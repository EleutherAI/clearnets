from argparse import ArgumentParser


def calculate_attention_flops(config, batch_size, seq_length):
    """Calculate FLOPs for attention layer."""
    hidden_size = config.hidden_size
    num_heads = config.num_heads
    head_dim = hidden_size // num_heads
    
    qkv_flops = 3 * batch_size * seq_length * hidden_size * hidden_size
    attn_flops = batch_size * num_heads * seq_length * seq_length * head_dim
    # Attention output projection
    out_flops = batch_size * seq_length * hidden_size * hidden_size
    
    return qkv_flops + attn_flops + out_flops

def calculate_dense_mlp_flops(config, batch_size, seq_length):
    """Calculate FLOPs for dense MLP layer."""
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
    
    # First linear layer + activation
    first_linear_flops = 2 * batch_size * seq_length * hidden_size * intermediate_size
    
    # Second linear layer
    second_linear_flops = 2 * batch_size * seq_length * intermediate_size * hidden_size
    
    return first_linear_flops + second_linear_flops

def calculate_sparse_mlp_flops(config, batch_size, seq_length):
    """Calculate FLOPs for sparse MLP layer with top-k activation."""
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
    expanded_dim = intermediate_size * 8  # From the SparseMLP implementation
    k = config.k
    
    # First linear layer + activation (full computation)
    first_linear_flops = 2 * batch_size * seq_length * hidden_size * expanded_dim
    
    # Top-k selection (O(n) operations per position for partial sort)
    topk_flops = batch_size * seq_length * expanded_dim
    
    # Sparse second linear (only k activations per position, assumes Triton kernel)
    sparse_second_flops = 2 * batch_size * seq_length * k * hidden_size
    
    return first_linear_flops + topk_flops + sparse_second_flops

def calculate_transformer_flops(config, batch_size, seq_length, is_sparse=False):
    """Calculate total FLOPs for the transformer model."""
    # Embedding layer
    embedding_flops = batch_size * seq_length * config.hidden_size
    
    # Layer calculations
    layer_flops = 0
    for _ in range(config.num_layers):
        # Attention
        layer_flops += calculate_attention_flops(config, batch_size, seq_length)
        
        # MLP
        if is_sparse:
            layer_flops += calculate_sparse_mlp_flops(config, batch_size, seq_length)
        else:
            layer_flops += calculate_dense_mlp_flops(config, batch_size, seq_length)
        
        # Layer norms (2 per layer)
        layer_flops += 2 * batch_size * seq_length * config.hidden_size
    
    # Final layer norm
    final_ln_flops = batch_size * seq_length * config.hidden_size
    
    total_flops = embedding_flops + layer_flops + final_ln_flops
    return total_flops

def calculate_sae_forward_flops(config, sae_config, batch_size, seq_length, auxk_loss=False):
    """Calculate FLOPs for a single forward pass through the SAE."""
    d_in = config.hidden_size 
    num_latents = sae_config.num_latents or d_in * sae_config.expansion_factor
    k = sae_config.k
    
    # Encoder linear + ReLU
    encoder_flops = 2 * batch_size * seq_length * d_in * num_latents
    
    # Top-k selection (O(n) partial sort)
    topk_flops = batch_size * seq_length * num_latents
    
    # Sparse decoder using k selected features
    decoder_flops = 2 * batch_size * seq_length * k * d_in
    
    # L2 loss calculation
    loss_flops = batch_size * seq_length * d_in
    
    auxk_decode_flops = 0
    if auxk_loss:
        auxk_k = d_in // 2  # From SAE code: k_aux = x.shape[-1] // 2
        auxk_decode_flops = 2 * batch_size * seq_length * auxk_k * d_in

    variance_flops = batch_size * seq_length * d_in * 2  # mean and squared diff
    
    total_flops = (encoder_flops + topk_flops + decoder_flops + loss_flops + 
                  auxk_decode_flops + variance_flops)
    
    return total_flops


def calculate_total_sae_training_flops(
    transformer_config,
    sae_config,
    batch_size,
    seq_length,
):
    """Calculate total FLOPs for SAE training process."""
    # FLOPs per forward pass through transformer to get activations
    transformer_flops = calculate_transformer_flops(
        transformer_config, 
        batch_size, 
        seq_length,
        is_sparse=False
    )
    
    # FLOPs for SAE forward/backward
    sae_forward_flops = calculate_sae_forward_flops(
        transformer_config,
        sae_config, 
        batch_size, 
        seq_length,
        auxk_loss=sae_config.multi_topk
    )
    
    # Approximate backward pass as 2x forward pass
    sae_backward_flops = 2 * sae_forward_flops
    
    # Total FLOPs per step including forward and backward
    return transformer_flops + sae_forward_flops + sae_backward_flops


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1280)
    parser.add_argument("--sae_batch_size", type=int, default=1280)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=18_500, # Rough globals steps from WandB 
                        help='Number of training steps with gradient accumulation steps not included')
    return parser.parse_args()


if __name__ == "__main__":   
    # The step count in wandb will match your actual optimization steps, not the individual 
    # forward/backward passes during gradient accumulation. 
    # This matches the standard convention of counting actual weight updates as steps.

    args = parse_args()
    print("Batch size:", args.batch_size)
    print("Sequence length:", args.seq_len)
    print("Number of training steps (gradient accumulation steps not included):", args.num_steps)

    from clearnets.train.sparse_gptneox_config import SparseGPTNeoConfig
    from clearnets.train.train_transformer import MODEL_CONFIG
    from sparsify.config import SaeConfig

    sparse_config = SparseGPTNeoConfig(**MODEL_CONFIG["roneneldan/TinyStories-8M"], sparse_mlp=True)
    sparse_flops_per_step = calculate_transformer_flops(sparse_config, batch_size=args.batch_size, seq_length=args.seq_len, is_sparse=True)
    sparse_flops = sparse_flops_per_step * args.num_steps
    print('sparse_flops', f'{sparse_flops:.2e}')

    dense_config = SparseGPTNeoConfig(**MODEL_CONFIG["roneneldan/TinyStories-8M"], sparse_mlp=False)
    dense_flops_per_step = calculate_transformer_flops(dense_config, batch_size=args.batch_size, seq_length=args.seq_len, is_sparse=False)
    dense_flops = dense_flops_per_step * args.num_steps
    
    print('dense_flops', f'{dense_flops:.2e}')
    # ~1 : 3.2 in favor of sparse FLOPs

    print('Leftover FLOP for training SAE on dense transformer', f'{(sparse_flops - dense_flops):.2e}')

    sae_config = SaeConfig(
        expansion_factor=8,
        k=32,
        multi_topk=True   
    )
    sae_flop_per_step = calculate_total_sae_training_flops(
        dense_config,
        sae_config,
        batch_size=args.sae_batch_size,
        seq_length=args.seq_len,
    )
    print(sae_flop_per_step)

    flop_budget = (sparse_flops - dense_flops)
    # ~5 million @ batch size of 8
    # 78,500 @ batch size of 512
    # 31,400 @ batch size of 1280
    sae_steps_in_budget = flop_budget // sae_flop_per_step
    sae_sequences_per_step = args.sae_batch_size

    # We have far more steps in our budget than we need (referring to previous runs: https://wandb.ai/eleutherai/sae?nw=nwuserluciarosequirke&panelDisplayName=fvu%2Ftransformer.h.1.mlp)
    # So we can simply run to convergence which seems to take around 6k steps
    sae_max_sequences = sae_steps_in_budget * sae_sequences_per_step
    print('sae_max_sequences', f'{sae_max_sequences:.2e}') 

    print('SAE sequences in 6k steps', sae_sequences_per_step * 6_000)
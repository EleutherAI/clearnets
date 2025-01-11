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

def calculate_sae_forward_flops(config, sae_config, batch_size, seq_length, auxk_loss=True):
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
    num_training_steps,
    grad_accumulation_steps=1
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
        seq_length
    )
    
    # Approximate backward pass as 2x forward pass
    sae_backward_flops = 2 * sae_forward_flops
    
    # Total FLOPs per step including forward and backward
    flops_per_step = transformer_flops + sae_forward_flops + sae_backward_flops
    
    # Scale by number of steps and gradient accumulation
    total_flops = flops_per_step * num_training_steps * grad_accumulation_steps
    
    return total_flops


if __name__ == "__main__":    
    # max_position_embeddings=context_length
    from clearnets.train.sparse_gptneox_config import SparseGPTNeoConfig
    from clearnets.train.train_transformer import MODEL_CONFIG
    from sae.config import SaeConfig

    # Effective batch size = 80 * 16 = 1280
    sparse_batch_size_scalar = 2
    batch_size = 80 // sparse_batch_size_scalar
    gradient_accumulation_steps = 16 * sparse_batch_size_scalar

    sparse_config = SparseGPTNeoConfig(**MODEL_CONFIG["roneneldan/TinyStories8M"], sparse_mlp=True)
    sparse_flops = calculate_transformer_flops(sparse_config, batch_size=1, seq_length=512, is_sparse=True)
    print('sparse_flops per step, batch size of 1', f'{sparse_flops:.2e}')

    # The step count in wandb will match your actual optimization steps, not the individual 
    # forward/backward passes during gradient accumulation. 
    # This matches the standard convention of counting actual weight updates as steps.
    sparse_batch_size_scalar = 1
    batch_size = 80 // sparse_batch_size_scalar
    gradient_accumulation_steps = 16 * sparse_batch_size_scalar

    dense_config = SparseGPTNeoConfig(**MODEL_CONFIG["roneneldan/TinyStories8M"], sparse_mlp=False)
    dense_flops = calculate_transformer_flops(dense_config, batch_size=1, seq_length=512, is_sparse=False)
    print('dense_flops per step, batch size of 1', f'{dense_flops:.2e}')

    sae_config = SaeConfig(
        k=32,
        expansion_factor=4,
        normalize_decoder=True
    )
    total_sae_flops = calculate_total_sae_training_flops(
        dense_config,
        sae_config,
        batch_size=1,
        seq_length=512,
        num_training_steps=1
    )
    print('total_sae_flops per step, batch size of 1', f'{total_sae_flops:.2e}')


    # ~1 : 3.2 in favor of sparse FLOPs

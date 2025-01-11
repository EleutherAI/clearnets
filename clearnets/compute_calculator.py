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


if __name__ == "__main__":    
    # max_position_embeddings=context_length
    from clearnets.train.sparse_gptneox_config import SparseGPTNeoConfig
    from clearnets.train.train_transformer import MODEL_CONFIG

    config = SparseGPTNeoConfig(**MODEL_CONFIG["roneneldan/TinyStories8M"], sparse_mlp=False)
    dense_flops = calculate_transformer_flops(config, batch_size=1, seq_length=512, is_sparse=False)
    print('dense_flops', f'{dense_flops:.2e}')

    config = SparseGPTNeoConfig(**MODEL_CONFIG["roneneldan/TinyStories8M"], sparse_mlp=True)
    sparse_flops = calculate_transformer_flops(config, batch_size=1, seq_length=512, is_sparse=True)
    print('sparse_flops', f'{sparse_flops:.2e}')
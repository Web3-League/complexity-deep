"""
Utility functions for Complexity Concept models.
"""

from complexity_deep.models.config import ComplexityConfig
from complexity_deep.models.modeling import ComplexityForCausalLM


SIZE_PRESETS = {
    "tiny": ComplexityConfig.complexity_tiny,
    "20m": ComplexityConfig.complexity_20m,
    "small": ComplexityConfig.complexity_small,
    "150m": ComplexityConfig.complexity_150m,
    "base": ComplexityConfig.complexity_base,
    "350m": ComplexityConfig.complexity_350m,
    "medium": ComplexityConfig.complexity_medium,
    "large": ComplexityConfig.complexity_large,
    "1b": ComplexityConfig.complexity_1b,
    "3b": ComplexityConfig.complexity_3b,
}


def create_complexity_model(
    size: str = "base",
    vocab_size: int = 100000,
    **kwargs,
) -> ComplexityForCausalLM:
    """
    Create a Complexity model by size name.

    Args:
        size: One of "tiny", "small", "base", "medium", "large", "1b", "3b"
        vocab_size: Vocabulary size (default: 100K for Complexity-4)
        **kwargs: Additional config overrides

    Returns:
        ComplexityForCausalLM model

    Example:
        model = create_complexity_model("base")
        model = create_complexity_model("small", vocab_size=50000)
    """
    if size not in SIZE_PRESETS:
        raise ValueError(f"Unknown size: {size}. Choose from {list(SIZE_PRESETS.keys())}")

    config = SIZE_PRESETS[size]()
    config.vocab_size = vocab_size

    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return ComplexityForCausalLM(config)


def estimate_model_size(config: ComplexityConfig) -> dict:
    """
    Estimate model size and memory requirements.

    Args:
        config: Model configuration

    Returns:
        Dictionary with size estimates
    """
    # Embedding parameters
    embed_params = config.vocab_size * config.hidden_size

    # Per-layer parameters
    # Attention: Q, K, V, O projections
    attn_params = (
        config.hidden_size * config.num_attention_heads * (config.hidden_size // config.num_attention_heads) +  # Q
        config.hidden_size * config.num_key_value_heads * (config.hidden_size // config.num_attention_heads) +  # K
        config.hidden_size * config.num_key_value_heads * (config.hidden_size // config.num_attention_heads) +  # V
        config.hidden_size * config.hidden_size  # O
    )

    # MLP: gate, up, down
    mlp_params = (
        config.hidden_size * config.intermediate_size +  # gate
        config.hidden_size * config.intermediate_size +  # up
        config.intermediate_size * config.hidden_size    # down
    )

    # Layer norms
    norm_params = config.hidden_size * 2  # input_layernorm + post_attention_layernorm

    # Per layer total
    layer_params = attn_params + mlp_params + norm_params

    # Total
    total_params = embed_params + (layer_params * config.num_hidden_layers) + config.hidden_size  # final norm

    # If not tying embeddings, add LM head
    if not config.tie_word_embeddings:
        total_params += config.vocab_size * config.hidden_size

    # Memory estimates (in bytes)
    fp32_memory = total_params * 4
    fp16_memory = total_params * 2
    bf16_memory = total_params * 2

    return {
        "total_params": total_params,
        "embed_params": embed_params,
        "layer_params": layer_params,
        "num_layers": config.num_hidden_layers,
        "memory_fp32_gb": fp32_memory / (1024 ** 3),
        "memory_fp16_gb": fp16_memory / (1024 ** 3),
        "memory_bf16_gb": bf16_memory / (1024 ** 3),
    }

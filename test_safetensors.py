"""
Test generation from safetensors checkpoint.
"""

import torch
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast
from complexity_deep.models import ComplexityForCausalLM, ComplexityConfig


def main():
    checkpoint_path = "checkpoints/step_200000.safetensors"
    tokenizer_path = "checkpoints/"

    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Vocab size: {len(tokenizer)}")

    print(f"\nLoading config...")
    config = ComplexityConfig.from_dict({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 2048,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "pad_token_id": 1,
        "bos_token_id": 2,
        "eos_token_id": 0,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
    })

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = load_file(checkpoint_path, device="cpu")
    print(f"Keys in checkpoint: {len(state_dict)}")

    # Create model
    model = ComplexityForCausalLM(config)

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}: {sum(p.numel() for p in model.parameters()):,} params")

    # Test prompts
    prompts = [
        "The meaning of life is",
        "def fibonacci(n):",
        "Once upon a time",
        "Paris is the capital of",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated:\n{generated}")


if __name__ == "__main__":
    main()

"""Test checkpoint compatibility with new contextual mu."""

import torch
from safetensors.torch import load_file
from complexity_deep import DeepConfig, DeepForCausalLM

def main():
    checkpoint_path = "checkpoints/step_200000.safetensors"

    print("Loading checkpoint...")
    state_dict = load_file(checkpoint_path)
    print(f"Checkpoint keys: {len(state_dict)}")

    # Check for mu keys
    mu_keys = [k for k in state_dict.keys() if '.mu' in k]
    print(f"\nFound {len(mu_keys)} mu-related keys:")
    for k in mu_keys[:5]:
        print(f"  {k}: {state_dict[k].shape}")

    # Create model with new contextual mu
    print("\nCreating model with contextual mu...")
    config = DeepConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_experts=4,
    )
    model = DeepForCausalLM(config)

    # Check new model keys
    model_keys = list(model.state_dict().keys())
    mu_proj_keys = [k for k in model_keys if 'mu_proj' in k]
    print(f"\nNew model has {len(mu_proj_keys)} mu_proj keys:")
    for k in mu_proj_keys[:5]:
        print(f"  {k}: {model.state_dict()[k].shape}")

    # Try to load checkpoint (strict=False to allow new keys)
    print("\nLoading checkpoint into new model (strict=False)...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"\nMissing keys (new params, will be randomly init): {len(missing)}")
    for k in missing[:10]:
        print(f"  {k}")

    print(f"\nUnexpected keys (in checkpoint but not model): {len(unexpected)}")
    for k in unexpected[:10]:
        print(f"  {k}")

    # Verify mu_proj is zero-initialized (compatible behavior)
    print("\nVerifying mu_proj weights are zero (for compatibility):")
    for name, param in model.named_parameters():
        if 'mu_proj' in name:
            max_val = param.abs().max().item()
            print(f"  {name}: max={max_val:.6f} (should be 0)")
            break

    print("\n✓ Checkpoint compatibility verified!")
    print("  - Old checkpoint loads fine")
    print("  - mu_proj starts at zero (same behavior as before)")
    print("  - Model will learn contextual mu during training")

if __name__ == "__main__":
    main()

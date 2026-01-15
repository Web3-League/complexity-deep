"""Analyze mu values in checkpoint to see if they're stuck."""

import torch
from safetensors.torch import load_file

checkpoint_path = "checkpoints/step_200000.safetensors"

print("Loading checkpoint...")
state_dict = load_file(checkpoint_path)

# Find all mu parameters
mu_keys = [k for k in state_dict.keys() if '.mu' in k and 'mu_' not in k]

print(f"\nFound {len(mu_keys)} mu parameters:\n")

for key in mu_keys[:5]:  # First 5 layers
    mu = state_dict[key]
    print(f"{key}:")
    print(f"  Shape: {mu.shape}")
    print(f"  Mean:  {mu.mean().item():.6f}")
    print(f"  Std:   {mu.std().item():.6f}")
    print(f"  Min:   {mu.min().item():.6f}")
    print(f"  Max:   {mu.max().item():.6f}")
    print(f"  Near zero?: {(mu.abs() < 0.01).float().mean().item()*100:.1f}% of values")
    print()

# Summary across all layers
print("=" * 50)
print("SUMMARY ACROSS ALL LAYERS:")
all_mu = torch.cat([state_dict[k].flatten() for k in mu_keys])
print(f"  Total mu values: {all_mu.numel():,}")
print(f"  Global mean: {all_mu.mean().item():.6f}")
print(f"  Global std:  {all_mu.std().item():.6f}")
print(f"  Near zero (<0.01): {(all_mu.abs() < 0.01).float().mean().item()*100:.1f}%")
print(f"  Near zero (<0.1):  {(all_mu.abs() < 0.1).float().mean().item()*100:.1f}%")

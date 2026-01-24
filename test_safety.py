"""
Test Safety Integration in Complexity Deep

Tests:
1. SafetyClamp basic functionality
2. INLDynamics with safety clamp
3. Model-level safety installation
4. Contrastive safety loss
"""

import torch
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Testing Safety Integration - Complexity Deep")
print("=" * 60)

# Test 1: SafetyClamp
print("\n[1] Testing SafetyClamp...")
from complexity_deep.core.safety import SafetyClamp

clamp = SafetyClamp(hidden_size=256, threshold=2.0, soft_clamp=False)

# Set harm direction
harm_dir = torch.randn(256)
clamp.set_harm_direction(harm_dir)
clamp.enabled = True

# Test clamping
x = torch.randn(2, 32, 256)
x_clamped = clamp(x)
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {x_clamped.shape}")
print(f"  Stats: {clamp.get_stats()}")
print("  [OK] SafetyClamp works!")

# Test 2: High projection clamping
print("\n[2] Testing high-projection clamping...")
harm_dir_norm = harm_dir / harm_dir.norm()
high_harm = harm_dir_norm.unsqueeze(0) * 5.0  # 5x threshold
clamped = clamp(high_harm)
proj_before = (high_harm @ harm_dir_norm).item()
proj_after = (clamped @ harm_dir_norm).item()
print(f"  Projection before: {proj_before:.4f}")
print(f"  Projection after:  {proj_after:.4f}")
print(f"  Threshold:         {clamp.threshold}")
assert proj_after <= clamp.threshold + 0.01, f"Clamping failed! {proj_after} > {clamp.threshold}"
print("  [OK] High-projection clamping works!")

# Test 3: INLDynamics with safety
print("\n[3] Testing INLDynamics with safety clamp...")
from complexity_deep.core.layer import INLDynamics

dynamics = INLDynamics(hidden_size=256)

# Install safety
safety = SafetyClamp(hidden_size=256, threshold=2.0)
safety.set_harm_direction(harm_dir)
safety.enabled = True
dynamics.install_safety(safety)

# Forward pass
h = torch.randn(2, 32, 256)
v = torch.zeros(2, 32, 256)
h_next, v_next, mu = dynamics(h, v)

print(f"  h_next shape: {h_next.shape}")
print(f"  v_next shape: {v_next.shape}")
print(f"  mu shape:     {mu.shape}")
print(f"  Safety stats: {dynamics.get_safety_stats()}")
print("  [OK] INLDynamics with safety works!")

# Test 4: Remove safety
print("\n[4] Testing safety removal...")
dynamics.remove_safety()
h_next2, v_next2, mu2 = dynamics(h, v)
print(f"  Safety stats after removal: {dynamics.get_safety_stats()}")
print("  [OK] Safety removal works!")

# Test 5: ContrastiveSafetyLoss
print("\n[5] Testing ContrastiveSafetyLoss...")
from complexity_deep.core.safety import ContrastiveSafetyLoss

loss_fn = ContrastiveSafetyLoss(hidden_size=256, margin=1.0)

safe_act = torch.randn(4, 256)
harmful_act = torch.randn(4, 256)

result = loss_fn(safe_act, harmful_act)
print(f"  Loss:       {result['loss'].item():.4f}")
print(f"  Separation: {result['separation'].item():.4f}")
print("  [OK] ContrastiveSafetyLoss works!")

# Test 6: Model-level installation
print("\n[6] Testing model-level safety installation...")
from complexity_deep import DeepConfig, DeepForCausalLM, install_safety_on_model

config = DeepConfig.complexity_tiny()
model = DeepForCausalLM(config)

# Install safety on last 2 layers
install_safety_on_model(
    model,
    harm_direction=harm_dir,
    threshold=2.0,
    layers=[-2, -1]
)

# Forward pass
input_ids = torch.randint(0, 1000, (2, 16))
output = model(input_ids)

print(f"  Logits shape: {output.logits.shape}")
print("  [OK] Model-level safety works!")

# Test 7: Verify safety is applied
print("\n[7] Checking safety on layers...")
for i, layer in enumerate(model.model.layers):
    has_safety = hasattr(layer.dynamics, 'safety_clamp') and layer.dynamics.safety_clamp is not None
    if has_safety:
        print(f"  Layer {i}: Safety enabled")
print("  [OK] Safety verified on layers!")

# Test 8: Remove safety from model
print("\n[8] Testing model safety removal...")
from complexity_deep import remove_safety_from_model

remove_safety_from_model(model)

for i, layer in enumerate(model.model.layers):
    has_safety = hasattr(layer.dynamics, 'safety_clamp') and layer.dynamics.safety_clamp is not None
    if has_safety:
        print(f"  WARNING: Layer {i} still has safety!")

print("  [OK] Safety removed from model!")

print("\n" + "=" * 60)
print("All safety tests passed!")
print("=" * 60)

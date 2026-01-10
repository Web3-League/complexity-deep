"""
Complexity Deep Core Components
===============================

Building blocks for the Complexity Deep architecture with full INL Dynamics.

Architecture per layer:
    1. KQV Attention (perception)
    2. INL Dynamics (control with velocity)
    3. Token-Routed MLP (transformation)
"""

from complexity_deep.core.normalization import RMSNorm
from complexity_deep.core.rotary import RotaryEmbedding, apply_rotary_pos_emb
from complexity_deep.core.attention import ComplexityAttention
from complexity_deep.core.mlp import ComplexityMLP
from complexity_deep.core.token_routed_mlp import TokenRoutedMLP
from complexity_deep.core.layer import DeepDecoderLayer, INLDynamics

# Backward compatibility aliases
ComplexityDecoderLayer = DeepDecoderLayer
INLDynamicsLite = INLDynamics

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "ComplexityAttention",
    "ComplexityMLP",
    "TokenRoutedMLP",
    "DeepDecoderLayer",
    "INLDynamics",
    # Aliases
    "ComplexityDecoderLayer",
    "INLDynamicsLite",
]

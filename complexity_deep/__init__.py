"""
Complexity Deep - Llama + Full INL Dynamics (Robotics)
======================================================

Multicouche architecture with robotics-grade control.

Each layer has 3 components:
    1. KQV Attention (perception)
    2. INL Dynamics (control with velocity tracking)
    3. Token-Routed MLP (transformation)

Features:
- Full velocity tracking (smooth trajectories)
- Adaptive controller (alpha, beta, gate)
- Learnable equilibrium (mu)
- Real-time robotics capable

Usage:
    from complexity_deep import DeepConfig, DeepForCausalLM

    config = DeepConfig.deep_base()
    model = DeepForCausalLM(config)
"""

from complexity_deep.core import (
    RMSNorm,
    RotaryEmbedding,
    ComplexityAttention,
    ComplexityMLP,
    DeepDecoderLayer,
    INLDynamics,
)

from complexity_deep.models import (
    ComplexityConfig as DeepConfig,
    ComplexityModel as DeepModel,
    ComplexityForCausalLM as DeepForCausalLM,
    create_complexity_model as create_deep_model,
)

__version__ = "0.5.0"
__all__ = [
    # Core
    "RMSNorm",
    "RotaryEmbedding",
    "ComplexityAttention",
    "ComplexityMLP",
    "DeepDecoderLayer",
    "INLDynamics",
    # Models
    "DeepConfig",
    "DeepModel",
    "DeepForCausalLM",
    "create_deep_model",
]

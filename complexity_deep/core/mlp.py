"""
MLP (Feed-Forward) layers for Complexity architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexityMLP(nn.Module):
    """
    SwiGLU MLP (Gated Linear Unit with Swish activation).

    Used in Llama, Mistral, and other modern architectures.
    More expressive than standard FFN with similar parameter count.

    Architecture:
        out = down_proj(swish(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Three linear projections (no bias for efficiency)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Activation function
        self.act_fn = self._get_activation(hidden_act)

    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            "silu": F.silu,
            "swish": F.silu,
            "gelu": F.gelu,
            "relu": F.relu,
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # SwiGLU: swish(gate) * up
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

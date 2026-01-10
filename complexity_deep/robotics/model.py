"""
Robotics Complexity Model
=========================

Adapted Complexity model for real-time robotics control.

Key adaptations:
- Multi-modal input (vision, proprio, actions, language)
- Action output head with continuous decoding
- Optimized for low-latency inference
- Token-Routed MLP naturally specializes per modality

Control loop:
    observe → tokenize → model → decode → actuate
    ~10-20ms total latency on edge GPU

Author: Pacific Prime
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
from dataclasses import dataclass, field

try:
    from ..cuda import HAS_TRITON
    from ..cuda.optimized_layer import OptimizedTransformerLayer, OptimizationConfig
    OPTIMIZATIONS_AVAILABLE = HAS_TRITON
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False

from .tokenizer import RoboticsTokenizer, TokenizerConfig


@dataclass
class RoboticsConfig:
    """Configuration for robotics model."""
    # Model architecture
    vocab_size: int = 100000
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 1408
    num_experts: int = 4  # Maps to 4 modalities!
    max_position_embeddings: int = 512

    # Robotics specific
    action_dim: int = 7  # 6-DOF + gripper
    proprio_dim: int = 32
    action_horizon: int = 8  # Predict N future actions
    control_freq: int = 50  # Hz

    # Tokenizer config
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)

    # Optimization
    use_cuda_optimizations: bool = True
    use_fp16: bool = True


class ActionHead(nn.Module):
    """
    Action prediction head for robotics.

    Decodes transformer hidden states to:
    1. Discrete action tokens (for autoregressive generation)
    2. Continuous action values (for direct control)
    """

    def __init__(self, config: RoboticsConfig):
        super().__init__()
        self.config = config

        # Token prediction (for autoregressive)
        self.token_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Continuous action prediction (for direct control)
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.action_dim * config.action_horizon),
        )

        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(config.hidden_size, config.action_dim * config.action_horizon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_continuous: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict actions from hidden states.

        Args:
            hidden_states: [batch, seq, hidden] transformer output
            return_continuous: Whether to return continuous actions

        Returns:
            dict with:
                - logits: [batch, seq, vocab_size] token logits
                - actions: [batch, action_horizon, action_dim] continuous actions
                - uncertainty: [batch, action_horizon, action_dim] uncertainty estimates
        """
        batch_size = hidden_states.shape[0]

        # Token logits (for training)
        logits = self.token_head(hidden_states)

        output = {"logits": logits}

        if return_continuous:
            # Use last hidden state for action prediction
            last_hidden = hidden_states[:, -1, :]  # [batch, hidden]

            # Continuous actions
            actions_flat = self.action_head(last_hidden)  # [batch, action_horizon * action_dim]
            actions = actions_flat.view(batch_size, self.config.action_horizon, self.config.action_dim)
            actions = torch.tanh(actions)  # Bound to [-1, 1]

            # Uncertainty
            uncertainty_flat = self.uncertainty_head(last_hidden)
            uncertainty = F.softplus(uncertainty_flat.view(batch_size, self.config.action_horizon, self.config.action_dim))

            output["actions"] = actions
            output["uncertainty"] = uncertainty

        return output


class RoboticsComplexity(nn.Module):
    """
    Complexity model adapted for robotics control.

    Architecture:
        Input: [vision_tokens, proprio_tokens, action_history_tokens, language_tokens]
        Model: Transformer with Token-Routed MLP (4 experts = 4 modalities)
        Output: Next action tokens + continuous action values

    The Token-Routed MLP naturally routes:
        - Vision tokens (0-25k) → Expert 0 (visual features)
        - Proprio tokens (25k-50k) → Expert 1 (body state)
        - Action tokens (50k-75k) → Expert 2 (action patterns)
        - Language tokens (75k-100k) → Expert 3 (semantic understanding)
    """

    def __init__(self, config: RoboticsConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Modality embeddings (added to token embeddings)
        self.modality_embed = nn.Embedding(4, config.hidden_size)  # 4 modalities

        # Position embeddings
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Transformer layers
        if config.use_cuda_optimizations and OPTIMIZATIONS_AVAILABLE:
            opt_config = OptimizationConfig(
                use_fused_attention=True,
                use_fused_mlp=True,
                use_fused_residual=True,
                use_persistent_cggr=True,
                use_int8_quantization=False,
            )
            self.layers = nn.ModuleList([
                OptimizedTransformerLayer(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    intermediate_size=config.intermediate_size,
                    num_experts=config.num_experts,
                    vocab_size=config.vocab_size,
                )
                for _ in range(config.num_hidden_layers)
            ])
        else:
            # Standard layers (import from main module)
            from ..core.layer import ComplexityDecoderLayer
            from ..models.config import ComplexityConfig as BaseConfig

            base_config = BaseConfig(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                vocab_size=config.vocab_size,
            )
            self.layers = nn.ModuleList([
                ComplexityDecoderLayer(base_config)
                for _ in range(config.num_hidden_layers)
            ])

        # Output norm
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-6)

        # Action head
        self.action_head = ActionHead(config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_continuous_actions: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for robotics control.

        Args:
            input_ids: [batch, seq] token IDs
            modality_ids: [batch, seq] modality ID for each token (0-3)
            attention_mask: [batch, seq] attention mask
            return_continuous_actions: Whether to return continuous action predictions

        Returns:
            dict with:
                - logits: [batch, seq, vocab_size]
                - actions: [batch, action_horizon, action_dim]
                - uncertainty: [batch, action_horizon, action_dim]
                - hidden_states: [batch, seq, hidden]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Add modality embeddings
        if modality_ids is not None:
            hidden_states = hidden_states + self.modality_embed(modality_ids)

        # Add position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = hidden_states + self.pos_embed(positions)

        # Transformer layers
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                # Check if layer expects token_ids
                try:
                    hidden_states = layer(hidden_states, token_ids=input_ids)
                except TypeError:
                    hidden_states = layer(hidden_states)

        # Output norm
        hidden_states = self.norm(hidden_states)

        # Action head
        outputs = self.action_head(hidden_states, return_continuous=return_continuous_actions)
        outputs["hidden_states"] = hidden_states

        return outputs

    def predict_action(
        self,
        observation: Dict[str, torch.Tensor],
        tokenizer: RoboticsTokenizer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict action from observation (convenience method).

        Args:
            observation: Dict with 'image', 'proprio', 'prev_actions'
            tokenizer: RoboticsTokenizer instance

        Returns:
            actions: [batch, action_horizon, action_dim] predicted actions
            uncertainty: [batch, action_horizon, action_dim] uncertainty
        """
        # Tokenize observation
        tokens = tokenizer.tokenize_observation(
            image=observation.get("image"),
            proprio=observation.get("proprio"),
            prev_actions=observation.get("prev_actions"),
            language=observation.get("language"),
        )

        # Forward pass
        with torch.no_grad():
            outputs = self.forward(
                input_ids=tokens["token_ids"],
                modality_ids=tokens["modality_ids"],
                return_continuous_actions=True,
            )

        return outputs["actions"], outputs["uncertainty"]


# =============================================================================
# LIGHTWEIGHT MODEL FOR EDGE DEPLOYMENT
# =============================================================================

class RoboticsComplexityTiny(RoboticsComplexity):
    """
    Tiny version optimized for edge devices (Jetson Nano, etc).

    ~20M parameters, ~15-25ms inference on Jetson Orin.
    """

    def __init__(
        self,
        action_dim: int = 7,
        proprio_dim: int = 32,
    ):
        config = RoboticsConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=704,
            num_experts=4,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            use_cuda_optimizations=True,
            use_fp16=True,
        )
        super().__init__(config)


class RoboticsComplexitySmall(RoboticsComplexity):
    """
    Small version for desktop GPUs.

    ~50M parameters, ~8-12ms inference on RTX 3060.
    """

    def __init__(
        self,
        action_dim: int = 7,
        proprio_dim: int = 32,
    ):
        config = RoboticsConfig(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=1408,
            num_experts=4,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            use_cuda_optimizations=True,
            use_fp16=True,
        )
        super().__init__(config)

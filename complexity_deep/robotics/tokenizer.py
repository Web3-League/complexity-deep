"""
Multi-Modal Robotics Tokenizer
==============================

Tokenizes multiple input modalities for robotics:
- Vision: Images → tokens via vision encoder
- Proprioception: Joint states → tokens via linear projection
- Actions: Previous actions → tokens
- Language: Text instructions → tokens

Token ID ranges (100k vocab):
- 0-25k:      Vision tokens (Expert 0)
- 25k-50k:    Proprio tokens (Expert 1)
- 50k-75k:    Action tokens (Expert 2)
- 75k-100k:   Language tokens (Expert 3)

This maps directly to Token-Routed MLP experts!

Author: Pacific Prime
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import numpy as np

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


@dataclass
class TokenizerConfig:
    """Configuration for robotics tokenizer."""
    vocab_size: int = 100000
    vision_vocab_start: int = 0
    vision_vocab_size: int = 25000
    proprio_vocab_start: int = 25000
    proprio_vocab_size: int = 25000
    action_vocab_start: int = 50000
    action_vocab_size: int = 25000
    language_vocab_start: int = 75000
    language_vocab_size: int = 25000

    # Vision
    image_size: int = 224
    patch_size: int = 14
    vision_dim: int = 384  # DINOv2-small

    # Proprio
    proprio_dim: int = 32  # Joint positions + velocities + forces
    proprio_hidden: int = 256

    # Actions
    action_dim: int = 7  # 6-DOF + gripper
    action_hidden: int = 256
    action_bins: int = 256  # Discretization bins per dimension


# =============================================================================
# VISION ENCODER
# =============================================================================

class VisionEncoder(nn.Module):
    """
    Vision encoder for robotics.

    Converts images to discrete tokens using:
    1. Pretrained vision backbone (DINOv2, CLIP, etc.)
    2. Vector quantization to discrete tokens

    Each token ID falls in [0, 25k) → routed to Expert 0.
    """

    def __init__(
        self,
        config: TokenizerConfig,
        backbone: str = "dinov2-small",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.config = config

        # Vision backbone
        self.backbone_name = backbone
        self.backbone = self._load_backbone(backbone)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection to token space
        backbone_dim = self._get_backbone_dim(backbone)
        self.proj = nn.Linear(backbone_dim, config.vision_vocab_size)

        # Learnable codebook for VQ
        self.codebook = nn.Embedding(config.vision_vocab_size, backbone_dim)
        nn.init.uniform_(self.codebook.weight, -1/config.vision_vocab_size, 1/config.vision_vocab_size)

    def _load_backbone(self, name: str) -> nn.Module:
        """Load pretrained vision backbone."""
        if "dinov2" in name.lower():
            try:
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                return model
            except:
                # Fallback to simple CNN
                return self._simple_cnn()
        elif "clip" in name.lower():
            try:
                import clip
                model, _ = clip.load("ViT-B/16")
                return model.visual
            except:
                return self._simple_cnn()
        else:
            return self._simple_cnn()

    def _simple_cnn(self) -> nn.Module:
        """Simple CNN fallback."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten(2),
            nn.Linear(256, 384),
        )

    def _get_backbone_dim(self, name: str) -> int:
        """Get backbone output dimension."""
        if "dinov2" in name.lower() and "small" in name.lower():
            return 384
        elif "dinov2" in name.lower() and "base" in name.lower():
            return 768
        elif "clip" in name.lower():
            return 512
        else:
            return 384

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to discrete tokens.

        Args:
            images: [batch, 3, H, W] normalized images

        Returns:
            token_ids: [batch, num_patches] token IDs in [0, 25k)
            features: [batch, num_patches, dim] continuous features
        """
        batch_size = images.shape[0]

        # Get features from backbone
        with torch.no_grad() if not self.training else torch.enable_grad():
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(images)
                # Remove CLS token if present
                if features.shape[1] > 196:  # 14x14 patches + CLS
                    features = features[:, 1:]
            else:
                features = self.backbone(images)
                if features.dim() == 2:
                    features = features.unsqueeze(1)

        # features: [batch, num_patches, dim]
        num_patches = features.shape[1]

        # Vector quantization: find nearest codebook entry
        # Compute distances to all codebook entries
        flat_features = features.view(-1, features.shape[-1])  # [batch*patches, dim]

        # Efficient distance computation
        distances = (
            flat_features.pow(2).sum(dim=-1, keepdim=True)
            - 2 * flat_features @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=-1)
        )

        # Get nearest codebook indices
        token_ids = distances.argmin(dim=-1)  # [batch*patches]
        token_ids = token_ids.view(batch_size, num_patches)

        # Add vocab offset for vision tokens
        token_ids = token_ids + self.config.vision_vocab_start

        return token_ids, features


# =============================================================================
# PROPRIOCEPTION ENCODER
# =============================================================================

class ProprioEncoder(nn.Module):
    """
    Proprioception encoder for robotics.

    Converts continuous proprioceptive state to discrete tokens:
    - Joint positions
    - Joint velocities
    - Joint torques/forces
    - End-effector pose

    Token IDs fall in [25k, 50k) → routed to Expert 1.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config

        # MLP to process proprio
        self.encoder = nn.Sequential(
            nn.Linear(config.proprio_dim, config.proprio_hidden),
            nn.ReLU(),
            nn.Linear(config.proprio_hidden, config.proprio_hidden),
            nn.ReLU(),
        )

        # Quantization head
        self.quantizer = nn.Linear(config.proprio_hidden, config.proprio_vocab_size)

        # Learnable codebook
        self.codebook = nn.Embedding(config.proprio_vocab_size, config.proprio_hidden)

    def forward(
        self,
        proprio: torch.Tensor,
        num_tokens: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode proprioception to discrete tokens.

        Args:
            proprio: [batch, proprio_dim] proprioceptive state
            num_tokens: Number of tokens to generate

        Returns:
            token_ids: [batch, num_tokens] token IDs in [25k, 50k)
            features: [batch, num_tokens, hidden] continuous features
        """
        batch_size = proprio.shape[0]

        # Encode
        features = self.encoder(proprio)  # [batch, hidden]

        # Generate multiple tokens by projecting to different subspaces
        # This allows representing the full proprio state with multiple tokens
        features_expanded = features.unsqueeze(1).expand(-1, num_tokens, -1)

        # Add positional info
        pos = torch.arange(num_tokens, device=proprio.device).float()
        pos_embed = torch.sin(pos.unsqueeze(0).unsqueeze(-1) * 0.1)
        features_expanded = features_expanded + pos_embed.expand(batch_size, -1, features.shape[-1])

        # Quantize
        logits = self.quantizer(features_expanded)  # [batch, num_tokens, vocab_size]
        token_ids = logits.argmax(dim=-1)  # [batch, num_tokens]

        # Add vocab offset
        token_ids = token_ids + self.config.proprio_vocab_start

        return token_ids, features_expanded


# =============================================================================
# ACTION ENCODER
# =============================================================================

class ActionEncoder(nn.Module):
    """
    Action encoder/decoder for robotics.

    Converts actions to/from discrete tokens:
    - Discretizes continuous actions into bins
    - Each action dimension gets its own token

    Token IDs fall in [50k, 75k) → routed to Expert 2.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.num_bins = config.action_bins

        # Per-dimension bin edges (learned or fixed)
        self.register_buffer(
            "bin_edges",
            torch.linspace(-1, 1, config.action_bins + 1).unsqueeze(0).expand(config.action_dim, -1)
        )

        # Vocab offset per action dimension
        tokens_per_dim = config.action_vocab_size // config.action_dim
        self.tokens_per_dim = tokens_per_dim

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous actions to discrete tokens.

        Args:
            actions: [batch, action_dim] continuous actions in [-1, 1]

        Returns:
            token_ids: [batch, action_dim] token IDs in [50k, 75k)
        """
        batch_size = actions.shape[0]

        # Clip actions to valid range
        actions = actions.clamp(-1, 1)

        # Discretize each dimension
        token_ids = torch.zeros(batch_size, self.action_dim, dtype=torch.long, device=actions.device)

        for dim in range(self.action_dim):
            # Find bin for each action
            bins = torch.bucketize(actions[:, dim], self.bin_edges[dim, 1:-1])
            # Add offset for this dimension
            token_ids[:, dim] = bins + dim * self.tokens_per_dim

        # Add vocab offset
        token_ids = token_ids + self.config.action_vocab_start

        return token_ids

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete tokens to continuous actions.

        Args:
            token_ids: [batch, action_dim] token IDs in [50k, 75k)

        Returns:
            actions: [batch, action_dim] continuous actions in [-1, 1]
        """
        batch_size = token_ids.shape[0]

        # Remove vocab offset
        token_ids = token_ids - self.config.action_vocab_start

        actions = torch.zeros(batch_size, self.action_dim, device=token_ids.device)

        for dim in range(self.action_dim):
            # Get bin index for this dimension
            bins = token_ids[:, dim] - dim * self.tokens_per_dim
            bins = bins.clamp(0, self.num_bins - 1)

            # Convert bin to continuous value (bin center)
            bin_width = 2.0 / self.num_bins
            actions[:, dim] = -1 + (bins.float() + 0.5) * bin_width

        return actions


# =============================================================================
# UNIFIED ROBOTICS TOKENIZER
# =============================================================================

class RoboticsTokenizer(nn.Module):
    """
    Unified tokenizer for multi-modal robotics input.

    Combines:
    - Vision encoder (images → tokens 0-25k)
    - Proprio encoder (joint states → tokens 25k-50k)
    - Action encoder (actions → tokens 50k-75k)
    - Language tokenizer (text → tokens 75k-100k)

    The token ID ranges map directly to Token-Routed MLP experts!
    """

    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        vision_backbone: str = "dinov2-small",
        language_tokenizer: Optional[object] = None,
    ):
        super().__init__()
        self.config = config or TokenizerConfig()

        # Modality encoders
        self.vision_encoder = VisionEncoder(self.config, backbone=vision_backbone)
        self.proprio_encoder = ProprioEncoder(self.config)
        self.action_encoder = ActionEncoder(self.config)

        # Language tokenizer (external, e.g., from transformers)
        self.language_tokenizer = language_tokenizer

    def tokenize_observation(
        self,
        image: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        prev_actions: Optional[torch.Tensor] = None,
        language: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a complete robotics observation.

        Args:
            image: [batch, 3, H, W] camera image
            proprio: [batch, proprio_dim] proprioceptive state
            prev_actions: [batch, history, action_dim] previous actions
            language: Text instruction

        Returns:
            dict with:
                - token_ids: [batch, total_tokens] concatenated token IDs
                - attention_mask: [batch, total_tokens]
                - modality_ids: [batch, total_tokens] which modality each token is from
        """
        batch_size = image.shape[0] if image is not None else proprio.shape[0]
        device = image.device if image is not None else proprio.device

        all_tokens = []
        all_modalities = []

        # Vision tokens
        if image is not None:
            vision_tokens, _ = self.vision_encoder(image)
            all_tokens.append(vision_tokens)
            all_modalities.append(torch.zeros_like(vision_tokens))  # Modality 0

        # Proprio tokens
        if proprio is not None:
            proprio_tokens, _ = self.proprio_encoder(proprio)
            all_tokens.append(proprio_tokens)
            all_modalities.append(torch.ones_like(proprio_tokens))  # Modality 1

        # Action history tokens
        if prev_actions is not None:
            # Flatten action history
            batch_size, history_len, action_dim = prev_actions.shape
            flat_actions = prev_actions.view(batch_size * history_len, action_dim)
            action_tokens = self.action_encoder.encode(flat_actions)
            action_tokens = action_tokens.view(batch_size, history_len * action_dim)
            all_tokens.append(action_tokens)
            all_modalities.append(torch.full_like(action_tokens, 2))  # Modality 2

        # Language tokens
        if language is not None and self.language_tokenizer is not None:
            lang_tokens = self.language_tokenizer.encode(language, return_tensors="pt")
            lang_tokens = lang_tokens.to(device)
            # Offset to language vocab range
            lang_tokens = lang_tokens + self.config.language_vocab_start
            lang_tokens = lang_tokens.expand(batch_size, -1)
            all_tokens.append(lang_tokens)
            all_modalities.append(torch.full_like(lang_tokens, 3))  # Modality 3

        # Concatenate all tokens
        token_ids = torch.cat(all_tokens, dim=1)
        modality_ids = torch.cat(all_modalities, dim=1)

        # Attention mask (all ones for now)
        attention_mask = torch.ones_like(token_ids)

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "modality_ids": modality_ids,
        }

    def decode_actions(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Decode action tokens to continuous actions.

        Args:
            token_ids: [batch, action_dim] action token IDs

        Returns:
            actions: [batch, action_dim] continuous actions
        """
        return self.action_encoder.decode(token_ids)


# =============================================================================
# CUDA-OPTIMIZED TOKENIZATION
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _fast_bucketize_kernel(
        values_ptr,
        edges_ptr,
        output_ptr,
        num_values,
        num_bins,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fast bucketization for action discretization."""
        pid = tl.program_id(0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < num_values

        value = tl.load(values_ptr + offset, mask=mask, other=0.0)

        # Binary search for bin
        low = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        high = tl.full([BLOCK_SIZE], num_bins, dtype=tl.int32)

        for _ in range(10):  # log2(1024) iterations max
            mid = (low + high) // 2
            edge = tl.load(edges_ptr + mid, mask=mask, other=0.0)
            low = tl.where(value >= edge, mid + 1, low)
            high = tl.where(value < edge, mid, high)

        tl.store(output_ptr + offset, low - 1, mask=mask)


def fast_discretize_actions(actions: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    """
    Fast action discretization using Triton.

    Args:
        actions: [batch, action_dim] continuous actions in [-1, 1]
        num_bins: Number of discretization bins

    Returns:
        bins: [batch, action_dim] bin indices
    """
    if not HAS_TRITON or not actions.is_cuda:
        # Fallback to PyTorch
        edges = torch.linspace(-1, 1, num_bins + 1, device=actions.device)
        return torch.bucketize(actions, edges[1:-1])

    batch_size, action_dim = actions.shape
    num_values = batch_size * action_dim

    edges = torch.linspace(-1, 1, num_bins + 1, device=actions.device)
    output = torch.empty(num_values, dtype=torch.int32, device=actions.device)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_values, BLOCK_SIZE),)

    _fast_bucketize_kernel[grid](
        actions.view(-1), edges, output,
        num_values, num_bins,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(batch_size, action_dim)

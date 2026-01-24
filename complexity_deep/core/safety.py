"""
Safety Module for Complexity Deep

Representation Engineering approach for inference safety.
Clamps activations along learned harm directions.

Integration points:
- INLDynamics: clamp mu_contextual and h_next
- DeepDecoderLayer: clamp hidden_states after dynamics
- DeepModel: clamp accumulated mu_residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union
from dataclasses import dataclass


@dataclass
class SafetyConfig:
    """Safety configuration."""
    enabled: bool = False
    threshold: float = 2.0
    soft_clamp: bool = True
    temperature: float = 1.0
    clamp_mu: bool = True
    clamp_hidden: bool = True
    clamp_velocity: bool = False  # Already clamped by dynamics


class SafetyClamp(nn.Module):
    """
    Clamps activations along harm direction.

    projection = activation @ harm_direction
    if projection > threshold:
        activation -= (projection - threshold) * harm_direction
    """

    def __init__(
        self,
        hidden_size: int,
        threshold: float = 2.0,
        soft_clamp: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.soft_clamp = soft_clamp
        self.temperature = max(temperature, 1e-8)

        # Harm direction (initialized to zero = no effect)
        # Will be loaded from trained direction or learned during SFT
        self.register_buffer(
            'harm_direction',
            torch.zeros(hidden_size)
        )

        # Stats
        self.register_buffer('num_clamped', torch.tensor(0))
        self.register_buffer('total_processed', torch.tensor(0))

        self.enabled = False  # Disabled by default

    def set_harm_direction(self, direction: torch.Tensor):
        """Set harm direction (normalized)."""
        direction = F.normalize(direction.float(), dim=0)
        self.harm_direction.copy_(direction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp activations."""
        if not self.enabled or self.harm_direction.norm() < 1e-6:
            return x

        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])

        # Project onto harm direction
        projection = x_flat @ self.harm_direction

        # Track stats
        self.total_processed += projection.numel()
        self.num_clamped += (projection > self.threshold).sum()

        if self.soft_clamp:
            excess = projection - self.threshold
            clamp_factor = torch.sigmoid(excess / self.temperature)
            correction = clamp_factor * F.relu(excess)
        else:
            correction = F.relu(projection - self.threshold)

        # Subtract excess projection
        x_clamped = x_flat - correction.unsqueeze(-1) * self.harm_direction

        return x_clamped.view(original_shape)

    def get_stats(self) -> Dict[str, float]:
        """Get clamping statistics."""
        total = self.total_processed.item()
        if total == 0:
            return {'clamp_rate': 0.0, 'total': 0}
        return {
            'clamp_rate': self.num_clamped.item() / total,
            'num_clamped': self.num_clamped.item(),
            'total': total
        }

    def reset_stats(self):
        self.num_clamped.zero_()
        self.total_processed.zero_()


class MultiDirectionSafetyClamp(nn.Module):
    """
    Clamps multiple harm directions (categories).
    E.g., separate directions for: violence, drugs, weapons, etc.
    """

    def __init__(
        self,
        hidden_size: int,
        num_directions: int = 8,
        threshold: float = 2.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.threshold = threshold

        # Multiple harm directions [num_directions, hidden_size]
        self.register_buffer(
            'harm_directions',
            torch.zeros(num_directions, hidden_size)
        )

        # Per-direction thresholds
        self.register_buffer(
            'thresholds',
            torch.full((num_directions,), threshold)
        )

        # Which directions are active
        self.register_buffer(
            'active_mask',
            torch.zeros(num_directions, dtype=torch.bool)
        )

        self.enabled = False

    def set_direction(self, index: int, direction: torch.Tensor, threshold: float = None):
        """Set a specific harm direction."""
        direction = F.normalize(direction.float(), dim=0)
        self.harm_directions[index] = direction
        self.active_mask[index] = True
        if threshold is not None:
            self.thresholds[index] = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp all active directions."""
        if not self.enabled or not self.active_mask.any():
            return x

        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])

        # Get active directions
        active_dirs = self.harm_directions[self.active_mask]  # [K, H]
        active_thresholds = self.thresholds[self.active_mask]  # [K]

        # Project onto all directions at once
        projections = x_flat @ active_dirs.T  # [N, K]

        # Compute corrections for each direction
        for i in range(active_dirs.shape[0]):
            correction = F.relu(projections[:, i] - active_thresholds[i])
            x_flat = x_flat - correction.unsqueeze(-1) * active_dirs[i]

        return x_flat.view(original_shape)


class ContrastiveSafetyLoss(nn.Module):
    """
    Contrastive loss for learning harm direction during SFT.

    Learns harm_direction such that:
    - harmful_activations @ harm_direction is HIGH
    - safe_activations @ harm_direction is LOW
    """

    def __init__(
        self,
        hidden_size: int,
        margin: float = 1.0,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.margin = margin
        self.temperature = max(temperature, 1e-8)

        # Learnable harm direction
        self.harm_direction = nn.Parameter(
            F.normalize(torch.randn(hidden_size), dim=0)
        )

    def forward(
        self,
        safe_activations: torch.Tensor,
        harmful_activations: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive safety loss.

        Args:
            safe_activations: [batch, seq, hidden] or [batch, hidden]
            harmful_activations: Same shape

        Returns:
            Dict with loss and metrics
        """
        # Mean pool if sequence
        if safe_activations.dim() == 3:
            safe_activations = safe_activations.mean(dim=1)
            harmful_activations = harmful_activations.mean(dim=1)

        direction = F.normalize(self.harm_direction, dim=0)

        # Project
        safe_proj = safe_activations @ direction
        harmful_proj = harmful_activations @ direction

        # Margin loss
        margin_loss = F.relu(self.margin - (harmful_proj - safe_proj)).mean()

        # Contrastive loss
        logits = torch.stack([harmful_proj, safe_proj], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        contrast_loss = F.cross_entropy(logits, labels)

        total_loss = margin_loss + contrast_loss

        return {
            'loss': total_loss,
            'margin_loss': margin_loss,
            'contrast_loss': contrast_loss,
            'separation': (harmful_proj - safe_proj).mean(),
        }

    def get_direction(self) -> torch.Tensor:
        """Get normalized harm direction."""
        return F.normalize(self.harm_direction, dim=0)


def load_safety_directions(path: str, device: torch.device = None) -> Dict[str, torch.Tensor]:
    """Load safety directions from file."""
    data = torch.load(path, map_location=device or 'cpu')

    if isinstance(data, torch.Tensor):
        return {'default': data}
    elif isinstance(data, dict):
        if 'harm_direction' in data:
            return {'default': data['harm_direction']}
        elif 'directions' in data:
            return data['directions']
        else:
            return data
    else:
        raise ValueError(f"Unknown format in {path}")


def install_safety_on_model(
    model: nn.Module,
    harm_direction: torch.Tensor,
    threshold: float = 2.0,
    layers: List[int] = None,
):
    """
    Install safety clamping on a model.

    Args:
        model: Model with .layers attribute (list of DeepDecoderLayer)
        harm_direction: [hidden_size] harm direction vector
        threshold: Clamping threshold
        layers: Which layers to install on (default: last 3)
    """
    # Find layers
    if hasattr(model, 'layers'):
        model_layers = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    else:
        raise ValueError("Cannot find layers in model")

    if layers is None:
        layers = [-3, -2, -1]  # Last 3 layers by default

    # Install on each layer
    for idx in layers:
        layer = model_layers[idx]
        if hasattr(layer, 'dynamics'):
            # Create and configure safety clamp
            hidden_size = layer.dynamics.hidden_size
            safety = SafetyClamp(hidden_size, threshold=threshold)
            safety.set_harm_direction(harm_direction)
            safety.enabled = True

            # Store on dynamics module
            layer.dynamics.safety_clamp = safety


def remove_safety_from_model(model: nn.Module):
    """Remove safety clamping from model."""
    if hasattr(model, 'layers'):
        model_layers = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    else:
        return

    for layer in model_layers:
        if hasattr(layer, 'dynamics') and hasattr(layer.dynamics, 'safety_clamp'):
            del layer.dynamics.safety_clamp

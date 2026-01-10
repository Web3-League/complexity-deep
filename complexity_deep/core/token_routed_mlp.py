"""
Token-Routed MLP for Complexity architecture.

Innovation: Route tokens to specialized experts based on token ID.
Deterministic routing = no router to learn, stable, 100% parallel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP (Deterministic MoE).

    Each token is routed to a specific expert based on its token ID.

    Benefits:
    - 1/num_experts compute per token (faster)
    - Specialized experts per token frequency range
    - Deterministic = stable training, no load balancing loss
    - 100% parallel (no routing decisions at runtime)

    Token routing strategy:
        Expert 0: Token IDs 0 to vocab_size/4       (most frequent)
        Expert 1: Token IDs vocab_size/4 to /2
        Expert 2: Token IDs vocab_size/2 to 3/4
        Expert 3: Token IDs 3*vocab_size/4 to end   (most rare)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        vocab_size: int = 100000,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        # Expert size = intermediate_size / num_experts (same total params)
        self.expert_intermediate_size = intermediate_size // num_experts

        # Create experts (SwiGLU style)
        self.experts = nn.ModuleList([
            Expert(hidden_size, self.expert_intermediate_size, hidden_act)
            for _ in range(num_experts)
        ])

        # Precompute token -> expert mapping
        self.register_buffer(
            "token_to_expert",
            self._create_token_mapping(vocab_size, num_experts),
        )

    def _create_token_mapping(self, vocab_size: int, num_experts: int) -> torch.Tensor:
        """
        Create deterministic mapping from token ID to expert ID.

        Strategy: Divide vocabulary into equal ranges.
        BPE tokenizers typically sort by frequency, so:
            - Low IDs = frequent tokens
            - High IDs = rare tokens
        """
        mapping = torch.zeros(vocab_size, dtype=torch.long)
        tokens_per_expert = vocab_size // num_experts

        for expert_id in range(num_experts):
            start_idx = expert_id * tokens_per_expert
            end_idx = start_idx + tokens_per_expert if expert_id < num_experts - 1 else vocab_size
            mapping[start_idx:end_idx] = expert_id

        return mapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with token-based routing.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len] - original token IDs for routing

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        if token_ids is None:
            # Fallback: use all experts equally (for inference without token_ids)
            return self._forward_all_experts(hidden_states)

        # Get expert assignment for each token
        # Clamp to valid range
        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        expert_ids = self.token_to_expert[token_ids_clamped]  # [batch, seq_len]

        # Process each expert's tokens
        output = torch.zeros_like(hidden_states)

        for expert_id in range(self.num_experts):
            # Mask for tokens routed to this expert
            mask = (expert_ids == expert_id)  # [batch, seq_len]

            if not mask.any():
                continue

            # Get tokens for this expert
            # Flatten for efficient processing
            expert_input = hidden_states[mask]  # [num_tokens, hidden_size]

            # Process through expert
            expert_output = self.experts[expert_id](expert_input)

            # Put back in output
            output[mask] = expert_output

        return output

    def _forward_all_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback: average all experts (for inference without token_ids)."""
        outputs = [expert(hidden_states) for expert in self.experts]
        return torch.stack(outputs, dim=0).mean(dim=0)


class Expert(nn.Module):
    """Single expert MLP (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.act_fn = F.silu if hidden_act == "silu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TokenRoutedMLPParallel(nn.Module):
    """
    Optimized Token-Routed MLP using batched operations.

    Instead of looping over experts, process all at once with scatter/gather.
    Better GPU utilization for large batches.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        vocab_size: int = 100000,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        self.expert_intermediate_size = intermediate_size // num_experts

        # Batched expert weights [num_experts, hidden, intermediate]
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, self.expert_intermediate_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, self.expert_intermediate_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, self.expert_intermediate_size, hidden_size) * 0.02
        )

        self.act_fn = F.silu if hidden_act == "silu" else F.gelu

        # Token mapping
        self.register_buffer(
            "token_to_expert",
            self._create_token_mapping(vocab_size, num_experts),
        )

    def _create_token_mapping(self, vocab_size: int, num_experts: int) -> torch.Tensor:
        mapping = torch.zeros(vocab_size, dtype=torch.long)
        tokens_per_expert = vocab_size // num_experts

        for expert_id in range(num_experts):
            start_idx = expert_id * tokens_per_expert
            end_idx = start_idx + tokens_per_expert if expert_id < num_experts - 1 else vocab_size
            mapping[start_idx:end_idx] = expert_id

        return mapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batched forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        if token_ids is None:
            # Fallback: use expert 0
            expert_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
        else:
            token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
            expert_ids = self.token_to_expert[token_ids_clamped]

        # Flatten
        flat_hidden = hidden_states.view(-1, self.hidden_size)  # [B*S, H]
        flat_expert_ids = expert_ids.view(-1)  # [B*S]

        # Gather weights for each token's expert
        # gate_proj: [num_experts, H, I] -> [B*S, H, I]
        gate_weights = self.gate_proj[flat_expert_ids]  # [B*S, H, I]
        up_weights = self.up_proj[flat_expert_ids]      # [B*S, H, I]
        down_weights = self.down_proj[flat_expert_ids]  # [B*S, I, H]

        # SwiGLU: down(act(gate(x)) * up(x))
        # [B*S, 1, H] @ [B*S, H, I] -> [B*S, 1, I]
        gate_out = torch.bmm(flat_hidden.unsqueeze(1), gate_weights).squeeze(1)  # [B*S, I]
        up_out = torch.bmm(flat_hidden.unsqueeze(1), up_weights).squeeze(1)      # [B*S, I]

        intermediate = self.act_fn(gate_out) * up_out  # [B*S, I]

        # [B*S, 1, I] @ [B*S, I, H] -> [B*S, 1, H]
        output = torch.bmm(intermediate.unsqueeze(1), down_weights).squeeze(1)  # [B*S, H]

        return output.view(batch_size, seq_len, self.hidden_size)

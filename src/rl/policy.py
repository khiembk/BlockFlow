"""Adaptive token-revelation policy (docs/mypapers/main.tex, Stage 2).

Implements the factorized action distribution of spec Eq. 200-211: the policy
first samples a cardinality `m_k` (how many positions to commit this round),
then samples `m_k` distinct position indices sequentially without replacement
(a standard Plackett-Luce / pointer-network factorization). `sample_action` and
`log_prob` share the same scoring core (`_score`) so a stored action can be
re-scored under updated parameters (needed for the GRPO/PPO ratio) without the
two code paths drifting apart.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Per-position features: [hidden_state (hidden_dim), entropy, top_prob, margin,
# position_frac]. Global features: [pooled_hidden (hidden_dim), round_frac].
NUM_POSITION_SCALAR_FEATURES = 4
NUM_GLOBAL_SCALAR_FEATURES = 1


class RevelationPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden: int = 128,
        m_max: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.m_max = m_max
        self.position_mlp = nn.Sequential(
            nn.Linear(hidden_dim + NUM_POSITION_SCALAR_FEATURES, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.cardinality_mlp = nn.Sequential(
            nn.Linear(hidden_dim + NUM_GLOBAL_SCALAR_FEATURES, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, m_max),
        )

    def _score(
        self,
        position_features: torch.Tensor,
        global_features: torch.Tensor,
        active_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (position_logits (1, L), cardinality_logits (1, m_max))."""
        position_logits = self.position_mlp(position_features).squeeze(-1)
        neg_inf = torch.finfo(position_logits.dtype).min
        position_logits = position_logits.masked_fill(~active_mask, neg_inf)
        cardinality_logits = self.cardinality_mlp(global_features)
        return position_logits, cardinality_logits

    def _mask_cardinality(
        self, cardinality_logits: torch.Tensor, m_max_eff: int
    ) -> torch.Tensor:
        neg_inf = torch.finfo(cardinality_logits.dtype).min
        idx = torch.arange(cardinality_logits.shape[-1], device=cardinality_logits.device)
        valid = idx < m_max_eff
        return cardinality_logits.masked_fill(~valid[None, :], neg_inf)

    @staticmethod
    def _sequential_select(
        position_logits_row: torch.Tensor,
        m: int,
        fixed_indices: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """Samples (or re-scores, if `fixed_indices` given) an ordered sequence of
        `m` distinct indices via sequential sampling without replacement."""
        remaining = position_logits_row
        log_prob = torch.zeros((), device=position_logits_row.device, dtype=position_logits_row.dtype)
        selected = []
        neg_inf = torch.finfo(remaining.dtype).min
        for r in range(m):
            dist = torch.distributions.Categorical(logits=remaining)
            idx = fixed_indices[r] if fixed_indices is not None else dist.sample()
            log_prob = log_prob + dist.log_prob(idx)
            selected.append(idx)
            remaining = remaining.clone()
            remaining[idx] = neg_inf
        return torch.stack(selected), log_prob

    def sample_action(
        self,
        position_features: torch.Tensor,
        global_features: torch.Tensor,
        active_mask: torch.BoolTensor,
        m_max_eff: int,
    ) -> Dict[str, Any]:
        """Samples a nonempty ordered action (spec Eq. 200-209). `m_max_eff` must
        be `min(self.m_max, |A_k|)` (number of currently-active positions),
        enforced by the caller so a nonempty, in-range action is always valid."""
        assert m_max_eff >= 1, "m_max_eff must be >= 1 (nonempty active set required)."
        position_logits, cardinality_logits = self._score(
            position_features, global_features, active_mask
        )
        cardinality_logits = self._mask_cardinality(cardinality_logits, m_max_eff)
        m_dist = torch.distributions.Categorical(logits=cardinality_logits[0])
        m_idx = m_dist.sample()
        log_prob = m_dist.log_prob(m_idx)
        m = int(m_idx.item()) + 1

        indices, seq_log_prob = self._sequential_select(position_logits[0], m)
        log_prob = log_prob + seq_log_prob
        return {"m": m, "indices": indices, "log_prob": log_prob}

    def log_prob(
        self,
        position_features: torch.Tensor,
        global_features: torch.Tensor,
        active_mask: torch.BoolTensor,
        m_max_eff: int,
        action: Dict[str, Any],
    ) -> torch.Tensor:
        """Recomputes `log pi_phi(action | obs)` under current parameters (with
        grad) for a previously-sampled action — used by the GRPO/PPO ratio."""
        position_logits, cardinality_logits = self._score(
            position_features, global_features, active_mask
        )
        cardinality_logits = self._mask_cardinality(cardinality_logits, m_max_eff)
        m_dist = torch.distributions.Categorical(logits=cardinality_logits[0])
        m_idx = torch.as_tensor(
            action["m"] - 1, device=cardinality_logits.device, dtype=torch.long
        )
        log_prob = m_dist.log_prob(m_idx)

        _, seq_log_prob = self._sequential_select(
            position_logits[0], action["m"], fixed_indices=action["indices"]
        )
        return log_prob + seq_log_prob

"""Continuous-input DIT backbone for Adaptive BlockFlow's Gaussian-to-categorical VFM.

Reuses the transformer blocks, time embedder, final layer, and rotary embeddings
from ``src.backbone.dit`` unchanged. Adds a parallel continuous-state embedding
path so uncommitted (active) target positions can be fed as continuous vectors in
``R^V`` instead of integer token ids, while source/committed positions keep using
the ordinary token-embedding lookup.
"""

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.backbone.dit import DIT

log = logging.getLogger(__name__)


class ContinuousEmbedding(nn.Module):
    """Projects continuous vocabulary-space vectors (Gaussian/simplex) to model width."""

    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.proj = nn.Linear(vocab_dim, dim, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class ContinuousDIT(DIT):
    """DIT variant accepting a hybrid clean-token-id / continuous-state input.

    At each position, the embedded representation is either:
      - ``self.vocab_embed(input_ids)`` (standard integer lookup) if the position
        is clean (source or already-committed), or
      - ``self.continuous_embed(continuous_states)`` if the position is active
        (uncommitted, holding a Gaussian-to-categorical flow state).

    Time conditioning (``t``) is applied globally via the existing AdaLN-Zero
    mechanism inherited from ``DIT``/``DDiTBlock``. This is a documented
    simplification: the spec's caching contract (Section 6.1) wants clean-context
    representations independent of flow time so they can be safely cached, but
    this backbone is only ever used uncached (Stage 1, M1/M2); revisit when
    KV-caching (M3) is implemented.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not self.adaLN:
            raise ValueError("ContinuousDIT requires adaln=true for time conditioning.")
        self.continuous_embed = ContinuousEmbedding(
            dim=self.vocab_embed.embedding.shape[1], vocab_dim=self.vocab_size
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        continuous_states: torch.FloatTensor,
        active_mask: torch.BoolTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del kwargs  # No KV-cache / legacy-signature kwargs are supported here.

        clean_embed = self.vocab_embed(input_ids)
        active_embed = self.continuous_embed(continuous_states)
        x = torch.where(active_mask[..., None], active_embed, clean_embed)

        if t is None:
            raise ValueError("ContinuousDIT.forward requires `t` (flow time).")
        t_cond = F.silu(self.sigma_map(t))

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device)[
                None, :
            ].expand(input_ids.shape[0], -1)
        rotary_cos_sin = self.rotary_emb(position_ids)

        for i in range(len(self.blocks)):
            x = self.blocks[i](
                x,
                rotary_cos_sin,
                c=t_cond,
                causal=False,
                mask=attention_mask,
                use_cache=False,
            )
        logits = self.output_layer(x, t_cond)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)

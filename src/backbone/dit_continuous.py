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

    def __init__(
        self,
        *args: Any,
        pretrained_embedding_model: Optional[str] = None,
        freeze_pretrained_embeddings: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if not self.adaLN:
            raise ValueError("ContinuousDIT requires adaln=true for time conditioning.")
        self.continuous_embed = ContinuousEmbedding(
            dim=self.vocab_embed.embedding.shape[1], vocab_dim=self.vocab_size
        )
        if pretrained_embedding_model is not None:
            self._init_pretrained_token_embeddings(
                pretrained_embedding_model, freeze=freeze_pretrained_embeddings
            )

    def _init_pretrained_token_embeddings(
        self, pretrained_model_name: str, freeze: bool = True
    ) -> None:
        """Initializes `vocab_embed` (clean-token lookup) and `output_layer`'s
        projection from a pretrained model's token embedding table (e.g.
        T5-small), instead of random init. Only the vocab-tied matrices are
        affected -- `continuous_embed` (the active-position continuous-input
        projection) and the transformer body have no pretrained equivalent and
        are left as-is.

        Requires `hidden_size` to exactly equal the pretrained model's
        embedding dimension (e.g. 512 for t5-small) -- there is no learned
        projection bridging a dimension mismatch here, by design (freezing a
        projected pretrained embedding would just be training a new random
        projection anyway).
        """
        from transformers import T5EncoderModel

        log.info(
            "Initializing vocab_embed/output_layer from pretrained '%s' token embeddings",
            pretrained_model_name,
        )
        pretrained = T5EncoderModel.from_pretrained(pretrained_model_name)
        embedding = pretrained.shared.weight.detach()
        hidden_dim = self.vocab_embed.embedding.shape[1]
        if embedding.shape[1] != hidden_dim:
            raise ValueError(
                f"hidden_size={hidden_dim} must equal the pretrained model's "
                f"embedding dim={embedding.shape[1]} (e.g. 512 for t5-small) "
                "to reuse its token embeddings directly."
            )
        if embedding.shape[0] < self.vocab_size:
            raise ValueError(
                f"Pretrained embedding table has {embedding.shape[0]} rows, "
                f"fewer than our vocab_size={self.vocab_size}."
            )
        embedding = embedding[: self.vocab_size].to(
            dtype=self.vocab_embed.embedding.dtype
        )
        # T5's raw embedding table has a *much* larger absolute scale than this
        # architecture's own conventions (observed row-norm mean ~520 for
        # t5-small, vs. ~0.58 for this backbone's default EmbeddingLayer init).
        # That scale is only well-behaved inside T5's own co-designed stack
        # (its own attention/LayerNorm scaling, no absolute position embeddings,
        # tied-weight model_dim**-0.5 output correction); copying it as-is into a
        # differently-scaled architecture (rotary attention, AdaLN, standard SDPA
        # scaling) produced loss in the thousands, and even applying T5's own
        # model_dim**-0.5 correction alone still left large logit spread (row
        # norms ~23). So: unit-normalize each row first (preserving the directions
        # / cosine similarities -- the actual transferred semantic information --
        # while discarding T5's absolute scale), then apply a target scale
        # appropriate for *this* architecture's conventions.
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            # Input lookup: unit-norm rows is a reasonable embedding-layer scale
            # on its own (comparable order of magnitude to many trained embedding
            # tables), and the first LayerNorm in each transformer block
            # renormalizes activations regardless.
            self.vocab_embed.embedding.copy_(embedding)
            # Output projection: DDiTFinalLayer.linear is zero-initialized by
            # default (so initial predictions are exactly uniform); replacing it
            # with unit-norm rows scaled by hidden_dim**-0.5 keeps logits small
            # (empirically: std ~0.04 for hidden_dim=512) so initial predictions
            # stay close to uniform while still encoding the pretrained token
            # geometry, rather than reintroducing a large, unstable logit spread.
            self.output_layer.linear.weight.copy_(embedding * (hidden_dim**-0.5))
            self.output_layer.linear.bias.zero_()
        del pretrained
        if freeze:
            self.vocab_embed.embedding.requires_grad_(False)
            self.output_layer.linear.weight.requires_grad_(False)
            self.output_layer.linear.bias.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        continuous_states: torch.FloatTensor,
        active_mask: torch.BoolTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
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
        hidden_states = (x,) if output_hidden_states else None
        return CausalLMOutputWithPast(
            logits=logits, past_key_values=None, hidden_states=hidden_states
        )

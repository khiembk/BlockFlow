"""DIT backbone for Adaptive BlockFlow Stage 1 v3: Gaussian-to-T5-embedding VFM
with a separately trained decoder branch (ELF-informed, see
docs/mypapers/main.tex and the Stage 1 v3 plan).

This is a *parallel* backbone to `src.backbone.dit_continuous.ContinuousDIT`
(not a modification of it): the flow now operates in a frozen T5-small
encoder's 512-dim output space instead of one-hot vocabulary space, bridged
into this project's own `hidden_size` via a trainable `BottleneckAdapter`
shared by both the clean-context path and the active-flow path. There is no
`vocab_embed`/`output_layer`-as-classifier pair tied to pretrained weights
(the earlier, abandoned approach) -- the only thing ever consumed from T5 is
its frozen forward-pass output activations (computed upstream, outside this
backbone, by `T5ContextEncoder`), and token decoding happens through a
dedicated, separately trained decoder branch.
"""

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.backbone.dit import DDiTFinalLayer, DIT
from src.backbone.t5_context_encoder import BottleneckAdapter

log = logging.getLogger(__name__)


class EmbeddingSpaceContinuousDIT(DIT):
    """DIT variant operating on a continuous T5-embedding-space input.

    At each position, the pre-adapter representation is either:
      - `context_latent` (the frozen T5 encoder's contextualized output,
        normalized) if the position is clean (source or already-committed), or
      - `continuous_states` (the Gaussian-to-T5-embedding flow state) if the
        position is active (uncommitted).
    Both paths are bridged to `hidden_size` by the *same* `BottleneckAdapter`
    instance, since both live in the same 512-dim T5 latent space.

    Two forward modes, selected by `decoder_mode`:
      - `decoder_mode=False` (the main flow/denoiser branch): the existing
        AdaLN-conditioned `output_layer` is repurposed as a zero-init
        regression head predicting the clean T5 embedding (`hidden_size ->
        t5_embed_dim`), trained with masked MSE.
      - `decoder_mode=True` (the separately trained decoder branch, per ELF):
        a factored unembedding head (`hidden_size -> bottleneck_dim ->
        vocab_size`) predicts token logits, trained with masked CE on a
        dedicated input-corruption schedule (see `AdaptiveBlockFlowEmbed`).

    Mode-gating simplification (documented, analogous to ContinuousDIT's own
    documented AdaLN simplification): ELF marks decoder-branch forward passes
    by prepending learnable "mode tokens" to the sequence. Doing that here
    would require extending this backbone's position-id/rotary/attention-mask
    bookkeeping to a variable, mode-dependent sequence length. Instead, since
    this architecture already threads a global per-forward-pass conditioning
    signal through every block via AdaLN (`t_cond`), decoder-mode is signaled
    the same way: a learned `decoder_mode_bias` vector is added to `t_cond`
    whenever `decoder_mode=True`. This still gates the *entire* transformer
    body's behavior on a learned, mode-specific signal (ELF's actual intent),
    without changing sequence length or mask shapes.
    """

    def __init__(
        self,
        *args: Any,
        t5_embed_dim: int = 512,
        bottleneck_dim: int = 128,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if not self.adaLN:
            raise ValueError("EmbeddingSpaceContinuousDIT requires adaln=true.")
        dim = self.vocab_embed.embedding.shape[1]
        cond_dim = self.sigma_map.mlp[-1].out_features
        # `vocab_embed` (discrete token lookup) has no role here -- both clean
        # and active positions enter through the T5-embedding-space adapter.
        del self.vocab_embed

        self.t5_embed_dim = t5_embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.adapter = BottleneckAdapter(
            embed_dim=t5_embed_dim, bottleneck_dim=bottleneck_dim, hidden_size=dim
        )
        # Flow/denoiser branch: zero-init regression head (hidden -> t5_embed_dim),
        # replacing the vocab-sized classifier `DIT.__init__` built by default.
        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=t5_embed_dim,
            cond_dim=cond_dim,
            adaLN=True,
            tie_word_embeddings=False,
        )
        # Decoder branch: learned mode signal + factored unembedding head.
        self.decoder_mode_bias = nn.Parameter(torch.zeros(cond_dim))
        self.decoder_proj = nn.Linear(dim, bottleneck_dim, bias=True)
        self.decoder_unembed = nn.Linear(bottleneck_dim, self.vocab_size, bias=True)
        nn.init.kaiming_uniform_(self.decoder_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.decoder_unembed.weight)
        nn.init.zeros_(self.decoder_unembed.bias)

    def forward(
        self,
        context_latent: torch.FloatTensor,
        continuous_states: torch.FloatTensor,
        active_mask: torch.BoolTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_mode: bool = False,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del kwargs  # No KV-cache / legacy-signature kwargs are supported here.

        merged_latent = torch.where(active_mask[..., None], continuous_states, context_latent)
        x = self.adapter(merged_latent)

        if t is None:
            raise ValueError("EmbeddingSpaceContinuousDIT.forward requires `t`.")
        t_cond = F.silu(self.sigma_map(t))
        if decoder_mode:
            t_cond = t_cond + self.decoder_mode_bias

        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device)[None, :].expand(
                x.shape[0], -1
            )
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

        if decoder_mode:
            pred = self.decoder_unembed(F.gelu(self.decoder_proj(x)))
        else:
            pred = self.output_layer(x, t_cond)

        hidden_states = (x,) if output_hidden_states else None
        return CausalLMOutputWithPast(
            logits=pred, past_key_values=None, hidden_states=hidden_states
        )

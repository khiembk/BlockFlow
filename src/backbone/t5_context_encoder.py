"""Frozen T5 encoder + trainable bottleneck adapter, for Adaptive BlockFlow Stage 1
v3's embedding-space flow matching (ELF-informed, see docs/mypapers/main.tex and
the Stage 1 v3 plan).

Unlike the earlier (abandoned) attempt that froze *raw T5 embedding-table weights*
inside this project's own vocab-tied layers -- which hit an unrecoverable scale
mismatch (T5's embedding rows have ~1000x the norm of this architecture's own
init convention) -- this module only ever consumes the frozen T5 encoder's
*forward-pass output activations*, which are already well-scaled by T5's own
internal LayerNorms (empirically measured std ~0.15 for t5-small, see the
`latent_std` default below), and bridges them into this project's own hidden
size via a small trainable adapter, never copying T5's weights anywhere.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import T5EncoderModel

log = logging.getLogger(__name__)


class T5ContextEncoder(nn.Module):
    """Wraps a frozen `T5EncoderModel`, exposing a normalized latent space.

    `latent_std=0.15` is an empirical measurement of t5-small's own
    last_hidden_state std (see scratch measurement script referenced in the
    Stage 1 v3 plan), in the same ballpark as ELF/CorrFlow's hand-picked 0.2 --
    both are activation-scale normalization constants, not weight copies, so
    neither carries the earlier raw-embedding-weight scale bug.
    """

    def __init__(
        self,
        pretrained_model_name: str = "t5-small",
        latent_mean: float = 0.0,
        latent_std: float = 0.15,
    ):
        super().__init__()
        log.info("Loading frozen T5 context encoder from '%s'", pretrained_model_name)
        self.encoder = T5EncoderModel.from_pretrained(pretrained_model_name)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.embed_dim = self.encoder.config.d_model

    def train(self, mode: bool = True) -> "T5ContextEncoder":
        # Keep the frozen encoder in eval mode (no dropout) regardless of the
        # parent module's train()/eval() calls.
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Runs the frozen encoder and normalizes its output activations."""
        latents = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return (latents - self.latent_mean) / self.latent_std


class BottleneckAdapter(nn.Module):
    """Bridges T5's embed_dim (512 for t5-small) to this architecture's own
    hidden size, through a low-rank bottleneck (mirrors CorrFlow's
    `BottleneckTextProj`). Trainable, and shared by both the clean-context
    path (frozen T5 encoder output) and the active-flow path (continuous
    interpolant), since both inputs live in the same 512-dim T5 latent space.
    """

    def __init__(self, embed_dim: int, bottleneck_dim: int, hidden_size: int):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.down_proj(x))

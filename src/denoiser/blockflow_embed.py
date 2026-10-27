"""Stage 1 v3 of Adaptive BlockFlow: Gaussian-to-T5-embedding variational flow
matching with a separately trained decoder branch (ELF-informed, see
docs/mypapers/main.tex and the Stage 1 v3 plan). Subclasses `AdaptiveBlockFlow`
to reuse its revelation-round sampling (`_sample_revelation_masks`), its 3-way
attention contract (`_build_attention_mask`), and its round loop/fixed-b
commitment/EOS logic (`generate`, `rollout_for_grpo`) completely unchanged --
only the representation the flow operates in, and how continuous predictions
become discrete tokens, differ from `AdaptiveBlockFlow`.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput

from src.backbone.t5_context_encoder import T5ContextEncoder
from src.denoiser.base import DenoiserInput, LossAndNllOutput
from src.denoiser.blockflow import AdaptiveBlockFlow, AdaptiveBlockFlowConfig

log = logging.getLogger(__name__)


class AdaptiveBlockFlowEmbedConfig(AdaptiveBlockFlowConfig):
    """Config for Stage 1 v3 (embedding-space Adaptive BlockFlow)."""

    def __init__(
        self,
        t5_model_name: str = "t5-small",
        t5_embed_dim: int = 512,
        latent_mean: float = 0.0,
        latent_std: float = 0.15,
        noise_scale: float = 2.0,
        decoder_prob: float = 0.2,
        decoder_p_mean: float = -1.5,
        decoder_p_std: float = 0.8,
        decoder_noise_scale: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.t5_model_name = t5_model_name
        self.t5_embed_dim = t5_embed_dim
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.noise_scale = noise_scale
        self.decoder_prob = decoder_prob
        self.decoder_p_mean = decoder_p_mean
        self.decoder_p_std = decoder_p_std
        self.decoder_noise_scale = decoder_noise_scale


class AdaptiveBlockFlowEmbed(AdaptiveBlockFlow):
    """Stage-1 v3: the flow target is a frozen T5-small encoder's contextualized
    512-dim output embedding (not one-hot vocab space). Every training step is
    *either* the denoiser (masked MSE, Gaussian-to-T5-embedding flow) branch
    *or* the decoder (masked CE, factored unembedding head) branch, chosen by a
    `decoder_prob`-weighted coin flip per step -- not summed every step, per
    ELF's own recipe. The revelation-round structure (predict all remaining
    positions jointly, commit `b` per round) and the 3-way attention contract
    are inherited from `AdaptiveBlockFlow` unmodified.
    """

    config_class = AdaptiveBlockFlowEmbedConfig

    def __init__(
        self,
        config: AdaptiveBlockFlowEmbedConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs: Any,
    ):
        super().__init__(config, tokenizer, **kwargs)
        self.t5_embed_dim = config.t5_embed_dim
        self.noise_scale = config.noise_scale
        self.decoder_prob = config.decoder_prob
        self.decoder_p_mean = config.decoder_p_mean
        self.decoder_p_std = config.decoder_p_std
        self.decoder_noise_scale = config.decoder_noise_scale
        self.t5_encoder = T5ContextEncoder(
            pretrained_model_name=config.t5_model_name,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
        )
        backbone_t5_embed_dim = getattr(self.backbone, "t5_embed_dim", None)
        if backbone_t5_embed_dim is not None and backbone_t5_embed_dim != self.t5_embed_dim:
            raise ValueError(
                f"backbone_config.t5_embed_dim={backbone_t5_embed_dim} must match "
                f"config.t5_embed_dim={self.t5_embed_dim}."
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
    ) -> DenoiserInput:
        del past_key_values
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(input_ids)
        attention_mask = attention_mask.to(torch.bool)
        context_mask = context_mask.to(torch.bool)

        _committed_mask, active_mask, clean_mask, position_idx = (
            self._sample_revelation_masks(input_ids, attention_mask, context_mask)
        )

        # One frozen-encoder pass over the whole teacher-forced sequence gets
        # every position's contextualized embedding at once; used as the
        # clean-context input at clean positions *and* the regression/decoder
        # target at active positions (spec Stage 1 v3 plan, Key design #2).
        context_latent = self.t5_encoder.encode(
            input_ids.clamp(min=0), attention_mask.to(torch.long)
        )

        is_decoder_step = bool(torch.rand((), device=device).item() < self.decoder_prob)
        if is_decoder_step:
            lambda_t = torch.sigmoid(
                torch.randn(batch_size, seq_len, device=device) * self.decoder_p_std
                + self.decoder_p_mean
            )
            noise = (
                torch.randn(batch_size, seq_len, self.t5_embed_dim, device=device)
                * self.decoder_noise_scale
            )
            continuous_states = (
                lambda_t[..., None] * context_latent + (1 - lambda_t[..., None]) * noise
            )
            t_cond = lambda_t.mean(dim=-1)
        else:
            if t is None:
                t = (
                    torch.rand(batch_size, device=device) * (1 - 2 * self.sampling_eps)
                    + self.sampling_eps
                )
            t = t.to(torch.float32)
            epsilon = (
                torch.randn(batch_size, seq_len, self.t5_embed_dim, device=device)
                * self.noise_scale
            )
            t_broadcast = t[:, None, None]
            continuous_states = (1 - t_broadcast) * epsilon + t_broadcast * context_latent
            t_cond = t

        bool_mask = self._build_attention_mask(
            clean_mask=clean_mask, active_mask=active_mask, position_idx=position_idx
        )
        bool_mask = bool_mask & attention_mask[:, None, :]
        float_mask = self._preprocess_attention_mask(
            bool_mask[:, None, :, :], dtype=torch.float32
        )

        tokens_mask = active_mask.to(torch.float32)

        return DenoiserInput(
            xt=input_ids,
            x0=input_ids,
            attention_mask=float_mask,
            context_mask=context_mask.to(torch.float32),
            tokens_mask=tokens_mask,
            active_mask=active_mask,
            continuous_states=continuous_states,
            t=t_cond,
            backbone_kwargs={
                "position_ids": position_idx,
                "t": t_cond,
                "decoder_mode": is_decoder_step,
                "context_latent": context_latent,
            },
        )

    def _backbone_forward(
        self,
        denoiser_inputs: DenoiserInput,
        **backbone_kwargs: Any,
    ) -> ModelOutput:
        return self.backbone(
            continuous_states=denoiser_inputs.continuous_states,
            active_mask=denoiser_inputs.active_mask,
            attention_mask=denoiser_inputs.attention_mask,
            **denoiser_inputs.backbone_kwargs,
            **backbone_kwargs,
        )

    def _forward(
        self,
        backbone_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        del kwargs
        if denoiser_inputs.backbone_kwargs.get("decoder_mode", False):
            return backbone_output.log_softmax(-1)
        return backbone_output  # raw T5-embedding regression prediction

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        del kwargs
        active = denoiser_inputs.tokens_mask
        count = active.sum(dim=-1)
        is_decoder_step = denoiser_inputs.backbone_kwargs.get("decoder_mode", False)
        if is_decoder_step:
            log_p_theta = torch.gather(
                model_output, dim=-1, index=denoiser_inputs.x0[:, :, None]
            ).squeeze(-1)
            loss_per_token = -log_p_theta
        else:
            target = denoiser_inputs.backbone_kwargs["context_latent"]
            loss_per_token = F.mse_loss(model_output, target, reduction="none").mean(dim=-1)
        nlls = loss_per_token * active
        batch_nll = nlls.sum(dim=-1)
        token_nll = torch.where(
            count > 0, batch_nll / count, torch.zeros_like(batch_nll)
        ).mean()
        return LossAndNllOutput(
            loss=token_nll,
            nlls=nlls,
            other_loss_terms={"is_decoder_step": float(is_decoder_step)},
        )

    # ------------------------------------------------------------------
    # Generation: shared round loop/fixed-b commitment/EOS logic is inherited
    # unchanged from `AdaptiveBlockFlow.generate`/`rollout_for_grpo`, which
    # call `self._solve_flow_round(...)` polymorphically -- only the flow
    # solve itself (representation space + decoder read-out) differs here.
    # ------------------------------------------------------------------
    def _solve_flow_round(
        self,
        sequence: torch.LongTensor,
        committed_mask: torch.BoolTensor,
        attention_mask_valid: torch.BoolTensor,
        canvas_length: int,
        position_idx: torch.LongTensor,
        num_flow_steps: int,
        request_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, torch.BoolTensor, Optional[torch.Tensor], int]:
        """Solves the Gaussian-to-T5-embedding flow over active positions for one
        revelation round, then runs one extra decoder-branch forward pass on the
        converged latent to obtain token probabilities (the flow branch itself
        only predicts continuous T5 embeddings, not token logits -- see Stage 1
        v3 plan, Key design #3). NFE counts both the Euler steps and this final
        decoder read-out pass.

        Clean (committed/source) positions are always a contiguous prefix of
        the canvas under left-to-right commitment, so their contextual T5
        embedding is obtained by encoding just that prefix (the only tokens
        actually known at generation time) -- unlike training, where the whole
        teacher-forced sequence is known and encoded at once.
        """
        device = sequence.device
        active_mask = (~committed_mask) & attention_mask_valid
        clean_mask = committed_mask & attention_mask_valid
        clean_len = int(clean_mask[0].sum().item())

        context_latent = torch.zeros(1, canvas_length, self.t5_embed_dim, device=device)
        if clean_len > 0:
            context_latent[:, :clean_len] = self.t5_encoder.encode(
                sequence[:, :clean_len], attention_mask_valid[:, :clean_len].to(torch.long)
            )

        bool_mask = self._build_attention_mask(
            clean_mask=clean_mask, active_mask=active_mask, position_idx=position_idx
        )
        bool_mask = bool_mask & attention_mask_valid[:, None, :]
        float_mask = self._preprocess_attention_mask(
            bool_mask[:, None, :, :], dtype=torch.float32
        )

        z = (
            torch.randn(1, canvas_length, self.t5_embed_dim, device=device)
            * self.noise_scale
        )
        timesteps = torch.linspace(0.0, 1.0, num_flow_steps + 1, device=device)
        nfe = 0
        for step in range(num_flow_steps):
            t_cur = timesteps[step].expand(1)
            t_next = timesteps[step + 1]
            backbone_output = self._backbone_forward(
                DenoiserInput(
                    xt=sequence,
                    continuous_states=z,
                    active_mask=active_mask,
                    attention_mask=float_mask,
                    backbone_kwargs={
                        "position_ids": position_idx,
                        "t": t_cur,
                        "decoder_mode": False,
                        "context_latent": context_latent,
                    },
                )
            )
            nfe += 1
            x0_pred = backbone_output.logits
            velocity = (x0_pred - z) / (1.0 - t_cur)
            z_new = z + (t_next - t_cur) * velocity
            z = torch.where(active_mask[..., None], z_new, z)

        decoder_output = self._backbone_forward(
            DenoiserInput(
                xt=sequence,
                continuous_states=z,
                active_mask=active_mask,
                attention_mask=float_mask,
                backbone_kwargs={
                    "position_ids": position_idx,
                    "t": torch.ones(1, device=device),
                    "decoder_mode": True,
                    "context_latent": context_latent,
                },
            ),
            output_hidden_states=request_hidden_states,
        )
        nfe += 1
        p_theta = decoder_output.logits.softmax(dim=-1)
        hidden_state = (
            decoder_output.hidden_states[0] if request_hidden_states else None
        )

        return p_theta, active_mask, hidden_state, nfe

"""Stage 1 of Adaptive BlockFlow (docs/mypapers/main.tex): a Gaussian-to-categorical
variational flow matching (VFM) denoiser with fixed-size, all-remaining-token
commitment. See the spec's Sections 1-7 for the full algorithmic contract; this
module implements milestones M1 (VFM training) and M2 (uncached fixed-b
generation) only. No KV-caching (M3) or GRPO (M4) is implemented here.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput

from src.denoiser.base import (
    Denoiser,
    DenoiserConfig,
    DenoiserInput,
    LossAndNllOutput,
)

log = logging.getLogger(__name__)


@dataclass
class BlockFlowGenerationOutput(ModelOutput):
    """Output of `AdaptiveBlockFlow.generate`."""

    sequences: Optional[torch.LongTensor] = None
    nfe: Optional[int] = None
    num_rounds: Optional[int] = None


class AdaptiveBlockFlowConfig(DenoiserConfig):
    """Config for Adaptive BlockFlow (Stage 1)."""

    def __init__(
        self,
        commit_size: int = 1,
        sampling_eps: float = 1e-3,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.commit_size = commit_size
        self.sampling_eps = sampling_eps


class AdaptiveBlockFlow(Denoiser):
    """Stage-1 Adaptive BlockFlow denoiser.

    Every uncommitted ("active") target position is given a continuous
    interpolant z_t = (1-t) x0 + t * one_hot(y) (spec Eq. 1); the backbone
    predicts a categorical endpoint distribution for all active positions
    jointly, supervised by cross-entropy over the full active set (spec Eq. 3).
    Training uses a fixed left-to-right commitment schedule of size
    `commit_size` (`b`, spec Section 5.1). `generate` performs a full, uncached
    flow solve over all remaining positions each round and commits only `b` of
    them (spec Algorithm 2).
    """

    config_class = AdaptiveBlockFlowConfig

    def __init__(
        self,
        config: AdaptiveBlockFlowConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs: Any,
    ):
        super().__init__(config, tokenizer, **kwargs)
        self.commit_size = config.commit_size
        self.sampling_eps = config.sampling_eps

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

        # tokens_mask convention matches setdlm.py/mdlm.py: valid & non-context.
        target_valid = attention_mask & ~context_mask
        position_idx = torch.arange(seq_len, device=device)[None, :].expand(
            batch_size, -1
        )

        target_len = target_valid.sum(dim=-1)  # (B,)
        has_target = target_len > 0
        target_start = target_valid.float().argmax(dim=-1)  # (B,); 0 if no target

        # Sample a nonterminal revelation round k in {0, ..., num_rounds-1}
        # (spec Algorithm 1): committed_count = min(k*b, target_len) is always
        # strictly less than target_len for k < ceil(target_len / b), so the
        # active set is guaranteed nonempty whenever has_target is True.
        safe_target_len = torch.where(
            has_target, target_len, torch.ones_like(target_len)
        )
        num_rounds = torch.div(
            safe_target_len + self.commit_size - 1,
            self.commit_size,
            rounding_mode="floor",
        )
        round_idx = (
            (torch.rand(batch_size, device=device) * num_rounds.float())
            .floor()
            .long()
        )
        round_idx = round_idx.clamp(min=0)
        round_idx = torch.minimum(round_idx, (num_rounds - 1).clamp(min=0))
        committed_count = torch.clamp(round_idx * self.commit_size, max=safe_target_len)
        committed_count = torch.where(
            has_target, committed_count, torch.zeros_like(committed_count)
        )

        target_rank = position_idx - target_start[:, None]
        committed_mask = target_valid & (target_rank < committed_count[:, None])
        active_mask = target_valid & ~committed_mask
        clean_mask = (context_mask | committed_mask) & attention_mask

        # Continuous interpolant (spec Eq. 1).
        if t is None:
            t = (
                torch.rand(batch_size, device=device) * (1 - 2 * self.sampling_eps)
                + self.sampling_eps
            )
        t = t.to(torch.float32)
        x0_noise = torch.randn(
            batch_size, seq_len, self.vocab_size, device=device, dtype=torch.float32
        )
        one_hot_target = F.one_hot(input_ids.clamp(min=0), self.vocab_size).to(
            torch.float32
        )
        t_broadcast = t[:, None, None]
        continuous_states = (1 - t_broadcast) * x0_noise + t_broadcast * one_hot_target

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
            t=t,
            backbone_kwargs={"position_ids": position_idx, "t": t},
        )

    @staticmethod
    def _build_attention_mask(
        clean_mask: torch.BoolTensor,
        active_mask: torch.BoolTensor,
        position_idx: torch.LongTensor,
    ) -> torch.BoolTensor:
        """Three-way source/committed/active attention contract (spec Section 6.1).

        clean queries may only attend causally to earlier-or-equal clean keys
        (source always precedes committed targets, and committed targets are
        ordered by their absolute position under left-to-right commitment);
        active queries attend to all clean keys and all other active keys.
        """
        causal_among_clean = position_idx[:, None, :] <= position_idx[:, :, None]
        clean_q = clean_mask[:, :, None]
        clean_k = clean_mask[:, None, :]
        active_q = active_mask[:, :, None]
        active_k = active_mask[:, None, :]
        return (
            (clean_q & clean_k & causal_among_clean)
            | (active_q & clean_k)
            | (active_q & active_k)
        )

    def _backbone_forward(
        self,
        denoiser_inputs: DenoiserInput,
        **backbone_kwargs: Any,
    ) -> ModelOutput:
        return self.backbone(
            input_ids=denoiser_inputs.xt,
            continuous_states=denoiser_inputs.continuous_states,
            active_mask=denoiser_inputs.active_mask,
            attention_mask=denoiser_inputs.attention_mask,
            **denoiser_inputs.backbone_kwargs,
            **backbone_kwargs,
        )

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        del kwargs
        log_p_theta = torch.gather(
            model_output, dim=-1, index=denoiser_inputs.x0[:, :, None]
        ).squeeze(-1)
        loss = -log_p_theta
        nlls = loss * denoiser_inputs.tokens_mask
        count = denoiser_inputs.tokens_mask.sum(dim=-1)
        batch_nll = nlls.sum(dim=-1)
        token_nll = torch.where(
            count > 0, batch_nll / count, torch.zeros_like(batch_nll)
        ).mean()
        return LossAndNllOutput(loss=token_nll, nlls=nlls, other_loss_terms={})

    # ------------------------------------------------------------------
    # Generation (spec Algorithm 2: all-remaining flow, fixed commitment, uncached)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
        num_flow_steps: int = 16,
        do_sample: bool = False,
        return_dict_in_generate: bool = False,
        batch_size: int = 1,
        device: Optional[str] = None,
        history: Optional[list] = None,
        **kwargs: Any,
    ):
        del kwargs
        assert inputs is not None, "AdaptiveBlockFlow.generate requires `inputs`."
        assert inputs.shape[0] == 1, "Only batch size 1 is supported for inference."
        assert max_new_tokens is not None and max_new_tokens > 0
        device = inputs.device if device is None else device
        input_length = inputs.shape[-1]
        canvas_length = input_length + max_new_tokens

        if attention_mask is None:
            attention_mask = torch.ones_like(inputs)
        attention_mask_valid = torch.cat(
            [attention_mask.bool(), torch.ones(1, max_new_tokens, dtype=torch.bool, device=device)],
            dim=-1,
        )
        sequence = torch.cat(
            [inputs, torch.zeros(1, max_new_tokens, dtype=torch.long, device=device)],
            dim=-1,
        )
        committed_mask = torch.cat(
            [
                torch.ones_like(inputs, dtype=torch.bool),
                torch.zeros(1, max_new_tokens, dtype=torch.bool, device=device),
            ],
            dim=-1,
        )
        position_idx = torch.arange(canvas_length, device=device)[None, :]

        total_nfe = 0
        num_rounds = 0
        finished = False
        while committed_mask[:, input_length:].sum() < max_new_tokens and not finished:
            active_mask = (~committed_mask) & attention_mask_valid
            num_active = int(active_mask.sum().item())
            if num_active == 0:
                break
            clean_mask = committed_mask & attention_mask_valid
            bool_mask = self._build_attention_mask(
                clean_mask=clean_mask, active_mask=active_mask, position_idx=position_idx
            )
            bool_mask = bool_mask & attention_mask_valid[:, None, :]
            float_mask = self._preprocess_attention_mask(
                bool_mask[:, None, :, :], dtype=torch.float32
            )

            z = torch.randn(1, canvas_length, self.vocab_size, device=device)
            timesteps = torch.linspace(0.0, 1.0, num_flow_steps + 1, device=device)
            p_theta = None
            for step in range(num_flow_steps):
                t_cur = timesteps[step].expand(1)
                t_next = timesteps[step + 1]
                backbone_output = self._backbone_forward(
                    DenoiserInput(
                        xt=sequence,
                        continuous_states=z,
                        active_mask=active_mask,
                        attention_mask=float_mask,
                        backbone_kwargs={"position_ids": position_idx, "t": t_cur},
                    )
                )
                total_nfe += 1
                logits = getattr(backbone_output, "logits", backbone_output)
                p_theta = logits.softmax(dim=-1)
                velocity = (p_theta - z) / (1.0 - t_cur)
                z_new = z + (t_next - t_cur) * velocity
                z = torch.where(active_mask[..., None], z_new, z)

            candidate_ids = self._sample_categorical(p_theta, do_sample=do_sample)
            active_positions = active_mask[0].nonzero(as_tuple=True)[0]
            commit_positions = active_positions[: self.commit_size]
            sequence[0, commit_positions] = candidate_ids[0, commit_positions]
            committed_mask[0, commit_positions] = True
            num_rounds += 1
            if history is not None:
                # Testing/debugging hook (spec correctness test: commit immutability).
                history.append((commit_positions.detach().clone(), sequence.detach().clone()))

            if self.eos_token_id is not None:
                committed_ids = sequence[0, commit_positions]
                eos_hits = (committed_ids == self.eos_token_id).nonzero(as_tuple=True)[0]
                if eos_hits.numel() > 0:
                    eos_pos = commit_positions[eos_hits[0]].item()
                    sequence = sequence[:, : eos_pos + 1]
                    finished = True

        if return_dict_in_generate:
            return BlockFlowGenerationOutput(
                sequences=sequence, nfe=total_nfe, num_rounds=num_rounds
            )
        return sequence

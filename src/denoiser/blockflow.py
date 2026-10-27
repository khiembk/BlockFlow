"""Stage 1 of Adaptive BlockFlow (docs/mypapers/main.tex): a Gaussian-to-categorical
variational flow matching (VFM) denoiser with fixed-size, all-remaining-token
commitment. See the spec's Sections 1-7 for the full algorithmic contract; this
module implements milestones M1 (VFM training) and M2 (uncached fixed-b
generation) only. No KV-caching (M3) or GRPO (M4) is implemented here.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
from src.rl.policy import RevelationPolicy

log = logging.getLogger(__name__)


@dataclass
class BlockFlowGenerationOutput(ModelOutput):
    """Output of `AdaptiveBlockFlow.generate`."""

    sequences: Optional[torch.LongTensor] = None
    nfe: Optional[int] = None
    num_rounds: Optional[int] = None


@dataclass
class RolloutStep:
    """One revelation round of a Stage-2 GRPO rollout (spec Section "Stage 2:
    Policy state and action"). Observation tensors are detached (generator theta
    is frozen; policy features never backprop into it)."""

    position_features: torch.Tensor  # (1, L, hidden_dim + 4), detached
    global_features: torch.Tensor  # (1, hidden_dim + 1), detached
    active_mask: torch.BoolTensor  # (1, L), detached
    m_max_eff: int
    action: Dict[str, Any]  # {"m": int, "indices": LongTensor, "log_prob": Tensor|None}
    is_forced: bool  # True if this round's commitment was a forced-budget action
    # (spec: "Forced last-round actions must be deterministic and excluded from
    # the stochastic policy loss") rather than a stochastic policy sample.


@dataclass
class RolloutTrace:
    """Full Stage-2 rollout trajectory for one source, used by GRPO."""

    steps: list[RolloutStep]
    sequence: torch.LongTensor
    total_nfe: int
    num_rounds: int


class AdaptiveBlockFlowConfig(DenoiserConfig):
    """Config for Adaptive BlockFlow (Stage 1)."""

    def __init__(
        self,
        commit_size: int = 1,
        sampling_eps: float = 1e-3,
        window_size: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.commit_size = commit_size
        self.sampling_eps = sampling_eps
        self.window_size = window_size


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

    `window_size` (optional) bounds what the *loss* scores to the nearest
    `window_size` active positions in canvas order, instead of every active
    position (which can be nearly the full remaining target length in early
    rounds). This doesn't change the round/commit mechanics, the attention
    contract, or the continuous interpolant at all -- every active position is
    still given an interpolant and attended to exactly as before, so there is
    no risk of degenerate all-masked attention rows. It only concentrates
    gradient signal on the near-term positions that actually get committed
    soon, matching this repo's BD3LM/SetDLM configs, which denoise a bounded
    8-16 token unit per step rather than an unbounded remaining span.
    `window_size=None` (default) preserves scoring every active position.
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
        self.window_size = config.window_size

    def _window_tokens_mask(self, active_mask: torch.BoolTensor) -> torch.FloatTensor:
        """Restricts the scored (loss) positions to the nearest `window_size`
        active positions in canvas order. See class docstring."""
        if self.window_size is None:
            return active_mask.to(torch.float32)
        active_rank = torch.cumsum(active_mask.to(torch.long), dim=-1) - 1
        windowed = active_mask & (active_rank < self.window_size)
        return windowed.to(torch.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _sample_revelation_masks(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        context_mask: torch.BoolTensor,
    ) -> tuple[
        torch.BoolTensor, torch.BoolTensor, torch.BoolTensor, torch.LongTensor
    ]:
        """Samples a nonterminal revelation round and builds the resulting
        committed/active/clean masks (spec Algorithm 1). Pure mask logic, no
        interpolant construction -- shared by `AdaptiveBlockFlow` (one-hot
        vocab-space interpolant) and `AdaptiveBlockFlowEmbed` (T5-embedding-space
        interpolant), which only differ in what continuous representation they
        build on top of these masks.

        Returns `(committed_mask, active_mask, clean_mask, position_idx)`.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

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
        return committed_mask, active_mask, clean_mask, position_idx

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

        tokens_mask = self._window_tokens_mask(active_mask)

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
        """Solve the Gaussian-to-categorical flow over all active positions for one
        revelation round (spec Algorithm 2, inner loop / Eq. 4 Euler sampler).

        Shared by both `generate` (fixed-`b` commitment) and `rollout_for_grpo`
        (policy-driven commitment) so there is exactly one tested implementation
        of "solve the flow over everything not yet committed."

        Returns `(p_theta, active_mask, hidden_state, nfe)`, where `hidden_state`
        is the backbone's final pre-output-layer representation from the last
        Euler step (only computed/returned if `request_hidden_states=True`).
        """
        device = sequence.device
        active_mask = (~committed_mask) & attention_mask_valid
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
        hidden_state = None
        nfe = 0
        for step in range(num_flow_steps):
            t_cur = timesteps[step].expand(1)
            t_next = timesteps[step + 1]
            is_last_step = step == num_flow_steps - 1
            backbone_output = self._backbone_forward(
                DenoiserInput(
                    xt=sequence,
                    continuous_states=z,
                    active_mask=active_mask,
                    attention_mask=float_mask,
                    backbone_kwargs={"position_ids": position_idx, "t": t_cur},
                ),
                output_hidden_states=(request_hidden_states and is_last_step),
            )
            nfe += 1
            logits = getattr(backbone_output, "logits", backbone_output)
            p_theta = logits.softmax(dim=-1)
            if request_hidden_states and is_last_step:
                hidden_state = backbone_output.hidden_states[0]
            velocity = (p_theta - z) / (1.0 - t_cur)
            z_new = z + (t_next - t_cur) * velocity
            z = torch.where(active_mask[..., None], z_new, z)

        return p_theta, active_mask, hidden_state, nfe

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
            if int(active_mask.sum().item()) == 0:
                break
            p_theta, active_mask, _, nfe = self._solve_flow_round(
                sequence=sequence,
                committed_mask=committed_mask,
                attention_mask_valid=attention_mask_valid,
                canvas_length=canvas_length,
                position_idx=position_idx,
                num_flow_steps=num_flow_steps,
            )
            total_nfe += nfe

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

    # ------------------------------------------------------------------
    # Stage 2: policy-driven rollout for GRPO (spec Algorithm 3, rollout half)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_observation(
        p_theta: torch.Tensor,
        hidden_state: torch.Tensor,
        active_mask: torch.BoolTensor,
        clean_mask: torch.BoolTensor,
        position_idx: torch.LongTensor,
        canvas_length: int,
        round_frac: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds reference-free policy features (spec Section "Stage 2: Policy
        state and action", Eq. o_k): per-position [hidden, entropy, top_prob,
        margin, position_frac] and a pooled-source global feature. Everything is
        detached — the frozen generator never receives policy gradients."""
        probs = p_theta.detach()
        hidden = hidden_state.detach()
        top2 = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
        top_prob = top2[..., 0]
        second_prob = top2[..., -1] if top2.shape[-1] > 1 else torch.zeros_like(top_prob)
        margin = top_prob - second_prob
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        position_frac = (position_idx.to(hidden.dtype) / float(canvas_length))
        position_features = torch.cat(
            [
                hidden,
                entropy[..., None],
                top_prob[..., None],
                margin[..., None],
                position_frac[..., None],
            ],
            dim=-1,
        )
        clean_f = clean_mask[..., None].to(hidden.dtype)
        pooled_hidden = (hidden * clean_f).sum(dim=1) / clean_f.sum(dim=1).clamp_min(1.0)
        round_frac_t = torch.full(
            (hidden.shape[0], 1), round_frac, device=hidden.device, dtype=hidden.dtype
        )
        global_features = torch.cat([pooled_hidden, round_frac_t], dim=-1)
        return position_features, global_features

    @torch.no_grad()
    def rollout_for_grpo(
        self,
        inputs: torch.LongTensor,
        policy: RevelationPolicy,
        max_new_tokens: int,
        num_flow_steps: int = 16,
        policy_max_rounds: int = 40,
        do_sample: bool = True,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> RolloutTrace:
        """Runs one policy-driven rollout (spec Algorithm 3's rollout half): each
        round solves the flow over all active positions (shared `_solve_flow_round`
        with Stage 1's `generate`), then the policy chooses how many and which
        positions to commit instead of a fixed `commit_size`. Generator `theta` is
        frozen throughout (no_grad); only observation tensors (detached) and the
        sampled action/log-prob are recorded for the later GRPO update, which
        back-propagates through `policy` only.
        """
        assert inputs.shape[0] == 1, "Only batch size 1 is supported for rollouts."
        device = inputs.device
        input_length = inputs.shape[-1]
        canvas_length = input_length + max_new_tokens

        if attention_mask is None:
            attention_mask = torch.ones_like(inputs)
        attention_mask_valid = torch.cat(
            [
                attention_mask.bool(),
                torch.ones(1, max_new_tokens, dtype=torch.bool, device=device),
            ],
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

        steps: list[RolloutStep] = []
        total_nfe = 0
        round_idx = 0
        finished = False
        while committed_mask[:, input_length:].sum() < max_new_tokens and not finished:
            active_mask = (~committed_mask) & attention_mask_valid
            num_active = int(active_mask.sum().item())
            if num_active == 0:
                break
            clean_mask = committed_mask & attention_mask_valid

            p_theta, active_mask, hidden_state, nfe = self._solve_flow_round(
                sequence=sequence,
                committed_mask=committed_mask,
                attention_mask_valid=attention_mask_valid,
                canvas_length=canvas_length,
                position_idx=position_idx,
                num_flow_steps=num_flow_steps,
                request_hidden_states=True,
            )
            total_nfe += nfe

            position_features, global_features = self._build_observation(
                p_theta=p_theta,
                hidden_state=hidden_state,
                active_mask=active_mask,
                clean_mask=clean_mask,
                position_idx=position_idx,
                canvas_length=canvas_length,
                round_frac=round_idx / max(1, policy_max_rounds),
            )

            is_forced = round_idx >= policy_max_rounds - 1
            if is_forced:
                active_positions = active_mask[0].nonzero(as_tuple=True)[0]
                action = {"m": num_active, "indices": active_positions, "log_prob": None}
                m_max_eff = num_active
            else:
                m_max_eff = min(policy.m_max, num_active)
                action = policy.sample_action(
                    position_features=position_features,
                    global_features=global_features,
                    active_mask=active_mask,
                    m_max_eff=m_max_eff,
                )

            candidate_ids = self._sample_categorical(p_theta, do_sample=do_sample)
            commit_positions = action["indices"]
            sequence[0, commit_positions] = candidate_ids[0, commit_positions]
            committed_mask[0, commit_positions] = True
            steps.append(
                RolloutStep(
                    position_features=position_features,
                    global_features=global_features,
                    active_mask=active_mask,
                    m_max_eff=m_max_eff,
                    action=action,
                    is_forced=is_forced,
                )
            )
            round_idx += 1

            if self.eos_token_id is not None:
                committed_ids = sequence[0, commit_positions]
                eos_hits = (committed_ids == self.eos_token_id).nonzero(as_tuple=True)[0]
                if eos_hits.numel() > 0:
                    eos_pos = commit_positions[eos_hits[0]].item()
                    sequence = sequence[:, : eos_pos + 1]
                    finished = True

        return RolloutTrace(
            steps=steps, sequence=sequence, total_nfe=total_nfe, num_rounds=round_idx
        )

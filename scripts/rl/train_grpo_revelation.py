"""Stage 2 (spec Milestone M4): GRPO training for the adaptive token-revelation
policy. Not a Composer script — the rollout/update loop has fundamentally
different control flow from supervised denoising training (see
scripts/composer_scripts/train_discrete_denoiser.py for that).

The generator (theta) is loaded from a frozen Stage-1 AdaptiveBlockFlow
checkpoint and never updated here; only the policy (phi) is trained.
"""

import logging
import os
import random

import hydra
import torch
from omegaconf import DictConfig

from scripts.eval.model_loading import load_eval_model
from scripts.utils import (
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.rl.grpo import group_advantages, grpo_update
from src.rl.policy import RevelationPolicy
from src.rl.rewards import compute_reward, pairwise_diversity, sentence_quality

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/rl", config_name="grpo_wmt14")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    log.info("Loading frozen Stage-1 generator from %s", cfg.generator.pretrained_model_name_or_path)
    model = load_eval_model(
        pretrained_model_name_or_path=cfg.generator.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        device=device,
        load_ema_weights=cfg.generator.load_ema_weights,
        ckpt_file=cfg.generator.ckpt_file,
        force_legacy_if_no_generate=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = hydra.utils.instantiate(cfg.dataset, tokenizer=tokenizer)
    source_lang = dataset.source_lang
    target_lang = dataset.target_lang

    policy = RevelationPolicy(
        hidden_dim=cfg.policy.hidden_dim,
        mlp_hidden=cfg.policy.mlp_hidden,
        m_max=cfg.policy.m_max,
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.grpo.policy_lr)

    del source_lang  # only target_lang is needed for reference lookup below

    for step in range(cfg.grpo.num_steps):
        all_traces = []
        all_advantages = []
        step_qualities, step_costs, step_rewards = [], [], []

        for _ in range(cfg.grpo.sources_per_step):
            idx = random.randrange(len(dataset))
            example = dataset[idx]
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            reference = dataset.dataset[idx]["translation"][target_lang]

            traces, hypotheses, qualities, costs = [], [], [], []
            for _ in range(cfg.grpo.group_size):
                trace = model.rollout_for_grpo(
                    inputs=input_ids,
                    policy=policy,
                    max_new_tokens=cfg.generation.max_new_tokens,
                    num_flow_steps=cfg.generation.num_flow_steps,
                    policy_max_rounds=cfg.generation.policy_max_rounds,
                    do_sample=cfg.generation.do_sample,
                )
                generated_ids = trace.sequence[0, input_ids.shape[-1] :]
                hyp = tokenizer.decode(generated_ids, skip_special_tokens=True)
                quality = sentence_quality(hyp, reference)
                cost = trace.total_nfe / max(1, cfg.generation.num_flow_steps)
                traces.append(trace)
                hypotheses.append(hyp)
                qualities.append(quality)
                costs.append(cost)

            if cfg.grpo.lambda_diversity > 0:
                diversities = pairwise_diversity(hypotheses)
            else:
                diversities = [None] * len(hypotheses)
            rewards = [
                compute_reward(
                    quality=q,
                    cost=c,
                    diversity=d,
                    lambda_quality=cfg.grpo.lambda_quality,
                    lambda_cost=cfg.grpo.lambda_cost,
                    lambda_diversity=cfg.grpo.lambda_diversity,
                )
                for q, c, d in zip(qualities, costs, diversities)
            ]
            advantages = group_advantages(rewards)

            all_traces.extend(traces)
            all_advantages.extend(advantages)
            step_qualities.extend(qualities)
            step_costs.extend(costs)
            step_rewards.extend(rewards)

        stats = grpo_update(
            policy,
            optimizer,
            all_traces,
            all_advantages,
            clip_eps=cfg.grpo.clip_eps,
            grpo_epochs=cfg.grpo.grpo_epochs,
            max_grad_norm=cfg.grpo.max_grad_norm,
        )

        if step % cfg.logging.log_interval == 0:
            log.info(
                "step=%d loss=%s mean_reward=%.4f mean_chrf=%.4f mean_rounds=%.2f "
                "trajectories_updated=%d",
                step,
                stats["loss"],
                sum(step_rewards) / len(step_rewards),
                sum(step_qualities) / len(step_qualities),
                sum(step_costs) / len(step_costs),
                stats["num_trajectories_updated"],
            )

        if step % cfg.logging.save_interval == 0 or step == cfg.grpo.num_steps - 1:
            ckpt_path = os.path.join(cfg.output_dir, f"policy_step{step}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            log.info("Saved policy checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    register_useful_resolvers()
    main()

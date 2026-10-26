"""Group-relative policy optimization for the revelation policy (docs/mypapers/
main.tex, "Group-relative policy optimization", Eq. 227-245).

Rollout collection (`AdaptiveBlockFlow.rollout_for_grpo`) and the policy update
here are deliberately separate: rollouts run fully under `torch.no_grad()` (the
generator `theta` is always frozen, and `log pi_old` is just a stored number);
`grpo_update` re-scores the *stored* observations through the *current* policy
parameters with grad enabled to compute the PPO-style probability ratio. This
means `theta` can never receive a gradient from this module, by construction.
"""

from typing import Any, Dict, List

import torch

from src.denoiser.blockflow import RolloutTrace
from src.rl.policy import RevelationPolicy


def group_advantages(
    rewards: List[float], eps: float = 1e-4, std_floor: float = 1e-6
) -> List[float]:
    """Spec Eq. 229-230: group-normalized advantage; zero (not a divide-by-eps
    blowup) when every reward in the group is identical."""
    n = len(rewards)
    if n == 0:
        return []
    mean_r = sum(rewards) / n
    variance = sum((r - mean_r) ** 2 for r in rewards) / n
    std = variance**0.5
    if std < std_floor:
        return [0.0] * n
    return [(r - mean_r) / (std + eps) for r in rewards]


def grpo_update(
    policy: RevelationPolicy,
    optimizer: torch.optim.Optimizer,
    traces: List[RolloutTrace],
    advantages: List[float],
    clip_eps: float = 0.2,
    grpo_epochs: int = 1,
    max_grad_norm: float = 1.0,
) -> Dict[str, Any]:
    """Spec Eq. 239 clipped GRPO objective (no reference regularizer, Eq. 243's
    `R_ref` term is left off per spec: "leave it off initially unless implemented
    and verified"). Updates `policy.parameters()` only; `traces` were produced by
    a frozen generator, so there is no path for gradients to reach `theta`.
    """
    assert len(traces) == len(advantages), "traces and advantages must align 1:1."
    stats: Dict[str, Any] = {"loss": None, "num_trajectories_updated": 0, "epochs_run": 0}
    for _ in range(grpo_epochs):
        optimizer.zero_grad()
        total_loss = torch.zeros((), device=next(policy.parameters()).device)
        num_trajectories = 0
        for trace, advantage in zip(traces, advantages):
            stochastic_steps = [
                step
                for step in trace.steps
                if not step.is_forced and step.action.get("log_prob") is not None
            ]
            if not stochastic_steps:
                continue
            traj_loss = torch.zeros((), device=total_loss.device)
            for step in stochastic_steps:
                log_prob_old = step.action["log_prob"].detach()
                log_prob_new = policy.log_prob(
                    position_features=step.position_features,
                    global_features=step.global_features,
                    active_mask=step.active_mask,
                    m_max_eff=step.m_max_eff,
                    action=step.action,
                )
                ratio = torch.exp(log_prob_new - log_prob_old)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                traj_loss = traj_loss - torch.minimum(unclipped, clipped)
            total_loss = total_loss + traj_loss / len(stochastic_steps)
            num_trajectories += 1

        if num_trajectories == 0:
            break
        mean_loss = total_loss / num_trajectories
        mean_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        stats["loss"] = float(mean_loss.detach().item())
        stats["num_trajectories_updated"] = num_trajectories
        stats["epochs_run"] += 1
    return stats

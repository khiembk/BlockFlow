"""Correctness checks for the Stage 2 reward formula (docs/mypapers/main.tex,
Stage 2 "Rewards" section): R_g = Q_g + lambda_d*D_g - lambda_c*C_g, where
Q_g = chrF/100 has no separate weight, and for WMT14 lambda_d = 0.
"""

from src.rl.rewards import compute_reward


def test_reward_is_normalized_quality_minus_cost():
    # quality=100 (perfect chrF), cost=0 -> reward == Q_g == 1.0
    assert compute_reward(quality=100.0, cost=0.0, lambda_cost=0.02) == 1.0
    # quality=0, cost=0 -> reward == 0.0
    assert compute_reward(quality=0.0, cost=0.0, lambda_cost=0.02) == 0.0


def test_reward_subtracts_scaled_cost():
    reward = compute_reward(quality=50.0, cost=8.0, lambda_cost=0.02)
    assert abs(reward - (0.5 - 0.02 * 8.0)) < 1e-9


def test_reward_ignores_diversity_when_lambda_zero():
    r_with = compute_reward(quality=60.0, cost=4.0, diversity=0.9, lambda_cost=0.02, lambda_diversity=0.0)
    r_without = compute_reward(quality=60.0, cost=4.0, diversity=None, lambda_cost=0.02, lambda_diversity=0.0)
    assert r_with == r_without


def test_reward_includes_diversity_when_enabled():
    base = compute_reward(quality=60.0, cost=4.0, diversity=0.5, lambda_cost=0.02, lambda_diversity=0.0)
    boosted = compute_reward(quality=60.0, cost=4.0, diversity=0.5, lambda_cost=0.02, lambda_diversity=0.1)
    assert abs(boosted - base - 0.1 * 0.5) < 1e-9

"""Correctness checks for Adaptive BlockFlow Stage 2 GRPO (spec Section 8, items
9-10: policy log-probability validity and GRPO frozen-generator correctness).
"""

import copy
import itertools

import torch

from src.denoiser.blockflow import AdaptiveBlockFlow, AdaptiveBlockFlowConfig
from src.rl.grpo import group_advantages, grpo_update
from src.rl.policy import RevelationPolicy

VOCAB_SIZE = 20
SOURCE_LEN = 4
TARGET_LEN = 6
LENGTH = SOURCE_LEN + TARGET_LEN
COMMIT_SIZE = 2
HIDDEN_DIM = 8


def _make_model(seed: int = 0) -> AdaptiveBlockFlow:
    torch.manual_seed(seed)
    backbone_config = {
        "_target_": "src.backbone.dit_continuous.ContinuousDIT",
        "length": LENGTH,
        "causal_attention": False,
        "adaln": True,
        "hidden_size": HIDDEN_DIM,
        "cond_dim": HIDDEN_DIM,
        "n_heads": 2,
        "num_layers": 2,
        "dropout": 0.0,
        "tie_word_embeddings": True,
        "vocab_size": VOCAB_SIZE,
        "block_size": COMMIT_SIZE,
        "attn_backend": "sdpa",
        "pretrained_model_name_or_path": None,
    }
    tokenization_config = {
        "vocab_size": VOCAB_SIZE,
        "mask_token_id": None,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_vocab_size_multiple": None,
    }
    config = AdaptiveBlockFlowConfig(
        commit_size=COMMIT_SIZE,
        sampling_eps=1e-3,
        length=LENGTH,
        backbone_config=backbone_config,
        noise_config=None,
        tokenization_config=tokenization_config,
        attn_backend="sdpa",
    )
    model = AdaptiveBlockFlow(config, tokenizer=object())
    model.eval()
    return model


def _make_policy(seed: int = 0, m_max: int = 3) -> RevelationPolicy:
    torch.manual_seed(seed)
    return RevelationPolicy(hidden_dim=HIDDEN_DIM, mlp_hidden=8, m_max=m_max)


def test_policy_action_validity_and_log_prob_consistency():
    torch.manual_seed(0)
    policy = _make_policy()
    length = 5
    active_mask = torch.zeros(1, length, dtype=torch.bool)
    active_positions = torch.tensor([1, 2, 4])
    active_mask[0, active_positions] = True
    position_features = torch.randn(1, length, HIDDEN_DIM + 4)
    global_features = torch.randn(1, HIDDEN_DIM + 1)
    m_max_eff = min(policy.m_max, int(active_mask.sum().item()))

    action = policy.sample_action(
        position_features=position_features,
        global_features=global_features,
        active_mask=active_mask,
        m_max_eff=m_max_eff,
    )

    assert 1 <= action["m"] <= m_max_eff
    assert action["indices"].shape[0] == action["m"]
    indices_list = action["indices"].tolist()
    assert len(set(indices_list)) == len(indices_list), "action indices must be distinct"
    assert all(i in active_positions.tolist() for i in indices_list), (
        "action must only select active positions"
    )

    recomputed_log_prob = policy.log_prob(
        position_features=position_features,
        global_features=global_features,
        active_mask=active_mask,
        m_max_eff=m_max_eff,
        action=action,
    )
    assert torch.allclose(recomputed_log_prob, action["log_prob"], atol=1e-5)


def test_policy_log_prob_normalized_over_enumerable_actions():
    """Brute-force: sum of exp(log_prob) over every valid ordered action for a
    tiny active set must equal 1 (spec item 9: correct normalized probability)."""
    torch.manual_seed(1)
    policy = _make_policy(m_max=2)
    length = 3
    active_mask = torch.ones(1, length, dtype=torch.bool)
    position_features = torch.randn(1, length, HIDDEN_DIM + 4)
    global_features = torch.randn(1, HIDDEN_DIM + 1)
    m_max_eff = min(policy.m_max, length)

    total_prob = 0.0
    for m in range(1, m_max_eff + 1):
        for ordered_indices in itertools.permutations(range(length), m):
            action = {
                "m": m,
                "indices": torch.tensor(ordered_indices, dtype=torch.long),
            }
            with torch.no_grad():
                log_prob = policy.log_prob(
                    position_features=position_features,
                    global_features=global_features,
                    active_mask=active_mask,
                    m_max_eff=m_max_eff,
                    action=action,
                )
            total_prob += float(torch.exp(log_prob).item())
    assert abs(total_prob - 1.0) < 1e-4


def test_group_advantages_zero_variance():
    assert group_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    adv = group_advantages([1.0, 2.0, 3.0])
    assert abs(sum(adv)) < 1e-6  # mean-centered


def test_grpo_update_freezes_generator_and_handles_zero_advantage():
    model = _make_model()
    policy = _make_policy(seed=2)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    source = torch.randint(3, VOCAB_SIZE, (1, SOURCE_LEN))
    traces = [
        model.rollout_for_grpo(
            inputs=source,
            policy=policy,
            max_new_tokens=6,
            num_flow_steps=2,
            policy_max_rounds=10,
        )
        for _ in range(2)
    ]

    generator_state_before = copy.deepcopy(model.state_dict())

    # Zero-variance group: identical rewards -> zero advantage -> no-op update.
    policy_state_before = copy.deepcopy(policy.state_dict())
    zero_advantages = group_advantages([1.0, 1.0])
    grpo_update(policy, optimizer, traces, zero_advantages, grpo_epochs=1)
    for key, value in policy.state_dict().items():
        assert torch.equal(value, policy_state_before[key]), (
            f"zero-advantage update changed policy param {key}"
        )

    # Nonzero-advantage update: generator must remain exactly frozen.
    nonzero_advantages = group_advantages([1.0, 3.0])
    stats = grpo_update(policy, optimizer, traces, nonzero_advantages, grpo_epochs=1)
    assert stats["loss"] is not None
    for key, value in model.state_dict().items():
        assert torch.equal(value, generator_state_before[key]), (
            f"GRPO update modified frozen generator param {key}"
        )


def test_grpo_update_skips_all_forced_trace():
    model = _make_model()
    policy = _make_policy(seed=3)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    source = torch.randint(3, VOCAB_SIZE, (1, SOURCE_LEN))
    # policy_max_rounds=1 forces every round to be a deterministic, excluded step.
    trace = model.rollout_for_grpo(
        inputs=source,
        policy=policy,
        max_new_tokens=6,
        num_flow_steps=2,
        policy_max_rounds=1,
    )
    assert all(step.is_forced for step in trace.steps)

    policy_state_before = copy.deepcopy(policy.state_dict())
    stats = grpo_update(policy, optimizer, [trace], [1.0], grpo_epochs=1)
    assert stats["num_trajectories_updated"] == 0
    assert stats["loss"] is None
    for key, value in policy.state_dict().items():
        assert torch.equal(value, policy_state_before[key])

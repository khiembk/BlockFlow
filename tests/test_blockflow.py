"""Correctness checks for Adaptive BlockFlow Stage 1 (docs/mypapers/main.tex, Sec. 8).

Small, direct, CPU-only checks against tiny random tensors. These are not a
substitute for the spec's full mandatory test list -- they cover the subset
that's checkable without a trained checkpoint or a real dataset.
"""

import torch

from src.denoiser.blockflow import AdaptiveBlockFlow, AdaptiveBlockFlowConfig

VOCAB_SIZE = 20
SOURCE_LEN = 4
TARGET_LEN = 6
LENGTH = SOURCE_LEN + TARGET_LEN
COMMIT_SIZE = 2


def _make_model(seed: int = 0) -> AdaptiveBlockFlow:
    torch.manual_seed(seed)
    backbone_config = {
        "_target_": "src.backbone.dit_continuous.ContinuousDIT",
        "length": LENGTH,
        "causal_attention": False,
        "adaln": True,
        "hidden_size": 16,
        "cond_dim": 16,
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


def _toy_batch(batch_size: int = 2, pad: int = 0):
    length = LENGTH + pad
    input_ids = torch.randint(3, VOCAB_SIZE, (batch_size, length))
    attention_mask = torch.ones(batch_size, length, dtype=torch.long)
    if pad:
        attention_mask[:, -pad:] = 0
    context_mask = torch.zeros(batch_size, length, dtype=torch.long)
    context_mask[:, :SOURCE_LEN] = 1
    return input_ids, attention_mask, context_mask


def test_noise_path_endpoints():
    """z_1 == one_hot(y) exactly at t=1; z_0 is ~N(0, I) in distribution at t=0."""
    model = _make_model()
    input_ids, attention_mask, context_mask = _toy_batch(batch_size=4)
    batch_size = input_ids.shape[0]

    # t=1: interpolant collapses exactly to the one-hot target at active positions.
    denoiser_inputs = model._prepare_inputs(
        input_ids, attention_mask, context_mask, t=torch.ones(batch_size)
    )
    expected = torch.nn.functional.one_hot(input_ids, VOCAB_SIZE).float()
    active = denoiser_inputs.active_mask
    assert torch.allclose(
        denoiser_inputs.continuous_states[active], expected[active], atol=1e-5
    )

    # t=0: interpolant is exactly the Gaussian prior (no target signal at all).
    # Accumulate samples across repeated calls (fresh randn each time) to check
    # the empirical mean/std without relying on a single noisy draw.
    samples = []
    for _ in range(50):
        di = model._prepare_inputs(
            input_ids, attention_mask, context_mask, t=torch.zeros(batch_size)
        )
        samples.append(di.continuous_states[di.active_mask])
    samples = torch.cat(samples)
    assert samples.numel() > 1000
    assert abs(samples.mean().item()) < 0.15
    assert abs(samples.std().item() - 1.0) < 0.15


def test_all_remaining_objective_and_padding():
    """Loss is finite; tokens_mask is exactly the active set (zero on
    source/committed/padding)."""
    model = _make_model()
    input_ids, attention_mask, context_mask = _toy_batch(batch_size=3, pad=2)
    denoiser_inputs = model._prepare_inputs(input_ids, attention_mask, context_mask)
    backbone_output = model._backbone_forward(denoiser_inputs)
    model_output = model._forward(backbone_output.logits, denoiser_inputs)
    loss_and_nll = model._compute_loss(model_output, denoiser_inputs)

    assert torch.isfinite(loss_and_nll.loss)
    assert loss_and_nll.loss.item() != 0.0

    tokens_mask = denoiser_inputs.tokens_mask.bool()
    assert torch.equal(tokens_mask, denoiser_inputs.active_mask)
    # Padding (last 2 positions) must never be active/committed.
    assert not tokens_mask[:, -2:].any()
    # Source is never active.
    assert not tokens_mask[:, :SOURCE_LEN].any()
    # At least one active position exists per (unpadded) example.
    assert (loss_and_nll.nlls * tokens_mask).abs().sum() > 0


def test_no_reference_leakage_into_clean_context():
    """Perturbing target ids at active (not-yet-revealed) positions must not
    change backbone logits at clean (source/committed) positions."""
    model = _make_model()
    input_ids_a, attention_mask, context_mask = _toy_batch(batch_size=2)

    torch.manual_seed(123)
    denoiser_inputs_a = model._prepare_inputs(input_ids_a, attention_mask, context_mask)
    active = denoiser_inputs_a.active_mask

    input_ids_b = input_ids_a.clone()
    perturbation = torch.randint(3, VOCAB_SIZE, input_ids_b.shape)
    input_ids_b[active] = ((input_ids_b[active] + 1 + perturbation[active]) % (VOCAB_SIZE - 3)) + 3

    torch.manual_seed(123)
    denoiser_inputs_b = model._prepare_inputs(input_ids_b, attention_mask, context_mask)
    assert torch.equal(denoiser_inputs_a.active_mask, denoiser_inputs_b.active_mask)

    clean = ~active
    with torch.no_grad():
        out_a = model._backbone_forward(denoiser_inputs_a).logits
        out_b = model._backbone_forward(denoiser_inputs_b).logits
    assert torch.allclose(out_a[clean], out_b[clean], atol=1e-5)


def test_sampler_validity_endpoint_distribution():
    """p_theta sums to 1 and is nonnegative at every active position."""
    model = _make_model()
    input_ids, attention_mask, context_mask = _toy_batch(batch_size=2)
    denoiser_inputs = model._prepare_inputs(input_ids, attention_mask, context_mask)
    backbone_output = model._backbone_forward(denoiser_inputs)
    log_p = model._forward(backbone_output.logits, denoiser_inputs)
    p_theta = log_p.exp()
    active = denoiser_inputs.active_mask
    assert (p_theta[active] >= -1e-6).all()
    assert torch.allclose(
        p_theta[active].sum(dim=-1), torch.ones(int(active.sum())), atol=1e-3
    )


def test_commit_immutability_during_generation():
    """Once committed, a position's id is never overwritten by a later round."""
    model = _make_model()
    source = torch.randint(3, VOCAB_SIZE, (1, SOURCE_LEN))
    history: list = []
    sequence = model.generate(
        inputs=source,
        max_new_tokens=6,
        num_flow_steps=2,
        do_sample=False,
        history=history,
    )
    assert sequence.shape[0] == 1
    assert len(history) >= 1
    for i, (commit_positions_i, snapshot_i) in enumerate(history):
        for commit_positions_j, snapshot_j in history[i:]:
            assert torch.equal(
                snapshot_j[0, commit_positions_i], snapshot_i[0, commit_positions_i]
            )

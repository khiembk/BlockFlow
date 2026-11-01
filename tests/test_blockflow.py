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


def test_windowed_tokens_mask_restricts_scored_positions():
    """With `window_size` set, tokens_mask scores only the nearest
    `window_size` active positions (in canvas order), not the whole active
    set -- concentrating loss signal on the near-term positions that actually
    get committed soon."""
    WINDOW = 2
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
        window_size=WINDOW,
        length=LENGTH,
        backbone_config=backbone_config,
        noise_config=None,
        tokenization_config=tokenization_config,
        attn_backend="sdpa",
    )
    model = AdaptiveBlockFlow(config, tokenizer=object())
    model.eval()

    input_ids, attention_mask, context_mask = _toy_batch(batch_size=4)
    denoiser_inputs = model._prepare_inputs(input_ids, attention_mask, context_mask)
    tokens_mask = denoiser_inputs.tokens_mask.bool()
    active_mask = denoiser_inputs.active_mask

    # Windowed mask is always a subset of the full active set.
    assert torch.equal(tokens_mask & active_mask, tokens_mask)
    for i in range(input_ids.shape[0]):
        num_active = int(active_mask[i].sum().item())
        num_scored = int(tokens_mask[i].sum().item())
        assert num_scored == min(WINDOW, num_active)
        # The scored positions are exactly the first WINDOW active ones in
        # canvas order (lowest position indices among active positions).
        active_positions = active_mask[i].nonzero(as_tuple=True)[0]
        expected_scored = set(active_positions[:WINDOW].tolist())
        actual_scored = set(tokens_mask[i].nonzero(as_tuple=True)[0].tolist())
        assert actual_scored == expected_scored


def test_active_blocks_get_independent_flow_times():
    """Different blocks within the active span are noised at independently
    sampled flow times, not one shared scalar t for the whole active span."""
    model = _make_model()
    input_ids, attention_mask, context_mask = _toy_batch(batch_size=8)
    denoiser_inputs = model._prepare_inputs(input_ids, attention_mask, context_mask)
    t = denoiser_inputs.t
    active = denoiser_inputs.active_mask
    assert t.dim() == 2  # per-position, not one scalar per example

    found_example_with_multiple_blocks = False
    for i in range(input_ids.shape[0]):
        active_positions = active[i].nonzero(as_tuple=True)[0]
        if active_positions.numel() < 2 * COMMIT_SIZE:
            continue  # this example's sampled round only has one active block
        found_example_with_multiple_blocks = True
        block0 = active_positions[:COMMIT_SIZE]
        block1 = active_positions[COMMIT_SIZE : 2 * COMMIT_SIZE]
        # Within a block, t is identical for every position (block-level, not
        # per-token, granularity); across blocks, it generally differs.
        assert torch.allclose(t[i, block0], t[i, block0[0]].expand_as(t[i, block0]))
        assert torch.allclose(t[i, block1], t[i, block1[0]].expand_as(t[i, block1]))
        assert not torch.allclose(t[i, block0[0]], t[i, block1[0]])
    assert found_example_with_multiple_blocks, (
        "No example in this batch sampled a round with >=2 active blocks; "
        "widen TARGET_LEN/batch_size or reseed."
    )


def test_no_cross_block_leakage_between_active_blocks():
    """Active blocks are isolated from each other: perturbing one active
    block's input ids must not change backbone logits at a DIFFERENT active
    block's positions (each block only attends to the clean prefix and its
    own block, exactly like `generate`'s per-round, one-block-at-a-time
    behavior -- a block being denoised never gets to peek at a sibling
    block's different, unrelated noise level)."""
    model = _make_model()
    batch_size = 2
    # Force two full active blocks and nothing committed (round k=0): clean
    # prefix is just the source; the whole TARGET_LEN-length target is active.
    assert TARGET_LEN >= 2 * COMMIT_SIZE
    attention_mask = torch.ones(batch_size, LENGTH, dtype=torch.bool)
    context_mask = torch.zeros(batch_size, LENGTH, dtype=torch.bool)
    context_mask[:, :SOURCE_LEN] = True
    active_mask = torch.zeros(batch_size, LENGTH, dtype=torch.bool)
    active_mask[:, SOURCE_LEN : SOURCE_LEN + 2 * COMMIT_SIZE] = True
    clean_mask = context_mask.clone()
    position_idx = torch.arange(LENGTH)[None, :].expand(batch_size, -1)
    block_idx = model._active_block_idx(active_mask, COMMIT_SIZE)
    block0 = slice(SOURCE_LEN, SOURCE_LEN + COMMIT_SIZE)
    block1 = slice(SOURCE_LEN + COMMIT_SIZE, SOURCE_LEN + 2 * COMMIT_SIZE)

    def run(ids: torch.LongTensor, seed: int):
        torch.manual_seed(seed)
        t_per_block = torch.rand(batch_size, int(block_idx.max().item()) + 1) * 0.998 + 0.001
        t = torch.gather(t_per_block, dim=1, index=block_idx)
        t_cond = torch.where(active_mask, t, torch.ones_like(t))
        x0_noise = torch.randn(batch_size, LENGTH, VOCAB_SIZE)
        one_hot_target = torch.nn.functional.one_hot(ids, VOCAB_SIZE).float()
        continuous_states = (1 - t[..., None]) * x0_noise + t[..., None] * one_hot_target
        bool_mask = model._build_attention_mask(
            clean_mask=clean_mask,
            active_mask=active_mask,
            position_idx=position_idx,
            block_idx=block_idx,
        )
        bool_mask = bool_mask & attention_mask[:, None, :]
        float_mask = model._preprocess_attention_mask(
            bool_mask[:, None, :, :], dtype=torch.float32
        )
        from src.denoiser.base import DenoiserInput

        denoiser_inputs = DenoiserInput(
            xt=ids,
            x0=ids,
            attention_mask=float_mask,
            active_mask=active_mask,
            continuous_states=continuous_states,
            backbone_kwargs={"position_ids": position_idx, "t": t_cond},
        )
        with torch.no_grad():
            # Compare pre-output-layer hidden states, not logits: the output
            # head is zero-initialized (standard AdaLN-Zero DiT init), so
            # logits are trivially all-zero on a fresh, untrained model.
            return model._backbone_forward(
                denoiser_inputs, output_hidden_states=True
            ).hidden_states[0]

    ids_a = torch.randint(3, VOCAB_SIZE, (batch_size, LENGTH))
    ids_b = ids_a.clone()
    perturbation = torch.randint(1, VOCAB_SIZE - 3, ids_b[:, block1].shape)
    ids_b[:, block1] = ((ids_b[:, block1] - 3 + perturbation) % (VOCAB_SIZE - 3)) + 3

    out_a = run(ids_a, seed=7)
    out_b = run(ids_b, seed=7)
    # Block 1 was perturbed; block 0 (a different active block) and the clean
    # prefix must be completely unaffected.
    assert torch.allclose(out_a[:, block0], out_b[:, block0], atol=1e-5)
    assert torch.allclose(out_a[:, :SOURCE_LEN], out_b[:, :SOURCE_LEN], atol=1e-5)
    # Sanity: block 1's own hidden state DOES change (otherwise the
    # perturbation/test setup itself would be a no-op and the test vacuous).
    assert not torch.allclose(out_a[:, block1], out_b[:, block1], atol=1e-5)


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

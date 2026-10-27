"""Correctness checks for Adaptive BlockFlow Stage 1 v3 (embedding-space VFM
with a frozen T5 encoder + separately trained decoder branch, see
docs/mypapers/main.tex and the Stage 1 v3 plan). Same tiny-tensor CPU style as
`tests/test_blockflow.py`; uses a real (but small) pretrained t5-small as the
frozen context encoder since that's the whole point of this architecture.
"""

import torch

from src.denoiser.blockflow_embed import AdaptiveBlockFlowEmbed, AdaptiveBlockFlowEmbedConfig

VOCAB_SIZE = 20
SOURCE_LEN = 4
TARGET_LEN = 6
LENGTH = SOURCE_LEN + TARGET_LEN
COMMIT_SIZE = 2
T5_EMBED_DIM = 512


def _make_model(seed: int = 0, decoder_prob: float = 0.0) -> AdaptiveBlockFlowEmbed:
    torch.manual_seed(seed)
    backbone_config = {
        "_target_": "src.backbone.dit_continuous_embed.EmbeddingSpaceContinuousDIT",
        "length": LENGTH,
        "causal_attention": False,
        "adaln": True,
        "hidden_size": 16,
        "cond_dim": 16,
        "n_heads": 2,
        "num_layers": 2,
        "dropout": 0.0,
        "tie_word_embeddings": False,
        "vocab_size": VOCAB_SIZE,
        "block_size": COMMIT_SIZE,
        "attn_backend": "sdpa",
        "pretrained_model_name_or_path": None,
        "t5_embed_dim": T5_EMBED_DIM,
        "bottleneck_dim": 8,
    }
    tokenization_config = {
        "vocab_size": VOCAB_SIZE,
        "mask_token_id": None,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_vocab_size_multiple": None,
    }
    config = AdaptiveBlockFlowEmbedConfig(
        commit_size=COMMIT_SIZE,
        sampling_eps=1e-3,
        length=LENGTH,
        backbone_config=backbone_config,
        noise_config=None,
        tokenization_config=tokenization_config,
        attn_backend="sdpa",
        t5_model_name="t5-small",
        t5_embed_dim=T5_EMBED_DIM,
        latent_mean=0.0,
        latent_std=0.15,
        noise_scale=2.0,
        decoder_prob=decoder_prob,
        decoder_p_mean=-1.5,
        decoder_p_std=0.8,
        decoder_noise_scale=2.0,
    )
    model = AdaptiveBlockFlowEmbed(config, tokenizer=object())
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
    """At t=1 (flow branch), continuous_states == true T5 target embedding
    exactly at active positions; at t=0, continuous_states == N(0,I)*noise_scale."""
    model = _make_model(decoder_prob=0.0)
    input_ids, attention_mask, context_mask = _toy_batch(batch_size=4)
    batch_size = input_ids.shape[0]

    denoiser_inputs = model._prepare_inputs(
        input_ids, attention_mask, context_mask, t=torch.ones(batch_size)
    )
    active = denoiser_inputs.active_mask
    context_latent = denoiser_inputs.backbone_kwargs["context_latent"]
    assert torch.allclose(
        denoiser_inputs.continuous_states[active], context_latent[active], atol=1e-5
    )

    samples = []
    for _ in range(50):
        di = model._prepare_inputs(
            input_ids, attention_mask, context_mask, t=torch.zeros(batch_size)
        )
        samples.append(di.continuous_states[di.active_mask])
    samples = torch.cat(samples)
    assert samples.numel() > 1000
    assert abs(samples.mean().item()) < 0.3
    assert abs(samples.std().item() - model.noise_scale) < 0.3


def test_all_remaining_objective_mse_branch():
    """Flow (MSE) branch: loss finite/nonzero on active positions, tokens_mask
    exactly the active set (zero on source/committed/padding)."""
    model = _make_model(decoder_prob=0.0)
    input_ids, attention_mask, context_mask = _toy_batch(batch_size=3, pad=2)
    denoiser_inputs = model._prepare_inputs(input_ids, attention_mask, context_mask)
    assert denoiser_inputs.backbone_kwargs["decoder_mode"] is False
    backbone_output = model._backbone_forward(denoiser_inputs)
    model_output = model._forward(backbone_output.logits, denoiser_inputs)
    loss_and_nll = model._compute_loss(model_output, denoiser_inputs)

    assert torch.isfinite(loss_and_nll.loss)
    assert loss_and_nll.loss.item() != 0.0

    tokens_mask = denoiser_inputs.tokens_mask.bool()
    assert torch.equal(tokens_mask, denoiser_inputs.active_mask)
    assert not tokens_mask[:, -2:].any()
    assert not tokens_mask[:, :SOURCE_LEN].any()
    assert (loss_and_nll.nlls * tokens_mask).abs().sum() > 0


def test_all_remaining_objective_decoder_branch():
    """Decoder (CE) branch: loss finite/nonzero on active positions only."""
    model = _make_model(decoder_prob=1.0)
    input_ids, attention_mask, context_mask = _toy_batch(batch_size=3, pad=2)
    denoiser_inputs = model._prepare_inputs(input_ids, attention_mask, context_mask)
    assert denoiser_inputs.backbone_kwargs["decoder_mode"] is True
    backbone_output = model._backbone_forward(denoiser_inputs)
    model_output = model._forward(backbone_output.logits, denoiser_inputs)
    loss_and_nll = model._compute_loss(model_output, denoiser_inputs)

    assert torch.isfinite(loss_and_nll.loss)
    assert loss_and_nll.loss.item() != 0.0
    tokens_mask = denoiser_inputs.tokens_mask.bool()
    assert not tokens_mask[:, -2:].any()
    assert not tokens_mask[:, :SOURCE_LEN].any()


def test_no_reference_leakage_into_clean_context():
    """Perturbing target ids at active positions must not change the flow
    branch's regression prediction nor the decoder branch's logits at clean
    (source/committed) positions."""
    for decoder_prob in (0.0, 1.0):
        model = _make_model(decoder_prob=decoder_prob)
        input_ids_a, attention_mask, context_mask = _toy_batch(batch_size=2)

        torch.manual_seed(123)
        denoiser_inputs_a = model._prepare_inputs(input_ids_a, attention_mask, context_mask)
        active = denoiser_inputs_a.active_mask

        input_ids_b = input_ids_a.clone()
        perturbation = torch.randint(3, VOCAB_SIZE, input_ids_b.shape)
        input_ids_b[active] = (
            (input_ids_b[active] + 1 + perturbation[active]) % (VOCAB_SIZE - 3)
        ) + 3

        torch.manual_seed(123)
        denoiser_inputs_b = model._prepare_inputs(input_ids_b, attention_mask, context_mask)
        assert torch.equal(denoiser_inputs_a.active_mask, denoiser_inputs_b.active_mask)

        clean = ~active
        with torch.no_grad():
            out_a = model._backbone_forward(denoiser_inputs_a).logits
            out_b = model._backbone_forward(denoiser_inputs_b).logits
        assert torch.allclose(out_a[clean], out_b[clean], atol=1e-4)


def test_sampler_validity_decoder_branch():
    """Decoder branch's softmax output sums to 1 and is nonnegative at active
    positions."""
    model = _make_model(decoder_prob=1.0)
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


def test_frozen_t5_encoder_unchanged_after_training_step():
    """The frozen T5 encoder's parameters are bit-identical after a training
    step; the adapter/backbone body do receive gradients and update."""
    model = _make_model(decoder_prob=0.0)
    model.train()
    frozen_before = [p.detach().clone() for p in model.t5_encoder.encoder.parameters()]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    input_ids, attention_mask, context_mask = _toy_batch(batch_size=2)

    optimizer.zero_grad()
    out = model(input_ids=input_ids, attention_mask=attention_mask, context_mask=context_mask)
    out.loss.backward()
    optimizer.step()

    frozen_after = list(model.t5_encoder.encoder.parameters())
    for before, after in zip(frozen_before, frozen_after):
        assert torch.equal(before, after)
        assert after.grad is None or torch.all(after.grad == 0)

    assert model.backbone.adapter.up_proj.weight.grad is not None
    assert model.backbone.blocks[0].attn_qkv.weight.grad is not None


def test_commit_immutability_during_generation():
    """Once committed, a position's id is never overwritten by a later round."""
    model = _make_model(decoder_prob=0.2)
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

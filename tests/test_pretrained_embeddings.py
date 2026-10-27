"""Correctness checks for initializing/freezing ContinuousDIT's vocab-tied
matrices from a pretrained model's token embeddings (t5-small)."""

import copy

import pytest
import torch
from transformers import T5EncoderModel

from src.backbone.dit_continuous import ContinuousDIT
from src.denoiser.blockflow import AdaptiveBlockFlow, AdaptiveBlockFlowConfig

T5_HIDDEN_DIM = 512
VOCAB_SIZE = 500  # small slice of t5-small's vocab, enough to exercise the path


def _t5_shared_embedding():
    return T5EncoderModel.from_pretrained("t5-small").shared.weight.detach()


@pytest.mark.parametrize("freeze", [True, False])
def test_pretrained_embeddings_copied_and_frozen(freeze):
    backbone = ContinuousDIT(
        length=16,
        causal_attention=False,
        adaln=True,
        hidden_size=T5_HIDDEN_DIM,
        cond_dim=32,
        n_heads=8,
        num_layers=2,
        dropout=0.0,
        tie_word_embeddings=True,
        vocab_size=VOCAB_SIZE,
        block_size=1,
        attn_backend="sdpa",
        pretrained_model_name_or_path=None,
        pretrained_embedding_model="t5-small",
        freeze_pretrained_embeddings=freeze,
    )
    t5_embedding = _t5_shared_embedding()[:VOCAB_SIZE].float()
    unit_embedding = t5_embedding / t5_embedding.norm(dim=-1, keepdim=True)

    # T5's raw embedding table has a much larger absolute scale (row norms
    # ~500+) than this architecture's own conventions, and that scale is only
    # well-behaved inside T5's own co-designed stack. So we transfer only the
    # *directions* (unit-normalized rows -- the actual learned semantic
    # geometry) and apply our own architecture-appropriate target scale, not
    # T5's raw values or even T5's own tied-embedding output correction alone
    # (both were tried and still produced exploding/unstable initial loss).
    assert not torch.allclose(backbone.vocab_embed.embedding, t5_embedding, atol=1e-3)
    assert torch.allclose(backbone.vocab_embed.embedding, unit_embedding, atol=1e-5)
    assert torch.allclose(
        backbone.vocab_embed.embedding.norm(dim=-1), torch.ones(VOCAB_SIZE), atol=1e-4
    )

    expected_output_weight = unit_embedding * (T5_HIDDEN_DIM**-0.5)
    assert torch.allclose(
        backbone.output_layer.linear.weight, expected_output_weight, atol=1e-5
    )
    assert torch.all(backbone.output_layer.linear.bias == 0.0)

    assert backbone.vocab_embed.embedding.requires_grad == (not freeze)
    assert backbone.output_layer.linear.weight.requires_grad == (not freeze)
    assert backbone.output_layer.linear.bias.requires_grad == (not freeze)


def test_hidden_size_mismatch_raises():
    with pytest.raises(ValueError, match="hidden_size"):
        ContinuousDIT(
            length=16,
            causal_attention=False,
            adaln=True,
            hidden_size=256,  # != t5-small's 512
            cond_dim=32,
            n_heads=8,
            num_layers=2,
            dropout=0.0,
            tie_word_embeddings=True,
            vocab_size=VOCAB_SIZE,
            block_size=1,
            attn_backend="sdpa",
            pretrained_model_name_or_path=None,
            pretrained_embedding_model="t5-small",
        )


def test_frozen_embeddings_unchanged_after_training_step():
    backbone_config = {
        "_target_": "src.backbone.dit_continuous.ContinuousDIT",
        "length": 10,
        "causal_attention": False,
        "adaln": True,
        "hidden_size": T5_HIDDEN_DIM,
        "cond_dim": 32,
        "n_heads": 8,
        "num_layers": 2,
        "dropout": 0.0,
        "tie_word_embeddings": True,
        "vocab_size": VOCAB_SIZE,
        "block_size": 2,
        "attn_backend": "sdpa",
        "pretrained_model_name_or_path": None,
        "pretrained_embedding_model": "t5-small",
        "freeze_pretrained_embeddings": True,
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
        commit_size=2,
        sampling_eps=1e-3,
        length=10,
        backbone_config=backbone_config,
        noise_config=None,
        tokenization_config=tokenization_config,
        attn_backend="sdpa",
    )
    model = AdaptiveBlockFlow(config, tokenizer=object())
    model.train()

    frozen_vocab_embed_before = model.backbone.vocab_embed.embedding.detach().clone()
    frozen_output_weight_before = model.backbone.output_layer.linear.weight.detach().clone()

    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad), lr=0.1
    )
    input_ids = torch.randint(3, VOCAB_SIZE, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    context_mask = torch.zeros(2, 10, dtype=torch.long)
    context_mask[:, :4] = 1

    optimizer.zero_grad()
    out = model(input_ids=input_ids, attention_mask=attention_mask, context_mask=context_mask)
    out.loss.backward()
    optimizer.step()

    assert torch.equal(model.backbone.vocab_embed.embedding, frozen_vocab_embed_before)
    assert torch.equal(model.backbone.output_layer.linear.weight, frozen_output_weight_before)
    # Sanity: something else (a transformer block weight) DID get a gradient/update.
    assert model.backbone.blocks[0].attn_qkv.weight.grad is not None


def test_initial_loss_is_sane_with_pretrained_embeddings():
    """Regression test for the T5 tied-embedding output-scale bug: omitting the
    model_dim**-0.5 compensating scale on output_layer made logits ~sqrt(hidden_dim)x
    too large, saturating the softmax and producing a loss in the thousands
    (observed in a live training run) instead of near log(vocab_size)."""
    backbone_config = {
        "_target_": "src.backbone.dit_continuous.ContinuousDIT",
        "length": 10,
        "causal_attention": False,
        "adaln": True,
        "hidden_size": T5_HIDDEN_DIM,
        "cond_dim": 32,
        "n_heads": 8,
        "num_layers": 2,
        "dropout": 0.0,
        "tie_word_embeddings": True,
        "vocab_size": VOCAB_SIZE,
        "block_size": 2,
        "attn_backend": "sdpa",
        "pretrained_model_name_or_path": None,
        "pretrained_embedding_model": "t5-small",
        "freeze_pretrained_embeddings": True,
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
        commit_size=2,
        sampling_eps=1e-3,
        length=10,
        backbone_config=backbone_config,
        noise_config=None,
        tokenization_config=tokenization_config,
        attn_backend="sdpa",
    )
    model = AdaptiveBlockFlow(config, tokenizer=object())
    model.eval()

    input_ids = torch.randint(3, VOCAB_SIZE, (4, 10))
    attention_mask = torch.ones(4, 10, dtype=torch.long)
    context_mask = torch.zeros(4, 10, dtype=torch.long)
    context_mask[:, :4] = 1

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, context_mask=context_mask)

    import math

    assert torch.isfinite(out.loss)
    # A random-init transformer body with correctly-scaled frozen embeddings
    # should land in the same ballpark as log(vocab_size); a 1500+ loss (as seen
    # with the unscaled bug) is off by ~2-3 orders of magnitude, not noise.
    assert out.loss.item() < 5 * math.log(VOCAB_SIZE)

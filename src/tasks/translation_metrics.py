"""Corpus-level BLEU for translation evaluation (WMT14 etc.).

Mirrors the convention used in the sibling CorrFlow project
(`src/utils/metrics_utils.py::compute_bleu`): plain `sacrebleu.corpus_bleu`
with `lowercase=True` and `use_effective_order=True` (the latter matters here
since AdaptiveBlockFlow can produce short/degenerate hypotheses early in
training). Single-reference only (WMT14 has one reference per source), scored
against raw reference text -- never retokenized/decoded through the model's
own tokenizer, so scoring is insensitive to that tokenizer's normalization.
"""

from typing import List

import sacrebleu


def compute_corpus_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Corpus BLEU (0-100) over a full evaluation set."""
    assert len(hypotheses) == len(references), (
        f"Got {len(hypotheses)} hypotheses but {len(references)} references."
    )
    return sacrebleu.corpus_bleu(
        hypotheses, [references], lowercase=True, use_effective_order=True
    ).score


def compute_corpus_chrf(hypotheses: List[str], references: List[str]) -> float:
    """Corpus chrF (0-100); the spec's recommended primary metric (less brittle
    than sentence BLEU at the small/early-training translation lengths this
    model can produce)."""
    assert len(hypotheses) == len(references), (
        f"Got {len(hypotheses)} hypotheses but {len(references)} references."
    )
    return sacrebleu.corpus_chrf(hypotheses, [references]).score

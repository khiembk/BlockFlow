"""GRPO reward computation (docs/mypapers/main.tex, Stage 2: "Rewards: quality,
diversity, computational cost"). Quality uses sentence chrF directly via
`sacrebleu` (not the `evaluate` library wrapper that `scripts/eval/seq2seq_eval.py`
uses for one-shot corpus BLEU) because the reward is computed once per rollout —
potentially thousands of times per training run — and `sacrebleu` is already an
installed, offline-capable dependency with no per-call download overhead.
"""

from typing import List, Optional

from sacrebleu.metrics import CHRF

_CHRF = CHRF()


def sentence_quality(hypothesis: str, reference: str) -> float:
    """Sentence-level chrF in [0, 100] (spec: "a reasonable first training signal
    because it is less brittle than sentence-level BLEU")."""
    if not hypothesis.strip():
        return 0.0
    return float(_CHRF.sentence_score(hypothesis, [reference]).score)


def pairwise_diversity(hypotheses: List[str]) -> List[float]:
    """Spec Eq.: D_g = (G-1)^-1 * sum_{h!=g} [1 - chrF(y_g, y_h)/100]. Only used
    when `lambda_diversity > 0` (off by default, per spec's recommended ordering
    of ablations: quality+cost first, diversity afterward)."""
    group_size = len(hypotheses)
    if group_size <= 1:
        return [0.0] * group_size
    diversities = []
    for g, hyp_g in enumerate(hypotheses):
        if not hyp_g.strip():
            diversities.append(0.0)
            continue
        scores = []
        for h, hyp_h in enumerate(hypotheses):
            if h == g:
                continue
            if not hyp_h.strip():
                continue
            scores.append(1.0 - _CHRF.sentence_score(hyp_g, [hyp_h]).score / 100.0)
        diversities.append(sum(scores) / len(scores) if scores else 0.0)
    return diversities


def compute_reward(
    quality: float,
    cost: float,
    diversity: Optional[float] = None,
    lambda_cost: float = 0.02,
    lambda_diversity: float = 0.0,
) -> float:
    """Spec Eq.: R_g = Q_g + lambda_d * D_g - lambda_c * C_g, where
    Q_g = chrF(hyp, ref) / 100 (quality has no separate weight -- for WMT14
    the spec sets lambda_d = 0 and considers only quality and cost).

    `quality` is the raw sentence_quality() output (chrF in [0, 100]); the
    /100 normalization to match Q_g happens here, not at the call site, so
    callers can keep logging/reporting the raw chrF score.
    """
    reward = (quality / 100.0) - lambda_cost * cost
    if lambda_diversity > 0.0 and diversity is not None:
        reward += lambda_diversity * diversity
    return reward

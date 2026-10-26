from src.tasks.translation_metrics import compute_corpus_bleu, compute_corpus_chrf


def test_bleu_identical_hypotheses_scores_near_100():
    refs = ["the quick brown fox jumps over the lazy dog", "a second test sentence here"]
    hyps = list(refs)
    assert compute_corpus_bleu(hyps, refs) > 99.0
    assert compute_corpus_chrf(hyps, refs) > 99.0


def test_bleu_unrelated_hypotheses_scores_low():
    refs = ["the quick brown fox jumps over the lazy dog", "a second test sentence here"]
    hyps = ["completely unrelated text", "nothing like the reference at all"]
    assert compute_corpus_bleu(hyps, refs) < 20.0
    assert compute_corpus_chrf(hyps, refs) < 40.0


def test_bleu_length_mismatch_raises():
    try:
        compute_corpus_bleu(["a"], ["a", "b"])
    except AssertionError:
        return
    raise AssertionError("expected AssertionError on length mismatch")

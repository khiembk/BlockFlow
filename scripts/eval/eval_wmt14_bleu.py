"""WMT14 corpus BLEU/chrF evaluation for AdaptiveBlockFlow.

A lightweight, single-process evaluation driver (mirrors the sibling CorrFlow
project's rank-0-only, non-distributed BLEU evaluation convention --
`sacrebleu.corpus_bleu(hyps, [refs], lowercase=True, use_effective_order=True)`
-- rather than `scripts/eval/seq2seq_eval.py`'s full multi-rank DDP harness,
which isn't adapted to AdaptiveBlockFlow's batch-size-1 `generate` contract).

References are the dataset's raw target text (never retokenized/decoded
through the model's tokenizer), matching CorrFlow's convention.
"""

import json
import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from scripts.eval.model_loading import load_eval_model
from scripts.utils import maybe_add_missing_special_tokens, register_useful_resolvers, set_seed
from src.tasks.translation_metrics import compute_corpus_bleu, compute_corpus_chrf

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="wmt14_bleu")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    log.info("Loading model from %s", cfg.model.pretrained_model_name_or_path)
    model = load_eval_model(
        pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        device=device,
        load_ema_weights=cfg.model.load_ema_weights,
        ckpt_file=cfg.model.ckpt_file,
        force_legacy_if_no_generate=True,
    )
    model.eval()

    dataset = hydra.utils.instantiate(cfg.dataset, tokenizer=tokenizer)
    target_lang = dataset.target_lang
    num_samples = cfg.num_samples if cfg.num_samples is not None else len(dataset)
    num_samples = min(num_samples, len(dataset))
    log.info("Evaluating %d / %d examples", num_samples, len(dataset))

    hypotheses, references = [], []
    records = []
    with torch.no_grad():
        for idx in tqdm(range(num_samples)):
            example = dataset[idx]
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            reference = dataset.dataset[idx]["translation"][target_lang]

            outputs = model.generate(
                inputs=input_ids,
                max_new_tokens=cfg.generation.max_new_tokens,
                num_flow_steps=cfg.generation.num_flow_steps,
                do_sample=cfg.generation.do_sample,
                return_dict_in_generate=True,
            )
            generated_ids = outputs.sequences[0, input_ids.shape[-1] :]
            hypothesis = tokenizer.decode(generated_ids, skip_special_tokens=True)

            hypotheses.append(hypothesis)
            references.append(reference)
            records.append(
                {
                    "index": idx,
                    "hypothesis": hypothesis,
                    "reference": reference,
                    "nfe": outputs.nfe,
                    "num_rounds": outputs.num_rounds,
                }
            )

    bleu = compute_corpus_bleu(hypotheses, references)
    chrf = compute_corpus_chrf(hypotheses, references)
    mean_nfe = sum(r["nfe"] for r in records) / len(records)
    mean_rounds = sum(r["num_rounds"] for r in records) / len(records)

    log.info(
        "BLEU: %.2f  chrF: %.2f  mean_nfe: %.2f  mean_rounds: %.2f  (n=%d)",
        bleu,
        chrf,
        mean_nfe,
        mean_rounds,
        len(records),
    )

    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "bleu": bleu,
                "chrf": chrf,
                "mean_nfe": mean_nfe,
                "mean_rounds": mean_rounds,
                "num_samples": len(records),
            },
            f,
            indent=2,
        )
    with open(os.path.join(cfg.output_dir, "generations.jsonl"), "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("Wrote results to %s", cfg.output_dir)


if __name__ == "__main__":
    register_useful_resolvers()
    main()

"""Build the GSM8K distillation datasets used by the training configs.

The GSM8K distillation configs expect Hugging Face datasets saved at:

    outputs/distillation/Qwen3-32B-AWQ/gsm8k_train
    outputs/distillation/Qwen3-32B-AWQ/gsm8k_eval

Each row contains tokenized prompt plus teacher answer in the student tokenizer's
vocabulary. Context masks are injected later by
configs/dataset/gsm8k_train_distill.yaml.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from datasets import Dataset, load_dataset
from scripts.utils import maybe_add_missing_special_tokens

SOURCE_PROMPT_TEXT = (
    "Please reason step by step, and put your final answer within $\\boxed{}$. "
)
TARGET_PROMPT_TEXT = "Answer: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and save GSM8K distillation datasets."
    )
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--student-tokenizer", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output-dir", default="outputs/distillation/Qwen3-32B-AWQ")
    parser.add_argument("--dataset-path", default="openai/gsm8k")
    parser.add_argument("--config-name", default="main")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--use-gold-answers",
        action="store_true",
        help=(
            "Build a structurally compatible dataset from GSM8K labels instead "
            "of teacher generations. This is useful for smoke tests, but is not "
            "the paper distillation dataset."
        ),
    )
    return parser.parse_args()


def normalize_gsm8k_answer(answer: str) -> str:
    return re.sub(
        r"^####\s*(\d+)\s*$",
        r"$\\boxed{\1}$",
        answer,
        flags=re.MULTILINE,
    )


def render_chat_prompt(tokenizer: PreTrainedTokenizerBase, question: str) -> str:
    messages = [{"role": "user", "content": SOURCE_PROMPT_TEXT + question}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages, **kwargs
        )


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_torch_dtype(dtype_name: str):
    normalized = dtype_name.lower()
    if normalized == "auto":
        return "auto"
    dtype = getattr(torch, normalized, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype


def load_teacher_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        revision=args.revision,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        revision=args.revision,
        trust_remote_code=True,
        device_map=args.device_map,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )
    model.eval()
    return tokenizer, model


def model_input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def generate_teacher_responses(
    questions: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> list[str]:
    responses: list[str] = []
    batches = list(batched(questions, args.batch_size))
    for question_batch in tqdm(batches, desc="Generating teacher answers"):
        prompts = [
            render_chat_prompt(tokenizer, question) for question in question_batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
            add_special_tokens=False,
        ).to(model_input_device(model))
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.do_sample:
            generation_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )
        output_ids = model.generate(**inputs, **generation_kwargs)
        prompt_width = inputs["input_ids"].shape[-1]
        responses.extend(
            tokenizer.batch_decode(
                output_ids[:, prompt_width:],
                skip_special_tokens=False,
            )
        )
    return responses


def tokenize_rows(
    questions: list[str],
    responses: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> list[dict[str, list[int] | int]]:
    rows: list[dict[str, list[int] | int]] = []
    eos_token = tokenizer.eos_token or ""
    for idx, (question, response) in enumerate(zip(questions, responses)):
        prompt = render_chat_prompt(tokenizer, question)
        text = prompt + response.strip()
        if eos_token and not text.endswith(eos_token):
            text += eos_token
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
        rows.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "index": idx,
            }
        )
    return rows


def build_split(
    split_name: str,
    output_name: str,
    limit: int | None,
    responses: list[str] | None,
    student_tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> None:
    dataset = load_dataset(
        args.dataset_path,
        args.config_name,
        split=split_name,
        trust_remote_code=True,
    )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    questions = list(dataset["question"])
    if responses is None:
        responses = [
            TARGET_PROMPT_TEXT + normalize_gsm8k_answer(answer)
            for answer in dataset["answer"]
        ]
    rows = tokenize_rows(questions, responses, student_tokenizer, args.max_length)
    output_path = Path(args.output_dir) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(output_path))
    print(f"Saved {len(rows)} examples to {output_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer,
        trust_remote_code=True,
    )
    student_tokenizer = maybe_add_missing_special_tokens(student_tokenizer)

    teacher_tokenizer = None
    teacher_model = None
    if not args.use_gold_answers:
        teacher_tokenizer, teacher_model = load_teacher_model(args)

    split_plan = [
        (args.train_split, "gsm8k_train", args.limit_train),
        (args.eval_split, "gsm8k_eval", args.limit_eval),
    ]
    for split_name, output_name, limit in split_plan:
        responses = None
        if teacher_model is not None and teacher_tokenizer is not None:
            raw_dataset = load_dataset(
                args.dataset_path,
                args.config_name,
                split=split_name,
                trust_remote_code=True,
            )
            if limit is not None:
                raw_dataset = raw_dataset.select(range(min(limit, len(raw_dataset))))
            responses = generate_teacher_responses(
                list(raw_dataset["question"]),
                teacher_tokenizer,
                teacher_model,
                args,
            )
        build_split(
            split_name,
            output_name,
            limit,
            responses,
            student_tokenizer,
            args,
        )


if __name__ == "__main__":
    main()

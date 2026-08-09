#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"

ARGS=(
  --teacher-model "${DISTILL_TEACHER_MODEL:-Qwen/Qwen3-32B-AWQ}"
  --student-tokenizer "${DISTILL_STUDENT_TOKENIZER:-Qwen/Qwen3-1.7B-Base}"
  --output-dir "${DLLM_DISTILL_DATA_DIR:-outputs/distillation/Qwen3-32B-AWQ}"
  --batch-size "${DISTILL_BATCH_SIZE:-1}"
  --max-length "${DISTILL_MAX_LENGTH:-1024}"
  --max-prompt-length "${DISTILL_MAX_PROMPT_LENGTH:-1024}"
  --max-new-tokens "${DISTILL_MAX_NEW_TOKENS:-1024}"
  --device-map "${DISTILL_DEVICE_MAP:-auto}"
  --torch-dtype "${DISTILL_TORCH_DTYPE:-bfloat16}"
)

if [ -n "${DISTILL_LIMIT_TRAIN:-}" ]; then
  ARGS+=(--limit-train "${DISTILL_LIMIT_TRAIN}")
fi
if [ -n "${DISTILL_LIMIT_EVAL:-}" ]; then
  ARGS+=(--limit-eval "${DISTILL_LIMIT_EVAL}")
fi
if [ "${DISTILL_DO_SAMPLE:-false}" == "true" ]; then
  ARGS+=(
    --do-sample
    --temperature "${DISTILL_TEMPERATURE:-0.6}"
    --top-p "${DISTILL_TOP_P:-0.95}"
    --top-k "${DISTILL_TOP_K:-20}"
  )
fi
if [ "${DISTILL_USE_GOLD_ANSWERS:-false}" == "true" ]; then
  ARGS+=(--use-gold-answers)
fi

python scripts/build_gsm8k_distill.py "${ARGS[@]}" "$@"

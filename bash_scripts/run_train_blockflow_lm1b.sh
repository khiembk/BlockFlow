#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"

NUM_VISIBLE_DEVICES="${NUM_VISIBLE_DEVICES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
RUN_DIR="${RUN_DIR:-/scratch/${USER}/blockflow_outputs}"

# Model arch: AdaptiveBlockFlow (discrete vocab-space VFM) scaled to ~135M
# params (measured: 132.2M), applied to LM1B as a follow-on dataset per the
# original Stage 1 plan's deferred scope. Unlike the WMT14 runs, there's no
# pretrained-T5-embedding warm start here -- LM1B uses bert-base-uncased's
# tokenizer/vocab (matching this repo's existing BD3LM/SetDLM lm1b scripts),
# which doesn't correspond to T5's vocab at all, so vocab_embed/output_layer
# are randomly initialized (no hidden_size=512 constraint from that path, but
# kept at 512 anyway for consistency/reuse of the validated WMT14 shape).
LENGTH=128
BLOCK_SIZE="${BLOCK_SIZE:-16}"
EVAL_BLOCK_SIZE=${BLOCK_SIZE}
WINDOW_SIZE="${WINDOW_SIZE:-16}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
N_LAYERS="${N_LAYERS:-24}"
N_HEADS="${N_HEADS:-8}"
TOKENIZER_NAME="${TOKENIZER_NAME:-bert-base-uncased}"
VOCAB_SIZE="${VOCAB_SIZE:-30522}"
ATTN_BACKEND=sdpa

# Preprocessed LM1B binaries found on scratch (bert-base-uncased tokenized,
# length-128, no `context_mask` field -- AdaptiveBlockFlow/the collator both
# default a missing context_mask to all-zeros, i.e. pure LM: the whole
# sequence is "target", nothing is fixed context, same convention already
# used by this repo's BD3LM/MDLM lm1b training).
LM1B_TRAIN_PATH="${LM1B_TRAIN_PATH:-/scratch/khiem.tran@ad.wsu.edu/mdlm/data_cache/lm1b_train_bs128_unwrapped.dat}"
LM1B_EVAL_PATH="${LM1B_EVAL_PATH:-/scratch/khiem.tran@ad.wsu.edu/mdlm/data_cache/lm1b_test_bs128_unwrapped.dat}"

# Hyperparameters
LR=3e-4
WARMUP_DURATION="2500ba"
# Sized so each of 2 GPUs gets a micro-batch of 272 (grad_accum=1) -- empirically
# measured to land in the ~40-60GB/80GB range (memory scaling was noisy/non-
# linear in this batch-size range via coarse external polling, but consistently
# well clear of the 80GB ceiling).
BATCH_SIZE="${BATCH_SIZE:-544}"
MAX_DURATION="${MAX_DURATION:-200000ba}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000ba}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000ba}"

PRETRAINED_MODEL_NAME_OR_PATH=null

TAG="blockflow_vfm_stage1_discrete_lm1b_135M_win${WINDOW_SIZE}"
LAYERS="layers${N_LAYERS}"
RUN_NAME=lm1b_commit${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_${LAYERS}_hidden${HIDDEN_SIZE}_${TAG}

GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -E 's/.*(A[0-9]+|H100|A6000).*/\1/' | head -n 1)
if [[ -n "${MICRO_BATCH_SIZE:-}" ]]; then
    : # explicit override wins
elif [[ "$GPU_TYPE" == "A100" || "$GPU_TYPE" == "H100" ]]; then
    MICRO_BATCH_SIZE=272
elif [[ "$GPU_TYPE" == "A6000" ]]; then
    MICRO_BATCH_SIZE=64
else
    MICRO_BATCH_SIZE=16
fi
NUM_WORKERS=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  tokenizer=autotokenizer \
  tokenizer.pretrained_model_name_or_path=${TOKENIZER_NAME} \
  collator=blockflow \
  collator@eval_collator=blockflow \
  dataset@train_dataset=lm1b_train \
  dataset@eval_dataset=lm1b_eval \
  train_dataset.dataset_path=${LM1B_TRAIN_PATH} \
  eval_dataset.dataset_path=${LM1B_EVAL_PATH} \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval=${EVAL_INTERVAL} \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  composer.lr_scheduler.t_max=${MAX_DURATION} \
  composer.lr_scheduler.alpha_f=0.0 \
  composer/callbacks=[hf_compatible_checkpointing,lr_monitor,runtime_estimator,save_best_checkpointing,speed_monitor] \
  model=blockflow \
  model.config.attn_backend="sdpa" \
  training.compile_backbone=false \
  model.config.length=${LENGTH} \
  model/backbone@model.config.backbone_config=dit_continuous \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  model.config.backbone_config.n_heads=${N_HEADS} \
  model.config.backbone_config.vocab_size=${VOCAB_SIZE} \
  model.config.backbone_config.attn_backend=${ATTN_BACKEND} \
  model.config.backbone_config.pretrained_embedding_model=null \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  eval_dataloader.batch_size=${MICRO_BATCH_SIZE} \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  window_size=${WINDOW_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval=${SAVE_INTERVAL} \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true

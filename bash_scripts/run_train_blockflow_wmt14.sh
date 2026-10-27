#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"

NUM_VISIBLE_DEVICES="${NUM_VISIBLE_DEVICES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
# Checkpoints are large (~6GB each) and saved frequently; default to local
# scratch (fast NVMe, not the NFS-mounted home dir) rather than REPO_ROOT/outputs.
RUN_DIR="${RUN_DIR:-/scratch/${USER}/blockflow_outputs}"

# Model arch (Adaptive BlockFlow Stage 1: Gaussian-to-categorical VFM, see
# docs/mypapers/main.tex). `block_size` here is the spec's fixed commit size `b`.
LENGTH=200
BLOCK_SIZE="${BLOCK_SIZE:-8}"
EVAL_BLOCK_SIZE=${BLOCK_SIZE}
# hidden_size=512/n_heads=8 matches t5-small's own d_model/head-count exactly --
# required (not just convenient) when PRETRAINED_EMBEDDING_MODEL is set, since
# vocab_embed/output_layer are initialized directly from t5-small's embedding
# table with no projection layer bridging a dimension mismatch.
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
N_LAYERS=12
N_HEADS="${N_HEADS:-8}"
# Default to a de/en-scoped ~32k vocab (T5-small's tokenizer, same choice as the
# sibling CorrFlow/ELF-B project) rather than bert-base-multilingual-cased's
# 119547 (spread across 100+ languages we don't need): smaller dense (L,V)
# buffers -> bigger batches, and less wasted output-space capacity.
# NOTE: VOCAB_SIZE must equal len(tokenizer) *after* maybe_add_missing_special_tokens
# (which may add one token, e.g. T5-small: raw vocab_size=32100 -> len=32101),
# not the tokenizer's raw `.vocab_size` -- verify with scripts/utils.py if you
# change TOKENIZER_NAME.
TOKENIZER_NAME="${TOKENIZER_NAME:-t5-small}"
VOCAB_SIZE="${VOCAB_SIZE:-32101}"
# Initialize + freeze vocab_embed/output_layer from this pretrained model's
# token embeddings (e.g. "t5-small"); null disables this (random init, as before).
PRETRAINED_EMBEDDING_MODEL="${PRETRAINED_EMBEDDING_MODEL:-t5-small}"
ATTN_BACKEND=sdpa

# Hyperparameters
LR=3e-4
WARMUP_DURATION="2500ba"
BATCH_SIZE=256
MAX_DURATION="${MAX_DURATION:-200000ba}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000ba}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000ba}"

PRETRAINED_MODEL_NAME_OR_PATH=null

TAG="blockflow_vfm_stage1_v3_t5embed"
LAYERS="layers${N_LAYERS}"
RUN_NAME=wmt14_commit${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_${LAYERS}_hidden${HIDDEN_SIZE}_${TAG}

# NOTE: AdaptiveBlockFlow materializes dense (micro_batch, LENGTH, VOCAB_SIZE)
# float32 buffers (Gaussian prior, one-hot target, interpolant; spec Section 7
# flags this cost explicitly). With VOCAB_SIZE=119547 this is far more
# memory-hungry per example than token-id-only discrete denoisers, so the
# micro-batch size here is much smaller than other bash_scripts/ use at this
# hidden size.
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -E 's/.*(A[0-9]+|H100|A6000).*/\1/' | head -n 1)
if [[ -n "${MICRO_BATCH_SIZE:-}" ]]; then
    : # explicit override wins
elif [[ "$GPU_TYPE" == "A100" || "$GPU_TYPE" == "H100" ]]; then
    MICRO_BATCH_SIZE=8
elif [[ "$GPU_TYPE" == "A6000" ]]; then
    MICRO_BATCH_SIZE=4
else
    MICRO_BATCH_SIZE=2
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
  dataset@train_dataset=wmt14_train \
  dataset@eval_dataset=wmt14_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval=${EVAL_INTERVAL} \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
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
  model.config.backbone_config.pretrained_embedding_model=${PRETRAINED_EMBEDDING_MODEL} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  eval_dataloader.batch_size=${MICRO_BATCH_SIZE} \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${RUN_DIR}/${RUN_NAME} \
  composer.trainer.save_interval=${SAVE_INTERVAL} \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true

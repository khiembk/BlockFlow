#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh"

NUM_VISIBLE_DEVICES="${NUM_VISIBLE_DEVICES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
# Checkpoints are large and saved frequently; default to local scratch (fast
# NVMe, not the NFS-mounted home dir) rather than REPO_ROOT/outputs.
RUN_DIR="${RUN_DIR:-/scratch/${USER}/blockflow_outputs}"

# Model arch (Adaptive BlockFlow Stage 1 v3: Gaussian-to-T5-embedding VFM with
# a frozen T5-small context encoder + a separately trained decoder branch, see
# docs/mypapers/main.tex and the Stage 1 v3 plan). `block_size` is the spec's
# fixed commit size `b`. predict==commit (16==16), matching BD3LM/SetDLM's own
# proven convention (e.g. BD3LM cnn/lm1b use block_size=16 as a single atomic
# predict-and-commit unit) rather than Stage 1's "predict more than you
# commit" variant -- scores and commits the exact same 16-token block.
LENGTH=200
BLOCK_SIZE="${BLOCK_SIZE:-16}"
EVAL_BLOCK_SIZE=${BLOCK_SIZE}
# Bounds the loss to the nearest WINDOW_SIZE active positions (canvas order)
# per round instead of every remaining target position (which can be nearly
# the full remaining target length in early rounds, diluting gradient signal
# away from the near-term positions that actually get committed). Matches
# this repo's BD3LM/SetDLM bounded denoising granularity (8-16 tokens) rather
# than Stage 1's original unbounded "predict everything remaining" default.
# Set equal to BLOCK_SIZE so the scored window and the committed block coincide.
WINDOW_SIZE="${WINDOW_SIZE:-16}"
# hidden_size is free now (no longer forced to equal T5's 512-dim output) --
# the trainable BottleneckAdapter bridges T5's space to this width. Sized to
# match ELF-B (ELF's own base config for this exact WMT14 de-en task: depth=12,
# hidden_size=768, num_heads=12, mlp_ratio=4.0 -- see ELF's src/modules/model.py
# and src/configs/training_configs/train_de-en_ELF-B.yml), which lands our
# trainable param count at ~97M (measured: 97.2M trainable / 132.5M total).
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
N_LAYERS=12
N_HEADS="${N_HEADS:-12}"
BOTTLENECK_DIM="${BOTTLENECK_DIM:-128}"
T5_MODEL_NAME="${T5_MODEL_NAME:-t5-small}"
T5_EMBED_DIM="${T5_EMBED_DIM:-512}"
# Flow-interpolant Gaussian prior scale and decoder-branch corruption schedule
# (ELF's own WMT14 hyperparameters, see Stage 1 v3 plan Key design #2-3).
NOISE_SCALE="${NOISE_SCALE:-2.0}"
DECODER_PROB="${DECODER_PROB:-0.2}"
DECODER_P_MEAN="${DECODER_P_MEAN:--1.5}"
DECODER_P_STD="${DECODER_P_STD:-0.8}"
DECODER_NOISE_SCALE="${DECODER_NOISE_SCALE:-2.0}"
# Empirically measured t5-small encoder-output std (see Stage 1 v3 plan); an
# activation-scale normalization constant, not a weight copy.
LATENT_STD="${LATENT_STD:-0.15}"
# NOTE: VOCAB_SIZE must equal len(tokenizer) *after* maybe_add_missing_special_tokens
# (which may add one token, e.g. T5-small: raw vocab_size=32100 -> len=32101),
# not the tokenizer's raw `.vocab_size` -- verify with scripts/utils.py if you
# change TOKENIZER_NAME. This is only the decoder branch's output size now
# (the flow itself runs in T5_EMBED_DIM-space), not a dense flow buffer width.
TOKENIZER_NAME="${TOKENIZER_NAME:-t5-small}"
VOCAB_SIZE="${VOCAB_SIZE:-32101}"
ATTN_BACKEND=sdpa

# Hyperparameters
LR=3e-4
WARMUP_DURATION="2500ba"
BATCH_SIZE=256
MAX_DURATION="${MAX_DURATION:-200000ba}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000ba}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000ba}"

PRETRAINED_MODEL_NAME_OR_PATH=null

TAG="blockflow_vfm_stage1_v3_t5embedspace_win${WINDOW_SIZE}"
LAYERS="layers${N_LAYERS}"
RUN_NAME=wmt14_commit${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_${LAYERS}_hidden${HIDDEN_SIZE}_${TAG}

# NOTE: AdaptiveBlockFlowEmbed materializes dense (micro_batch, LENGTH,
# T5_EMBED_DIM) float32 buffers (Gaussian prior, T5 target embedding,
# interpolant) instead of (micro_batch, LENGTH, VOCAB_SIZE) -- a ~63x
# reduction vs. the one-hot-vocab-space Stage 1 for VOCAB_SIZE=32101, so
# micro-batch sizes here can be much larger. Each training step also runs one
# frozen T5-small encoder forward pass (cheap relative to the trainable body
# at this scale) plus the usual backbone forward/backward.
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -E 's/.*(A[0-9]+|H100|A6000).*/\1/' | head -n 1)
if [[ -n "${MICRO_BATCH_SIZE:-}" ]]; then
    : # explicit override wins
elif [[ "$GPU_TYPE" == "A100" || "$GPU_TYPE" == "H100" ]]; then
    # Empirically measured: hidden_size=512/num_layers=12 at this micro-batch
    # peaks at ~46GB/80GB on an H100 (5-batch probe, grad_accum=1) -- comfortable
    # headroom at a micro-batch equal to the full global batch size.
    MICRO_BATCH_SIZE=256
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
  dataset@train_dataset=wmt14_train \
  dataset@eval_dataset=wmt14_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval=${EVAL_INTERVAL} \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  composer/callbacks=[hf_compatible_checkpointing,lr_monitor,runtime_estimator,save_best_checkpointing,speed_monitor,bleu_eval] \
  composer.callbacks.bleu_eval.num_samples=${BLEU_NUM_SAMPLES:-20} \
  composer.callbacks.bleu_eval.eval_interval=${BLEU_EVAL_INTERVAL:-2000} \
  model=blockflow_embed \
  model.config.attn_backend="sdpa" \
  training.compile_backbone=false \
  model.config.length=${LENGTH} \
  model.config.t5_model_name=${T5_MODEL_NAME} \
  model.config.t5_embed_dim=${T5_EMBED_DIM} \
  model.config.latent_std=${LATENT_STD} \
  model.config.noise_scale=${NOISE_SCALE} \
  model.config.decoder_prob=${DECODER_PROB} \
  model.config.decoder_p_mean=${DECODER_P_MEAN} \
  model.config.decoder_p_std=${DECODER_P_STD} \
  model.config.decoder_noise_scale=${DECODER_NOISE_SCALE} \
  model/backbone@model.config.backbone_config=dit_continuous_embed \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  model.config.backbone_config.n_heads=${N_HEADS} \
  model.config.backbone_config.vocab_size=${VOCAB_SIZE} \
  model.config.backbone_config.attn_backend=${ATTN_BACKEND} \
  model.config.backbone_config.t5_embed_dim=${T5_EMBED_DIM} \
  model.config.backbone_config.bottleneck_dim=${BOTTLENECK_DIM} \
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

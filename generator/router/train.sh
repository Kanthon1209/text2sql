#!/usr/bin/env bash
# Router R0：问题 -> source/databases/tables。Qwen3.5-4B LoRA。
# 复现：先运行 export_jsonl.py，再运行本脚本。超参全部写在这里。
# Qwen3.5 在 swift 里走视觉加载器，context_8 需要：
#   pip install "qwen-vl-utils>=0.0.14" decord -i https://pypi.org/simple
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
export NO_PROXY='*'

ROOT="/data/k/runs/router-r0"
DATA="$ROOT/train.jsonl"
mkdir -p "$ROOT"
test -f "$DATA"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONHASHSEED=20261001

swift sft \
  --model /data/models/Qwen3.5-4B \
  --model_type qwen3_5 \
  --template qwen3_5 \
  --use_hf true \
  --enable_thinking false \
  --dataset "$DATA" \
  --split_dataset_ratio 0 \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 1e-4 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --warmup_ratio 0.05 \
  --gradient_checkpointing true \
  --seed 20261001 \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --eval_strategy no \
  --report_to none \
  --dataloader_num_workers 2 \
  --dataset_num_proc 1 \
  --output_dir "$ROOT"

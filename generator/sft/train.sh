#!/usr/bin/env bash
# 生成器 E09：完整提示词 -> 标注查询。Qwen3.5-4B LoRA。
# 复现：先运行 export_jsonl.py，卡空出来后再运行本脚本。超参全部写在这里。
# 默认 1 号卡。E09 推理若仍占着 1 号卡，不要同时启动。
# 套上 Qwen3.5 对话模板后最长 2728 token，max_length 4096 不会截断。
# 4 个 epoch：SQL 比 Router 的短 JSON 长，2 遍不够；再多主要是贴训练题原句。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
export NO_PROXY='*'

ROOT="/data/k/runs/generator-e09"
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
  --num_train_epochs 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 4096 \
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

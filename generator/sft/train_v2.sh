#!/usr/bin/env bash
# 第二阶段生成器 LoRA：训练输入与测试时 v3 的提示词完全一致。
# 数据由 export_jsonl_v2.py 生成：检索规则 0.65 过滤、相似案例同库 0.6、
# 附加系统句、失败重写多轮样本（首轮 loss=False）。
# 0 和 1 两张卡一起跑，两张卡都空着。
# 每卡批次 4，梯度累计 2，全局批次 16。expandable_segments 避免长样本时的显存碎片。
# 4 个 epoch 的权重都留下，测试集上第 2 个 epoch 优于最后一轮。
# max_length 6144 保证不截断。
set -euo pipefail

ROOT="/data/k/runs/generator-e09-sft2"
MODEL="/data/models/Qwen3.5-4B"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_PORT="${MASTER_PORT:-29502}"
export PYTHONHASHSEED=20261001
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

swift sft \
    --model "$MODEL" \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --use_hf true \
    --enable_thinking false \
    --dataset "$ROOT/train.jsonl" \
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
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 6144 \
    --warmup_ratio 0.05 \
    --gradient_checkpointing true \
    --seed 20261001 \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 4 \
    --eval_strategy no \
    --report_to none \
    --dataloader_num_workers 2 \
    --dataset_num_proc 1 \
    --output_dir "$ROOT"
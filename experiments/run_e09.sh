#!/usr/bin/env bash
# 改进后的 E09：表卡片、同数据源相似案例、按数据源的方言约束。
# 固定 1 号卡、端口 8003，避免占用 0 号卡上已有的 8002 服务。
# 结果写到 /data/k/runs/e09。
set -euo pipefail
export GPU=1
export PORT=8003
OUT=/data/k/runs/e09
ROOT=/data/k
PIDF="$ROOT/run/Qwen3.5-4B-gpu1.pid"
LOGF="$ROOT/log/serve_Qwen3.5-4B-gpu1.log"
mkdir -p "$OUT" "$ROOT/log" "$ROOT/run"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
export NO_PROXY='*'

if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[e09] 启动 Qwen3.5-4B gpu=$GPU port=$PORT"
  CUDA_VISIBLE_DEVICES="$GPU" nohup swift deploy \
    --model /data/models/Qwen3.5-4B \
    --host 127.0.0.1 \
    --port "$PORT" \
    --served_model_name Qwen3.5-4B \
    --infer_backend vllm \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --enable_thinking false \
    --torch_dtype bfloat16 \
    --max_length 8192 \
    --max_new_tokens 1024 \
    --temperature 0 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_max_model_len 8192 \
    >"$LOGF" 2>&1 &
  echo $! >"$PIDF"
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
fi
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null
python /data/k/experiments/run_ready.py \
  --output "$OUT" \
  --experiments E09 \
  --base-url "http://127.0.0.1:${PORT}/v1"

#!/usr/bin/env bash
# 只启动 Qwen3.5-4B。环境 context_8，端口 8002。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/log"
PID_DIR="$ROOT/run"
mkdir -p "$LOG" "$PID_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
export NO_PROXY='*'

GPU="${GPU:-0}"
PORT="${PORT:-8002}"
PIDF="$PID_DIR/Qwen3.5-4B.pid"
LOGF="$LOG/serve_Qwen3.5-4B.log"

if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "[serve] Qwen3.5-4B 已在运行 pid=$(cat "$PIDF") port=$PORT"
  exit 0
fi

echo "[serve] 启动 Qwen3.5-4B gpu=$GPU port=$PORT"
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
echo "[serve] pid=$! log=$LOGF"

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[serve] :${PORT} 就绪"
    exit 0
  fi
  sleep 3
done
echo "[serve] 未就绪，查看 $LOGF"
exit 1

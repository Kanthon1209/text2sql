#!/usr/bin/env bash
# 用 ms-swift deploy 启动两个常驻 OpenAI 兼容推理服务（vLLM 后端）
#   GPU0 -> XiYanSQL-QwenCoder-3B-2504  :8001
#   GPU1 -> Qwen3.5-4B                  :8002
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src"
LOG="$ROOT/log"
PID_DIR="$ROOT/run"
mkdir -p "$LOG" "$PID_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
export NO_PROXY='*'

MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
# 两个 4B 模型同跑一张 40G A100，每家给 ~42% 显存（≈16.8G），权重 8G + KV cache 余量足够
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.42}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
XIYAN_GPU="${XIYAN_GPU:-1}"
QWEN_GPU="${QWEN_GPU:-1}"
XIYAN_PORT="${XIYAN_PORT:-8001}"
QWEN_PORT="${QWEN_PORT:-8002}"

start_one() {
  local name="$1" gpu="$2" port="$3" model="$4"
  shift 4
  local pidf="$PID_DIR/${name}.pid"
  local logf="$LOG/serve_${name}.log"

  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "[serve] $name 已在运行 pid=$(cat "$pidf") port=$port"
    return 0
  fi

  echo "[serve] 启动 $name gpu=$gpu port=$port -> $logf"
  CUDA_VISIBLE_DEVICES="$gpu" nohup swift deploy \
    --model "$model" \
    --host 0.0.0.0 \
    --port "$port" \
    --served_model_name "$name" \
    --infer_backend vllm \
    --torch_dtype bfloat16 \
    --max_length "$MAX_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 0.1 \
    --top_p 0.8 \
    --vllm_gpu_memory_utilization "$GPU_MEM_UTIL" \
    --vllm_max_model_len "$MAX_LENGTH" \
    --vllm_max_num_seqs "$MAX_NUM_SEQS" \
    "$@" \
    >"$logf" 2>&1 &
  echo $! >"$pidf"
  echo "[serve] $name pid=$! log=$logf"
}

start_one "XiYanSQL-3B" "$XIYAN_GPU" "$XIYAN_PORT" \
  "/data/models/XiYanSQL-QwenCoder-3B-2504" \
  --model_type qwen2 \
  --template qwen2_5

start_one "Qwen3.5-4B" "$QWEN_GPU" "$QWEN_PORT" \
  "/data/models/Qwen3.5-4B" \
  --model_type qwen3_5 \
  --template qwen3_5 \
  --enable_thinking false

echo "[serve] 等待 /v1/models 就绪 ..."
for port in "$XIYAN_PORT" "$QWEN_PORT"; do
  ok=0
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ok=1; break
    fi
    sleep 3
  done
  if [[ $ok -eq 1 ]]; then
    echo "[serve] :${port} 就绪"
  else
    echo "[serve] :${port} 60s 内未就绪，请查 $LOG/serve_*.log"
    exit 1
  fi
done
echo "[serve] 两个服务均已就绪"

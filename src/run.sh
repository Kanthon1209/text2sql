#!/usr/bin/env bash
# 一键：启动服务 -> 推理前50条 -> 停服务。完整日志写 /data/k/log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src"
LOG="$ROOT/log"
mkdir -p "$LOG"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER="$LOG/run_${TS}.log"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
export NO_PROXY='*'

echo "[run] ts=$TS" | tee "$MASTER"

# 1. 启动服务
echo "[run] 启动推理服务 ..." | tee -a "$MASTER"
bash "$SRC/serve.sh" 2>&1 | tee -a "$MASTER"

# 2. 推理
echo "[run] 开始推理前50条 ..." | tee -a "$MASTER"
python "$SRC/infer.py" --limit 50 --workers 8 --ts "$TS" 2>&1 | tee -a "$MASTER"

# 3. 停服务
echo "[run] 停止服务 ..." | tee -a "$MASTER"
bash "$SRC/stop.sh" 2>&1 | tee -a "$MASTER"

echo "[run] 全部完成，日志目录: $LOG" | tee -a "$MASTER"
ls -la "$LOG" | tee -a "$MASTER"

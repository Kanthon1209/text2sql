#!/usr/bin/env bash
# 测试集目前只跑 E01。结果写到 /data/k/runs/experiments-test-v1。
set -euo pipefail
export GPU="${GPU:-0}"
export PORT="${PORT:-8002}"
mkdir -p /data/k/runs/experiments-test-v1
bash /data/k/generator/serve.sh
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null
source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8
python /data/k/experiments/run_test.py --base-url "http://127.0.0.1:${PORT}/v1" "$@"

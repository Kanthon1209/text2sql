#!/usr/bin/env bash
# 先确保 Qwen3.5-4B 在 8002，再按 E01、E02、E03、E05 的顺序跑完。
set -euo pipefail
export GPU="${GPU:-0}"
export PORT="${PORT:-8002}"
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
cd /data/k/generator
python /data/k/experiments/run_ready.py --base-url "http://127.0.0.1:${PORT}/v1" "$@"

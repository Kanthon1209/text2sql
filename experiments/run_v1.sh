#!/usr/bin/env bash
# 编号仍是 E02、E03、E05。PostgreSQL 与 Elasticsearch 使用改过的提示词，
# 结果写到 /data/k/runs/experiments-v1。MySQL 和 SQLite 提示词未改，沿用 experiments 里的预测。
set -euo pipefail
export GPU="${GPU:-0}"
export PORT="${PORT:-8002}"
OUT=/data/k/runs/experiments-v1
mkdir -p "$OUT"
python3 - << 'PY'
import json
from pathlib import Path
src_root = Path("/data/k/runs/experiments")
dst_root = Path("/data/k/runs/experiments-v1")
keep = {"MYSQL", "SQLite"}
for name in ("E02", "E03", "E05"):
    dst = dst_root / name / "predictions.jsonl"
    if dst.exists() and dst.stat().st_size:
        print(f"{name} 已有预测，不覆盖", flush=True)
        continue
    src = src_root / name / "predictions.jsonl"
    kept = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source") in keep:
            kept.append(line)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"{name} 沿用 {len(kept)} 条", flush=True)
PY
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
python /data/k/experiments/run_ready.py \
  --output "$OUT" \
  --experiments E02,E03,E05 \
  --base-url "http://127.0.0.1:${PORT}/v1"

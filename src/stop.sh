#!/usr/bin/env bash
# 停止本套 ms-swift 推理服务
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT/run"
for name in XiYanSQL-3B Qwen3.5-4B; do
  pidf="$PID_DIR/${name}.pid"
  if [[ -f "$pidf" ]]; then
    pid="$(cat "$pidf")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop] kill $name pid=$pid"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidf"
  fi
done
# 兜底：按 served_model_name 杀残留 swift deploy
pkill -f "served_model_name XiYanSQL-3B" 2>/dev/null || true
pkill -f "served_model_name Qwen3.5-4B" 2>/dev/null || true
echo "[stop] done"

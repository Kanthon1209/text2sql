#!/usr/bin/env bash
# 用里程碑里的代码和权重重跑 v5 推理。不改 /data/k 里的工作区文件。
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
export NO_PROXY='*'
export PYTHONUNBUFFERED=1
M=/data/k/_milestone/v5-727
OUT=${1:-/tmp/v5-rerun}
mkdir -p "$OUT"
cp "$M/data/QA_test.json" "$OUT/QA_test.json"
TMP=$(mktemp -d)
sed -e "s|/data/k/experiments|$M/code|" -e "s|/data/k/generator|$M/generator|" \
  "$M/code/run_test_full.py" > "$TMP/run_test_full.py"
cp "$M/code/guard.py" "$M/code/run_ready.py" "$TMP/"
CKPT=$M/weights
CUDA_VISIBLE_DEVICES=0 python -u "$TMP/run_test_full.py" infer \
  --output "$OUT" --shard 0 --shards 2 --device cuda --adapter "$CKPT" &
p0=$!
CUDA_VISIBLE_DEVICES=1 python -u "$TMP/run_test_full.py" infer \
  --output "$OUT" --shard 1 --shards 2 --device cuda --adapter "$CKPT" &
p1=$!
fail=0
wait "$p0" || fail=1
wait "$p1" || fail=1
if [[ "$fail" -ne 0 ]]; then
  echo "推理失败" >&2
  exit 1
fi
python -u "$TMP/run_test_full.py" merge --output "$OUT"
python -u "$TMP/run_test_full.py" submit --output "$OUT"
echo "完成 $OUT/result.json"

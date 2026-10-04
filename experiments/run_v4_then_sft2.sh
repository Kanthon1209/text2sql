#!/usr/bin/env bash
# 等第一版生成器 LoRA 训练结束，用最终权重跑 test-full-v4，
# 然后自动开始第二阶段训练（generator-e09-sft2）。
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8

V1_RUN=/data/k/runs/generator-e09/v0-20261004-182257
CKPT=$V1_RUN/checkpoint-1480
V4=/data/k/runs/test-full-v4
V2=/data/k/runs/test-full-v2

echo "$(date '+%F %T') 等第一版训练出 checkpoint-1480"
until [ -f "$CKPT/adapter_model.safetensors" ]; do
  sleep 60
done
echo "$(date '+%F %T') checkpoint-1480 已保存"

# 等训练进程完全退出，释放显存
for _ in $(seq 1 30); do
  if ! pgrep -f "swift/cli/sft.py" > /dev/null 2>&1; then
    break
  fi
  sleep 60
done
sleep 20
echo "$(date '+%F %T') 第一版训练结束，开始 v4"

mkdir -p "$V4"
for name in router.jsonl router_meta.json similar_top10.json knowledge_rules.json QA_test.json manifest.json; do
  cp "$V2/$name" "$V4/$name"
done

CUDA_VISIBLE_DEVICES=0 python /data/k/experiments/run_test_full.py infer \
  --output "$V4" --device cuda --adapter "$CKPT" 2>&1 | tee "$V4/infer.log"

python /data/k/experiments/run_test_full.py merge --output "$V4"
python /data/k/experiments/run_test_full.py submit --output "$V4"

python - << 'PY'
import json, sys
from collections import Counter
sys.path.insert(0, "/data/lj")
import os
os.environ["VERIFY_SQL_LOG"] = "/dev/null"
from verify_sql import check_detail
ours = {r["question_id"]: r for r in json.load(open("/data/k/runs/test-full-v4/result.json"))}
gpt = {r["question_id"]: r for r in json.load(open("/data/submit/gpt-v1/result.json"))}
MAP = {"MYSQL": "mysql", "POSTGRESQL": "postgresql", "SQLite": "sqlite", "Elasticsearch": "elasticsearch"}
def kind(r):
    return r.get("db") or ("CROSS" if "db1" in r else "?")
equal = diff = fail = mismatch = 0
by = Counter()
for qid, o in ours.items():
    g = gpt[qid]
    ko, kg = kind(o), kind(g)
    if ko != kg:
        mismatch += 1
        continue
    if ko == "CROSS":
        pred = {"db1": o["db1"], "query1": o["query1"], "db2": o["db2"], "query2": o["query2"]}
        gold = {"db1": g["db1"], "query1": g["query1"], "db2": g["db2"], "query2": g["query2"]}
        db = "federated"
    else:
        pred, gold, db = o.get("query"), g.get("query"), MAP[ko]
    d = check_detail(db, pred, gold, column_order_sensitive=True, timeout=12)
    if d["equal"]:
        equal += 1
        by[ko + " equal"] += 1
    elif not d["ok"]:
        fail += 1
        by[ko + " fail"] += 1
    else:
        diff += 1
        by[ko + " diff"] += 1
print("V4 vs gpt equal", equal, "diff", diff, "pair_fail", fail, "source_mismatch", mismatch)
print(dict(by))
PY

echo "$(date '+%F %T') 开始第二阶段训练"
mkdir -p /data/k/runs/generator-e09-sft2
bash /data/k/generator/sft/train_v2.sh 2>&1 | tee /data/k/runs/generator-e09-sft2/train.log
echo "$(date '+%F %T') 全部完成"
#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate context_8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
export NO_PROXY='*'
export PYTHONUNBUFFERED=1
OUT=/data/k/runs/test-full-v5b
CKPT=/data/k/runs/generator-e09-sft2/v4-20261005-094854/checkpoint-740
PY=/data/k/experiments/run_test_full.py
echo "$(date '+%F %T') infer start gpu0 $CKPT"
CUDA_VISIBLE_DEVICES=0 python -u "$PY" infer --output "$OUT" --device cuda --adapter "$CKPT" >"$OUT/infer.log" 2>&1
echo "$(date '+%F %T') merge"
python -u "$PY" merge --output "$OUT" | tee "$OUT/merge.log"
python -u "$PY" submit --output "$OUT" | tee "$OUT/submit.log"
python - << 'PY'
import json, os, sys
from collections import Counter
sys.path.insert(0, "/data/lj")
os.environ["VERIFY_SQL_LOG"] = "/dev/null"
from verify_sql import check_detail
ours = {r["question_id"]: r for r in json.load(open("/data/k/runs/test-full-v5b/result.json"))}
gpt = {r["question_id"]: r for r in json.load(open("/data/submit/gpt-v1/result.json"))}
v5 = {r["question_id"]: r for r in json.load(open("/data/k/runs/test-full-v5/result.json"))}
MAP = {"MYSQL": "mysql", "POSTGRESQL": "postgresql", "SQLite": "sqlite", "Elasticsearch": "elasticsearch"}
def kind(r):
    return r.get("db") or ("CROSS" if "db1" in r else "?")
def compare(other, label):
    equal = diff = fail = mismatch = 0
    by = Counter()
    for qid, o in ours.items():
        g = other[qid]
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
    print(label, "equal", equal, "diff", diff, "pair_fail", fail, "source_mismatch", mismatch)
    print(dict(sorted(by.items())))
compare(gpt, "V5B vs gpt")
compare(v5, "V5B vs v5")
PY
echo "$(date '+%F %T') done"

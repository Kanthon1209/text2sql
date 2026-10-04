#!/usr/bin/env python3
"""把 E09 的提示词导出成生成器的 ms-swift jsonl。

user 与 /data/k/experiments/run_ready.py 里 E09 的推理提示词相同：
标注的数据源、库名、表名，表卡片，标注知识正文，同一数据源的 2 条相似案例。
assistant 是该步的标注查询。跨源题拆成两条，分别学习 query1 和 query2。
不含本题自己的查询。
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/data/k/experiments")
from run_ready import DATA, bare_index, load_cases, load_knowledge, tasks_for  # noqa: E402
from schema import load_cards, load_tables  # noqa: E402

OUT = Path("/data/k/runs/generator-e09/train.jsonl")
MANIFEST = Path("/data/k/runs/generator-e09/export_manifest.json")


def main() -> None:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    tables = load_tables()
    bare = bare_index(tables)
    knowledge = load_knowledge()
    cases = load_cases(rows)
    cards = load_cards()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    steps: Counter[int] = Counter()
    chars = []
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            for task in tasks_for("E09", row, tables, bare, knowledge, cases, cards):
                messages = list(task["messages"])
                messages.append({"role": "assistant", "content": task["gold"]})
                record = {
                    "question_id": task["question_id"],
                    "step": task["step"],
                    "source": task["source"],
                    "messages": messages,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[task["source"]] += 1
                steps[task["step"]] += 1
                chars.append(sum(len(item["content"]) for item in messages))
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    chars.sort()
    manifest = {
        "variant": "generator-e09",
        "source": str(DATA),
        "output": str(OUT),
        "prompt": "与 experiments/run_ready.py 的 E09 相同。数据源、库名、表名和知识用标注，相似案例来自 similar_top10_scored.json。",
        "rows": sum(counts.values()),
        "questions": len(rows),
        "by_source": dict(counts),
        "by_step": {str(key): steps[key] for key in sorted(steps)},
        "chars": {
            "min": chars[0],
            "p50": chars[len(chars) // 2],
            "p90": chars[int(len(chars) * 0.9)],
            "p99": chars[int(len(chars) * 0.99)],
            "max": chars[-1],
        },
        "sha256": digest,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""把 v2-k 题库导出成 Router R0 的 ms-swift jsonl。

R0 的 user 只有问题，不含库清单。assistant 是 source、databases、tables；
跨源再加 steps。不写入 SQL、DDL 和知识正文。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

SRC = Path("/data/data/v2-k/QA_train.json")
OUT = Path("/data/k/runs/router-r0/train.jsonl")
MANIFEST = Path("/data/k/runs/router-r0/export_manifest.json")

SYSTEM = (
    "你是数据源路由器。根据问题判断数据源、数据库和数据表。"
    "只输出一个 JSON 对象，不要 SQL，不要解释。\n"
    '单源格式：{"source":"MYSQL","databases":["库名"],"tables":["库名.表名"]}\n'
    '跨源格式：{"source":"CROSS","databases":["库名"],"tables":["库名.表名"],'
    '"steps":[{"db":"MYSQL","databases":["库名"],"tables":["库名.表名"]}]}\n'
    "source 只能是 MYSQL、SQLite、POSTGRESQL、Elasticsearch、CROSS。"
)


def target(row: dict) -> dict:
    body = {
        "source": row["source"],
        "databases": list(row.get("databases") or []),
        "tables": list(row.get("tables") or []),
    }
    if row["source"] == "CROSS":
        body["steps"] = [
            {
                "db": step["db"],
                "databases": list(step.get("databases") or []),
                "tables": list(step.get("tables") or []),
            }
            for step in row["steps"]
        ]
    return body


def main() -> None:
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    skipped = []
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not row.get("parse_ok", True):
                skipped.append(row["question_id"])
                continue
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"问题：{row['question']}"},
                    {"role": "assistant", "content": json.dumps(target(row), ensure_ascii=False)},
                ]
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[row["source"]] += 1
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    manifest = {
        "source": str(SRC),
        "output": str(OUT),
        "variant": "router-r0",
        "seed_note": "导出无随机。题库顺序与 QA_train.json 一致。",
        "rows": sum(counts.values()),
        "by_source": dict(counts),
        "skipped_parse_failed": skipped,
        "sha256": digest,
        "system": SYSTEM,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "by_source", "sha256", "output")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

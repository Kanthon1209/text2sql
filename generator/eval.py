#!/usr/bin/env python3
"""用未微调的 Qwen3.5-4B 测试生成器。

Schema 来自 v2-k 的标准表，不是模型自己选的。
这样测的是「表已经给对时，基座能不能写出查询」。
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

from prompt import build_user, load_knowledge, system_for
from schema import load_tables, lookup

DATA = Path("/data/data/v2-k/QA_train.json")
OUT = Path("/data/k/runs/generator-base")


def norm(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "")
    text = re.sub(r"^```(?:sql|json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    return re.sub(r"\s+", " ", text.strip().rstrip(";")).lower()


def gold_query(row: dict, step: int) -> str:
    if row["source"] == "CROSS":
        return row["query1"] if step == 1 else row["query2"]
    return row["query"]


def tasks_for(row: dict, tables: dict[str, str], knowledge: dict[int, str] | None) -> list[dict]:
    source = row["source"]
    if source != "CROSS":
        schema, missing = lookup(tables, row["tables"])
        if source == "Elasticsearch":
            schema, missing = lookup(tables, ["medical_institutions"])
        texts = []
        if knowledge is not None:
            texts = [knowledge[kid] for kid in row.get("knowledge_ids") or [] if kid in knowledge]
        return [{
            "question_id": row["question_id"],
            "step": 0,
            "source": source,
            "gold": gold_query(row, 0),
            "missing_tables": missing,
            "messages": [
                {"role": "system", "content": system_for(source)},
                {"role": "user", "content": build_user(row["question"], source, schema, texts)},
            ],
        }]

    made = []
    for step_index, step in enumerate(row["steps"], start=1):
        schema, missing = lookup(tables, step["tables"])
        note = "这是跨源第一步，查询结果将作为第二步的输入。" if step_index == 1 else (
            "这是跨源第二步。请用 result 代表第一步的执行结果。\n第一步查询：" + row["query1"]
        )
        made.append({
            "question_id": row["question_id"],
            "step": step_index,
            "source": step["db"],
            "gold": gold_query(row, step_index),
            "missing_tables": missing,
            "messages": [
                {"role": "system", "content": system_for(step["db"])},
                {"role": "user", "content": build_user(row["question"], step["db"], schema, None, note)},
            ],
        })
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--sources", default="MYSQL,SQLite,POSTGRESQL,Elasticsearch")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--with-knowledge", action="store_true")
    parser.add_argument("--include-cross", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    wanted = {item.strip() for item in args.sources.split(",") if item.strip()}
    if args.include_cross:
        wanted.add("CROSS")
    rows = [row for row in json.loads(args.data.read_text(encoding="utf-8")) if row["source"] in wanted and row.get("parse_ok")]
    if args.limit:
        rows = rows[: args.limit]
    tables = load_tables()
    knowledge = load_knowledge() if args.with_knowledge else None
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    args.output.mkdir(parents=True, exist_ok=True)
    pred_path = args.output / "predictions.jsonl"

    stats = Counter()
    by_source: dict[str, Counter] = {}
    with pred_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            for task in tasks_for(row, tables, knowledge):
                started = time.time()
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=task["messages"],
                    temperature=0,
                    max_tokens=1024,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                text = completion.choices[0].message.content or ""
                exact = norm(text) == norm(task["gold"])
                group = by_source.setdefault(task["source"], Counter())
                group["n"] += 1
                group["exact"] += int(exact)
                stats["n"] += 1
                stats["exact"] += int(exact)
                handle.write(json.dumps({
                    "question_id": task["question_id"],
                    "step": task["step"],
                    "source": task["source"],
                    "exact": exact,
                    "missing_tables": task["missing_tables"],
                    "gold": task["gold"],
                    "pred": text,
                    "seconds": round(time.time() - started, 3),
                }, ensure_ascii=False) + "\n")
            print(f"{index}/{len(rows)} exact {stats['exact']}/{stats['n']}", flush=True)

    summary = {
        "n": stats["n"],
        "exact": stats["exact"],
        "exact_rate": stats["exact"] / stats["n"] if stats["n"] else 0,
        "by_source": {key: {"n": value["n"], "exact": value["exact"]} for key, value in by_source.items()},
        "note": "字符串完全匹配，不是执行正确率。Schema 使用标准表。",
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

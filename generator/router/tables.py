#!/usr/bin/env python3
"""库名已经给定时，让基座模型从该库的表名单里勾选表。不加载 Router LoRA。

默认使用 /data/data/v2-k/QA_train.json 的标注库名。
结果写到 /data/k/runs/router-tables/gold-db/。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/data/k/generator")
from schema import load_tables  # noqa: E402

DATA = Path("/data/data/v2-k/QA_train.json")
OUT = Path("/data/k/runs/router-tables/gold-db")
MODEL = "/data/models/Qwen3.5-4B"

SYSTEM = (
    "你是选表器。数据库已经确定。下面列出这些库中的全部数据表和字段。"
    "只选择撰写这条查询需要的表，表名必须来自名单。"
    '只输出一个 JSON 对象，不要解释。格式：{"tables":["库名.表名"]}。'
    "Elasticsearch 没有数据表，tables 输出空数组。"
)


def columns_of(key: str, ddl: str) -> list[str]:
    if key == "medical_institutions" or ddl.lstrip().startswith("PUT"):
        body = re.split(r"\n(?:-- |CREATE TABLE)", ddl, maxsplit=1)[0]
        return re.findall(r'"(\w+)"\s*:\s*\{', body)
    skip = {"constraint", "primary", "foreign", "unique", "key", "index", "check", "create"}
    found = []
    for line in ddl.splitlines():
        text = line.strip().rstrip(",")
        if not text or text.startswith((")", "--", "CREATE", "create")):
            continue
        match = re.match(r'["`\[]?([A-Za-z_]\w*)["`\]]?', text)
        if not match:
            continue
        name = match.group(1)
        if name.lower() in skip:
            continue
        if name not in found:
            found.append(name)
    return found


def candidates_for(tables: dict[str, str], databases: list[str]) -> list[str]:
    chosen = {item.lower() for item in databases}
    found = [key for key in tables if key.split(".", 1)[0] in chosen or key in chosen]
    return sorted(found)


def build_user(question: str, source: str, databases: list[str], catalog: list[tuple[str, list[str]]]) -> str:
    lines = [f"问题：{question}", f"数据源：{source}", "库名：" + ("、".join(databases) or "无"), "可选表："]
    if not catalog:
        lines.append("无")
    for name, columns in catalog:
        lines.append(f"{name}：" + ", ".join(columns))
    return "\n".join(lines)


def parse_tables(text: str) -> list[str]:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return []
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(value, dict):
        return []
    tables = value.get("tables") or []
    if not isinstance(tables, list):
        return []
    return [str(item).lower() for item in tables]


def mismatch(gold: set[str], pred: set[str]) -> str:
    if gold == pred:
        return "一致"
    if pred < gold:
        return "少表"
    if gold < pred:
        return "多表"
    if gold & pred:
        return "有对有错"
    return "完全不同"


def summarize(rows: list[dict]) -> dict:
    by_source: dict[str, Counter] = {}
    kinds: Counter = Counter()
    exact = 0
    for row in rows:
        gold = set(row["gold_tables"])
        pred = set(row["pred_tables"])
        kind = mismatch(gold, pred)
        kinds[kind] += 1
        exact += kind == "一致"
        bucket = by_source.setdefault(row["source"], Counter())
        bucket["n"] += 1
        bucket["exact"] += kind == "一致"
    n = len(rows) or 1
    return {
        "n": len(rows),
        "table_exact": exact,
        "table_exact_rate": exact / n,
        "mismatch": dict(kinds),
        "by_source": {key: {"n": value["n"], "exact": value["exact"]} for key, value in by_source.items()},
        "note": "库名用训练集标注。基座 Qwen3.5-4B 从该库的表名单勾选，未加载 Router LoRA。",
    }


def prepare(rows: list[dict], tables: dict[str, str]) -> list[dict]:
    made = []
    for row in rows:
        databases = list(row.get("databases") or [])
        names = candidates_for(tables, databases)
        catalog = [(name, columns_of(name, tables[name])) for name in names]
        made.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "source": row["source"],
            "databases": databases,
            "gold_tables": [item.lower() for item in row.get("tables") or []],
            "catalog": catalog,
            "user": build_user(row["question"], row["source"], databases, catalog),
        })
    return made


def done_ids(path: Path) -> set[int]:
    found = set()
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            found.add(json.loads(line)["question_id"])
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = json.loads(args.data.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    tasks = prepare(rows, load_tables())
    if args.dry_run:
        for task in tasks[:3]:
            print("=" * 20, task["question_id"], task["source"], "tables", len(task["catalog"]))
            print(SYSTEM)
            print(task["user"][:1200])
        lengths = [len(task["user"]) for task in tasks]
        print("n", len(tasks), "user chars p50", sorted(lengths)[len(lengths) // 2], "max", max(lengths))
        return

    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    engine = TransformersEngine(
        args.model,
        model_type="qwen3_5",
        template_type="qwen3_5",
        torch_dtype="bfloat16",
        attn_impl="sdpa",
        use_hf=True,
        max_batch_size=max(1, args.batch_size),
    )
    engine.template.enable_thinking = False
    config = RequestConfig(max_tokens=256, temperature=0)
    args.output.mkdir(parents=True, exist_ok=True)
    pred_path = args.output / "predictions.jsonl"
    finished = done_ids(pred_path)
    pending = [task for task in tasks if task["question_id"] not in finished]
    print(f"待跑 {len(pending)} 已完成 {len(finished)}", flush=True)
    batch = max(1, args.batch_size)
    with pred_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch):
            chunk = pending[start:start + batch]
            responses = engine.infer(
                [InferRequest(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["user"]}]) for task in chunk],
                config,
            )
            for task, response in zip(chunk, responses):
                text = response.choices[0].message.content or ""
                chosen = parse_tables(text)
                allowed = {name for name, _ in task["catalog"]}
                record = {
                    "question_id": task["question_id"],
                    "source": task["source"],
                    "databases": task["databases"],
                    "gold_tables": task["gold_tables"],
                    "pred_tables": chosen,
                    "outside_catalog": [item for item in chosen if item not in allowed],
                    "exact": set(chosen) == set(task["gold_tables"]),
                    "pred": text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done = min(start + batch, len(pending))
            print(f"{done}/{len(pending)}", flush=True)
    saved = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = summarize(saved)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

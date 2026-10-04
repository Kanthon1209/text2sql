#!/usr/bin/env python3
"""跑标注就能支撑的生成实验：E01、E02、E03、E05、E09。

数据是 /data/data/v2-w/train_eknow.json。Router 用其中的 db、database_names、table_names，
不调用已训练的 Router。知识点只用该文件里的 knowledge_ids。
E09 用表卡片代替整段 DDL。相似案例取 similar_top10_scored.json 里与本题同一数据源的前 2 条，排除本题，附上问题和 Golden SQL。
E04、E06–E08、E10–E14 需要 Router 预测或生成器 LoRA，本脚本不跑。

结果写到 /data/k/runs/experiments/<编号>/。中断后重跑会跳过已完成的题目。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/data/k/generator")
from prompt import present_schema, system_for  # noqa: E402
from schema import load_cards, load_tables, lookup, lookup_cards  # noqa: E402

DATA = Path("/data/data/v2-w/train_eknow.json")
KNOWLEDGE = Path("/data/data/v2-w/knowledge.json")
SIMILAR = Path("/data/data/v2-w/similar_top10_scored.json")
OUT = Path("/data/k/runs/experiments")
READY = ("E01", "E02", "E03", "E05", "E09")
WITH_SCHEMA = {"E03", "E05", "E09"}
WITH_KNOWLEDGE = {"E05", "E09"}
CASE_K = 2


def norm(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "")
    text = re.sub(r"^```(?:sql|json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    return re.sub(r"\s+", " ", text.strip().rstrip(";")).lower()


def load_knowledge() -> dict[int, str]:
    rows = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
    return {int(row["knowledge_id"]): row["knowledge_text"] for row in rows}


def bare_index(tables: dict[str, str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for key in tables:
        name = key.split(".", 1)[-1]
        found.setdefault(name, [])
        if key not in found[name]:
            found[name].append(key)
    return found


def resolve(row: dict, tables: dict[str, str], bare: dict[str, list[str]]) -> dict:
    source = row.get("db") or "CROSS"
    databases = [item for item in (row.get("database_names") or [])]
    table_names = [item for item in (row.get("table_names") or [])]
    labeled_empty_db = not databases
    if source == "Elasticsearch":
        return {
            "source": source,
            "databases": databases or ["medical_institutions"],
            "tables": table_names,
            "qualified": ["medical_institutions"],
            "missing": [],
            "ambiguous": [],
            "database_filled": labeled_empty_db,
        }
    allowed = {item.lower() for item in databases}
    qualified: list[str] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for table in table_names:
        name = table.lower().strip('`"')
        candidates = list(bare.get(name, []))
        if name == "vessel_ais_data" and "public.vessel_ais_data" in tables:
            if not allowed or "public" in allowed or source in ("POSTGRESQL", "CROSS"):
                candidates = ["public.vessel_ais_data"]
        elif allowed:
            candidates = [key for key in candidates if key.split(".", 1)[0] in allowed]
        if not candidates:
            missing.append(table)
            continue
        if len(candidates) > 1:
            ambiguous.append(table)
        for key in candidates:
            if key not in qualified:
                qualified.append(key)
    if not databases:
        for key in qualified:
            db_name = key.split(".", 1)[0] if "." in key else key
            if db_name not in databases:
                databases.append(db_name)
    return {
        "source": source,
        "databases": databases,
        "tables": table_names,
        "qualified": qualified,
        "missing": missing,
        "ambiguous": ambiguous,
        "database_filled": labeled_empty_db,
    }


def cross_steps(row: dict, resolved: dict) -> list[dict]:
    by_engine: dict[str, list[str]] = {}
    for key in resolved["qualified"]:
        engine = "POSTGRESQL" if key.startswith("public.") else "MYSQL"
        by_engine.setdefault(engine, []).append(key)
    made = []
    for index, source, gold in ((1, row["db1"], row["query1"]), (2, row["db2"], row["query2"])):
        keys = by_engine.get(source, [])
        databases: list[str] = []
        tables: list[str] = []
        for key in keys:
            db_name, table = key.split(".", 1)
            if db_name not in databases:
                databases.append(db_name)
            tables.append(table)
        note = "这是跨源第一步。查询结果将作为第二步的输入。" if index == 1 else "这是跨源第二步。用 result 代表第一步的执行结果。"
        made.append({
            "step": index,
            "source": source,
            "databases": databases,
            "tables": tables,
            "qualified": keys,
            "gold": gold,
            "note": note,
        })
    return made


def system_text(experiment: str, source: str) -> str:
    if experiment == "E01":
        return "你是查询生成器。根据问题生成一条查询。只输出查询本身，不要解释，不要 Markdown。"
    if experiment == "E02":
        if source == "Elasticsearch":
            return "你是 Elasticsearch 查询生成器。根据问题、数据源、库名和表名生成一个 JSON DSL。只使用 bool、geo_distance、geo_polygon、geo_bounding_box、term、range 这些查询，外层使用 aggs 与 size，或 _source。圆形范围用 geo_distance，矩形范围用 geo_bounding_box，不要改成 geo_shape。只输出 JSON，不要解释，不要 Markdown。"
        if source == "POSTGRESQL":
            return "你是 POSTGRESQL 查询生成器。根据问题、数据源、库名和表名生成一条只读 SQL。表名不要加 public. 前缀，标识符不要加双引号。地理条件使用 geom 和 PostGIS 函数，例如 ST_DWithin。只输出 SQL，不要解释，不要 Markdown。"
        extra = "表名保持数据库前缀。" if source == "SQLite" else ""
        return f"你是 {source} 查询生成器。根据问题、数据源、库名和表名生成一条只读 SQL。{extra}只输出 SQL，不要解释，不要 Markdown。"
    return system_for(source)


def gold_text(row: dict) -> str:
    if row.get("query"):
        return row["query"]
    return "query1:\n" + row.get("query1", "") + "\nquery2:\n" + row.get("query2", "")


def question_source(row: dict) -> str:
    return row.get("db") or "CROSS"


def load_cases(rows: list[dict]) -> dict[int, list[dict]]:
    """每题最多 2 条相似案例。丢掉本题，并且只保留同一数据源。"""
    scored = json.loads(SIMILAR.read_text(encoding="utf-8"))
    by_id = {int(row["question_id"]): row for row in rows}
    found: dict[int, list[dict]] = {}
    for row in rows:
        qid = int(row["question_id"])
        source = question_source(row)
        picked = []
        for item in scored.get(str(qid), []):
            other = int(item["question_id"])
            if other == qid or other not in by_id:
                continue
            neighbor = by_id[other]
            if question_source(neighbor) != source:
                continue
            picked.append({
                "question_id": other,
                "question": neighbor["question"],
                "query": gold_text(neighbor),
            })
            if len(picked) >= CASE_K:
                break
        found[qid] = picked
    return found


def render_cases(cases: list[dict]) -> str:
    if not cases:
        return "相似案例：无"
    lines = ["相似案例："]
    for index, case in enumerate(cases, start=1):
        lines.append(f"示例{index}问题：{case['question']}")
        lines.append(f"示例{index}查询：{case['query']}")
    return "\n".join(lines)


def compose(experiment: str, question: str, route: dict | None, schema: str | None, knowledge: list[str] | None, note: str, cases: list[dict] | None = None) -> str:
    parts = [f"问题：{question}"]
    if route is not None:
        parts.append("数据源：" + route["source"])
        parts.append("库名：" + ("、".join(route["databases"]) or "无"))
        parts.append("表名：" + ("、".join(route["tables"]) or "无"))
    if note:
        parts.append(note)
    if experiment in WITH_SCHEMA:
        shown = present_schema((route or {}).get("source", ""), schema or "")
        title = "表卡片" if experiment == "E09" else "Schema"
        parts.append(f"{title}：\n" + (shown or "无"))
    if experiment in WITH_KNOWLEDGE:
        parts.append("知识：\n" + ("\n".join(knowledge or []) or "无"))
    if experiment == "E09":
        parts.append(render_cases(cases or []))
    return "\n".join(parts)


def tasks_for(experiment: str, row: dict, tables: dict[str, str], bare: dict[str, list[str]], knowledge: dict[int, str], cases: dict[int, list[dict]] | None = None, cards: dict | None = None) -> list[dict]:
    resolved = resolve(row, tables, bare)
    if row.get("db"):
        steps = [{
            "step": 0,
            "source": resolved["source"],
            "databases": resolved["databases"],
            "tables": resolved["tables"],
            "qualified": resolved["qualified"],
            "gold": row["query"],
            "note": "",
        }]
    elif experiment == "E01":
        steps = [{
            "step": 0,
            "source": "CROSS",
            "databases": resolved["databases"],
            "tables": resolved["tables"],
            "qualified": resolved["qualified"],
            "gold": row["query1"].rstrip() + "\n" + row["query2"].rstrip(),
            "note": "",
        }]
    else:
        steps = cross_steps(row, resolved)
    made = []
    for step in steps:
        schema, missing = ("", [])
        if experiment == "E09":
            schema, missing = lookup_cards(cards or {}, step["qualified"])
        elif experiment in WITH_SCHEMA:
            schema, missing = lookup(tables, step["qualified"])
        texts = []
        unknown = []
        if experiment in WITH_KNOWLEDGE:
            for kid in row.get("knowledge_ids") or []:
                if int(kid) in knowledge:
                    texts.append(knowledge[int(kid)])
                else:
                    unknown.append(int(kid))
        examples = []
        if experiment == "E09":
            examples = [
                case for case in (cases or {}).get(int(row["question_id"]), [])
                if case["question_id"] != int(row["question_id"])
            ]
        route = None if experiment == "E01" else {
            "source": step["source"],
            "databases": step["databases"],
            "tables": step["tables"],
        }
        made.append({
            "question_id": row["question_id"],
            "step": step["step"],
            "source": step["source"],
            "gold": step["gold"],
            "missing_tables": list(resolved["missing"]) + list(missing),
            "ambiguous_tables": resolved["ambiguous"],
            "database_filled": resolved["database_filled"],
            "knowledge_ids": list(row.get("knowledge_ids") or []),
            "unknown_knowledge_ids": unknown,
            "messages": [
                {"role": "system", "content": system_text(experiment, step["source"])},
                {"role": "user", "content": compose(experiment, row["question"], route, schema, texts, step["note"], examples)},
            ],
        })
    return made


def done_keys(path: Path) -> set[tuple[int, int]]:
    found = set()
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found.add((row["question_id"], row["step"]))
    return found


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_source: dict[str, Counter] = {}
    questions: dict[int, list[bool]] = {}
    for row in rows:
        bucket = by_source.setdefault(row["source"], Counter())
        bucket["n"] += 1
        bucket["exact"] += int(row["exact"])
        questions.setdefault(row["question_id"], []).append(row["exact"])
    n = len(rows)
    exact = sum(int(row["exact"]) for row in rows)
    full = sum(all(flags) for flags in questions.values())
    return {
        "calls": n,
        "call_exact": exact,
        "call_exact_rate": exact / n if n else 0,
        "questions": len(questions),
        "question_exact": full,
        "question_exact_rate": full / len(questions) if questions else 0,
        "by_source": {key: {"n": value["n"], "exact": value["exact"]} for key, value in by_source.items()},
        "missing_schema_calls": sum(1 for row in rows if row.get("missing_tables")),
        "database_filled_calls": sum(1 for row in rows if row.get("database_filled")),
        "calls_with_knowledge_ids": sum(1 for row in rows if row.get("knowledge_ids")),
    }


def run_one(experiment: str, rows: list[dict], tables, bare, knowledge, cases, client, out: Path, limit: int, cards: dict | None = None) -> dict:
    dest = out / experiment
    dest.mkdir(parents=True, exist_ok=True)
    pred = dest / "predictions.jsonl"
    finished = done_keys(pred)
    chosen = rows[:limit] if limit else rows
    pending = []
    for row in chosen:
        for task in tasks_for(experiment, row, tables, bare, knowledge, cases, cards):
            if (task["question_id"], task["step"]) not in finished:
                pending.append(task)
    print(f"{experiment} 待跑 {len(pending)} 已完成 {len(finished)}", flush=True)
    with pred.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(pending, start=1):
            started = time.time()
            completion = client.chat.completions.create(
                model="Qwen3.5-4B",
                messages=task["messages"],
                temperature=0,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = completion.choices[0].message.content or ""
            record = {
                "experiment": experiment,
                "question_id": task["question_id"],
                "step": task["step"],
                "source": task["source"],
                "exact": norm(text) == norm(task["gold"]),
                "missing_tables": task["missing_tables"],
                "ambiguous_tables": task["ambiguous_tables"],
                "database_filled": task["database_filled"],
                "knowledge_ids": task["knowledge_ids"],
                "unknown_knowledge_ids": task["unknown_knowledge_ids"],
                "gold": task["gold"],
                "pred": text,
                "seconds": round(time.time() - started, 3),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 20 == 0 or index == len(pending):
                print(f"{experiment} {index}/{len(pending)}", flush=True)
    metrics = summarize(pred)
    metrics.update({
        "experiment": experiment,
        "data": str(DATA),
        "note": "字符串是否一致。评测集是带标注的训练题，不是 QA_test.json，也不是执行正确率。E09 的相似案例已排除本题。",
    })
    (dest / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--experiments", default=",".join(READY))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    names = [item.strip() for item in args.experiments.split(",") if item.strip()]
    unknown = [item for item in names if item not in READY]
    if unknown:
        raise SystemExit("本脚本只跑 " + ",".join(READY) + "，不能跑 " + ",".join(unknown))
    rows = json.loads(args.data.read_text(encoding="utf-8"))
    tables = load_tables()
    bare = bare_index(tables)
    knowledge = load_knowledge()
    cases = load_cases(rows) if "E09" in names else {}
    cards = load_cards() if "E09" in names else {}
    if args.dry_run:
        sample = rows[:1] + [row for row in rows if row["question_id"] in (2070, 2491, 2936)]
        for name in names:
            for row in sample:
                for task in tasks_for(name, row, tables, bare, knowledge, cases, cards):
                    print("=" * 20, name, row["question_id"], "step", task["step"])
                    print(task["messages"][0]["content"])
                    print(task["messages"][1]["content"][:800])
        return
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    args.output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"experiments": {}}
    summary["data"] = str(args.data)
    summary["sha256"] = digest
    for name in names:
        summary["experiments"][name] = run_one(name, rows, tables, bare, knowledge, cases, client, args.output, args.limit, cards)
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

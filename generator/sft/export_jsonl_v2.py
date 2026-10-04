#!/usr/bin/env python3
"""第二阶段生成器 LoRA 训练数据：和测试时 v3 的提示词完全一致。

相比第一版 train.jsonl 的改动：
1. 知识点用检索器真实输出：表字段说明 + 相似度 >= 0.65 的 top-3 业务规则，
   替代金标 knowledge_ids 原文。
2. 相似案例改为同库且相似度 >= 0.6 的 top-2，替代同数据源 top-2。
3. 系统提示追加测试时的约束句（规则不进提示词就不使用、ES geo 限制、
   CROSS 两步说明），复用 run_test_full.build_tasks，保证一字不差。
4. 加入失败重写样本：e09 在训练集上的推理里执行失败的步骤改成多轮对话，
   首轮回答不计损失（ms-swift 消息级 loss=False），末轮是金标 SQL。

三元组、表卡片、CROSS 按步拆行、金标目标都和第一版一致。
输出 /data/k/runs/generator-e09-sft2/train.jsonl，不覆盖正在训练用的那份。
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/data/k/experiments")
sys.path.insert(0, "/data/k/generator")

import guard  # noqa: E402
from knowledge import KnowledgeRetriever  # noqa: E402
from run_ready import bare_index, resolve  # noqa: E402
from run_test_full import build_tasks, knowledge_items, pick_cases  # noqa: E402
from schema import load_cards, load_tables  # noqa: E402

TRAIN = Path("/data/data/v2-w/train_eknow.json")
SIMILAR = Path("/data/data/v2-w/similar_top10_scored.json")
E09_PREDS = Path("/data/k/runs/e09/E09/predictions.jsonl")
OUT_DIR = Path("/data/k/runs/generator-e09-sft2")
REPAIR_TEXT = "这条查询不能用。\n{error}\n请重写一条。只使用表卡片中的表和字段。"
WORKERS = 8


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def route_of_row(row: dict, resolved: dict) -> dict:
    source = resolved["source"]
    qualified = resolved["qualified"]
    if source != "CROSS":
        return {
            "source": source,
            "databases": sorted(guard.databases_of(row)),
            "tables": qualified,
            "steps": [],
        }
    pg_keys = [key for key in qualified if key.startswith("public.")]
    mysql_keys = [key for key in qualified if not key.startswith("public.")]
    fallback = [item.lower() for item in (row.get("database_names") or [])]

    def step(db: str) -> dict:
        keys = pg_keys if db == "POSTGRESQL" else mysql_keys
        if keys:
            databases = []
            for key in keys:
                name = key.split(".", 1)[0]
                if name not in databases:
                    databases.append(name)
        else:
            databases = ["public"] if db == "POSTGRESQL" else list(fallback)
        return {"db": db, "databases": databases, "tables": keys}

    steps = [step(row["db1"]), step(row["db2"])]
    union: list[str] = []
    for item in steps:
        for name in item["databases"]:
            if name not in union:
                union.append(name)
    return {"source": "CROSS", "databases": union, "tables": qualified, "steps": steps}


def mine_repair(tables_by_step: dict[tuple[int, int], list[str]]) -> dict[tuple[int, int], tuple[str, str]]:
    """e09 训练集推理里执行失败的步骤 -> (预测原文, 错误信息)。"""
    preds = read_jsonl(E09_PREDS)
    found: dict[tuple[int, int], tuple[str, str]] = {}

    def check(item: dict):
        key = (int(item["question_id"]), int(item.get("step") or 0))
        error = guard.problems(item["source"], key[1], item["pred"], tables_by_step.get(key, []))
        return key, item, error

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for key, item, error in pool.map(check, preds):
            if error:
                found[key] = (item["pred"], error)
    return found


def main() -> None:
    import torch

    rows = json.loads(TRAIN.read_text(encoding="utf-8"))
    tables = load_tables()
    bare = bare_index(tables)
    cards = load_cards()
    retriever = KnowledgeRetriever()
    similar = json.loads(SIMILAR.read_text(encoding="utf-8"))
    by_id = {int(row["question_id"]): row for row in rows}

    routes: dict[int, dict] = {}
    tables_by_step: dict[tuple[int, int], list[str]] = {}
    for row in rows:
        qid = int(row["question_id"])
        resolved = resolve(row, tables, bare)
        route = route_of_row(row, resolved)
        routes[qid] = (row, resolved, route)
        if route["source"] == "CROSS":
            for step in route["steps"]:
                tables_by_step[(qid, route["steps"].index(step) + 1)] = list(step["tables"])
        else:
            tables_by_step[(qid, 0)] = list(route["tables"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"规则检索 {len(rows)} 题 device={device}", flush=True)
    ranked = retriever.rule_notes_many([row["question"] for row in rows], device=device, batch_size=32)

    print("挖掘失败重写样本", flush=True)
    repair = mine_repair(tables_by_step)
    print(f"失败步骤 {len(repair)}", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen3.5-4B", trust_remote_code=True)

    out_rows = []
    case_counts = Counter()
    rule_rows = 0
    table_rows = 0
    repair_used = 0
    repair_too_long = 0
    for index, row in enumerate(rows):
        qid = int(row["question_id"])
        _, resolved, route = routes[qid]
        knowledge = knowledge_items(route, ranked[index], retriever)
        if any(item["kind"] == "rule" for item in knowledge):
            rule_rows += 1
        if any(item["kind"] == "table" for item in knowledge):
            table_rows += 1
        candidates = [
            item for item in similar.get(str(qid), [])
            if int(item["question_id"]) != qid
        ]
        cases = pick_cases(
            candidates,
            row["question"],
            route["source"],
            by_id,
            {name.lower() for name in route["databases"]},
        )
        case_counts[len(cases)] += 1
        assembled = {
            "question_id": qid,
            "question": row["question"],
            "source": route["source"],
            "databases": route["databases"],
            "tables": route["tables"],
            "steps": route["steps"],
            "router_ok": True,
            "knowledge": knowledge,
            "similar_cases": cases,
        }
        for task in build_tasks(assembled, cards):
            if route["source"] == "CROSS":
                gold = row["query1"] if task["step"] == 1 else row["query2"]
            else:
                gold = row["query"]
            key = (int(task["question_id"]), int(task["step"]))
            if key in repair:
                pred_text, error = repair.pop(key)
                first = guard.extract_query(pred_text) or pred_text.strip()
                messages = task["messages"] + [
                    {"role": "assistant", "content": first, "loss": False},
                    {"role": "user", "content": REPAIR_TEXT.format(error=error)},
                    {"role": "assistant", "content": gold},
                ]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False)
                if len(tokenizer.encode(rendered)) <= 4000:
                    out_rows.append({"messages": messages})
                    repair_used += 1
                    continue
                repair_too_long += 1
            out_rows.append({"messages": task["messages"] + [{"role": "assistant", "content": gold}]})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "train.jsonl"
    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in out_rows)
    dest.write_text(text, encoding="utf-8")

    lengths = []
    for item in out_rows:
        rendered = tokenizer.apply_chat_template(item["messages"], tokenize=False)
        lengths.append(len(tokenizer.encode(rendered)))
    lengths.sort()
    manifest = {
        "version": "sft-v2",
        "rows": len(out_rows),
        "repair_rows": repair_used,
        "repair_too_long": repair_too_long,
        "by_source": dict(Counter(row.get("db") or "CROSS" for row in rows)),
        "similar_cases": {str(key): value for key, value in sorted(case_counts.items())},
        "with_rule_knowledge": rule_rows,
        "with_table_knowledge": table_rows,
        "failed_steps_left": len(repair),
        "token_min": lengths[0],
        "token_p50": lengths[len(lengths) // 2],
        "token_p90": lengths[int(len(lengths) * 0.9)],
        "token_p99": lengths[int(len(lengths) * 0.99)],
        "token_max": lengths[-1],
        "over_4096": sum(1 for value in lengths if value > 4096),
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "train": str(TRAIN),
        "e09_preds": str(E09_PREDS),
    }
    (OUT_DIR / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""评测 Router R0。默认评全部训练题，指标是拟合程度，不是留出集。"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path("/data/k/runs/router-r0/train.jsonl")
RUN = Path("/data/k/runs/router-r0")
MODEL = "/data/models/Qwen3.5-4B"


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def same_set(left, right) -> bool:
    return set(left or []) == set(right or [])


def score(gold: dict, pred: dict) -> dict:
    result = {
        "json_ok": bool(pred),
        "source": pred.get("source") == gold.get("source"),
        "databases": same_set(pred.get("databases"), gold.get("databases")),
        "tables": same_set(pred.get("tables"), gold.get("tables")),
        "steps": True,
    }
    if gold.get("source") == "CROSS":
        g_steps = gold.get("steps") or []
        p_steps = pred.get("steps") or []
        result["steps"] = len(g_steps) == len(p_steps) and all(
            a.get("db") == b.get("db") and same_set(a.get("databases"), b.get("databases")) and same_set(a.get("tables"), b.get("tables"))
            for a, b in zip(g_steps, p_steps)
        )
    result["full"] = result["source"] and result["databases"] and result["tables"] and result["steps"]
    return result


def latest_checkpoint(run: Path) -> Path:
    checkpoints = sorted(run.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    if not checkpoints:
        raise SystemExit(f"没有 checkpoint：{run}")
    return checkpoints[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    checkpoint = latest_checkpoint(args.run)
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    engine = TransformersEngine(
        args.model,
        adapters=[str(checkpoint)],
        model_type="qwen3_5",
        template_type="qwen3_5",
        torch_dtype="bfloat16",
        attn_impl="sdpa",
        use_hf=True,
    )
    config = RequestConfig(max_tokens=512, temperature=0)
    totals = Counter()
    by_source = {}
    details = []
    for index, row in enumerate(rows, start=1):
        messages = row["messages"]
        gold = json.loads(messages[-1]["content"])
        response = engine.infer([InferRequest(messages=messages[:-1])], config)[0]
        text = response.choices[0].message.content or ""
        pred = parse_json(text)
        item = score(gold, pred)
        item.update({"source_gold": gold.get("source"), "pred": text})
        details.append(item)
        bucket = by_source.setdefault(gold.get("source"), Counter())
        for key in ("json_ok", "source", "databases", "tables", "steps", "full"):
            totals[key] += int(item[key])
            bucket[key] += int(item[key])
        bucket["n"] += 1
        totals["n"] += 1
        if index % 20 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} full {totals['full']}/{totals['n']}", flush=True)

    n = totals["n"] or 1
    summary = {
        "checkpoint": str(checkpoint),
        "n": totals["n"],
        "note": "评测集就是训练集，表示模型是否学会这 5830 条映射。",
        "source_acc": totals["source"] / n,
        "database_acc": totals["databases"] / n,
        "table_acc": totals["tables"] / n,
        "step_acc": totals["steps"] / n,
        "full_acc": totals["full"] / n,
        "json_ok": totals["json_ok"] / n,
        "by_source": {key: {"n": value["n"], "full": value["full"], "source": value["source"]} for key, value in by_source.items()},
    }
    out = args.output or (args.run / "eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "details.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

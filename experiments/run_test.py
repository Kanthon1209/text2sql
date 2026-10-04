#!/usr/bin/env python3
"""在 QA_test.json 上跑当前不依赖标注的实验。

测试集只有 question_id 和 question。E01 只使用问题，可以跑。
E02、E03 还要数据源、库名、表名，需先用已训练的 Router 对这 400 题推理。
E05 还要知识点，测试集没有 knowledge_ids，需先检索。
没有标准查询，不计算字符串一致率。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/k/experiments")
from run_ready import compose, system_text  # noqa: E402

DATA = Path("/data/raw/contest_8/QA_test.json")
OUT = Path("/data/k/runs/experiments-test-v1")
READY = ("E01",)


def done_ids(path: Path) -> set[int]:
    found = set()
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        found.add(json.loads(line)["question_id"])
    return found


def run_one(name: str, rows: list[dict], client, out: Path) -> dict:
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    pred = dest / "predictions.jsonl"
    finished = done_ids(pred)
    pending = [row for row in rows if row["question_id"] not in finished]
    print(f"{name} 待跑 {len(pending)} 已完成 {len(finished)}", flush=True)
    with pred.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(pending, start=1):
            messages = [
                {"role": "system", "content": system_text(name, "")},
                {"role": "user", "content": compose(name, row["question"], None, None, None, "")},
            ]
            started = time.time()
            completion = client.chat.completions.create(
                model="Qwen3.5-4B",
                messages=messages,
                temperature=0,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = completion.choices[0].message.content or ""
            record = {
                "experiment": name,
                "question_id": row["question_id"],
                "question": row["question"],
                "pred": text,
                "seconds": round(time.time() - started, 3),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 20 == 0 or index == len(pending):
                print(f"{name} {index}/{len(pending)}", flush=True)
    n = sum(1 for line in pred.read_text(encoding="utf-8").splitlines() if line.strip())
    metrics = {
        "experiment": name,
        "questions": n,
        "data": str(DATA),
        "note": "测试集没有标准查询，只保存生成结果，没有字符串一致率。",
    }
    (dest / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--experiments", default=",".join(READY))
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    args = parser.parse_args()
    names = [item.strip() for item in args.experiments.split(",") if item.strip()]
    unknown = [item for item in names if item not in READY]
    if unknown:
        raise SystemExit("测试集目前只能跑 " + ",".join(READY) + "，不能跑 " + ",".join(unknown))
    rows = json.loads(args.data.read_text(encoding="utf-8"))
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "data": str(args.data),
        "sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "questions": len(rows),
        "experiments": {},
    }
    for name in names:
        summary["experiments"][name] = run_one(name, rows, client, args.output)
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

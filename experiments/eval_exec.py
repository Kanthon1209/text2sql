#!/usr/bin/env python3
"""用 verify_sql 判断实验生成语句和 Golden 语句的执行结果是否一致。

跨库题按联邦查询整题判定：第一段的结果填进第二段的 (result) 后再比较。
单库题一条预测对一条 Golden。结果写到每个实验目录的 exec_accuracy.json。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

EXPERIMENTS = Path("/data/k/runs/experiments")
TRAIN = Path("/data/data/v2-w/train_eknow.json")
SOURCE_DB = {
    "MYSQL": "mysql",
    "POSTGRESQL": "postgresql",
    "SQLite": "sqlite",
    "Elasticsearch": "elasticsearch",
    "CROSS": "federated",
}


def extract_query(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.I)
    text = text.strip()
    fences = re.findall(r"```(?:sql|json)?\s*([\s\S]*?)```", text, flags=re.I)
    if fences:
        text = "\n".join(part.strip() for part in fences if part.strip())
    return text.strip()


def load_train() -> dict[int, dict]:
    rows = json.loads(TRAIN.read_text(encoding="utf-8"))
    return {int(row["question_id"]): row for row in rows}


def load_predictions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def federated_from_steps(steps: list[dict]) -> tuple[dict, dict, bool]:
    steps = sorted(steps, key=lambda item: item["step"])
    first, second = steps[0], steps[1]
    gold = {
        "db1": first["source"],
        "query1": first["gold"],
        "db2": second["source"],
        "query2": second["gold"],
    }
    pred = {
        "db1": first["source"],
        "query1": extract_query(first["pred"]),
        "db2": second["source"],
        "query2": extract_query(second["pred"]),
    }
    exact = bool(first["exact"] and second["exact"])
    return pred, gold, exact


def federated_from_cross(row: dict, train_row: dict) -> tuple[dict, dict, bool]:
    query1, _, query2 = row["gold"].partition("\n")
    gold = {
        "db1": train_row.get("db1") or "MYSQL",
        "query1": query1,
        "db2": train_row.get("db2") or "POSTGRESQL",
        "query2": query2,
    }
    pred_text = extract_query(row["pred"])
    parsed = None
    if pred_text.startswith("{"):
        try:
            parsed = json.loads(pred_text)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict) and "query1" in parsed and "query2" in parsed:
        pred = {
            "db1": parsed.get("db1") or gold["db1"],
            "query1": parsed["query1"],
            "db2": parsed.get("db2") or gold["db2"],
            "query2": parsed["query2"],
        }
    else:
        parts = [part.strip() for part in re.split(r"\n\s*\n", pred_text) if part.strip()]
        if len(parts) >= 2 and re.search(r"\(result\)", parts[-1], re.I):
            pred = {
                "db1": gold["db1"],
                "query1": parts[0],
                "db2": gold["db2"],
                "query2": parts[-1],
            }
        else:
            pred = pred_text
    return pred, gold, bool(row["exact"])


def jobs_for(rows: list[dict], train: dict[int, dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["question_id"])].append(row)
    jobs = []
    for question_id, items in grouped.items():
        items.sort(key=lambda item: item["step"])
        if len(items) == 1 and items[0]["source"] != "CROSS":
            row = items[0]
            jobs.append({
                "question_id": question_id,
                "source": row["source"],
                "db_type": SOURCE_DB[row["source"]],
                "pred": extract_query(row["pred"]),
                "gold": row["gold"],
                "string_exact": bool(row["exact"]),
            })
            continue
        if len(items) == 1 and items[0]["source"] == "CROSS":
            pred, gold, exact = federated_from_cross(items[0], train[question_id])
            source = "CROSS"
        else:
            pred, gold, exact = federated_from_steps(items)
            source = "CROSS"
        jobs.append({
            "question_id": question_id,
            "source": source,
            "db_type": "federated",
            "pred": pred,
            "gold": gold,
            "string_exact": exact,
        })
    jobs.sort(key=lambda item: item["question_id"])
    return jobs


def _init_worker() -> None:
    os.environ["VERIFY_SQL_LOG"] = "/dev/null"
    import sys
    sys.path.insert(0, "/data/lj")
    from verify_sql import set_log_file
    set_log_file("/dev/null")


def _eval_job(job: dict) -> dict:
    import sys
    sys.path.insert(0, "/data/lj")
    from verify_sql import check_detail

    started = time.time()
    detail = check_detail(
        job["db_type"],
        job["pred"],
        job["gold"],
        column_order_sensitive=True,
        treat_both_empty_as_equal=True,
        timeout=15,
    )
    both_empty = bool(detail["equal"] and detail["gold_rows"] == 0 and detail["pred_rows"] == 0)
    return {
        "question_id": job["question_id"],
        "source": job["source"],
        "string_exact": job["string_exact"],
        "exec_equal": bool(detail["equal"]),
        "ok": bool(detail["ok"]),
        "gold_rows": detail["gold_rows"],
        "pred_rows": detail["pred_rows"],
        "both_empty": both_empty,
        "error": detail["error"],
        "seconds": round(time.time() - started, 3),
    }


def _rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def summarize(experiment: str, records: list[dict], elapsed: float) -> dict:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)

    def pack(name: str, group: list[dict]) -> dict:
        total = len(group)
        correct = sum(1 for item in group if item["exec_equal"])
        exact = sum(1 for item in group if item["string_exact"])
        gained = sum(1 for item in group if item["exec_equal"] and not item["string_exact"])
        exact_fail = sum(1 for item in group if item["string_exact"] and not item["exec_equal"])
        errors = sum(1 for item in group if not item["ok"])
        both_empty = sum(1 for item in group if item["both_empty"])
        return {
            "数据源": name,
            "题目数": total,
            "执行正确数": correct,
            "执行正确率": _rate(correct, total),
            "字符串一致数": exact,
            "字符串一致率": _rate(exact, total),
            "字符串不同但执行正确": gained,
            "字符串一致但执行不一致": exact_fail,
            "执行失败数": errors,
            "双方结果为空且记为正确": both_empty,
        }

    total = pack("全部", records)
    sources = [pack(name, by_source[name]) for name in ("MYSQL", "POSTGRESQL", "SQLite", "Elasticsearch", "CROSS") if name in by_source]
    return {
        "实验": experiment,
        "判定": "verify_sql.check_detail：在真实库执行预测语句与 Golden 语句，结果集一致记为正确。跨库题按联邦查询整题判定。",
        "参数": {
            "column_order_sensitive": True,
            "treat_both_empty_as_equal": True,
            "timeout_seconds": 15,
        },
        "题目数": total["题目数"],
        "执行正确数": total["执行正确数"],
        "执行正确率": total["执行正确率"],
        "字符串一致数": total["字符串一致数"],
        "字符串一致率": total["字符串一致率"],
        "字符串不同但执行正确": total["字符串不同但执行正确"],
        "字符串一致但执行不一致": total["字符串一致但执行不一致"],
        "执行失败数": total["执行失败数"],
        "双方结果为空且记为正确": total["双方结果为空且记为正确"],
        "去掉空结果后的执行正确数": total["执行正确数"] - total["双方结果为空且记为正确"],
        "去掉空结果后的执行正确率": _rate(total["执行正确数"] - total["双方结果为空且记为正确"], total["题目数"]),
        "分数据源": sources,
        "耗时秒": round(elapsed, 1),
        "明细": "exec_details.jsonl",
    }


def done_ids(path: Path) -> dict[int, dict]:
    found = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[int(row["question_id"])] = row
    return found


def run_experiment(name: str, train: dict[int, dict], workers: int, limit: int) -> dict:
    dest = EXPERIMENTS / name
    predictions = load_predictions(dest / "predictions.jsonl")
    jobs = jobs_for(predictions, train)
    if limit:
        jobs = jobs[:limit]
    detail_path = dest / ("exec_details_limit.jsonl" if limit else "exec_details.jsonl")
    finished = {} if limit else done_ids(detail_path)
    pending = [job for job in jobs if job["question_id"] not in finished]
    print(f"{name} 题目 {len(jobs)} 待判定 {len(pending)} 已完成 {len(finished)}", flush=True)
    started = time.time()
    mode = "a" if finished else "w"
    with detail_path.open(mode, encoding="utf-8") as handle:
        if pending:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
                futures = [pool.submit(_eval_job, job) for job in pending]
                done = 0
                for future in as_completed(futures):
                    record = future.result()
                    finished[record["question_id"]] = record
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    done += 1
                    if done % 200 == 0 or done == len(pending):
                        correct = sum(1 for item in finished.values() if item["exec_equal"])
                        elapsed = time.time() - started
                        print(
                            f"{name} {done}/{len(pending)} 目前正确 {correct}/{len(finished)} "
                            f"用时 {elapsed:.0f}s",
                            flush=True,
                        )
    records = [finished[job["question_id"]] for job in jobs]
    elapsed = time.time() - started
    summary = summarize(name, records, elapsed)
    if not limit:
        (dest / "exec_accuracy.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({k: summary[k] for k in ("实验", "题目数", "执行正确数", "执行正确率", "字符串一致数", "字符串不同但执行正确")}, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default="E01,E02,E03,E05")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    names = [item.strip() for item in args.experiments.split(",") if item.strip()]
    train = load_train()
    summaries = []
    for name in names:
        summaries.append(run_experiment(name, train, args.workers, args.limit))
    if not args.limit:
        (EXPERIMENTS / "exec_summary.json").write_text(
            json.dumps({"experiments": summaries}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

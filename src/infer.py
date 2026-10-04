#!/usr/bin/env python3
"""对 QA_train.json 前 N 条做 oracle 模式批量推理，并和 gold query 做轻量比对。

对每个模型调用常驻 OpenAI 兼容服务（ms-swift deploy），并发请求。
输出：
  /data/k/log/result_<MODEL>_<TS>.jsonl   每条一行：prompt/response/parsed/gold/latency
  /data/k/log/summary_<TS>.json           两模型汇总指标
  /data/k/log/infer_<TS>.log              完整运行日志
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/data/k")
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI  # noqa: E402

from prompt_builder import get_store, build_prompt  # noqa: E402

TRAIN = Path("/data/raw/contest_8/QA_train.json")


def split_think(text: str) -> str:
    """去掉 Qwen3.5 的 <think>...</think> 或 ħ...ȗ 包裹，只留最终答案。"""
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    if "ȗ" in text:
        parts = re.split(r"ȗ", text, maxsplit=1, flags=re.I)
        text = parts[-1].lstrip("\n").strip()
    return text.strip()


def extract_sql(text: str) -> str:
    text = split_think(text).strip()
    m = re.search(r"```sql\s*([\s\S]*?)```", text, flags=re.I)
    if m:
        return m.group(1).strip().rstrip(";").strip()
    m = re.search(r"```\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip().rstrip(";").strip()
    for key in ("select", "with", "put", "post", "{"):
        idx = text.lower().find(key)
        if idx >= 0:
            return text[idx:].strip().rstrip(";").strip()
    return text.rstrip(";").strip()


def norm_sql(s: str) -> str:
    """归一化 SQL 用于宽松比对：去注释/空白/大小写/尾分号。"""
    if not s:
        return ""
    s = re.sub(r"--[^\n]*", "", s)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.rstrip(";").strip()
    return s


def exact_match(pred: str, gold: str) -> bool:
    return norm_sql(pred) == norm_sql(gold)


def call_model(client: OpenAI, model: str, prompt: str, max_tokens: int, extra_body: dict) -> dict:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.8,
        stream=False,
        extra_body=extra_body,
    )
    dt = time.time() - t0
    choice = resp.choices[0]
    content = choice.message.content or ""
    usage = resp.usage
    return {
        "raw_text": content,
        "finish_reason": choice.finish_reason,
        "usage": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens} if usage else None,
        "latency_s": round(dt, 3),
    }


def run_model(name: str, base_url: str, served: str, items: list[dict],
              store, max_tokens: int, workers: int, extra_body: dict, logf) -> list[dict]:
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=300.0)
    results: list[dict] = [None] * len(items)

    def work(i: int):
        item = items[i]
        prompt, dialect = build_prompt(store, item)
        try:
            out = call_model(client, served, prompt, max_tokens, extra_body)
            parsed = extract_sql(out["raw_text"])
            em = exact_match(parsed, item["query"])
            return {
                "idx": i,
                "question_id": item["question_id"],
                "question": item["question"],
                "db": item["db"],
                "dialect": dialect,
                "gold_query": item["query"],
                "prompt": prompt,
                "raw_text": out["raw_text"],
                "parsed_sql": parsed,
                "exact_match": em,
                "finish_reason": out["finish_reason"],
                "usage": out["usage"],
                "latency_s": out["latency_s"],
            }
        except Exception as e:
            print(f"[{name}] idx={i} qid={item.get('question_id')} ERROR: {e}", file=logf, flush=True)
            return {
                "idx": i,
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "db": item.get("db"),
                "gold_query": item.get("query"),
                "prompt": prompt if "prompt" in locals() else "",
                "raw_text": "",
                "parsed_sql": "",
                "exact_match": False,
                "error": str(e),
                "finish_reason": None,
                "usage": None,
                "latency_s": 0.0,
            }

    print(f"[{name}] 开始推理 {len(items)} 条，workers={workers}", file=logf, flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in range(len(items))}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            results[r["idx"]] = r
            done += 1
            if done % 10 == 0 or done == len(items):
                print(f"[{name}] 进度 {done}/{len(items)}", file=logf, flush=True)
    dt = time.time() - t0
    print(f"[{name}] 完成，总耗时 {dt:.1f}s", file=logf, flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--xiyan-port", type=int, default=8001)
    ap.add_argument("--qwen-port", type=int, default=8002)
    ap.add_argument("--ts", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--skip-serve-check", action="store_true")
    args = ap.parse_args()

    LOG = ROOT / "log"
    LOG.mkdir(parents=True, exist_ok=True)
    log_path = LOG / f"infer_{args.ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        print(msg, file=logf, flush=True)
        print(msg)

    log(f"=== 推理运行 ts={args.ts} limit={args.limit} workers={args.workers} ===")

    # 就绪检查
    if not args.skip_serve_check:
        import urllib.request
        for name, port in (("XiYanSQL-3B", args.xiyan_port), ("Qwen3.5-4B", args.qwen_port)):
            url = f"http://127.0.0.1:{port}/v1/models"
            try:
                urllib.request.urlopen(url, timeout=5)
                log(f"[ready] {name} :{port} OK")
            except Exception as e:
                log(f"[ready] {name} :{port} 未就绪: {e}")
                logf.close()
                sys.exit(1)

    # 数据
    data = json.load(open(TRAIN, encoding="utf-8"))
    items = data[: args.limit]
    log(f"载入 QA_train 前 {len(items)} 条")

    store = get_store()
    log(f"schema store 加载完成，units={len(store.units)}")

    # 两个模型并发跑（各自独立线程池）
    jobs = [
        {
            "name": "XiYanSQL-3B",
            "base_url": f"http://127.0.0.1:{args.xiyan_port}/v1",
            "served": "XiYanSQL-3B",
            "extra_body": {},
        },
        {
            "name": "Qwen3.5-4B",
            "base_url": f"http://127.0.0.1:{args.qwen_port}/v1",
            "served": "Qwen3.5-4B",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    ]

    all_results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {
            ex.submit(run_model, j["name"], j["base_url"], j["served"], items, store,
                      args.max_tokens, args.workers, j["extra_body"], logf): j["name"]
            for j in jobs
        }
        for f in as_completed(futs):
            name = futs[f]
            all_results[name] = f.result()

    # 写每模型明细
    for name, rs in all_results.items():
        out = LOG / f"result_{name}_{args.ts}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"已写 {out}")

    # 汇总
    summary = {"ts": args.ts, "limit": len(items), "models": {}}
    for name, rs in all_results.items():
        ok = sum(1 for r in rs if r.get("exact_match"))
        err = sum(1 for r in rs if r.get("error"))
        lat = [r["latency_s"] for r in rs if r.get("latency_s")]
        summary["models"][name] = {
            "exact_match": ok,
            "exact_match_rate": round(ok / len(rs), 4),
            "errors": err,
            "avg_latency_s": round(sum(lat) / len(lat), 3) if lat else 0,
            "max_latency_s": round(max(lat), 3) if lat else 0,
        }
    sum_path = LOG / f"summary_{args.ts}.json"
    json.dump(summary, sum_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"已写汇总 {sum_path}")
    log(json.dumps(summary, ensure_ascii=False, indent=2))

    # 打印前 3 条对比
    log("\n=== 前 3 条样例对比 ===")
    for i in range(min(3, len(items))):
        log(f"\n--- #{i} qid={items[i]['question_id']} db={items[i]['db']} ---")
        log(f"Q: {items[i]['question']}")
        log(f"GOLD: {items[i]['query']}")
        for name in all_results:
            r = all_results[name][i]
            log(f"[{name}] EM={r['exact_match']}")
            log(f"  RAW: {r['raw_text'][:300]!r}")
            log(f"  PARSED: {r['parsed_sql'][:200]}")

    logf.close()
    print(f"\n完成。日志: {log_path}")


if __name__ == "__main__":
    main()

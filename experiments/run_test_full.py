#!/usr/bin/env python3
"""测试集完全体，无训练生成器。

1. route：Router R0 给 QA_test.json 补数据源、库名、表名。
2. retrieve：相似案例用训练题向量检索；知识点规则用 KnowledgeRetriever。
3. assemble：写成 /data/k/runs/test-full-v1/QA_test.json。
4. infer：按 E09 的表卡片提示词，调用未微调的 Qwen3.5-4B。
5. merge：合并两张卡的预测。

测试集没有标准查询，不计算字符串一致率。
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

sys.path.insert(0, "/data/k/experiments")
sys.path.insert(0, "/data/k/generator")

import guard  # noqa: E402
from knowledge import KnowledgeRetriever  # noqa: E402
from run_ready import compose, gold_text, question_source, system_text  # noqa: E402
from schema import load_cards, lookup_cards  # noqa: E402

TEST = Path("/data/raw/contest_8/QA_test.json")
TRAIN = Path("/data/data/v2-w/train_eknow.json")
OUT = Path("/data/k/runs/test-full-v1")
ROUTER_RUN = Path("/data/k/runs/router-r0")
MODEL = "/data/models/Qwen3.5-4B"
EMBED = Path("/data/models/Qwen3-Embedding-0.6B")
CASE_K = 2
SIMILAR_K = 10
SOURCES = {"MYSQL", "SQLite", "POSTGRESQL", "Elasticsearch", "CROSS"}
STEP_DB = {"MYSQL", "SQLite", "POSTGRESQL", "Elasticsearch"}
SIMILAR_TASK = "根据自然语言问题，检索语义最相近的训练问题"
ROUTER_SYSTEM = (
    "你是数据源路由器。根据问题判断数据源、数据库和数据表。"
    "只输出一个 JSON 对象，不要 SQL，不要解释。\n"
    '单源格式：{"source":"MYSQL","databases":["库名"],"tables":["库名.表名"]}\n'
    '跨源格式：{"source":"CROSS","databases":["库名"],"tables":["库名.表名"],'
    '"steps":[{"db":"MYSQL","databases":["库名"],"tables":["库名.表名"]}]}\n'
    "source 只能是 MYSQL、SQLite、POSTGRESQL、Elasticsearch、CROSS。"
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def latest_checkpoint(run: Path) -> Path:
    checkpoints = [
        path for path in run.rglob("checkpoint-*")
        if path.is_dir() and (path / "adapter_model.safetensors").exists()
    ]
    if not checkpoints:
        raise SystemExit(f"没有 checkpoint：{run}")
    return max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))


def as_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    found = []
    for item in value:
        text = str(item).strip()
        if text and text not in found:
            found.append(text)
    return found


def normalize_route(pred: dict) -> dict | None:
    source = pred.get("source")
    if source not in SOURCES:
        return None
    databases = as_list(pred.get("databases"))
    tables = as_list(pred.get("tables"))
    steps = []
    if source == "CROSS":
        for step in pred.get("steps") or []:
            if not isinstance(step, dict) or step.get("db") not in STEP_DB:
                continue
            steps.append({
                "db": step["db"],
                "databases": as_list(step.get("databases")),
                "tables": as_list(step.get("tables")),
            })
        if not steps:
            return None
    if source == "Elasticsearch":
        databases = databases or ["medical_institutions"]
        tables = tables or ["medical_institutions"]
    return {"source": source, "databases": databases, "tables": tables, "steps": steps}


def load_test(path: Path = TEST) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def route(out: Path, batch_size: int) -> None:
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    out.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_checkpoint(ROUTER_RUN)
    rows = load_test()
    dest = out / "router.jsonl"
    done = {int(row["question_id"]): row for row in read_jsonl(dest)}
    pending = [row for row in rows if int(row["question_id"]) not in done]
    print(f"router 待跑 {len(pending)} 已完成 {len(done)} checkpoint {checkpoint}", flush=True)
    if pending:
        engine = TransformersEngine(
            MODEL,
            adapters=[str(checkpoint)],
            model_type="qwen3_5",
            template_type="qwen3_5",
            torch_dtype="bfloat16",
            attn_impl="sdpa",
            use_hf=True,
            max_batch_size=batch_size,
        )
        engine.template.enable_thinking = False
        config = RequestConfig(max_tokens=512, temperature=0)
        with dest.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), batch_size):
                chunk = pending[start:start + batch_size]
                requests = [
                    InferRequest(messages=[
                        {"role": "system", "content": ROUTER_SYSTEM},
                        {"role": "user", "content": f"问题：{row['question']}"},
                    ])
                    for row in chunk
                ]
                responses = engine.infer(requests, config)
                for row, response in zip(chunk, responses):
                    text = response.choices[0].message.content or ""
                    pred = normalize_route(parse_json(text))
                    record = {
                        "question_id": row["question_id"],
                        "raw": text,
                        "ok": pred is not None,
                        "route": pred,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done[int(row["question_id"])] = record
                handle.flush()
                print(f"router {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)
    ok = sum(1 for row in done.values() if row.get("ok"))
    meta = {"checkpoint": str(checkpoint), "questions": len(rows), "router_ok": ok}
    (out / "router_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)


def _embed(texts: list[str], tokenizer, model, device: str, batch_size: int = 64):
    import torch
    import torch.nn.functional as F

    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"]
            if mask[:, -1].sum() == mask.shape[0]:
                pooled = hidden[:, -1]
            else:
                lengths = mask.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
            pooled = F.normalize(pooled, p=2, dim=1)
        vectors.append(pooled)
        done = min(start + batch_size, len(texts))
        if done == len(texts) or start % (batch_size * 20) == 0:
            print(f"encoded {done}/{len(texts)}", flush=True)
    return torch.cat(vectors, dim=0)


def retrieve(out: Path) -> None:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_rows = load_test()
    train_rows = json.loads(TRAIN.read_text(encoding="utf-8"))
    train_questions = [row["question"] for row in train_rows]
    train_ids = [int(row["question_id"]) for row in train_rows]
    digest = hashlib.sha256("\n".join(train_questions).encode()).hexdigest()
    print("retrieve device", device, "train", len(train_rows), "test", len(test_rows), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(EMBED, padding_side="left", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMBED,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()
    cache = out / "train_question_vectors.npz"
    docs = None
    if cache.exists():
        stored = np.load(cache)
        if str(stored["digest"]) == digest and stored["vectors"].shape[0] == len(train_questions):
            docs = torch.from_numpy(stored["vectors"]).to(device)
            print("loaded cached train vectors", tuple(docs.shape), flush=True)
    if docs is None:
        print("embed train questions", flush=True)
        docs = _embed(train_questions, tokenizer, model, device)
        np.savez(cache, vectors=docs.detach().cpu().numpy(), digest=np.array(digest))
    print("embed test similar queries", flush=True)
    queries = _embed(
        [f"Instruct: {SIMILAR_TASK}\nQuery:{row['question']}" for row in test_rows],
        tokenizer,
        model,
        device,
    )
    scores = (queries @ docs.T).float()
    values, index = torch.topk(scores, k=SIMILAR_K, dim=1)
    similar = {}
    for row, row_index, row_score in zip(test_rows, index.cpu(), values.cpu()):
        similar[str(row["question_id"])] = [
            {"question_id": train_ids[int(item)], "score": round(float(score), 6)}
            for item, score in zip(row_index, row_score)
        ]
    (out / "similar_top10.json").write_text(json.dumps(similar, ensure_ascii=False) + "\n", encoding="utf-8")
    del model, docs, queries, scores
    if device == "cuda":
        torch.cuda.empty_cache()

    retriever = KnowledgeRetriever()
    print("retrieve knowledge rules", flush=True)
    ranked = retriever.rule_notes_many([row["question"] for row in test_rows], device=device, batch_size=32)
    rules = {
        str(row["question_id"]): [
            {"knowledge_id": item["knowledge_id"], "text": item["text"], "score": item["score"]}
            for item in picked
        ]
        for row, picked in zip(test_rows, ranked)
    }
    (out / "knowledge_rules.json").write_text(json.dumps(rules, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote similar {len(similar)} rules {len(rules)}", flush=True)


def pick_cases(candidates: list[dict], question: str, source: str, by_id: dict[int, dict], databases: set[str]) -> list[dict]:
    picked = []
    for item in candidates:
        if float(item.get("score") or 0) < guard.CASE_MIN_SCORE:
            continue
        neighbor = by_id.get(int(item["question_id"]))
        if neighbor is None or neighbor["question"] == question:
            continue
        if question_source(neighbor) != source:
            continue
        neighbor_dbs = guard.databases_of(neighbor)
        if databases and neighbor_dbs and databases.isdisjoint(neighbor_dbs):
            continue
        if source == "SQLite" and not neighbor_dbs:
            continue
        picked.append({
            "question_id": int(neighbor["question_id"]),
            "question": neighbor["question"],
            "query": gold_text(neighbor),
            "score": item["score"],
            "source": question_source(neighbor),
        })
        if len(picked) >= CASE_K:
            break
    return picked


def knowledge_items(route: dict | None, rules: list[dict], retriever: KnowledgeRetriever) -> list[dict]:
    items = []
    if route:
        for note in retriever.table_notes(route["tables"], route["databases"]):
            items.append({
                "knowledge_id": note["knowledge_id"],
                "text": note["text"],
                "kind": "table",
                "database": note.get("database"),
                "table": note.get("table"),
            })
    for rule in rules:
        if not guard.keep_rule(rule):
            continue
        items.append({
            "knowledge_id": rule["knowledge_id"],
            "text": rule["text"],
            "kind": "rule",
            "score": rule.get("score"),
        })
    return items


def texts_for_step(items: list[dict], tables: list[str], cross: bool) -> list[str]:
    if not cross:
        return [item["text"] for item in items]
    wanted = {table.lower() for table in tables}
    table_texts = []
    rules = []
    for item in items:
        if item.get("kind") == "rule":
            rules.append(item["text"])
            continue
        key = f"{item.get('database') or ''}.{item.get('table') or ''}".lower()
        bare = str(item.get("table") or "").lower()
        if key in wanted or bare in wanted:
            table_texts.append(item["text"])
    return table_texts + rules


def route_of(row: dict) -> dict | None:
    nested = row.get("route")
    if isinstance(nested, dict) and nested.get("source"):
        return nested
    if not row.get("source"):
        return None
    return {
        "source": row["source"],
        "databases": list(row.get("databases") or []),
        "tables": list(row.get("tables") or []),
        "steps": list(row.get("steps") or []),
    }


def build_tasks(row: dict, cards: dict) -> list[dict]:
    route = route_of(row)
    if not route:
        return []
    source = route["source"]
    cases = [
        {"question": item["question"], "query": item["query"]}
        for item in row.get("similar_cases") or []
    ]
    if source == "CROSS":
        steps = []
        for index, step in enumerate(route["steps"], start=1):
            note = "这是跨源第一步。查询结果将作为第二步的输入。" if index == 1 else "这是跨源第二步。用 result 代表第一步的执行结果。"
            steps.append((index, step, note))
    else:
        steps = [(0, {"db": source, "databases": route["databases"], "tables": route["tables"]}, "")]
    tasks = []
    for step_id, step, note in steps:
        tables = list(step.get("tables") or [])
        schema, missing = lookup_cards(cards, tables)
        texts = texts_for_step(row.get("knowledge") or [], tables, source == "CROSS")
        route_step = {"source": step["db"], "databases": list(step.get("databases") or []), "tables": tables}
        system = system_text("E09", step["db"])
        system += "提示词里没有出现的业务规则不要使用，不要据此添加过滤或分段。"
        if step["db"] == "Elasticsearch":
            system += "不要使用 geo_shape、match、match_phrase。"
        if source == "CROSS" and step_id == 1:
            system += "本步只查询当前数据源的表，不要写另一个数据源的函数或表。"
        if source == "CROSS" and step_id == 2:
            system += "本步查询必须包含 result，不要引用第一步的库表。"
        tasks.append({
            "question_id": row["question_id"],
            "step": step_id,
            "source": step["db"],
            "databases": route_step["databases"],
            "tables": tables,
            "missing_tables": missing,
            "knowledge_ids": [item["knowledge_id"] for item in row.get("knowledge") or []],
            "similar_question_ids": [item["question_id"] for item in row.get("similar_cases") or []],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": compose("E09", row["question"], route_step, schema, texts, note, cases)},
            ],
        })
    return tasks


def assemble(out: Path) -> None:
    rows = load_test()
    routed = {int(row["question_id"]): row for row in read_jsonl(out / "router.jsonl")}
    similar = json.loads((out / "similar_top10.json").read_text(encoding="utf-8"))
    rules = json.loads((out / "knowledge_rules.json").read_text(encoding="utf-8"))
    train = {int(row["question_id"]): row for row in json.loads(TRAIN.read_text(encoding="utf-8"))}
    retriever = KnowledgeRetriever()
    cards = load_cards()
    enriched = []
    case_counts = Counter()
    by_source = Counter()
    for row in rows:
        qid = int(row["question_id"])
        found = routed.get(qid) or {}
        route = found.get("route") if found.get("ok") else None
        item_rules = rules.get(str(qid), [])
        knowledge = knowledge_items(route, item_rules, retriever)
        route_dbs = {item.lower() for item in (route.get("databases") or [])} if route else set()
        cases = pick_cases(similar.get(str(qid), []), row["question"], route["source"], train, route_dbs) if route else []
        case_counts[len(cases)] += 1
        if route:
            by_source[route["source"]] += 1
        else:
            by_source["FAILED"] += 1
        enriched.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "source": route["source"] if route else None,
            "databases": route["databases"] if route else [],
            "tables": route["tables"] if route else [],
            "steps": route["steps"] if route else [],
            "router_ok": bool(route),
            "knowledge": knowledge,
            "similar_cases": cases,
        })
    if by_source["FAILED"] == len(rows):
        raise SystemExit("Router 没有解析出任何题目")
    dest = out / "QA_test.json"
    dest.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sample = next(row for row in enriched if row["router_ok"])
    task = build_tasks(sample, cards)[0]
    prompt = task["messages"][0]["content"] + "\n\n" + task["messages"][1]["content"]
    for title in ("问题：", "数据源：", "表卡片：", "知识：", "相似案例："):
        if title not in prompt:
            raise SystemExit(f"提示词缺少 {title}")
    (out / "sample_prompt.txt").write_text(prompt, encoding="utf-8")
    meta_path = out / "router_meta.json"
    router_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    manifest = {
        "source": str(TEST),
        "sha256": hashlib.sha256(TEST.read_bytes()).hexdigest(),
        "output": str(dest),
        "questions": len(enriched),
        "router_checkpoint": router_meta.get("checkpoint"),
        "router_ok": len(enriched) - by_source["FAILED"],
        "by_source": dict(by_source),
        "similar_cases": {str(key): value for key, value in sorted(case_counts.items())},
        "with_table_knowledge": sum(1 for row in enriched if any(item["kind"] == "table" for item in row["knowledge"])),
        "with_rule_knowledge": sum(1 for row in enriched if any(item["kind"] == "rule" for item in row["knowledge"])),
        "prompt": "E09 表卡片。业务规则只保留相似度 >= 0.65 的条目。相似案例必须同一库且相似度 >= 0.6。生成后执行检查，失败重写一次。生成器不加 LoRA。",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def _complete(messages: list[dict], client, engine, config) -> str:
    if engine is not None:
        from swift.infer_engine import InferRequest

        response = engine.infer([InferRequest(messages=messages)], config)[0]
        return response.choices[0].message.content or ""
    completion = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model="Qwen3.5-4B",
                messages=messages,
                temperature=0,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            break
        except Exception as exc:
            print(f"api retry {attempt + 1} {exc}", flush=True)
            if attempt == 2:
                raise
            time.sleep(3)
    return completion.choices[0].message.content or ""


def _generate(task: dict, client, engine, config) -> tuple[str, str, bool]:
    text = _complete(task["messages"], client, engine, config)
    error = guard.problems(task["source"], task["step"], text, task["tables"])
    if not error:
        return text, "", False
    repair = list(task["messages"]) + [
        {"role": "assistant", "content": text},
        {"role": "user", "content": "这条查询不能用。\n" + error + "\n请重写一条。只使用表卡片中的表和字段。"},
    ]
    second = _complete(repair, client, engine, config)
    second_error = guard.problems(task["source"], task["step"], second, task["tables"])
    if not second_error:
        return second, error, True
    return text, error, True


def infer(out: Path, shard: int, shards: int, base_url: str, device: str) -> None:
    rows = json.loads((out / "QA_test.json").read_text(encoding="utf-8"))
    cards = load_cards()
    chosen = [row for index, row in enumerate(rows) if index % shards == shard]
    dest_dir = out / f"gpu{shard}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "predictions.jsonl"
    done = {(row["question_id"], row["step"]) for row in read_jsonl(dest)}
    pending = []
    for row in chosen:
        for task in build_tasks(row, cards):
            if (task["question_id"], task["step"]) not in done:
                pending.append(task)
    print(f"shard {shard} 待跑 {len(pending)} 已完成 {len(done)} device={device}", flush=True)
    client = None
    engine = None
    config = None
    if device == "cpu":
        import torch
        from swift.infer_engine import RequestConfig, TransformersEngine

        engine = TransformersEngine(
            MODEL,
            model_type="qwen3_5",
            template_type="qwen3_5",
            torch_dtype=torch.float32,
            attn_impl="sdpa",
            device_map="cpu",
            use_hf=True,
            max_batch_size=1,
        )
        engine.template.enable_thinking = False
        config = RequestConfig(max_tokens=1024, temperature=0)
    else:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=180)
    repaired = 0
    with dest.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(pending, start=1):
            started = time.time()
            text, error, retried = _generate(task, client, engine, config)
            repaired += int(retried)
            record = {
                "experiment": out.name,
                "question_id": task["question_id"],
                "step": task["step"],
                "source": task["source"],
                "databases": task["databases"],
                "tables": task["tables"],
                "missing_tables": task["missing_tables"],
                "knowledge_ids": task["knowledge_ids"],
                "similar_question_ids": task["similar_question_ids"],
                "pred": text,
                "check_error": error,
                "retried": retried,
                "seconds": round(time.time() - started, 3),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(pending):
                print(f"shard {shard} {index}/{len(pending)} 重写 {repaired}", flush=True)


def merge(out: Path) -> None:
    rows = []
    seen = set()
    for path in sorted(out.glob("gpu*/predictions.jsonl")):
        for row in read_jsonl(path):
            key = (row["question_id"], row["step"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (row["question_id"], row["step"]))
    dest = out / "predictions.jsonl"
    dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    enriched = json.loads((out / "QA_test.json").read_text(encoding="utf-8"))
    by_source = Counter(row["source"] for row in rows)
    questions = {row["question_id"] for row in rows}
    metrics = {
        "experiment": "test-full-v1",
        "data": str(out / "QA_test.json"),
        "questions_in_data": len(enriched),
        "router_ok": sum(1 for row in enriched if row.get("router_ok")),
        "questions_predicted": len(questions),
        "calls": len(rows),
        "by_source": dict(by_source),
        "missing_table_calls": sum(1 for row in rows if row.get("missing_tables")),
        "note": "未微调生成器。业务规则相似度 >= 0.65，相似案例同一库且相似度 >= 0.6。执行失败会重写一次。",
        "retried": sum(1 for row in rows if row.get("retried")),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def submit(out: Path) -> None:
    predictions = read_jsonl(out / "predictions.jsonl")
    grouped: dict[int, list[dict]] = {}
    for row in predictions:
        grouped.setdefault(int(row["question_id"]), []).append(row)
    order = [int(row["question_id"]) for row in json.loads((out / "QA_test.json").read_text(encoding="utf-8"))]
    result = []
    for question_id in order:
        steps = sorted(grouped.get(question_id, []), key=lambda item: item["step"])
        if not steps:
            continue
        if len(steps) == 1:
            result.append({
                "question_id": question_id,
                "db": steps[0]["source"],
                "query": guard.extract_query(steps[0]["pred"]),
            })
            continue
        result.append({
            "question_id": question_id,
            "db1": steps[0]["source"],
            "query1": guard.extract_query(steps[0]["pred"]),
            "db2": steps[1]["source"],
            "query2": guard.extract_query(steps[1]["pred"]),
        })
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'result.json'} {len(result)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["route", "retrieve", "assemble", "infer", "merge", "submit"])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--device", choices=["api", "cpu"], default="api")
    args = parser.parse_args()
    if args.stage == "route":
        route(args.output, args.batch_size)
    elif args.stage == "retrieve":
        retrieve(args.output)
    elif args.stage == "assemble":
        assemble(args.output)
    elif args.stage == "infer":
        infer(args.output, args.shard, args.shards, args.base_url, args.device)
    elif args.stage == "submit":
        submit(args.output)
    else:
        merge(args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""用 Qwen3-Embedding-0.6B 为 v2-w 生成两份 top10。

只使用 CUDA_VISIBLE_DEVICES 里可见的卡。调用时固定为 1 号卡，避开 0 号卡上的生成实验。
知识点：question_id -> 10 条 {knowledge_id, score}。
相似案例：question_id -> 10 条 {question_id, score}，不含本题。score 是余弦相似度。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL = Path("/data/models/Qwen3-Embedding-0.6B")
TRAIN = Path("/data/data/v2-w/train_eknow.json")
KNOWLEDGE = Path("/data/data/v2-w/knowledge.json")
OUT_KNOWLEDGE = Path("/data/data/v2-w/knowledge_top10_scored.json")
OUT_SIMILAR = Path("/data/data/v2-w/similar_top10_scored.json")
MAX_LENGTH = 512
BATCH = 64

KNOWLEDGE_TASK = "根据自然语言问题，检索撰写数据库查询所需要的领域知识"
SIMILAR_TASK = "根据自然语言问题，检索语义最相近的训练问题"


def last_token_pool(hidden, mask):
    left = mask[:, -1].sum() == mask.shape[0]
    if left:
        return hidden[:, -1]
    lengths = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def embed(texts, tokenizer, model, device):
    vectors = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            pooled = last_token_pool(hidden, tokens["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
        vectors.append(pooled)
        if start % (BATCH * 20) == 0:
            print(f"encoded {min(start + BATCH, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(vectors, dim=0)


def instruct(task, text):
    return f"Instruct: {task}\nQuery:{text}"


def top10(scores, ids, id_key):
    values, index = torch.topk(scores, k=10, dim=1)
    values = values.cpu()
    index = index.cpu()
    mapping = {}
    for qid, row_index, row_score in zip(question_ids, index, values):
        mapping[str(qid)] = [
            {id_key: int(ids[item]), "score": round(float(score), 6)}
            for item, score in zip(row_index, row_score)
        ]
    return mapping


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, "visible", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16 if device == "cuda" else torch.float32, trust_remote_code=True)
    model = model.to(device).eval()

    rows = json.loads(TRAIN.read_text(encoding="utf-8"))
    knowledge = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
    global question_ids
    question_ids = [row["question_id"] for row in rows]
    questions = [row["question"] for row in rows]
    knowledge_ids = [item["knowledge_id"] for item in knowledge]
    knowledge_texts = [item["knowledge_text"] for item in knowledge]

    print("embed knowledge", len(knowledge_texts), flush=True)
    knowledge_vectors = embed(knowledge_texts, tokenizer, model, device)
    print("embed questions as documents", len(questions), flush=True)
    question_vectors = embed(questions, tokenizer, model, device)
    print("embed knowledge queries", flush=True)
    knowledge_queries = embed([instruct(KNOWLEDGE_TASK, text) for text in questions], tokenizer, model, device)
    print("embed similar queries", flush=True)
    similar_queries = embed([instruct(SIMILAR_TASK, text) for text in questions], tokenizer, model, device)

    knowledge_scores = (knowledge_queries @ knowledge_vectors.T).float()
    knowledge_map = top10(knowledge_scores, knowledge_ids, "knowledge_id")
    OUT_KNOWLEDGE.write_text(json.dumps(knowledge_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    similar_scores = (similar_queries @ question_vectors.T).float()
    similar_scores.fill_diagonal_(float("-inf"))
    similar_map = top10(similar_scores, question_ids, "question_id")
    OUT_SIMILAR.write_text(json.dumps(similar_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT_KNOWLEDGE, OUT_SIMILAR, "questions", len(question_ids), flush=True)


if __name__ == "__main__":
    main()

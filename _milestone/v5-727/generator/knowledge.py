#!/usr/bin/env python3
"""知识点检索。

字段说明按调用方给出的表精确取回，业务规则只在非字段知识里做向量检索。

- 「在数据库 X 的表 Y 中……」是字段说明。命中 `库.表` 就全部附上，不再按相似度排序。
- 不以「在数据库」开头的条目是业务规则。用 Qwen3-Embedding-0.6B 的余弦相似度取前 3 条。
- 「在数据库 X 中……」但没有点明表名的，是没有挂到题目上的计算公式。整库附上会把无关公式带进提示词，因此两条路径都不返回它们。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

KNOWLEDGE = Path("/data/data/v2-w/knowledge.json")
TRAIN = Path("/data/data/v2-w/train_eknow.json")
MODEL = Path("/data/models/Qwen3-Embedding-0.6B")
RUN = Path("/data/k/runs/knowledge")
VECTORS = RUN / "rule_vectors.npz"
RULE_K = 3
MAX_LENGTH = 512
RULE_TASK = "根据自然语言问题，检索撰写数据库查询所需要的领域知识"
_TABLE = re.compile(r"在数据库([A-Za-z0-9_]+)的表([A-Za-z0-9_]+)中")


def _kind(text: str) -> str:
    if _TABLE.search(text):
        return "table"
    if text.startswith("在数据库"):
        return "recipe"
    return "rule"


def _instruct(text: str) -> str:
    return f"Instruct: {RULE_TASK}\nQuery:{text}"


class KnowledgeRetriever:
    """表名用 `库.表`。只有表名时，必须同时给出库名，避免 country 这类重名表串库。"""

    def __init__(self, path: Path = KNOWLEDGE):
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        self.digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.by_key: dict[tuple[str, str], list[dict]] = {}
        self.bare: dict[str, list[tuple[str, str]]] = {}
        self.rules: list[dict] = []
        self.recipes: list[dict] = []
        for row in rows:
            text = row["knowledge_text"]
            kind = _kind(text)
            item = {"knowledge_id": int(row["knowledge_id"]), "text": text}
            if kind == "rule":
                self.rules.append(item)
                continue
            if kind == "recipe":
                self.recipes.append(item)
                continue
            match = _TABLE.search(text)
            key = (match.group(1).lower(), match.group(2).lower())
            item["database"], item["table"] = key
            self.by_key.setdefault(key, []).append(item)
            self.bare.setdefault(key[1], [])
            if key not in self.bare[key[1]]:
                self.bare[key[1]].append(key)
        for notes in self.by_key.values():
            notes.sort(key=lambda item: item["knowledge_id"])
        self.rules.sort(key=lambda item: item["knowledge_id"])
        # 库级配方（"在数据库X中"无表名）：join 关系/计算口径说明，按库索引。
        # 原两通道（表键/规则嵌入）都不返回它们，是 v3 补的第三通道。
        self.recipes_by_db: dict[str, list[dict]] = {}
        for item in self.recipes:
            m = re.search(r"在数据库([A-Za-z0-9_]+)中", item["text"])
            if m:
                self.recipes_by_db.setdefault(m.group(1).lower(), []).append(item)
        self._matrix: np.ndarray | None = None

    def recipes_for(self, databases: list[str], k: int = 3) -> list[dict]:
        """召回库的库级配方（每库至多 k 条，按 knowledge_id 稳定排序）。"""
        out: list[dict] = []
        for db in databases or []:
            items = self.recipes_by_db.get(str(db).lower(), [])
            out.extend(items[:k])
            if len(out) >= 6:
                break
        return out

    def table_notes(self, tables: list[str], databases: list[str] | None = None) -> list[dict]:
        notes = []
        for key in sorted(self._keys(tables, databases)):
            notes.extend(self.by_key.get(key, []))
        return notes

    def rule_notes(self, question: str, k: int = RULE_K) -> list[dict]:
        return self.rule_notes_many([question], k)[0]

    def rule_notes_many(self, questions: list[str], k: int = RULE_K, device: str = "cpu", batch_size: int = 32) -> list[list[dict]]:
        if not questions:
            return []
        matrix = self._rule_matrix()
        vectors = _encode([_instruct(text) for text in questions], device, batch_size)
        scores = vectors @ matrix.T
        picked = []
        for row in scores:
            order = np.argsort(-row)[:k]
            picked.append([
                {**self.rules[int(index)], "score": round(float(row[index]), 6)}
                for index in order
            ])
        return picked

    def retrieve(self, question: str, tables: list[str], databases: list[str] | None = None, k: int = RULE_K, device: str = "cpu") -> dict:
        notes = self.table_notes(tables, databases)
        rules = self.rule_notes(question, k) if question else []
        return {
            "table_notes": notes,
            "rules": rules,
            "texts": [item["text"] for item in notes] + [item["text"] for item in rules],
        }

    def _keys(self, tables: list[str], databases: list[str] | None) -> set[tuple[str, str]]:
        allowed = {item.lower() for item in (databases or [])}
        found: set[tuple[str, str]] = set()
        for raw in tables or []:
            name = raw.strip().strip('`"').lower()
            if not name:
                continue
            if "." in name:
                database, table = name.split(".", 1)
                found.add((database, table.split(".")[-1]))
                continue
            candidates = list(self.bare.get(name, []))
            if allowed:
                candidates = [key for key in candidates if key[0] in allowed]
            elif len(candidates) != 1:
                continue
            found.update(candidates)
        return found

    def _rule_matrix(self) -> np.ndarray:
        if self._matrix is not None:
            return self._matrix
        if not VECTORS.exists():
            raise FileNotFoundError(f"还没有业务规则向量，先运行 python {Path(__file__)} --build")
        stored = np.load(VECTORS, allow_pickle=False)
        if str(stored["digest"]) != self.digest:
            raise RuntimeError("knowledge.json 已变化，需要重新 --build")
        ids = [int(item) for item in stored["ids"]]
        if ids != [item["knowledge_id"] for item in self.rules]:
            raise RuntimeError("业务规则顺序和向量不一致，需要重新 --build")
        self._matrix = stored["vectors"].astype(np.float32)
        return self._matrix


_ENCODER: dict[str, tuple] = {}


def _encoder(device: str):
    if device not in _ENCODER:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", trust_remote_code=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModel.from_pretrained(MODEL, dtype=dtype, trust_remote_code=True).to(device).eval()
        _ENCODER[device] = (tokenizer, model)
    return _ENCODER[device]


def _encode(texts: list[str], device: str, batch_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    tokenizer, model = _encoder(device)
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"]
            if mask[:, -1].sum() == mask.shape[0]:
                pooled = hidden[:, -1]
            else:
                lengths = mask.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
            pooled = F.normalize(pooled.float(), p=2, dim=1)
        vectors.append(pooled.cpu().numpy())
        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % (batch_size * 20) == 0:
            print(f"encoded {done}/{len(texts)}", flush=True)
    return np.concatenate(vectors, axis=0)


def build_vectors(device: str, batch_size: int) -> None:
    retriever = KnowledgeRetriever()
    RUN.mkdir(parents=True, exist_ok=True)
    matrix = _encode([item["text"] for item in retriever.rules], device, batch_size)
    np.savez(
        VECTORS,
        vectors=matrix.astype(np.float32),
        ids=np.array([item["knowledge_id"] for item in retriever.rules]),
        digest=np.array(retriever.digest),
    )
    print(f"wrote {VECTORS} rules {len(retriever.rules)}", flush=True)


def _rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def evaluate(device: str, batch_size: int) -> dict:
    retriever = KnowledgeRetriever()
    rows = json.loads(TRAIN.read_text(encoding="utf-8"))
    rule_ids = {item["knowledge_id"] for item in retriever.rules}
    table_ids = {item["knowledge_id"] for notes in retriever.by_key.values() for item in notes}

    table_hit = table_gold = table_questions = 0
    empty_db = 0
    for row in rows:
        gold = [int(item) for item in (row.get("knowledge_ids") or []) if int(item) in table_ids]
        if not gold:
            continue
        table_questions += 1
        if not row.get("database_names"):
            empty_db += 1
        got = {item["knowledge_id"] for item in retriever.table_notes(row.get("table_names") or [], row.get("database_names") or [])}
        table_hit += len(set(gold) & got)
        table_gold += len(gold)

    rule_rows = [row for row in rows if any(int(item) in rule_ids for item in (row.get("knowledge_ids") or []))]
    rule_hit = rule_gold_hit = rule_gold = 0
    rule_macro = 0.0
    if rule_rows:
        ranked = retriever.rule_notes_many([row["question"] for row in rule_rows], RULE_K, device, batch_size)
        macros = []
        for row, picked in zip(rule_rows, ranked):
            gold = {int(item) for item in row["knowledge_ids"] if int(item) in rule_ids}
            got = {item["knowledge_id"] for item in picked}
            rule_gold_hit += len(gold & got)
            rule_gold += len(gold)
            rule_hit += bool(gold & got)
            macros.append(len(gold & got) / len(gold))
        rule_macro = sum(macros) / len(macros)

    summary = {
        "table_notes": len(table_ids),
        "rules": len(retriever.rules),
        "recipes_held_out": len(retriever.recipes),
        "rule_k": RULE_K,
        "table_questions": table_questions,
        "table_gold": table_gold,
        "table_gold_recalled": table_hit,
        "table_recall": _rate(table_hit, table_gold),
        "table_questions_without_database": empty_db,
        "rule_questions": len(rule_rows),
        "rule_gold": rule_gold,
        "rule_gold_recalled": rule_gold_hit,
        "rule_recall_at_3": _rate(rule_gold_hit, rule_gold),
        "rule_recall_at_3_mean": round(rule_macro, 6),
        "rule_hit_at_3": _rate(rule_hit, len(rule_rows)),
        "note": "字段说明用题目标注的库名和表名精确匹配。业务规则只在 247 条非字段知识中取前 3 条。143 条未点明表名的计算公式不返回。",
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not args.build and not args.eval:
        parser.print_help()
        return
    if args.build:
        build_vectors(args.device, args.batch_size)
    if args.eval:
        evaluate(args.device, args.batch_size)


if __name__ == "__main__":
    main()

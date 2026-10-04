"""构建 oracle 模式 prompt（给模型喂 gold schema），复用 track8 的 schema_store。

oracle 模式：用 QA_train 里的 gold db + gold query 推断该问题对应的真实 schema，
再按 XiYanSQL 风格 prompt 模板组装。这样测试的是模型在「已知目标库」下的生成能力。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 复用 track8-unified-query 的 schema 解析与 prompt 模板
TRACK8 = Path("/data/track8-unified-query")
sys.path.insert(0, str(TRACK8))

from src.data.schema_store import SchemaStore  # noqa: E402
from src.pipeline.prompt import build_oracle_prompt  # noqa: E402

SCHEMA_DIR = Path("/data/raw/contest_8/Schema")


def get_store() -> SchemaStore:
    return SchemaStore(SCHEMA_DIR)


def build_prompt(store: SchemaStore, item: dict) -> tuple[str, str]:
    """返回 (prompt, dialect)。item 需含 question / db / query。"""
    dialect, schema = store.oracle_schema_for_gold(item)
    prompt = build_oracle_prompt(
        question=item["question"],
        dialect=dialect,
        schema=schema,
        evidence="",  # 不注入 knowledge，纯测模型基线
    )
    return prompt, dialect

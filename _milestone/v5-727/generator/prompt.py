"""生成器提示词。训练和这次基线测试用同一套。"""
from __future__ import annotations

import json
from pathlib import Path

KNOWLEDGE = Path("/data/raw/contest_8/knowledge.json")

_SYSTEM = {
    "MYSQL": "你是 MYSQL 查询生成器。根据问题和 Schema 生成一条只读 SQL。只输出 SQL，不要解释，不要 Markdown。",
    "SQLite": "你是 SQLite 查询生成器。根据问题和 Schema 生成一条只读 SQL。表名保持数据库前缀。只输出 SQL，不要解释，不要 Markdown。",
    "POSTGRESQL": "你是 POSTGRESQL 查询生成器。根据问题和 Schema 生成一条只读 SQL。表名不要加 public. 前缀，标识符不要加双引号。地理条件使用 geom 和 PostGIS 函数，例如 ST_DWithin。只输出 SQL，不要解释，不要 Markdown。",
    "Elasticsearch": "你是 Elasticsearch 查询生成器。根据问题和 mapping 生成一个 JSON DSL。只使用 bool、geo_distance、geo_polygon、geo_bounding_box、term、range 这些查询，外层使用 aggs 与 size，或 _source。圆形范围用 geo_distance，矩形范围用 geo_bounding_box，不要改成 geo_shape。只输出 JSON，不要解释，不要 Markdown。",
}


def system_for(source: str) -> str:
    return _SYSTEM.get(source, _SYSTEM["MYSQL"])


_CARD_TAIL = "只使用表卡片中给出的表和字段。不要添加题目里没有的过滤条件。知识中的单位换算照做。只输出查询本身，不要解释，不要 Markdown。"

_CARD_SYSTEM = {
    "MYSQL": (
        "你是 MYSQL 查询生成器。根据问题、表卡片和知识生成一条只读 SQL。"
        "表名写库名.表名，例如 transportation.vehicle。"
        "业务库和 common、education_2024 和 education_2025 写在同一条 SQL 里。"
        "日期用 DATE_FORMAT。education 的 date_key 是整数，写成 20240101，不加引号。不要用 STRFTIME。"
        + _CARD_TAIL
    ),
    "SQLite": (
        "你是 SQLite 查询生成器。根据问题、表卡片和知识生成一条只读 SQL。"
        "表名保持数据库前缀，例如 book_publishing_company.sales。"
        "年份和月份用 STRFTIME('%Y', 列)、STRFTIME('%m', 列)。不要用 DATE_FORMAT、NOW()、INTERVAL。"
        + _CARD_TAIL
    ),
    "POSTGRESQL": (
        "你是 POSTGRESQL 查询生成器。根据问题、表卡片和知识生成一条只读 SQL。"
        "表名不要加 public. 前缀，标识符不要加双引号。"
        "ST_MakePoint(经度, 纬度)，经度在前。距离用 geom::geography，单位是米。"
        "时间窗写成 NOW() - INTERVAL '24 hours'。不要用 DATE_FORMAT。"
        + _CARD_TAIL
    ),
    "Elasticsearch": (
        "你是 Elasticsearch 查询生成器。根据问题、表卡片和知识生成一个 JSON DSL。"
        "只使用 bool、geo_distance、geo_polygon、geo_bounding_box、term、range。"
        "圆形用 geo_distance，矩形用 geo_bounding_box，不要改成 geo_shape。"
        "location 里先写 lat 再写 lon。返回字段用 \"_source\"。"
        "只有纯聚合才写 size: 0。city、level 这类 keyword 用 term，不要用 match。"
        + _CARD_TAIL
    ),
}


def system_for_cards(source: str) -> str:
    """E09 使用。按数据源附带方言约束，上下文是表卡片而不是整段 DDL。"""
    return _CARD_SYSTEM.get(source, _CARD_SYSTEM["MYSQL"])


def present_schema(source: str, schema: str) -> str:
    """PostgreSQL 标注查询不写 public.，展示给模型的 Schema 去掉这个前缀。"""
    text = schema or ""
    if source == "POSTGRESQL":
        text = text.replace("public.", "")
    return text


def load_knowledge() -> dict[int, str]:
    rows = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
    return {int(row["knowledge_id"]): row["knowledge_text"] for row in rows}


def build_user(question: str, source: str, schema: str, knowledge: list[str] | None = None, step_note: str = "") -> str:
    parts = [f"问题：{question}", f"方言：{source}"]
    if step_note:
        parts.append(step_note)
    parts.append("Schema：\n" + (present_schema(source, schema) or "无"))
    if knowledge:
        parts.append("知识：\n" + "\n".join(knowledge))
    return "\n".join(parts)

"""测试集无训练生成器的约束。

业务规则相似度低于 0.65 不放进提示词。第 4 题的载重分段规则是 0.647，会把
「各车型总载重」写成轻型车、中型车、重型车，所以卡在 0.65。
相似案例必须和本题同一个库，余弦相似度低于 0.6 不使用。
生成之后检查表名、数据源是否串库、Elasticsearch 查询类型，并在库里执行。
失败时由调用方重写一次。
"""
from __future__ import annotations

import json
import os
import re

os.environ.setdefault("VERIFY_SQL_LOG", "/dev/null")

RULE_MIN_SCORE = 0.65
CASE_MIN_SCORE = 0.6

_FROM = re.compile(r"(?:from|join)\s+([`\"]?[A-Za-z_][\w.]*)", re.I)
_ES_FORBIDDEN = {
    "geo_shape",
    "match",
    "match_phrase",
    "match_phrase_prefix",
    "multi_match",
    "query_string",
    "simple_query_string",
}
_PG_IN_MYSQL = re.compile(r"vessel_ais_data|ST_DWithin|ST_MakePoint|::geography|::geometry", re.I)
_MYSQL_IN_PG = re.compile(
    r"\b(?:transportation|finance_tax|common|education_2024|education_2025)\.",
    re.I,
)


def extract_query(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.I)
    text = text.strip()
    fences = re.findall(r"```(?:sql|json)?\s*([\s\S]*?)```", text, flags=re.I)
    if fences:
        text = "\n".join(part.strip() for part in fences if part.strip())
    return text.strip()


def databases_of(row: dict) -> set[str]:
    names = {item.lower() for item in (row.get("database_names") or []) if item}
    if names:
        return names
    source = row.get("db") or ""
    if source == "POSTGRESQL":
        return {"public"}
    if source == "Elasticsearch":
        return {"medical_institutions"}
    return set()


def keep_rule(rule: dict) -> bool:
    score = rule.get("score")
    return score is None or float(score) >= RULE_MIN_SCORE


def _bare(name: str) -> str:
    return name.strip('`"').lower().split(".")[-1]


def _allowed_tables(tables: list[str]) -> set[str]:
    allowed = set()
    for name in tables:
        key = name.lower()
        allowed.add(key)
        allowed.add(_bare(key))
    return allowed


def _referenced_tables(sql: str) -> list[str]:
    return [item.strip('`"') for item in _FROM.findall(sql or "")]


def _es_forbidden(sql: str) -> list[str]:
    try:
        body = json.loads(sql)
    except json.JSONDecodeError as exc:
        return [f"不是合法 JSON：{exc}"]
    found = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _ES_FORBIDDEN:
                    found.append(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(body)
    return found


def static_error(source: str, step: int, sql: str, tables: list[str]) -> str:
    query = (sql or "").strip()
    if not query:
        return "没有输出查询"
    if source == "Elasticsearch":
        forbidden = _es_forbidden(query)
        if forbidden:
            return "Elasticsearch 使用了不允许的查询类型：" + "、".join(forbidden)
        return ""
    if step == 2 and not re.search(r"\bresult\b", query, re.I):
        return "跨源第二步必须包含 result，用它代表第一步的执行结果"
    if source == "MYSQL" and _PG_IN_MYSQL.search(query):
        return "这条是 MySQL，不要写 vessel_ais_data 或 PostGIS 函数"
    if source == "POSTGRESQL" and _MYSQL_IN_PG.search(query):
        return "这条是 PostgreSQL，不要写 MySQL 的库名前缀"
    allowed = _allowed_tables(tables)
    extra = []
    for name in _referenced_tables(query):
        key = name.lower()
        if key in allowed or _bare(key) in allowed:
            continue
        extra.append(name)
    if extra:
        return "使用了表卡片以外的表：" + "、".join(extra)
    return ""


def execution_error(source: str, step: int, sql: str) -> str:
    if source == "Elasticsearch":
        trial = sql
    else:
        trial = re.sub(r"\(result\)", "(NULL)", sql or "", flags=re.I)
    try:
        import sys
        sys.path.insert(0, "/data/lj")
        from verify_sql import _run_one

        db = {
            "MYSQL": "mysql",
            "POSTGRESQL": "postgresql",
            "SQLite": "sqlite",
            "Elasticsearch": "elasticsearch",
        }[source]
        _run_one(db, trial, 12, None)
    except Exception as exc:
        return str(exc).strip().splitlines()[0][:300]
    return ""


def problems(source: str, step: int, text: str, tables: list[str]) -> str:
    sql = extract_query(text)
    found = static_error(source, step, sql, tables)
    if found:
        return found
    return execution_error(source, step, sql)

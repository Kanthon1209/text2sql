"""按 db.table 取回 DDL。供生成器测试使用，不负责选表。"""
from __future__ import annotations

import re
from pathlib import Path

SCHEMA_DIR = Path("/data/raw/contest_8/Schema")
TABLE_RE = re.compile(
    r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["`]?([\w.-]+)["`]?',
    re.I,
)
BIRD_DB_RE = re.compile(r"数据库:\s*([\w-]+)")
USE_RE = re.compile(r"^USE\s+([\w-]+)", re.I)
CREATE_DB_RE = re.compile(r"CREATE\s+DATABASE\s+IF\s+NOT\s+EXISTS\s+([\w-]+)", re.I)


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(path)


def _slice_create(lines: list[str], index: int) -> tuple[str, int]:
    body = [lines[index]]
    depth = lines[index].count("(") - lines[index].count(")")
    end = index + 1
    if depth <= 0:
        while end < len(lines) and "(" not in lines[end]:
            body.append(lines[end])
            end += 1
        if end < len(lines):
            body.append(lines[end])
            depth = lines[end].count("(") - lines[end].count(")")
            end += 1
    while end < len(lines) and depth > 0:
        body.append(lines[end])
        depth += lines[end].count("(") - lines[end].count(")")
        end += 1
    return "\n".join(body).strip(), end


def load_tables() -> dict[str, str]:
    """键是小写的 db.table，值是 CREATE TABLE 正文。"""
    tables: dict[str, str] = {}

    def add(db: str, raw_name: str, ddl: str) -> None:
        name = raw_name.split(".")[-1].strip('"`')
        tables[f"{db}.{name}".lower()] = ddl

    def consume(path: Path) -> None:
        lines = read_text(path).splitlines()
        database = ""
        index = 0
        while index < len(lines):
            line = lines[index]
            for pattern in (BIRD_DB_RE, USE_RE, CREATE_DB_RE):
                match = pattern.search(line.strip())
                if match:
                    database = match.group(1)
                    break
            match = TABLE_RE.search(line)
            if match and database:
                ddl, index = _slice_create(lines, index)
                raw = match.group(1)
                if raw.lower().startswith("public."):
                    add("public", raw, ddl)
                else:
                    add(database, raw, ddl)
                continue
            index += 1

    for name in ("bird_schemas.txt", "commom_schema.txt", "core_schema.txt"):
        consume(SCHEMA_DIR / name)

    core = read_text(SCHEMA_DIR / "core_schema.txt")
    es = re.search(r"PUT\s+/medical_institutions[\s\S]*", core)
    if es:
        tables["medical_institutions"] = es.group(0).strip()
    return tables


def lookup(tables: dict[str, str], names: list[str]) -> tuple[str, list[str]]:
    chunks = []
    missing = []
    for name in names:
        ddl = tables.get(name.lower())
        if ddl is None and name.lower() == "public.vessel_ais_data":
            ddl = tables.get("vessel_ais_data")
        if ddl is None:
            missing.append(name)
            continue
        chunks.append(f"[{name}]\n{ddl}")
    return "\n\n".join(chunks), missing

"""按 db.table 取回 DDL 或表卡片。供生成器测试使用，不负责选表。

表卡片在 schema_cards.json：32 个库、265 张表，每张表一张卡片。
来源是活库目录加上赛题 schema 的字段注释，由 lj/pipeline 生成。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SCHEMA_DIR = Path("/data/raw/contest_8/Schema")
CARDS_PATH = Path(__file__).with_name("schema_cards.json")
TABLE_RE = re.compile(
    r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["`]?([\w.-]+)["`]?',
    re.I,
)
BIRD_DB_RE = re.compile(r"数据库:\s*([\w-]+)")
USE_RE = re.compile(r"^USE\s+([\w-]+)", re.I)
CREATE_DB_RE = re.compile(r"CREATE\s+DATABASE\s+IF\s+NOT\s+EXISTS\s+([\w-]+)", re.I)

# v3 值画像渲染上限：低基数字段才值得占提示词；纯数字长值是标识符，
# 超长文本值（BLOB/base64，如 movie_3.staff.picture 存了 7 万字符的图片内容）不是类别
VALUE_MAX = 10
VALUE_MAX_NDISTINCT = 20
VALUE_CHAR_MAX = 60
_ID_VALUE_RE = re.compile(r"^\d{16,}$")


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


def load_cards(path: Path = CARDS_PATH) -> dict[str, dict]:
    """键与 load_tables 相同，是小写的 db.table。值是一张表的卡片。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    cards: dict[str, dict] = {}
    for key, card in data["tables"].items():
        name = key.split(":", 1)[-1].lower()
        cards[name] = card
    return cards


def render_card(card: dict, full: bool = True) -> str:
    """一张表卡片渲染成提示词里的摘要。

    full=False 为 v3 紧凑卡：只给表描述、字段名和 FK——扩展表（FK/共现
    带进来的）主要供 join，不需要逐字段注释，省上下文。
    full 卡在 v3 下额外渲染值画像（低基数字段的真实 DISTINCT 值），
    模型可直接对齐字面量（type: business/mod_cook 这类）。
    """
    lines = [f"- 表 {card['table']}"
             + (f"（{card['desc']}）" if card.get("desc") else "")
             + f" [库: {card['db']}]"]
    if not full:
        fields = ", ".join(f["name"] for f in card.get("fields", []))
        if fields:
            lines.append(f"    字段: {fields}")
        for link in card.get("fks", []):
            lines.append(
                f"    FK: {link['col']} -> {link['ref_db']}.{link['ref_table']}({link['ref_col']})"
            )
        return "\n".join(lines)
    for field in card.get("fields", []):
        segment = f"    {field['name']} {field.get('type', '')}".rstrip()
        if field.get("comment"):
            segment += f" -- {field['comment']}"
        if field.get("enum"):
            segment += " 枚举:" + "|".join(field["enum"])
        values = field.get("values") or []
        n_distinct = field.get("n_distinct")
        # 纯数字长值（身份证/证件号/长编码）是标识符不是类别，渲染只有噪声；
        # 公司名这类长文本值恰恰是字面量对齐最需要的，保留
        if values and (n_distinct is None or n_distinct <= VALUE_MAX_NDISTINCT) \
                and not any(len(str(v)) > VALUE_CHAR_MAX
                            or _ID_VALUE_RE.match(str(v))
                            for v in values[:VALUE_MAX]):
            segment += " 值:" + "|".join(str(v) for v in values[:VALUE_MAX])
        if field.get("unverified"):
            segment += " [注:该列未见于赛题schema/知识库，非必要勿用]"
        lines.append(segment)
    for link in card.get("fks", []):
        lines.append(
            f"    FK: {link['col']} -> {link['ref_db']}.{link['ref_table']}({link['ref_col']})"
        )
    return "\n".join(lines)


def lookup_cards(cards: dict[str, dict], names: list[str],
                 full_names: set[str] | None = None) -> tuple[str, list[str]]:
    """full_names 里的表渲染全量卡（含值画像），其余紧凑卡。

    full_names=None 时全部全量（v3 之前的行为）。
    """
    chunks = []
    missing = []
    for name in names:
        key = name.lower()
        card = cards.get(key)
        if card is None and key == "public.vessel_ais_data":
            card = cards.get("vessel_ais_data")
        if card is None:
            missing.append(name)
            continue
        full = full_names is None or key in full_names or name in (full_names or ())
        chunks.append(render_card(card, full=full))
    return "\n\n".join(chunks), missing


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

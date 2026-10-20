"""Load CSV / Excel / Parquet into a clean DataFrame."""
import sqlite3
from contextlib import contextmanager

import pandas as pd


def _dedupe_columns(cols):
    seen, out = {}, []
    for c in map(str, cols):
        c = c.strip()
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        out.append(c)
    return out


@contextmanager
def _sqlite_memory_database(source):
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(source.getvalue())
        yield connection
    finally:
        connection.close()


def sqlite_tables(source) -> list[str]:
    """List user tables in an uploaded SQLite database without modifying it."""
    with _sqlite_memory_database(source) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


def load_sqlite_table(source, table: str) -> pd.DataFrame:
    """Load one selected table from an uploaded SQLite database."""
    if table not in sqlite_tables(source):
        raise ValueError(f"Table not found in uploaded SQLite database: {table}")
    escaped_table = table.replace('"', '""')
    with _sqlite_memory_database(source) as connection:
        return pd.read_sql_query(f'SELECT * FROM "{escaped_table}"', connection)


def load(source, name=None) -> pd.DataFrame:
    name = (name or getattr(source, "name", str(source))).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(source)
    elif name.endswith(".jsonl"):
        df = pd.read_json(source, lines=True)
    elif name.endswith(".json"):
        df = pd.read_json(source)
    else:
        df = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_csv(source, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Could not decode file")
    df.columns = _dedupe_columns(df.columns)
    for c in df.select_dtypes("object").columns:  # detect date columns
        s = df[c].dropna().astype(str).head(200)
        if len(s) and s.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean() > 0.8:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[c] = parsed
    return df

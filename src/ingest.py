from __future__ import annotations
import os
import pandas as pd
from typing import Optional

def load_csv(path: str, sample_rows: int = 50000, seed: int = 42) -> pd.DataFrame:
    # Try fast read; fall back to chunked sampling for big files
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        # If separators/encodings are messy, let pandas guess
        df = pd.read_csv(path, low_memory=False, sep=None, engine="python")
    if len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=seed).reset_index(drop=True)
    return _postprocess(df)

def load_sql(conn_str: str, table_or_query: str, sample_rows: int = 50000, seed: int = 42) -> pd.DataFrame:
    import sqlalchemy as sa
    engine = sa.create_engine(conn_str)
    query = table_or_query if table_or_query.strip().lower().startswith(("select","with")) else f"SELECT * FROM {table_or_query}"
    df = pd.read_sql_query(query, engine)
    if len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=seed).reset_index(drop=True)
    return _postprocess(df)

def _postprocess(df):
    # Downcast numeric dtypes to save memory
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df
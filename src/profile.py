from __future__ import annotations
import pandas as pd
import numpy as np

def schema_summary(df: pd.DataFrame):
    rows, cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    nulls = df.isna().sum().to_dict()
    uniques = df.nunique(dropna=True).to_dict()
    example = {c: _example_val(df[c]) for c in df.columns}
    return {
        "shape": {"rows": int(rows), "cols": int(cols)},
        "dtypes": dtypes,
        "nulls": {k: int(v) for k, v in nulls.items()},
        "uniques": {k: int(v) for k, v in uniques.items()},
        "examples": example,
    }

def _example_val(s: pd.Series):
    try:
        val = s.dropna().iloc[0]
        if isinstance(val, (np.floating, float)):
            return float(val)
        if isinstance(val, (np.integer, int)):
            return int(val)
        return str(val)[:120]
    except Exception:
        return None

def correlations(df: pd.DataFrame, max_vars: int = 40):
    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] == 0:
        return None
    if num.shape[1] > max_vars:
        num = num.iloc[:, :max_vars]
    corr = num.corr(numeric_only=True).fillna(0.0)
    return corr
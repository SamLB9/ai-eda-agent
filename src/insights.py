from __future__ import annotations
import pandas as pd

def basic_insights(profile: dict, target: str | None = None) -> list[str]:
    out = []
    shape = profile["shape"]
    out.append(f"Dataset has {shape['rows']} rows × {shape['cols']} columns.")
    null_cols = [c for c, n in profile["nulls"].items() if n > 0]
    if null_cols:
        out.append(f"{len(null_cols)} columns have missing values (e.g., {', '.join(null_cols[:5])}).")
    high_card = [c for c, u in profile["uniques"].items() if u > 0.8 * shape["rows"]]
    if high_card:
        out.append(f"{len(high_card)} high-cardinality columns (e.g., {', '.join(high_card[:5])}).")
    if target and target in profile["dtypes"]:
        out.append(f"Target set to '{target}'. Consider plots vs {target} for top predictors.")
    return out
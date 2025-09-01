from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

def hist_plot(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    df[col].dropna().plot(kind="hist", bins=30, ax=ax)
    ax.set_title(f"Distribution: {col}")
    ax.set_xlabel(col); ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def bar_top_k(df: pd.DataFrame, col: str, k: int = 20):
    fig, ax = plt.subplots()
    vc = df[col].astype(str).value_counts().head(k)
    vc.plot(kind="bar", ax=ax)
    ax.set_title(f"Top {k} categories: {col}")
    ax.set_xlabel(col); ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def scatter(df: pd.DataFrame, x: str, y: str):
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y], s=12, alpha=0.6)
    ax.set_title(f"{y} vs {x}")
    ax.set_xlabel(x); ax.set_ylabel(y)
    fig.tight_layout()
    return fig

def heatmap_corr(corr):
    import numpy as np
    fig, ax = plt.subplots()
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(corr.shape[1])); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(corr.shape[0])); ax.set_yticklabels(corr.index)
    ax.set_title("Correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
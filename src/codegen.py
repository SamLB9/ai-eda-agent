from __future__ import annotations

import re
from typing import Callable, Dict


def _imports_block() -> str:
    return (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n\n"
    )


def generate_code(step: str) -> str:
    """Return Python code (as a string) to perform a simple EDA action.

    Assumptions
    - The DataFrame is named `df`.
    - Only uses pandas, numpy, matplotlib.pyplot.
    - Outputs either printouts or a matplotlib figure.
    """
    if not isinstance(step, str) or not step.strip():
        return _imports_block() + "print('No step instruction provided.')\n"

    step_norm = step.strip().lower()

    # Exact/contains match registry
    handlers: Dict[str, Callable[[], str]] = {
        "preview dataset shape and column dtypes": _code_preview_shape_dtypes,
        "summarize missing values": _code_missing_values,
        "identify duplicated rows and potential unique keys": _code_duplicates_and_keys,
        "summary statistics for numeric columns": _code_numeric_summary,
        "plot distributions (histogram/boxplot) for numeric columns": _code_numeric_distributions,
        "check pairwise correlations among numeric columns": _code_correlation_heatmap,
        "value counts for categorical and boolean columns": _code_categorical_value_counts,
        "group by categorical columns to compare summary metrics": _code_groupby_categorical_summary,
        "plot trends over time for datetime columns": _code_datetime_trends,
        "inspect text length distribution and frequent tokens for text columns": _code_text_lengths_and_tokens,
        "bivariate analysis: each feature vs target": _code_bivariate_vs_target,
        "assess numeric-feature correlation with target (e.g., pearson/spearman)": _code_corr_with_target,
        "compare target rates across categorical groups": _code_target_rates_by_category,
        "check class balance of the target variable": _code_target_class_balance,
        "compare feature distributions by target class": _code_feature_dists_by_target,
    }

    # Direct mapping where step matches a key or contains it
    for key, func in handlers.items():
        if step_norm == key or key in step_norm:
            return func()

    # Pattern: "Plot distribution of <Column>"
    m = re.search(r"plot\s+(?:the\s+)?distribution[s]?\s+of\s+(.+)$", step, flags=re.IGNORECASE)
    if m:
        column_expr = m.group(1).strip().strip('"\'')
        return _code_plot_distribution_of(column_expr)

    # Fallback: unrecognized instruction
    return _imports_block() + (
        "print('Unrecognized step instruction: %s')\n" % step.replace("'", "\'")
    )


# -------------------- Generators --------------------

def _code_preview_shape_dtypes() -> str:
    return _imports_block() + (
        "print('Shape:', df.shape)\n"
        "print('\\nDtypes:')\n"
        "print(df.dtypes)\n"
    )


def _code_missing_values() -> str:
    return _imports_block() + (
        "missing_counts = df.isnull().sum().sort_values(ascending=False)\n"
        "missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)\n"
        "print('Missing values by column:')\n"
        "print(missing_counts)\n"
        "print('\\nMissing percent by column:')\n"
        "print(missing_pct.round(2))\n"
    )


def _code_duplicates_and_keys() -> str:
    return _imports_block() + (
        "dup_count = df.duplicated().sum()\n"
        "print(f'Duplicated rows: {dup_count}')\n"
        "unique_cols = [c for c in df.columns if df[c].is_unique]\n"
        "print('Potential unique key columns:', unique_cols)\n"
    )


def _code_numeric_summary() -> str:
    return _imports_block() + (
        "num_desc = df.describe(include=[np.number]).T\n"
        "print(num_desc)\n"
    )


def _code_numeric_distributions() -> str:
    return _imports_block() + (
        "numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
        "if not numeric_cols:\n"
        "    print('No numeric columns found.')\n"
        "else:\n"
        "    n = len(numeric_cols)\n"
        "    fig, axes = plt.subplots(n, 1, figsize=(8, 3*n))\n"
        "    if n == 1:\n"
        "        axes = [axes]\n"
        "    for ax, col in zip(axes, numeric_cols):\n"
        "        ax.hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='white')\n"
        "        ax.set_title(f'Distribution of {col}')\n"
        "        ax.set_xlabel(col)\n"
        "        ax.set_ylabel('Frequency')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _code_correlation_heatmap() -> str:
    return _imports_block() + (
        "num_df = df.select_dtypes(include=[np.number])\n"
        "if num_df.shape[1] < 2:\n"
        "    print('Need at least 2 numeric columns for correlation heatmap.')\n"
        "else:\n"
        "    corr = num_df.corr(numeric_only=True)\n"
        "    fig, ax = plt.subplots(figsize=(0.6*corr.shape[1]+3, 0.6*corr.shape[0]+3))\n"
        "    cax = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)\n"
        "    ax.set_xticks(range(corr.shape[1]))\n"
        "    ax.set_xticklabels(corr.columns, rotation=90)\n"
        "    ax.set_yticks(range(corr.shape[0]))\n"
        "    ax.set_yticklabels(corr.index)\n"
        "    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)\n"
        "    ax.set_title('Correlation heatmap (numeric columns)')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _code_categorical_value_counts() -> str:
    return _imports_block() + (
        "cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()\n"
        "if not cat_cols:\n"
        "    print('No categorical/boolean columns found.')\n"
        "else:\n"
        "    for col in cat_cols:\n"
        "        print(f'\\n=== {col} ===')\n"
        "        print(df[col].value_counts(dropna=False).head(20))\n"
    )


def _code_groupby_categorical_summary() -> str:
    return _imports_block() + (
        "cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()\n"
        "num_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
        "if not cat_cols or not num_cols:\n"
        "    print('Need both categorical and numeric columns.')\n"
        "else:\n"
        "    for col in cat_cols:\n"
        "        print(f'\\n=== Grouped by {col} ===')\n"
        "        print(df.groupby(col)[num_cols].agg(['count', 'mean']).head(10))\n"
    )


def _code_datetime_trends() -> str:
    return _imports_block() + (
        "dt_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, tz]']).columns.tolist()\n"
        "num_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
        "if not dt_cols or not num_cols:\n"
        "    print('Need at least one datetime column and one numeric column.')\n"
        "else:\n"
        "    dt = dt_cols[0]\n"
        "    cols_to_plot = num_cols[:3]\n"
        "    fig, ax = plt.subplots(figsize=(10, 5))\n"
        "    for col in cols_to_plot:\n"
        "        ax.plot(df[dt], df[col], label=str(col))\n"
        "    ax.legend()\n"
        "    ax.set_title(f'Trends over time by {dt}')\n"
        "    ax.set_xlabel(dt)\n"
        "    ax.set_ylabel('Value')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _code_text_lengths_and_tokens() -> str:
    return _imports_block() + (
        "text_cols = df.select_dtypes(include=['object']).columns.tolist()\n"
        "if not text_cols:\n"
        "    print('No text/object columns found.')\n"
        "else:\n"
        "    col = text_cols[0]\n"
        "    lengths = df[col].astype(str).str.len()\n"
        "    plt.figure(figsize=(8, 4))\n"
        "    plt.hist(lengths.dropna(), bins=50, color='slateblue', edgecolor='white')\n"
        "    plt.title(f'Text length distribution for {col}')\n"
        "    plt.xlabel('Length')\n"
        "    plt.ylabel('Frequency')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    tokens = df[col].astype(str).str.split(expand=True).stack().value_counts().head(20)\n"
        "    print('\\nTop tokens:')\n"
        "    print(tokens)\n"
    )


def _code_bivariate_vs_target() -> str:
    return _imports_block() + (
        "if 'target' not in df.columns:\n"
        "    print(\"Column 'target' not found. Please ensure a 'target' column exists.\")\n"
        "else:\n"
        "    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'target']\n"
        "    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()\n"
        "    for col in num_cols:\n"
        "        corr = df[[col, 'target']].corr(numeric_only=True).iloc[0, 1]\n"
        "        print(f'Numeric vs target correlation: {col}: {corr:.3f}')\n"
        "    for col in cat_cols:\n"
        "        rates = df.groupby(col)['target'].mean().sort_values(ascending=False)\n"
        "        print(f'\\nTarget rate by {col}:')\n"
        "        print(rates.head(20))\n"
    )


def _code_corr_with_target() -> str:
    return _imports_block() + (
        "if 'target' not in df.columns:\n"
        "    print(\"Column 'target' not found. Please ensure a 'target' column exists.\")\n"
        "else:\n"
        "    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'target']\n"
        "    corrs = {}\n"
        "    for col in num_cols:\n"
        "        corrs[col] = df[[col, 'target']].corr(numeric_only=True).iloc[0, 1]\n"
        "    s = pd.Series(corrs).sort_values(ascending=False)\n"
        "    print(s)\n"
    )


def _code_target_rates_by_category() -> str:
    return _imports_block() + (
        "if 'target' not in df.columns:\n"
        "    print(\"Column 'target' not found. Please ensure a 'target' column exists.\")\n"
        "else:\n"
        "    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()\n"
        "    if not cat_cols:\n"
        "        print('No categorical/boolean columns found.')\n"
        "    else:\n"
        "        for col in cat_cols:\n"
        "            rates = df.groupby(col)['target'].mean().sort_values(ascending=False)\n"
        "            print(f'\\nTarget rate by {col}:')\n"
        "            print(rates.head(20))\n"
    )


def _code_target_class_balance() -> str:
    return _imports_block() + (
        "if 'target' not in df.columns:\n"
        "    print(\"Column 'target' not found. Please ensure a 'target' column exists.\")\n"
        "else:\n"
        "    counts = df['target'].value_counts(dropna=False)\n"
        "    pct = df['target'].value_counts(normalize=True, dropna=False) * 100\n"
        "    print('Class counts:')\n"
        "    print(counts)\n"
        "    print('\\nClass percentage:')\n"
        "    print(pct.round(2))\n"
    )


def _code_feature_dists_by_target() -> str:
    return _imports_block() + (
        "if 'target' not in df.columns:\n"
        "    print(\"Column 'target' not found. Please ensure a 'target' column exists.\")\n"
        "else:\n"
        "    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'target']\n"
        "    classes = df['target'].dropna().unique()\n"
        "    if len(classes) < 2:\n"
        "        print('Need at least two target classes to compare distributions.')\n"
        "    else:\n"
        "        cols_to_plot = num_cols[:3]\n"
        "        fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(8, 3*len(cols_to_plot)))\n"
        "        if len(cols_to_plot) == 1:\n"
        "            axes = [axes]\n"
        "        for ax, col in zip(axes, cols_to_plot):\n"
        "            for cls in classes[:2]:\n"
        "                data = df.loc[df['target'] == cls, col].dropna()\n"
        "                ax.hist(data, bins=30, alpha=0.6, label=str(cls))\n"
        "            ax.set_title(f'{col} by target class')\n"
        "            ax.set_xlabel(col)\n"
        "            ax.set_ylabel('Frequency')\n"
        "            ax.legend(title='target')\n"
        "        plt.tight_layout()\n"
        "        plt.show()\n"
    )


def _code_plot_distribution_of(column_expr: str) -> str:
    safe_col = column_expr.strip().strip('"\'')
    return _imports_block() + (
        f"col = '{safe_col}'\n"
        "if col not in df.columns:\n"
        "    print(f\"Column '{col}' not found in df.\")\n"
        "else:\n"
        "    plt.figure(figsize=(8, 4))\n"
        "    plt.hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='white')\n"
        "    plt.title(f'Distribution of {col}')\n"
        "    plt.xlabel(col)\n"
        "    plt.ylabel('Frequency')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    ) 
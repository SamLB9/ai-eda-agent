from __future__ import annotations

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env in project root (if present)
load_dotenv()


def _client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key.")
    base_url = os.getenv("OPENAI_BASE_URL")
    organization = os.getenv("OPENAI_ORG")
    return OpenAI(api_key=key, base_url=base_url, organization=organization)


def _resolve_model(model: str) -> str:
    return (model or "gpt-5-mini").strip()


def llm_autonomous_eda_code(
    schema: Dict[str, str],
    missing_pct: Dict[str, float],
    distinct_counts: Dict[str, int],
    goal: str,
    time_budget_min: int,
    model: str = "gpt-5-mini",
    topic: Optional[str] = None,
    request_timeout: Optional[int] = None,
) -> str:
    base_client = _client()
    client = base_client.with_options(timeout=request_timeout) if request_timeout else base_client
    model = _resolve_model(model)
    sys = (
        "You are an expert data scientist.")
    topic_line = f"Topic of interest: {topic}\n" if (topic and topic.strip()) else "Topic of interest: general data analysis\n"
    user = (
        "Generate a SINGLE self-contained Python script that performs an autonomous exploratory data analysis (EDA) on a DataFrame named df.\n"
        "Constraints:\n"
        "- Only use these libraries: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, sklearn.* (for ML), tabulate.\n"
        "- Do not import any other libraries. Do not read or write files. No internet access.\n"
        "- IMPORTANT: Do NOT call plt.show() and do NOT close figures (e.g., no plt.close('all')). The environment will capture figures automatically.\n"
        "- CRITICAL: You MUST create plots! Include comprehensive visualizations using plt.plot(), plt.hist(), plt.scatter(), sns.histplot(), sns.boxplot(), etc. Create 5-8 plots maximum for performance.\n"
        "- PLOT EXAMPLES: Use these exact patterns:\n"
        "  * Histogram example:\n"
        "    plt.figure(figsize=(10, 6))\n"
        "    plt.hist(df['column'], bins=20)\n"
        "    plt.title('Distribution of Column Name')\n"
        "    plt.xlabel('Column Name')\n"
        "    plt.ylabel('Frequency')\n"
        "  * Boxplot example:\n"
        "    plt.figure(figsize=(10, 6))\n"
        "    sns.boxplot(data=df, x='cat_col', y='num_col')\n"
        "    plt.title('Distribution of Numeric Column by Category')\n"
        "    plt.xlabel('Category Column')\n"
        "    plt.ylabel('Numeric Column')\n"
        "  * For boxplots with categorical data, ensure sufficient data per category:\n"
        "    top_categories = df['cat_col'].value_counts().head(10).index.tolist()\n"
        "    df_filtered = df[df['cat_col'].isin(top_categories)]\n"
        "    plt.figure(figsize=(10, 6))\n"
        "    sns.boxplot(data=df_filtered, x='cat_col', y='num_col')\n"
        "    plt.title('Distribution of Numeric Column by Top Categories')\n"
        "    plt.xlabel('Category Column')\n"
        "    plt.ylabel('Numeric Column')\n"
        "  * Scatter plot example:\n"
        "    plt.figure(figsize=(10, 6))\n"
        "    plt.scatter(df['x_col'], df['y_col'])\n"
        "    plt.title('Relationship between X Column and Y Column')\n"
        "    plt.xlabel('X Column')\n"
        "    plt.ylabel('Y Column')\n"
        "- Create 5-8 plots maximum for performance. Focus on most important visualizations.\n"
        "- MANDATORY: ALL plots must have descriptive titles and axis labels:\n"
        "  * ALWAYS use plt.title('Descriptive Title') for every plot\n"
        "  * ALWAYS use plt.xlabel('X-axis Label') and plt.ylabel('Y-axis Label')\n"
        "  * Use specific, descriptive titles (e.g., 'Distribution of Popularity by Genre', not just 'Box plot')\n"
        "  * Include variable names in titles and axis labels for clarity\n"
        "  * Example: plt.title('Distribution of Danceability Scores'); plt.xlabel('Danceability'); plt.ylabel('Frequency')\n"
        "- CRITICAL: Generate FAST, EFFICIENT code. Avoid operations that could hang:\n"
        "  * NEVER use loops to create plots (e.g., for col in columns: plt.figure())\n"
        "  * Create individual plot statements instead of loops\n"
        "  * Use the FULL dataset for analysis - do not sample\n"
        "  * For plots: use sampling only for visualization (e.g., df.sample(1000) for scatter plots)\n"
        "  * For correlation: limit to most important numeric columns (max 10-15 columns)\n"
        "  * For groupby: always limit results to top 10-20 groups with .head()\n"
        "  * Use simple ML models only (no complex ensembles)\n"
        "  * Avoid seaborn.pairplot() and other heavy operations\n"
        "- Print clear section markers BEFORE each section exactly as:\n"
        "  ### SECTION: Dataset Summary\n"
        "  ### SECTION: Visualizations\n"
        "  ### SECTION: Statistical Insights\n"
        "  ### SECTION: ML Results\n"
        "  ### SECTION: Final Explanation\n"
        "- SECTION CONTENT GUIDELINES:\n"
        "  * Dataset Summary: Basic info (sample data, descriptive stats, missing values, data types)\n"
        "  * Statistical Insights: Correlations, grouped statistics, feature relationships\n"
        "  * ML Results: Model results, predictions, evaluation metrics\n"
        "- Use random_state=42 where applicable.\n"
        f"- Time budget: {time_budget_min} minutes. Adapt depth accordingly.\n"
        "- The DataFrame is already provided as df.\n"
        "- PERFECT TABLE FORMATTING: Use st.dataframe() for all data tables:\n"
        "  * NEVER use print() for DataFrames - ALWAYS use st.dataframe()\n"
        "  * CRITICAL: Add a descriptive title comment BEFORE each st.dataframe() call:\n"
        "    # Table: Sample Data (First 5 rows)\n"
        "    st.dataframe(df.head(), use_container_width=True, hide_index=True)\n"
        "    # Table: Descriptive Statistics\n"
        "    st.dataframe(df.describe().round(3), use_container_width=True, hide_index=True)\n"
        "    # Table: Missing Values Summary\n"
        "    st.dataframe(missing_df, use_container_width=True, hide_index=True)\n"
        "    # Table: Column Data Types\n"
        "    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)\n"
        "    # Table: Top Genres by Count\n"
        "    st.dataframe(genre_counts.head(10), use_container_width=True, hide_index=True)\n"
        "  * CRITICAL: Title comments MUST be in the EXACT same order as st.dataframe() calls.\n"
        "  * CRITICAL: Each st.dataframe() call must have exactly one title comment immediately before it.\n"
        "  * CRITICAL: Do NOT skip title comments or add extra ones - match 1:1 with st.dataframe() calls.\n"
        "  * MANDATORY: Always create missing_df and dtypes_df:\n"
        "    missing_df = (df.isna().sum() / len(df) * 100).round(3).reset_index()\n"
        "    missing_df.columns = ['column', 'missing_pct']\n"
        "    dtypes_df = pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str)})\n"
        "  * For descriptive statistics: ALWAYS select only numeric columns for describe():\n"
        "    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
        "    desc_stats = df[numeric_cols].describe()\n"
        "    st.dataframe(desc_stats.round(3), use_container_width=True, hide_index=True)\n"
        "  * For correlation matrix: ALWAYS create a readable correlation matrix with feature names as first column in the Statistical Insights section:\n"
        "    # Example for correlation matrix (place in Statistical Insights section):\n"
        "    # IMPORTANT: Limit to most important numeric columns (max 10-15) for performance\n"
        "    numeric_cols = ['col1', 'col2', 'col3']  # Select only most important numeric columns\n"
        "    corr = df[numeric_cols].corr()\n"
        "    corr_with_names = corr.reset_index().rename(columns={'index': 'feature'})\n"
        "    # Table: Correlation Matrix\n"
        "    st.dataframe(corr_with_names.round(3), use_container_width=True, hide_index=True)\n"
        "    # This creates a table with 'feature' as first column, then correlation values\n"
        "  * For correlation with target variable: Create ONLY simple correlation (not absolute) to show direction:\n"
        "    # Example for correlation with popularity (simple correlation, not absolute):\n"
        "    if 'popularity' in corr.columns:\n"
        "        corr_with_target = corr['popularity'].sort_values(ascending=False).reset_index().rename(columns={'index': 'feature', 'popularity': 'correlation_with_popularity'})\n"
        "        # Table: Correlation with Popularity\n"
        "        st.dataframe(corr_with_target.round(3), use_container_width=True, hide_index=True)\n"
        "    # DO NOT create absolute correlation tables - they provide less information than simple correlations\n"
        "  * For value counts: st.dataframe(vc.head(10).reset_index(), use_container_width=True, hide_index=True)\n"
        "  * For sample data: st.dataframe(df.head(), use_container_width=True, hide_index=True)\n"
        "  * For grouped data: st.dataframe(grouped.round(3), use_container_width=True, hide_index=True)\n"
        "  * IMPORTANT: For groupby operations, always limit to top groups: .head(10) or .head(20)\n"
        "  * IMPORTANT: Use the full dataset for groupby but limit results for performance\n"
        "  * EFFICIENCY: Use st.dataframe() calls as needed for comprehensive data display\n"
        "  * PRIORITY: Focus on most important tables (describe, corr, head)\n"
        "  * Always use .round(3) for numeric data\n"
        "  * For value_counts(): use .reset_index() to convert to DataFrame first\n"
        "  * IMPORTANT: Replace ALL print(df.xxx) with st.dataframe(df.xxx)\n"
        "  * EFFICIENCY: Keep code simple and fast - avoid complex operations\n"
        "  * VARIABLES: Only use 'df' - do not reference undefined variables like 'topic' or 'goal'\n"
        "  * CRITICAL: DO NOT create custom classes like _St or _FallbackST - use the provided 'st' object directly\n"
        "  * PERFORMANCE: Create 5-8 plots maximum, use sampling for large datasets, avoid heavy computations\n"
        "  * TIME BUDGET RULES: For 5min budget - NO ML models, NO train_test_split, NO sklearn models, NO confusion matrices. Only basic EDA.\n\n"
        f"Schema (name: dtype): {schema}\n"
        f"Missing % per column: {missing_pct}\n"
        f"Distinct counts per column: {distinct_counts}\n"
        f"User focus/context: {goal}\n"
        f"{topic_line}"
        "Guidelines by time budget:\n"
        "- 5min: Basic EDA - dataset summary (missing values, column types, value counts), comprehensive plots (distributions, correlations, relationships), statistical insights (correlations, key metrics tables), final explanation. NO ML models.\n"
        "- 10min: + correlations, grouped stats, richer plots; optionally a very light model.\n"
        "- 20min: + simple ML (logistic regression, decision tree) with basic evaluation.\n"
        "- 30min: + limited advanced ML (random forest, gradient boosting), deeper plots and explanations.\n\n"
        "REQUIRED TABLES: Every EDA MUST include these 4 essential tables in the Dataset Summary section:\n"
        "1. # Table: Sample Data (First 5 rows)\n"
        "   st.dataframe(df.head(), use_container_width=True, hide_index=True)\n"
        "2. # Table: Descriptive Statistics\n"
        "   numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
        "   desc_stats = df[numeric_cols].describe()\n"
        "   st.dataframe(desc_stats.round(3), use_container_width=True, hide_index=True)\n"
        "3. # Table: Missing Values Summary\n"
        "   st.dataframe(missing_df, use_container_width=True, hide_index=True)\n"
        "4. # Table: Column Data Types\n"
        "   st.dataframe(dtypes_df, use_container_width=True, hide_index=True)\n"
        "These are MANDATORY - always include them before any other tables.\n"
        "IMPORTANT: Correlation matrices belong in Statistical Insights section, NOT Dataset Summary.\n\n"
        "Return ONLY the Python code. No markdown fences, no explanations."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def llm_plan(schema: Dict[str, str], goal: str, model: str = "gpt-5-mini") -> List[str]:
    client = _client()
    model = _resolve_model(model)
    sys = (
        "You are an expert data scientist. Generate a concise, actionable EDA plan as a numbered list of short steps."
        " Keep it 6-10 items, avoid long sentences."
    )
    user = f"Schema: {schema}\nGoal: {goal}\nReturn only the list of steps."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or ""
    steps = [line.split(".", 1)[-1].strip() if "." in line[:4] else line.strip() for line in content.splitlines() if line.strip()]
    steps = [s for s in steps if s]
    return steps


def llm_codegen(step: str, model: str = "gpt-5-mini") -> str:
    client = _client()
    model = _resolve_model(model)
    sys = (
        "Return only Python code that assumes a DataFrame named df and only imports pandas as pd, numpy as np, and matplotlib.pyplot as plt."
        " Code should either print results or show a matplotlib plot."
    )
    user = f"Step: {step}\nOnly return code, no explanations."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def llm_explain(stdout: str, figures: List[str], step: str, goal: str, model: str = "gpt-5-mini") -> str:
    client = _client()
    model = _resolve_model(model)
    sys = "Write a concise (3-5 sentences) explanation of EDA results in plain English. Mention figures, summarize key numerics, and tie back to the goal."
    user = (
        f"Goal: {goal}\nStep: {step}\nStdout:\n{stdout}\nFigures: {len(figures)} images."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def llm_generate_key_findings(dataframes: List[Dict], goal: str, model: str = "gpt-5-mini") -> str:
    """Generate key findings based on the data tables produced by the EDA."""
    client = _client()
    model = _resolve_model(model)
    
    # Prepare the data tables for analysis
    tables_summary = []
    for i, df_info in enumerate(dataframes):
        df = df_info['data']
        title = df_info.get('title', f'Table {i+1}')
        
        # Create a summary of the table
        table_summary = f"Table {i+1}: {title}\n"
        table_summary += f"Shape: {df.shape}\n"
        table_summary += f"Columns: {list(df.columns)}\n"
        
        # Add sample data for analysis
        if len(df) <= 10:
            table_summary += f"Data:\n{df.to_string()}\n"
        else:
            table_summary += f"Sample data (first 5 rows):\n{df.head().to_string()}\n"
        
        tables_summary.append(table_summary)
    
    all_tables = "\n\n".join(tables_summary)
    
    sys = """You are an expert data analyst. Based on the data tables provided from an EDA, write a concise paragraph (3-5 sentences) summarizing the KEY FINDINGS and INSIGHTS. Focus on:

1. Most important patterns or trends in the data
2. Notable statistics or correlations
3. Surprising or interesting discoveries
4. What these findings mean in the context of the user's goal

Write in clear, professional language. Avoid generic statements like "the data shows various patterns" - be specific about what you found."""
    
    user = f"""Goal: {goal}

Data Tables from EDA:
{all_tables}

Please provide a concise summary of the key findings and insights from this data."""
    
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def llm_fix_syntax_error(
    original_code: str,
    syntax_error: str,
    schema: Dict[str, str],
    missing_pct: Dict[str, float],
    distinct_counts: Dict[str, int],
    goal: str,
    model: str = "gpt-5-mini",
    request_timeout: Optional[int] = None,
) -> str:
    """Ask the LLM to fix a syntax error in its generated code."""
    base_client = _client()
    client = base_client.with_options(timeout=request_timeout) if request_timeout else base_client
    model = _resolve_model(model)

    sys = "You are an expert Python programmer. Fix the syntax error in the provided code."

    user = f"""The code below has a syntax error: {syntax_error}

Please fix this syntax error and return ONLY the corrected Python code. Do not add explanations or markdown formatting.

Original code:
{original_code}

Context:
- DataFrame is named 'df'
- Available libraries: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, sklearn.*, tabulate
- Schema: {schema}
- Missing %: {missing_pct}
- Goal: {goal}

Return ONLY the fixed Python code."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or "" 
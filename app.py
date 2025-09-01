import os
import io
import streamlit as st
import pandas as pd
from src.ingest import load_csv
from src.profile import schema_summary, correlations
from src.charts import hist_plot, bar_top_k, scatter, heatmap_corr
from src.insights import basic_insights

st.set_page_config(page_title="AI Data Explorer", layout="wide")

st.title("📊 AI Data Explorer — Milestone 1")
st.caption("Upload a CSV (or connect a DB later), auto-profile the dataset, and view first-pass EDA.")

with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("CSV file", type=["csv"])
    sample_rows = st.number_input("Sample rows (max)", 1000, 200000, 50000, step=1000)
    seed = st.number_input("Random seed", 1, 999999, 42, step=1)
    target = st.text_input("Target column (optional)", "")

    run = st.button("Run Profiling")

# Load data
df = None
if up is not None and run:
    bytes_data = up.getvalue()
    tmp_path = os.path.join("data", "uploads", up.name)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(bytes_data)
    with st.spinner("Reading CSV..."):
        df = load_csv(tmp_path, sample_rows=sample_rows, seed=seed)

if df is not None:
    st.success(f"Loaded {len(df)} rows × {df.shape[1]} columns (sampled if large).")
    # Profile
    prof = schema_summary(df)
    col1, col2, col3 = st.columns([1.5,1,1])
    with col1:
        st.subheader("Schema & Types")
        st.json({"shape": prof["shape"], "dtypes": prof["dtypes"]})
    with col2:
        st.subheader("Nulls (per column)")
        st.json(prof["nulls"])
    with col3:
        st.subheader("Uniques (per column)")
        st.json(prof["uniques"])

    # Insights
    st.subheader("Insights")
    bullets = basic_insights(prof, target.strip() or None)
    for b in bullets:
        st.write("• " + b)

    # Charts
    st.subheader("Auto-EDA Charts")
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = [c for c in df.columns if c not in num_cols]

    # 1) Up to 3 numeric histograms
    for c in num_cols[:3]:
        st.pyplot(hist_plot(df, c))

    # 2) Up to 2 categorical bar charts
    for c in cat_cols[:2]:
        st.pyplot(bar_top_k(df, c))

    # 3) One scatter of first two numeric columns (if exist)
    if len(num_cols) >= 2:
        st.pyplot(scatter(df, num_cols[0], num_cols[1]))

    # 4) Correlation heatmap
    corr = correlations(df)
    if corr is not None and corr.shape[1] >= 2:
        st.pyplot(heatmap_corr(corr))

else:
    st.info("Upload a CSV and click **Run Profiling** to begin.")
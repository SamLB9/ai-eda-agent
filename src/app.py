from __future__ import annotations

import base64
import io
import time
import re
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from planner import Planner
from codegen import generate_code
from executor import execute_code
from explainer import explain_results
from llm import llm_plan, llm_codegen, llm_explain
from session_runner import run_autonomous_session, SECTION_HEADERS


# -------------------- Utilities --------------------

def _get_data_dir() -> Path:
    """Get the correct path to the data directory."""
    # Get the directory where this app.py file is located
    app_dir = Path(__file__).parent
    # Go up one level and into data/uploads
    data_dir = app_dir.parent / "data" / "uploads"
    return data_dir

def _init_session_state():
    if "df" not in st.session_state:
        st.session_state.df = None  # type: Optional[pd.DataFrame]
    if "plan" not in st.session_state:
        st.session_state.plan = []  # type: List[str]
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "logs" not in st.session_state:
        st.session_state.logs = []  # type: List[Dict]
    if "last_run_idx" not in st.session_state:
        st.session_state.last_run_idx = -1
    if "refine_text" not in st.session_state:
        st.session_state.refine_text = ""
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "hide_on_error" not in st.session_state:
        st.session_state.hide_on_error = True


def _parse_schema_input(raw: str) -> Dict[str, str]:
    raw = (raw or "").strip()
    mapping: Dict[str, str] = {}
    if not raw:
        return mapping
    # Try JSON-like dict first
    try:
        import json

        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    # Fallback: lines like "col: type"
    for line in raw.splitlines():
        if ":" in line:
            name, dtype = line.split(":", 1)
            mapping[name.strip()] = dtype.strip()
    return mapping


def _fig_b64_to_st_image(fig_b64: str):
    data = base64.b64decode(fig_b64)
    st.image(io.BytesIO(data), use_container_width=True)





def _get_table_title(df: pd.DataFrame, index: int, section_name: str) -> str:
    """Generate a descriptive title for a DataFrame based on its content."""
    if df.empty:
        return f"Table {index + 1}"
    
    # Check DataFrame content to determine appropriate title
    columns = [col.lower() for col in df.columns]
    
    # Dataset Summary section titles
    if section_name == "Dataset Summary":
        # Check for sample data first (df.head()) - this should be the highest priority
        if len(df) <= 5:
            return "Sample Data"
        elif any('missing' in col for col in columns):
            return "Missing Values Summary"
        elif any('dtype' in col for col in columns) or any('type' in col for col in columns):
            return "Column Data Types"
        elif len(df) == 1 and any('shape' in col for col in columns):
            return "Dataset Overview"
        elif any('count' in col for col in columns) and any('mean' in col for col in columns):
            return "Descriptive Statistics"
        elif any('genre' in col for col in columns) and len(df) <= 10:
            return "Top Genres by Count"
        elif any('artist' in col for col in columns) and len(df) <= 10:
            return "Top Artists by Count"
        else:
            return "Dataset Summary Table"
    
    # Statistical Insights section titles
    elif section_name == "Statistical Insights":
        if any('corr' in col for col in columns) or any('correlation' in col for col in columns):
            # Check if this is a correlation with popularity table
            if any('popularity' in col for col in columns) and len(df) <= 15:
                return "Correlation with Popularity"
            else:
                return "Correlation Matrix"
        elif any('popularity' in col for col in columns) and any('artist' in col for col in columns):
            return "Artist Popularity Analysis"
        elif any('group' in col for col in columns) or any('aggregate' in col for col in columns):
            return "Grouped Statistics"
        elif any('skew' in col for col in columns) or any('kurt' in col for col in columns):
            return "Distribution Statistics"
        else:
            return "Statistical Analysis"
    
    # Default title
    return f"Analysis Table {index + 1}"





def _get_table_description(df: pd.DataFrame, section_name: str) -> str:
    """Generate a description for a DataFrame based on its content."""
    if df.empty:
        return ""
    
    columns = [col.lower() for col in df.columns]
    
    # Dataset Summary descriptions
    if section_name == "Dataset Summary":
        if any('missing' in col for col in columns):
            return "Shows the percentage of missing values for each column in the dataset."
        elif any('dtype' in col for col in columns):
            return "Displays the data type of each column in the dataset."
        elif any('count' in col for col in columns) and any('mean' in col for col in columns):
            return "Provides descriptive statistics including count, mean, standard deviation, and quartiles for numeric columns."
        elif any('genre' in col for col in columns):
            return "Lists the most common genres in the dataset with their frequency counts."
        elif any('artist' in col for col in columns):
            return "Shows the artists with the most tracks in the dataset."
        elif len(df) <= 5:
            return "Sample of the first few rows from the dataset."
    
    # Statistical Insights descriptions
    elif section_name == "Statistical Insights":
        if any('corr' in col for col in columns):
            # Check if this is a correlation with popularity table
            if any('popularity' in col for col in columns) and len(df) <= 15:
                return "Shows the correlation coefficients between each feature and popularity, ranked by absolute correlation strength."
            else:
                return "Correlation matrix showing the strength and direction of relationships between all numeric features."
        elif any('popularity' in col for col in columns) and any('artist' in col for col in columns):
            return "Analysis of artist popularity metrics including track counts and average popularity scores."
        elif any('group' in col for col in columns):
            return "Aggregated statistics grouped by categories or features."
        elif any('skew' in col for col in columns):
            return "Distribution statistics showing skewness and kurtosis of numeric features."
    
    return ""


def _display_captured_dataframes(dataframes: List[Dict], section_name: str):
    """Display captured dataframes with proper Streamlit formatting."""
    if not dataframes:
        return
    
    # Don't add duplicate subtitles - the main section header is already there
    # st.subheader(f"{section_name} - Data Tables")
    
    # Track displayed dataframes to prevent duplicates
    displayed_dataframes = set()
    
    for i, df_info in enumerate(dataframes):
        df = df_info['data']
        kwargs = df_info.get('kwargs', {})
        
        # Use captured title if available, otherwise generate one
        captured_title = df_info.get('title')
        call_order = df_info.get('call_order', i + 1)
        
        # Use LLM-generated title if available, otherwise fallback to generated title
        table_title = captured_title or _get_table_title(df, i, section_name)
        
        # Create a unique identifier for this dataframe to prevent duplicates
        df_signature = f"{table_title}_{df.shape}_{list(df.columns)[:3]}"  # Use title, shape, and first 3 columns
        if df_signature in displayed_dataframes:
            print(f"DEBUG: Skipping duplicate dataframe: {table_title}")
            continue
        displayed_dataframes.add(df_signature)
        
        # Debug: log title assignment for troubleshooting
        print(f"DEBUG: Table {call_order} in {section_name} - Captured: '{captured_title}', Final: '{table_title}'")
        
        # Display the title cleanly
        st.markdown(f"**{table_title}**")
        
        # Add description based on DataFrame content
        table_description = _get_table_description(df, section_name)
        if table_description:
            st.caption(table_description)
        
        # Handle duplicate column names by adding suffixes
        if len(df.columns) != len(set(df.columns)):
            # Find duplicate columns and rename them
            seen_columns = {}
            new_columns = []
            for col in df.columns:
                if col in seen_columns:
                    seen_columns[col] += 1
                    new_columns.append(f"{col}_{seen_columns[col]}")
                else:
                    seen_columns[col] = 0
                    new_columns.append(col)
            df = df.copy()
            df.columns = new_columns
            st.warning(f"⚠️ Duplicate column names detected and renamed in table {i+1}")
        
        # Apply proper formatting
        column_config = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check if column contains numeric data - use safer dtype checking
            try:
                col_dtype = str(df[col].dtype)
            except AttributeError:
                # Fallback to dtypes if dtype fails
                try:
                    col_dtype = str(df.dtypes[col])
                except:
                    col_dtype = 'object'  # Default fallback
            
            if col_dtype in ['int64', 'float64']:
                # Check if it's a percentage column
                if '%' in col or 'pct' in col_lower or 'rate' in col_lower:
                    # Format as percentage
                    column_config[col] = st.column_config.NumberColumn(
                        col,
                        format="%.1f%%",
                        help=f"Column: {col}"
                    )
                else:
                    # Format as number with 2 decimal places
                    column_config[col] = st.column_config.NumberColumn(
                        col,
                        format="%.2f",
                        help=f"Column: {col}"
                    )
            elif 'date' in col_lower or 'time' in col_lower:
                # Format as date
                column_config[col] = st.column_config.DateColumn(
                    col,
                    format="YYYY-MM-DD",
                    help=f"Column: {col}"
                )
            elif col_dtype == 'object':
                # Text column with max characters to avoid ugly wrapping
                column_config[col] = st.column_config.TextColumn(
                    col,
                    max_chars=50,
                    help=f"Column: {col}"
                )
        
        # Determine height based on number of rows
        height = min(400, max(200, len(df) * 35 + 100))
        
        # Display with pagination if large
        if len(df) > 50:
            # Add pagination controls
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                rows_per_page = st.selectbox(f"Rows per page - Table {i+1}", [10, 25, 50, 100], index=2, key=f"rows_{i}")
            
            total_pages = (len(df) + rows_per_page - 1) // rows_per_page
            
            with col1:
                if total_pages > 1:
                    page = st.selectbox(f"Page - Table {i+1}", range(1, total_pages + 1), index=0, key=f"page_{i}")
                else:
                    page = 1
            
            # Slice DataFrame for current page
            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            df_page = df.iloc[start_idx:end_idx]
            
            # Display page info
            with col3:
                st.text(f"Page {page} of {total_pages}")
                st.text(f"Rows {start_idx + 1}-{min(end_idx, len(df))} of {len(df)}")
            
            # Display the DataFrame
            st.dataframe(
                df_page,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
                height=height
            )
        else:
            # Display full DataFrame without pagination
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
                height=height
            )
        
        st.divider()


def _parse_and_display_dataframe(text_content: str):
    """Parse text content and display any pandas DataFrames using st.dataframe with proper formatting."""
    if not text_content:
        return
    
    lines = text_content.split('\n')
    current_text = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this looks like a pandas table header or DATAFRAME marker
        if (any(term in line.lower() for term in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'popularity', 'acousticness', 'danceability', 'energy', 'dtype', 'missing', 'object', 'int64', 'float64']) or
            '### DATAFRAME:' in line or
            'Dtypes:' in line or
            'Missing % by column:' in line):
            # Display any accumulated text first
            if current_text:
                st.text('\n'.join(current_text))
                current_text = []
            
            # Collect table lines
            table_lines = [line]
            j = i + 1
            
            while j < len(lines):
                next_line = lines[j]
                # Check if this line contains numeric data with proper spacing
                if (any(char.isdigit() for char in next_line) and 
                    next_line.count('  ') >= 2 and 
                    len(next_line.strip()) > 10):
                    table_lines.append(next_line)
                    j += 1
                else:
                    break
            
            if len(table_lines) >= 2:
                # Convert to DataFrame and display
                try:
                    df = _parse_table_to_dataframe(table_lines)
                    if df is not None and not df.empty:
                        _display_dataframe_with_formatting(df)
                    else:
                        # Fallback to text display
                        st.text('\n'.join(table_lines))
                except Exception:
                    # Fallback to text display if parsing fails
                    st.text('\n'.join(table_lines))
                
                i = j - 1
            else:
                current_text.append(line)
        else:
            current_text.append(line)
        
        i += 1
    
    # Display any remaining text
    if current_text:
        st.text('\n'.join(current_text))


def _parse_table_to_dataframe(table_lines: List[str]) -> Optional[pd.DataFrame]:
    """Parse table lines into a pandas DataFrame."""
    if len(table_lines) < 2:
        return None
    
    try:
        # Handle special cases like "Dtypes:" and "Missing % by column:"
        if any('Dtypes:' in line for line in table_lines) or any('Missing % by column:' in line for line in table_lines):
            # This is a key-value table format
            data_rows = []
            for line in table_lines:
                line = line.strip()
                if not line or ':' in line and line.endswith(':'):
                    continue
                
                # Split by multiple spaces to separate key and value
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    data_rows.append([key, value])
            
            if data_rows:
                # Determine column names based on content
                if any('Dtypes:' in line for line in table_lines):
                    columns = ['Column', 'Data Type']
                elif any('Missing % by column:' in line for line in table_lines):
                    columns = ['Column', 'Missing %']
                else:
                    columns = ['Key', 'Value']
                
                return pd.DataFrame(data_rows, columns=columns)
        
        # Parse header (first line)
        header_line = table_lines[0].strip()
        
        # Handle different table formats
        if '|' in header_line:
            # Tabulate grid format
            columns = [col.strip() for col in header_line.split('|')[1:-1] if col.strip()]
        else:
            # Space-separated format
            columns = re.split(r'\s{2,}', header_line)
            columns = [col.strip() for col in columns if col.strip()]
        
        if not columns:
            return None
        
        # Parse data rows
        data_rows = []
        for line in table_lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # Skip separator lines (like +---+---+)
            if line.startswith('+') and line.endswith('+'):
                continue
            
            # Handle different table formats
            if '|' in line:
                # Tabulate grid format
                values = [col.strip() for col in line.split('|')[1:-1]]
            else:
                # Space-separated format
                values = re.split(r'\s{2,}', line)
                values = [val.strip() for val in values]
            
            # Ensure we have the right number of values
            if len(values) >= len(columns):
                # Take only the first len(columns) values
                row_data = values[:len(columns)]
                data_rows.append(row_data)
        
        if not data_rows:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=columns)
        
        # Convert numeric columns more efficiently
        for col in df.columns:
            try:
                # Only process object columns that might be numeric
                if df[col].dtype == 'object':
                    # Quick check if column might be numeric
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        # Check if most values look numeric
                        numeric_count = sum(1 for v in sample_values if str(v).replace('.', '').replace('-', '').replace('%', '').replace(',', '').isdigit())
                        if numeric_count >= len(sample_values) * 0.8:  # 80% are numeric
                            # Clean and convert
                            cleaned = df[col].astype(str).str.replace('%', '').str.replace(',', '')
                            df[col] = pd.to_numeric(cleaned, errors='coerce')
            except Exception:
                # Keep as object if conversion fails
                pass
        
        return df
    
    except Exception as e:
        # Don't show warning for every parsing failure - just return None
        return None


def _display_dataframe_with_formatting(df: pd.DataFrame):
    """Display DataFrame using st.dataframe with clean, consistent formatting."""
    if df.empty:
        return
    
    # Configure column formatting
    column_config = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if column contains numeric data - use safer dtype checking
        try:
            col_dtype = str(df[col].dtype)
        except AttributeError:
            # Fallback to dtypes if dtype fails
            try:
                col_dtype = str(df.dtypes[col])
            except:
                col_dtype = 'object'  # Default fallback
        
        if col_dtype in ['int64', 'float64']:
            # Check if it's a percentage column
            if '%' in col or 'pct' in col_lower or 'rate' in col_lower:
                # Format as percentage
                column_config[col] = st.column_config.NumberColumn(
                    col,
                    format="%.1f%%",
                    help=f"Column: {col}"
                )
            else:
                # Format as number with 2 decimal places
                column_config[col] = st.column_config.NumberColumn(
                    col,
                    format="%.2f",
                    help=f"Column: {col}"
                )
        elif 'date' in col_lower or 'time' in col_lower:
            # Format as date
            column_config[col] = st.column_config.DateColumn(
                col,
                format="YYYY-MM-DD",
                help=f"Column: {col}"
            )
        elif col_dtype == 'object':
            # Text column with max characters to avoid ugly wrapping
            column_config[col] = st.column_config.TextColumn(
                col,
                max_chars=50,
                help=f"Column: {col}"
            )
    
    # Determine height based on number of rows (fixed height as requested)
    height = min(400, max(200, len(df) * 35 + 100))
    
    # Display with pagination if large (more than 50 rows)
    if len(df) > 50:
        # Add pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=2, key=f"rows_{id(df)}")
        
        total_pages = (len(df) + rows_per_page - 1) // rows_per_page
        
        with col1:
            if total_pages > 1:
                page = st.selectbox("Page", range(1, total_pages + 1), index=0, key=f"page_{id(df)}")
            else:
                page = 1
        
        # Slice DataFrame for current page
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        df_page = df.iloc[start_idx:end_idx]
        
        # Display page info
        with col3:
            st.text(f"Page {page} of {total_pages}")
            st.text(f"Rows {start_idx + 1}-{min(end_idx, len(df))} of {len(df)}")
        
        # Display the DataFrame with all requested formatting
        st.dataframe(
            df_page,
            use_container_width=True,  # Use full container width
            hide_index=True,  # Hide index unless explicitly needed
            column_config=column_config,  # Apply column formatting
            height=height  # Fixed height
        )
    else:
        # Display full DataFrame without pagination
        st.dataframe(
            df,
            use_container_width=True,  # Use full container width
            hide_index=True,  # Hide index unless explicitly needed
            column_config=column_config,  # Apply column formatting
            height=height  # Fixed height
        )


# -------------------- UI --------------------

def main():
    st.set_page_config(page_title="AI Data Explorer", layout="wide")
    _init_session_state()

    with st.sidebar:
        st.header("Data & Focus")
        
        # Dataset selection
        dataset_option = st.selectbox(
            "Choose a dataset:",
            ["Upload your own CSV", "Spotify Features (Music)", "Titanic Survival", "Video Game Sales"],
            index=0
        )
        
        if dataset_option == "Upload your own CSV":
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded is not None:
                try:
                    st.session_state.df = pd.read_csv(uploaded)
                    st.success(f"Loaded DataFrame with shape {st.session_state.df.shape}")
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")
            else:
                st.info("No CSV uploaded. A small example DataFrame will be used.")
                if st.session_state.df is None:
                    st.session_state.df = pd.DataFrame(
                        {
                            "Age": [22, 35, 58, None, 41, 29, 18, 22, 35, 60],
                            "Sex": [
                                "male",
                                "female",
                                "female",
                                "male",
                                "male",
                                "female",
                                "female",
                                "male",
                                "female",
                                "male",
                            ],
                            "target": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
                        }
                    )
        
        elif dataset_option == "Spotify Features (Music)":
            try:
                if st.session_state.df is None or st.session_state.df.shape != (232725, 18):
                    data_dir = _get_data_dir()
                    spotify_path = data_dir / "SpotifyFeatures.csv"
                    st.info(f"Loading from: {spotify_path}")
                    st.session_state.df = pd.read_csv(spotify_path)
                    st.success(f"Loaded Spotify Features dataset with shape {st.session_state.df.shape}")
            except Exception as e:
                st.error(f"Failed to load Spotify dataset: {e}")
                st.info(f"Data directory: {_get_data_dir()}")
                st.info("Falling back to example DataFrame")
                if st.session_state.df is None:
                    st.session_state.df = pd.DataFrame(
                        {
                            "Age": [22, 35, 58, None, 41, 29, 18, 22, 35, 60],
                            "Sex": [
                                "male",
                                "female",
                                "female",
                                "male",
                                "male",
                                "female",
                                "female",
                                "male",
                                "female",
                                "male",
                            ],
                            "target": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
                        }
                    )
        
        elif dataset_option == "Titanic Survival":
            try:
                if st.session_state.df is None or st.session_state.df.shape != (888, 12):
                    data_dir = _get_data_dir()
                    titanic_path = data_dir / "titanic.csv"
                    st.info(f"Loading from: {titanic_path}")
                    st.session_state.df = pd.read_csv(titanic_path)
                    st.success(f"Loaded Titanic dataset with shape {st.session_state.df.shape}")
            except Exception as e:
                st.error(f"Failed to load Titanic dataset: {e}")
                st.info(f"Data directory: {_get_data_dir()}")
                st.info("Falling back to example DataFrame")
                if st.session_state.df is None:
                    st.session_state.df = pd.DataFrame(
                        {
                            "Age": [22, 35, 58, None, 41, 29, 18, 22, 35, 60],
                            "Sex": [
                                "male",
                                "female",
                                "female",
                                "male",
                                "male",
                                "female",
                                "female",
                                "male",
                                "female",
                                "male",
                            ],
                            "target": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
                        }
                    )
        
        elif dataset_option == "Video Game Sales":
            try:
                if st.session_state.df is None:
                    data_dir = _get_data_dir()
                    vgsales_path = data_dir / "vgsales.csv"
                    st.info(f"Loading from: {vgsales_path}")
                    st.session_state.df = pd.read_csv(vgsales_path)
                    st.success(f"Loaded Video Game Sales dataset with shape {st.session_state.df.shape}")
            except Exception as e:
                st.error(f"Failed to load Video Game Sales dataset: {e}")
                st.info(f"Data directory: {_get_data_dir()}")
                st.info("Falling back to example DataFrame")
                if st.session_state.df is None:
                    st.session_state.df = pd.DataFrame(
                        {
                            "Age": [22, 35, 58, None, 41, 29, 18, 22, 35, 60],
                            "Sex": [
                                "male",
                                "female",
                                "female",
                                "male",
                                "male",
                                "female",
                                "female",
                                "male",
                                "female",
                                "male",
                            ],
                            "target": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
                        }
                    )

        st.session_state.topic = st.text_input(
            "Focus (what exactly you want to analyze)",
            value=st.session_state.topic,
            placeholder="e.g., drivers of churn for senior customers",
        )

        st.divider()
        # LLM Settings and Display Settings removed for beta version
        # Model: gpt-5-mini (fixed)
        # hide_on_error is set to True by default

        reset_btn = st.button("Reset Session", use_container_width=True)

        if reset_btn:
            st.session_state.plan = []
            st.session_state.current_idx = 0
            st.session_state.logs = []
            st.session_state.last_run_idx = -1
            st.session_state.refine_text = ""
            st.session_state.topic = ""
            st.rerun()

    st.header("Autonomous EDA (One-Click)")
    
    # Time budget selection - currently only showing simple analysis option
    # Other options (10, 20, 30 min) are commented out for beta release but can be re-enabled later
    time_budget = st.selectbox(
        "Analysis Type", 
        ["Simple dataset analysis (roughly 2min)"],  # [5, 10, 20, 30] - other options commented for beta
        index=0
    )
    
    # Convert the display text to actual time budget value
    time_budget_minutes = 5  # Always 5 minutes for beta version
    
    run_auto = st.button("Run Autonomous EDA", type="primary")

    if run_auto:
        if st.session_state.df is None or st.session_state.df.empty:
            st.error("Please upload a CSV first.")
            st.stop()
        # Capture data outside of background thread to avoid accessing session state there
        df_local = st.session_state.df
        topic_local = st.session_state.topic
        hide_on_error_local = True  # Fixed to True for beta version

        # Time-budget-based progress bar (Beta version - fixed 5-minute analysis)
        total_seconds = int(time_budget_minutes) * 60
        # Extend actual sandbox timeout with a grace factor so long steps can finish
        grace_factor = 2.0
        timeout_seconds_local = int(total_seconds * grace_factor)
        progress = st.progress(0)
        status = st.empty()
        start_ts = time.time()
        from concurrent.futures import ThreadPoolExecutor

        def _run_session(df_obj: pd.DataFrame, topic_str: str, hide_flag: bool):
            timeout_seconds = timeout_seconds_local
            return run_autonomous_session(
                df_obj,
                topic_str,
                time_budget_min=int(time_budget_minutes),
                model="gpt-5-mini",
                timeout_seconds=timeout_seconds,
                topic=None,
                hide_on_error=hide_flag,
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_session, df_local, topic_local, hide_on_error_local)
                # Update progress until done or time budget reached
                while not future.done():
                    elapsed = time.time() - start_ts
                    pct = min(int((elapsed / total_seconds) * 100), 99)
                    progress.progress(pct)
                    remaining = max(0, int(total_seconds - elapsed))
                    status.text(f"Running autonomous EDA... {pct}% | ~{remaining}s remaining (elapsed {int(elapsed)}s of {total_seconds}s)")
                    time.sleep(0.2)
                session_res = future.result()
                progress.progress(100)
                status.text(f"Completed in {int(time.time() - start_ts)}s")
        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f"Autonomous session failed: {e}")
            st.stop()

        # Display results in the required structure
        
        # Display all captured dataframes first (these are the actual DataFrames)
        dataframes = session_res.get("dataframes", [])
        
        # Parse sections to understand what content should go where
        sections = session_res.get("sections", {})
        
        # Display Dataset Summary section
        st.subheader("1. Dataset Summary")
        if dataframes:
            # For Dataset Summary, take the first few dataframes (typically: sample, describe, missing, genre, artist)
            dataset_summary_dfs = dataframes[:5] if len(dataframes) >= 5 else dataframes
            _display_captured_dataframes(dataset_summary_dfs, "Dataset Summary")
        else:
            _parse_and_display_dataframe(sections.get("Dataset Summary", ""))

        st.subheader("2. Visualizations")
        figs = session_res.get("figures", [])
        if figs:
            for b64 in figs:
                _fig_b64_to_st_image(b64)
        else:
            st.info("No figures produced.")

        st.subheader("3. Statistical Insights")
        if dataframes and len(dataframes) > 5:
            # For Statistical Insights, take the remaining dataframes (typically: correlations, grouped stats)
            statistical_insights_dfs = dataframes[5:]
            _display_captured_dataframes(statistical_insights_dfs, "Statistical Insights")
        else:
            _parse_and_display_dataframe(sections.get("Statistical Insights", ""))

        # st.subheader("4. ML Results")
        # # ML Results typically don't have dataframes for 5min budget
        # _parse_and_display_dataframe(sections.get("ML Results", ""))

        st.subheader("4. Final Explanation")
        st.write(session_res["sections"].get("Final Explanation", ""))

        # Display the GPT-generated code for analysis
        st.subheader("5. Generated Code")
        st.info("Below is the Python code generated by GPT for this analysis. You can review it to identify any errors or improvements.")
        
        # Debug: Check if code exists and show its length
        if "code" in session_res:
            code_length = len(session_res["code"])
            st.info(f"Code length: {code_length} characters")
            if code_length > 0:
                # Check if this is fallback code
                if "# ---- FALLBACK" in session_res["code"]:
                    if "(due to SyntaxError)" in session_res["code"]:
                        st.warning("⚠️ Fallback code was used due to syntax errors in the original GPT-generated code.")
                    else:
                        st.warning("⚠️ Fallback code was used because the original GPT-generated code was too minimal or produced insufficient output.")
                    st.info("The code below shows the original GPT code followed by the fallback code that was actually executed.")
                    
                    # Extract and show just the original GPT code
                    parts = session_res["code"].split("# ---- FALLBACK")
                    if len(parts) > 1:
                        original_code = parts[0].strip()
                        st.subheader("Original GPT Code (with errors):")
                        st.code(original_code, language="python")
                        st.subheader("Fallback Code (what actually ran):")
                        st.code(parts[1].strip(), language="python")
                    else:
                        st.code(session_res["code"], language="python")
                else:
                    st.code(session_res["code"], language="python")
                
                # Show a summary of what happened
                if "# ---- FALLBACK" in session_res["code"]:
                    st.info("💡 The original GPT code was too minimal or had errors, so the fallback code was executed instead.")
            else:
                st.warning("Generated code is empty!")
        else:
            st.error("No 'code' field found in session results!")
            st.write("Available keys:", list(session_res.keys()))

        if session_res.get("error"):
            st.subheader("Execution Error")
            st.warning("An error occurred during execution. Showing raw error details.")
            st.code(session_res["error"], language="text")

        # Save a condensed log entry; errors intentionally hidden from UI
        st.session_state.logs.append({
            "mode": "autonomous",
            "time_budget": time_budget,
            "topic": topic_local,
            "code": session_res["code"],
            "sections": session_res["sections"],
            "figures": figs,
            "error": session_res.get("error"),
        })

    st.header("Session Logs")
    if not st.session_state.logs:
        st.info("No logs yet. Run a session to populate logs.")
    else:
        for i, entry in enumerate(st.session_state.logs):
            title = f"Autonomous Session ({entry.get('time_budget','-')} min)"
            with st.expander(title):
                st.caption("Focus")
                st.text(entry.get("topic", "<none>"))
                st.caption("Code")
                st.code(entry["code"], language="python")
                for h in SECTION_HEADERS:
                    st.caption(h)
                    content = entry["sections"].get(h, "")
                    if h in ("Dataset Summary", "Statistical Insights"):  # "ML Results" commented out for future use
                        _parse_and_display_dataframe(content or "<none>")
                    elif h == "Final Explanation":
                        st.write(content or "<none>")
                    elif h == "Visualizations":
                        if entry["figures"]:
                            for b64 in entry["figures"]:
                                _fig_b64_to_st_image(b64)
                        else:
                            st.text("<no figures>")


if __name__ == "__main__":
    main() 
from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional

import pandas as pd

try:
    from .executor import execute_code
    from .llm import llm_autonomous_eda_code, llm_generate_key_findings
except ImportError:
    from executor import execute_code
    from llm import llm_autonomous_eda_code, llm_generate_key_findings


def _remove_duplicate_sentences(text: str) -> str:
    """Remove duplicate sentences from text, keeping only the first occurrence."""
    if not text or not text.strip():
        return text
    
    lines = text.split('\n')
    seen_sentences = set()
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append('')
            continue
            
        # Check if this line is a duplicate sentence
        # Normalize the sentence for comparison (remove extra spaces, convert to lowercase)
        normalized = re.sub(r'\s+', ' ', line.lower().strip())
        
        if normalized in seen_sentences:
            # Skip duplicate sentence
            continue
        else:
            seen_sentences.add(normalized)
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


SECTION_HEADERS = [
    "Dataset Summary",
    "Visualizations",
    "Statistical Insights",
    "ML Results",
    "Final Explanation",
]


def compute_dataset_info(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, int]]:
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_pct = {col: float(df[col].isna().mean() * 100.0) for col in df.columns}
    distinct_counts = {col: int(df[col].nunique(dropna=False)) for col in df.columns}
    return schema, missing_pct, distinct_counts


def extract_code(text: str) -> str:
    # If the model returns code fences, extract; else return as-is
    m = re.search(r"```(?:python)?\n([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def convert_pandas_to_tabulate(stdout: str) -> str:
    """Convert pandas table output to tabulate format for better display."""
    import re
    
    # Look for pandas-style table patterns and convert them
    lines = stdout.split('\n')
    converted_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this looks like a pandas table header (contains column names)
        if any(col in line.lower() for col in ['popularity', 'acousticness', 'danceability', 'energy', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']):
            # This might be a pandas table - look ahead to see if it's a full table
            table_lines = [line]
            j = i + 1
            
            # Collect subsequent lines that look like table data
            while j < len(lines) and (any(char.isdigit() for char in lines[j]) and lines[j].count('  ') >= 2):
                table_lines.append(lines[j])
                j += 1
            
            if len(table_lines) >= 2:  # At least header + one data row
                # Convert this table section to tabulate format
                converted_lines.append("```")
                converted_lines.extend(table_lines)
                converted_lines.append("```")
                i = j - 1  # Skip the lines we just processed
            else:
                converted_lines.append(line)
        else:
            converted_lines.append(line)
        
        i += 1
    
    return '\n'.join(converted_lines)


def auto_convert_to_tabulate(stdout: str) -> str:
    """Automatically convert pandas-style table output to tabulate format."""
    import re
    
    # Split into lines and process
    lines = stdout.split('\n')
    result_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for pandas table patterns
        if any(term in line.lower() for term in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'popularity', 'acousticness', 'danceability']):
            # This might be a table header - collect the table
            table_lines = [line]
            j = i + 1
            
            # Collect subsequent lines that look like table data
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
                # Convert to tabulate format
                result_lines.append("```")
                result_lines.extend(table_lines)
                result_lines.append("```")
                i = j - 1
            else:
                result_lines.append(line)
        else:
            result_lines.append(line)
        
        i += 1
    
    return '\n'.join(result_lines)


def format_pandas_output(stdout: str) -> str:
    """Post-process stdout to format raw pandas output for better readability."""
    # Add section separators if missing
    if "### SECTION:" in stdout and "=" not in stdout:
        stdout = stdout.replace("### SECTION:", "\n" + "="*60 + "\n### SECTION:")
    
    # Simple approach: wrap pandas tables in code blocks for better display
    lines = stdout.split('\n')
    formatted_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this looks like a pandas table header
        if any(term in line.lower() for term in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']):
            # Start a code block
            formatted_lines.append("```")
            formatted_lines.append(line)
            
            # Continue adding lines until we hit a non-table line
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if (any(char.isdigit() for char in next_line) and 
                    next_line.count('  ') >= 2):
                    formatted_lines.append(next_line)
                    j += 1
                else:
                    break
            
            # End the code block
            formatted_lines.append("```")
            i = j - 1
        else:
            formatted_lines.append(line)
        
        i += 1
    
    return '\n'.join(formatted_lines)


def parse_sections(stdout: str) -> Dict[str, str]:
    pattern = re.compile(r"^### SECTION:\s*(.+)\s*$", re.MULTILINE)
    parts: Dict[str, str] = {h: "" for h in SECTION_HEADERS}
    indices = [(m.start(), m.group(1).strip()) for m in pattern.finditer(stdout)]
    
    # DEBUG: Print parsing info
    print(f"DEBUG: parse_sections - stdout length: {len(stdout)}")
    print(f"DEBUG: parse_sections - found {len(indices)} section markers")
    for start, title in indices:
        print(f"DEBUG: parse_sections - found marker: '{title}' at position {start}")
    
    if not indices:
        # Fallback: try to intelligently parse content without section markers
        print(f"DEBUG: parse_sections - no markers found, trying intelligent parsing")
        
        # Look for common patterns to identify sections
        lines = stdout.split('\n')
        current_section = "Dataset Summary"
        section_content = {h: [] for h in SECTION_HEADERS}
        
        for line in lines:
            line_lower = line.lower()
            
            # Try to identify sections based on content
            if any(term in line_lower for term in ['key findings', 'summary', 'conclusion', 'final', 'explanation']):
                current_section = "Final Explanation"
            elif any(term in line_lower for term in ['correlation', 'statistical', 'insights', 'analysis']):
                current_section = "Statistical Insights"
            elif any(term in line_lower for term in ['visualization', 'plot', 'chart', 'figure']):
                current_section = "Visualizations"
            elif any(term in line_lower for term in ['ml', 'model', 'prediction', 'training']):
                current_section = "ML Results"
            elif any(term in line_lower for term in ['dataset', 'shape', 'dtype', 'missing']):
                current_section = "Dataset Summary"
            
            section_content[current_section].append(line)
        
        # Join content for each section
        for section in SECTION_HEADERS:
            if section_content[section]:
                content = '\n'.join(section_content[section]).strip()
                
                # Remove markdown code blocks from the content
                content = re.sub(r'```[a-zA-Z]*\n', '', content)  # Remove ```python, ```, etc.
                content = re.sub(r'\n```\n', '\n', content)  # Remove closing ```
                content = re.sub(r'^```\n', '', content)  # Remove opening ``` at start
                content = re.sub(r'\n```$', '', content)  # Remove closing ``` at end
                content = re.sub(r'^```$', '', content)  # Remove standalone ```
                content = re.sub(r'```$', '', content)  # Remove ``` at end of line
                
                parts[section] = content.strip()
        
        return parts
    indices.append((len(stdout), "__END__"))
    for i in range(len(indices) - 1):
        start, title = indices[i]
        end, _ = indices[i + 1]
        # Skip the header line itself
        header_line_end = stdout.find("\n", start)
        body_start = header_line_end + 1 if header_line_end != -1 else start
        content = stdout[body_start:end].strip()
        # Normalize title to our known headers if possible
        for h in SECTION_HEADERS:
            if h.lower() == title.lower():
                # Format the content for better readability
                formatted_content = format_pandas_output(content)
                
                # Remove markdown code blocks from the content
                formatted_content = re.sub(r'```[a-zA-Z]*\n', '', formatted_content)  # Remove ```python, ```, etc.
                formatted_content = re.sub(r'\n```\n', '\n', formatted_content)  # Remove closing ```
                formatted_content = re.sub(r'^```\n', '', formatted_content)  # Remove opening ``` at start
                formatted_content = re.sub(r'\n```$', '', formatted_content)  # Remove closing ``` at end
                formatted_content = re.sub(r'^```$', '', formatted_content)  # Remove standalone ```
                formatted_content = re.sub(r'```$', '', formatted_content)  # Remove ``` at end of line
                
                parts[h] = formatted_content.strip()
                print(f"DEBUG: parse_sections - assigned '{title}' to '{h}' with {len(formatted_content)} chars")
                break
    return parts


def _fallback_script(df: pd.DataFrame, topic: str = None, time_budget_min: int = 5) -> str:
    """Return a simple, deterministic EDA script with required sections.

    Uses pandas/numpy/matplotlib/seaborn only, avoids plt.show/close, and limits plots.
    """
    return (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "print('### SECTION: Dataset Summary')\n"
        "print('Shape:', df.shape)\n"
        "print('Dtypes:')\n"
        "print(df.dtypes)\n"
        "print('\\nMissing % by column:')\n"
        "print((df.isnull().mean()*100).round(2))\n\n"
        "num_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n"
        "cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()\n\n"
        "print('### SECTION: Visualizations')\n"
        "plotted = 0\n"
        "# Up to 3 numeric histograms\n"
        "for col in num_cols[:3]:\n"
        "    fig, ax = plt.subplots(figsize=(6,3))\n"
        "    ax.hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='white')\n"
        "    ax.set_title(f'Distribution of {col}')\n"
        "    ax.set_xlabel(col)\n"
        "    ax.set_ylabel('Frequency')\n"
        "    plotted += 1\n"
        "# One categorical countplot\n"
        "if cat_cols:\n"
        "    col = cat_cols[0]\n"
        "    vc = df[col].astype(str).value_counts().head(12)\n"
        "    fig, ax = plt.subplots(figsize=(6,3))\n"
        "    sns.barplot(x=vc.index, y=vc.values, ax=ax)\n"
        "    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')\n"
        "    ax.set_title(f'Counts of {col} (top 12)')\n"
        "    ax.set_xlabel(col)\n"
        "    ax.set_ylabel('Count')\n\n"
        "print('### SECTION: Statistical Insights')\n"
        "if num_cols:\n"
        "    desc = df[num_cols].describe().T\n"
        "    print(desc)\n"
        "    if len(num_cols) >= 2:\n"
        "        corr = df[num_cols].corr(numeric_only=True)\n"
        "        fig, ax = plt.subplots(figsize=(min(10, 0.6*corr.shape[1]+3), min(8, 0.6*corr.shape[0]+3)))\n"
        "        cax = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)\n"
        "        ax.set_xticks(range(corr.shape[1]))\n"
        "        ax.set_xticklabels(corr.columns, rotation=90)\n"
        "        ax.set_yticks(range(corr.shape[0]))\n"
        "        ax.set_yticklabels(corr.index)\n"
        "        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)\n"
        "        ax.set_title('Correlation heatmap')\n\n"
        "print('### SECTION: ML Results')\n"
        "print('Skipped in fallback to ensure robustness.')\n\n"
        "print('### SECTION: Final Explanation')\n"
        "print('This fallback analysis summarizes structure, missingness, basic distributions, and correlations to provide quick insights when the autonomous agent encounters an error.')\n"
    )


def _last_resort_sections(df: pd.DataFrame) -> Dict[str, str]:
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing = (df.isnull().mean() * 100).round(2).to_string()
    summary = [
        f"Shape: {df.shape}",
        "Dtypes:",
        pd.Series(schema).to_string(),
        "",
        "Missing % by column:",
        missing,
    ]
    sections = {h: "" for h in SECTION_HEADERS}
    sections["Dataset Summary"] = "\n".join(summary)
    sections["Final Explanation"] = (
        "Minimal summary generated due to repeated execution errors."
    )
    return sections


def sanitize_code(code: str, df: pd.DataFrame = None, time_budget_min: int = None) -> str:
    """Simplified sanitizer that only does essential fixes without breaking the code."""
    # Remove plt.show/close to allow capture
    code = re.sub(r"plt\s*\.\s*show\s*\(\s*\)", "", code)
    code = re.sub(r"plt\s*\.\s*close\s*\(\s*(['\"])all\1\s*\)", "", code)

    # Remove import streamlit statements since we provide a mock
    code = re.sub(r"import\s+streamlit\s*", "", code)
    code = re.sub(r"from\s+streamlit\s+import\s+.*\n", "", code)
    
    # Remove markdown code blocks
    code = re.sub(r'```[a-zA-Z]*\n', '', code)
    code = re.sub(r'\n```\n', '\n', code)
    
    # Fix seaborn boxplot issues with sampling - ensure sufficient data per category
    # This regex is too aggressive and breaks indentation, so we'll handle it differently
    # code = re.sub(
    #     r"sns\.boxplot\(data=([^,]+),\s*x='([^']+)',\s*y='([^']+)'\)",
    #     r"# Filter data to ensure sufficient points per category for boxplot\n    boxplot_data = \1.groupby('\2').filter(lambda x: len(x) >= 3)\n    if len(boxplot_data) > 0:\n        sns.boxplot(data=boxplot_data, x='\2', y='\3')\n    else:\n        plt.text(0.5, 0.5, 'Boxplot skipped - insufficient data per category', ha='center', va='center', transform=plt.gca().transAxes)",
    #     code
    # )
    
    # Add try/except around problematic seaborn operations to prevent crashes
    # This regex is also too aggressive and breaks indentation, so we'll handle it differently
    # code = re.sub(
    #     r"(sns\.boxplot\([^)]+\))",
    #     r"try:\n    \1\nexcept (ValueError, IndexError) as e:\n    print(f'Boxplot skipped due to data issue: {e}')\n    plt.text(0.5, 0.5, 'Boxplot skipped - insufficient data', ha='center', va='center', transform=plt.gca().transAxes)",
    #     code
    # )

    return code


def run_autonomous_session(
    df: pd.DataFrame,
    goal: str,
    time_budget_min: int,
    model: str,
    timeout_seconds: int,
    topic: Optional[str] = None,
    hide_on_error: bool = True,
) -> Dict[str, object]:
    schema, missing_pct, distinct_counts = compute_dataset_info(df)

    # bound LLM codegen time to 20% of budget (min 5s, max budget-10s)
    llm_timeout = max(5, min(int(0.2 * timeout_seconds), max(5, timeout_seconds - 10)))
    
    print(f"DEBUG: Attempting LLM call with timeout {llm_timeout}s")
    try:
        code_text = llm_autonomous_eda_code(
            schema,
            missing_pct,
            distinct_counts,
            goal,
            time_budget_min,
            model=model,
            topic=topic,
            request_timeout=llm_timeout,
        )
        print(f"DEBUG: LLM call successful, response length: {len(code_text)}")
        original_code = extract_code(code_text)
        print(f"DEBUG: Extracted original code length: {len(original_code)}")
        code = sanitize_code(original_code, df, time_budget_min)
        print(f"DEBUG: Sanitized code length: {len(code)}")
        
        # Try to compile the code to catch any syntax errors
        try:
            compile(code, "<string>", "exec")
            print(f"DEBUG: Code compiles successfully")
        except SyntaxError as se:
            print(f"DEBUG: Syntax error: {se}")
            # Skip aggressive syntax fixing to avoid breaking LLM code
            pass
        
        # Execute the code
        print(f"DEBUG: Executing code with timeout {timeout_seconds}s")
        result = execute_code(code, df, timeout_seconds)
        
        if result.get("error"):
            print(f"DEBUG: Execution error: {result.get('error')}")
            # Use fallback if execution fails
            fb_code = _fallback_script(df, topic, time_budget_min)
            fb_code = sanitize_code(fb_code, df, time_budget_min)
            fb_res = execute_code(fb_code, df, timeout_seconds=max(30, timeout_seconds // 2))
            if fb_res.get("error"):
                print(f"DEBUG: Fallback also failed: {fb_res.get('error')}")
                # Use last resort
                sections = _last_resort_sections(df)
                figures = []
                dataframes = []
            else:
                sections = parse_sections(fb_res.get("stdout", ""))
                figures = fb_res.get("figures", [])
                dataframes = fb_res.get("dataframes", [])
        else:
            sections = parse_sections(result.get("stdout", ""))
            figures = result.get("figures", [])
            dataframes = result.get("dataframes", [])
            
    except Exception as e:
        print(f"DEBUG: LLM call failed: {e}")
        # Use fallback
        fb_code = _fallback_script(df, topic, time_budget_min)
        fb_code = sanitize_code(fb_code, df, time_budget_min)
        fb_res = execute_code(fb_code, df, timeout_seconds=max(30, timeout_seconds // 2))
        if fb_res.get("error"):
            sections = _last_resort_sections(df)
            figures = []
            dataframes = []
        else:
            sections = parse_sections(fb_res.get("stdout", ""))
            figures = fb_res.get("figures", [])
            dataframes = fb_res.get("dataframes", [])
    
    # Generate key findings if we have dataframes
    if dataframes and not result.get("error"):
        print(f"DEBUG: Generating key findings from {len(dataframes)} data tables")
        try:
            key_findings = llm_generate_key_findings(dataframes, goal, model)
            print(f"DEBUG: Generated key findings: {len(key_findings)} characters")
            
            # Update the Final Explanation section with key findings
            if "Final Explanation" in sections:
                existing_content = sections["Final Explanation"]
                if existing_content.strip():
                    if "Key Findings:" not in existing_content:
                        sections["Final Explanation"] = f"{existing_content}\n\nKey Findings:\n{key_findings}"
                    else:
                        sections["Final Explanation"] = existing_content
                else:
                    sections["Final Explanation"] = f"Key Findings:\n{key_findings}"
            else:
                sections["Final Explanation"] = f"Key Findings:\n{key_findings}"
            
            # Clean up duplicate sentences in Final Explanation
            if "Final Explanation" in sections:
                sections["Final Explanation"] = _remove_duplicate_sentences(sections["Final Explanation"])
        except Exception as e:
            print(f"DEBUG: Failed to generate key findings: {e}")
    
    return {
        "sections": sections,
        "figures": figures,
        "dataframes": dataframes,
        "code": code if 'code' in locals() else "",
        "error": result.get("error") if 'result' in locals() else None
    }

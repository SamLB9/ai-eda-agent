from __future__ import annotations

import re
from typing import List


def explain_results(stdout: str, figures: List[str], step: str) -> str:
    """Produce a short 3–5 sentence explanation of EDA results.

    - Summarizes key numeric output (e.g., top nulls, correlations, group stats)
    - Mentions whether a figure was produced
    - Links back to the user's goal via the step context (heuristic)
    """
    text = stdout or ""
    step_l = (step or "").strip().lower()

    sentences: List[str] = []

    # 1) Opening sentence: what we did
    opening = _opening_sentence(step)
    if opening:
        sentences.append(opening)

    # 2) Numeric summary (heuristic parsing by known markers)
    numeric_bits: List[str] = []

    # Missing values: top counts and top pct
    numeric_bits.extend(_summarize_missing(text))

    # Class balance (counts/pct)
    numeric_bits.extend(_summarize_class_balance(text))

    # Target rate by category
    numeric_bits.extend(_summarize_target_rates(text))

    # Numeric vs target correlations
    numeric_bits.extend(_summarize_numeric_vs_target_corrs(text))

    # Generic series-like top entries (value counts, correlations series)
    if not numeric_bits:
        numeric_bits.extend(_summarize_generic_series(text))

    if numeric_bits:
        sentences.append("; ".join(numeric_bits))

    # 3) Figure mention
    fig_sentence = _figure_sentence(figures, step_l)
    if fig_sentence:
        sentences.append(fig_sentence)

    # 4) Link back to goal via step context
    goal_sentence = _goal_link_sentence(step_l)
    if goal_sentence:
        sentences.append(goal_sentence)

    # Keep it concise: 3–5 sentences
    if len(sentences) < 3:
        # Add generic closing if needed
        sentences.append("These findings provide a quick, actionable snapshot to guide next EDA actions.")
    return " ".join(sentences[:5])


# -------------------- Helpers --------------------

def _opening_sentence(step: str) -> str:
    s = (step or "").strip()
    if not s:
        return "Executed the requested EDA step."
    return f"Completed step: {s}."


def _summarize_missing(text: str) -> List[str]:
    bits: List[str] = []
    if "Missing values by column:" in text:
        section = text.split("Missing values by column:", 1)[1]
        section = section.split("Missing percent by column:", 1)[0] if "Missing percent by column:" in section else section
        top_counts = _parse_series_like(section, top_k=3)
        if top_counts:
            parts = [f"{name} ({value})" for name, value in top_counts]
            bits.append("Top null counts: " + ", ".join(parts))
    if "Missing percent by column:" in text:
        section = text.split("Missing percent by column:", 1)[1]
        top_pct = _parse_series_like(section, top_k=3)
        if top_pct:
            parts = [f"{name} ({value}%)" for name, value in top_pct]
            bits.append("Top null %: " + ", ".join(parts))
    return bits


def _summarize_class_balance(text: str) -> List[str]:
    bits: List[str] = []
    if "Class counts:" in text:
        section = text.split("Class counts:", 1)[1]
        section = section.split("Class percentage:", 1)[0] if "Class percentage:" in section else section
        top_counts = _parse_series_like(section, top_k=3)
        if top_counts:
            parts = [f"{name}={value}" for name, value in top_counts]
            bits.append("Class counts: " + ", ".join(parts))
    if "Class percentage:" in text:
        section = text.split("Class percentage:", 1)[1]
        top_pct = _parse_series_like(section, top_k=3)
        if top_pct:
            parts = [f"{name}={value}%" for name, value in top_pct]
            bits.append("Class %: " + ", ".join(parts))
    return bits


def _summarize_target_rates(text: str) -> List[str]:
    bits: List[str] = []
    # Look for lines like: "Target rate by Sex:" followed by series entries
    for m in re.finditer(r"Target rate by\s+(.+?):\s*$", text, flags=re.MULTILINE):
        col = m.group(1).strip()
        section = text[m.end() :]
        top_rates = _parse_series_like(section, top_k=3)
        if top_rates:
            parts = [f"{name}={value}" for name, value in top_rates]
            bits.append(f"Top target rates by {col}: " + ", ".join(parts))
            break
    return bits


def _summarize_numeric_vs_target_corrs(text: str) -> List[str]:
    # Lines like: "Numeric vs target correlation: age: 0.321"
    matches = re.findall(r"Numeric vs target correlation:\s*([^:]+):\s*([+-]?(?:\d+\.\d+|\d+))", text)
    if not matches:
        return []
    pairs = [(name.strip(), float(val)) for name, val in matches]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = [f"{name}={val:.3f}" for name, val in pairs[:3]]
    return ["Top numeric vs target correlations: " + ", ".join(top)]


def _summarize_generic_series(text: str) -> List[str]:
    # Try to parse first series-like block for a quick summary (e.g., value_counts or correlation series)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Skip pandas footer lines
    cleaned = [ln for ln in lines if not ln.lstrip().startswith(("Name:", "dtype:"))]
    series = _parse_series_like("\n".join(cleaned), top_k=3)
    if series:
        parts = [f"{name}={value}" for name, value in series]
        return ["Top entries: " + ", ".join(parts)]
    return []


def _parse_series_like(block: str, top_k: int = 3):
    # Parse lines like: "label    123" or "label    12.3"; stop when header/footer lines appear
    results = []
    for ln in block.splitlines():
        s = ln.strip()
        if not s or s.startswith(("Name:", "dtype:")):
            continue
        m = re.match(r"^(.+?)\s+([+-]?(?:\d+\.\d+|\d+))\s*$", s)
        if m:
            name = m.group(1).strip()
            val = m.group(2).strip()
            # Keep numeric as string to avoid formatting changes; for percentages caller will add '%'
            results.append((name, val))
        if len(results) >= top_k:
            break
    return results


def _figure_sentence(figures: List[str], step_l: str) -> str:
    if not figures:
        return "No figure was produced."
    desc = "a matplotlib figure"
    if "distribution" in step_l or "histogram" in step_l:
        desc = "a distribution histogram"
    elif "correlation" in step_l or "heatmap" in step_l:
        desc = "a correlation heatmap"
    elif "trend" in step_l or "time" in step_l:
        desc = "a time series plot"
    elif "target class" in step_l or "by target" in step_l:
        desc = "overlaid histograms by target class"
    if len(figures) == 1:
        return f"A figure was produced: {desc}."
    return f"{len(figures)} figures were produced, including {desc}."


def _goal_link_sentence(step_l: str) -> str:
    # Heuristic tie-back using step keywords
    if any(k in step_l for k in ("driver", "influenc", "impact", "factor", "surviv", "churn", "target")):
        return "This helps assess relationships with the target and potential drivers relevant to your goal."
    if any(k in step_l for k in ("quality", "clean", "missing", "null")):
        return "This highlights data quality issues to address before modeling, supporting your goal."
    return "This supports your goal by surfacing key patterns to guide the next steps." 
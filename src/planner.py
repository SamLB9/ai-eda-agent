from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Union


SchemaInput = Union[
    Dict[str, str],
    Iterable[Tuple[str, str]],
    Iterable[Dict[str, str]],
]


@dataclass
class ColumnGroups:
    numeric: List[str]
    categorical: List[str]
    boolean: List[str]
    datetime: List[str]
    text: List[str]
    unknown: List[str]


class Planner:
    """Generate a simple, deterministic EDA plan.

    Parameters
    ----------
    schema:
        A list/dict describing columns and their types. Supported formats:
        - {column_name: type}
        - [(column_name, type), ...]
        - [{"name": column_name, "type": type}, ...]
    goal:
        A short description of the analysis objective, e.g. "find drivers of survival".

    Methods
    -------
    plan() -> List[str]
        Returns a list of short EDA instructions.
    """

    def __init__(self, schema: SchemaInput, goal: str) -> None:
        self.goal_original: str = goal or ""
        self.goal: str = (goal or "").strip().lower()
        self.schema_map: Dict[str, str] = self._normalize_schema(schema)
        self.groups: ColumnGroups = self._categorize_columns(self.schema_map)

    # -------------------- Public API --------------------
    def plan(self) -> List[str]:  # noqa: D401
        """Return a deterministic list of short EDA steps as strings."""
        steps: List[str] = []

        # Always useful, general overview
        steps.append("Preview dataset shape and column dtypes")
        steps.append("Summarize missing values by column")
        steps.append("Identify duplicated rows and potential unique keys")

        # Numeric-focused steps
        if self.groups.numeric:
            steps.append("Summary statistics for numeric columns")
            steps.append("Plot distributions (histogram/boxplot) for numeric columns")
            if len(self.groups.numeric) >= 2:
                steps.append("Check pairwise correlations among numeric columns")

        # Categorical/boolean-focused steps
        if self.groups.categorical or self.groups.boolean:
            steps.append("Value counts for categorical and boolean columns")
            steps.append("Group by categorical columns to compare summary metrics")

        # Datetime-focused steps
        if self.groups.datetime:
            steps.append("Plot trends over time for datetime columns")

        # Text-focused steps
        if self.groups.text:
            steps.append("Inspect text length distribution and frequent tokens for text columns")

        # Goal-aware additions (deterministic keyword checks)
        steps.extend(self._goal_specific_steps())

        # Remove accidental duplicates while preserving order
        steps = self._deduplicate_preserve_order(steps)
        return steps

    # -------------------- Internal helpers --------------------
    def _normalize_schema(self, schema: SchemaInput) -> Dict[str, str]:
        """Normalize supported schema formats into a name->type mapping."""
        if schema is None:
            return {}

        if isinstance(schema, dict):
            # Already a mapping
            return {str(k): str(v) for k, v in schema.items()}

        # Iterable of tuples or dicts
        result: Dict[str, str] = {}
        for item in schema:  # type: ignore[assignment]
            if isinstance(item, tuple) and len(item) >= 2:
                name, dtype = item[0], item[1]
                result[str(name)] = str(dtype)
            elif isinstance(item, dict):
                name = item.get("name")
                dtype = item.get("type")
                if name is not None and dtype is not None:
                    result[str(name)] = str(dtype)
        return result

    def _categorize_columns(self, mapping: Dict[str, str]) -> ColumnGroups:
        numeric: List[str] = []
        categorical: List[str] = []
        boolean: List[str] = []
        datetime_cols: List[str] = []
        text: List[str] = []
        unknown: List[str] = []

        for col, dtype in mapping.items():
            inferred = self._infer_type(str(dtype))
            if inferred == "numeric":
                numeric.append(col)
            elif inferred == "categorical":
                categorical.append(col)
            elif inferred == "boolean":
                boolean.append(col)
            elif inferred == "datetime":
                datetime_cols.append(col)
            elif inferred == "text":
                text.append(col)
            else:
                unknown.append(col)

        return ColumnGroups(
            numeric=numeric,
            categorical=categorical,
            boolean=boolean,
            datetime=datetime_cols,
            text=text,
            unknown=unknown,
        )

    def _infer_type(self, raw_dtype: str) -> str:
        t = raw_dtype.strip().lower()

        # Datetime first (specific tokens)
        if any(k in t for k in ("datetime", "timestamp", "date", "time")):
            return "datetime"

        # Boolean
        if "bool" in t:
            return "boolean"

        # Numeric tokens (covers pandas, SQL-ish, and general terms)
        numeric_tokens = (
            "int",
            "float",
            "double",
            "decimal",
            "numeric",
            "number",
            "real",
        )
        if any(k in t for k in numeric_tokens):
            return "numeric"

        # Text vs categorical heuristics
        # - "object", "string", SQL char types often behave like categorical in EDA
        # - explicit "text" treated as text
        if "text" in t:
            return "text"
        if any(k in t for k in ("object", "string", "varchar", "char", "category", "categorical")):
            return "categorical"

        return "unknown"

    def _goal_specific_steps(self) -> List[str]:
        g = self.goal
        steps: List[str] = []

        if any(k in g for k in ("driver", "influenc", "impact", "factor")):
            steps.append("Bivariate analysis: each feature vs target")
            if self.groups.numeric:
                steps.append("Assess numeric-feature correlation with target (e.g., Pearson/Spearman)")
            if self.groups.categorical or self.groups.boolean:
                steps.append("Compare target rates across categorical groups")

        if any(k in g for k in ("surviv", "churn", "default", "conversion", "retention")):
            steps.append("Check class balance of the target variable")
            steps.append("Compare feature distributions by target class")

        return steps

    def _deduplicate_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for s in items:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result 
"""Tools that change workbench UI state rather than the dataset.

The feature selection lives in the frontend, so these tools run inside the
execution agent against the UI state sent with the chat turn, and their result
is returned to the frontend in ``final_state``.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Optional, Sequence

SET_MODEL_FEATURES = "set_model_features"

_PATTERN_HELP = "Supports * wildcards, e.g. '*_lag*' or 'T*'."

UI_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": SET_MODEL_FEATURES,
            "description": (
                "Change which columns are selected in the Feature Selection step: the "
                "target variable (y) and the input features (X) used for training. "
                "For requests like 'use only X and Y' or 'select the new features and "
                "unselect the old ones', pass the complete new list in `select` "
                "(e.g. select=['*_lag*', '*_rolling*']). Use `add` / `remove` only for "
                "small edits to the current selection. Does not modify the dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "description": "Column to predict. Omit to keep the current target.",
                    },
                    "select": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Replace the selected features with exactly these columns. {_PATTERN_HELP}",
                    },
                    "add": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Columns to add to the current selection. {_PATTERN_HELP}",
                    },
                    "remove": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Columns to remove from the current selection. {_PATTERN_HELP}",
                    },
                },
            },
        },
    },
]


def _expand(patterns: Optional[Sequence[str]], columns: Sequence[str]) -> tuple[list[str], list[str]]:
    """Resolve names and wildcards against ``columns``; return (matches, unknown)."""
    by_lower = {c.lower(): c for c in columns}
    matched: list[str] = []
    unknown: list[str] = []
    for raw in patterns or []:
        p = (raw or "").strip()
        if not p:
            continue
        if any(ch in p for ch in "*?["):
            hits = [c for c in columns if fnmatch.fnmatchcase(c, p)] or [
                c for c in columns if fnmatch.fnmatch(c.lower(), p.lower())
            ]
        else:
            hits = [p] if p in columns else ([by_lower[p.lower()]] if p.lower() in by_lower else [])
        if hits:
            matched.extend(h for h in hits if h not in matched)
        else:
            unknown.append(p)
    return matched, unknown


def apply_feature_selection(
    *,
    columns: Sequence[str],
    numeric_columns: Sequence[str],
    current_features: Sequence[str],
    current_target: Optional[str],
    target_column: Optional[str] = None,
    select: Optional[Sequence[str]] = None,
    add: Optional[Sequence[str]] = None,
    remove: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    target = current_target if current_target in columns else None
    if target_column:
        found, _ = _expand([target_column], columns)
        if len(found) != 1:
            raise ValueError(f"Target column {target_column!r} not found. Available columns: {list(columns)}")
        target = found[0]

    selected, unknown_select = _expand(select, columns) if select is not None else (
        [c for c in current_features if c in columns], []
    )
    added, unknown_add = _expand(add, columns)
    removed, unknown_remove = _expand(remove, columns)
    unknown = unknown_select + unknown_add + unknown_remove
    matched_any = (select is not None and selected) or added or removed or target_column
    if unknown and not matched_any:
        raise ValueError(f"No such columns: {unknown}. Available columns: {list(columns)}")

    before = set(selected) if select is None else set(c for c in current_features if c in columns)
    chosen = set(selected) | set(added)
    chosen -= set(removed)
    chosen.discard(target)
    numeric = set(numeric_columns)
    non_numeric = sorted(c for c in chosen if c not in numeric)
    chosen -= set(non_numeric)

    features = [c for c in columns if c in chosen]
    result: dict[str, Any] = {
        "target_column": target,
        "selected_features": features,
        "now_selected": [c for c in features if c not in before],
        "now_unselected": [c for c in columns if c in before and c not in chosen],
        "unselected_numeric_columns": [
            c for c in columns if c in numeric and c not in chosen and c != target
        ],
        "ignored_unknown": unknown,
        "ignored_non_numeric": non_numeric,
    }
    if not features:
        result["warning"] = (
            "No input features are selected now, so training is impossible. If the "
            "user asked for other columns, call set_model_features again with the "
            "full list in `select`."
        )
    return result

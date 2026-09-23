"""Grounding shared by the chat agents.

Gives the model the facts it would otherwise guess at: the real columns of the
active dataset version, the models already trained on its lineage, what the
user has selected in the UI, and the recent conversation.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from backend.database import AsyncSessionLocal
from backend.mcp_server.eda_tools import _resolve_storage_path
from backend.mcp_server.model_registry import lineage_models
from backend.models import DatasetVersion

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 1500
MAX_COLUMNS_LISTED = 300
MAX_MODELS_LISTED = 5


@dataclass
class DatasetContext:
    version_id: str
    action: str
    n_rows: int
    columns: list[tuple[str, str, bool]]  # (name, dtype, is_numeric)
    models: list[dict[str, Any]] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c[0] for c in self.columns]

    @property
    def numeric_columns(self) -> list[str]:
        return [c[0] for c in self.columns if c[2]]


def _is_numeric(dtype: pa.DataType) -> bool:
    return pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)


async def load_dataset_context(dataset_version_id: Optional[str]) -> Optional[DatasetContext]:
    """Read the version's schema from parquet metadata (no row data) plus its models."""
    if not dataset_version_id:
        return None
    try:
        async with AsyncSessionLocal() as db:
            version = await db.get(DatasetVersion, uuid.UUID(dataset_version_id))
            if version is None:
                return None
            parquet = pq.ParquetFile(_resolve_storage_path(version.file_path))
            columns = [
                (f.name, str(f.type), _is_numeric(f.type))
                for f in parquet.schema_arrow
                if not f.name.startswith("__index_level")
            ]
            models = await lineage_models(db, dataset_version_id)
        return DatasetContext(
            version_id=str(version.id),
            action=version.action_performed,
            n_rows=parquet.metadata.num_rows,
            columns=columns,
            models=models[:MAX_MODELS_LISTED],
        )
    except Exception:  # noqa: BLE001 — a missing context must not break the chat
        logger.warning("Could not load dataset context for %s", dataset_version_id, exc_info=True)
        return None


def _fmt_metric(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) else "n/a"


def render_dataset_context(ctx: Optional[DatasetContext]) -> str:
    if ctx is None:
        return "DATASET: schema unavailable — call get_column_statistics to discover the columns."

    lines = [
        f"ACTIVE DATASET VERSION: {ctx.version_id} (created by: {ctx.action}) — "
        f"{ctx.n_rows:,} rows × {len(ctx.columns)} columns",
        "COLUMNS (authoritative — these are the only column names that exist, spelled exactly):",
    ]
    lines += [f"  - {name} ({dtype})" for name, dtype, _ in ctx.columns[:MAX_COLUMNS_LISTED]]
    if len(ctx.columns) > MAX_COLUMNS_LISTED:
        lines.append(f"  … and {len(ctx.columns) - MAX_COLUMNS_LISTED} more")
    lines.append(f"NUMERIC COLUMNS: {', '.join(ctx.numeric_columns) or 'none'}")

    if ctx.models:
        lines.append("TRAINED MODELS on this lineage (newest first; plot tools use the newest when model_id is omitted):")
        for m in ctx.models:
            lines.append(
                f"  - {m['model_id']}: {m.get('algorithm')} predicting {m.get('target_column')} "
                f"from {m.get('feature_columns')} — R² {_fmt_metric(m.get('r2_score'))}, "
                f"RMSE {_fmt_metric(m.get('rmse'))}"
            )
    else:
        lines.append("TRAINED MODELS: none yet on this lineage.")
    return "\n".join(lines)


def parse_ui_state(ui_context: Optional[str]) -> dict[str, Any]:
    if not ui_context:
        return {}
    try:
        value = json.loads(ui_context)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def render_ui_state(ui: dict[str, Any]) -> str:
    target = ui.get("targetVariable")
    features = ui.get("selectedFeatures") or []
    metrics = ui.get("modelMetrics") or {}
    lines = [
        "UI STATE (what the user has selected in the workbench):",
        f"  target variable: {target or 'not selected'}",
        f"  selected features: {features if features else 'none selected'}",
    ]
    if metrics.get("model_id"):
        lines.append(
            f"  model shown in UI: {metrics['model_id']} ({metrics.get('algorithm')}, "
            f"R² {_fmt_metric(metrics.get('r2_score'))}, RMSE {_fmt_metric(metrics.get('rmse'))})"
        )
    if ui.get("activePlot"):
        lines.append(f"  plot on canvas: {ui['activePlot']}")
    return "\n".join(lines)


def _turn_field(turn: Any, name: str) -> Any:
    return turn.get(name) if isinstance(turn, dict) else getattr(turn, name, None)


def _recent_turns(history: Optional[Iterable[Any]]) -> list[Any]:
    return [
        t for t in list(history or [])[-MAX_HISTORY_MESSAGES:]
        if _turn_field(t, "role") in ("user", "assistant") and _turn_field(t, "content")
    ]


def history_messages(history: Optional[Iterable[Any]]) -> list[dict[str, str]]:
    """Turn prior chat turns into OpenAI messages, newest ``MAX_HISTORY_MESSAGES`` only."""
    out: list[dict[str, str]] = []
    for turn in _recent_turns(history):
        content = _turn_field(turn, "content")
        text = content if len(content) <= MAX_HISTORY_CHARS else content[:MAX_HISTORY_CHARS] + " …[truncated]"
        out.append({"role": _turn_field(turn, "role"), "content": text})
    return out


def render_tool_history(history: Optional[Iterable[Any]]) -> str:
    """Which tools ran in earlier turns, for the system prompt.

    Kept out of the assistant messages themselves, which small models imitate.
    """
    lines = []
    assistant_turns = [t for t in _recent_turns(history) if _turn_field(t, "role") == "assistant"]
    for n, turn in enumerate(assistant_turns, 1):
        tools = _turn_field(turn, "tools")
        if tools:
            lines.append(f"  reply {n}: {', '.join(tools)}")
    return "TOOLS RUN IN EARLIER REPLIES (oldest first):\n" + "\n".join(lines) if lines else ""

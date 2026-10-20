"""Downloads: a dataset version (CSV / Parquet), a trained model, the pipeline recipe and an HTML report."""

from __future__ import annotations

import html
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import pyarrow.parquet as pq
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import User, get_current_user
from backend.database import AsyncSessionLocal
from backend.mcp_server.eda_tools import _load_dataframe, _resolve_storage_path
from backend.mcp_server.model_registry import lineage_ids, lineage_models
from backend.models import Dataset, DatasetVersion, TrainedModel
from backend.services.profile import cached_profile
from backend.services.projects import owned_version
from backend.storage import get_storage

router = APIRouter(tags=["exports"])

_CSV_BATCH_ROWS = 50_000
RECIPE_FORMAT = 1


def _attachment(name: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{name}"'}


def _stem(dataset: Optional[Dataset], version: DatasetVersion) -> str:
    base = Path(dataset.file_name).stem if dataset and dataset.file_name else "dataset"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "dataset"
    return f"{base}_{str(version.id)[:8]}"


async def _versions_root_first(session: AsyncSession, version: DatasetVersion) -> list[DatasetVersion]:
    ids = await lineage_ids(session, str(version.id))
    chain = [await session.get(DatasetVersion, uuid.UUID(i)) for i in reversed(ids)]
    return [v for v in chain if v is not None]


def _csv_chunks(path: Path) -> Iterator[bytes]:
    parquet = pq.ParquetFile(path)
    header = True
    for batch in parquet.iter_batches(batch_size=_CSV_BATCH_ROWS):
        frame = batch.to_pandas()
        frame = frame[[c for c in frame.columns if not str(c).startswith("__index_level")]]
        yield frame.to_csv(index=False, header=header).encode()
        header = False
    if header:
        yield ",".join(f.name for f in parquet.schema_arrow if not f.name.startswith("__index_level")).encode() + b"\n"


@router.get("/datasets/versions/{dataset_version_id}/download")
async def download_version(
    dataset_version_id: str,
    format: Literal["csv", "parquet"] = Query("csv"),
    user: User = Depends(get_current_user),
) -> Response:
    """Every row of the version. CSV is streamed in batches, so large datasets never sit in memory as text."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
        dataset = await session.get(Dataset, version.dataset_id)
    try:
        path = await run_in_threadpool(_resolve_storage_path, version.file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    name = _stem(dataset, version)
    if format == "parquet":
        return FileResponse(path, media_type="application/vnd.apache.parquet", filename=f"{name}.parquet")
    return StreamingResponse(_csv_chunks(path), media_type="text/csv", headers=_attachment(f"{name}.csv"))


@router.get("/models/{model_id}/download")
async def download_model(model_id: str, user: User = Depends(get_current_user)) -> FileResponse:
    """The saved model (a joblib file: a dict whose ``model`` entry is the fitted estimator)."""
    try:
        model_uuid = uuid.UUID(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Model not found.") from exc
    async with AsyncSessionLocal() as session:
        row = await session.get(TrainedModel, model_uuid)
        dataset = await session.get(Dataset, row.dataset_id) if row else None
    if row is None or dataset is None or dataset.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Model not found.")
    try:
        path = await run_in_threadpool(get_storage().fetch, row.artifact_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="The model file is missing from storage.") from exc
    meta = row.meta or {}
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{meta.get('algorithm') or 'model'}_{meta.get('target_column') or ''}").strip("_")
    return FileResponse(path, media_type="application/octet-stream", filename=f"{label}_{model_id[:8]}.joblib")


async def _recipe(session: AsyncSession, version: DatasetVersion) -> dict[str, Any]:
    dataset = await session.get(Dataset, version.dataset_id)
    chain = await _versions_root_first(session, version)
    return {
        "recipe_format": RECIPE_FORMAT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": dataset.file_name if dataset else None,
        "dataset_version_id": str(version.id),
        "steps": [
            {
                "step": i,
                "action": v.action_performed,
                "parameters": v.parameters_used or {},
                "version_id": str(v.id),
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for i, v in enumerate(chain)
        ],
        "models": await lineage_models(session, str(version.id)),
    }


@router.get("/datasets/versions/{dataset_version_id}/recipe")
async def download_recipe(dataset_version_id: str, user: User = Depends(get_current_user)) -> Response:
    """Every step from the upload to this version, with its parameters, and the models trained along the way."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
        recipe = await _recipe(session, version)
        dataset = await session.get(Dataset, version.dataset_id)
    body = json.dumps(recipe, indent=2, default=str)
    return Response(body, media_type="application/json", headers=_attachment(f"{_stem(dataset, version)}_recipe.json"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return html.escape(str(value))


def _params(parameters: dict[str, Any]) -> str:
    shown = {k: v for k, v in parameters.items() if k not in ("dataset_version_id",)}
    return html.escape(json.dumps(shown, default=str)) if shown else ""


def _report_html(recipe: dict[str, Any], profile: dict[str, Any]) -> str:
    e = html.escape
    time = profile.get("time")
    target = profile.get("target")
    flagged = [c for c in profile.get("columns", []) if c["status"] != "ok"]
    rows = [
        ("Source file", e(recipe.get("source_file") or "—")),
        ("Version", f"<code>{e(recipe['dataset_version_id'])}</code>"),
        ("Size", f"{profile['row_count']:,} rows × {profile['column_count']} columns"),
        ("Complete rows", f"{profile['complete_rows']:,}"),
    ]
    if time:
        rows.append(("Time", f"{e(str(profile['time_column']))}: {e(str(time.get('start')))} → {e(str(time.get('end')))}"
                     f" · median step {_fmt(time.get('median_interval_s'))} s · {time.get('n_gaps', 0)} gaps"))
    if target:
        rows.append(("Target", f"{e(target['column'])}: {target['samples']:,} samples ({target['share'] * 100:.1f}% of rows)"))

    def table(head: list[str], body: list[list[str]]) -> str:
        if not body:
            return "<p class='muted'>None.</p>"
        th = "".join(f"<th>{h}</th>" for h in head)
        trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"

    steps = [[str(s["step"]), e(s["action"]), f"<code>{_params(s['parameters'])}</code>", e(str(s["created_at"] or ""))]
             for s in recipe["steps"]]
    models = [
        [
            f"<code>{e(m['model_id'][:8])}</code>",
            e(str(m.get("algorithm"))),
            e(str(m.get("target_column"))),
            str(len(m.get("feature_columns") or [])),
            _fmt((m.get("cv") or {}).get("r2")),
            _fmt(m.get("r2_score")),
            _fmt(m.get("rmse")),
            e(str(m.get("trained_at") or "")[:19].replace("T", " ")),
        ]
        for m in recipe["models"]
    ]
    issues = [[e(c["name"]), e(c["kind"]), e(c["status"]), e("; ".join(c["issues"]))] for c in flagged]
    summary = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Soft sensor report · {e(recipe.get('source_file') or '')}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;color:#1d2433;max-width:980px;margin:32px auto;padding:0 20px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 8px;border-bottom:1px solid #dde2ea;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #eef1f5;vertical-align:top}}
thead th{{background:#f4f6f9;font-weight:600}}code{{font:12px ui-monospace,monospace;word-break:break-all}}.muted{{color:#6b7385}}
</style></head><body>
<h1>Soft sensor report</h1>
<p class="muted">Generated {e(recipe['generated_at'][:19].replace('T', ' '))} UTC</p>
<h2>Data</h2><table><tbody>{summary}</tbody></table>
<h2>Pipeline ({len(steps)} steps)</h2>{table(['#', 'Action', 'Parameters', 'When'], steps)}
<h2>Models</h2>{table(['Model', 'Algorithm', 'Target', 'Features', 'CV R²', 'Test R²', 'Test RMSE', 'Trained'], models)}
<h2>Data quality findings</h2>{table(['Column', 'Type', 'Status', 'Findings'], issues)}
</body></html>"""


@router.get("/datasets/versions/{dataset_version_id}/report")
async def download_report(
    dataset_version_id: str,
    target: Optional[str] = Query(None),
    time_column: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
) -> Response:
    """A self-contained HTML report: data summary, pipeline steps, models and quality findings."""
    async with AsyncSessionLocal() as session:
        version = await owned_version(session, dataset_version_id, user)
        recipe = await _recipe(session, version)
        dataset = await session.get(Dataset, version.dataset_id)
    try:
        profile = await run_in_threadpool(
            cached_profile,
            (str(version.id), target, time_column),
            lambda: _load_dataframe(version),
            target=target,
            time_column=time_column,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        _report_html(recipe, profile),
        media_type="text/html",
        headers=_attachment(f"{_stem(dataset, version)}_report.html"),
    )

"""Deployments: compile a trained model's lineage into a live predictor and serve it."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import secrets
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import auth_required
from backend.deployment import monitor, runtime
from backend.deployment.replay import Compiled, VersionInfo, compile_lineage
from backend.mcp_server.eda_tools import _load_dataframe
from backend.mcp_server.model_registry import lineage_ids
from backend.mcp_server.modeling_tools import _load_model_artifact
from backend.models import Dataset, DatasetVersion, Deployment, DeploymentLab, DeploymentRecord, TrainedModel
from backend.storage import deployment_key, get_storage

KEY_PREFIX = "ssk_"
MAX_ROWS_PER_REQUEST = 1000
MAX_LABS_PER_REQUEST = 1000
KEEP_RECORDS = 5000
KEEP_LABS = 2000
DEFAULT_GAIN = 0.3
_LAB_MATCH_MIN_S = 60.0


def max_deployments_per_user() -> int:
    """0 means unlimited. Online, each deployment pins its dataset, so they are capped."""
    default = 3 if auth_required() else 0
    try:
        return int(os.getenv("MAX_DEPLOYMENTS_PER_USER", str(default)))
    except ValueError:
        return default


def live_rate_per_minute() -> int:
    try:
        return int(os.getenv("LIVE_RATE_LIMIT_PER_MINUTE", "120"))
    except ValueError:
        return 120


# --------------------------------------------------------------------------- lineage


async def load_chain(session: AsyncSession, dataset_version_id: str) -> list[VersionInfo]:
    """The lineage of ``dataset_version_id`` from the upload to it, each version loadable on demand."""
    ids = await lineage_ids(session, dataset_version_id)
    if not ids:
        raise ValueError(f"Dataset version not found: {dataset_version_id}")
    chain: list[VersionInfo] = []
    for vid in reversed(ids):
        version = await session.get(DatasetVersion, uuid.UUID(vid))
        assert version is not None
        params: dict[str, Any] = dict(version.parameters_used or {})
        chain.append(VersionInfo(
            id=vid, action=version.action_performed or "", params=params,
            load=lambda v=version: _load_dataframe(v),
        ))
    return chain


class _Cache:
    """A small thread-safe LRU (compiling or loading a bundle takes seconds)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def drop(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_compiled_cache = _Cache(6)
_bundle_cache = _Cache(16)
_locks: dict[str, asyncio.Lock] = {}


def clear_caches() -> None:
    _compiled_cache.clear()
    _bundle_cache.clear()


def _lock_for(deployment_id: str) -> asyncio.Lock:
    lock = _locks.get(deployment_id)
    if lock is None:
        lock = _locks[deployment_id] = asyncio.Lock()
    return lock


class Prepared:
    """A model with its compiled lineage and the data needed to build a bundle."""

    def __init__(self, row: TrainedModel, artifact: dict[str, Any], chain: list[VersionInfo], compiled: Compiled) -> None:
        self.row = row
        self.artifact = artifact
        self.chain = chain
        self.compiled = compiled


async def prepare_model(session: AsyncSession, model: TrainedModel) -> Prepared:
    key = str(model.id)
    cached = _compiled_cache.get(key)
    if cached is not None:
        return cached
    meta = dict(model.meta or {})
    training_version = meta.get("dataset_version_id")
    if not training_version:
        raise ValueError("This model does not record which dataset version it was trained on.")
    if await session.get(DatasetVersion, uuid.UUID(training_version)) is None:
        raise ValueError("The dataset version this model was trained on no longer exists.")
    chain = await load_chain(session, training_version)
    artifact = await run_in_threadpool(_load_model_artifact, model.artifact_key)
    compiled = await run_in_threadpool(
        compile_lineage, chain,
        feature_columns=list(artifact["feature_columns"]), target_column=str(artifact["target_column"]),
    )
    prepared = Prepared(model, artifact, chain, compiled)
    _compiled_cache.put(key, prepared)
    return prepared


def _num(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def plan_payload(prepared: Prepared) -> dict[str, Any]:
    compiled, artifact = prepared.compiled, prepared.artifact
    metrics = artifact.get("metrics") or {}
    cut = compiled.cut_index
    cut_title = next((s.title for s in compiled.steps if s.index == cut), "Upload")
    return {
        "model": {
            "model_id": str(artifact["model_id"]),
            "algorithm": artifact.get("algorithm"),
            "target_column": artifact.get("target_column"),
            "feature_columns": list(artifact["feature_columns"]),
            "r2_score": _num(metrics.get("r2_score")),
            "rmse": _num(metrics.get("rmse")),
            "trained_on": str(artifact.get("dataset_version_id") or ""),
        },
        "ready": not compiled.blocking,
        "inputs": [{"name": c, "dtype": compiled.input_dtypes.get(c), "is_time": c == compiled.time_column} for c in compiled.inputs],
        "time_column": compiled.time_column,
        "buffer_rows": compiled.buffer_rows,
        "cut": {"index": cut, "version_id": compiled.cut_version_id, "title": "Upload" if cut == 0 else cut_title},
        "steps": [s.as_dict() for s in compiled.steps],
        "checks": compiled.checks,
        "warnings": compiled.warnings,
        "blocking": compiled.blocking,
        "approx_features": compiled.approx_features,
    }


# --------------------------------------------------------------------------- deployments


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_api_key() -> tuple[str, str, str]:
    """(key, sha256 hash, prefix shown in lists)."""
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    return key, hash_key(key), key[:12]


def _build_bundle_file(prepared: Prepared, key: str) -> runtime.Bundle:
    training = prepared.chain[-1].load()
    cut_data = prepared.chain[prepared.compiled.cut_index].load()
    bundle = runtime.build_bundle(prepared.compiled, prepared.artifact, training, cut_data)
    get_storage().write(key, lambda path: joblib.dump(bundle, path))
    return bundle


def _spec(prepared: Prepared, bundle: runtime.Bundle) -> dict[str, Any]:
    plan = plan_payload(prepared)
    stats = {name: st.as_dict() for name, st in bundle.stats.items()}
    for item in plan["inputs"]:
        item["stats"] = stats.get(item["name"])
    plan["region"] = bundle.region.as_dict() if bundle.region else None
    plan["history_rows"] = bundle.history_rows
    return plan


async def create_deployment(
    session: AsyncSession,
    *,
    owner_id: str,
    dataset: Dataset,
    model: TrainedModel,
    name: str,
    gain: float,
) -> tuple[Deployment, str]:
    prepared = await prepare_model(session, model)
    if prepared.compiled.blocking:
        raise ValueError("This model cannot be deployed yet: " + " ".join(prepared.compiled.blocking))
    dep_id = uuid.uuid4()
    key = deployment_key(owner_id, dataset.id, dep_id)
    bundle = await run_in_threadpool(_build_bundle_file, prepared, key)
    api_key, key_hash, prefix = new_api_key()
    deployment = Deployment(
        id=dep_id, owner_id=owner_id, dataset_id=dataset.id, model_id=str(model.id), name=name,
        status="active", api_key_hash=key_hash, api_key_prefix=prefix, bundle_key=key,
        spec=_spec(prepared, bundle), state=runtime.new_state(gain),
    )
    session.add(deployment)
    await session.commit()
    _bundle_cache.put(str(dep_id), bundle)
    return deployment, api_key


async def load_bundle(deployment: Deployment) -> runtime.Bundle:
    key = str(deployment.id)
    bundle = _bundle_cache.get(key)
    if bundle is None:
        bundle = await run_in_threadpool(lambda: joblib.load(get_storage().fetch(deployment.bundle_key)))
        _bundle_cache.put(key, bundle)
    return bundle


def _stamp(value: str) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


async def ingest_rows(session: AsyncSession, deployment: Deployment, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict for live rows, store the records and the new state; one request at a time per deployment."""
    bundle = await load_bundle(deployment)
    async with _lock_for(str(deployment.id)):
        await session.refresh(deployment)
        records, state = await run_in_threadpool(runtime.predict, bundle, dict(deployment.state or {}), rows)
        for rec in records:
            if rec["status"] == runtime.IGNORED:
                continue
            session.add(DeploymentRecord(
                deployment_id=deployment.id, row_time=_stamp(rec["time"]), received_at=_stamp(rec["received_at"]),
                status=rec["status"], prediction=rec["prediction"], corrected=rec["corrected"], bias=rec["bias"],
                t2=rec["t2"], spe=rec["spe"], reasons=rec["reasons"], inputs=rec["inputs"],
            ))
        deployment.state = state
        deployment.updated_at = datetime.now(timezone.utc)
        await session.flush()
        await _trim(session, DeploymentRecord, deployment.id, KEEP_RECORDS)
        await session.commit()
    return records


async def _trim(session: AsyncSession, table, deployment_id: uuid.UUID, keep: int) -> None:
    cutoff = (
        await session.execute(
            select(table.id).where(table.deployment_id == deployment_id).order_by(table.id.desc()).offset(keep).limit(1)
        )
    ).scalar_one_or_none()
    if cutoff is not None:
        await session.execute(delete(table).where(table.deployment_id == deployment_id, table.id <= cutoff))


def _match_tolerance(times: list[datetime]) -> float:
    if len(times) < 3:
        return _LAB_MATCH_MIN_S * 10
    gaps = np.diff(np.array([t.timestamp() for t in sorted(times)]))
    gaps = gaps[gaps > 0]
    median = float(np.median(gaps)) if len(gaps) else 0.0
    return max(_LAB_MATCH_MIN_S, 3.0 * median)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


async def ingest_labs(session: AsyncSession, deployment: Deployment, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match lab results to the prediction for their sample time and update the bias, oldest sample first.

    Results come back in the order of ``samples``.
    """
    parsed: list[tuple[datetime, float, int]] = []
    results: list[dict[str, Any]] = [{} for _ in samples]
    for position, sample in enumerate(samples):
        stamp = runtime.parse_time(sample.get("time"))
        value = _num(sample.get("value"))
        if stamp is None or value is None:
            results[position] = {"time": sample.get("time"), "value": sample.get("value"), "status": "rejected",
                                 "reason": "A lab result needs the sample time and a numeric value."}
            continue
        parsed.append((stamp.to_pydatetime(), value, position))
    parsed.sort(key=lambda item: (item[0], item[2]))
    received = datetime.now(timezone.utc)
    seen: set[datetime] = set()

    async with _lock_for(str(deployment.id)):
        await session.refresh(deployment)
        state = dict(deployment.state or {})
        gain = float(state.get("gain", DEFAULT_GAIN))
        bias = float(state.get("bias") or 0.0)
        recent = (
            await session.execute(
                select(DeploymentRecord.row_time).where(DeploymentRecord.deployment_id == deployment.id)
                .order_by(DeploymentRecord.id.desc()).limit(200)
            )
        ).scalars().all()
        tolerance = _match_tolerance([_aware(t) for t in recent])
        newest = runtime.parse_time(state.get("last_time"))
        for sample_time, value, position in parsed:
            if newest is None or sample_time > newest.to_pydatetime():
                results[position] = {"time": sample_time.isoformat(), "value": value, "status": "early",
                                     "reason": "No reading up to this sample time has been predicted yet; send it again later."}
                continue
            duplicate = sample_time in seen or (
                await session.execute(
                    select(DeploymentLab.id).where(
                        DeploymentLab.deployment_id == deployment.id, DeploymentLab.sample_time == sample_time,
                    ).limit(1)
                )
            ).scalar_one_or_none() is not None
            if duplicate:
                results[position] = {"time": sample_time.isoformat(), "value": value, "status": "duplicate",
                                     "reason": "A lab result for this sample time was already received."}
                continue
            seen.add(sample_time)
            match = (
                await session.execute(
                    select(DeploymentRecord).where(
                        DeploymentRecord.deployment_id == deployment.id,
                        DeploymentRecord.row_time <= sample_time,
                    ).order_by(DeploymentRecord.row_time.desc(), DeploymentRecord.id.desc()).limit(1)
                )
            ).scalar_one_or_none()
            near = match is not None and (sample_time - _aware(match.row_time)).total_seconds() <= tolerance
            lab = DeploymentLab(deployment_id=deployment.id, sample_time=sample_time, received_at=received, value=value)
            if near and match.prediction is None:
                results[position] = {"time": sample_time.isoformat(), "value": value, "status": "unmatched",
                                     "reason": "The prediction for this sample time was suppressed, so the bias is not updated."}
            elif near:
                bias = monitor.update_bias(bias, gain, value - float(match.prediction))
                lab.record_time = match.row_time
                lab.prediction = match.prediction
                lab.corrected = match.corrected
                lab.bias_after = bias
                results[position] = {"time": sample_time.isoformat(), "value": value, "status": "matched",
                                     "prediction": match.prediction, "corrected": match.corrected, "bias_after": bias}
            else:
                results[position] = {"time": sample_time.isoformat(), "value": value, "status": "unmatched",
                                     "reason": "No prediction was made at or shortly before this sample time."}
            session.add(lab)
        state["bias"] = bias
        deployment.state = state
        deployment.updated_at = datetime.now(timezone.utc)
        await session.flush()
        await _trim(session, DeploymentLab, deployment.id, KEEP_LABS)
        await session.commit()
    return results


# --------------------------------------------------------------------------- reading back


def record_dict(rec: DeploymentRecord) -> dict[str, Any]:
    return {
        "id": rec.id, "time": _aware(rec.row_time).isoformat(), "received_at": _aware(rec.received_at).isoformat(),
        "status": rec.status, "prediction": rec.prediction, "corrected": rec.corrected, "bias": rec.bias,
        "t2": rec.t2, "spe": rec.spe, "reasons": list(rec.reasons or []), "inputs": dict(rec.inputs or {}),
    }


def lab_dict(lab: DeploymentLab) -> dict[str, Any]:
    return {
        "time": _aware(lab.sample_time).isoformat(), "value": lab.value,
        "record_time": _aware(lab.record_time).isoformat() if lab.record_time else None,
        "prediction": lab.prediction, "corrected": lab.corrected, "bias_after": lab.bias_after,
    }


def deployment_dict(dep: Deployment) -> dict[str, Any]:
    state = dict(dep.state or {})
    spec = dict(dep.spec or {})
    return {
        "id": str(dep.id), "name": dep.name, "status": dep.status, "model_id": dep.model_id,
        "dataset_id": str(dep.dataset_id), "api_key_prefix": dep.api_key_prefix,
        "created_at": _aware(dep.created_at).isoformat() if dep.created_at else None,
        "updated_at": _aware(dep.updated_at).isoformat() if dep.updated_at else None,
        "bias": state.get("bias", 0.0), "bias_gain": state.get("gain", DEFAULT_GAIN),
        "counts": state.get("counts", {}), "last_time": state.get("last_time"),
        "last_received": state.get("last_received"), "history_rows": len(state.get("buffer") or []),
        "model": spec.get("model"), "inputs": spec.get("inputs", []), "time_column": spec.get("time_column"),
    }


async def monitor_payload(session: AsyncSession, dep: Deployment, limit: int) -> dict[str, Any]:
    records = (
        await session.execute(
            select(DeploymentRecord).where(DeploymentRecord.deployment_id == dep.id)
            .order_by(DeploymentRecord.id.desc()).limit(limit)
        )
    ).scalars().all()
    labs = (
        await session.execute(
            select(DeploymentLab).where(DeploymentLab.deployment_id == dep.id)
            .order_by(DeploymentLab.sample_time.desc()).limit(limit)
        )
    ).scalars().all()
    total = (
        await session.execute(select(func.count()).select_from(DeploymentRecord).where(DeploymentRecord.deployment_id == dep.id))
    ).scalar_one()
    matched = [lab for lab in reversed(labs) if lab.corrected is not None]
    residuals = [lab.value - lab.corrected for lab in matched]
    test_rmse = ((dep.spec or {}).get("model") or {}).get("rmse")
    return {
        "deployment": deployment_dict(dep),
        "records": [record_dict(r) for r in reversed(records)],
        "labs": [lab_dict(lab) for lab in reversed(labs)],
        "stored_records": total,
        "accuracy": monitor.accuracy(residuals, test_rmse),
    }


# --------------------------------------------------------------------------- cleanup


async def deployments_of_datasets(session: AsyncSession, dataset_ids: list[uuid.UUID]) -> list[Deployment]:
    if not dataset_ids:
        return []
    return list((await session.execute(select(Deployment).where(Deployment.dataset_id.in_(dataset_ids)))).scalars().all())


async def delete_deployment_rows(session: AsyncSession, deployments: list[Deployment]) -> list[str]:
    """Delete the rows (children first, as SQLite does not cascade); returns the bundle keys to remove."""
    ids = [d.id for d in deployments]
    if not ids:
        return []
    await session.execute(delete(DeploymentLab).where(DeploymentLab.deployment_id.in_(ids)))
    await session.execute(delete(DeploymentRecord).where(DeploymentRecord.deployment_id.in_(ids)))
    await session.execute(delete(Deployment).where(Deployment.id.in_(ids)))
    for dep in deployments:
        _bundle_cache.drop(str(dep.id))
        _locks.pop(str(dep.id), None)
    return [d.bundle_key for d in deployments]


def forget_models(model_ids: list[str]) -> None:
    for model_id in model_ids:
        _compiled_cache.drop(str(model_id))


# --------------------------------------------------------------------------- simulation


def simulation_rows(prepared: Prepared, start: int, count: int) -> dict[str, Any]:
    """Rows of the data the live feed corresponds to, as a connector would send them, with the lab values."""
    compiled = prepared.compiled
    data = prepared.chain[compiled.cut_index].load()
    target = str(prepared.artifact["target_column"])
    total = len(data)
    start = max(0, min(start, total))
    part = data.iloc[start:start + count]
    rows: list[dict[str, Any]] = []
    labs: list[Optional[float]] = []
    for _, row in part.iterrows():
        rows.append({c: _plain(row[c]) for c in compiled.inputs})
        labs.append(_num(row[target]) if target in data.columns else None)
    return {"start": start, "total": total, "rows": rows, "labs": labs, "target_column": target,
            "time_column": compiled.time_column}


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _num(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value) if not isinstance(value, (int, str)) else value

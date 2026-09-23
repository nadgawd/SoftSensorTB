"""Index of trained soft-sensor models and their place in the dataset lineage.

Each saved artifact has a ``trained_models`` row holding its metadata, so
listing models never unpickles estimators. Lookups are scoped to the dataset of
the active version: a model id from another dataset (another user's) never
resolves.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Dataset, DatasetVersion, TrainedModel
from backend.storage import LOCAL_ROOT_NAME, LocalStorage

logger = logging.getLogger(__name__)

_MAX_LINEAGE_DEPTH = 500


def _meta_from_artifact(artifact: dict[str, Any], trained_at: str) -> dict[str, Any]:
    metrics = artifact.get("metrics") or {}
    return {
        "model_id": str(artifact.get("model_id")),
        "algorithm": artifact.get("algorithm"),
        "target_column": artifact.get("target_column"),
        "feature_columns": list(artifact.get("feature_columns") or []),
        "dataset_version_id": str(artifact.get("dataset_version_id") or ""),
        "new_version_id": str(artifact.get("new_version_id") or ""),
        "r2_score": metrics.get("r2_score"),
        "rmse": metrics.get("rmse"),
        "trained_at": artifact.get("trained_at") or trained_at,
    }


async def register_model(
    session: AsyncSession,
    *,
    dataset_id: uuid.UUID,
    artifact: dict[str, Any],
    artifact_key: str,
) -> dict[str, Any]:
    meta = _meta_from_artifact(artifact, datetime.now(timezone.utc).isoformat())
    session.add(
        TrainedModel(
            id=uuid.UUID(meta["model_id"]),
            dataset_id=dataset_id,
            meta=meta,
            artifact_key=artifact_key,
        )
    )
    await session.commit()
    return meta


async def load_model_meta(session: AsyncSession, dataset_id: uuid.UUID) -> list[dict[str, Any]]:
    rows = await session.execute(select(TrainedModel.meta).where(TrainedModel.dataset_id == dataset_id))
    return [dict(meta) for meta in rows.scalars().all()]


async def lineage_ids(session: AsyncSession, dataset_version_id: str) -> list[str]:
    """``dataset_version_id`` followed by each ancestor, nearest first."""
    out: list[str] = []
    current: Optional[uuid.UUID] = uuid.UUID(dataset_version_id)
    while current is not None and len(out) < _MAX_LINEAGE_DEPTH:
        if str(current) in out:
            break
        version = await session.get(DatasetVersion, current)
        if version is None:
            break
        out.append(str(version.id))
        current = version.parent_version_id
    return out


def models_for_lineage(metas: list[dict[str, Any]], lineage: list[str]) -> list[dict[str, Any]]:
    """Models trained on, or producing, any version in ``lineage``; newest first."""
    members = set(lineage)
    hits = [
        m for m in metas
        if m.get("dataset_version_id") in members or m.get("new_version_id") in members
    ]
    return sorted(hits, key=lambda m: m.get("trained_at") or "", reverse=True)


async def lineage_models(session: AsyncSession, dataset_version_id: str) -> list[dict[str, Any]]:
    """Models on the lineage of ``dataset_version_id``, newest first."""
    version = await session.get(DatasetVersion, uuid.UUID(dataset_version_id))
    if version is None:
        return []
    metas = await load_model_meta(session, version.dataset_id)
    return models_for_lineage(metas, await lineage_ids(session, dataset_version_id))


def _as_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
    try:
        return uuid.UUID(str(value)) if value else None
    except ValueError:
        return None


async def resolve_model(
    session: AsyncSession,
    dataset_version_id: str,
    model_id: Optional[str] = None,
) -> TrainedModel:
    """``model_id`` if it names a model of this dataset, else the newest one in the lineage."""
    version = await session.get(DatasetVersion, uuid.UUID(dataset_version_id))
    if version is None:
        raise ValueError(f"Dataset version not found: {dataset_version_id}")
    explicit = _as_uuid(model_id)
    if explicit is not None:
        row = await session.get(TrainedModel, explicit)
        if row is not None and row.dataset_id == version.dataset_id:
            return row
    candidates = await lineage_models(session, dataset_version_id)
    if not candidates:
        raise ValueError(
            "No trained model exists for this dataset version or any version it was "
            "derived from. Train one with train_soft_sensor first."
        )
    row = await session.get(TrainedModel, uuid.UUID(candidates[0]["model_id"]))
    assert row is not None
    return row


async def resolve_model_id(
    session: AsyncSession,
    dataset_version_id: str,
    model_id: Optional[str] = None,
) -> str:
    return str((await resolve_model(session, dataset_version_id, model_id)).id)


async def import_legacy_models(session: AsyncSession, storage: LocalStorage) -> int:
    """Index ``data_storage/models/*.pkl`` files saved before the table existed.

    Their JSON sidecar (or the artifact itself) names the dataset version they
    were trained on; artifacts whose version no longer exists are skipped.
    """
    models_dir = storage.root / "models"
    if not models_dir.is_dir():
        return 0
    known = set((await session.execute(select(TrainedModel.id))).scalars().all())
    added = 0
    for pkl in models_dir.glob("*.pkl"):
        model_uuid = _as_uuid(pkl.stem)
        if model_uuid is None or model_uuid in known:
            continue
        try:
            sidecar = pkl.with_suffix(".json")
            if sidecar.exists():
                meta = json.loads(sidecar.read_text())
            else:
                artifact = joblib.load(pkl)
                if not isinstance(artifact, dict):
                    continue
                artifact.setdefault("model_id", pkl.stem)
                mtime = datetime.fromtimestamp(pkl.stat().st_mtime, tz=timezone.utc).isoformat()
                meta = _meta_from_artifact(artifact, mtime)
            version_uuid = _as_uuid(meta.get("dataset_version_id"))
            version = await session.get(DatasetVersion, version_uuid) if version_uuid else None
            if version is None or await session.get(Dataset, version.dataset_id) is None:
                continue
            session.add(
                TrainedModel(
                    id=model_uuid,
                    dataset_id=version.dataset_id,
                    meta=meta,
                    artifact_key=str(Path(LOCAL_ROOT_NAME) / "models" / pkl.name),
                )
            )
            added += 1
        except Exception:  # noqa: BLE001 — one unreadable artifact must not hide the rest
            logger.warning("Skipping unreadable model artifact %s", pkl, exc_info=True)
    if added:
        await session.commit()
    return added

"""Deployment step: plan, create and manage live deployments, and the live endpoints a plant connector calls."""

from __future__ import annotations

import io
import json
import secrets
import uuid
import zipfile
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import User, get_current_user
from backend.database import AsyncSessionLocal
from backend.deployment import service
from backend.limits import SlidingWindowLimiter
from backend.mcp_server.model_registry import resolve_model
from backend.models import Dataset, Deployment
from backend.services.projects import owned_version
from backend.storage import get_storage

router = APIRouter(tags=["deployments"])
live_limiter = SlidingWindowLimiter(60.0)

CONNECTOR_DIR = Path(__file__).resolve().parent.parent / "deployment" / "plant_connector"


def _uuid(value: str, what: str = "Deployment") -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"{what} not found.") from exc


async def _owned(session: AsyncSession, deployment_id: str, user: User) -> Deployment:
    dep = await session.get(Deployment, _uuid(deployment_id))
    if dep is None or dep.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Deployment not found.")
    return dep


async def _model_for(session: AsyncSession, dataset_version_id: str, model_id: Optional[str], user: User):
    version = await owned_version(session, dataset_version_id, user)
    try:
        model = await resolve_model(session, str(version.id), model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if model_id and str(model.id) != model_id:
        raise HTTPException(status_code=404, detail="Model not found for this dataset.")
    return version, model


async def _prepare(session: AsyncSession, model) -> service.Prepared:
    try:
        return await service.prepare_model(session, model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="A stored dataset version or the model file is missing.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --------------------------------------------------------------------------- management


@router.get("/datasets/versions/{dataset_version_id}/deployment-plan")
async def deployment_plan(
    dataset_version_id: str,
    model_id: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """What deploying the model needs: the live inputs, how each step runs on live rows, and the checks."""
    async with AsyncSessionLocal() as session:
        _, model = await _model_for(session, dataset_version_id, model_id, user)
        prepared = await _prepare(session, model)
    return service.plan_payload(prepared)


class CreateDeployment(BaseModel):
    dataset_version_id: str
    model_id: str
    name: str = Field("", max_length=200)
    bias_gain: float = Field(service.DEFAULT_GAIN, ge=0.0, le=1.0)


@router.post("/deployments")
async def create_deployment(body: CreateDeployment, user: User = Depends(get_current_user)) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        version, model = await _model_for(session, body.dataset_version_id, body.model_id, user)
        dataset = await session.get(Dataset, version.dataset_id)
        cap = service.max_deployments_per_user()
        if cap > 0:
            count = (
                await session.execute(select(func.count()).select_from(Deployment).where(Deployment.owner_id == user.id))
            ).scalar_one()
            if count >= cap:
                raise HTTPException(status_code=409, detail=f"You can have at most {cap} deployments; delete one first.")
        meta = model.meta or {}
        name = body.name.strip() or f"{meta.get('algorithm') or 'Model'} → {meta.get('target_column') or 'target'}"
        try:
            deployment, api_key = await service.create_deployment(
                session, owner_id=user.id, dataset=dataset, model=model, name=name, gain=body.bias_gain,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="A stored dataset version or the model file is missing.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deployment": service.deployment_dict(deployment), "api_key": api_key}


@router.get("/deployments")
async def list_deployments(user: User = Depends(get_current_user)) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(Deployment, Dataset.file_name).join(Dataset, Dataset.id == Deployment.dataset_id)
                .where(Deployment.owner_id == user.id).order_by(Deployment.created_at.desc())
            )
        ).all()
    return {
        "deployments": [{**service.deployment_dict(dep), "dataset_name": file_name} for dep, file_name in rows],
        "max_deployments": service.max_deployments_per_user(),
    }


@router.get("/deployments/{deployment_id}")
async def get_deployment(deployment_id: str, user: User = Depends(get_current_user)) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        dep = await _owned(session, deployment_id, user)
    return {**service.deployment_dict(dep), "spec": dep.spec}


class UpdateDeployment(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    status: Optional[Literal["active", "paused"]] = None
    bias_gain: Optional[float] = Field(None, ge=0.0, le=1.0)
    reset_bias: bool = False


@router.patch("/deployments/{deployment_id}")
async def update_deployment(deployment_id: str, body: UpdateDeployment, user: User = Depends(get_current_user)) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        dep = await _owned(session, deployment_id, user)
        async with service._lock_for(str(dep.id)):
            await session.refresh(dep)
            if body.name is not None and body.name.strip():
                dep.name = body.name.strip()
            if body.status is not None:
                dep.status = body.status
            state = dict(dep.state or {})
            if body.bias_gain is not None:
                state["gain"] = body.bias_gain
            if body.reset_bias:
                state["bias"] = 0.0
            dep.state = state
            await session.commit()
    return service.deployment_dict(dep)


@router.delete("/deployments/{deployment_id}")
async def delete_deployment(deployment_id: str, user: User = Depends(get_current_user)) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        dep = await _owned(session, deployment_id, user)
        keys = await service.delete_deployment_rows(session, [dep])
        await session.flush()
        get_storage().delete(keys)
        await session.commit()
    return {"deleted": deployment_id}


@router.post("/deployments/{deployment_id}/rotate-key")
async def rotate_key(deployment_id: str, user: User = Depends(get_current_user)) -> dict[str, Any]:
    """A new API key; the old one stops working at once."""
    async with AsyncSessionLocal() as session:
        dep = await _owned(session, deployment_id, user)
        api_key, key_hash, prefix = service.new_api_key()
        dep.api_key_hash, dep.api_key_prefix = key_hash, prefix
        await session.commit()
    return {"deployment": service.deployment_dict(dep), "api_key": api_key}


@router.get("/deployments/{deployment_id}/monitor")
async def deployment_monitor(
    deployment_id: str,
    limit: int = Query(500, ge=1, le=service.KEEP_RECORDS),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """The latest predictions and lab results, and the lab accuracy."""
    async with AsyncSessionLocal() as session:
        dep = await _owned(session, deployment_id, user)
        return await service.monitor_payload(session, dep, limit)


async def _prepared_for(session: AsyncSession, dep: Deployment) -> service.Prepared:
    from backend.models import TrainedModel

    model = await session.get(TrainedModel, _uuid(dep.model_id, "Model"))
    if model is None:
        raise HTTPException(status_code=404, detail="The deployed model no longer exists.")
    return await _prepare(session, model)


@router.get("/deployments/{deployment_id}/simulation")
async def simulation(
    deployment_id: str,
    start: int = Query(0, ge=0),
    count: int = Query(200, ge=1, le=service.MAX_ROWS_PER_REQUEST),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Rows of the deployment's own data in the live format, to replay as a simulated feed."""
    async with AsyncSessionLocal() as session:
        dep = await _owned(session, deployment_id, user)
        prepared = await _prepared_for(session, dep)
    return await run_in_threadpool(service.simulation_rows, prepared, start, count)


def connector_config(dep: Deployment, base_url: str) -> dict[str, Any]:
    spec = dep.spec or {}
    inputs = [i["name"] for i in spec.get("inputs", []) if not i.get("is_time")]
    target = (spec.get("model") or {}).get("target_column") or "target"
    return {
        "server": base_url.rstrip("/"),
        "deployment_id": str(dep.id),
        "api_key_env": "SOFTSENSOR_API_KEY",
        "interval_seconds": 60,
        "time_column": spec.get("time_column"),
        "inputs": {name: name for name in inputs},
        "source": {
            "type": "csv",
            "csv": {"path": "live_data.csv", "follow": True, "time_column": spec.get("time_column")},
            "opcua": {"endpoint": "opc.tcp://historian:4840", "nodes": {name: f"ns=2;s={name}" for name in inputs},
                      "username": None, "password_env": None},
            "mqtt": {"host": "broker.local", "port": 1883, "topics": {name: f"plant/unit1/{name}" for name in inputs},
                     "value_key": "value", "time_key": None, "max_age_seconds": 300, "qos": 1,
                     "tls": False, "username": None, "password_env": None},
            "piweb": {"base_url": "https://pi-server/piwebapi", "web_ids": {name: "<WebId>" for name in inputs},
                      "username": None, "password_env": None, "verify_tls": True},
        },
        "lab": {"type": "none", "csv": {"path": "lab_results.csv", "time_column": "sample_time", "value_column": target}},
        "write_back": {"type": "none", "opcua_node": "ns=2;s=SoftSensor." + target,
                       "mqtt_topic": "plant/unit1/softsensor/" + target,
                       "piweb_web_id": "<WebId of the prediction tag>", "csv_path": "predictions.csv"},
    }


@router.get("/deployments/{deployment_id}/connector")
async def download_connector(deployment_id: str, request: Request, user: User = Depends(get_current_user)) -> Response:
    """A zip with the plant connector script and a config filled in for this deployment (no key inside)."""
    async with AsyncSessionLocal() as session:
        dep = await _owned(session, deployment_id, user)
    base = str(request.base_url)
    forwarded = request.headers.get("x-forwarded-proto")
    if forwarded:
        base = base.replace(f"{request.url.scheme}://", f"{forwarded}://", 1)
    config = connector_config(dep, base)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("softsensor_connector.py", "README.md", "requirements.txt"):
            archive.write(CONNECTOR_DIR / name, f"softsensor_connector/{name}")
        archive.writestr("softsensor_connector/config.json", json.dumps(config, indent=2))
    return Response(
        buffer.getvalue(), media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="softsensor_connector.zip"'},
    )


# --------------------------------------------------------------------------- live


async def _live(session: AsyncSession, deployment_id: str, request: Request) -> Deployment:
    """The deployment, for its API key (a plant connector) or for its owner (the in-app simulator)."""
    key = request.headers.get("x-api-key")
    try:
        dep_uuid = uuid.UUID(deployment_id)
    except ValueError:
        dep_uuid = None
    dep = await session.get(Deployment, dep_uuid) if dep_uuid else None
    if key:
        if dep is None or not secrets.compare_digest(dep.api_key_hash, service.hash_key(key)):
            raise HTTPException(status_code=401, detail="Invalid API key for this deployment.")
    else:
        user = await get_current_user(request)
        if dep is None or dep.owner_id != user.id:
            raise HTTPException(status_code=404, detail="Deployment not found.")
    live_limiter.check(
        str(dep.id), service.live_rate_per_minute(),
        "Too many requests for this deployment ({limit} per minute); send rows in batches",
    )
    return dep


class LiveRows(BaseModel):
    rows: list[dict[str, Any]] = Field(..., description="Readings, one object per time stamp: input name → value.")

    @field_validator("rows")
    @classmethod
    def _bounded(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            raise ValueError("Send at least one row.")
        if len(rows) > service.MAX_ROWS_PER_REQUEST:
            raise ValueError(f"Send at most {service.MAX_ROWS_PER_REQUEST} rows per request.")
        return rows


class LabSamples(BaseModel):
    samples: list[dict[str, Any]] = Field(..., description="Lab results: {time: sample time, value: measured target}.")

    @field_validator("samples")
    @classmethod
    def _bounded(cls, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not samples:
            raise ValueError("Send at least one lab result.")
        if len(samples) > service.MAX_LABS_PER_REQUEST:
            raise ValueError(f"Send at most {service.MAX_LABS_PER_REQUEST} lab results per request.")
        return samples


@router.get("/live/{deployment_id}")
async def live_info(deployment_id: str, request: Request) -> dict[str, Any]:
    """What to send: lets a connector check its key and settings before it starts."""
    async with AsyncSessionLocal() as session:
        dep = await _live(session, deployment_id, request)
    info = service.deployment_dict(dep)
    return {key: info[key] for key in ("id", "name", "status", "inputs", "time_column", "model", "last_time")}


@router.post("/live/{deployment_id}/predict")
async def live_predict(deployment_id: str, body: LiveRows, request: Request) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        dep = await _live(session, deployment_id, request)
        if dep.status != "active":
            raise HTTPException(status_code=409, detail="This deployment is paused.")
        try:
            records = await service.ingest_rows(session, dep, body.rows)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="The deployment's bundle is missing from storage.") from exc
    return {"predictions": records}


@router.post("/live/{deployment_id}/lab")
async def live_lab(deployment_id: str, body: LabSamples, request: Request) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        dep = await _live(session, deployment_id, request)
        results = await service.ingest_labs(session, dep, body.samples)
        bias = (dep.state or {}).get("bias", 0.0)
    return {"results": results, "bias": bias}

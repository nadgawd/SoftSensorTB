"""The caller's saved project: UI state kept until New project deletes it all."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.auth import User, get_current_user
from backend.database import AsyncSessionLocal
from backend.services.projects import delete_project, load_state, save_state

router = APIRouter(tags=["project"])


def _max_state_bytes() -> int:
    try:
        return int(os.getenv("MAX_PROJECT_STATE_MB", "5")) * 1024 * 1024
    except ValueError:
        return 5 * 1024 * 1024


@router.get("/project/state")
async def get_project_state(user: User = Depends(get_current_user)) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        return {"user_id": user.id, "state": await load_state(session, user)}


@router.put("/project/state")
async def put_project_state(request: Request, user: User = Depends(get_current_user)) -> dict[str, Any]:
    limit = _max_state_bytes()
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Project state is larger than {limit // (1024 * 1024)} MB; older plots are not saved.",
            )
    try:
        payload = json.loads(body or b"null")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Project state must be JSON.") from exc
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise HTTPException(status_code=400, detail='Send {"state": {...}}.')
    async with AsyncSessionLocal() as session:
        await save_state(session, user, state)
    return {"ok": True, "bytes": len(body)}


@router.delete("/project")
async def delete_current_project(user: User = Depends(get_current_user)) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        return await delete_project(session, user)

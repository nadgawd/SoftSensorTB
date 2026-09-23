"""The public deployment: signed-in users only see and delete their own work."""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend import auth
from backend.api import datasets as datasets_api
from backend.database import AsyncSessionLocal, engine
from backend.limits import chat_limiter
from backend.main import app
from backend.models import Dataset, TrainedModel
from backend.storage import LocalStorage, model_key, set_storage

SUPABASE_URL = "https://test-project.supabase.co"
SIGNING_KEY = ec.generate_private_key(ec.SECP256R1())
CSV = b"x,y\n1,2\n3,4\n5,6\n"


def make_token(sub, *, key=SIGNING_KEY, aud="authenticated", role="authenticated",
               iss=f"{SUPABASE_URL}/auth/v1", expires_in=3600):
    now = int(time.time())
    claims = {"sub": sub, "aud": aud, "role": role, "iss": iss,
              "iat": now, "exp": now + expires_in, "is_anonymous": True}
    return jwt.encode(claims, key, algorithm="ES256", headers={"kid": "test"})


def as_user(sub):
    return {"Authorization": f"Bearer {make_token(sub)}"}


class FakeJWKS:
    def get_signing_key_from_jwt(self, token):
        return SimpleNamespace(key=SIGNING_KEY.public_key())


@pytest.fixture
def storage(tmp_path):
    store = LocalStorage(tmp_path / "storage")
    set_storage(store)
    yield store
    set_storage(None)


@pytest.fixture
def client(monkeypatch, storage):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SUPABASE_URL", SUPABASE_URL)
    for name in ("SUPABASE_JWT_SECRET", "MAX_DATASETS_PER_USER", "CHAT_RATE_LIMIT_PER_HOUR",
                 "MAX_UPLOAD_MB", "MAX_PROJECT_STATE_MB"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(auth, "_jwks_client", lambda url: FakeJWKS())
    chat_limiter.reset()
    with TestClient(app) as test_client:
        yield test_client
        # Pooled connections belong to this client's event loop.
        test_client.portal.call(engine.dispose)


def new_user():
    return str(uuid.uuid4())


def upload(client, user, body=CSV, name="data.csv"):
    resp = client.post("/datasets/upload", headers=as_user(user), files={"file": (name, body, "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_valid_token_is_accepted(client):
    resp = client.get("/project/state", headers=as_user("user-1"))
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "user-1"


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer not-a-jwt"},
        {"Authorization": f"Bearer {make_token('u', expires_in=-3600)}"},
        {"Authorization": f"Bearer {make_token('u', aud='someone-else')}"},
        {"Authorization": f"Bearer {make_token('u', iss='https://other.supabase.co/auth/v1')}"},
        {"Authorization": f"Bearer {make_token('u', role='anon')}"},
        {"Authorization": f"Bearer {make_token('u', key=ec.generate_private_key(ec.SECP256R1()))}"},
    ],
    ids=["missing", "garbage", "expired", "wrong-audience", "wrong-issuer", "anon-role", "wrong-key"],
)
def test_invalid_tokens_are_rejected(client, headers):
    assert client.get("/project/state", headers=headers).status_code == 401


def test_legacy_shared_secret_tokens(client, monkeypatch):
    now = int(time.time())
    claims = {"sub": "legacy-user", "aud": "authenticated", "role": "authenticated",
              "iss": f"{SUPABASE_URL}/auth/v1", "exp": now + 600}
    hs256 = jwt.encode(claims, "s" * 32, algorithm="HS256")
    headers = {"Authorization": f"Bearer {hs256}"}

    assert client.get("/project/state", headers=headers).status_code == 401
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "s" * 32)
    assert client.get("/project/state", headers=headers).json()["user_id"] == "legacy-user"


def test_unreachable_signing_keys_are_a_503_not_a_logout(client, monkeypatch):
    class DownJWKS:
        def get_signing_key_from_jwt(self, token):
            raise jwt.PyJWKClientConnectionError("timed out")

    monkeypatch.setattr(auth, "_jwks_client", lambda url: DownJWKS())
    assert client.get("/project/state", headers=as_user("u")).status_code == 503


def test_other_users_cannot_reach_a_dataset(client):
    alice, bob = new_user(), new_user()
    up = upload(client, alice)
    version, dataset = up["dataset_version_id"], up["dataset_id"]

    assert client.get(f"/datasets/versions/{version}/preview", headers=as_user(alice)).status_code == 200

    bob_h = as_user(bob)
    assert client.get(f"/datasets/versions/{version}/preview", headers=bob_h).status_code == 404
    assert client.get(f"/datasets/versions/{version}/meta", headers=bob_h).status_code == 404
    assert client.post(f"/datasets/versions/{version}/rollback", headers=bob_h).status_code == 404
    assert client.get(f"/datasets/{dataset}/history", headers=bob_h).status_code == 404
    chat = client.post("/chat", headers=bob_h, json={"message": "describe it", "dataset_version_id": version})
    assert chat.status_code == 404


def test_project_state_is_per_user(client):
    alice, bob = new_user(), new_user()
    state = {"activeStep": 2, "chatMessages": [{"role": "user", "content": "hi"}]}
    assert client.put("/project/state", headers=as_user(alice), json={"state": state}).status_code == 200

    assert client.get("/project/state", headers=as_user(alice)).json()["state"] == state
    assert client.get("/project/state", headers=as_user(bob)).json()["state"] is None


def test_new_project_deletes_only_the_callers_work(client, storage):
    alice, bob = new_user(), new_user()
    first = upload(client, alice)
    upload(client, alice)
    bobs = upload(client, bob)
    client.put("/project/state", headers=as_user(alice), json={"state": {"activeStep": 3}})

    artifact = model_key(alice, first["dataset_id"], uuid.uuid4())
    storage.write(artifact, lambda path: path.write_bytes(b"model"))

    async def add_model():
        async with AsyncSessionLocal() as session:
            session.add(TrainedModel(id=uuid.uuid4(), dataset_id=uuid.UUID(first["dataset_id"]),
                                     meta={}, artifact_key=artifact))
            await session.commit()

    client.portal.call(add_model)

    resp = client.delete("/project", headers=as_user(alice))
    assert resp.status_code == 200
    assert resp.json() == {"deleted_datasets": 2}

    async def remaining():
        async with AsyncSessionLocal() as session:
            datasets = await session.scalar(
                select(func.count()).select_from(Dataset).where(Dataset.owner_id == alice))
            models = await session.scalar(
                select(func.count()).select_from(TrainedModel).where(TrainedModel.artifact_key == artifact))
            return datasets, models

    assert client.portal.call(remaining) == (0, 0)
    assert not (storage.root / alice).exists()
    assert client.get("/project/state", headers=as_user(alice)).json()["state"] is None

    bob_version = bobs["dataset_version_id"]
    assert client.get(f"/datasets/versions/{bob_version}/preview", headers=as_user(bob)).status_code == 200


def test_oldest_dataset_is_evicted_past_the_cap(client, storage, monkeypatch):
    monkeypatch.setenv("MAX_DATASETS_PER_USER", "2")
    alice = new_user()
    ups = [upload(client, alice) for _ in range(3)]

    h = as_user(alice)
    assert client.get(f"/datasets/{ups[0]['dataset_id']}/history", headers=h).status_code == 404
    assert not [p for p in (storage.root / alice / ups[0]["dataset_id"]).rglob("*") if p.is_file()]
    for up in ups[1:]:
        assert client.get(f"/datasets/{up['dataset_id']}/history", headers=h).status_code == 200


def test_upload_size_is_capped(client, monkeypatch):
    monkeypatch.setattr(datasets_api, "max_upload_bytes", lambda: len(CSV) - 1)
    resp = client.post("/datasets/upload", headers=as_user(new_user()),
                       files={"file": ("data.csv", CSV, "text/csv")})
    assert resp.status_code == 413


def test_project_state_size_is_capped(client, monkeypatch):
    monkeypatch.setenv("MAX_PROJECT_STATE_MB", "1")
    big = {"state": {"plotHistory": "x" * (1024 * 1024 + 1)}}
    assert client.put("/project/state", headers=as_user(new_user()), json=big).status_code == 413


def test_chat_is_rate_limited_per_user(client, monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_HOUR", "2")
    alice, bob = new_user(), new_user()
    # Alice's version: Bob's requests stop at the ownership check, before any LLM call.
    version = upload(client, alice)["dataset_version_id"]
    body = {"message": "hi", "dataset_version_id": version}

    assert [client.post("/chat", headers=as_user(bob), json=body).status_code for _ in range(2)] == [404, 404]
    limited = client.post("/chat", headers=as_user(bob), json=body)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) > 0
    assert client.post("/chat", headers=as_user(alice), json={**body, "dataset_version_id": str(uuid.uuid4())}).status_code == 404


def test_local_mode_needs_no_token(client, monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    resp = client.get("/project/state")
    assert resp.status_code == 200
    assert resp.json()["user_id"] == auth.LOCAL_USER_ID

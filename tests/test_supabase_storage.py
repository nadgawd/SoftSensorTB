"""Supabase Storage backend against a fake Storage API."""

from __future__ import annotations

import json

import httpx
import pytest

from backend.storage import SupabaseStorage


class FakeStorageAPI:
    """Just enough of /storage/v1 for SupabaseStorage: objects in a dict."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/storage/v1")
        if path.startswith("/object/list/sst-data"):
            prefix = json.loads(request.content)["prefix"].rstrip("/")
            children: dict[str, bool] = {}
            for key in self.objects:
                if key.startswith(prefix + "/"):
                    head, _, rest = key[len(prefix) + 1:].partition("/")
                    children[head] = children.get(head, False) or not rest
            return httpx.Response(200, json=[{"name": n, "id": "x" if is_file else None}
                                             for n, is_file in children.items()])
        if path == "/object/sst-data" and request.method == "DELETE":
            for key in json.loads(request.content)["prefixes"]:
                self.objects.pop(key, None)
            return httpx.Response(200, json=[])
        if path.startswith("/object/sst-data/"):
            key = path.removeprefix("/object/sst-data/")
            if request.method == "POST":
                self.objects[key] = request.content
                return httpx.Response(200, json={"Key": key})
            if request.method == "GET":
                if key not in self.objects:
                    return httpx.Response(400, json={"error": "not_found"})
                return httpx.Response(200, content=self.objects[key])
        if path == "/bucket/sst-data":
            return httpx.Response(200, json={"id": "sst-data"})
        return httpx.Response(404)


@pytest.fixture
def api():
    return FakeStorageAPI()


def make_storage(api, tmp_path, key="sb_secret_test", **kwargs):
    return SupabaseStorage("https://p.supabase.co", key, "sst-data", tmp_path / "cache",
                           transport=httpx.MockTransport(api), **kwargs)


def test_write_uploads_and_fetch_reads_back(api, tmp_path):
    store = make_storage(api, tmp_path)
    store.write("u1/d1/v1.parquet", lambda path: path.write_bytes(b"parquet-bytes"))
    assert api.objects == {"u1/d1/v1.parquet": b"parquet-bytes"}

    fresh = make_storage(api, tmp_path / "other")
    assert fresh.fetch("u1/d1/v1.parquet").read_bytes() == b"parquet-bytes"


def test_missing_object_is_file_not_found(api, tmp_path):
    with pytest.raises(FileNotFoundError):
        make_storage(api, tmp_path).fetch("u1/d1/missing.parquet")


def test_delete_prefix_removes_nested_objects_of_that_user_only(api, tmp_path):
    store = make_storage(api, tmp_path)
    for key in ("u1/d1/v1.parquet", "u1/d1/models/m1.pkl", "u1/d2/v2.parquet", "u2/d3/v3.parquet"):
        store.write(key, lambda path: path.write_bytes(b"x"))

    store.delete_prefix("u1/")
    assert list(api.objects) == ["u2/d3/v3.parquet"]
    assert not (tmp_path / "cache" / "u1").exists()


@pytest.mark.parametrize("key", ["../etc/passwd", "u1/../../x", "/abs/key", ""])
def test_unsafe_keys_are_refused(api, tmp_path, key):
    with pytest.raises(ValueError):
        make_storage(api, tmp_path).fetch(key)


def test_secret_keys_go_only_in_the_apikey_header(api, tmp_path):
    make_storage(api, tmp_path, key="sb_secret_abc").ping()
    make_storage(api, tmp_path, key="eyJlegacy.service.role").ping()
    new, legacy = api.requests
    assert new.headers["apikey"] == "sb_secret_abc" and "authorization" not in new.headers
    assert legacy.headers["authorization"] == "Bearer eyJlegacy.service.role"


def test_cache_is_pruned_past_its_limit(api, tmp_path):
    store = make_storage(api, tmp_path, cache_max_bytes=10)
    for i in range(4):
        store.write(f"u1/d1/v{i}.parquet", lambda path: path.write_bytes(b"12345"))
    cached = [p for p in (tmp_path / "cache").rglob("*") if p.is_file()]
    assert sum(p.stat().st_size for p in cached) <= 10
    assert len(api.objects) == 4

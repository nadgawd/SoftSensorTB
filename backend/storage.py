"""Where dataset versions and model artifacts live.

Objects are addressed by key: ``{owner_id}/{dataset_id}/{version_id}.parquet``
and ``{owner_id}/{dataset_id}/models/{model_id}.pkl``. Every object is written
once and never changed, so a downloaded copy can be cached indefinitely.

``STORAGE_BACKEND=local`` (default) keeps files under ``data_storage/``.
``STORAGE_BACKEND=supabase`` stores them in a private Supabase Storage bucket
and caches reads under ``/tmp``, because the host's disk is wiped on restart.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_ROOT_NAME = "data_storage"


def parquet_key(owner_id: str, dataset_id: object, version_id: object) -> str:
    return f"{owner_id}/{dataset_id}/{version_id}.parquet"


def model_key(owner_id: str, dataset_id: object, model_id: object) -> str:
    return f"{owner_id}/{dataset_id}/models/{model_id}.pkl"


def _check_key(key: str) -> str:
    parts = Path(key).parts
    if not key or Path(key).is_absolute() or any(p in {"..", ""} for p in parts):
        raise ValueError(f"Invalid storage key: {key!r}")
    return key


class Storage(Protocol):
    def write(self, key: str, writer: Callable[[Path], None]) -> None: ...
    def fetch(self, key: str) -> Path: ...
    def delete(self, keys: Iterable[str]) -> None: ...
    def delete_prefix(self, prefix: str) -> None: ...
    def ping(self) -> None: ...


class LocalStorage:
    """Files under ``data_storage/``.

    Rows created before keys existed hold paths like ``data_storage/<id>.parquet``
    (relative to the project root) or absolute paths; those still resolve.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or _PROJECT_ROOT / LOCAL_ROOT_NAME).resolve()

    def path_for(self, key: str) -> Path:
        path = Path(key)
        if path.is_absolute():
            return path.resolve()
        if path.parts and path.parts[0] == LOCAL_ROOT_NAME:
            path = Path(*path.parts[1:])
        resolved = (self.root / _check_key(str(path))).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Invalid storage key: {key!r}")
        return resolved

    def write(self, key: str, writer: Callable[[Path], None]) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path)

    def fetch(self, key: str) -> Path:
        path = self.path_for(key)
        if not path.exists():
            raise FileNotFoundError(f"Stored file missing: {key}")
        return path

    def delete(self, keys: Iterable[str]) -> None:
        for key in keys:
            try:
                path = self.path_for(key)
            except ValueError:
                continue
            if path.is_relative_to(self.root):
                path.unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        target = self.path_for(prefix.rstrip("/"))
        if target != self.root and target.is_dir():
            shutil.rmtree(target, ignore_errors=True)

    def ping(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


class SupabaseStorage:
    """A private Supabase Storage bucket, reached with the service-role key."""

    _PAGE = 1000

    def __init__(
        self,
        url: str,
        service_key: str,
        bucket: str,
        cache_dir: Path,
        *,
        cache_max_bytes: int = 1024 * 1024 * 1024,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.bucket = bucket
        self.cache_dir = cache_dir
        self.cache_max_bytes = cache_max_bytes
        self._lock = threading.Lock()
        headers = {"apikey": service_key}
        # Secret keys (sb_secret_...) are not JWTs and are refused as a Bearer
        # token; only the legacy service_role JWT goes in Authorization.
        if not service_key.startswith("sb_"):
            headers["Authorization"] = f"Bearer {service_key}"
        self._client = httpx.Client(
            base_url=f"{url.rstrip('/')}/storage/v1",
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
            transport=transport,
        )

    def _object_url(self, key: str) -> str:
        return f"/object/{self.bucket}/{quote(_check_key(key), safe='/')}"

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / _check_key(key)

    def write(self, key: str, writer: Callable[[Path], None]) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path)
        resp = self._client.post(
            self._object_url(key),
            content=path.read_bytes(),
            headers={"Content-Type": "application/octet-stream", "x-upsert": "true"},
        )
        if resp.status_code >= 300:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"Storage upload failed ({resp.status_code}): {resp.text[:200]}")
        self._prune_cache()

    def fetch(self, key: str) -> Path:
        path = self._cache_path(key)
        if path.exists():
            return path
        resp = self._client.get(self._object_url(key))
        # Supabase answers a missing object with 400 or 404.
        if resp.status_code in (400, 404):
            raise FileNotFoundError(f"Stored file missing: {key}")
        if resp.status_code >= 300:
            raise RuntimeError(f"Storage download failed ({resp.status_code}): {resp.text[:200]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.part")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
        self._prune_cache()
        return path

    def _list(self, prefix: str) -> list[str]:
        """All object keys under ``prefix`` (folders are walked recursively)."""
        keys: list[str] = []
        folders = [prefix.rstrip("/")]
        while folders:
            folder = folders.pop()
            offset = 0
            while True:
                resp = self._client.post(
                    f"/object/list/{self.bucket}",
                    json={"prefix": folder, "limit": self._PAGE, "offset": offset},
                )
                resp.raise_for_status()
                items = resp.json()
                for item in items:
                    child = f"{folder}/{item['name']}" if folder else item["name"]
                    # Folders are listed with a null id.
                    (keys if item.get("id") else folders).append(child)
                if len(items) < self._PAGE:
                    break
                offset += self._PAGE
        return keys

    def delete(self, keys: Iterable[str]) -> None:
        batch = [_check_key(k) for k in keys]
        for start in range(0, len(batch), self._PAGE):
            chunk = batch[start : start + self._PAGE]
            resp = self._client.request("DELETE", f"/object/{self.bucket}", json={"prefixes": chunk})
            resp.raise_for_status()
        for key in batch:
            self._cache_path(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        prefix = _check_key(prefix.rstrip("/"))
        self.delete(self._list(prefix))
        shutil.rmtree(self.cache_dir / prefix, ignore_errors=True)

    def ping(self) -> None:
        """Raise unless the bucket is reachable with the service-role key."""
        resp = self._client.get(f"/bucket/{self.bucket}")
        if resp.status_code >= 300:
            raise RuntimeError(f"bucket {self.bucket!r}: HTTP {resp.status_code} {resp.text[:200]}")

    def _prune_cache(self) -> None:
        with self._lock:
            files = [p for p in self.cache_dir.rglob("*") if p.is_file()]
            total = sum(p.stat().st_size for p in files)
            if total <= self.cache_max_bytes:
                return
            for path in sorted(files, key=lambda p: p.stat().st_mtime):
                total -= path.stat().st_size
                path.unlink(missing_ok=True)
                if total <= self.cache_max_bytes * 0.8:
                    break


_storage: Optional[Storage] = None


def _build_storage() -> Storage:
    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    if backend == "local":
        return LocalStorage()
    if backend == "supabase":
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not url or not key:
            raise RuntimeError("STORAGE_BACKEND=supabase needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
        cache = Path(os.getenv("STORAGE_CACHE_DIR") or Path(tempfile.gettempdir()) / "sst_cache")
        return SupabaseStorage(
            url,
            key,
            os.getenv("SUPABASE_BUCKET", "sst-data"),
            cache,
            cache_max_bytes=int(os.getenv("STORAGE_CACHE_MB", "1024")) * 1024 * 1024,
        )
    raise RuntimeError(f"Unknown STORAGE_BACKEND: {backend!r} (use 'local' or 'supabase').")


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = _build_storage()
    return _storage


def set_storage(storage: Optional[Storage]) -> None:
    """Swap the backend (tests), or reset it so the next call rebuilds from env."""
    global _storage
    _storage = storage

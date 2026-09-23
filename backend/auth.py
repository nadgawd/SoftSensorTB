"""Who is calling: Supabase access-token verification.

The frontend signs every visitor in with Supabase (anonymously at first, by
email or Google later, keeping the same user id) and sends the access token as
``Authorization: Bearer``. Tokens are verified locally against the project's
published signing keys (JWKS), or with the legacy shared secret when
``SUPABASE_JWT_SECRET`` is set.

With ``AUTH_REQUIRED`` off (the local default) every request is the fixed
local user, so a single-user ``run.sh`` setup needs no Supabase at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool

LOCAL_USER_ID = "local"

_ASYMMETRIC_ALGORITHMS = ["ES256", "RS256", "EdDSA"]
_LEEWAY_S = 30


@dataclass(frozen=True)
class User:
    id: str
    is_anonymous: bool = False
    email: Optional[str] = None


LOCAL_USER = User(id=LOCAL_USER_ID)


def auth_required() -> bool:
    return os.getenv("AUTH_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _supabase_url() -> str:
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    if not url:
        raise RuntimeError("AUTH_REQUIRED is on but SUPABASE_URL is not set.")
    return url


@lru_cache(maxsize=4)
def _jwks_client(jwks_url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=3600, timeout=10)


def verify_token(token: str) -> User:
    """Decode and validate a Supabase access token. Raises ``jwt.InvalidTokenError``."""
    base = _supabase_url()
    options = {"require": ["exp", "sub", "aud"]}
    secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    header = jwt.get_unverified_header(token)
    if header.get("alg") == "HS256":
        if not secret:
            raise jwt.InvalidTokenError("HS256 token but SUPABASE_JWT_SECRET is not configured.")
        key, algorithms = secret, ["HS256"]
    else:
        key = _jwks_client(f"{base}/auth/v1/.well-known/jwks.json").get_signing_key_from_jwt(token).key
        algorithms = _ASYMMETRIC_ALGORITHMS
    claims = jwt.decode(
        token,
        key,
        algorithms=algorithms,
        audience="authenticated",
        issuer=f"{base}/auth/v1",
        leeway=_LEEWAY_S,
        options=options,
    )
    if claims.get("role") != "authenticated":
        raise jwt.InvalidTokenError("Token is not for a signed-in user.")
    return User(
        id=str(claims["sub"]),
        is_anonymous=bool(claims.get("is_anonymous", False)),
        email=claims.get("email") or None,
    )


def _bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


async def get_current_user(request: Request) -> User:
    """FastAPI dependency: the verified caller, or the local user when auth is off."""
    if not auth_required():
        return LOCAL_USER
    token = _bearer_token(request)
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Sign-in required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return await run_in_threadpool(verify_token, token)
    except jwt.PyJWKClientConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not reach the sign-in service. Try again shortly.",
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=401,
            detail="Your session has expired. Reload the page to sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

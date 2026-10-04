"""Passphrase + Google ID token login, HS256 JWTs, bearer-token dependency."""

from __future__ import annotations

import time
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt

from .settings import Settings, get_settings


ALGORITHM = "HS256"


def hash_passphrase(passphrase: str) -> str:
    return bcrypt.hashpw(passphrase.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_passphrase(passphrase: str, stored_hash: str) -> bool:
    if not stored_hash or not passphrase:
        return False
    try:
        return bcrypt.checkpw(passphrase.encode("utf-8"), stored_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_jwt(
    sub: str,
    email: str,
    display_name: str,
    settings: Settings,
) -> str:
    if not settings.roadguard_jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="jwt_secret_not_configured",
        )
    now = int(time.time())
    payload = {
        "sub": sub,
        "email": email,
        "name": display_name,
        "iat": now,
        "exp": now + settings.roadguard_jwt_ttl,
    }
    return jwt.encode(payload, settings.roadguard_jwt_secret, algorithm=ALGORITHM)


def decode_jwt(token: str, settings: Settings) -> dict[str, Any]:
    try:
        return jwt.decode(
            token, settings.roadguard_jwt_secret, algorithms=[ALGORITHM]
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid_jwt: {exc}",
        ) from exc


def verify_google_id_token(
    id_token: str,
    client_id: str,
    allowed_emails: set[str],
    http_client: Any | None = None,
) -> dict[str, Any]:
    """Verify a Google-issued ID token via the tokeninfo endpoint.

    `http_client` lets tests inject a stubbed httpx-compatible client.
    """
    close_after = False
    if http_client is None:
        import httpx

        http_client = httpx.Client(timeout=10.0)
        close_after = True
    try:
        resp = http_client.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": id_token},
        )
    finally:
        if close_after:
            http_client.close()
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"google_token_invalid: {resp.status_code}",
        )
    data = resp.json()
    aud = data.get("aud") or ""
    email = (data.get("email") or "").lower()
    if client_id and aud != client_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="google_aud_mismatch",
        )
    verified = data.get("email_verified")
    if verified in {"false", False}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="google_email_unverified",
        )
    if allowed_emails and email not in allowed_emails:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="google_email_not_allowed",
        )
    return {
        "sub": f"google:{data.get('sub')}",
        "email": email,
        "name": data.get("name") or email.split("@")[0] or "unknown",
    }


def get_current_user(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing_bearer_token",
        )
    token = auth_header.split(" ", 1)[1].strip()
    return decode_jwt(token, settings)

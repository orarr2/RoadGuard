"""slowapi limiter keyed by JWT subject when present, IP otherwise."""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from .auth import decode_jwt
from .settings import get_settings


def _rate_limit_key(request: Request) -> str:
    """Prefer the authenticated user; fall back to the remote address."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        try:
            payload = decode_jwt(token, get_settings())
        except Exception:
            payload = None
        if payload and payload.get("sub"):
            return f"user:{payload['sub']}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_rate_limit_key)

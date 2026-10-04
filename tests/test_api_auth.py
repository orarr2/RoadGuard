"""Auth helpers: passphrase hashing, JWT, Google ID token verification."""

import time

import httpx
import pytest

from api import auth
from api.settings import Settings


def _settings_with(**overrides) -> Settings:
    base = dict(
        roadguard_jwt_secret="test-secret",
        roadguard_jwt_ttl=3600,
        google_oauth_client_id="",
        roadguard_passphrase_hash="",
        roadguard_allowed_emails="orarbeli1@gmail.com",
    )
    base.update(overrides)
    return Settings(**base)


def test_hash_and_verify_passphrase_roundtrip():
    stored = auth.hash_passphrase("secret-phrase")
    assert auth.verify_passphrase("secret-phrase", stored) is True
    assert auth.verify_passphrase("wrong", stored) is False
    assert auth.verify_passphrase("", stored) is False


def test_verify_passphrase_rejects_corrupt_hash():
    assert auth.verify_passphrase("secret", "not-a-bcrypt-hash") is False


def test_create_and_decode_jwt():
    settings = _settings_with()
    token = auth.create_jwt("sub-1", "a@b", "Alice", settings)
    payload = auth.decode_jwt(token, settings)
    assert payload["sub"] == "sub-1"
    assert payload["email"] == "a@b"
    assert payload["name"] == "Alice"


def test_decode_jwt_rejects_bogus_token():
    settings = _settings_with()
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        auth.decode_jwt("not.a.jwt", settings)


def test_verify_google_id_token_happy_path():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "aud": "cid",
                "sub": "123",
                "email": "orarbeli1@gmail.com",
                "email_verified": "true",
                "name": "Or",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    payload = auth.verify_google_id_token(
        id_token="idtoken",
        client_id="cid",
        allowed_emails={"orarbeli1@gmail.com"},
        http_client=client,
    )
    assert payload["email"] == "orarbeli1@gmail.com"
    assert payload["sub"] == "google:123"


def test_verify_google_id_token_aud_mismatch():
    def handler(request):
        return httpx.Response(200, json={"aud": "other", "sub": "1", "email": "a@b", "email_verified": "true"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        auth.verify_google_id_token(
            id_token="x",
            client_id="cid",
            allowed_emails={"a@b"},
            http_client=client,
        )
    assert exc_info.value.detail == "google_aud_mismatch"


def test_verify_google_id_token_not_allowed_email():
    def handler(request):
        return httpx.Response(200, json={"aud": "cid", "sub": "1", "email": "stranger@example.com", "email_verified": "true"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        auth.verify_google_id_token(
            id_token="x",
            client_id="cid",
            allowed_emails={"orarbeli1@gmail.com"},
            http_client=client,
        )
    assert exc_info.value.status_code == 403

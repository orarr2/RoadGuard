"""Shared pytest fixtures.

Ensures `src/` is on sys.path so tests can `import agent`, `import api`,
`import storage`, `import db`, and `import jobs` the same way the
runtime does. Also defines a few throwaway fixtures used across files.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

try:  # pydantic 2.13 only
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimum env the API expects. Individual tests may override."""
    monkeypatch.setenv("ROADGUARD_JWT_SECRET", "test-secret-not-for-prod")
    monkeypatch.setenv("ROADGUARD_ALLOWED_EMAILS", "orarbeli1@gmail.com")
    monkeypatch.setenv("ROADGUARD_PASSPHRASE_HASH", "")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("ROADGUARD_LOCAL_LLM_MODEL", "")
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "")
    monkeypatch.setenv("ROADGUARD_DB_PATH", ":memory:")
    # pydantic-settings caches via lru_cache; clear before each test
    from api.settings import reset_settings_cache

    reset_settings_cache()

    # slowapi keeps a module-level in-memory storage. Reset before each
    # test so the per-minute limits are counted from scratch.
    from api.rate_limits import limiter

    storage = getattr(limiter, "_storage", None)
    for attr in ("storage", "events", "locks", "expirations", "base_exp"):
        bucket = getattr(storage, attr, None)
        if bucket is not None and hasattr(bucket, "clear"):
            bucket.clear()


@pytest.fixture()
def passphrase_hash() -> str:
    import bcrypt

    return bcrypt.hashpw(b"test-passphrase", bcrypt.gensalt(rounds=4)).decode()

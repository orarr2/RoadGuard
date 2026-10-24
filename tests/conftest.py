"""Shared pytest fixtures.

Puts the repository root on sys.path so the tests import roadguard_agent
and roadguard_api the same way the service does, and sets the minimum
environment the API expects.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

try:  # pydantic 2.13 only
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Minimum env the API expects. Individual tests may override."""
    monkeypatch.setenv("ROADGUARD_JWT_SECRET", "test-secret-not-for-prod")
    monkeypatch.setenv("ROADGUARD_ALLOWED_EMAILS", "tester@example.com")
    monkeypatch.setenv("ROADGUARD_PASSPHRASE_HASH", "")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("ROADGUARD_DB_PATH", ":memory:")
    # Point every on-disk side effect at a per-test tempdir; the defaults
    # are VM paths that are permission-denied on CI and dev machines.
    _test_tmp = tmp_path_factory.mktemp("roadguard_test")
    monkeypatch.setenv("ROADGUARD_CHECKPOINT_PATH", str(_test_tmp / "checkpoints.sqlite"))
    monkeypatch.setenv("ROADGUARD_RAG_PATH", str(_test_tmp / "chroma"))
    # pydantic-settings caches via lru_cache; clear before each test
    from roadguard_agent import reset_settings_cache

    reset_settings_cache()

    # slowapi keeps a module-level in-memory storage. Reset before each
    # test so the per-minute limits are counted from scratch.
    from roadguard_api import limiter

    storage = getattr(limiter, "_storage", None)
    for attr in ("storage", "events", "locks", "expirations", "base_exp"):
        bucket = getattr(storage, attr, None)
        if bucket is not None and hasattr(bucket, "clear"):
            bucket.clear()


@pytest.fixture()
def passphrase_hash() -> str:
    import bcrypt

    return bcrypt.hashpw(b"test-passphrase", bcrypt.gensalt(rounds=4)).decode()

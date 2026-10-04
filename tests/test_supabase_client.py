"""Supabase client against an httpx MockTransport."""

import datetime

import httpx
import pytest

from storage.supabase_client import SupabaseClient, SupabaseError


def _client_with(handler) -> SupabaseClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return SupabaseClient(
        url="https://proj.supabase.co",
        service_role_key="srkey",
        bucket="roadguard-hazards",
        http_client=http,
    )


def test_new_object_key_shape():
    sb = _client_with(lambda r: httpx.Response(200))
    key = sb.new_object_key("passphrase:owner@example.com")
    today = datetime.datetime.utcnow().strftime("%Y%m%d")
    assert key.startswith(today + "/")
    assert key.endswith(".jpg")
    assert key.count("/") == 2


def test_new_object_key_sanitizes_user():
    sb = _client_with(lambda r: httpx.Response(200))
    key = sb.new_object_key("google:12345/sub:/x")
    assert "/" not in key.split("/", 2)[1]  # middle segment has no stray slashes


def test_presign_upload_qualifies_relative_url():
    def handler(request):
        return httpx.Response(200, json={"url": "/storage/v1/object/upload/signed/x", "token": "T"})

    sb = _client_with(handler)
    out = sb.presign_upload("20261004/u/x.jpg")
    assert out["upload_url"].startswith("https://proj.supabase.co")
    assert out["token"] == "T"


def test_signed_read_url_qualifies_relative_url():
    def handler(request):
        return httpx.Response(200, json={"signedURL": "/storage/v1/object/signed/x?token=T"})

    sb = _client_with(handler)
    url = sb.signed_read_url("20261004/u/x.jpg")
    assert url.startswith("https://proj.supabase.co")


def test_delete_object_treats_404_as_success():
    def handler(request):
        return httpx.Response(404, json={"message": "not found"})

    sb = _client_with(handler)
    assert sb.delete_object("20261004/u/x.jpg") is True


def test_delete_object_raises_on_5xx():
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    sb = _client_with(handler)
    with pytest.raises(SupabaseError):
        sb.delete_object("20261004/u/x.jpg")


def test_list_expired_returns_only_old_objects():
    now = datetime.datetime.now(datetime.timezone.utc)
    old = (now - datetime.timedelta(hours=72)).isoformat().replace("+00:00", "Z")
    fresh = (now - datetime.timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    def handler(request):
        return httpx.Response(
            200,
            json=[
                {"name": "old.jpg", "created_at": old},
                {"name": "fresh.jpg", "created_at": fresh},
            ],
        )

    sb = _client_with(handler)
    expired = sb.list_expired(ttl_hours=48)
    assert [e.name for e in expired] == ["old.jpg"]


def test_init_requires_url_and_key():
    with pytest.raises(ValueError):
        SupabaseClient(url="", service_role_key="k", bucket="b")
    with pytest.raises(ValueError):
        SupabaseClient(url="x", service_role_key="", bucket="b")
    with pytest.raises(ValueError):
        SupabaseClient(url="x", service_role_key="k", bucket="")

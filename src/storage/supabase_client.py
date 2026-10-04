"""Minimal Supabase Storage client used by the VM.

Only the four operations the VM actually needs are implemented, directly
against the REST endpoints: presign an upload, mint a short-lived signed
read URL, delete an object, and list objects whose server-side
`created_at` is older than a cutoff (used as a safety net alongside the
DB-driven TTL cleanup).

The service role key is read from the environment; nothing here is
callable from the browser.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import httpx


DEFAULT_UPLOAD_EXPIRES_S = 900  # 15 minutes, matches Supabase default
DEFAULT_READ_EXPIRES_S = 3600  # 1 hour
LIST_PAGE_SIZE = 100
LIST_MAX_PAGES = 50


@dataclass
class ExpiredObject:
    name: str
    created_at: str


class SupabaseClient:
    """Thin REST client for the Supabase Storage API.

    `http_client` is injectable so tests can swap in an httpx MockTransport.
    """

    def __init__(
        self,
        url: str,
        service_role_key: str,
        bucket: str,
        http_client: Optional[httpx.Client] = None,
        timeout_s: float = 15.0,
    ) -> None:
        if not url or not service_role_key or not bucket:
            raise ValueError("supabase_client: url, service_role_key, and bucket are all required")
        self.url = url.rstrip("/")
        self.service_role_key = service_role_key
        self.bucket = bucket
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout_s)

    def __enter__(self) -> "SupabaseClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    # ---- Helpers ---------------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.service_role_key}",
            "apikey": self.service_role_key,
        }

    @staticmethod
    def new_object_key(user_id: str, extension: str = "jpg") -> str:
        """Deterministic path: YYYYMMDD/user/uuid.ext. Scoped per user."""
        today = datetime.datetime.utcnow().strftime("%Y%m%d")
        safe_user = (user_id or "anon").replace(":", "_").replace("/", "_")
        safe_user = safe_user[:40] or "anon"
        return f"{today}/{safe_user}/{uuid.uuid4().hex}.{extension.lstrip('.')}"

    # ---- Operations ------------------------------------------------------

    def presign_upload(
        self,
        object_key: str,
        expires_in: int = DEFAULT_UPLOAD_EXPIRES_S,
    ) -> dict[str, Any]:
        """Return a signed URL and token the browser PUTs the image to."""
        endpoint = f"{self.url}/storage/v1/object/upload/sign/{self.bucket}/{object_key}"
        resp = self._http.post(endpoint, headers=self._headers, json={})
        if resp.status_code >= 400:
            raise SupabaseError(
                f"presign_upload_failed: {resp.status_code} {resp.text[:200]}"
            )
        data = resp.json()
        upload_url = data.get("url") or data.get("signedURL") or data.get("signedUrl")
        if upload_url and upload_url.startswith("/"):
            upload_url = f"{self.url}{upload_url}"
        return {
            "upload_url": upload_url,
            "object_key": object_key,
            "token": data.get("token"),
            "expires_in": expires_in,
        }

    def signed_read_url(
        self,
        object_key: str,
        expires_in: int = DEFAULT_READ_EXPIRES_S,
    ) -> str:
        """Mint a short-lived download URL the webapp can embed in <img>."""
        endpoint = f"{self.url}/storage/v1/object/sign/{self.bucket}/{object_key}"
        resp = self._http.post(
            endpoint,
            headers=self._headers,
            json={"expiresIn": int(expires_in)},
        )
        if resp.status_code >= 400:
            raise SupabaseError(
                f"signed_read_url_failed: {resp.status_code} {resp.text[:200]}"
            )
        data = resp.json()
        signed = data.get("signedURL") or data.get("signedUrl") or data.get("url")
        if signed and signed.startswith("/"):
            signed = f"{self.url}{signed}"
        if not signed:
            raise SupabaseError(f"signed_read_url_empty: {data!r}")
        return signed

    def delete_object(self, object_key: str) -> bool:
        """Delete one object. Returns True on 2xx or 404 (idempotent)."""
        endpoint = f"{self.url}/storage/v1/object/{self.bucket}/{object_key}"
        resp = self._http.delete(endpoint, headers=self._headers)
        if resp.status_code == 404:
            return True
        if 200 <= resp.status_code < 300:
            return True
        raise SupabaseError(
            f"delete_object_failed: {resp.status_code} {resp.text[:200]}"
        )

    def list_expired(
        self,
        ttl_hours: int,
        prefix: str = "",
        page_size: int = LIST_PAGE_SIZE,
        max_pages: int = LIST_MAX_PAGES,
    ) -> list[ExpiredObject]:
        """Scan the bucket and return objects whose server-side created_at
        is older than `ttl_hours` ago.

        This is a safety net for orphans that the DB-driven cleanup missed.
        Walks a flat prefix; nested folders need to be enumerated by the
        caller for complete coverage.
        """
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            hours=int(ttl_hours)
        )
        expired: list[ExpiredObject] = []
        offset = 0
        for _ in range(max_pages):
            endpoint = f"{self.url}/storage/v1/object/list/{self.bucket}"
            resp = self._http.post(
                endpoint,
                headers=self._headers,
                json={
                    "prefix": prefix,
                    "limit": int(page_size),
                    "offset": int(offset),
                    "sortBy": {"column": "created_at", "order": "asc"},
                },
            )
            if resp.status_code >= 400:
                raise SupabaseError(
                    f"list_expired_failed: {resp.status_code} {resp.text[:200]}"
                )
            rows = resp.json() or []
            if not isinstance(rows, list) or not rows:
                break
            for row in rows:
                name = row.get("name")
                created_raw = row.get("created_at")
                if not name or not created_raw:
                    continue
                try:
                    created = datetime.datetime.fromisoformat(
                        created_raw.replace("Z", "+00:00")
                    )
                except ValueError:
                    continue
                if created.tzinfo is None:
                    created = created.replace(tzinfo=datetime.timezone.utc)
                if created <= cutoff:
                    full_name = f"{prefix.rstrip('/')}/{name}" if prefix else name
                    expired.append(
                        ExpiredObject(name=full_name, created_at=created_raw)
                    )
            if len(rows) < page_size:
                break
            offset += page_size
        return expired


class SupabaseError(RuntimeError):
    """Raised when a Supabase REST call returns an unexpected status."""

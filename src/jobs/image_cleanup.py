"""Nightly image cleanup: purge Supabase objects past the DB-tracked TTL.

Reads candidates from `hazards.image_expires_at < now`, deletes each
Supabase object, and null-outs the row's `image_object_key`. Idempotent
by design: a 404 from Supabase is treated as success, and the row
update is safe to re-run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve()
if str(_HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[1]))

from api.settings import Settings, get_settings  # noqa: E402
from db.models import Database  # noqa: E402
from storage.supabase_client import SupabaseClient, SupabaseError  # noqa: E402


LOGGER = logging.getLogger("roadguard.jobs.image_cleanup")


def run(
    settings: Settings,
    *,
    dry_run: bool = False,
    db: Optional[Database] = None,
    supabase: Optional[SupabaseClient] = None,
    now_ts: Optional[float] = None,
) -> dict[str, int]:
    """Delete Supabase objects whose TTL has passed and clear their keys.

    Returns a counts dict so a test or the systemd unit can log the result.
    `db`, `supabase`, and `now_ts` are injectable for tests.
    """
    now = now_ts if now_ts is not None else time.time()
    owns_db = db is None
    if db is None:
        db = Database(settings.roadguard_db_path)

    candidates = db.list_expired_images(now_ts=now, limit=1000)
    counts = {"scanned": len(candidates), "deleted": 0, "failed": 0}
    LOGGER.info("cleanup_scan: %d past-ttl candidates", counts["scanned"])
    if not candidates:
        if owns_db:
            db.close()
        return counts

    owns_supabase = supabase is None
    if supabase is None:
        if not settings.supabase_url or not settings.supabase_service_role_key:
            LOGGER.error("supabase_not_configured; skipping")
            if owns_db:
                db.close()
            counts["failed"] = counts["scanned"]
            return counts
        supabase = SupabaseClient(
            url=settings.supabase_url,
            service_role_key=settings.supabase_service_role_key,
            bucket=settings.supabase_bucket,
        )

    try:
        for cand in candidates:
            key = cand["image_object_key"]
            hazard_id = cand["hazard_id"]
            if dry_run:
                LOGGER.info("would_delete: %s (hazard=%s)", key, hazard_id)
                continue
            try:
                supabase.delete_object(key)
                db.clear_image_object_key(hazard_id)
                counts["deleted"] += 1
            except SupabaseError as exc:
                LOGGER.warning("delete_failed: %s %s", key, exc)
                counts["failed"] += 1
    finally:
        if owns_supabase:
            supabase.close()
        if owns_db:
            db.close()
    LOGGER.info(
        "cleanup_done: deleted=%d failed=%d scanned=%d",
        counts["deleted"],
        counts["failed"],
        counts["scanned"],
    )
    return counts


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Purge expired Supabase images.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    counts = run(get_settings(), dry_run=args.dry_run)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

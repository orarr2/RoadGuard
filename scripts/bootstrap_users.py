"""Bootstrap the step-1 auth upgrade users: orpsp8 (admin) + batch@internal.

Reads credentials from env OR CLI args and inserts the two users
idempotently. Safe to run multiple times.

Usage (on the VM):
    sudo -u roadguard env ROADGUARD_DB_PATH=/var/lib/roadguard/roadguard.db \\
        OWNER_PASSWORD='<the real password>' \\
        /opt/roadguard/venv/bin/python3 /opt/roadguard/repo/scripts/bootstrap_users.py

The owner password must satisfy the policy (8+ chars, upper, lower, special).
On conflict the script does an UPSERT and overwrites the password hash; the
rest of the row (username, is_admin flag) is preserved.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from roadguard_agent import (  # noqa: E402
    Database, get_settings, reset_settings_cache,
    hash_password, validate_password, validate_username,
)


def _ensure_user(db, user_id, username, email, display_name,
                 password_hash, is_admin, is_internal):
    existing = db.get_user(user_id)
    db.create_user_full(
        user_id=user_id,
        username=username,
        email=email,
        display_name=display_name,
        password_hash=password_hash,
        google_sub=None,
        is_admin=is_admin,
        is_internal=is_internal,
    )
    action = "updated" if existing else "created"
    print(f"  {action}: id={user_id} username={username} "
          f"admin={is_admin} internal={is_internal}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--owner-username", default="orpsp8")
    parser.add_argument("--owner-email", default=os.environ.get("OWNER_EMAIL",
                                                                "orarbeli1@gmail.com"))
    parser.add_argument("--owner-password", default=os.environ.get("OWNER_PASSWORD", ""),
                        help="Plaintext password; validated against the policy, then bcrypted.")
    parser.add_argument("--owner-hash", default=os.environ.get("OWNER_HASH", ""),
                        help="Already-bcrypted hash; use when the plaintext is not on hand "
                             "(e.g. migrate the existing ROADGUARD_PASSPHRASE_HASH).")
    parser.add_argument("--batch-username", default="batch")
    args = parser.parse_args()

    reset_settings_cache()
    settings = get_settings()
    print(f"db: {settings.roadguard_db_path}")

    ok, reason = validate_username(args.owner_username)
    if not ok:
        print(f"ERROR: owner username invalid: {reason}", file=sys.stderr)
        return 2

    if args.owner_password:
        ok, reason = validate_password(args.owner_password)
        if not ok:
            print(f"ERROR: owner password rejects: {reason}", file=sys.stderr)
            return 2
        owner_hash = hash_password(args.owner_password)
    elif args.owner_hash:
        if not args.owner_hash.startswith("$2"):
            print("ERROR: --owner-hash does not look like a bcrypt hash ($2...)",
                  file=sys.stderr)
            return 2
        owner_hash = args.owner_hash
    else:
        print("ERROR: provide --owner-password OR --owner-hash "
              "(migrate from ROADGUARD_PASSPHRASE_HASH)",
              file=sys.stderr)
        return 2

    with Database(settings.roadguard_db_path) as db:
        existing = db.get_user_by_username(args.owner_username)
        owner_id = existing.id if existing else f"u_owner_{args.owner_username}"
        _ensure_user(db, owner_id, args.owner_username, args.owner_email,
                     args.owner_username, owner_hash,
                     is_admin=True, is_internal=False)

        existing_batch = db.get_user_by_username(args.batch_username)
        batch_id = existing_batch.id if existing_batch else "u_batch_internal"
        # The batch user has no password - internal users only authenticate via
        # a JWT the operator mints (scripts/mint_batch_token.py would go here).
        # Set an unusable hash so username+password login cannot work on it.
        _ensure_user(db, batch_id, args.batch_username,
                     "batch@internal.local", "Batch runner",
                     password_hash="!unusable_internal_user",
                     is_admin=False, is_internal=True)

    print(f"total users now: {db.count_users()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

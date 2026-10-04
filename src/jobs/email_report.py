"""Daily cron: render and send a previous-24h hazard summary over Gmail SMTP.

Picks up every hazard reported in the last 24 hours, groups them by class
and agent decision, and lists the urgent and human_review cases with a
link into /hazards/{id}. Sent only to recipients that have opted in
through the settings screen; the daily email is off by default so the
mailbox stays clean during development.
"""

from __future__ import annotations

import argparse
import datetime
import html
import logging
import smtplib
import sys
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve()
if str(_HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[1]))

from api.settings import Settings, get_settings  # noqa: E402
from db.models import Database, Hazard  # noqa: E402


LOGGER = logging.getLogger("roadguard.jobs.email_report")

DEFAULT_SUMMARY_WINDOW_H = 24
HAZARD_LINK_BASE = "https://orarr2.github.io/RoadGuard/#/hazards/"


def _fmt_time(ts: float) -> str:
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M UTC")


def _bucket_counts(counts: dict[str, int]) -> dict[str, dict[str, int]]:
    """Reshape `{class:decision: n}` into `{class: {decision: n}}`."""
    out: dict[str, dict[str, int]] = {}
    for key, n in counts.items():
        if ":" not in key:
            continue
        cls, decision = key.split(":", 1)
        out.setdefault(cls, {})[decision] = n
    return out


def build_summary_html(
    db: Database,
    *,
    window_hours: int = DEFAULT_SUMMARY_WINDOW_H,
    now_ts: Optional[float] = None,
) -> str:
    now = now_ts if now_ts is not None else time.time()
    since = now - window_hours * 3600
    counts = db.count_hazards_since(since_ts=since)
    buckets = _bucket_counts(counts)
    critical = db.recent_critical_hazards(since_ts=since, limit=100)

    class_rows = ""
    class_order = ["D00", "D10", "D20", "D40"]
    decisions = ["urgent", "standard", "needs_more_evidence", "human_review"]
    for cls in class_order:
        row = buckets.get(cls, {})
        if not row:
            continue
        total = sum(row.values())
        cells = "".join(
            f"<td style='padding:4px 10px;text-align:right'>{row.get(d, 0)}</td>"
            for d in decisions
        )
        class_rows += (
            f"<tr><td style='padding:4px 10px'><b>{cls}</b></td>"
            f"{cells}"
            f"<td style='padding:4px 10px;text-align:right'>{total}</td></tr>"
        )
    if not class_rows:
        class_rows = (
            "<tr><td colspan='6' style='padding:12px;text-align:center;color:#666'>"
            "No hazards submitted in this window.</td></tr>"
        )

    critical_rows = ""
    for hz in critical:
        link = f"{HAZARD_LINK_BASE}{hz.id}"
        decision = hz.agent_decision
        colour = "#c0392b" if decision == "urgent" else "#8e44ad"
        rationale = html.escape(hz.agent_rationale or "")[:220]
        critical_rows += (
            "<tr>"
            f"<td style='padding:6px 10px'><a href='{html.escape(link)}'>{html.escape(hz.id)}</a></td>"
            f"<td style='padding:6px 10px'>{_fmt_time(hz.reported_at)}</td>"
            f"<td style='padding:6px 10px'>{html.escape(hz.hazard_class)}</td>"
            f"<td style='padding:6px 10px'>{html.escape(hz.road_type)}</td>"
            f"<td style='padding:6px 10px'>{html.escape(hz.lane_blocked)}</td>"
            f"<td style='padding:6px 10px;color:{colour};font-weight:600'>{html.escape(decision)}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{hz.priority}</td>"
            f"<td style='padding:6px 10px;color:#333'>{rationale}</td>"
            "</tr>"
        )
    if not critical_rows:
        critical_rows = (
            "<tr><td colspan='8' style='padding:12px;text-align:center;color:#666'>"
            "No urgent or human-review cases in this window.</td></tr>"
        )

    header_date = datetime.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d")
    return f"""<!doctype html>
<html><body style='font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:960px;margin:0 auto;color:#222'>
<h2 style='margin:0 0 4px 0'>RoadGuard Daily Summary - {header_date}</h2>
<p style='margin:0 0 20px 0;color:#555'>
Previous {window_hours} hours. External actions (contractor dispatch,
road closure) remain mocked and still require a human to execute.
</p>
<h3 style='margin:16px 0 8px 0'>Submissions by class</h3>
<table style='border-collapse:collapse;border:1px solid #ddd;width:100%'>
<thead style='background:#f5f6f7'>
<tr>
<th style='padding:6px 10px;text-align:left'>Class</th>
<th style='padding:6px 10px;text-align:right'>Urgent</th>
<th style='padding:6px 10px;text-align:right'>Standard</th>
<th style='padding:6px 10px;text-align:right'>Needs evidence</th>
<th style='padding:6px 10px;text-align:right'>Human review</th>
<th style='padding:6px 10px;text-align:right'>Total</th>
</tr>
</thead>
<tbody>{class_rows}</tbody>
</table>

<h3 style='margin:20px 0 8px 0'>Urgent and human-review cases</h3>
<table style='border-collapse:collapse;border:1px solid #ddd;width:100%;font-size:14px'>
<thead style='background:#f5f6f7'>
<tr>
<th style='padding:6px 10px;text-align:left'>Hazard</th>
<th style='padding:6px 10px;text-align:left'>Reported</th>
<th style='padding:6px 10px;text-align:left'>Class</th>
<th style='padding:6px 10px;text-align:left'>Road</th>
<th style='padding:6px 10px;text-align:left'>Lane</th>
<th style='padding:6px 10px;text-align:left'>Decision</th>
<th style='padding:6px 10px;text-align:right'>Priority</th>
<th style='padding:6px 10px;text-align:left'>Rationale</th>
</tr>
</thead>
<tbody>{critical_rows}</tbody>
</table>

<p style='margin-top:20px;font-size:12px;color:#777'>
Reminder: RoadGuard is a decision-support system. These are
recommendations, not actions. A human must review every urgent and
human-review item before any field work is initiated.
</p>
</body></html>
"""


def send_summary(
    settings: Settings,
    html_body: str,
    recipients: list[str],
    smtp_client: Optional[smtplib.SMTP] = None,
    now_ts: Optional[float] = None,
) -> int:
    if not recipients:
        LOGGER.info("no_recipients; skipping send")
        return 0
    if not settings.email_address or not settings.email_app_password:
        LOGGER.error("smtp_not_configured")
        return 0
    subject_date = datetime.datetime.utcfromtimestamp(
        now_ts if now_ts is not None else time.time()
    ).strftime("%Y-%m-%d")
    msg = EmailMessage()
    msg["Subject"] = f"RoadGuard Daily Summary - {subject_date}"
    msg["From"] = settings.email_address
    msg["To"] = ", ".join(recipients)
    msg.set_content(
        "This is the HTML daily summary. Enable HTML in your mail client."
    )
    msg.add_alternative(html_body, subtype="html")

    owns_smtp = smtp_client is None
    try:
        if smtp_client is None:
            smtp_client = smtplib.SMTP(
                settings.email_smtp_host, settings.email_smtp_port, timeout=30
            )
            smtp_client.ehlo()
            smtp_client.starttls()
            smtp_client.login(settings.email_address, settings.email_app_password)
        smtp_client.send_message(msg)
    finally:
        if owns_smtp and smtp_client is not None:
            try:
                smtp_client.quit()
            except Exception:
                pass
    return len(recipients)


def run(
    settings: Settings,
    *,
    dry_run: bool = False,
    db: Optional[Database] = None,
    smtp_client: Optional[smtplib.SMTP] = None,
    recipients_override: Optional[list[str]] = None,
    window_hours: int = DEFAULT_SUMMARY_WINDOW_H,
    now_ts: Optional[float] = None,
) -> dict[str, int]:
    """Render the summary and send to opted-in recipients.

    Everything is injectable so tests can run without the DB or SMTP.
    """
    owns_db = db is None
    if db is None:
        db = Database(settings.roadguard_db_path)
    try:
        html_body = build_summary_html(
            db, window_hours=window_hours, now_ts=now_ts
        )
        if recipients_override is not None:
            recipients = recipients_override
        else:
            recipients = [u.email for u in db.report_recipients() if u.email]
            if settings.report_recipient and settings.report_recipient not in recipients:
                recipients.append(settings.report_recipient)
        LOGGER.info(
            "summary: html_len=%d recipients=%d dry_run=%s",
            len(html_body),
            len(recipients),
            dry_run,
        )
        if dry_run:
            return {"html_len": len(html_body), "recipients": len(recipients), "sent": 0}
        sent = send_summary(
            settings=settings,
            html_body=html_body,
            recipients=recipients,
            smtp_client=smtp_client,
            now_ts=now_ts,
        )
        return {"html_len": len(html_body), "recipients": len(recipients), "sent": sent}
    finally:
        if owns_db:
            db.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Send the daily hazard summary.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--window-hours", type=int, default=DEFAULT_SUMMARY_WINDOW_H)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    counts = run(
        get_settings(),
        dry_run=args.dry_run,
        window_hours=args.window_hours,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

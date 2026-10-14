"""Email a supervisor when an approval row is created.

Decoupled from email_report.py so the daily summary and the per-event
approval mail can evolve independently. Shares the Settings-based
SMTP configuration.
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Optional

from api.settings import Settings


LOGGER = logging.getLogger("roadguard.jobs.approval_email")


def _build_html(
    approval_id: str,
    hazard_id: str,
    action: str,
    priority: int,
    rationale: str,
    cited_sources: list[str],
    base_url: str,
) -> str:
    """Approval email body with Approve / Deny links.

    The links are plain HTTP GETs so the mail client doesn't need to
    POST. The GET endpoint only renders the UI; the decision itself is
    a POST that the UI performs.
    """
    approve_url = f"{base_url.rstrip('/')}/approvals/{approval_id}"
    cited = (
        "<ul>" + "".join(f"<li>{s}</li>" for s in cited_sources) + "</ul>"
        if cited_sources
        else "<em>no cited sources</em>"
    )
    return f"""\
<!doctype html>
<html><body style="font-family:system-ui,sans-serif;max-width:600px">
<h2 style="color:#B91C1C">RoadGuard · {action.upper()} awaiting approval</h2>
<p>A hazard was triaged as <strong>{action}</strong> with priority
<strong>{priority}</strong> and needs a human decision before anything
downstream can act.</p>

<p><strong>Hazard</strong>: <code>{hazard_id}</code></p>
<p><strong>Rationale</strong>: {rationale}</p>
<p><strong>Cited sources</strong>:</p>
{cited}

<p style="margin-top:24px">
  <a href="{approve_url}"
     style="background:#1B5E20;color:white;padding:12px 24px;
            text-decoration:none;border-radius:4px">
    Open approval page
  </a>
</p>

<hr/>
<p style="font-size:12px;color:#6B7280">
Nothing has been dispatched. No contractor has been contacted. No road
has been closed. This email is a request for a human to approve or
deny the proposed action.
</p>
</body></html>
"""


def send_approval_request(
    settings: Settings,
    *,
    approval_id: str,
    hazard_id: str,
    action: str,
    priority: int,
    rationale: str,
    cited_sources: list[str],
    base_url: str,
    recipient: Optional[str] = None,
    smtp_client: Optional[smtplib.SMTP] = None,
) -> bool:
    """Send a single approval-request email. Returns True on success.

    When smtp_client is injected (tests), no network call happens
    beyond ``smtp_client.send_message``. In production the function
    builds its own SMTP client using the Settings SMTP block.
    """
    if not settings.email_address or not settings.email_app_password:
        LOGGER.warning("smtp_not_configured; approval email skipped")
        return False

    to = recipient or settings.report_recipient
    if not to:
        LOGGER.warning("no_approval_recipient; email skipped")
        return False

    msg = EmailMessage()
    msg["Subject"] = f"RoadGuard approval needed · {action.upper()} · {hazard_id}"
    msg["From"] = settings.email_address
    msg["To"] = to
    msg.set_content(
        f"RoadGuard needs your approval for action={action} on {hazard_id}. "
        f"Open {base_url.rstrip('/')}/approvals/{approval_id} to decide."
    )
    msg.add_alternative(
        _build_html(
            approval_id=approval_id,
            hazard_id=hazard_id,
            action=action,
            priority=priority,
            rationale=rationale,
            cited_sources=cited_sources,
            base_url=base_url,
        ),
        subtype="html",
    )

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
    except Exception as exc:
        LOGGER.error("approval_email_failed: %s", exc)
        return False
    finally:
        if owns_smtp and smtp_client is not None:
            try:
                smtp_client.quit()
            except Exception:
                pass
    return True

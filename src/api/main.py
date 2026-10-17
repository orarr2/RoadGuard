"""FastAPI shell wiring the agent, auth, rate limits, and placeholder routes.

Routes that depend on later-sprint components (Supabase client, SQLite
tables, ChromaDB index, daily report renderer) return 501 Not Implemented
until those commits land.
"""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from agent import loop, tools
from agent.multiagent.adapter import multiagent_decide, multiagent_decide_live

from .auth import (
    create_jwt,
    get_current_user,
    verify_google_id_token,
    verify_passphrase,
)
from .decision_cache import DecisionCache, get_decision_cache
from .rate_limits import limiter
from .schemas import (
    AgentVerdict,
    HazardDetail,
    HazardHistoryResponse,
    HazardSubmit,
    HazardSubmitResponse,
    HealthResponse,
    LiveHazardResponse,
    LiveHazardSubmit,
    LoginRequest,
    LoginResponse,
    Me,
    PresignRequest,
    PresignResponse,
    RagQueryRequest,
    RagQueryResponse,
)
from .settings import Settings, get_settings


LOGGER = logging.getLogger("roadguard.api")
VERSION = "0.1.0"


def _render_approval_html(row: Any) -> str:
    """Minimal HTML page for the approval gate.

    Shows the pending verdict + Approve/Deny form, or a frozen status
    line once the decision has been recorded. No JavaScript, so it
    renders in any mail client's preview and any browser.
    """
    import html as _html
    import json as _json

    sources = []
    try:
        sources = _json.loads(row.cited_sources_json or "[]")
    except Exception:
        sources = []
    cited_html = (
        "<ul>" + "".join(f"<li>{_html.escape(s)}</li>" for s in sources) + "</ul>"
        if sources
        else "<em>no cited sources</em>"
    )
    if row.status == "pending":
        decision_block = f"""\
<form method="POST" action="/approvals/{row.id}/decide"
      style="display:flex;gap:12px;margin-top:16px">
  <button name="decision" value="approved"
          style="background:#1B5E20;color:white;border:none;padding:12px 24px;
                 font-size:16px;border-radius:4px;cursor:pointer">
    Approve
  </button>
  <button name="decision" value="denied"
          style="background:#B91C1C;color:white;border:none;padding:12px 24px;
                 font-size:16px;border-radius:4px;cursor:pointer">
    Deny
  </button>
</form>
"""
    else:
        color = "#1B5E20" if row.status == "approved" else "#B91C1C"
        decision_block = (
            f'<p style="margin-top:16px"><strong style="color:{color}">'
            f"Status: {_html.escape(row.status)}</strong> · decided by "
            f"{_html.escape(row.decided_by or 'unknown')}</p>"
        )

    return f"""\
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>RoadGuard approval · {_html.escape(row.id)}</title></head>
<body style="font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;
             padding:0 16px;color:#111">
<h2 style="color:#B91C1C">RoadGuard · {_html.escape(row.action).upper()} awaiting approval</h2>
<p><strong>Hazard</strong>: <code>{_html.escape(row.hazard_id)}</code></p>
<p><strong>Priority</strong>: {row.priority}</p>
<p><strong>Rationale</strong>: {_html.escape(row.rationale)}</p>
<p><strong>Cited sources</strong>:</p>
{cited_html}
{decision_block}
<hr style="margin-top:32px">
<p style="font-size:12px;color:#6B7280">
Nothing has been dispatched. No contractor has been contacted. No road
has been closed. Approve creates the local draft work order; Deny
closes the case without action.
</p>
</body></html>
"""
CORS_ORIGINS = [
    "https://orarr2.github.io",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]


# ---------------------------------------------------------------------------
# Readiness helpers
# ---------------------------------------------------------------------------


def _db_ready(settings: Settings) -> bool:
    return os.path.isfile(settings.roadguard_db_path)


def _agent_ready(settings: Settings) -> bool:
    return bool(settings.groq_api_key)


def _fallback_ready(settings: Settings) -> bool:
    path = settings.roadguard_local_llm_model
    return bool(path) and os.path.isfile(path)


def _rag_ready(settings: Settings) -> bool:
    return bool(settings.roadguard_rag_path) and os.path.isdir(settings.roadguard_rag_path)


def _supabase_ready(settings: Settings) -> bool:
    return bool(settings.supabase_url) and bool(settings.supabase_service_role_key)


# ---------------------------------------------------------------------------
# Request-body helpers (dedupe between /hazards/submit and /hazards/live)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Reported-at timestamp in the shape every agent path expects."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dump_image_quality(body: Any, *, missing: Any) -> Any:
    """Serialize a HazardSubmit/LiveHazardSubmit image_quality block or
    return `missing` when the client did not send one.

    Both endpoints had the same `body.image_quality.model_dump(exclude_none=True)
    if body.image_quality is not None else <default>` ternary - this is the one
    place that owns it.
    """
    if body.image_quality is None:
        return missing
    return body.image_quality.model_dump(exclude_none=True)


def _hazard_input_from_batch(body: Any, hazard_id: str, reported_at: str) -> dict[str, Any]:
    """Build the flat hazard dict for /hazards/submit (batch flow)."""
    return {
        "hazard_id": hazard_id,
        "class": body.hazard_class,
        "confidence": float(body.confidence),
        "bboxes": body.bboxes,
        "road_type": body.road_type,
        "lane_blocked": body.lane_blocked,
        "location": body.location or "",
        "image_object_key": body.image_object_key,
        "reported_at": reported_at,
        # Optional rich fields - absent on older clients
        "top1_margin": body.top1_margin,
        "area_ratio": body.area_ratio,
        "n_detections_by_class": body.n_detections_by_class,
        "center_y_rel": body.center_y_rel,
        "image_quality": _dump_image_quality(body, missing=None),
    }


def _submit_dict_from_live(body: Any, reported_at: str) -> dict[str, Any]:
    """Build the dict passed to multiagent_decide_live from /hazards/live."""
    return {
        "class": body.hazard_class,
        "confidence": float(body.confidence),
        "top1_margin": body.top1_margin,
        "area_ratio": body.area_ratio,
        "n_detections_by_class": body.n_detections_by_class,
        "bbox": body.bbox,
        "center_y_rel": body.center_y_rel,
        "road_type": body.road_type,
        "lane_blocked": body.lane_blocked,
        "location": body.location or "",
        "session_id": body.session_id,
        "track_id": body.track_id or "",
        "image_quality": _dump_image_quality(body, missing={}),
        "reported_at": reported_at,
    }


def _live_fallback_hazard(body: Any, hazard_id: str, reported_at: str) -> dict[str, Any]:
    """Build the monolith-shaped hazard dict for /hazards/live when
    AGENT_ARCHITECTURE is not multiagent. The monolith has no idea what
    the live-only fields mean, so this is the lossy down-shape."""
    return {
        "hazard_id": hazard_id,
        "class": body.hazard_class,
        "confidence": float(body.confidence),
        "bboxes": [body.bbox] if body.bbox else [],
        "road_type": body.road_type,
        "lane_blocked": body.lane_blocked,
        "location": body.location or "",
        "image_object_key": f"live/{body.session_id}/{hazard_id}.jpg",
        "reported_at": reported_at,
    }


# ---------------------------------------------------------------------------
# Shared DB handle (FastAPI dependency)
# ---------------------------------------------------------------------------
#
# Database() opens a sqlite3 connection AND runs the schema migration from
# schema.sql on every __init__. Each `with Database(path)` under an API
# handler therefore paid that cost on every request. One Database per path
# is safe across FastAPI's threadpool because the connection is opened with
# check_same_thread=False and WAL journaling.
#
# A plain dict (not functools.lru_cache) keeps the cache clearable - tests
# routinely rebuild the app against a fresh tmp DB path and want the old
# handle released so the tmp dir can be deleted on Windows.
_DB_CACHE: dict[str, Any] = {}


def _get_shared_db(path: str):
    """Return a long-lived Database for `path`, creating it on first call."""
    cached = _DB_CACHE.get(path)
    if cached is None:
        from db.models import Database
        cached = Database(path)
        _DB_CACHE[path] = cached
    return cached


def get_db(settings: Settings = Depends(get_settings)):
    """FastAPI dependency injecting the shared Database for this path."""
    return _get_shared_db(settings.roadguard_db_path)


def _reset_db_cache() -> None:
    """Close any cached Databases and drop their handles.

    create_app() calls this so a rebuilt app (what tests do between cases)
    does not keep a connection pointing at a tmp DB that pytest is about
    to delete.
    """
    for _db in list(_DB_CACHE.values()):
        try:
            _db.close()
        except Exception:
            pass
    _DB_CACHE.clear()


def _primary_provider(settings: Settings) -> loop.LLMProvider:
    if settings.agent_provider == "groq":
        return loop.GroqProvider(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"unknown_agent_provider: {settings.agent_provider}",
    )


def _fallback_provider(settings: Settings) -> Optional[loop.LLMProvider]:
    if not _fallback_ready(settings):
        return None
    return loop.LocalLlamaProvider(
        model_path=settings.roadguard_local_llm_model,
        n_ctx=settings.roadguard_local_llm_ctx,
        n_threads=settings.roadguard_local_llm_threads,
        max_tokens=settings.roadguard_local_llm_max_tokens,
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    # The @limiter.limit decorators below each append a Limit object to the
    # shared Limiter instance's _route_limits dict. If create_app() is
    # called more than once (which tests routinely do), each build would
    # add another duplicate limit per route, which then counts each
    # request twice, three times, and so on. Clearing the dict at the
    # start of the build keeps every app instance to one limit per route.
    for attr in ("_route_limits", "_dynamic_route_limits"):
        bucket = getattr(limiter, attr, None)
        if bucket is not None and hasattr(bucket, "clear"):
            bucket.clear()

    _reset_db_cache()

    app = FastAPI(title="RoadGuard", version=VERSION)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
        agent_ok = _agent_ready(settings)
        fallback_ok = _fallback_ready(settings)
        return HealthResponse(
            status="ok" if (agent_ok or fallback_ok) else "degraded",
            version=VERSION,
            db_ready=_db_ready(settings),
            agent_ready=agent_ok,
            llm_fallback_ready=fallback_ok,
            rag_ready=_rag_ready(settings),
            supabase_ready=_supabase_ready(settings),
        )

    @app.post("/auth/login", response_model=LoginResponse)
    @limiter.limit("10/minute")
    def login(
        request: Request,
        body: LoginRequest = Body(...),
        settings: Settings = Depends(get_settings),
    ) -> LoginResponse:
        if body.google_id_token:
            payload = verify_google_id_token(
                id_token=body.google_id_token,
                client_id=settings.google_oauth_client_id,
                allowed_emails=settings.allowed_emails_set(),
            )
            token = create_jwt(
                sub=payload["sub"],
                email=payload["email"],
                display_name=payload["name"],
                settings=settings,
            )
            return LoginResponse(token=token, expires_in=settings.roadguard_jwt_ttl)
        if body.passphrase:
            if not verify_passphrase(
                body.passphrase, settings.roadguard_passphrase_hash
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="passphrase_invalid",
                )
            emails = sorted(settings.allowed_emails_set())
            email = emails[0] if emails else ""
            token = create_jwt(
                sub=f"passphrase:{email or 'owner'}",
                email=email,
                display_name="RoadGuard",
                settings=settings,
            )
            return LoginResponse(token=token, expires_in=settings.roadguard_jwt_ttl)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="passphrase_or_google_id_token_required",
        )

    @app.get("/auth/me", response_model=Me)
    def me(user: dict[str, Any] = Depends(get_current_user)) -> Me:
        return Me(
            sub=user.get("sub", ""),
            email=user.get("email", ""),
            name=user.get("name", ""),
        )

    @app.post("/hazards/presign", response_model=PresignResponse)
    @limiter.limit("30/minute")
    def hazards_presign(
        request: Request,
        body: PresignRequest = Body(...),
        user: dict[str, Any] = Depends(get_current_user),
    ) -> PresignResponse:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="supabase_not_wired_yet",
        )

    @app.post("/hazards/submit", response_model=HazardSubmitResponse)
    @limiter.limit("20/minute")
    def hazards_submit(
        request: Request,
        body: HazardSubmit = Body(...),
        user: dict[str, Any] = Depends(get_current_user),
        settings: Settings = Depends(get_settings),
    ) -> HazardSubmitResponse:
        hazard_id = f"hz_{uuid.uuid4().hex[:12]}"
        hazard_input = _hazard_input_from_batch(body, hazard_id, _now_iso())
        if settings.agent_architecture == "multiagent":
            decision = multiagent_decide(hazard_input)
        else:
            primary = _primary_provider(settings)
            fallback = _fallback_provider(settings)
            ctx = tools.ToolContext()
            decision = loop.agent_decide(
                hazard=hazard_input,
                provider=primary,
                ctx=ctx,
                fallback_provider=fallback,
            )
        return HazardSubmitResponse(
            hazard_id=hazard_id,
            verdict=AgentVerdict(
                action=decision.action,
                rationale=decision.rationale,
                priority=decision.priority,
                cited_sources=decision.cited_sources,
            ),
            provider=decision.provider,
            fallback_reason=decision.fallback_reason,
        )

    @app.post("/hazards/live", response_model=LiveHazardResponse)
    @limiter.limit("90/minute")
    def hazards_live(
        request: Request,
        body: LiveHazardSubmit = Body(...),
        user: dict[str, Any] = Depends(get_current_user),
        settings: Settings = Depends(get_settings),
        cache: DecisionCache = Depends(get_decision_cache),
    ) -> LiveHazardResponse:
        """Streaming submit from the live-drive screen.

        Returns the cached verdict when the same session resubmits the
        same canonical hazard within the TTL. On a miss, runs the
        agent (multiagent when AGENT_ARCHITECTURE=multiagent, otherwise
        monolith with the richer YOLO shape) and caches the result.
        """
        cache_key = DecisionCache.canonical_key(
            session_id=body.session_id,
            hazard_class=body.hazard_class,
            road_type=body.road_type,
            lane_blocked=body.lane_blocked,
            track_id=body.track_id,
        )

        cached = cache.get(cache_key)
        if cached is not None:
            return LiveHazardResponse(
                hazard_id=f"hz_live_{uuid.uuid4().hex[:12]}",
                verdict=AgentVerdict(
                    action=cached.action,
                    rationale=cached.rationale,
                    priority=cached.priority,
                    cited_sources=cached.cited_sources,
                ),
                provider=cached.provider,
                fallback_reason=cached.fallback_reason,
                cached=True,
                cache_key=cache_key,
            )

        hazard_id = f"hz_live_{uuid.uuid4().hex[:12]}"
        reported_at = _now_iso()

        if settings.agent_architecture == "multiagent":
            decision = multiagent_decide_live(
                hazard_id=hazard_id,
                submit=_submit_dict_from_live(body, reported_at),
            )
        else:
            # Monolith does not consume the richer live fields; adapt down.
            hazard_input = _live_fallback_hazard(body, hazard_id, reported_at)
            primary = _primary_provider(settings)
            fallback = _fallback_provider(settings)
            ctx = tools.ToolContext()
            decision = loop.agent_decide(
                hazard=hazard_input,
                provider=primary,
                ctx=ctx,
                fallback_provider=fallback,
            )

        cache.put(cache_key, decision)

        return LiveHazardResponse(
            hazard_id=hazard_id,
            verdict=AgentVerdict(
                action=decision.action,
                rationale=decision.rationale,
                priority=decision.priority,
                cited_sources=decision.cited_sources,
            ),
            provider=decision.provider,
            fallback_reason=decision.fallback_reason,
            cached=False,
            cache_key=cache_key,
        )

    @app.get("/hazards/history", response_model=HazardHistoryResponse)
    def hazards_history(
        user: dict[str, Any] = Depends(get_current_user),
    ) -> HazardHistoryResponse:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="db_not_wired_yet",
        )

    @app.get("/hazards/{hazard_id}", response_model=HazardDetail)
    def hazard_detail(
        hazard_id: str,
        user: dict[str, Any] = Depends(get_current_user),
    ) -> HazardDetail:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="db_not_wired_yet",
        )

    @app.post("/rag/query", response_model=RagQueryResponse)
    @limiter.limit("20/minute")
    def rag_query(
        request: Request,
        body: RagQueryRequest = Body(...),
        user: dict[str, Any] = Depends(get_current_user),
    ) -> RagQueryResponse:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="rag_not_wired_yet",
        )

    @app.get("/report/preview")
    def report_preview(
        user: dict[str, Any] = Depends(get_current_user),
    ) -> JSONResponse:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="report_not_wired_yet",
        )

    # ---- Approval gate (Charter §5-§6) ----------------------------------

    @app.get("/approvals/{approval_id}", response_class=HTMLResponse)
    def approval_page(approval_id: str, db=Depends(get_db)):
        """Lightweight HTML page a supervisor opens from the email."""
        try:
            row = db.get_approval(approval_id)
        except Exception:
            row = None
        if row is None:
            return HTMLResponse(
                "<h2>Approval not found</h2>", status_code=status.HTTP_404_NOT_FOUND
            )
        return HTMLResponse(_render_approval_html(row))

    @app.post("/approvals/{approval_id}/decide")
    @limiter.limit("10/minute")
    def approval_decide(
        request: Request,
        approval_id: str,
        decision: str = Form(...),
        db=Depends(get_db),
    ):
        """Approve or deny a pending approval. Called by the HTML page."""
        if decision not in ("approved", "denied"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid_decision",
            )
        # In production we would attach the authenticated supervisor
        # identity. For now the gate is anonymous but rate-limited.
        ok = db.decide_approval(approval_id, decision, decided_by="supervisor")
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="approval_not_pending_or_missing",
            )
        return RedirectResponse(
            url=f"/approvals/{approval_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # Mount the webapp at the repo root last, so API routes above take
    # precedence over the static fallback. Phones can then hit the same
    # cloudflared tunnel URL both for the API and for the webapp assets.
    _webapp_dir = Path(__file__).resolve().parents[2] / "webapp"
    if _webapp_dir.is_dir():
        app.mount(
            "/", StaticFiles(directory=str(_webapp_dir), html=True), name="webapp"
        )

    return app


app = create_app()

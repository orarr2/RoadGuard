"""FastAPI shell wiring the agent, auth, rate limits, and placeholder routes.

Routes that depend on later-sprint components (Supabase client, SQLite
tables, ChromaDB index, daily report renderer) return 501 Not Implemented
until those commits land.
"""

import logging
import os
import time
import uuid
from typing import Any, Optional

from pathlib import Path as _Path

from fastapi import Body, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
        hazard_input: dict[str, Any] = {
            "hazard_id": hazard_id,
            "class": body.hazard_class,
            "confidence": float(body.confidence),
            "bboxes": body.bboxes,
            "road_type": body.road_type,
            "lane_blocked": body.lane_blocked,
            "location": body.location or "",
            "image_object_key": body.image_object_key,
            "reported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
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
        submit_dict: dict[str, Any] = {
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
            "image_quality": (
                body.image_quality.model_dump(exclude_none=True)
                if body.image_quality is not None
                else {}
            ),
            "reported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if settings.agent_architecture == "multiagent":
            decision = multiagent_decide_live(hazard_id=hazard_id, submit=submit_dict)
        else:
            # Monolith does not consume the richer live fields; adapt down.
            hazard_input: dict[str, Any] = {
                "hazard_id": hazard_id,
                "class": body.hazard_class,
                "confidence": float(body.confidence),
                "bboxes": [body.bbox] if body.bbox else [],
                "road_type": body.road_type,
                "lane_blocked": body.lane_blocked,
                "location": body.location or "",
                "image_object_key": f"live/{body.session_id}/{hazard_id}.jpg",
                "reported_at": submit_dict["reported_at"],
            }
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

    # Mount the webapp at the repo root last, so API routes above take
    # precedence over the static fallback. Phones can then hit the same
    # cloudflared tunnel URL both for the API and for the webapp assets.
    _webapp_dir = _Path(__file__).resolve().parents[2] / "webapp"
    if _webapp_dir.is_dir():
        app.mount(
            "/", StaticFiles(directory=str(_webapp_dir), html=True), name="webapp"
        )

    return app


app = create_app()

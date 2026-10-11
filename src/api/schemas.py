"""Pydantic request and response models for the FastAPI endpoints."""

import warnings
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.warnings import UnsupportedFieldAttributeWarning

# Pydantic 2.13 warns when Field(alias=...) is attached to a Literal field,
# claiming the alias has no effect. In practice both the alias ("class")
# and the field name ("hazard_class") deserialize correctly here, so the
# warning is a false positive we silence rather than route around.
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


HazardClass = Literal["D00", "D10", "D20", "D40"]
RoadType = Literal["main", "secondary", "bike_path", "parking"]
LaneBlocked = Literal["none", "partial", "full"]
AgentAction = Literal["urgent", "standard", "needs_more_evidence", "human_review"]


class LoginRequest(BaseModel):
    passphrase: Optional[str] = None
    google_id_token: Optional[str] = None


class LoginResponse(BaseModel):
    token: str
    expires_in: int


class Me(BaseModel):
    sub: str
    email: str
    name: str


class PresignRequest(BaseModel):
    content_type: Literal["image/jpeg", "image/png", "image/webp"] = "image/jpeg"


class PresignResponse(BaseModel):
    upload_url: str
    object_key: str
    expires_in: int


class HazardSubmit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hazard_class: Literal["D00", "D10", "D20", "D40"] = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    bboxes: list[list[Any]] = Field(default_factory=list)
    road_type: RoadType
    lane_blocked: LaneBlocked
    location: Optional[str] = None
    image_object_key: str


class AgentVerdict(BaseModel):
    action: AgentAction
    rationale: str
    priority: int = Field(ge=0, le=100)
    cited_sources: list[str] = Field(default_factory=list)


class HazardSubmitResponse(BaseModel):
    hazard_id: str
    verdict: AgentVerdict
    provider: str
    fallback_reason: Optional[str] = None


class ImageQualitySignal(BaseModel):
    """Optional client-side quality probe surfaced to the Perception node."""

    blur_score: Optional[float] = None
    brightness_mean: Optional[float] = None
    is_road_scene: Optional[bool] = None
    road_scene_confidence: Optional[float] = None
    occlusion_estimate: Optional[Literal["low", "medium", "high"]] = None
    ocr_text_detected: Optional[str] = None


class LiveHazardSubmit(BaseModel):
    """Streaming submission from the live-drive screen.

    One per confirmed hazard, not per frame. The client's IoU tracker
    collapses frames into one submission; the server's decision cache
    (keyed by session_id + track_id + canonical attrs) covers the gap
    when the tracker drops and reacquires the same hazard.
    """

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(min_length=4, max_length=64)
    track_id: Optional[str] = Field(default=None, max_length=64)
    hazard_class: Literal["D00", "D10", "D20", "D40"] = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    top1_margin: Optional[float] = Field(default=None, ge=0, le=1)
    area_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    n_detections_by_class: Optional[dict[str, int]] = None
    bbox: Optional[list[float]] = None
    center_y_rel: Optional[float] = Field(default=None, ge=0, le=1)
    road_type: RoadType
    lane_blocked: LaneBlocked
    location: Optional[str] = None
    image_quality: Optional[ImageQualitySignal] = None


class LiveHazardResponse(BaseModel):
    hazard_id: str
    verdict: AgentVerdict
    provider: str
    fallback_reason: Optional[str] = None
    cached: bool = False
    cache_key: Optional[str] = None


class HazardHistoryItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hazard_id: str
    reported_at: str
    hazard_class: Literal["D00", "D10", "D20", "D40"] = Field(alias="class")
    confidence: float
    road_type: RoadType
    lane_blocked: LaneBlocked
    location: Optional[str] = None
    priority: int
    agent_decision: AgentAction
    agent_rationale: str
    image_object_key: Optional[str] = None
    image_expires_at: Optional[str] = None
    status: str


class HazardHistoryResponse(BaseModel):
    items: list[HazardHistoryItem] = Field(default_factory=list)
    next_cursor: Optional[str] = None


class HazardDetail(HazardHistoryItem):
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)


class RagQueryRequest(BaseModel):
    hazard_class: HazardClass
    road_type: RoadType
    k: int = Field(default=3, ge=1, le=10)


class RagPassage(BaseModel):
    id: str
    text: str
    issuing_body: str
    publication_date: str
    hazard_type: str
    triggering_conditions: str
    source_url: str
    is_binding_in_il: bool


class RagQueryResponse(BaseModel):
    passages: list[RagPassage] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    db_ready: bool
    agent_ready: bool
    llm_fallback_ready: bool
    rag_ready: bool
    supabase_ready: bool

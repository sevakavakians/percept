"""Data models for PERCEPT UI.

Defines dataclasses and Pydantic models for API responses,
WebSocket events, and dashboard data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NodeType(str, Enum):
    """Pipeline node types."""
    INPUT = "input"
    PROCESS = "process"
    OUTPUT = "output"
    CONDITIONAL = "conditional"


class EventType(str, Enum):
    """WebSocket event types."""
    FRAME_PROCESSED = "frame_processed"
    OBJECT_DETECTED = "object_detected"
    OBJECT_UPDATED = "object_updated"
    REVIEW_NEEDED = "review_needed"
    ALERT = "alert"
    CONFIG_CHANGED = "config_changed"
    METRICS_UPDATE = "metrics_update"


# =============================================================================
# Dashboard Models
# =============================================================================

class CameraStatus(BaseModel):
    """Status of a single camera."""
    id: str
    name: str
    connected: bool = True
    fps: float = 0.0
    resolution: Tuple[int, int] = (640, 480)
    last_frame_time: Optional[datetime] = None


class Alert(BaseModel):
    """System alert."""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    source: str = "system"
    acknowledged: bool = False


class DashboardData(BaseModel):
    """Real-time dashboard data."""

    # System metrics
    fps: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    hailo_utilization: float = 0.0
    temperature: float = 0.0

    # Pipeline status
    active_pipelines: List[str] = Field(default_factory=list)
    queue_depth: int = 0

    # Object counts
    total_objects: int = 0
    objects_in_view: int = 0
    pending_review: int = 0

    # Camera status
    cameras: List[CameraStatus] = Field(default_factory=list)

    # Alerts
    alerts: List[Alert] = Field(default_factory=list)

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now)


class MetricsData(BaseModel):
    """Detailed metrics data."""

    # Processing times (ms)
    segmentation_latency: float = 0.0
    tracking_latency: float = 0.0
    reid_latency: float = 0.0
    classification_latency: float = 0.0
    total_latency: float = 0.0

    # Throughput
    frames_processed: int = 0
    objects_processed: int = 0

    # Accuracy (if review data available)
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    # History (last N samples)
    fps_history: List[float] = Field(default_factory=list)
    latency_history: List[float] = Field(default_factory=list)

    # Uptime
    uptime_seconds: float = 0.0


# =============================================================================
# Pipeline Visualization Models
# =============================================================================

class PipelineNode(BaseModel):
    """Node in the pipeline DAG."""
    id: str
    type: NodeType
    label: str
    description: str = ""
    module_name: Optional[str] = None
    status: str = "idle"  # idle, running, error
    timing_ms: float = 0.0
    throughput: float = 0.0


class PipelineEdge(BaseModel):
    """Edge connecting pipeline nodes."""
    source: str
    target: str
    label: str = ""
    data_type: str = ""


class PipelineGraph(BaseModel):
    """Complete pipeline DAG structure."""
    name: str
    nodes: List[PipelineNode]
    edges: List[PipelineEdge]
    active: bool = True


class StageOutput(BaseModel):
    """Intermediate output from a pipeline stage."""
    stage_id: str
    frame_id: int
    timestamp: datetime
    timing_ms: float
    has_image: bool = False
    image_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    output_count: int = 0  # Number of objects/masks output


# =============================================================================
# Object Models
# =============================================================================

class ObjectSummary(BaseModel):
    """Summary of an object for list views."""
    id: str
    primary_class: str
    confidence: float
    thumbnail_url: Optional[str] = None
    first_seen: datetime
    last_seen: datetime
    camera_id: Optional[str] = None
    status: str = "unclassified"


class ObjectDetail(BaseModel):
    """Detailed object information."""
    id: str
    primary_class: str
    subclass: Optional[str] = None
    confidence: float
    status: str

    # Spatial
    position_3d: Optional[Tuple[float, float, float]] = None
    bounding_box_2d: Optional[Tuple[int, int, int, int]] = None
    dimensions_3d: Optional[Tuple[float, float, float]] = None

    # Appearance
    color: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

    # Tracking
    camera_id: Optional[str] = None
    trajectory: List[Tuple[float, float, float, datetime]] = Field(default_factory=list)

    # Timestamps
    first_seen: datetime
    last_seen: datetime

    # Images
    thumbnail_url: Optional[str] = None
    crop_url: Optional[str] = None


class TrajectoryPoint(BaseModel):
    """Point in an object's trajectory."""
    x: float
    y: float
    z: float
    timestamp: datetime
    camera_id: Optional[str] = None


# =============================================================================
# Review Models
# =============================================================================

class ReviewItem(BaseModel):
    """Item in the review queue."""
    id: str
    object_id: str
    primary_class: str
    suggested_classes: List[Tuple[str, float]] = Field(default_factory=list)
    confidence: float
    reason: str
    priority: str = "normal"
    thumbnail_url: Optional[str] = None
    crop_url: Optional[str] = None
    created_at: datetime
    camera_id: Optional[str] = None


class ReviewSubmission(BaseModel):
    """Submission for reviewing an object."""
    primary_class: str
    subclass: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class ReviewResult(BaseModel):
    """Result of a review submission."""
    success: bool
    object_id: str
    new_status: str
    message: str = ""


# =============================================================================
# Configuration Models
# =============================================================================

class ValidationError(BaseModel):
    """Configuration validation error."""
    path: str
    message: str
    severity: str = "error"


class ValidationResult(BaseModel):
    """Result of configuration validation."""
    valid: bool
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationError] = Field(default_factory=list)


class ConfigUpdate(BaseModel):
    """Configuration update request."""
    config: Dict[str, Any]
    validate_only: bool = False


# =============================================================================
# WebSocket Event Models
# =============================================================================

class WebSocketEvent(BaseModel):
    """WebSocket event message."""
    event: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)


class FrameProcessedEvent(BaseModel):
    """Event data for frame processed."""
    frame_id: int
    camera_id: str
    objects_detected: int
    processing_time_ms: float


class ObjectDetectedEvent(BaseModel):
    """Event data for object detected."""
    object_id: str
    primary_class: str
    confidence: float
    position: Optional[Tuple[float, float, float]] = None
    camera_id: str


class ObjectUpdatedEvent(BaseModel):
    """Event data for object updated (re-identified)."""
    object_id: str
    camera_id: str
    was_matched: bool
    confidence: float


class ReviewNeededEvent(BaseModel):
    """Event data for review needed."""
    object_id: str
    reason: str
    confidence: float
    priority: str


# =============================================================================
# API Response Models
# =============================================================================

class PaginatedResponse(BaseModel):
    """Paginated API response."""
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = True
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

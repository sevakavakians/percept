"""REST API routes for PERCEPT UI.

Provides endpoints for dashboard data, camera feeds, pipeline info,
object queries, configuration management, and review queue.
"""

import io
import os
import psutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from ui.models import (
    Alert,
    AlertLevel,
    APIResponse,
    CameraStatus,
    ConfigUpdate,
    DashboardData,
    MetricsData,
    NodeType,
    ObjectDetail,
    ObjectSummary,
    PaginatedResponse,
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
    ReviewItem,
    ReviewResult,
    ReviewSubmission,
    StageOutput,
    TrajectoryPoint,
    ValidationError,
    ValidationResult,
)

router = APIRouter()


# =============================================================================
# Dependencies
# =============================================================================

def get_database():
    """Get database instance."""
    from ui.app import app_state
    return app_state.database


def get_config():
    """Get configuration instance."""
    from ui.app import app_state
    return app_state.config


def get_app_state():
    """Get application state."""
    from ui.app import app_state
    return app_state


# =============================================================================
# Dashboard Endpoints
# =============================================================================

@router.get("/dashboard", response_model=DashboardData)
async def get_dashboard_data(state=Depends(get_app_state)):
    """Get real-time dashboard data."""
    # System metrics
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()

    # Get temperature if available (Linux)
    temperature = 0.0
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if entries:
                    temperature = entries[0].current
                    break
    except Exception:
        pass

    # Object counts
    total_objects = 0
    pending_review = 0
    if state.database:
        try:
            total_objects = state.database.count_objects()
            pending_review = state.database.count_pending_review()
        except Exception:
            pass

    # Camera status (mock for now)
    cameras = []
    if state.config and state.config.cameras:
        for cam_cfg in state.config.cameras:
            cameras.append(CameraStatus(
                id=cam_cfg.id,
                name=cam_cfg.id,
                connected=cam_cfg.enabled,
                fps=float(cam_cfg.fps),
                resolution=cam_cfg.resolution,
                last_frame_time=datetime.now(),
            ))

    return DashboardData(
        fps=state.frame_count / max(state.uptime_seconds, 1),
        cpu_usage=cpu_usage,
        memory_usage=memory.percent,
        memory_used_mb=memory.used / 1024 / 1024,
        memory_total_mb=memory.total / 1024 / 1024,
        hailo_utilization=0.0,  # Would need Hailo SDK
        temperature=temperature,
        active_pipelines=["main"],
        queue_depth=0,
        total_objects=total_objects,
        objects_in_view=0,
        pending_review=pending_review,
        cameras=cameras,
        alerts=[],
        timestamp=datetime.now(),
    )


@router.get("/metrics", response_model=MetricsData)
async def get_metrics(state=Depends(get_app_state)):
    """Get detailed performance metrics."""
    return MetricsData(
        segmentation_latency=30.0,
        tracking_latency=3.0,
        reid_latency=5.0,
        classification_latency=20.0,
        total_latency=58.0,
        frames_processed=state.frame_count,
        objects_processed=0,
        fps_history=[],
        latency_history=[],
        uptime_seconds=state.uptime_seconds,
    )


# =============================================================================
# Camera Endpoints
# =============================================================================

@router.get("/cameras", response_model=List[CameraStatus])
async def list_cameras(config=Depends(get_config)):
    """List all configured cameras."""
    cameras = []
    if config and config.cameras:
        for cam_cfg in config.cameras:
            cameras.append(CameraStatus(
                id=cam_cfg.device_id,
                name=cam_cfg.name or cam_cfg.device_id,
                connected=True,
                fps=30.0,
                resolution=(cam_cfg.width, cam_cfg.height),
            ))
    return cameras


@router.get("/cameras/{camera_id}/frame")
async def get_camera_frame(camera_id: str):
    """Get current frame from camera as JPEG."""
    # Generate placeholder image
    try:
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            img, f"Camera: {camera_id}",
            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            img, datetime.now().strftime("%H:%M:%S"),
            (250, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1
        )
        _, jpeg = cv2.imencode('.jpg', img)
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    except ImportError:
        raise HTTPException(status_code=503, detail="OpenCV not available")


@router.get("/cameras/{camera_id}/depth")
async def get_camera_depth(camera_id: str):
    """Get depth visualization from camera."""
    try:
        import cv2
        # Generate colorized depth placeholder
        depth = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)
        depth_norm = ((depth - 0.5) / 4.5 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        _, jpeg = cv2.imencode('.jpg', depth_color)
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    except ImportError:
        raise HTTPException(status_code=503, detail="OpenCV not available")


# =============================================================================
# Pipeline Endpoints
# =============================================================================

@router.get("/pipeline/graph", response_model=PipelineGraph)
async def get_pipeline_graph():
    """Get pipeline DAG structure for visualization."""
    return PipelineGraph(
        name="main",
        nodes=[
            PipelineNode(
                id="capture",
                type=NodeType.INPUT,
                label="Camera Capture",
                description="RGB-D frame acquisition",
                module_name="RealSenseCamera",
                status="running",
            ),
            PipelineNode(
                id="segment",
                type=NodeType.PROCESS,
                label="Segmentation",
                description="FastSAM + depth fusion",
                module_name="SegmentationFusion",
                status="running",
                timing_ms=30.0,
            ),
            PipelineNode(
                id="tracking",
                type=NodeType.PROCESS,
                label="Tracking",
                description="ByteTrack object tracking",
                module_name="ByteTrackWrapper",
                status="running",
                timing_ms=3.0,
            ),
            PipelineNode(
                id="reid",
                type=NodeType.PROCESS,
                label="ReID",
                description="Re-identification matching",
                module_name="ReIDMatcher",
                status="running",
                timing_ms=5.0,
            ),
            PipelineNode(
                id="router",
                type=NodeType.CONDITIONAL,
                label="Router",
                description="Classification routing",
                module_name="PipelineRouter",
                status="running",
                timing_ms=0.1,
            ),
            PipelineNode(
                id="person",
                type=NodeType.PROCESS,
                label="Person Pipeline",
                description="Pose, clothing, face",
                module_name="PersonPipeline",
                status="idle",
                timing_ms=25.0,
            ),
            PipelineNode(
                id="vehicle",
                type=NodeType.PROCESS,
                label="Vehicle Pipeline",
                description="Color, type, plate",
                module_name="VehiclePipeline",
                status="idle",
                timing_ms=15.0,
            ),
            PipelineNode(
                id="generic",
                type=NodeType.PROCESS,
                label="Generic Pipeline",
                description="ImageNet, color, shape",
                module_name="GenericPipeline",
                status="idle",
                timing_ms=20.0,
            ),
            PipelineNode(
                id="persist",
                type=NodeType.OUTPUT,
                label="Database",
                description="SQLite + embedding store",
                module_name="PerceptDatabase",
                status="running",
                timing_ms=5.0,
            ),
        ],
        edges=[
            PipelineEdge(source="capture", target="segment", data_type="FrameData"),
            PipelineEdge(source="segment", target="tracking", data_type="ObjectMask[]"),
            PipelineEdge(source="tracking", target="reid", data_type="Track[]"),
            PipelineEdge(source="reid", target="router", data_type="ObjectSchema[]"),
            PipelineEdge(source="router", target="person", label="person"),
            PipelineEdge(source="router", target="vehicle", label="vehicle"),
            PipelineEdge(source="router", target="generic", label="other"),
            PipelineEdge(source="person", target="persist"),
            PipelineEdge(source="vehicle", target="persist"),
            PipelineEdge(source="generic", target="persist"),
        ],
        active=True,
    )


@router.get("/pipeline/stages", response_model=List[str])
async def list_pipeline_stages():
    """List all pipeline stages."""
    return [
        "capture", "segment", "tracking", "reid",
        "router", "person", "vehicle", "generic", "persist"
    ]


@router.get("/pipeline/stages/{stage_id}/output", response_model=StageOutput)
async def get_stage_output(stage_id: str, frame_id: int = 0):
    """Get intermediate output from a pipeline stage."""
    return StageOutput(
        stage_id=stage_id,
        frame_id=frame_id,
        timestamp=datetime.now(),
        timing_ms=10.0,
        has_image=stage_id in ["capture", "segment"],
        image_url=f"/api/pipeline/stages/{stage_id}/image?frame_id={frame_id}",
        metadata={"stage": stage_id, "frame": frame_id},
        output_count=5,
    )


# =============================================================================
# Object Endpoints
# =============================================================================

@router.get("/objects", response_model=PaginatedResponse)
async def list_objects(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    class_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    db=Depends(get_database),
):
    """List objects with pagination and filtering."""
    items = []
    total = 0

    if db:
        try:
            offset = (page - 1) * page_size

            if class_filter:
                objects = db.query_by_class(class_filter, limit=page_size, offset=offset)
            else:
                objects = db.get_recent_objects(limit=page_size, offset=offset)

            total = db.count_objects()

            for obj in objects:
                items.append(ObjectSummary(
                    id=obj.id,
                    primary_class=obj.primary_class,
                    confidence=obj.confidence,
                    first_seen=obj.first_seen,
                    last_seen=obj.last_seen,
                    camera_id=obj.camera_id,
                    status=obj.classification_status.value if obj.classification_status else "unclassified",
                ))
        except Exception as e:
            print(f"Error listing objects: {e}")

    pages = (total + page_size - 1) // page_size if total > 0 else 1

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.get("/objects/{object_id}", response_model=ObjectDetail)
async def get_object(object_id: str, db=Depends(get_database)):
    """Get detailed object information."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    obj = db.get_object(object_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found")

    return ObjectDetail(
        id=obj.id,
        primary_class=obj.primary_class,
        subclass=obj.subclass,
        confidence=obj.confidence,
        status=obj.classification_status.value if obj.classification_status else "unclassified",
        position_3d=obj.position_3d,
        bounding_box_2d=obj.bounding_box_2d,
        dimensions_3d=obj.dimensions_3d,
        color=obj.color,
        attributes=obj.attributes or {},
        camera_id=obj.camera_id,
        trajectory=[
            (p[0], p[1], p[2], p[3]) for p in (obj.trajectory or [])
        ],
        first_seen=obj.first_seen,
        last_seen=obj.last_seen,
    )


@router.get("/objects/{object_id}/trajectory", response_model=List[TrajectoryPoint])
async def get_object_trajectory(object_id: str, db=Depends(get_database)):
    """Get object position history."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    obj = db.get_object(object_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found")

    return [
        TrajectoryPoint(x=p[0], y=p[1], z=p[2], timestamp=p[3])
        for p in (obj.trajectory or [])
    ]


# =============================================================================
# Configuration Endpoints
# =============================================================================

@router.get("/config")
async def get_config_data(config=Depends(get_config)):
    """Get current configuration."""
    if not config:
        return {}

    # Convert config to dict
    return {
        "cameras": [
            {
                "id": c.id,
                "type": c.type,
                "serial": c.serial,
                "resolution": c.resolution,
                "fps": c.fps,
                "enabled": c.enabled,
            }
            for c in (config.cameras or [])
        ],
        "segmentation": {
            "primary_method": config.segmentation.primary_method,
            "fusion_enabled": config.segmentation.fusion_enabled,
            "min_object_pixels": config.segmentation.min_object_pixels,
        },
        "tracking": {
            "algorithm": config.tracking.algorithm,
            "min_track_confidence": config.tracking.min_track_confidence,
            "lost_track_buffer_frames": config.tracking.lost_track_buffer_frames,
        },
        "database": {
            "path": config.database.path,
            "embedding_index_type": config.database.embedding_index_type,
        },
    }


@router.put("/config", response_model=APIResponse)
async def update_config(update: ConfigUpdate, state=Depends(get_app_state)):
    """Update configuration."""
    # Validate first
    validation = await validate_config(update)
    if not validation.valid:
        return APIResponse(
            success=False,
            error="Configuration validation failed",
            data=validation,
        )

    if update.validate_only:
        return APIResponse(success=True, data=validation)

    # Apply configuration
    try:
        config_path = os.environ.get("PERCEPT_CONFIG", "config/percept_config.yaml")
        # Save to file
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(update.config, f)

        # Reload
        from percept.core.config import PerceptConfig
        state.config = PerceptConfig.from_yaml(config_path)

        return APIResponse(success=True, data={"reloaded": True})
    except Exception as e:
        return APIResponse(success=False, error=str(e))


@router.post("/config/validate", response_model=ValidationResult)
async def validate_config(update: ConfigUpdate):
    """Validate configuration without applying."""
    errors = []
    warnings = []
    config = update.config

    # Check required sections
    if "cameras" not in config:
        warnings.append(ValidationError(
            path="cameras",
            message="No cameras configured",
            severity="warning"
        ))

    # Check threshold ranges
    tracking = config.get("tracking", {})
    if tracking.get("match_threshold", 0.5) > 1.0:
        errors.append(ValidationError(
            path="tracking.match_threshold",
            message="Match threshold must be <= 1.0",
            severity="error"
        ))

    # Check paths exist
    seg = config.get("segmentation", {})
    model_path = seg.get("fastsam_model_path", "")
    if model_path and not Path(model_path).exists():
        warnings.append(ValidationError(
            path="segmentation.fastsam_model_path",
            message=f"Model file not found: {model_path}",
            severity="warning"
        ))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# Review Queue Endpoints
# =============================================================================

@router.get("/review", response_model=List[ReviewItem])
async def get_review_queue(
    limit: int = Query(20, ge=1, le=100),
    db=Depends(get_database),
):
    """Get pending review items."""
    items = []

    if db:
        try:
            pending = db.get_pending_review(limit=limit)
            for obj in pending:
                items.append(ReviewItem(
                    id=f"review-{obj.id}",
                    object_id=obj.id,
                    primary_class=obj.primary_class,
                    suggested_classes=[
                        (obj.primary_class, obj.confidence),
                    ],
                    confidence=obj.confidence,
                    reason="low_confidence" if obj.confidence < 0.5 else "ambiguous",
                    priority="normal",
                    created_at=obj.first_seen,
                    camera_id=obj.camera_id,
                ))
        except Exception as e:
            print(f"Error getting review queue: {e}")

    return items


@router.post("/review/{object_id}", response_model=ReviewResult)
async def submit_review(
    object_id: str,
    review: ReviewSubmission,
    db=Depends(get_database),
):
    """Submit human review for an object."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    obj = db.get_object(object_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found")

    try:
        # Update object
        obj.primary_class = review.primary_class
        obj.subclass = review.subclass
        if review.attributes:
            obj.attributes = {**(obj.attributes or {}), **review.attributes}

        from percept.core.schema import ClassificationStatus
        obj.classification_status = ClassificationStatus.CONFIRMED

        db.save_object(obj)

        return ReviewResult(
            success=True,
            object_id=object_id,
            new_status="confirmed",
            message="Review submitted successfully",
        )
    except Exception as e:
        return ReviewResult(
            success=False,
            object_id=object_id,
            new_status="error",
            message=str(e),
        )


@router.post("/review/{object_id}/skip", response_model=ReviewResult)
async def skip_review(object_id: str, db=Depends(get_database)):
    """Skip a review item."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    obj = db.get_object(object_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found")

    # Mark as skipped (keep provisional status)
    return ReviewResult(
        success=True,
        object_id=object_id,
        new_status="skipped",
        message="Review skipped",
    )

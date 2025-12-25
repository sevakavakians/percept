"""ObjectSchema and related data structures for PERCEPT.

The ObjectSchema is the central data structure that accumulates knowledge about
detected objects throughout their lifecycle in the system.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ClassificationStatus(Enum):
    """Status of object classification confidence."""

    CONFIRMED = "confirmed"      # High confidence, auto-confirmed
    PROVISIONAL = "provisional"  # Medium confidence, may need review
    NEEDS_REVIEW = "needs_review"  # Low confidence, queued for human review
    UNCLASSIFIED = "unclassified"  # Not yet processed


@dataclass
class ObjectSchema:
    """Central data structure for detected objects.

    Accumulates knowledge about objects as they flow through processing pipelines.
    Designed to be SLAM-ready with camera-relative spatial coordinates.

    Attributes:
        id: Unique identifier (UUID)
        reid_embedding: 512-dim L2-normalized feature vector for re-identification
        position_3d: Camera-relative (x, y, z) position in meters
        bounding_box_2d: Pixel coordinates (x1, y1, x2, y2)
        dimensions: Physical (width, height, depth) in meters
        distance_from_camera: Distance in meters
        primary_class: Main classification (person, vehicle, furniture, etc.)
        subclass: More specific classification (adult, car, chair)
        confidence: Classification confidence 0.0-1.0
        classification_status: Processing status for human review
        attributes: Flexible dict for type-specific attributes
        first_seen: First observation timestamp
        last_seen: Most recent observation timestamp
        camera_id: Which camera observed this object
        trajectory: Position history with timestamps
        pipelines_completed: Which processing pipelines have run
        processing_time_ms: Total processing time
        source_frame_ids: Frame IDs that contributed to this schema
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reid_embedding: Optional[np.ndarray] = None

    # Spatial (camera-relative, SLAM-ready)
    position_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_box_2d: Tuple[int, int, int, int] = (0, 0, 0, 0)
    dimensions: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    distance_from_camera: float = 0.0

    # Classification
    primary_class: str = "unknown"
    subclass: Optional[str] = None
    confidence: float = 0.0
    classification_status: ClassificationStatus = ClassificationStatus.UNCLASSIFIED

    # Type-specific attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    camera_id: str = ""
    trajectory: List[Tuple[float, float, float, datetime]] = field(default_factory=list)

    # Processing metadata
    pipelines_completed: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    source_frame_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize embedding after initialization."""
        if self.reid_embedding is not None:
            self.reid_embedding = self._normalize_embedding(self.reid_embedding)

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def set_embedding(self, embedding: np.ndarray) -> None:
        """Set and normalize the ReID embedding."""
        self.reid_embedding = self._normalize_embedding(embedding)

    def update_position(
        self,
        position_3d: Tuple[float, float, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update position and add to trajectory."""
        self.position_3d = position_3d
        self.last_seen = timestamp or datetime.now()
        self.trajectory.append((*position_3d, self.last_seen))

    def mark_pipeline_complete(self, pipeline_name: str, time_ms: float) -> None:
        """Record that a pipeline has completed processing."""
        if pipeline_name not in self.pipelines_completed:
            self.pipelines_completed.append(pipeline_name)
        self.processing_time_ms += time_ms

    def set_classification(
        self,
        primary_class: str,
        confidence: float,
        subclass: Optional[str] = None,
        auto_set_status: bool = True
    ) -> None:
        """Set classification with automatic status determination.

        Args:
            primary_class: Main object class
            confidence: Confidence score 0.0-1.0
            subclass: Optional more specific class
            auto_set_status: If True, automatically set status based on confidence
        """
        self.primary_class = primary_class
        self.confidence = confidence
        self.subclass = subclass

        if auto_set_status:
            if confidence >= 0.85:
                self.classification_status = ClassificationStatus.CONFIRMED
            elif confidence >= 0.5:
                self.classification_status = ClassificationStatus.PROVISIONAL
            else:
                self.classification_status = ClassificationStatus.NEEDS_REVIEW

    @property
    def needs_review(self) -> bool:
        """Check if object needs human review."""
        return self.classification_status == ClassificationStatus.NEEDS_REVIEW

    @property
    def is_confirmed(self) -> bool:
        """Check if classification is confirmed."""
        return self.classification_status == ClassificationStatus.CONFIRMED

    @property
    def age_seconds(self) -> float:
        """Time since first seen in seconds."""
        return (datetime.now() - self.first_seen).total_seconds()

    @property
    def time_since_last_seen(self) -> float:
        """Time since last observation in seconds."""
        return (datetime.now() - self.last_seen).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Handles numpy arrays and datetime objects.
        """
        data = asdict(self)

        # Convert numpy array to list
        if data["reid_embedding"] is not None:
            data["reid_embedding"] = data["reid_embedding"].tolist()

        # Convert datetime objects
        data["first_seen"] = data["first_seen"].isoformat()
        data["last_seen"] = data["last_seen"].isoformat()

        # Convert trajectory timestamps
        data["trajectory"] = [
            (x, y, z, t.isoformat()) for x, y, z, t in data["trajectory"]
        ]

        # Convert enum to string
        data["classification_status"] = data["classification_status"].value

        return data

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObjectSchema:
        """Create ObjectSchema from dictionary."""
        # Convert embedding back to numpy
        if data.get("reid_embedding") is not None:
            data["reid_embedding"] = np.array(data["reid_embedding"], dtype=np.float32)

        # Convert datetime strings
        data["first_seen"] = datetime.fromisoformat(data["first_seen"])
        data["last_seen"] = datetime.fromisoformat(data["last_seen"])

        # Convert trajectory timestamps
        data["trajectory"] = [
            (x, y, z, datetime.fromisoformat(t))
            for x, y, z, t in data["trajectory"]
        ]

        # Convert status enum
        data["classification_status"] = ClassificationStatus(data["classification_status"])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> ObjectSchema:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"ObjectSchema(id={self.id[:8]}..., "
            f"class={self.primary_class}, "
            f"conf={self.confidence:.2f}, "
            f"status={self.classification_status.value})"
        )


@dataclass
class Detection:
    """Raw detection from an object detector.

    Intermediate representation before full ObjectSchema creation.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: Optional[np.ndarray] = None

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class ObjectMask:
    """Segmentation mask for an object.

    Used as intermediate representation between segmentation and classification.
    """

    mask: np.ndarray  # Binary mask (H, W)
    bbox: Tuple[int, int, int, int]  # Bounding box of mask
    confidence: float = 1.0
    depth_median: Optional[float] = None  # Median depth value in meters
    point_count: int = 0  # Number of 3D points if available

    @property
    def area(self) -> int:
        """Number of pixels in mask."""
        return int(np.sum(self.mask > 0))

    def extract_crop(self, image: np.ndarray) -> np.ndarray:
        """Extract masked region from image."""
        x1, y1, x2, y2 = self.bbox
        crop = image[y1:y2, x1:x2].copy()
        mask_crop = self.mask[y1:y2, x1:x2]

        # Apply mask (set background to black)
        if len(crop.shape) == 3:
            crop[mask_crop == 0] = 0
        else:
            crop[mask_crop == 0] = 0

        return crop

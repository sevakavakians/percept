"""Pipeline router for PERCEPT.

Routes detected objects to appropriate classification pipelines
based on their class and characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.core.schema import ObjectSchema, Detection

if TYPE_CHECKING:
    from percept.pipelines.person import PersonPipeline
    from percept.pipelines.vehicle import VehiclePipeline
    from percept.pipelines.generic import GenericPipeline


class PipelineType(Enum):
    """Types of classification pipelines."""
    PERSON = "person"
    VEHICLE = "vehicle"
    GENERIC = "generic"
    FACE = "face"  # Sub-pipeline triggered by person


# Class name mappings to pipeline types
PERSON_CLASSES = {"person", "pedestrian", "human", "man", "woman", "child"}
VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle", "vehicle",
    "sedan", "suv", "van", "pickup", "motorbike", "bike",
}
ANIMAL_CLASSES = {"dog", "cat", "bird", "horse", "cow", "sheep", "elephant"}


@dataclass
class RouterConfig:
    """Configuration for pipeline router."""

    # Routing behavior
    default_pipeline: str = "generic"
    enable_person_pipeline: bool = True
    enable_vehicle_pipeline: bool = True

    # Confidence thresholds
    min_routing_confidence: float = 0.3

    # Size-based routing (area in pixels)
    min_person_area: int = 1000  # Minimum area for person pipeline
    min_vehicle_area: int = 2000  # Minimum area for vehicle pipeline

    # Custom class mappings
    custom_person_classes: Set[str] = field(default_factory=set)
    custom_vehicle_classes: Set[str] = field(default_factory=set)


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    pipeline_type: PipelineType
    confidence: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineRouter:
    """Routes objects to appropriate classification pipelines.

    Determines which pipeline to use based on:
    - Initial detection class
    - Object size and aspect ratio
    - Scene context (optional)
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        """Initialize router.

        Args:
            config: Configuration options
        """
        self.config = config or RouterConfig()
        self._pipelines: Dict[PipelineType, Any] = {}
        self._routing_stats: Dict[str, int] = {}

        # Build class sets
        self._person_classes = PERSON_CLASSES | self.config.custom_person_classes
        self._vehicle_classes = VEHICLE_CLASSES | self.config.custom_vehicle_classes

    def register_pipeline(self, pipeline_type: PipelineType, pipeline: Any) -> None:
        """Register a pipeline for a type.

        Args:
            pipeline_type: Type of pipeline
            pipeline: Pipeline instance
        """
        self._pipelines[pipeline_type] = pipeline

    def get_pipeline(self, pipeline_type: PipelineType) -> Optional[Any]:
        """Get registered pipeline.

        Args:
            pipeline_type: Type to get

        Returns:
            Pipeline instance or None
        """
        return self._pipelines.get(pipeline_type)

    def route(
        self,
        detection: Detection,
        mask: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Determine which pipeline to use for a detection.

        Args:
            detection: Detection to route
            mask: Optional segmentation mask
            context: Optional scene context

        Returns:
            RoutingDecision with pipeline type and confidence
        """
        class_name = (detection.class_name or "").lower().strip()
        confidence = detection.confidence

        # Check confidence threshold
        if confidence < self.config.min_routing_confidence:
            return RoutingDecision(
                pipeline_type=PipelineType.GENERIC,
                confidence=confidence,
                reason="confidence_below_threshold",
            )

        # Check for person classes
        if self.config.enable_person_pipeline and class_name in self._person_classes:
            # Verify minimum area
            area = self._compute_area(detection.bbox, mask)
            if area >= self.config.min_person_area:
                self._update_stats("person")
                return RoutingDecision(
                    pipeline_type=PipelineType.PERSON,
                    confidence=confidence,
                    reason="class_match",
                    metadata={"matched_class": class_name, "area": area},
                )

        # Check for vehicle classes
        if self.config.enable_vehicle_pipeline and class_name in self._vehicle_classes:
            area = self._compute_area(detection.bbox, mask)
            if area >= self.config.min_vehicle_area:
                self._update_stats("vehicle")
                return RoutingDecision(
                    pipeline_type=PipelineType.VEHICLE,
                    confidence=confidence,
                    reason="class_match",
                    metadata={"matched_class": class_name, "area": area},
                )

        # Default to generic pipeline
        self._update_stats("generic")
        return RoutingDecision(
            pipeline_type=PipelineType.GENERIC,
            confidence=confidence,
            reason="default_fallback",
            metadata={"original_class": class_name},
        )

    def route_batch(
        self,
        detections: List[Detection],
        masks: Optional[List[np.ndarray]] = None,
    ) -> List[RoutingDecision]:
        """Route multiple detections.

        Args:
            detections: List of detections
            masks: Optional list of masks

        Returns:
            List of routing decisions
        """
        decisions = []
        masks = masks or [None] * len(detections)

        for det, mask in zip(detections, masks):
            decision = self.route(det, mask)
            decisions.append(decision)

        return decisions

    def process_object(
        self,
        obj: ObjectSchema,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> ObjectSchema:
        """Route and process an object through appropriate pipeline.

        Args:
            obj: Object to process
            image: Source image
            mask: Optional mask

        Returns:
            Processed ObjectSchema with attributes populated
        """
        # Create detection from object
        detection = Detection(
            class_id=0,
            class_name=obj.primary_class,
            confidence=obj.confidence,
            bbox=obj.bounding_box_2d or (0, 0, 0, 0),
        )

        # Route to appropriate pipeline
        decision = self.route(detection, mask)

        # Get pipeline
        pipeline = self._pipelines.get(decision.pipeline_type)
        if pipeline is None:
            # No pipeline registered, return as-is
            obj.attributes["_routing"] = {
                "pipeline": decision.pipeline_type.value,
                "status": "no_pipeline_registered",
            }
            return obj

        # Extract crop
        bbox = obj.bounding_box_2d
        if bbox and bbox != (0, 0, 0, 0):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            crop = image[y1:y2, x1:x2]

            # Extract mask crop if available
            mask_crop = None
            if mask is not None:
                mask_crop = mask[y1:y2, x1:x2]
        else:
            crop = image
            mask_crop = mask

        # Process through pipeline
        if hasattr(pipeline, 'process_object'):
            obj = pipeline.process_object(obj, crop, mask_crop)
        elif hasattr(pipeline, 'process'):
            obj = pipeline.process(obj, crop)

        # Add routing metadata
        obj.attributes["_routing"] = {
            "pipeline": decision.pipeline_type.value,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }

        return obj

    def _compute_area(
        self,
        bbox: Tuple[float, float, float, float],
        mask: Optional[np.ndarray] = None,
    ) -> int:
        """Compute area of detection.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            mask: Optional mask for precise area

        Returns:
            Area in pixels
        """
        if mask is not None:
            return int((mask > 0).sum())

        x1, y1, x2, y2 = bbox
        return int((x2 - x1) * (y2 - y1))

    def _update_stats(self, pipeline_name: str) -> None:
        """Update routing statistics."""
        self._routing_stats[pipeline_name] = self._routing_stats.get(pipeline_name, 0) + 1

    def get_stats(self) -> Dict[str, int]:
        """Get routing statistics.

        Returns:
            Dictionary of pipeline -> count
        """
        return dict(self._routing_stats)

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._routing_stats.clear()


class RouterModule(PipelineModule):
    """Pipeline module for object routing.

    Routes objects to specialized pipelines and aggregates results.
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        """Initialize module.

        Args:
            config: Configuration options
        """
        self.config = config or RouterConfig()
        self.router = PipelineRouter(self.config)

    @property
    def name(self) -> str:
        return "router"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="detections",
            required_fields=["detections", "image"],
            optional_fields=["masks"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="routed_objects",
            required_fields=["objects", "routing_decisions"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Route detections to pipelines.

        Args:
            data: PipelineData with detections

        Returns:
            PipelineData with routing decisions
        """
        detections = data.get("detections", [])
        image = data.get("image")
        masks = data.get("masks", [])

        # Ensure masks list matches detections
        if len(masks) < len(detections):
            masks = masks + [None] * (len(detections) - len(masks))

        # Route all detections
        decisions = []
        objects = []

        for i, det in enumerate(detections):
            mask = masks[i] if i < len(masks) else None
            decision = self.router.route(det, mask)
            decisions.append(decision)

            # Create ObjectSchema from detection
            obj = ObjectSchema(
                primary_class=det.class_name or "unknown",
                confidence=det.confidence,
                bounding_box_2d=det.bbox,
            )
            objects.append(obj)

        result = data.copy()
        result.objects = objects
        result.routing_decisions = decisions
        return result


def get_pipeline_for_class(class_name: str) -> PipelineType:
    """Convenience function to get pipeline type for a class name.

    Args:
        class_name: Object class name

    Returns:
        Appropriate PipelineType
    """
    class_lower = class_name.lower().strip()

    if class_lower in PERSON_CLASSES:
        return PipelineType.PERSON
    elif class_lower in VEHICLE_CLASSES:
        return PipelineType.VEHICLE
    else:
        return PipelineType.GENERIC

"""Vehicle classification pipeline for PERCEPT.

Extracts detailed attributes from detected vehicles:
- Vehicle type classification
- Color analysis
- Shape features
- Optional license plate detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.core.schema import ObjectSchema


class VehicleType(Enum):
    """Vehicle type classification."""
    SEDAN = "sedan"
    SUV = "suv"
    TRUCK = "truck"
    VAN = "van"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    BUS = "bus"
    PICKUP = "pickup"
    UNKNOWN = "unknown"


class VehicleOrientation(Enum):
    """Vehicle orientation relative to camera."""
    FRONT = "front"
    REAR = "rear"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    ANGLE_FRONT_LEFT = "angle_front_left"
    ANGLE_FRONT_RIGHT = "angle_front_right"
    ANGLE_REAR_LEFT = "angle_rear_left"
    ANGLE_REAR_RIGHT = "angle_rear_right"
    UNKNOWN = "unknown"


@dataclass
class VehiclePipelineConfig:
    """Configuration for vehicle pipeline."""

    # Color analysis
    color_bins: int = 8
    saturation_threshold: float = 30.0  # Below this = grayscale

    # Type classification
    min_aspect_ratio: float = 0.3  # Width/Height minimum
    max_aspect_ratio: float = 4.0  # Width/Height maximum

    # License plate detection
    enable_license_plate: bool = True
    plate_min_area: int = 500  # Minimum plate area in pixels
    plate_aspect_min: float = 2.0  # Min width/height for plate
    plate_aspect_max: float = 6.0  # Max width/height for plate

    # Shape analysis
    contour_epsilon: float = 0.02  # Contour approximation epsilon


# Color ranges in HSV
COLOR_RANGES = {
    "red": ((0, 100, 100), (10, 255, 255)),
    "red2": ((170, 100, 100), (180, 255, 255)),  # Red wraps around
    "orange": ((10, 100, 100), (25, 255, 255)),
    "yellow": ((25, 100, 100), (35, 255, 255)),
    "green": ((35, 100, 100), (85, 255, 255)),
    "blue": ((85, 100, 100), (130, 255, 255)),
    "purple": ((130, 100, 100), (160, 255, 255)),
    "pink": ((160, 100, 100), (170, 255, 255)),
}


@dataclass
class VehicleColorResult:
    """Result of vehicle color analysis."""
    primary_color: str
    secondary_color: Optional[str]
    color_confidence: float
    is_metallic: bool
    hsv_values: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary": self.primary_color,
            "secondary": self.secondary_color,
            "confidence": self.color_confidence,
            "metallic": self.is_metallic,
        }


@dataclass
class VehicleTypeResult:
    """Result of vehicle type classification."""
    vehicle_type: VehicleType
    confidence: float
    orientation: VehicleOrientation
    aspect_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.vehicle_type.value,
            "confidence": self.confidence,
            "orientation": self.orientation.value,
            "aspect_ratio": self.aspect_ratio,
        }


@dataclass
class LicensePlateResult:
    """Result of license plate detection."""
    detected: bool
    bbox: Optional[Tuple[int, int, int, int]] = None
    text: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detected": self.detected,
            "bbox": self.bbox,
            "text": self.text,
            "confidence": self.confidence,
        }


class VehicleColorAnalyzer:
    """Analyze vehicle color from image.

    Uses color histograms with special handling for
    metallic and two-tone paint.
    """

    def __init__(self, config: Optional[VehiclePipelineConfig] = None):
        """Initialize color analyzer.

        Args:
            config: Configuration options
        """
        self.config = config or VehiclePipelineConfig()

    def analyze(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> VehicleColorResult:
        """Analyze vehicle color.

        Args:
            image: BGR image of vehicle
            mask: Optional mask for vehicle region

        Returns:
            VehicleColorResult with color info
        """
        if image.size == 0:
            return VehicleColorResult(
                primary_color="unknown",
                secondary_color=None,
                color_confidence=0.0,
                is_metallic=False,
                hsv_values=(0, 0, 0),
            )

        # Apply mask if provided
        if mask is not None and mask.size > 0:
            # Ensure mask is same size as image
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Sample body region (center portion, avoiding windows)
        h, w = image.shape[:2]
        body_region = hsv[int(h * 0.3):int(h * 0.8), int(w * 0.1):int(w * 0.9)]

        if body_region.size == 0:
            body_region = hsv

        # Calculate color statistics
        avg_hue = np.mean(body_region[:, :, 0])
        avg_sat = np.mean(body_region[:, :, 1])
        avg_val = np.mean(body_region[:, :, 2])

        # Check for metallic (high variance in value channel)
        val_variance = np.var(body_region[:, :, 2])
        is_metallic = val_variance > 1000

        # Determine primary color
        primary_color = self._classify_color(avg_hue, avg_sat, avg_val)

        # Check for secondary color (two-tone)
        secondary_color = self._detect_secondary_color(hsv, primary_color)

        # Calculate confidence based on saturation consistency
        sat_std = np.std(body_region[:, :, 1])
        confidence = max(0.5, 1.0 - (sat_std / 128))

        return VehicleColorResult(
            primary_color=primary_color,
            secondary_color=secondary_color,
            color_confidence=confidence,
            is_metallic=is_metallic,
            hsv_values=(float(avg_hue), float(avg_sat), float(avg_val)),
        )

    def _classify_color(
        self,
        hue: float,
        saturation: float,
        value: float,
    ) -> str:
        """Classify color from HSV values."""
        # Low saturation = grayscale
        if saturation < self.config.saturation_threshold:
            if value < 50:
                return "black"
            elif value > 200:
                return "white"
            elif value > 150:
                return "silver"
            else:
                return "gray"

        # Map hue to color
        for color_name, (low, high) in COLOR_RANGES.items():
            if low[0] <= hue <= high[0]:
                if color_name == "red2":
                    return "red"
                return color_name

        # Brown (low saturation orange/yellow)
        if 10 <= hue <= 35 and saturation < 100:
            return "brown"

        return "unknown"

    def _detect_secondary_color(
        self,
        hsv: np.ndarray,
        primary_color: str,
    ) -> Optional[str]:
        """Detect secondary color for two-tone vehicles."""
        h, w = hsv.shape[:2]

        # Sample top and bottom regions
        top_region = hsv[:int(h * 0.3), :]
        bottom_region = hsv[int(h * 0.7):, :]

        if top_region.size == 0 or bottom_region.size == 0:
            return None

        top_hue = np.mean(top_region[:, :, 0])
        bottom_hue = np.mean(bottom_region[:, :, 0])

        # Significant hue difference indicates two-tone
        if abs(top_hue - bottom_hue) > 30:
            top_sat = np.mean(top_region[:, :, 1])
            top_val = np.mean(top_region[:, :, 2])
            secondary = self._classify_color(top_hue, top_sat, top_val)

            if secondary != primary_color:
                return secondary

        return None


class VehicleTypeClassifier:
    """Classify vehicle type from image.

    Uses shape analysis and aspect ratio heuristics.
    """

    def __init__(self, config: Optional[VehiclePipelineConfig] = None):
        """Initialize type classifier.

        Args:
            config: Configuration options
        """
        self.config = config or VehiclePipelineConfig()

    def classify(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        class_hint: Optional[str] = None,
    ) -> VehicleTypeResult:
        """Classify vehicle type.

        Args:
            image: BGR image of vehicle
            mask: Optional segmentation mask
            class_hint: Optional class from detector

        Returns:
            VehicleTypeResult with type info
        """
        h, w = image.shape[:2]
        aspect_ratio = w / max(h, 1)

        # Use class hint if available
        if class_hint:
            vehicle_type = self._type_from_hint(class_hint)
            if vehicle_type != VehicleType.UNKNOWN:
                orientation = self._estimate_orientation(image, mask)
                return VehicleTypeResult(
                    vehicle_type=vehicle_type,
                    confidence=0.8,
                    orientation=orientation,
                    aspect_ratio=aspect_ratio,
                )

        # Classify based on shape
        vehicle_type = self._classify_from_shape(image, mask, aspect_ratio)
        orientation = self._estimate_orientation(image, mask)

        # Calculate confidence based on how well it matches expected ratios
        confidence = self._calculate_confidence(vehicle_type, aspect_ratio)

        return VehicleTypeResult(
            vehicle_type=vehicle_type,
            confidence=confidence,
            orientation=orientation,
            aspect_ratio=aspect_ratio,
        )

    def _type_from_hint(self, class_hint: str) -> VehicleType:
        """Map detection class to vehicle type."""
        hint_lower = class_hint.lower()

        mapping = {
            "car": VehicleType.SEDAN,
            "sedan": VehicleType.SEDAN,
            "suv": VehicleType.SUV,
            "truck": VehicleType.TRUCK,
            "pickup": VehicleType.PICKUP,
            "van": VehicleType.VAN,
            "bus": VehicleType.BUS,
            "motorcycle": VehicleType.MOTORCYCLE,
            "motorbike": VehicleType.MOTORCYCLE,
            "bicycle": VehicleType.BICYCLE,
            "bike": VehicleType.BICYCLE,
        }

        return mapping.get(hint_lower, VehicleType.UNKNOWN)

    def _classify_from_shape(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        aspect_ratio: float,
    ) -> VehicleType:
        """Classify vehicle type from shape features."""
        h, w = image.shape[:2]

        # Very narrow = motorcycle/bicycle
        if aspect_ratio < 0.8:
            # Check if it's two-wheeled
            if self._detect_wheels(image) <= 2:
                if h > w * 1.5:
                    return VehicleType.MOTORCYCLE
                else:
                    return VehicleType.BICYCLE

        # Very wide = bus or truck
        if aspect_ratio > 2.5:
            if h > 200:  # Large vehicle
                return VehicleType.BUS
            return VehicleType.TRUCK

        # Tall = SUV or van
        if aspect_ratio < 1.3:
            # Check overall size
            area = h * w
            if area > 50000:
                return VehicleType.VAN
            return VehicleType.SUV

        # Standard proportions = sedan
        if 1.3 <= aspect_ratio <= 2.0:
            return VehicleType.SEDAN

        # Wide but not extremely = pickup
        if 2.0 < aspect_ratio <= 2.5:
            return VehicleType.PICKUP

        return VehicleType.UNKNOWN

    def _detect_wheels(self, image: np.ndarray) -> int:
        """Count visible wheels (simplified)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect circles (wheels)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100,
        )

        if circles is not None:
            return len(circles[0])
        return 0

    def _estimate_orientation(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> VehicleOrientation:
        """Estimate vehicle orientation."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Analyze left vs right symmetry
        left_half = gray[:, :w // 2]
        right_half = gray[:, w // 2:]
        right_half_flipped = cv2.flip(right_half, 1)

        # Resize to same dimensions
        min_w = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_w]
        right_half_flipped = right_half_flipped[:, :min_w]

        # Compare symmetry
        diff = np.abs(left_half.astype(float) - right_half_flipped.astype(float))
        asymmetry = np.mean(diff)

        # High symmetry = front or rear view
        if asymmetry < 30:
            # Check for headlights (brighter top portion)
            top_brightness = np.mean(gray[:h // 3, :])
            bottom_brightness = np.mean(gray[2 * h // 3:, :])

            if top_brightness > bottom_brightness + 20:
                return VehicleOrientation.FRONT
            else:
                return VehicleOrientation.REAR

        # Asymmetric = side view
        left_brightness = np.mean(gray[:, :w // 3])
        right_brightness = np.mean(gray[:, 2 * w // 3:])

        if left_brightness > right_brightness + 10:
            return VehicleOrientation.SIDE_LEFT
        elif right_brightness > left_brightness + 10:
            return VehicleOrientation.SIDE_RIGHT

        return VehicleOrientation.UNKNOWN

    def _calculate_confidence(
        self,
        vehicle_type: VehicleType,
        aspect_ratio: float,
    ) -> float:
        """Calculate classification confidence."""
        # Expected aspect ratios for each type
        expected_ratios = {
            VehicleType.SEDAN: 1.5,
            VehicleType.SUV: 1.2,
            VehicleType.TRUCK: 2.0,
            VehicleType.VAN: 1.0,
            VehicleType.BUS: 3.0,
            VehicleType.PICKUP: 2.2,
            VehicleType.MOTORCYCLE: 0.5,
            VehicleType.BICYCLE: 0.6,
        }

        expected = expected_ratios.get(vehicle_type, 1.5)
        ratio_diff = abs(aspect_ratio - expected) / expected

        return max(0.3, 1.0 - ratio_diff)


class LicensePlateDetector:
    """Detect and localize license plates.

    Uses edge detection and contour analysis.
    """

    def __init__(self, config: Optional[VehiclePipelineConfig] = None):
        """Initialize plate detector.

        Args:
            config: Configuration options
        """
        self.config = config or VehiclePipelineConfig()

    def detect(self, image: np.ndarray) -> LicensePlateResult:
        """Detect license plate in vehicle image.

        Args:
            image: BGR image of vehicle

        Returns:
            LicensePlateResult with plate info
        """
        if not self.config.enable_license_plate:
            return LicensePlateResult(detected=False)

        h, w = image.shape[:2]

        # Focus on lower portion where plates typically are
        roi = image[int(h * 0.4):, :]
        roi_offset = int(h * 0.4)

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Filter contours by aspect ratio and size
        candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            if area < self.config.plate_min_area:
                continue

            aspect = cw / max(ch, 1)
            if not (self.config.plate_aspect_min <= aspect <= self.config.plate_aspect_max):
                continue

            # Approximate contour
            epsilon = self.config.contour_epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Plates are roughly rectangular (4 corners)
            if 4 <= len(approx) <= 6:
                candidates.append({
                    "bbox": (x, y + roi_offset, x + cw, y + roi_offset + ch),
                    "area": area,
                    "aspect": aspect,
                })

        if not candidates:
            return LicensePlateResult(detected=False)

        # Take largest candidate
        best = max(candidates, key=lambda c: c["area"])

        return LicensePlateResult(
            detected=True,
            bbox=best["bbox"],
            text=None,  # OCR would be needed for text
            confidence=0.7,
        )


class VehiclePipeline:
    """Complete vehicle classification pipeline.

    Combines color analysis, type classification, and plate detection.
    """

    def __init__(self, config: Optional[VehiclePipelineConfig] = None):
        """Initialize pipeline.

        Args:
            config: Configuration options
        """
        self.config = config or VehiclePipelineConfig()
        self.color_analyzer = VehicleColorAnalyzer(self.config)
        self.type_classifier = VehicleTypeClassifier(self.config)
        self.plate_detector = LicensePlateDetector(self.config)

    def process_object(
        self,
        obj: ObjectSchema,
        crop: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> ObjectSchema:
        """Process vehicle through all modules.

        Args:
            obj: ObjectSchema to populate
            crop: Image crop of vehicle
            mask: Optional segmentation mask

        Returns:
            ObjectSchema with vehicle attributes
        """
        # Get class hint from object
        class_hint = obj.primary_class

        # Color analysis
        color_result = self.color_analyzer.analyze(crop, mask)
        obj.attributes["color"] = color_result.to_dict()

        # Type classification
        type_result = self.type_classifier.classify(crop, mask, class_hint)
        obj.attributes["vehicle_type"] = type_result.to_dict()

        # Update primary class if we have high confidence
        if type_result.confidence > 0.7:
            obj.primary_class = type_result.vehicle_type.value

        # License plate detection
        plate_result = self.plate_detector.detect(crop)
        obj.attributes["license_plate"] = plate_result.to_dict()

        # Shape features
        obj.attributes["shape"] = {
            "aspect_ratio": type_result.aspect_ratio,
            "orientation": type_result.orientation.value,
        }

        return obj


class VehiclePipelineModule(PipelineModule):
    """Pipeline module wrapper for vehicle pipeline."""

    def __init__(self, config: Optional[VehiclePipelineConfig] = None):
        """Initialize module.

        Args:
            config: Configuration options
        """
        self.config = config or VehiclePipelineConfig()
        self.pipeline = VehiclePipeline(self.config)

    @property
    def name(self) -> str:
        return "vehicle_pipeline"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="vehicle_crop",
            required_fields=["object", "crop"],
            optional_fields=["mask"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="vehicle_attributes",
            required_fields=["object"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Process vehicle through pipeline.

        Args:
            data: PipelineData with object and crop

        Returns:
            PipelineData with populated attributes
        """
        obj = data.get("object")
        crop = data.get("crop")
        mask = data.get("mask")

        if obj and crop is not None:
            obj = self.pipeline.process_object(obj, crop, mask)

        result = data.copy()
        result.object = obj
        return result

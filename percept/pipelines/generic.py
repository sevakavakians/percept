"""Generic object classification pipeline for PERCEPT.

Handles unknown objects with:
- ImageNet classification (ResNet-50)
- Color analysis
- Shape features
- Size estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.core.schema import ObjectSchema


def is_hailo_available() -> bool:
    """Check if Hailo runtime is available."""
    try:
        from hailo_platform import HEF, VDevice
        return True
    except ImportError:
        return False


HAILO_AVAILABLE = is_hailo_available()

# ImageNet class names (subset of common categories)
IMAGENET_CLASSES = [
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
    "electric_ray", "stingray", "cock", "hen", "ostrich", "brambling",
    "goldfinch", "house_finch", "junco", "indigo_bunting", "robin",
    "bulbul", "jay", "magpie", "chickadee", "water_ouzel", "kite",
    "bald_eagle", "vulture", "great_grey_owl", "european_fire_salamander",
    "common_newt", "eft", "spotted_salamander", "axolotl", "bullfrog",
    "tree_frog", "tailed_frog", "loggerhead", "leatherback_turtle",
    "mud_turtle", "terrapin", "box_turtle", "banded_gecko", "common_iguana",
    "american_chameleon", "whiptail", "agama", "frilled_lizard",
    "alligator_lizard", "gila_monster", "green_lizard", "african_chameleon",
    "komodo_dragon", "african_crocodile", "american_alligator", "triceratops",
    # ... truncated for brevity, full list would have 1000 entries
    "laptop", "mouse", "remote_control", "keyboard", "cell_phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush",
    "table", "chair", "couch", "bed", "dining_table", "toilet",
    "monitor", "desk", "wardrobe", "lamp", "bathtub", "cushion",
    "pillow", "blanket", "curtain", "plant", "flower", "pot",
    "bottle", "cup", "bowl", "fork", "knife", "spoon", "plate",
    "bag", "backpack", "suitcase", "umbrella", "shoe", "hat", "glasses",
]


@dataclass
class GenericPipelineConfig:
    """Configuration for generic pipeline."""

    # Model paths
    imagenet_model_path: str = "/usr/share/hailo-models/resnet_v1_50_h8l.hef"

    # Classification
    classification_confidence: float = 0.3
    top_k_classes: int = 5

    # Color analysis
    color_bins: int = 8
    dominant_colors_count: int = 3

    # Shape analysis
    contour_epsilon: float = 0.02
    min_contour_area: int = 100


@dataclass
class ClassificationResult:
    """Result of ImageNet classification."""
    class_name: str
    class_id: int
    confidence: float
    top_k: List[Tuple[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "top_k": [{"class": name, "confidence": conf} for name, conf in self.top_k],
        }


@dataclass
class ColorResult:
    """Result of color analysis."""
    dominant_colors: List[Tuple[str, float]]  # (color_name, percentage)
    color_histogram: Optional[np.ndarray]
    is_multicolored: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dominant": [
                {"color": name, "percentage": pct}
                for name, pct in self.dominant_colors
            ],
            "multicolored": self.is_multicolored,
        }


@dataclass
class ShapeResult:
    """Result of shape analysis."""
    aspect_ratio: float
    solidity: float  # Area / convex hull area
    extent: float  # Area / bounding box area
    circularity: float
    num_vertices: int
    shape_type: str  # "rectangular", "circular", "irregular"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "aspect_ratio": self.aspect_ratio,
            "solidity": self.solidity,
            "extent": self.extent,
            "circularity": self.circularity,
            "vertices": self.num_vertices,
            "type": self.shape_type,
        }


# Color name mapping (HSV ranges)
COLOR_RANGES = {
    "red": [(0, 100, 100), (10, 255, 255)],
    "orange": [(10, 100, 100), (25, 255, 255)],
    "yellow": [(25, 100, 100), (35, 255, 255)],
    "green": [(35, 100, 100), (85, 255, 255)],
    "cyan": [(85, 100, 100), (100, 255, 255)],
    "blue": [(100, 100, 100), (130, 255, 255)],
    "purple": [(130, 100, 100), (160, 255, 255)],
    "pink": [(160, 100, 100), (180, 255, 255)],
}


class ImageNetClassifier:
    """ImageNet classification using ResNet-50 on Hailo-8.

    Falls back to returning "unknown" when Hailo unavailable.
    """

    def __init__(self, config: Optional[GenericPipelineConfig] = None):
        """Initialize classifier.

        Args:
            config: Configuration options
        """
        self.config = config or GenericPipelineConfig()
        self._inference = None
        self._configured = False
        self.input_shape = (224, 224)
        self._preprocess_info = {}

        if HAILO_AVAILABLE:
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Try to load the ImageNet model."""
        try:
            from hailo_platform import HEF

            model_path = Path(self.config.imagenet_model_path)
            if not model_path.exists():
                model_path = Path("/home/sevak/ClaudeHome/hailo-agents/models/resnet_v1_50_h8l.hef")

            if model_path.exists():
                self._hef = HEF(str(model_path))
                self._vdevice = None

                input_info = self._hef.get_input_vstream_infos()
                if input_info:
                    self.input_shape = (input_info[0].shape[0], input_info[0].shape[1])
                    self.input_name = input_info[0].name

                self._inference = True
        except Exception:
            self._inference = None

    def _configure(self) -> None:
        """Configure the Hailo device."""
        if self._configured or not self._inference:
            return

        try:
            from hailo_platform import VDevice, InputVStreamParams, OutputVStreamParams

            self._vdevice = VDevice()
            self.network_group = self._vdevice.configure(self._hef)[0]
            self._input_vstreams_params = InputVStreamParams.make(self.network_group)
            self._output_vstreams_params = OutputVStreamParams.make(self.network_group)
            self._configured = True
        except Exception:
            self._inference = None

    def is_available(self) -> bool:
        """Check if classification is available."""
        return self._inference is not None

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """Classify image using ImageNet model.

        Args:
            image: BGR image

        Returns:
            ClassificationResult with top classes
        """
        if not self.is_available():
            return ClassificationResult(
                class_name="unknown",
                class_id=-1,
                confidence=0.0,
                top_k=[],
            )

        if not self._configured:
            self._configure()
            if not self._configured:
                return ClassificationResult(
                    class_name="unknown",
                    class_id=-1,
                    confidence=0.0,
                    top_k=[],
                )

        try:
            from hailo_platform import InferVStreams

            # Preprocess
            preprocessed = self._preprocess(image)
            input_data = {self.input_name: np.expand_dims(preprocessed, 0)}

            # Run inference
            with InferVStreams(
                self.network_group,
                self._input_vstreams_params,
                self._output_vstreams_params
            ) as infer_pipeline:
                with self.network_group.activate():
                    results = infer_pipeline.infer(input_data)

            # Postprocess
            return self._postprocess(results)

        except Exception:
            return ClassificationResult(
                class_name="unknown",
                class_id=-1,
                confidence=0.0,
                top_k=[],
            )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ResNet."""
        target_h, target_w = self.input_shape

        # Resize to target size
        resized = cv2.resize(image, (target_w, target_h))

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize (ImageNet preprocessing)
        normalized = rgb.astype(np.float32) / 255.0

        return normalized

    def _postprocess(self, outputs: Dict[str, np.ndarray]) -> ClassificationResult:
        """Postprocess classification outputs."""
        for name, output in outputs.items():
            if isinstance(output, np.ndarray):
                # Get probabilities
                probs = output.flatten()

                # Apply softmax if needed
                if probs.max() > 1.0 or probs.min() < 0.0:
                    exp_probs = np.exp(probs - probs.max())
                    probs = exp_probs / exp_probs.sum()

                # Get top-k
                top_indices = np.argsort(probs)[::-1][:self.config.top_k_classes]

                top_k = []
                for idx in top_indices:
                    if idx < len(IMAGENET_CLASSES):
                        class_name = IMAGENET_CLASSES[idx]
                    else:
                        class_name = f"class_{idx}"
                    top_k.append((class_name, float(probs[idx])))

                best_idx = top_indices[0]
                if best_idx < len(IMAGENET_CLASSES):
                    best_class = IMAGENET_CLASSES[best_idx]
                else:
                    best_class = f"class_{best_idx}"

                return ClassificationResult(
                    class_name=best_class,
                    class_id=int(best_idx),
                    confidence=float(probs[best_idx]),
                    top_k=top_k,
                )

        return ClassificationResult(
            class_name="unknown",
            class_id=-1,
            confidence=0.0,
            top_k=[],
        )


class ColorAnalyzer:
    """Analyze object colors.

    Extracts dominant colors and color distribution.
    """

    def __init__(self, config: Optional[GenericPipelineConfig] = None):
        """Initialize color analyzer.

        Args:
            config: Configuration options
        """
        self.config = config or GenericPipelineConfig()

    def analyze(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> ColorResult:
        """Analyze colors in image.

        Args:
            image: BGR image
            mask: Optional binary mask

        Returns:
            ColorResult with color info
        """
        if image.size == 0:
            return ColorResult(
                dominant_colors=[("unknown", 1.0)],
                color_histogram=None,
                is_multicolored=False,
            )

        # Apply mask if provided
        if mask is not None and mask.size > 0:
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            masked_pixels = image[mask > 0]
        else:
            masked_pixels = image.reshape(-1, 3)

        if len(masked_pixels) == 0:
            return ColorResult(
                dominant_colors=[("unknown", 1.0)],
                color_histogram=None,
                is_multicolored=False,
            )

        # Convert to HSV
        hsv_pixels = cv2.cvtColor(
            masked_pixels.reshape(-1, 1, 3),
            cv2.COLOR_BGR2HSV
        ).reshape(-1, 3)

        # Calculate color histogram
        hue_hist, _ = np.histogram(hsv_pixels[:, 0], bins=180, range=(0, 180))
        sat_hist, _ = np.histogram(hsv_pixels[:, 1], bins=256, range=(0, 256))
        val_hist, _ = np.histogram(hsv_pixels[:, 2], bins=256, range=(0, 256))

        # Find dominant colors
        dominant_colors = self._find_dominant_colors(hsv_pixels)

        # Check if multicolored
        is_multicolored = len(dominant_colors) > 2 or (
            len(dominant_colors) == 2 and dominant_colors[1][1] > 0.3
        )

        return ColorResult(
            dominant_colors=dominant_colors,
            color_histogram=hue_hist,
            is_multicolored=is_multicolored,
        )

    def _find_dominant_colors(
        self,
        hsv_pixels: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """Find dominant colors in pixel array."""
        total_pixels = len(hsv_pixels)
        if total_pixels == 0:
            return [("unknown", 1.0)]

        color_counts: Dict[str, int] = {}

        for pixel in hsv_pixels:
            h, s, v = pixel

            # Classify pixel color
            if s < 30:
                # Grayscale
                if v < 50:
                    color = "black"
                elif v > 200:
                    color = "white"
                else:
                    color = "gray"
            else:
                # Chromatic
                color = self._classify_hue(h)

            color_counts[color] = color_counts.get(color, 0) + 1

        # Sort by count
        sorted_colors = sorted(
            color_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Convert to percentages
        dominant = [
            (color, count / total_pixels)
            for color, count in sorted_colors[:self.config.dominant_colors_count]
        ]

        return dominant

    def _classify_hue(self, hue: float) -> str:
        """Classify hue value to color name."""
        for color_name, (low, high) in COLOR_RANGES.items():
            if low[0] <= hue <= high[0]:
                return color_name

        # Red wraps around
        if hue > 170 or hue < 10:
            return "red"

        return "unknown"


class ShapeAnalyzer:
    """Analyze object shape features.

    Extracts geometric properties from contours.
    """

    def __init__(self, config: Optional[GenericPipelineConfig] = None):
        """Initialize shape analyzer.

        Args:
            config: Configuration options
        """
        self.config = config or GenericPipelineConfig()

    def analyze(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> ShapeResult:
        """Analyze shape features.

        Args:
            image: BGR image
            mask: Optional binary mask

        Returns:
            ShapeResult with shape info
        """
        h, w = image.shape[:2]

        if mask is not None and mask.size > 0:
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (w, h))
            binary = (mask > 127).astype(np.uint8) * 255
        else:
            # Use edge detection to find contour
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return ShapeResult(
                aspect_ratio=w / max(h, 1),
                solidity=1.0,
                extent=1.0,
                circularity=0.0,
                num_vertices=4,
                shape_type="rectangular",
            )

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area < self.config.min_contour_area:
            return ShapeResult(
                aspect_ratio=w / max(h, 1),
                solidity=1.0,
                extent=1.0,
                circularity=0.0,
                num_vertices=4,
                shape_type="rectangular",
            )

        # Calculate features
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect_ratio = cw / max(ch, 1)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)

        bbox_area = cw * ch
        extent = area / max(bbox_area, 1)

        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / max(perimeter ** 2, 1)

        # Approximate contour for vertex count
        epsilon = self.config.contour_epsilon * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)

        # Classify shape type
        shape_type = self._classify_shape(
            aspect_ratio, circularity, num_vertices, solidity
        )

        return ShapeResult(
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            extent=extent,
            circularity=circularity,
            num_vertices=num_vertices,
            shape_type=shape_type,
        )

    def _classify_shape(
        self,
        aspect_ratio: float,
        circularity: float,
        num_vertices: int,
        solidity: float,
    ) -> str:
        """Classify shape type from features."""
        # High circularity = circle/oval
        if circularity > 0.8:
            if 0.8 < aspect_ratio < 1.2:
                return "circular"
            else:
                return "oval"

        # Rectangle check
        if num_vertices == 4 and solidity > 0.9:
            if 0.8 < aspect_ratio < 1.2:
                return "square"
            else:
                return "rectangular"

        # Triangle
        if num_vertices == 3:
            return "triangular"

        # Low solidity = irregular
        if solidity < 0.7:
            return "irregular"

        return "polygon"


class SizeEstimator:
    """Estimate physical size of objects.

    Uses depth information if available.
    """

    def __init__(self, config: Optional[GenericPipelineConfig] = None):
        """Initialize size estimator.

        Args:
            config: Configuration options
        """
        self.config = config or GenericPipelineConfig()

    def estimate(
        self,
        bbox: Tuple[int, int, int, int],
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Estimate object size.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            depth: Optional depth map
            intrinsics: Optional camera intrinsics {fx, fy, cx, cy}

        Returns:
            Size estimation dict
        """
        x1, y1, x2, y2 = bbox
        pixel_width = x2 - x1
        pixel_height = y2 - y1

        result = {
            "pixel_width": pixel_width,
            "pixel_height": pixel_height,
            "pixel_area": pixel_width * pixel_height,
        }

        if depth is not None and intrinsics is not None:
            # Get median depth in bbox region
            depth_region = depth[y1:y2, x1:x2]
            if depth_region.size > 0:
                median_depth = np.median(depth_region[depth_region > 0])

                if median_depth > 0:
                    fx = intrinsics.get("fx", 600)
                    fy = intrinsics.get("fy", 600)

                    # Convert to meters
                    real_width = (pixel_width * median_depth) / fx
                    real_height = (pixel_height * median_depth) / fy

                    result["real_width_m"] = float(real_width)
                    result["real_height_m"] = float(real_height)
                    result["depth_m"] = float(median_depth)

        return result


class GenericPipeline:
    """Complete generic object classification pipeline.

    Combines classification, color, shape, and size analysis.
    """

    def __init__(self, config: Optional[GenericPipelineConfig] = None):
        """Initialize pipeline.

        Args:
            config: Configuration options
        """
        self.config = config or GenericPipelineConfig()
        self.classifier = ImageNetClassifier(self.config)
        self.color_analyzer = ColorAnalyzer(self.config)
        self.shape_analyzer = ShapeAnalyzer(self.config)
        self.size_estimator = SizeEstimator(self.config)

    def process_object(
        self,
        obj: ObjectSchema,
        crop: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> ObjectSchema:
        """Process object through all modules.

        Args:
            obj: ObjectSchema to populate
            crop: Image crop of object
            mask: Optional segmentation mask
            depth: Optional depth map
            intrinsics: Optional camera intrinsics

        Returns:
            ObjectSchema with attributes
        """
        # ImageNet classification
        classification = self.classifier.classify(crop)
        obj.attributes["classification"] = classification.to_dict()

        # Update primary class if confident
        if classification.confidence > self.config.classification_confidence:
            obj.primary_class = classification.class_name

        # Color analysis
        color_result = self.color_analyzer.analyze(crop, mask)
        obj.attributes["color"] = color_result.to_dict()

        # Shape analysis
        shape_result = self.shape_analyzer.analyze(crop, mask)
        obj.attributes["shape"] = shape_result.to_dict()

        # Size estimation
        if obj.bounding_box_2d and obj.bounding_box_2d != (0, 0, 0, 0):
            size_result = self.size_estimator.estimate(
                obj.bounding_box_2d, depth, intrinsics
            )
            obj.attributes["size"] = size_result

        return obj


class GenericPipelineModule(PipelineModule):
    """Pipeline module wrapper for generic pipeline."""

    def __init__(self, config: Optional[GenericPipelineConfig] = None):
        """Initialize module.

        Args:
            config: Configuration options
        """
        self.config = config or GenericPipelineConfig()
        self.pipeline = GenericPipeline(self.config)

    @property
    def name(self) -> str:
        return "generic_pipeline"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="object_crop",
            required_fields=["object", "crop"],
            optional_fields=["mask", "depth", "intrinsics"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="object_attributes",
            required_fields=["object"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Process object through pipeline.

        Args:
            data: PipelineData with object and crop

        Returns:
            PipelineData with populated attributes
        """
        obj = data.get("object")
        crop = data.get("crop")
        mask = data.get("mask")
        depth = data.get("depth")
        intrinsics = data.get("intrinsics")

        if obj and crop is not None:
            obj = self.pipeline.process_object(obj, crop, mask, depth, intrinsics)

        result = data.copy()
        result.object = obj
        return result

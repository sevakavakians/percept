"""Person classification pipeline for PERCEPT.

Extracts detailed attributes from detected persons:
- Pose estimation (17 keypoints)
- Posture classification
- Clothing analysis (colors, types)
- Face detection (triggers sub-pipeline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.core.schema import ObjectSchema


class Posture(Enum):
    """Person posture classification."""
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    CROUCHING = "crouching"
    UNKNOWN = "unknown"


# COCO keypoint indices
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

KEYPOINT_INDICES = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Skeleton connections for visualization
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]


def is_hailo_available() -> bool:
    """Check if Hailo runtime is available."""
    try:
        from hailo_platform import HEF, VDevice
        return True
    except ImportError:
        return False


HAILO_AVAILABLE = is_hailo_available()


@dataclass
class PersonPipelineConfig:
    """Configuration for person pipeline."""

    # Model paths
    pose_model_path: str = "/usr/share/hailo-models/yolov8s_pose_h8.hef"
    face_model_path: str = "/usr/share/hailo-models/scrfd_2.5g_h8l.hef"

    # Detection thresholds
    pose_confidence: float = 0.5
    keypoint_confidence: float = 0.3
    face_confidence: float = 0.5

    # Clothing analysis
    clothing_regions: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "upper": (0.1, 0.2, 0.9, 0.5),  # Upper body region (relative)
            "lower": (0.1, 0.5, 0.9, 0.95),  # Lower body region
        }
    )
    color_bins: int = 8  # Color histogram bins per channel

    # Processing options
    enable_pose: bool = True
    enable_clothing: bool = True
    enable_face: bool = True


@dataclass
class Keypoint:
    """A single pose keypoint."""
    x: float
    y: float
    confidence: float
    name: str

    def is_visible(self, threshold: float = 0.3) -> bool:
        """Check if keypoint is visible."""
        return self.confidence >= threshold


@dataclass
class PoseResult:
    """Result of pose estimation."""
    keypoints: List[Keypoint]
    bbox: Tuple[int, int, int, int]
    confidence: float
    posture: Posture = Posture.UNKNOWN

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        """Get keypoint by name."""
        idx = KEYPOINT_INDICES.get(name)
        if idx is not None and idx < len(self.keypoints):
            return self.keypoints[idx]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "keypoints": {
                kp.name: {"x": kp.x, "y": kp.y, "confidence": kp.confidence}
                for kp in self.keypoints if kp.confidence > 0.1
            },
            "posture": self.posture.value,
            "confidence": self.confidence,
        }


@dataclass
class ClothingResult:
    """Result of clothing analysis."""
    upper_color: str
    upper_type: str
    lower_color: str
    lower_type: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "upper": {"color": self.upper_color, "type": self.upper_type},
            "lower": {"color": self.lower_color, "type": self.lower_type},
            "confidence": self.confidence,
        }


class PoseEstimator:
    """Pose estimation using YOLOv8-pose on Hailo-8.

    Falls back to returning empty results when Hailo unavailable.
    """

    def __init__(self, config: Optional[PersonPipelineConfig] = None):
        """Initialize pose estimator.

        Args:
            config: Configuration options
        """
        self.config = config or PersonPipelineConfig()
        self._inference = None
        self._configured = False
        self.input_shape = (640, 640)
        self._preprocess_info = {}

        if HAILO_AVAILABLE:
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Try to load the pose model."""
        try:
            from hailo_platform import HEF, VDevice

            model_path = Path(self.config.pose_model_path)
            if not model_path.exists():
                # Try alternate path
                model_path = Path("/home/sevak/ClaudeHome/hailo-agents/models/yolov8s_pose_h8.hef")

            if model_path.exists():
                self._hef = HEF(str(model_path))
                self._vdevice = None

                # Get input info
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
        """Check if pose estimation is available."""
        return self._inference is not None

    def estimate(self, image: np.ndarray) -> List[PoseResult]:
        """Estimate poses in image.

        Args:
            image: BGR image

        Returns:
            List of PoseResult instances
        """
        if not self.is_available():
            return []

        if not self._configured:
            self._configure()
            if not self._configured:
                return []

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
            return []

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for pose model."""
        target_h, target_w = self.input_shape

        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to target size
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        self._preprocess_info = {
            'scale': scale,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'orig_h': h,
            'orig_w': w,
        }

        return rgb.astype(np.uint8)

    def _postprocess(self, outputs: Dict[str, np.ndarray]) -> List[PoseResult]:
        """Postprocess pose estimation outputs."""
        poses = []
        info = self._preprocess_info
        input_h, input_w = self.input_shape

        for name, output in outputs.items():
            # Handle various output formats
            if isinstance(output, list) and len(output) > 0:
                batch_output = output[0]

                # YOLOv8-pose output format
                if isinstance(batch_output, np.ndarray):
                    self._parse_pose_detections(batch_output, poses, info, input_h, input_w)

            elif isinstance(output, np.ndarray):
                if output.ndim >= 2:
                    output = output[0] if output.shape[0] == 1 else output
                    self._parse_pose_detections(output, poses, info, input_h, input_w)

        return poses

    def _parse_pose_detections(
        self,
        detections: np.ndarray,
        poses: List[PoseResult],
        info: Dict,
        input_h: int,
        input_w: int,
    ) -> None:
        """Parse pose detections from output tensor."""
        # Expected format: [x, y, w, h, conf, kp1_x, kp1_y, kp1_conf, ...]
        # 17 keypoints * 3 = 51, plus 5 for bbox = 56 total
        for det in detections:
            if len(det) < 56:
                continue

            conf = float(det[4])
            if conf < self.config.pose_confidence:
                continue

            # Parse bounding box
            cx, cy, w, h = det[0:4]

            # Convert to pixel coordinates
            x1 = (cx - w/2) * input_w
            y1 = (cy - h/2) * input_h
            x2 = (cx + w/2) * input_w
            y2 = (cy + h/2) * input_h

            # Remove padding and scale
            x1 = (x1 - info['pad_w']) / info['scale']
            y1 = (y1 - info['pad_h']) / info['scale']
            x2 = (x2 - info['pad_w']) / info['scale']
            y2 = (y2 - info['pad_h']) / info['scale']

            # Clamp to image bounds
            x1 = max(0, min(info['orig_w'], x1))
            y1 = max(0, min(info['orig_h'], y1))
            x2 = max(0, min(info['orig_w'], x2))
            y2 = max(0, min(info['orig_h'], y2))

            bbox = (int(x1), int(y1), int(x2), int(y2))

            # Parse keypoints
            keypoints = []
            kp_data = det[5:56]  # 17 keypoints * 3 values each

            for i in range(17):
                kp_x = float(kp_data[i * 3]) * input_w
                kp_y = float(kp_data[i * 3 + 1]) * input_h
                kp_conf = float(kp_data[i * 3 + 2])

                # Convert to original coordinates
                kp_x = (kp_x - info['pad_w']) / info['scale']
                kp_y = (kp_y - info['pad_h']) / info['scale']

                keypoints.append(Keypoint(
                    x=kp_x,
                    y=kp_y,
                    confidence=kp_conf,
                    name=KEYPOINT_NAMES[i],
                ))

            # Classify posture
            posture = self._classify_posture(keypoints)

            poses.append(PoseResult(
                keypoints=keypoints,
                bbox=bbox,
                confidence=conf,
                posture=posture,
            ))

        return poses

    def _classify_posture(self, keypoints: List[Keypoint]) -> Posture:
        """Classify posture from keypoints."""
        # Get key points
        def get_kp(name: str) -> Optional[Keypoint]:
            idx = KEYPOINT_INDICES.get(name)
            if idx is not None and idx < len(keypoints):
                kp = keypoints[idx]
                if kp.confidence >= self.config.keypoint_confidence:
                    return kp
            return None

        l_hip = get_kp("left_hip")
        r_hip = get_kp("right_hip")
        l_knee = get_kp("left_knee")
        r_knee = get_kp("right_knee")
        l_ankle = get_kp("left_ankle")
        r_ankle = get_kp("right_ankle")
        l_shoulder = get_kp("left_shoulder")
        r_shoulder = get_kp("right_shoulder")

        # Need at least hips and shoulders
        if not (l_hip or r_hip) or not (l_shoulder or r_shoulder):
            return Posture.UNKNOWN

        # Calculate torso angle
        hip_y = (l_hip.y if l_hip else r_hip.y)
        shoulder_y = (l_shoulder.y if l_shoulder else r_shoulder.y)
        hip_x = (l_hip.x if l_hip else r_hip.x)
        shoulder_x = (l_shoulder.x if l_shoulder else r_shoulder.x)

        # Vertical distance between shoulders and hips
        torso_height = abs(hip_y - shoulder_y)

        # Horizontal spread of torso
        torso_width = abs(hip_x - shoulder_x)

        # If torso is more horizontal than vertical -> lying
        if torso_width > torso_height * 1.5:
            return Posture.LYING

        # Check leg positions for sitting vs standing
        if l_knee and l_hip and l_ankle:
            # Knee angle check
            hip_knee_dist = abs(l_hip.y - l_knee.y)
            knee_ankle_dist = abs(l_knee.y - l_ankle.y)

            if hip_knee_dist < knee_ankle_dist * 0.5:
                return Posture.SITTING

        if r_knee and r_hip and r_ankle:
            hip_knee_dist = abs(r_hip.y - r_knee.y)
            knee_ankle_dist = abs(r_knee.y - r_ankle.y)

            if hip_knee_dist < knee_ankle_dist * 0.5:
                return Posture.SITTING

        return Posture.STANDING


class ClothingAnalyzer:
    """Analyze clothing colors and types.

    Uses color histograms and simple heuristics.
    """

    # Color name mapping
    COLOR_RANGES = {
        "red": ((0, 100, 100), (10, 255, 255)),
        "orange": ((10, 100, 100), (25, 255, 255)),
        "yellow": ((25, 100, 100), (35, 255, 255)),
        "green": ((35, 100, 100), (85, 255, 255)),
        "blue": ((85, 100, 100), (130, 255, 255)),
        "purple": ((130, 100, 100), (160, 255, 255)),
        "pink": ((160, 100, 100), (180, 255, 255)),
    }

    def __init__(self, config: Optional[PersonPipelineConfig] = None):
        """Initialize clothing analyzer.

        Args:
            config: Configuration options
        """
        self.config = config or PersonPipelineConfig()

    def analyze(
        self,
        image: np.ndarray,
        keypoints: Optional[List[Keypoint]] = None,
    ) -> ClothingResult:
        """Analyze clothing in image.

        Args:
            image: BGR image crop of person
            keypoints: Optional pose keypoints for better segmentation

        Returns:
            ClothingResult with color and type info
        """
        h, w = image.shape[:2]

        # Define upper and lower regions
        if keypoints:
            upper_region, lower_region = self._get_regions_from_pose(keypoints, w, h)
        else:
            upper_region = self._get_default_region("upper", w, h)
            lower_region = self._get_default_region("lower", w, h)

        # Extract regions
        upper_crop = self._extract_region(image, upper_region)
        lower_crop = self._extract_region(image, lower_region)

        # Analyze colors
        upper_color = self._analyze_color(upper_crop)
        lower_color = self._analyze_color(lower_crop)

        # Simple type classification based on colors
        upper_type = self._classify_upper_type(upper_crop)
        lower_type = self._classify_lower_type(lower_crop)

        return ClothingResult(
            upper_color=upper_color,
            upper_type=upper_type,
            lower_color=lower_color,
            lower_type=lower_type,
            confidence=0.7,  # Heuristic confidence
        )

    def _get_regions_from_pose(
        self,
        keypoints: List[Keypoint],
        w: int,
        h: int,
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Get clothing regions from pose keypoints."""
        def get_kp(name: str) -> Optional[Keypoint]:
            idx = KEYPOINT_INDICES.get(name)
            if idx is not None and idx < len(keypoints):
                kp = keypoints[idx]
                if kp.confidence > 0.3:
                    return kp
            return None

        l_shoulder = get_kp("left_shoulder")
        r_shoulder = get_kp("right_shoulder")
        l_hip = get_kp("left_hip")
        r_hip = get_kp("right_hip")
        l_knee = get_kp("left_knee")
        r_knee = get_kp("right_knee")

        # Upper region: shoulders to hips
        if l_shoulder and r_shoulder and l_hip and r_hip:
            x1 = max(0, int(min(l_shoulder.x, r_shoulder.x) - w * 0.1))
            y1 = max(0, int(min(l_shoulder.y, r_shoulder.y)))
            x2 = min(w, int(max(l_shoulder.x, r_shoulder.x) + w * 0.1))
            y2 = min(h, int(max(l_hip.y, r_hip.y)))
            upper_region = (x1, y1, x2, y2)
        else:
            upper_region = self._get_default_region("upper", w, h)

        # Lower region: hips to knees/ankles
        if l_hip and r_hip:
            x1 = max(0, int(min(l_hip.x, r_hip.x) - w * 0.1))
            y1 = max(0, int(min(l_hip.y, r_hip.y)))
            x2 = min(w, int(max(l_hip.x, r_hip.x) + w * 0.1))

            if l_knee and r_knee:
                y2 = min(h, int(max(l_knee.y, r_knee.y) + h * 0.1))
            else:
                y2 = int(h * 0.95)

            lower_region = (x1, y1, x2, y2)
        else:
            lower_region = self._get_default_region("lower", w, h)

        return upper_region, lower_region

    def _get_default_region(
        self,
        region_type: str,
        w: int,
        h: int,
    ) -> Tuple[int, int, int, int]:
        """Get default region based on config."""
        rel_region = self.config.clothing_regions.get(region_type, (0, 0, 1, 1))
        return (
            int(w * rel_region[0]),
            int(h * rel_region[1]),
            int(w * rel_region[2]),
            int(h * rel_region[3]),
        )

    def _extract_region(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Extract image region."""
        x1, y1, x2, y2 = region
        return image[y1:y2, x1:x2]

    def _analyze_color(self, image: np.ndarray) -> str:
        """Analyze dominant color in image region."""
        if image.size == 0:
            return "unknown"

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calculate average saturation and value
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])

        # Low saturation = grayscale
        if avg_sat < 30:
            if avg_val < 60:
                return "black"
            elif avg_val > 200:
                return "white"
            else:
                return "gray"

        # Find dominant hue
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hue_hist)

        # Map hue to color name
        for color_name, (low, high) in self.COLOR_RANGES.items():
            if low[0] <= dominant_hue <= high[0]:
                return color_name

        # Check for brown (low saturation orange/red)
        if dominant_hue < 25 and avg_sat < 100:
            return "brown"

        return "unknown"

    def _classify_upper_type(self, image: np.ndarray) -> str:
        """Classify upper body clothing type."""
        if image.size == 0:
            return "unknown"

        h, w = image.shape[:2]

        # Simple heuristics based on shape
        aspect = w / max(h, 1)

        if aspect > 1.5:
            return "jacket"
        elif aspect < 0.6:
            return "vest"
        else:
            return "shirt"

    def _classify_lower_type(self, image: np.ndarray) -> str:
        """Classify lower body clothing type."""
        if image.size == 0:
            return "unknown"

        h, w = image.shape[:2]

        # Simple heuristics
        aspect = h / max(w, 1)

        if aspect > 2.0:
            return "pants"
        else:
            return "shorts"


class FaceDetector:
    """Face detection using SCRFD on Hailo-8.

    Falls back to OpenCV cascade when Hailo unavailable.
    """

    def __init__(self, config: Optional[PersonPipelineConfig] = None):
        """Initialize face detector.

        Args:
            config: Configuration options
        """
        self.config = config or PersonPipelineConfig()
        self._inference = None
        self._cascade = None
        self._configured = False
        self.input_shape = (640, 640)
        self._preprocess_info = {}

        if HAILO_AVAILABLE:
            self._try_load_model()

        # Load OpenCV cascade as fallback
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def _try_load_model(self) -> None:
        """Try to load the face model."""
        try:
            from hailo_platform import HEF

            model_path = Path(self.config.face_model_path)
            if not model_path.exists():
                model_path = Path("/home/sevak/ClaudeHome/hailo-agents/models/scrfd_2.5g_h8l.hef")

            if model_path.exists():
                self._hef = HEF(str(model_path))

                input_info = self._hef.get_input_vstream_infos()
                if input_info:
                    self.input_shape = (input_info[0].shape[0], input_info[0].shape[1])
                    self.input_name = input_info[0].name

                self._inference = True
        except Exception:
            self._inference = None

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image.

        Args:
            image: BGR image

        Returns:
            List of face detections with bbox and landmarks
        """
        if self._inference and HAILO_AVAILABLE:
            return self._detect_hailo(image)
        else:
            return self._detect_cascade(image)

    def _detect_hailo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect using Hailo SCRFD model."""
        # Similar to PoseEstimator - configure and run inference
        # Simplified for now
        return self._detect_cascade(image)

    def _detect_cascade(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect using OpenCV Haar cascade."""
        if self._cascade is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        results = []
        for (x, y, w, h) in faces:
            results.append({
                "bbox": (int(x), int(y), int(x + w), int(y + h)),
                "confidence": 0.8,  # Cascade doesn't provide confidence
                "landmarks": None,
            })

        return results


class PersonPipeline:
    """Complete person classification pipeline.

    Combines pose estimation, clothing analysis, and face detection.
    """

    def __init__(self, config: Optional[PersonPipelineConfig] = None):
        """Initialize pipeline.

        Args:
            config: Configuration options
        """
        self.config = config or PersonPipelineConfig()
        self.pose_estimator = PoseEstimator(self.config)
        self.clothing_analyzer = ClothingAnalyzer(self.config)
        self.face_detector = FaceDetector(self.config)

    def process_object(
        self,
        obj: ObjectSchema,
        crop: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> ObjectSchema:
        """Process person through all modules.

        Args:
            obj: ObjectSchema to populate
            crop: Image crop of person
            mask: Optional segmentation mask

        Returns:
            ObjectSchema with person attributes
        """
        # Pose estimation
        if self.config.enable_pose:
            poses = self.pose_estimator.estimate(crop)
            if poses:
                pose = poses[0]  # Take first/best pose
                obj.attributes["pose"] = pose.to_dict()
                obj.attributes["posture"] = pose.posture.value

                # Use keypoints for clothing analysis
                keypoints = pose.keypoints
            else:
                obj.attributes["pose"] = None
                obj.attributes["posture"] = "unknown"
                keypoints = None
        else:
            keypoints = None

        # Clothing analysis
        if self.config.enable_clothing:
            clothing = self.clothing_analyzer.analyze(crop, keypoints)
            obj.attributes["clothing"] = clothing.to_dict()

        # Face detection
        if self.config.enable_face:
            faces = self.face_detector.detect(crop)
            if faces:
                obj.attributes["face"] = {
                    "detected": True,
                    "count": len(faces),
                    "primary": faces[0],
                }
            else:
                obj.attributes["face"] = {"detected": False}

        return obj


class PersonPipelineModule(PipelineModule):
    """Pipeline module wrapper for person pipeline."""

    def __init__(self, config: Optional[PersonPipelineConfig] = None):
        """Initialize module.

        Args:
            config: Configuration options
        """
        self.config = config or PersonPipelineConfig()
        self.pipeline = PersonPipeline(self.config)

    @property
    def name(self) -> str:
        return "person_pipeline"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="person_crop",
            required_fields=["object", "crop"],
            optional_fields=["mask"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="person_attributes",
            required_fields=["object"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Process person through pipeline.

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

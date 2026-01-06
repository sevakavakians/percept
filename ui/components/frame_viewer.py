"""Frame viewer component for PERCEPT UI.

Handles frame capture, annotation, and streaming
for live camera display.
"""

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BoundingBox:
    """Bounding box annotation."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""
    confidence: float = 1.0
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR
    track_id: Optional[int] = None


@dataclass
class Annotation:
    """Collection of annotations for a frame."""
    boxes: List[BoundingBox]
    masks: List[np.ndarray] = None
    keypoints: List[List[Tuple[int, int, float]]] = None
    text_overlays: List[Tuple[str, int, int]] = None


@dataclass
class FrameMetadata:
    """Metadata for a captured frame."""
    frame_id: int
    camera_id: str
    timestamp: datetime
    width: int
    height: int
    fps: float = 0.0
    processing_time_ms: float = 0.0


# =============================================================================
# Frame Annotator
# =============================================================================

class FrameAnnotator:
    """Annotate frames with detection results."""

    def __init__(
        self,
        font_scale: float = 0.5,
        line_thickness: int = 2,
        show_confidence: bool = True,
        show_track_id: bool = True,
    ):
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.show_confidence = show_confidence
        self.show_track_id = show_track_id

        # Color palette for different classes
        self.class_colors = {
            "person": (0, 255, 0),    # Green
            "car": (255, 0, 0),        # Blue
            "truck": (255, 128, 0),    # Orange-blue
            "bicycle": (0, 255, 255),  # Yellow
            "dog": (255, 0, 255),      # Magenta
            "cat": (128, 0, 255),      # Purple
            "default": (0, 200, 200),  # Cyan
        }

    def annotate(
        self,
        frame: np.ndarray,
        annotation: Annotation,
    ) -> np.ndarray:
        """Annotate frame with detections."""
        if not CV2_AVAILABLE:
            return frame

        annotated = frame.copy()

        # Draw masks first (behind boxes)
        if annotation.masks:
            for mask in annotation.masks:
                if mask is not None and mask.shape[:2] == frame.shape[:2]:
                    # Create colored overlay
                    overlay = annotated.copy()
                    overlay[mask > 0] = (0, 255, 0)
                    annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        # Draw bounding boxes
        for box in annotation.boxes:
            color = self.class_colors.get(box.label, self.class_colors["default"])

            # Draw rectangle
            cv2.rectangle(
                annotated,
                (box.x1, box.y1),
                (box.x2, box.y2),
                color,
                self.line_thickness,
            )

            # Build label text
            label_parts = [box.label]
            if self.show_confidence:
                label_parts.append(f"{box.confidence:.2f}")
            if self.show_track_id and box.track_id is not None:
                label_parts.append(f"#{box.track_id}")
            label = " ".join(label_parts)

            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            cv2.rectangle(
                annotated,
                (box.x1, box.y1 - text_height - 10),
                (box.x1 + text_width + 4, box.y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (box.x1 + 2, box.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
            )

        # Draw keypoints
        if annotation.keypoints:
            for person_kpts in annotation.keypoints:
                for x, y, conf in person_kpts:
                    if conf > 0.5:
                        cv2.circle(annotated, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Draw text overlays
        if annotation.text_overlays:
            for text, x, y in annotation.text_overlays:
                cv2.putText(
                    annotated,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    1,
                )

        return annotated

    def add_info_overlay(
        self,
        frame: np.ndarray,
        metadata: FrameMetadata,
        stats: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Add information overlay to frame."""
        if not CV2_AVAILABLE:
            return frame

        annotated = frame.copy()
        y_offset = 20

        # Draw semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (5, 5), (200, 80), (0, 0, 0), -1)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        # Draw info text
        info_lines = [
            f"Camera: {metadata.camera_id}",
            f"Frame: {metadata.frame_id}",
            f"FPS: {metadata.fps:.1f}",
            f"Latency: {metadata.processing_time_ms:.1f}ms",
        ]

        for line in info_lines:
            cv2.putText(
                annotated,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            y_offset += 15

        return annotated


# =============================================================================
# Frame Encoder
# =============================================================================

class FrameEncoder:
    """Encode frames for streaming."""

    def __init__(self, quality: int = 80):
        self.quality = quality

    def encode_jpeg(self, frame: np.ndarray) -> bytes:
        """Encode frame as JPEG bytes."""
        if not CV2_AVAILABLE:
            return b""

        encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        return encoded.tobytes()

    def encode_png(self, frame: np.ndarray) -> bytes:
        """Encode frame as PNG bytes."""
        if not CV2_AVAILABLE:
            return b""

        _, encoded = cv2.imencode('.png', frame)
        return encoded.tobytes()

    def encode_base64(self, frame: np.ndarray) -> str:
        """Encode frame as base64 string."""
        import base64
        jpeg_bytes = self.encode_jpeg(frame)
        return base64.b64encode(jpeg_bytes).decode('utf-8')


# =============================================================================
# Depth Visualizer
# =============================================================================

class DepthVisualizer:
    """Visualize depth images."""

    def __init__(
        self,
        min_depth: float = 0.3,
        max_depth: float = 10.0,
        colormap: int = None,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.colormap = colormap if colormap is not None else (
            cv2.COLORMAP_JET if CV2_AVAILABLE else 0
        )

    def colorize(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth image to colorized RGB."""
        if not CV2_AVAILABLE:
            # Simple grayscale fallback
            normalized = np.clip(
                (depth - self.min_depth) / (self.max_depth - self.min_depth),
                0, 1
            )
            gray = (normalized * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)

        # Normalize depth
        depth_clipped = np.clip(depth, self.min_depth, self.max_depth)
        normalized = (depth_clipped - self.min_depth) / (self.max_depth - self.min_depth)
        depth_uint8 = (normalized * 255).astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(depth_uint8, self.colormap)

        # Mark invalid depth (zeros) as black
        colored[depth < 0.1] = 0

        return colored

    def create_depth_overlay(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create RGB-D composite visualization."""
        if not CV2_AVAILABLE:
            return rgb

        depth_colored = self.colorize(depth)

        # Resize if needed
        if depth_colored.shape[:2] != rgb.shape[:2]:
            depth_colored = cv2.resize(
                depth_colored,
                (rgb.shape[1], rgb.shape[0]),
            )

        return cv2.addWeighted(rgb, 1 - alpha, depth_colored, alpha, 0)


# =============================================================================
# Frame Manager
# =============================================================================

class FrameManager:
    """Manage frame capture and distribution."""

    def __init__(self, max_history: int = 30):
        self.max_history = max_history
        self._frames: Dict[str, List[Tuple[np.ndarray, FrameMetadata]]] = {}
        self._latest: Dict[str, Tuple[np.ndarray, FrameMetadata]] = {}

    def add_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        metadata: FrameMetadata,
    ):
        """Add a new frame."""
        if camera_id not in self._frames:
            self._frames[camera_id] = []

        self._frames[camera_id].append((frame, metadata))
        self._latest[camera_id] = (frame, metadata)

        # Trim history
        if len(self._frames[camera_id]) > self.max_history:
            self._frames[camera_id] = self._frames[camera_id][-self.max_history:]

    def get_latest(
        self,
        camera_id: str,
    ) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Get latest frame for a camera."""
        return self._latest.get(camera_id)

    def get_frame(
        self,
        camera_id: str,
        frame_id: int,
    ) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Get specific frame by ID."""
        if camera_id not in self._frames:
            return None

        for frame, metadata in self._frames[camera_id]:
            if metadata.frame_id == frame_id:
                return (frame, metadata)

        return None

    def get_cameras(self) -> List[str]:
        """Get list of camera IDs with frames."""
        return list(self._latest.keys())


# =============================================================================
# Placeholder Frame Generator
# =============================================================================

def generate_placeholder_frame(
    camera_id: str,
    width: int = 640,
    height: int = 480,
    message: str = None,
) -> np.ndarray:
    """Generate a placeholder frame when camera not available."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add gradient background
    for y in range(height):
        gray = int(30 + (y / height) * 20)
        frame[y, :] = [gray, gray, gray + 10]

    if CV2_AVAILABLE:
        # Add camera ID text
        cv2.putText(
            frame,
            f"Camera: {camera_id}",
            (width // 2 - 80, height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (150, 150, 150),
            2,
        )

        # Add message or timestamp
        msg = message or datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            frame,
            msg,
            (width // 2 - 50, height // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 100, 100),
            1,
        )

    return frame

"""Stage-specific visualization functions for pipeline outputs.

Each function takes the base frame and stage-specific data,
returning a visualized frame ready for streaming.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ui.components.frame_viewer import (
    Annotation,
    BoundingBox,
    FrameAnnotator,
    FrameMetadata,
)


# Shared annotator instance
_annotator = FrameAnnotator(
    font_scale=0.5,
    line_thickness=2,
    show_confidence=True,
    show_track_id=True,
)

# Color palette for masks (distinct colors for up to 20 objects)
MASK_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
    (0, 128, 255),  # Light blue
    (128, 255, 0),  # Lime
    (255, 0, 128),  # Pink
    (0, 255, 128),  # Spring green
    (128, 128, 255),
    (255, 128, 128),
    (128, 255, 128),
    (255, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (192, 192, 192),
    (64, 64, 64),
]


def visualize_capture(
    frame: np.ndarray,
    camera_id: str = "",
    frame_id: int = 0,
    fps: float = 0.0,
    timestamp: Optional[datetime] = None,
) -> np.ndarray:
    """Visualize capture stage: raw frame with info overlay.

    Args:
        frame: BGR image from camera
        camera_id: Camera identifier
        frame_id: Frame number
        fps: Current frames per second
        timestamp: Capture timestamp

    Returns:
        Annotated frame with info overlay
    """
    if not CV2_AVAILABLE:
        return frame

    result = frame.copy()
    ts = timestamp or datetime.now()

    # Draw info box
    overlay = result.copy()
    cv2.rectangle(overlay, (5, 5), (220, 70), (0, 0, 0), -1)
    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

    # Draw info text
    lines = [
        f"Camera: {camera_id}",
        f"Frame: {frame_id}",
        f"FPS: {fps:.1f}",
        f"Time: {ts.strftime('%H:%M:%S.%f')[:-3]}",
    ]

    y = 20
    for line in lines:
        cv2.putText(result, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 15

    return result


def visualize_segmentation(
    frame: np.ndarray,
    masks: List[Any],
    alpha: float = 0.4,
) -> np.ndarray:
    """Visualize segmentation stage: frame with colored mask overlays.

    Args:
        frame: BGR base image
        masks: List of ObjectMask or similar with .mask attribute
        alpha: Transparency of mask overlay (0-1)

    Returns:
        Frame with colored mask overlays
    """
    if not CV2_AVAILABLE or not masks:
        return frame

    result = frame.copy()

    for i, mask_obj in enumerate(masks):
        # Handle different mask formats
        if hasattr(mask_obj, 'mask'):
            mask = mask_obj.mask
        elif isinstance(mask_obj, np.ndarray):
            mask = mask_obj
        else:
            continue

        if mask is None or mask.shape[:2] != frame.shape[:2]:
            continue

        # Get color for this mask
        color = MASK_COLORS[i % len(MASK_COLORS)]

        # Create colored overlay
        overlay = result.copy()
        overlay[mask > 0] = color
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

        # Draw bounding box if available
        if hasattr(mask_obj, 'bbox') and mask_obj.bbox:
            x1, y1, x2, y2 = mask_obj.bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw confidence if available
            if hasattr(mask_obj, 'confidence'):
                label = f"{mask_obj.confidence:.2f}"
                cv2.putText(result, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw mask count
    cv2.putText(result, f"Objects: {len(masks)}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result


def visualize_tracking(
    frame: np.ndarray,
    tracks: List[Any],
    show_trails: bool = True,
    trail_length: int = 30,
) -> np.ndarray:
    """Visualize tracking stage: bounding boxes with track IDs and trails.

    Args:
        frame: BGR base image
        tracks: List of track objects with bbox, track_id, etc.
        show_trails: Whether to draw motion trails
        trail_length: Maximum trail points to show

    Returns:
        Frame with tracking visualization
    """
    if not CV2_AVAILABLE:
        return frame

    result = frame.copy()
    boxes = []

    for track in tracks:
        # Extract track info
        if hasattr(track, 'bbox'):
            bbox = track.bbox
        elif hasattr(track, 'bounding_box_2d'):
            bbox = track.bounding_box_2d
        elif isinstance(track, dict) and 'bbox' in track:
            bbox = track['bbox']
        else:
            continue

        track_id = getattr(track, 'track_id', getattr(track, 'id', None))
        label = getattr(track, 'primary_class', getattr(track, 'label', 'object'))
        confidence = getattr(track, 'confidence', 1.0)

        # Get consistent color for track ID
        color = MASK_COLORS[hash(str(track_id)) % len(MASK_COLORS)] if track_id else (0, 255, 0)

        boxes.append(BoundingBox(
            x1=int(bbox[0]),
            y1=int(bbox[1]),
            x2=int(bbox[2]),
            y2=int(bbox[3]),
            label=label,
            confidence=confidence,
            color=color,
            track_id=track_id,
        ))

        # Draw trail if available
        if show_trails and hasattr(track, 'trajectory'):
            trail = track.trajectory[-trail_length:]
            if len(trail) > 1:
                points = [(int(p[0]), int(p[1])) for p in trail if len(p) >= 2]
                for j in range(1, len(points)):
                    thickness = max(1, int((j / len(points)) * 3))
                    cv2.line(result, points[j-1], points[j], color, thickness)

    # Use annotator for boxes
    annotation = Annotation(boxes=boxes)
    result = _annotator.annotate(result, annotation)

    # Draw track count
    cv2.putText(result, f"Tracks: {len(tracks)}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result


def visualize_reid(
    frame: np.ndarray,
    objects: List[Any],
    matched_ids: Optional[set] = None,
) -> np.ndarray:
    """Visualize ReID stage: highlight matched vs new objects.

    Args:
        frame: BGR base image
        objects: List of ObjectSchema or similar
        matched_ids: Set of IDs that were matched to existing objects

    Returns:
        Frame with ReID visualization
    """
    if not CV2_AVAILABLE:
        return frame

    result = frame.copy()
    matched_ids = matched_ids or set()
    matched_count = 0
    new_count = 0

    for obj in objects:
        if hasattr(obj, 'bounding_box_2d'):
            bbox = obj.bounding_box_2d
        elif hasattr(obj, 'bbox'):
            bbox = obj.bbox
        else:
            continue

        obj_id = getattr(obj, 'id', None)
        is_matched = obj_id in matched_ids

        # Green for matched, yellow for new
        color = (0, 255, 0) if is_matched else (0, 255, 255)
        label = "MATCHED" if is_matched else "NEW"

        if is_matched:
            matched_count += 1
        else:
            new_count += 1

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw stats
    cv2.putText(result, f"Matched: {matched_count}  New: {new_count}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result


def visualize_classification(
    frame: np.ndarray,
    objects: List[Any],
) -> np.ndarray:
    """Visualize classification stage: labels and attributes.

    Args:
        frame: BGR base image
        objects: List of classified ObjectSchema

    Returns:
        Frame with classification labels
    """
    if not CV2_AVAILABLE:
        return frame

    result = frame.copy()
    boxes = []

    for obj in objects:
        if hasattr(obj, 'bounding_box_2d'):
            bbox = obj.bounding_box_2d
        elif hasattr(obj, 'bbox'):
            bbox = obj.bbox
        else:
            continue

        primary_class = getattr(obj, 'primary_class', 'unknown')
        confidence = getattr(obj, 'confidence', 0.0)
        subclass = getattr(obj, 'subclass', None)

        # Build label
        label = primary_class
        if subclass:
            label = f"{primary_class}/{subclass}"

        boxes.append(BoundingBox(
            x1=int(bbox[0]),
            y1=int(bbox[1]),
            x2=int(bbox[2]),
            y2=int(bbox[3]),
            label=label,
            confidence=confidence,
        ))

    annotation = Annotation(boxes=boxes)
    result = _annotator.annotate(result, annotation)

    return result


def visualize_depth(
    depth: np.ndarray,
    min_depth: float = 0.3,
    max_depth: float = 10.0,
) -> np.ndarray:
    """Visualize depth image with colormap.

    Args:
        depth: Depth image in meters
        min_depth: Minimum depth for normalization
        max_depth: Maximum depth for normalization

    Returns:
        Colorized depth visualization
    """
    if not CV2_AVAILABLE:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    # Clip and normalize
    depth_clipped = np.clip(depth, min_depth, max_depth)
    depth_norm = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    # Handle invalid depth (zeros)
    depth_norm[depth < 0.01] = 0

    # Apply colormap
    colorized = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    return colorized


def visualize_placeholder(
    width: int = 640,
    height: int = 480,
    message: str = "No data",
    stage_id: str = "",
) -> np.ndarray:
    """Generate placeholder frame when stage has no data.

    Args:
        width: Frame width
        height: Frame height
        message: Message to display
        stage_id: Stage identifier

    Returns:
        Placeholder frame
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add gradient background
    for y in range(height):
        gray = int(20 + (y / height) * 30)
        frame[y, :] = [gray, gray, gray + 5]

    if CV2_AVAILABLE:
        # Draw stage name
        cv2.putText(frame, stage_id.upper(), (width // 2 - 50, height // 2 - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        # Draw message
        cv2.putText(frame, message, (width // 2 - 40, height // 2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    return frame

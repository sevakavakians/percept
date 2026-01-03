"""ByteTrack integration for PERCEPT.

Provides frame-to-frame object tracking using the ByteTrack algorithm.
Falls back to simple IoU-based tracking when ByteTrack is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData


def is_supervision_available() -> bool:
    """Check if supervision package is available."""
    try:
        import supervision as sv
        return hasattr(sv, 'ByteTrack')
    except ImportError:
        return False


SUPERVISION_AVAILABLE = is_supervision_available()


class TrackState(Enum):
    """State of a track."""
    TENTATIVE = "tentative"  # New track, not yet confirmed
    CONFIRMED = "confirmed"  # Confirmed active track
    LOST = "lost"  # Temporarily lost
    DELETED = "deleted"  # Track to be removed


@dataclass
class ByteTrackConfig:
    """Configuration for ByteTrack."""

    # Track activation
    track_activation_threshold: float = 0.5
    minimum_matching_threshold: float = 0.8

    # Track lifecycle
    lost_track_buffer: int = 30  # Frames before lost track deleted
    minimum_consecutive_frames: int = 3  # Frames to confirm track

    # Detection filtering
    frame_rate: int = 30


@dataclass
class Track:
    """A tracked object."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    state: TrackState = TrackState.TENTATIVE
    age: int = 0  # Frames since creation
    hits: int = 0  # Number of detections associated
    time_since_update: int = 0  # Frames since last detection
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy)
    features: Optional[np.ndarray] = None  # Optional embedding

    def get_centroid(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def predict_next_bbox(self) -> Tuple[float, float, float, float]:
        """Predict next bounding box using velocity."""
        if self.velocity is None:
            return self.bbox

        x1, y1, x2, y2 = self.bbox
        vx, vy = self.velocity
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)


class SimpleIoUTracker:
    """Simple IoU-based tracker as fallback.

    Tracks objects by matching detections to existing tracks
    using Intersection over Union (IoU).
    """

    def __init__(self, config: Optional[ByteTrackConfig] = None):
        """Initialize tracker.

        Args:
            config: Configuration options
        """
        self.config = config or ByteTrackConfig()
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._frame_count = 0

    def update(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Track]:
        """Update tracks with new detections.

        Args:
            detections: List of detection dicts with bbox, confidence, etc.

        Returns:
            List of active tracks
        """
        self._frame_count += 1

        # Extract detection bboxes
        det_bboxes = []
        det_infos = []
        for det in detections:
            bbox = det.get("bbox", (0, 0, 0, 0))
            conf = det.get("confidence", 1.0)
            if conf >= self.config.track_activation_threshold:
                det_bboxes.append(bbox)
                det_infos.append(det)

        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(
            det_bboxes, list(self._tracks.keys())
        )

        # Update matched tracks
        for det_idx, track_id in matched:
            track = self._tracks[track_id]
            det_info = det_infos[det_idx]

            # Update velocity
            old_centroid = track.get_centroid()
            new_bbox = det_bboxes[det_idx]
            new_centroid = ((new_bbox[0] + new_bbox[2]) / 2, (new_bbox[1] + new_bbox[3]) / 2)
            track.velocity = (new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1])

            # Update track
            track.bbox = new_bbox
            track.confidence = det_info.get("confidence", 1.0)
            track.class_id = det_info.get("class_id")
            track.class_name = det_info.get("class_name")
            track.hits += 1
            track.time_since_update = 0
            track.age += 1

            # Confirm track if enough hits
            if track.state == TrackState.TENTATIVE:
                if track.hits >= self.config.minimum_consecutive_frames:
                    track.state = TrackState.CONFIRMED
            elif track.state == TrackState.LOST:
                track.state = TrackState.CONFIRMED

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det_info = det_infos[det_idx]
            track = Track(
                track_id=self._next_id,
                bbox=det_bboxes[det_idx],
                confidence=det_info.get("confidence", 1.0),
                class_id=det_info.get("class_id"),
                class_name=det_info.get("class_name"),
                state=TrackState.TENTATIVE,
                hits=1,
            )
            self._tracks[self._next_id] = track
            self._next_id += 1

        # Handle unmatched tracks
        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.time_since_update += 1
            track.age += 1

            if track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST

            # Delete old lost tracks
            if track.time_since_update > self.config.lost_track_buffer:
                track.state = TrackState.DELETED

        # Remove deleted tracks
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if t.state != TrackState.DELETED
        }

        # Return active tracks
        return [t for t in self._tracks.values() if t.state == TrackState.CONFIRMED]

    def _match_detections(
        self,
        det_bboxes: List[Tuple[float, float, float, float]],
        track_ids: List[int],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using IoU.

        Returns:
            (matched_pairs, unmatched_det_indices, unmatched_track_ids)
        """
        if not det_bboxes or not track_ids:
            return [], list(range(len(det_bboxes))), list(track_ids)

        # Compute IoU matrix
        n_dets = len(det_bboxes)
        n_tracks = len(track_ids)
        iou_matrix = np.zeros((n_dets, n_tracks))

        for i, det_bbox in enumerate(det_bboxes):
            for j, track_id in enumerate(track_ids):
                track = self._tracks[track_id]
                # Use predicted bbox for lost tracks
                track_bbox = track.predict_next_bbox() if track.state == TrackState.LOST else track.bbox
                iou_matrix[i, j] = self._compute_iou(det_bbox, track_bbox)

        # Greedy matching
        matched = []
        unmatched_dets = list(range(n_dets))
        unmatched_tracks = list(track_ids)

        while True:
            if iou_matrix.size == 0:
                break

            max_iou = iou_matrix.max()
            if max_iou < self.config.minimum_matching_threshold:
                break

            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            det_idx, track_idx = max_idx

            track_id = track_ids[track_idx]
            matched.append((det_idx, track_id))

            if det_idx in unmatched_dets:
                unmatched_dets.remove(det_idx)
            if track_id in unmatched_tracks:
                unmatched_tracks.remove(track_id)

            # Zero out matched row and column
            iou_matrix[det_idx, :] = 0
            iou_matrix[:, track_idx] = 0

        return matched, unmatched_dets, unmatched_tracks

    def _compute_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-6)

    def reset(self):
        """Reset tracker state."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self._tracks.get(track_id)


class ByteTrackWrapper:
    """Wrapper for ByteTrack with fallback to SimpleIoUTracker.

    Uses supervision's ByteTrack when available, otherwise falls back
    to simple IoU-based tracking.
    """

    def __init__(self, config: Optional[ByteTrackConfig] = None):
        """Initialize tracker.

        Args:
            config: Configuration options
        """
        self.config = config or ByteTrackConfig()
        self._use_bytetrack = SUPERVISION_AVAILABLE
        self._tracker = None
        self._fallback = SimpleIoUTracker(config)
        self._track_history: Dict[int, List[Track]] = {}

        if self._use_bytetrack:
            self._init_bytetrack()

    def _init_bytetrack(self):
        """Initialize ByteTrack from supervision."""
        try:
            import supervision as sv
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.config.track_activation_threshold,
                lost_track_buffer=self.config.lost_track_buffer,
                minimum_matching_threshold=self.config.minimum_matching_threshold,
                minimum_consecutive_frames=self.config.minimum_consecutive_frames,
                frame_rate=self.config.frame_rate,
            )
        except Exception:
            self._use_bytetrack = False

    def update(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Track]:
        """Update tracking with new detections.

        Args:
            detections: List of detection dicts with bbox, confidence, etc.

        Returns:
            List of active Track instances
        """
        if not self._use_bytetrack or self._tracker is None:
            return self._fallback.update(detections)

        import supervision as sv

        # Convert detections to supervision format
        if not detections:
            # Create empty detections
            sv_detections = sv.Detections.empty()
        else:
            bboxes = []
            confidences = []
            class_ids = []

            for det in detections:
                bbox = det.get("bbox", (0, 0, 0, 0))
                bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                confidences.append(det.get("confidence", 1.0))
                class_ids.append(det.get("class_id", 0))

            sv_detections = sv.Detections(
                xyxy=np.array(bboxes, dtype=np.float32),
                confidence=np.array(confidences, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int),
            )

        # Run ByteTrack
        tracked = self._tracker.update_with_detections(sv_detections)

        # Convert to Track instances
        tracks = []
        if tracked.tracker_id is not None:
            for i in range(len(tracked)):
                bbox = tuple(tracked.xyxy[i])
                track_id = int(tracked.tracker_id[i])
                confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                class_id = int(tracked.class_id[i]) if tracked.class_id is not None else None

                # Get class name from original detections
                class_name = None
                for det in detections:
                    det_bbox = det.get("bbox")
                    if det_bbox and self._bbox_match(bbox, det_bbox):
                        class_name = det.get("class_name")
                        break

                track = Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    state=TrackState.CONFIRMED,
                )
                tracks.append(track)

                # Store history
                if track_id not in self._track_history:
                    self._track_history[track_id] = []
                self._track_history[track_id].append(track)

        return tracks

    def _bbox_match(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
        threshold: float = 0.5,
    ) -> bool:
        """Check if two bboxes match (high IoU)."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return (intersection / max(union, 1e-6)) > threshold

    def reset(self):
        """Reset tracker state."""
        if self._use_bytetrack and self._tracker is not None:
            self._tracker.reset()
        self._fallback.reset()
        self._track_history.clear()

    def get_track_history(self, track_id: int) -> List[Track]:
        """Get historical tracks for a track ID."""
        return self._track_history.get(track_id, [])

    def is_using_bytetrack(self) -> bool:
        """Check if using real ByteTrack or fallback."""
        return self._use_bytetrack


class TrackingModule(PipelineModule):
    """Pipeline module for frame-to-frame tracking.

    Wraps ByteTrack for integration into PERCEPT pipelines.
    """

    def __init__(self, config: Optional[ByteTrackConfig] = None):
        """Initialize module.

        Args:
            config: Configuration options
        """
        self.config = config or ByteTrackConfig()
        self.tracker = ByteTrackWrapper(self.config)

    @property
    def name(self) -> str:
        return "tracking"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="detections",
            required_fields=["detections"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="tracks",
            required_fields=["tracks"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Track detections across frames.

        Args:
            data: PipelineData with detections

        Returns:
            PipelineData with tracks
        """
        detections = data.get("detections", [])

        # Convert various detection formats to dict
        det_list = []
        for det in detections:
            if isinstance(det, dict):
                det_list.append(det)
            elif hasattr(det, 'bbox'):
                det_list.append({
                    "bbox": det.bbox,
                    "confidence": getattr(det, 'confidence', 1.0),
                    "class_id": getattr(det, 'class_id', None),
                    "class_name": getattr(det, 'class_name', None),
                })

        tracks = self.tracker.update(det_list)

        result = data.copy()
        result.tracks = tracks
        return result


def track_detections(
    detections: List[Dict[str, Any]],
    tracker: Optional[ByteTrackWrapper] = None,
) -> List[Track]:
    """Convenience function for tracking detections.

    Args:
        detections: List of detection dicts
        tracker: Optional pre-existing tracker (creates new if None)

    Returns:
        List of Track instances
    """
    if tracker is None:
        tracker = ByteTrackWrapper()
    return tracker.update(detections)

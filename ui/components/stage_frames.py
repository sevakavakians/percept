"""Stage frame buffer for pipeline visualization.

Stores the latest visualized frame for each pipeline stage,
allowing MJPEG streams to serve the most recent output.
"""

import threading
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


class StageFrameBuffer:
    """Thread-safe buffer for stage output frames.

    Each pipeline stage stores its latest visualized frame here.
    Stream endpoints read from this buffer to serve MJPEG video.
    """

    def __init__(self):
        self._frames: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def update(
        self,
        stage_id: str,
        frame: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the frame for a stage.

        Args:
            stage_id: Pipeline stage identifier (e.g., "capture", "segment")
            frame: BGR image as numpy array
            metadata: Optional stage-specific metadata
        """
        with self._lock:
            self._frames[stage_id] = frame.copy()
            self._timestamps[stage_id] = datetime.now()
            if metadata:
                self._metadata[stage_id] = metadata

    def get(self, stage_id: str) -> Optional[np.ndarray]:
        """Get the latest frame for a stage.

        Args:
            stage_id: Pipeline stage identifier

        Returns:
            BGR image as numpy array, or None if no frame available
        """
        with self._lock:
            frame = self._frames.get(stage_id)
            if frame is not None:
                return frame.copy()
            return None

    def get_with_metadata(
        self, stage_id: str
    ) -> tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[datetime]]:
        """Get frame with its metadata and timestamp.

        Args:
            stage_id: Pipeline stage identifier

        Returns:
            Tuple of (frame, metadata, timestamp), all None if not available
        """
        with self._lock:
            frame = self._frames.get(stage_id)
            if frame is not None:
                return (
                    frame.copy(),
                    self._metadata.get(stage_id, {}),
                    self._timestamps.get(stage_id),
                )
            return None, None, None

    def get_all_stage_ids(self) -> list[str]:
        """Get list of all stages with frames."""
        with self._lock:
            return list(self._frames.keys())

    def clear(self, stage_id: Optional[str] = None) -> None:
        """Clear frames.

        Args:
            stage_id: Specific stage to clear, or None to clear all
        """
        with self._lock:
            if stage_id:
                self._frames.pop(stage_id, None)
                self._metadata.pop(stage_id, None)
                self._timestamps.pop(stage_id, None)
            else:
                self._frames.clear()
                self._metadata.clear()
                self._timestamps.clear()

    def get_age_ms(self, stage_id: str) -> Optional[float]:
        """Get age of frame in milliseconds.

        Args:
            stage_id: Pipeline stage identifier

        Returns:
            Age in milliseconds, or None if no frame
        """
        with self._lock:
            ts = self._timestamps.get(stage_id)
            if ts:
                return (datetime.now() - ts).total_seconds() * 1000
            return None


# Global instance for the application
stage_buffer = StageFrameBuffer()

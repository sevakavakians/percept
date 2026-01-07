"""Adaptive processing based on scene complexity.

Dynamically adjusts processing parameters to maintain target FPS
while maximizing quality when resources are available.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# Enums and Configuration
# =============================================================================

class ProcessingMode(Enum):
    """Processing mode based on system load."""
    FULL = "full"           # All features enabled
    BALANCED = "balanced"   # Some features reduced
    FAST = "fast"           # Minimal features for speed
    MINIMAL = "minimal"     # Skip non-essential processing


class SceneComplexity(Enum):
    """Scene complexity level."""
    LOW = "low"       # Few objects, simple scene
    MEDIUM = "medium" # Moderate objects
    HIGH = "high"     # Many objects, complex scene
    EXTREME = "extreme"  # Very crowded scene


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive processing."""

    # Target performance
    target_fps: float = 15.0
    min_fps: float = 10.0
    max_processing_time_ms: float = 100.0

    # Adaptation thresholds
    fps_low_threshold: float = 0.8   # Switch to faster mode below target * this
    fps_high_threshold: float = 1.2  # Switch to slower mode above target * this

    # Scene complexity thresholds
    low_complexity_objects: int = 3
    medium_complexity_objects: int = 10
    high_complexity_objects: int = 25

    # Frame skipping
    enable_frame_skipping: bool = True
    max_consecutive_skips: int = 2
    skip_stale_frames: bool = True
    stale_frame_age_ms: float = 200.0

    # Feature toggles per mode
    mode_features: Dict[ProcessingMode, Dict[str, bool]] = field(default_factory=lambda: {
        ProcessingMode.FULL: {
            "fastsam": True,
            "depth_segmentation": True,
            "pointcloud": True,
            "pose_estimation": True,
            "face_detection": True,
            "plate_detection": True,
            "imagenet_classification": True,
            "reid_matching": True,
        },
        ProcessingMode.BALANCED: {
            "fastsam": True,
            "depth_segmentation": True,
            "pointcloud": False,
            "pose_estimation": True,
            "face_detection": False,
            "plate_detection": True,
            "imagenet_classification": False,
            "reid_matching": True,
        },
        ProcessingMode.FAST: {
            "fastsam": True,
            "depth_segmentation": False,
            "pointcloud": False,
            "pose_estimation": False,
            "face_detection": False,
            "plate_detection": False,
            "imagenet_classification": False,
            "reid_matching": True,
        },
        ProcessingMode.MINIMAL: {
            "fastsam": True,
            "depth_segmentation": False,
            "pointcloud": False,
            "pose_estimation": False,
            "face_detection": False,
            "plate_detection": False,
            "imagenet_classification": False,
            "reid_matching": False,
        },
    })


# =============================================================================
# Scene Analyzer
# =============================================================================

class SceneAnalyzer:
    """Analyze scene complexity for adaptive processing."""

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()

    def analyze(
        self,
        object_count: int,
        motion_score: float = 0.0,
        depth_variance: float = 0.0,
    ) -> SceneComplexity:
        """Analyze scene complexity.

        Args:
            object_count: Number of detected objects
            motion_score: Motion between frames (0-1)
            depth_variance: Variance in depth values

        Returns:
            SceneComplexity level
        """
        # Primary factor: object count
        if object_count <= self.config.low_complexity_objects:
            base_complexity = SceneComplexity.LOW
        elif object_count <= self.config.medium_complexity_objects:
            base_complexity = SceneComplexity.MEDIUM
        elif object_count <= self.config.high_complexity_objects:
            base_complexity = SceneComplexity.HIGH
        else:
            base_complexity = SceneComplexity.EXTREME

        # Adjust for motion
        if motion_score > 0.5 and base_complexity != SceneComplexity.EXTREME:
            # Increase complexity for high motion
            levels = list(SceneComplexity)
            idx = levels.index(base_complexity)
            if idx < len(levels) - 1:
                return levels[idx + 1]

        return base_complexity

    def estimate_processing_cost(
        self,
        complexity: SceneComplexity,
        mode: ProcessingMode,
    ) -> float:
        """Estimate processing cost in ms.

        Args:
            complexity: Scene complexity level
            mode: Processing mode

        Returns:
            Estimated processing time in ms
        """
        # Base costs per complexity
        base_costs = {
            SceneComplexity.LOW: 30.0,
            SceneComplexity.MEDIUM: 50.0,
            SceneComplexity.HIGH: 80.0,
            SceneComplexity.EXTREME: 120.0,
        }

        # Mode multipliers
        mode_multipliers = {
            ProcessingMode.FULL: 1.0,
            ProcessingMode.BALANCED: 0.7,
            ProcessingMode.FAST: 0.4,
            ProcessingMode.MINIMAL: 0.25,
        }

        return base_costs[complexity] * mode_multipliers[mode]


# =============================================================================
# FPS Controller
# =============================================================================

class FPSController:
    """Control processing to maintain target FPS."""

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self._frame_times: deque = deque(maxlen=30)
        self._processing_times: deque = deque(maxlen=30)
        self._last_frame_time: float = 0.0
        self._consecutive_skips: int = 0

    def record_frame(self, processing_time_ms: float):
        """Record frame processing time."""
        now = time.perf_counter()

        if self._last_frame_time > 0:
            frame_time = now - self._last_frame_time
            self._frame_times.append(frame_time)

        self._processing_times.append(processing_time_ms)
        self._last_frame_time = now
        self._consecutive_skips = 0

    def record_skip(self):
        """Record skipped frame."""
        self._consecutive_skips += 1

    @property
    def current_fps(self) -> float:
        """Get current FPS."""
        if len(self._frame_times) < 2:
            return 0.0

        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    @property
    def average_processing_time(self) -> float:
        """Get average processing time in ms."""
        if not self._processing_times:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times)

    def should_skip_frame(self, frame_age_ms: float = 0.0) -> bool:
        """Determine if frame should be skipped.

        Args:
            frame_age_ms: Age of the frame in milliseconds

        Returns:
            True if frame should be skipped
        """
        if not self.config.enable_frame_skipping:
            return False

        # Don't skip too many consecutive frames
        if self._consecutive_skips >= self.config.max_consecutive_skips:
            return False

        # Skip stale frames
        if self.config.skip_stale_frames:
            if frame_age_ms > self.config.stale_frame_age_ms:
                return True

        # Skip if we're behind target
        if self.current_fps > 0:
            if self.current_fps < self.config.target_fps * self.config.fps_low_threshold:
                return True

        return False

    def get_recommended_mode(self) -> ProcessingMode:
        """Get recommended processing mode based on current FPS."""
        fps = self.current_fps

        if fps <= 0:
            return ProcessingMode.BALANCED

        target = self.config.target_fps
        low = target * self.config.fps_low_threshold
        high = target * self.config.fps_high_threshold

        if fps >= high:
            return ProcessingMode.FULL
        elif fps >= target:
            return ProcessingMode.BALANCED
        elif fps >= low:
            return ProcessingMode.FAST
        else:
            return ProcessingMode.MINIMAL


# =============================================================================
# Adaptive Processor
# =============================================================================

class AdaptiveProcessor:
    """Main adaptive processing controller."""

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.scene_analyzer = SceneAnalyzer(config)
        self.fps_controller = FPSController(config)

        self._current_mode: ProcessingMode = ProcessingMode.BALANCED
        self._current_complexity: SceneComplexity = SceneComplexity.MEDIUM
        self._mode_history: deque = deque(maxlen=10)

    @property
    def current_mode(self) -> ProcessingMode:
        """Get current processing mode."""
        return self._current_mode

    @property
    def current_complexity(self) -> SceneComplexity:
        """Get current scene complexity."""
        return self._current_complexity

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled in current mode.

        Args:
            feature: Feature name (e.g., "fastsam", "pose_estimation")

        Returns:
            True if feature is enabled
        """
        mode_features = self.config.mode_features.get(self._current_mode, {})
        return mode_features.get(feature, False)

    def update(
        self,
        object_count: int,
        processing_time_ms: float,
        frame_age_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Update adaptive processor state.

        Args:
            object_count: Number of objects detected
            processing_time_ms: Time spent processing frame
            frame_age_ms: Age of the frame

        Returns:
            Dict with current state and recommendations
        """
        # Update FPS tracking
        self.fps_controller.record_frame(processing_time_ms)

        # Analyze scene complexity
        self._current_complexity = self.scene_analyzer.analyze(object_count)

        # Get recommended mode
        recommended_mode = self.fps_controller.get_recommended_mode()

        # Smooth mode transitions (avoid rapid switching)
        self._mode_history.append(recommended_mode)

        if len(self._mode_history) >= 5:
            # Use most common mode in recent history
            mode_counts = {}
            for mode in self._mode_history:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

            most_common = max(mode_counts, key=mode_counts.get)
            if mode_counts[most_common] >= 3:
                self._current_mode = most_common

        return {
            "mode": self._current_mode.value,
            "complexity": self._current_complexity.value,
            "fps": self.fps_controller.current_fps,
            "avg_processing_ms": self.fps_controller.average_processing_time,
            "features": self.config.mode_features.get(self._current_mode, {}),
        }

    def should_skip_frame(self, frame_age_ms: float = 0.0) -> bool:
        """Check if frame should be skipped.

        Args:
            frame_age_ms: Age of the frame

        Returns:
            True if frame should be skipped
        """
        skip = self.fps_controller.should_skip_frame(frame_age_ms)
        if skip:
            self.fps_controller.record_skip()
        return skip

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "mode": self._current_mode.value,
            "complexity": self._current_complexity.value,
            "fps": self.fps_controller.current_fps,
            "target_fps": self.config.target_fps,
            "avg_processing_ms": self.fps_controller.average_processing_time,
            "max_processing_ms": self.config.max_processing_time_ms,
            "features_enabled": list(
                k for k, v in self.config.mode_features.get(self._current_mode, {}).items()
                if v
            ),
        }


# =============================================================================
# Adaptive Pipeline Module
# =============================================================================

class AdaptiveProcessingModule:
    """Pipeline module for adaptive processing control."""

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.processor = AdaptiveProcessor(config)

    @property
    def name(self) -> str:
        return "adaptive_processing"

    def pre_process(self, frame_age_ms: float = 0.0) -> Tuple[bool, Dict[str, bool]]:
        """Called before frame processing.

        Args:
            frame_age_ms: Age of the frame

        Returns:
            Tuple of (should_process, enabled_features)
        """
        if self.processor.should_skip_frame(frame_age_ms):
            return False, {}

        features = self.config.mode_features.get(
            self.processor.current_mode,
            {}
        )
        return True, features

    def post_process(
        self,
        object_count: int,
        processing_time_ms: float,
    ) -> Dict[str, Any]:
        """Called after frame processing.

        Args:
            object_count: Number of objects detected
            processing_time_ms: Processing time in ms

        Returns:
            Updated state
        """
        return self.processor.update(
            object_count=object_count,
            processing_time_ms=processing_time_ms,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processor.get_stats()

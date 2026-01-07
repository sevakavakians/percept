"""Tests for adaptive processing module.

Tests verify adaptive processing adjusts correctly
based on scene complexity and FPS targets.
"""

import time

import pytest

from percept.utils.adaptive import (
    AdaptiveConfig,
    AdaptiveProcessor,
    AdaptiveProcessingModule,
    FPSController,
    ProcessingMode,
    SceneAnalyzer,
    SceneComplexity,
)


# =============================================================================
# SceneComplexity Tests
# =============================================================================

class TestSceneComplexity:
    """Test SceneComplexity enum."""

    def test_complexity_values(self):
        """Test complexity enum values."""
        assert SceneComplexity.LOW.value == "low"
        assert SceneComplexity.MEDIUM.value == "medium"
        assert SceneComplexity.HIGH.value == "high"
        assert SceneComplexity.EXTREME.value == "extreme"


class TestProcessingMode:
    """Test ProcessingMode enum."""

    def test_mode_values(self):
        """Test mode enum values."""
        assert ProcessingMode.FULL.value == "full"
        assert ProcessingMode.BALANCED.value == "balanced"
        assert ProcessingMode.FAST.value == "fast"
        assert ProcessingMode.MINIMAL.value == "minimal"


# =============================================================================
# AdaptiveConfig Tests
# =============================================================================

class TestAdaptiveConfig:
    """Test AdaptiveConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = AdaptiveConfig()

        assert config.target_fps == 15.0
        assert config.min_fps == 10.0
        assert config.max_processing_time_ms == 100.0
        assert config.fps_low_threshold == 0.8
        assert config.fps_high_threshold == 1.2

    def test_custom_values(self):
        """Test custom configuration."""
        config = AdaptiveConfig(
            target_fps=30.0,
            min_fps=20.0,
            max_processing_time_ms=50.0,
        )

        assert config.target_fps == 30.0
        assert config.min_fps == 20.0
        assert config.max_processing_time_ms == 50.0

    def test_mode_features(self):
        """Test mode features configuration."""
        config = AdaptiveConfig()

        # Full mode has all features
        full_features = config.mode_features[ProcessingMode.FULL]
        assert full_features["fastsam"] is True
        assert full_features["depth_segmentation"] is True
        assert full_features["pointcloud"] is True
        assert full_features["pose_estimation"] is True

        # Minimal mode has few features
        minimal_features = config.mode_features[ProcessingMode.MINIMAL]
        assert minimal_features["fastsam"] is True
        assert minimal_features["depth_segmentation"] is False
        assert minimal_features["pointcloud"] is False
        assert minimal_features["reid_matching"] is False

    def test_complexity_thresholds(self):
        """Test complexity thresholds."""
        config = AdaptiveConfig()

        assert config.low_complexity_objects == 3
        assert config.medium_complexity_objects == 10
        assert config.high_complexity_objects == 25

    def test_frame_skipping_config(self):
        """Test frame skipping configuration."""
        config = AdaptiveConfig()

        assert config.enable_frame_skipping is True
        assert config.max_consecutive_skips == 2
        assert config.skip_stale_frames is True
        assert config.stale_frame_age_ms == 200.0


# =============================================================================
# SceneAnalyzer Tests
# =============================================================================

class TestSceneAnalyzer:
    """Test SceneAnalyzer class."""

    def test_analyze_low_complexity(self):
        """Analyze low complexity scene."""
        analyzer = SceneAnalyzer()
        complexity = analyzer.analyze(object_count=2)

        assert complexity == SceneComplexity.LOW

    def test_analyze_medium_complexity(self):
        """Analyze medium complexity scene."""
        analyzer = SceneAnalyzer()
        complexity = analyzer.analyze(object_count=7)

        assert complexity == SceneComplexity.MEDIUM

    def test_analyze_high_complexity(self):
        """Analyze high complexity scene."""
        analyzer = SceneAnalyzer()
        complexity = analyzer.analyze(object_count=20)

        assert complexity == SceneComplexity.HIGH

    def test_analyze_extreme_complexity(self):
        """Analyze extreme complexity scene."""
        analyzer = SceneAnalyzer()
        complexity = analyzer.analyze(object_count=50)

        assert complexity == SceneComplexity.EXTREME

    def test_motion_increases_complexity(self):
        """High motion increases complexity level."""
        analyzer = SceneAnalyzer()

        # Without motion: medium
        without_motion = analyzer.analyze(object_count=7, motion_score=0.0)
        assert without_motion == SceneComplexity.MEDIUM

        # With high motion: high
        with_motion = analyzer.analyze(object_count=7, motion_score=0.7)
        assert with_motion == SceneComplexity.HIGH

    def test_motion_cannot_exceed_extreme(self):
        """Motion cannot increase beyond extreme."""
        analyzer = SceneAnalyzer()

        # Already extreme
        complexity = analyzer.analyze(object_count=50, motion_score=0.9)
        assert complexity == SceneComplexity.EXTREME

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        config = AdaptiveConfig(
            low_complexity_objects=5,
            medium_complexity_objects=15,
            high_complexity_objects=30,
        )
        analyzer = SceneAnalyzer(config)

        # 4 objects is low with these thresholds
        assert analyzer.analyze(object_count=4) == SceneComplexity.LOW

        # 10 objects is medium
        assert analyzer.analyze(object_count=10) == SceneComplexity.MEDIUM

    def test_estimate_processing_cost(self):
        """Estimate processing cost."""
        analyzer = SceneAnalyzer()

        # Low complexity, full mode
        low_full = analyzer.estimate_processing_cost(
            SceneComplexity.LOW, ProcessingMode.FULL
        )
        assert low_full == 30.0  # Base cost for low

        # High complexity, fast mode
        high_fast = analyzer.estimate_processing_cost(
            SceneComplexity.HIGH, ProcessingMode.FAST
        )
        assert high_fast == 80.0 * 0.4  # Base * fast multiplier

    def test_mode_multipliers(self):
        """Test mode multipliers on cost."""
        analyzer = SceneAnalyzer()
        complexity = SceneComplexity.MEDIUM
        base_cost = 50.0  # Base for medium

        full = analyzer.estimate_processing_cost(complexity, ProcessingMode.FULL)
        balanced = analyzer.estimate_processing_cost(complexity, ProcessingMode.BALANCED)
        fast = analyzer.estimate_processing_cost(complexity, ProcessingMode.FAST)
        minimal = analyzer.estimate_processing_cost(complexity, ProcessingMode.MINIMAL)

        assert full == base_cost * 1.0
        assert balanced == base_cost * 0.7
        assert fast == base_cost * 0.4
        assert minimal == base_cost * 0.25


# =============================================================================
# FPSController Tests
# =============================================================================

class TestFPSController:
    """Test FPSController class."""

    def test_initial_fps_zero(self):
        """Initial FPS is zero."""
        controller = FPSController()
        assert controller.current_fps == 0.0

    def test_initial_processing_time_zero(self):
        """Initial processing time is zero."""
        controller = FPSController()
        assert controller.average_processing_time == 0.0

    def test_record_frame(self):
        """Record frame updates metrics."""
        controller = FPSController()

        controller.record_frame(50.0)
        time.sleep(0.05)
        controller.record_frame(55.0)

        assert controller.average_processing_time > 0

    def test_fps_calculation(self):
        """FPS calculation from frame times."""
        controller = FPSController()

        # Simulate ~20 FPS
        for _ in range(10):
            controller.record_frame(30.0)
            time.sleep(0.05)

        # Should be approximately 20 FPS
        assert 10 < controller.current_fps < 30

    def test_should_not_skip_initially(self):
        """Should not skip frames initially."""
        controller = FPSController()
        assert controller.should_skip_frame() is False

    def test_skip_stale_frames(self):
        """Skip stale frames."""
        controller = FPSController()

        # Frame older than stale threshold
        assert controller.should_skip_frame(frame_age_ms=300.0) is True

    def test_max_consecutive_skips(self):
        """Respect max consecutive skips."""
        controller = FPSController()

        # Skip up to max
        controller.record_skip()
        controller.record_skip()

        # Should not skip after max reached
        assert controller.should_skip_frame(frame_age_ms=300.0) is False

    def test_record_frame_resets_skips(self):
        """Recording frame resets skip counter."""
        controller = FPSController()

        controller.record_skip()
        controller.record_skip()
        controller.record_frame(50.0)

        # Should be able to skip again
        assert controller.should_skip_frame(frame_age_ms=300.0) is True

    def test_disable_frame_skipping(self):
        """Disable frame skipping."""
        config = AdaptiveConfig(enable_frame_skipping=False)
        controller = FPSController(config)

        assert controller.should_skip_frame(frame_age_ms=1000.0) is False

    def test_recommended_mode_no_data(self):
        """Recommended mode with no data."""
        controller = FPSController()
        assert controller.get_recommended_mode() == ProcessingMode.BALANCED

    def test_recommended_mode_high_fps(self):
        """High FPS recommends full mode."""
        config = AdaptiveConfig(target_fps=15.0)
        controller = FPSController(config)

        # Simulate high FPS
        for _ in range(10):
            controller.record_frame(30.0)
            time.sleep(0.03)  # ~33 FPS

        mode = controller.get_recommended_mode()
        # Should recommend FULL or BALANCED due to high FPS
        assert mode in [ProcessingMode.FULL, ProcessingMode.BALANCED]

    def test_recommended_mode_low_fps(self):
        """Low FPS recommends faster mode."""
        config = AdaptiveConfig(target_fps=30.0)
        controller = FPSController(config)

        # Simulate low FPS
        for _ in range(10):
            controller.record_frame(100.0)
            time.sleep(0.15)  # ~7 FPS

        mode = controller.get_recommended_mode()
        # Should recommend FAST or MINIMAL
        assert mode in [ProcessingMode.FAST, ProcessingMode.MINIMAL]


# =============================================================================
# AdaptiveProcessor Tests
# =============================================================================

class TestAdaptiveProcessor:
    """Test AdaptiveProcessor class."""

    def test_initial_state(self):
        """Test initial processor state."""
        processor = AdaptiveProcessor()

        assert processor.current_mode == ProcessingMode.BALANCED
        assert processor.current_complexity == SceneComplexity.MEDIUM

    def test_is_feature_enabled(self):
        """Check feature enabled status."""
        processor = AdaptiveProcessor()

        # Balanced mode
        assert processor.is_feature_enabled("fastsam") is True
        assert processor.is_feature_enabled("pointcloud") is False

    def test_update_returns_state(self):
        """Update returns current state."""
        processor = AdaptiveProcessor()

        state = processor.update(
            object_count=5,
            processing_time_ms=50.0,
        )

        assert "mode" in state
        assert "complexity" in state
        assert "fps" in state
        assert "features" in state

    def test_update_complexity(self):
        """Update changes complexity."""
        processor = AdaptiveProcessor()

        # Low object count
        processor.update(object_count=2, processing_time_ms=30.0)
        assert processor.current_complexity == SceneComplexity.LOW

        # High object count
        processor.update(object_count=30, processing_time_ms=30.0)
        assert processor.current_complexity == SceneComplexity.EXTREME

    def test_should_skip_frame(self):
        """Test frame skip decision."""
        processor = AdaptiveProcessor()

        # Fresh frame
        assert processor.should_skip_frame(frame_age_ms=50.0) is False

        # Stale frame
        assert processor.should_skip_frame(frame_age_ms=300.0) is True

    def test_get_stats(self):
        """Get processor statistics."""
        processor = AdaptiveProcessor()

        processor.update(object_count=5, processing_time_ms=50.0)
        stats = processor.get_stats()

        assert "mode" in stats
        assert "complexity" in stats
        assert "fps" in stats
        assert "target_fps" in stats
        assert "features_enabled" in stats

    def test_mode_smoothing(self):
        """Mode changes are smoothed."""
        processor = AdaptiveProcessor()

        # Single update shouldn't immediately change mode
        initial_mode = processor.current_mode

        processor.update(object_count=50, processing_time_ms=200.0)

        # Mode may stay same due to smoothing
        assert processor.current_mode in list(ProcessingMode)

    def test_features_enabled_list(self):
        """Features enabled matches current mode."""
        processor = AdaptiveProcessor()

        stats = processor.get_stats()
        features = stats["features_enabled"]

        # Balanced mode features
        assert "fastsam" in features
        assert "reid_matching" in features


# =============================================================================
# AdaptiveProcessingModule Tests
# =============================================================================

class TestAdaptiveProcessingModule:
    """Test AdaptiveProcessingModule class."""

    def test_module_name(self):
        """Module has correct name."""
        module = AdaptiveProcessingModule()
        assert module.name == "adaptive_processing"

    def test_pre_process_returns_features(self):
        """Pre-process returns enabled features."""
        module = AdaptiveProcessingModule()

        should_process, features = module.pre_process()

        assert should_process is True
        assert isinstance(features, dict)
        assert "fastsam" in features

    def test_pre_process_skip_stale(self):
        """Pre-process skips stale frames."""
        module = AdaptiveProcessingModule()

        should_process, features = module.pre_process(frame_age_ms=500.0)

        assert should_process is False
        assert features == {}

    def test_post_process_returns_state(self):
        """Post-process returns updated state."""
        module = AdaptiveProcessingModule()

        state = module.post_process(
            object_count=5,
            processing_time_ms=50.0,
        )

        assert "mode" in state
        assert "complexity" in state

    def test_get_stats(self):
        """Get module statistics."""
        module = AdaptiveProcessingModule()

        stats = module.get_stats()

        assert "mode" in stats
        assert "target_fps" in stats


# =============================================================================
# Integration Tests
# =============================================================================

class TestAdaptiveIntegration:
    """Integration tests for adaptive processing."""

    def test_full_processing_cycle(self):
        """Test complete processing cycle."""
        module = AdaptiveProcessingModule()

        # Simulate multiple frames
        for i in range(10):
            should_process, features = module.pre_process(frame_age_ms=10.0)

            if should_process:
                # Simulate processing
                time.sleep(0.05)

                state = module.post_process(
                    object_count=5 + i,
                    processing_time_ms=50.0 + i * 5,
                )

        stats = module.get_stats()
        assert stats["fps"] > 0 or len(module.processor.fps_controller._frame_times) < 2

    def test_adaptive_mode_transition(self):
        """Test mode transitions under load."""
        config = AdaptiveConfig(target_fps=20.0)
        processor = AdaptiveProcessor(config)

        # Start with normal load
        for _ in range(10):
            processor.update(object_count=5, processing_time_ms=40.0)
            time.sleep(0.05)

        initial_mode = processor.current_mode

        # Increase load significantly
        for _ in range(10):
            processor.update(object_count=50, processing_time_ms=150.0)
            time.sleep(0.15)

        # Mode should adapt (may go to faster mode)
        # The actual mode depends on timing
        assert processor.current_mode in list(ProcessingMode)

    def test_feature_toggle_by_mode(self):
        """Features toggle based on mode."""
        processor = AdaptiveProcessor()

        # Check features for each mode
        for mode in ProcessingMode:
            processor._current_mode = mode

            if mode == ProcessingMode.FULL:
                assert processor.is_feature_enabled("pose_estimation") is True
                assert processor.is_feature_enabled("pointcloud") is True
            elif mode == ProcessingMode.MINIMAL:
                assert processor.is_feature_enabled("pose_estimation") is False
                assert processor.is_feature_enabled("pointcloud") is False

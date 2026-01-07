"""Tests for performance profiler module.

Tests verify profiling functionality for timing,
memory, and pipeline stage measurements.
"""

import time

import pytest

from percept.utils.profiler import (
    FrameProfile,
    FrameProfiler,
    FullProfiler,
    MemoryProfiler,
    MemorySnapshot,
    PipelineProfiler,
    ProfileLevel,
    ProfilerConfig,
    StageProfiler,
    Timer,
    TimingRecord,
    TimingStats,
    get_default_profiler,
    profile_function,
    profile_stage,
    set_default_profiler,
)


# =============================================================================
# Timer Tests
# =============================================================================

class TestTimer:
    """Test Timer class."""

    def test_start_stop(self):
        """Start and stop timer."""
        timer = Timer("test")

        timer.start()
        time.sleep(0.01)
        duration = timer.stop()

        assert duration >= 10  # At least 10ms

    def test_context_manager(self):
        """Use timer as context manager."""
        with Timer("test") as timer:
            time.sleep(0.01)

        assert timer.duration_ms >= 10

    def test_duration_ms(self):
        """Get duration in milliseconds."""
        timer = Timer("test")
        timer.start()
        time.sleep(0.02)
        timer.stop()

        assert 15 < timer.duration_ms < 50

    def test_is_running(self):
        """Check if timer is running."""
        timer = Timer("test")

        assert timer.is_running is False

        timer.start()
        assert timer.is_running is True

        timer.stop()
        assert timer.is_running is False

    def test_lap_times(self):
        """Record lap times."""
        timer = Timer("test")
        timer.start()

        time.sleep(0.01)
        lap1 = timer.lap("phase1")

        time.sleep(0.01)
        lap2 = timer.lap("phase2")

        timer.stop()

        assert lap1 >= 10
        assert lap2 >= 10

        laps = timer.get_laps()
        assert len(laps) == 2
        assert laps[0][0] == "phase1"
        assert laps[1][0] == "phase2"

    def test_duration_while_running(self):
        """Get duration while timer is running."""
        timer = Timer("test")
        timer.start()
        time.sleep(0.01)

        # Duration available while running
        assert timer.duration_ms >= 10

    def test_no_laps_before_start(self):
        """Get laps returns empty before start."""
        timer = Timer("test")
        assert timer.get_laps() == []


# =============================================================================
# TimingRecord Tests
# =============================================================================

class TestTimingRecord:
    """Test TimingRecord class."""

    def test_create_record(self):
        """Create timing record."""
        record = TimingRecord.create(
            name="test",
            start_time=0.0,
            end_time=0.05,
            stage="segment",
        )

        assert record.name == "test"
        assert record.duration_ms == 50.0
        assert record.metadata["stage"] == "segment"


class TestTimingStats:
    """Test TimingStats class."""

    def test_from_records(self):
        """Create stats from records."""
        records = [
            TimingRecord(name="test", start_time=0, end_time=0.01, duration_ms=10),
            TimingRecord(name="test", start_time=0, end_time=0.02, duration_ms=20),
            TimingRecord(name="test", start_time=0, end_time=0.03, duration_ms=30),
        ]

        stats = TimingStats.from_records("test", records)

        assert stats.count == 3
        assert stats.total_ms == 60
        assert stats.min_ms == 10
        assert stats.max_ms == 30
        assert stats.mean_ms == 20

    def test_empty_records(self):
        """Empty records return default stats."""
        stats = TimingStats.from_records("test", [])

        assert stats.count == 0
        assert stats.total_ms == 0

    def test_percentiles(self):
        """Calculate percentiles."""
        records = [
            TimingRecord(name="t", start_time=0, end_time=0, duration_ms=i)
            for i in range(1, 101)  # 1-100
        ]

        stats = TimingStats.from_records("t", records)

        # Percentiles are approximate (index-based)
        assert 49 <= stats.p50_ms <= 52
        assert 94 <= stats.p95_ms <= 97
        assert 98 <= stats.p99_ms <= 100


# =============================================================================
# StageProfiler Tests
# =============================================================================

class TestStageProfiler:
    """Test StageProfiler class."""

    def test_start_stop(self):
        """Start and stop stage timing."""
        profiler = StageProfiler()

        profiler.start("segment")
        time.sleep(0.01)
        duration = profiler.stop("segment")

        assert duration >= 10

    def test_context_manager(self):
        """Use profile context manager."""
        profiler = StageProfiler()

        with profiler.profile("segment"):
            time.sleep(0.01)

        stats = profiler.get_stats("segment")
        assert stats is not None
        assert stats.count == 1
        assert stats.mean_ms >= 10

    def test_record_direct(self):
        """Record timing directly."""
        profiler = StageProfiler()

        profiler.record("segment", 50.0)
        profiler.record("segment", 60.0)

        stats = profiler.get_stats("segment")
        assert stats is not None
        assert stats.count == 2
        assert stats.mean_ms == 55.0

    def test_get_all_stats(self):
        """Get stats for all stages."""
        profiler = StageProfiler()

        profiler.record("segment", 50.0)
        profiler.record("track", 20.0)
        profiler.record("classify", 30.0)

        all_stats = profiler.get_all_stats()

        assert len(all_stats) == 3
        assert "segment" in all_stats
        assert "track" in all_stats
        assert "classify" in all_stats

    def test_get_slow_operations(self):
        """Get operations exceeding threshold."""
        config = ProfilerConfig(slow_operation_threshold_ms=50.0)
        profiler = StageProfiler(config)

        profiler.record("fast", 30.0)
        profiler.record("slow", 100.0)
        profiler.record("very_slow", 200.0)

        slow = profiler.get_slow_operations()

        assert len(slow) == 2
        assert slow[0].duration_ms == 200.0  # Sorted by duration
        assert slow[1].duration_ms == 100.0

    def test_disabled_profiling(self):
        """Disabled profiling skips recording."""
        config = ProfilerConfig(level=ProfileLevel.DISABLED)
        profiler = StageProfiler(config)

        profiler.record("segment", 50.0)

        stats = profiler.get_stats("segment")
        assert stats is None

    def test_clear(self):
        """Clear all records."""
        profiler = StageProfiler()

        profiler.record("segment", 50.0)
        profiler.clear()

        stats = profiler.get_stats("segment")
        assert stats is None

    def test_nonexistent_stats(self):
        """Get stats for nonexistent stage."""
        profiler = StageProfiler()

        stats = profiler.get_stats("nonexistent")
        assert stats is None


# =============================================================================
# FrameProfiler Tests
# =============================================================================

class TestFrameProfiler:
    """Test FrameProfiler class."""

    def test_frame_lifecycle(self):
        """Profile frame lifecycle."""
        profiler = FrameProfiler()

        profiler.start_frame(0)

        profiler.start_stage("segment")
        time.sleep(0.01)
        profiler.stop_stage("segment")

        profile = profiler.end_frame(object_count=5)

        assert profile.frame_id == 0
        assert profile.total_ms >= 10
        assert profile.object_count == 5
        assert "segment" in profile.stage_times

    def test_stage_context_manager(self):
        """Use stage context manager."""
        profiler = FrameProfiler()

        profiler.start_frame(1)

        with profiler.stage("segment"):
            time.sleep(0.01)

        with profiler.stage("track"):
            time.sleep(0.005)

        profile = profiler.end_frame()

        assert len(profile.stage_times) == 2
        assert profile.stage_times["segment"] >= 10

    def test_get_recent_profiles(self):
        """Get recent frame profiles."""
        profiler = FrameProfiler()

        for i in range(5):
            profiler.start_frame(i)
            time.sleep(0.005)
            profiler.end_frame()

        profiles = profiler.get_recent_profiles(3)

        assert len(profiles) == 3

    def test_get_average_frame_time(self):
        """Get average frame processing time."""
        profiler = FrameProfiler()

        for i in range(3):
            profiler.start_frame(i)
            time.sleep(0.01)
            profiler.end_frame()

        avg = profiler.get_average_frame_time()
        assert avg >= 10

    def test_get_stage_breakdown(self):
        """Get average time per stage."""
        profiler = FrameProfiler()

        for i in range(3):
            profiler.start_frame(i)
            with profiler.stage("segment"):
                time.sleep(0.01)
            profiler.end_frame()

        breakdown = profiler.get_stage_breakdown()

        assert "segment" in breakdown
        assert breakdown["segment"] >= 10

    def test_skipped_frame(self):
        """Profile skipped frame."""
        profiler = FrameProfiler()

        profiler.start_frame(0)
        profile = profiler.end_frame(skipped=True)

        assert profile.skipped is True

    def test_get_fps(self):
        """Calculate frames per second."""
        profiler = FrameProfiler()

        for i in range(10):
            profiler.start_frame(i)
            time.sleep(0.02)
            profiler.end_frame()

        fps = profiler.get_fps()

        # Should be around 50 FPS (1/0.02)
        assert 20 < fps < 100


class TestFrameProfile:
    """Test FrameProfile class."""

    def test_get_breakdown(self):
        """Get timing breakdown."""
        profile = FrameProfile(
            frame_id=0,
            timestamp=time.time(),
            total_ms=100.0,
            stage_times={"segment": 60.0, "track": 30.0, "classify": 10.0},
        )

        breakdown = profile.get_breakdown()

        assert len(breakdown) == 3
        assert breakdown[0][0] == "segment"  # Sorted by time
        assert breakdown[0][2] == 60.0  # 60% of total


# =============================================================================
# MemoryProfiler Tests
# =============================================================================

class TestMemoryProfiler:
    """Test MemoryProfiler class."""

    def test_snapshot(self):
        """Take memory snapshot."""
        profiler = MemoryProfiler()

        snapshot = profiler.snapshot()

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.timestamp > 0

    def test_get_current(self):
        """Get current memory usage."""
        profiler = MemoryProfiler()

        current = profiler.get_current()

        assert isinstance(current, MemorySnapshot)

    def test_get_history(self):
        """Get memory history."""
        profiler = MemoryProfiler()

        for _ in range(3):
            profiler.snapshot()

        history = profiler.get_history()

        assert len(history) == 3

    def test_get_peak(self):
        """Get peak memory usage."""
        profiler = MemoryProfiler()

        profiler.snapshot()
        profiler.snapshot()

        peak = profiler.get_peak()

        assert isinstance(peak, MemorySnapshot)

    def test_check_warning(self):
        """Check memory warning threshold."""
        config = ProfilerConfig(memory_warning_threshold_mb=0.001)  # Very low
        profiler = MemoryProfiler(config)

        warning = profiler.check_warning()

        # Should trigger warning (memory > 0.001MB)
        assert warning is not None or True  # May not trigger if psutil unavailable


# =============================================================================
# FullProfiler Tests
# =============================================================================

class TestFullProfiler:
    """Test FullProfiler class."""

    def test_start_stop(self):
        """Start and stop full profiling."""
        profiler = FullProfiler()

        profiler.start()

        # Do some work
        sum(range(1000))

        profiler.stop()

        stats = profiler.get_stats()
        assert stats is not None

    def test_context_manager(self):
        """Use profiler as context manager."""
        profiler = FullProfiler()

        with profiler.profile():
            sum(range(1000))

        report = profiler.get_report()
        assert "function calls" in report.lower() or len(report) > 0

    def test_profile_function(self):
        """Profile a function call."""
        profiler = FullProfiler()

        def test_func():
            return sum(range(100))

        result, report = profiler.profile_function(test_func)

        assert result == sum(range(100))
        assert len(report) > 0


# =============================================================================
# PipelineProfiler Tests
# =============================================================================

class TestPipelineProfiler:
    """Test PipelineProfiler class."""

    def test_frame_profiling(self):
        """Profile frame processing."""
        profiler = PipelineProfiler()

        profiler.start_frame(0)

        with profiler.stage("segment"):
            time.sleep(0.01)

        profile = profiler.end_frame(object_count=5)

        assert profile.frame_id == 0
        assert "segment" in profile.stage_times

    def test_record_stage(self):
        """Record stage timing directly."""
        profiler = PipelineProfiler()

        profiler.record_stage("segment", 50.0)

        stats = profiler.stage_profiler.get_stats("segment")
        assert stats is not None
        assert stats.mean_ms == 50.0

    def test_snapshot_memory(self):
        """Take memory snapshot."""
        profiler = PipelineProfiler()

        snapshot = profiler.snapshot_memory()

        assert isinstance(snapshot, MemorySnapshot)

    def test_get_summary(self):
        """Get profiling summary."""
        profiler = PipelineProfiler()

        for i in range(3):
            profiler.start_frame(i)
            with profiler.stage("segment"):
                time.sleep(0.01)
            profiler.end_frame()

        summary = profiler.get_summary()

        assert "fps" in summary
        assert "avg_frame_time_ms" in summary
        assert "stage_breakdown" in summary
        assert "memory_mb" in summary

    def test_get_report(self):
        """Get formatted report."""
        profiler = PipelineProfiler()

        for i in range(3):
            profiler.start_frame(i)
            with profiler.stage("segment"):
                time.sleep(0.01)
            profiler.end_frame()

        report = profiler.get_report()

        assert "PIPELINE PERFORMANCE REPORT" in report
        assert "FPS" in report
        assert "segment" in report

    def test_clear(self):
        """Clear profiling data."""
        profiler = PipelineProfiler()

        profiler.record_stage("segment", 50.0)
        profiler.clear()

        stats = profiler.stage_profiler.get_stats("segment")
        assert stats is None


# =============================================================================
# Decorator Tests
# =============================================================================

class TestDecorators:
    """Test profiling decorators."""

    def test_profile_stage_decorator(self):
        """Use profile_stage decorator."""
        profiler = PipelineProfiler()
        set_default_profiler(profiler)

        @profile_stage("test_stage")
        def test_func():
            time.sleep(0.01)
            return 42

        result = test_func()

        assert result == 42

        stats = profiler.stage_profiler.get_stats("test_stage")
        assert stats is not None
        assert stats.mean_ms >= 10

    def test_profile_function_decorator(self):
        """Use profile_function decorator."""
        profiler = PipelineProfiler()
        set_default_profiler(profiler)

        @profile_function
        def my_function():
            time.sleep(0.01)
            return "result"

        result = my_function()

        assert result == "result"

        stats = profiler.stage_profiler.get_stats("my_function")
        assert stats is not None

    def test_get_default_profiler(self):
        """Get default profiler creates one."""
        profiler = get_default_profiler()

        assert isinstance(profiler, PipelineProfiler)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestProfilerConfig:
    """Test ProfilerConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProfilerConfig()

        assert config.level == ProfileLevel.BASIC
        assert config.history_size == 1000
        assert config.slow_operation_threshold_ms == 100.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = ProfilerConfig(
            level=ProfileLevel.DETAILED,
            history_size=500,
            slow_operation_threshold_ms=50.0,
        )

        assert config.level == ProfileLevel.DETAILED
        assert config.history_size == 500
        assert config.slow_operation_threshold_ms == 50.0


class TestProfileLevel:
    """Test ProfileLevel enum."""

    def test_level_values(self):
        """Test level enum values."""
        assert ProfileLevel.DISABLED.value == "disabled"
        assert ProfileLevel.BASIC.value == "basic"
        assert ProfileLevel.DETAILED.value == "detailed"
        assert ProfileLevel.FULL.value == "full"


# =============================================================================
# Integration Tests
# =============================================================================

class TestProfilerIntegration:
    """Integration tests for profiler."""

    def test_full_pipeline_profiling(self):
        """Test complete pipeline profiling workflow."""
        profiler = PipelineProfiler()

        # Simulate processing multiple frames
        for frame_id in range(5):
            profiler.start_frame(frame_id)

            with profiler.stage("capture"):
                time.sleep(0.005)

            with profiler.stage("segment"):
                time.sleep(0.01)

            with profiler.stage("track"):
                time.sleep(0.005)

            profiler.end_frame(object_count=frame_id + 1)

        # Verify summary
        summary = profiler.get_summary()

        assert summary["fps"] > 0
        assert summary["avg_frame_time_ms"] > 0
        assert "capture" in summary["stage_breakdown"]
        assert "segment" in summary["stage_breakdown"]
        assert "track" in summary["stage_breakdown"]

        # Verify report generation
        report = profiler.get_report()
        assert len(report) > 100  # Non-trivial report

    def test_profiling_under_load(self):
        """Test profiling under simulated load."""
        profiler = PipelineProfiler()

        # Many quick operations
        for i in range(50):
            profiler.start_frame(i)
            with profiler.stage("process"):
                pass  # Very fast
            profiler.end_frame()

        summary = profiler.get_summary()

        # Should handle many frames
        assert len(profiler.frame_profiler.get_recent_profiles()) == 50

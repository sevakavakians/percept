"""Performance profiling utilities.

Provides tools for profiling pipeline stages, identifying bottlenecks,
and generating performance reports.
"""

import cProfile
import io
import pstats
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# Enums and Configuration
# =============================================================================

class ProfileLevel(Enum):
    """Profiling detail level."""
    DISABLED = "disabled"   # No profiling
    BASIC = "basic"         # Only timing
    DETAILED = "detailed"   # Timing + memory
    FULL = "full"           # Full cProfile


@dataclass
class ProfilerConfig:
    """Configuration for profiler."""

    # Profiling level
    level: ProfileLevel = ProfileLevel.BASIC

    # History settings
    history_size: int = 1000
    aggregation_window_seconds: float = 60.0

    # Thresholds for warnings
    slow_operation_threshold_ms: float = 100.0
    memory_warning_threshold_mb: float = 100.0

    # Output settings
    report_top_n: int = 20
    include_call_graph: bool = False


# =============================================================================
# Timing Utilities
# =============================================================================

@dataclass
class TimingRecord:
    """Record of a single timing measurement."""

    name: str
    start_time: float
    end_time: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        start_time: float,
        end_time: float,
        **metadata,
    ) -> "TimingRecord":
        """Create timing record."""
        duration_ms = (end_time - start_time) * 1000
        return cls(
            name=name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            metadata=metadata,
        )


@dataclass
class TimingStats:
    """Statistics for timing measurements."""

    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    @classmethod
    def from_records(cls, name: str, records: List[TimingRecord]) -> "TimingStats":
        """Create stats from timing records."""
        if not records:
            return cls(name=name)

        durations = [r.duration_ms for r in records]
        durations.sort()

        count = len(durations)
        total = sum(durations)
        mean = total / count

        # Calculate standard deviation
        variance = sum((d - mean) ** 2 for d in durations) / count
        std = variance ** 0.5

        # Percentiles
        def percentile(p: float) -> float:
            idx = int(count * p / 100)
            return durations[min(idx, count - 1)]

        return cls(
            name=name,
            count=count,
            total_ms=total,
            min_ms=min(durations),
            max_ms=max(durations),
            mean_ms=mean,
            std_ms=std,
            p50_ms=percentile(50),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
        )


class Timer:
    """Simple timer for measuring operations."""

    def __init__(self, name: str = ""):
        self.name = name
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._laps: List[Tuple[str, float]] = []

    def start(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._laps = []
        return self

    def stop(self) -> float:
        """Stop the timer and return duration in ms."""
        self._end_time = time.perf_counter()
        return self.duration_ms

    def lap(self, name: str = "") -> float:
        """Record a lap time."""
        now = time.perf_counter()
        if self._start_time is None:
            return 0.0

        last_time = self._laps[-1][1] if self._laps else self._start_time
        lap_duration = (now - last_time) * 1000
        self._laps.append((name, now))
        return lap_duration

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.perf_counter()
        return (end - self._start_time) * 1000

    @property
    def is_running(self) -> bool:
        """Check if timer is running."""
        return self._start_time is not None and self._end_time is None

    def get_laps(self) -> List[Tuple[str, float]]:
        """Get lap times as (name, duration_ms) pairs."""
        if not self._laps or self._start_time is None:
            return []

        result = []
        prev_time = self._start_time

        for name, lap_time in self._laps:
            duration = (lap_time - prev_time) * 1000
            result.append((name, duration))
            prev_time = lap_time

        return result

    def __enter__(self) -> "Timer":
        """Context manager entry."""
        return self.start()

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()


# =============================================================================
# Stage Profiler
# =============================================================================

class StageProfiler:
    """Profile individual pipeline stages."""

    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self._records: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.history_size)
        )
        self._active_timers: Dict[str, Timer] = {}
        self._lock = threading.Lock()

    def start(self, stage_name: str, **metadata) -> None:
        """Start timing a stage."""
        if self.config.level == ProfileLevel.DISABLED:
            return

        timer = Timer(stage_name)
        timer.start()
        timer.metadata = metadata

        with self._lock:
            self._active_timers[stage_name] = timer

    def stop(self, stage_name: str) -> float:
        """Stop timing a stage and record."""
        if self.config.level == ProfileLevel.DISABLED:
            return 0.0

        with self._lock:
            timer = self._active_timers.pop(stage_name, None)

        if timer is None:
            return 0.0

        duration = timer.stop()

        record = TimingRecord.create(
            name=stage_name,
            start_time=timer._start_time,
            end_time=timer._end_time,
            **getattr(timer, "metadata", {}),
        )

        with self._lock:
            self._records[stage_name].append(record)

        return duration

    @contextmanager
    def profile(self, stage_name: str, **metadata):
        """Context manager for profiling a stage."""
        self.start(stage_name, **metadata)
        try:
            yield
        finally:
            self.stop(stage_name)

    def record(self, stage_name: str, duration_ms: float, **metadata) -> None:
        """Record a timing directly."""
        if self.config.level == ProfileLevel.DISABLED:
            return

        now = time.perf_counter()
        record = TimingRecord(
            name=stage_name,
            start_time=now - duration_ms / 1000,
            end_time=now,
            duration_ms=duration_ms,
            metadata=metadata,
        )

        with self._lock:
            self._records[stage_name].append(record)

    def get_stats(self, stage_name: str) -> Optional[TimingStats]:
        """Get statistics for a stage."""
        with self._lock:
            records = list(self._records.get(stage_name, []))

        if not records:
            return None

        return TimingStats.from_records(stage_name, records)

    def get_all_stats(self) -> Dict[str, TimingStats]:
        """Get statistics for all stages."""
        with self._lock:
            stage_names = list(self._records.keys())

        return {
            name: self.get_stats(name)
            for name in stage_names
            if self.get_stats(name) is not None
        }

    def get_slow_operations(self) -> List[TimingRecord]:
        """Get operations exceeding threshold."""
        threshold = self.config.slow_operation_threshold_ms
        slow = []

        with self._lock:
            for records in self._records.values():
                slow.extend(
                    r for r in records if r.duration_ms > threshold
                )

        return sorted(slow, key=lambda r: r.duration_ms, reverse=True)

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()
            self._active_timers.clear()


# =============================================================================
# Frame Profiler
# =============================================================================

@dataclass
class FrameProfile:
    """Profile for a single frame."""

    frame_id: int
    timestamp: float
    total_ms: float
    stage_times: Dict[str, float] = field(default_factory=dict)
    object_count: int = 0
    skipped: bool = False

    def get_breakdown(self) -> List[Tuple[str, float, float]]:
        """Get timing breakdown as (stage, ms, percentage)."""
        if self.total_ms <= 0:
            return []

        breakdown = []
        for stage, ms in sorted(
            self.stage_times.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = (ms / self.total_ms) * 100
            breakdown.append((stage, ms, pct))

        return breakdown


class FrameProfiler:
    """Profile complete frame processing."""

    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self._profiles: deque = deque(maxlen=self.config.history_size)
        self._current_frame: Optional[FrameProfile] = None
        self._stage_profiler = StageProfiler(config)
        self._lock = threading.Lock()

    def start_frame(self, frame_id: int) -> None:
        """Start profiling a new frame."""
        with self._lock:
            self._current_frame = FrameProfile(
                frame_id=frame_id,
                timestamp=time.time(),
                total_ms=0.0,
            )
            self._frame_timer = Timer(f"frame_{frame_id}")
            self._frame_timer.start()

    def start_stage(self, stage_name: str) -> None:
        """Start profiling a stage within current frame."""
        self._stage_profiler.start(stage_name)

    def stop_stage(self, stage_name: str) -> float:
        """Stop profiling a stage."""
        duration = self._stage_profiler.stop(stage_name)

        with self._lock:
            if self._current_frame:
                self._current_frame.stage_times[stage_name] = duration

        return duration

    @contextmanager
    def stage(self, stage_name: str):
        """Context manager for stage profiling."""
        self.start_stage(stage_name)
        try:
            yield
        finally:
            self.stop_stage(stage_name)

    def end_frame(self, object_count: int = 0, skipped: bool = False) -> FrameProfile:
        """End profiling current frame."""
        with self._lock:
            if self._current_frame is None:
                return FrameProfile(frame_id=-1, timestamp=0, total_ms=0)

            self._frame_timer.stop()
            self._current_frame.total_ms = self._frame_timer.duration_ms
            self._current_frame.object_count = object_count
            self._current_frame.skipped = skipped

            profile = self._current_frame
            self._profiles.append(profile)
            self._current_frame = None

        return profile

    def get_recent_profiles(self, count: int = 100) -> List[FrameProfile]:
        """Get recent frame profiles."""
        with self._lock:
            profiles = list(self._profiles)

        return profiles[-count:]

    def get_average_frame_time(self) -> float:
        """Get average frame processing time."""
        profiles = self.get_recent_profiles()
        if not profiles:
            return 0.0

        times = [p.total_ms for p in profiles if not p.skipped]
        return sum(times) / len(times) if times else 0.0

    def get_stage_breakdown(self) -> Dict[str, float]:
        """Get average time per stage."""
        profiles = self.get_recent_profiles()
        if not profiles:
            return {}

        stage_totals: Dict[str, List[float]] = defaultdict(list)

        for profile in profiles:
            if not profile.skipped:
                for stage, ms in profile.stage_times.items():
                    stage_totals[stage].append(ms)

        return {
            stage: sum(times) / len(times)
            for stage, times in stage_totals.items()
        }

    def get_fps(self) -> float:
        """Get frames per second."""
        profiles = self.get_recent_profiles(30)
        if len(profiles) < 2:
            return 0.0

        time_span = profiles[-1].timestamp - profiles[0].timestamp
        if time_span <= 0:
            return 0.0

        return len(profiles) / time_span


# =============================================================================
# Memory Profiler
# =============================================================================

@dataclass
class MemorySnapshot:
    """Snapshot of memory usage."""

    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float
    available_mb: float


class MemoryProfiler:
    """Profile memory usage."""

    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self._snapshots: deque = deque(maxlen=self.config.history_size)
        self._lock = threading.Lock()

    def snapshot(self) -> MemorySnapshot:
        """Take memory snapshot."""
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            sys_mem = psutil.virtual_memory()

            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss_mb=mem_info.rss / (1024 * 1024),
                vms_mb=mem_info.vms / (1024 * 1024),
                percent=process.memory_percent(),
                available_mb=sys_mem.available / (1024 * 1024),
            )
        except ImportError:
            # psutil not available
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss_mb=0.0,
                vms_mb=0.0,
                percent=0.0,
                available_mb=0.0,
            )

        with self._lock:
            self._snapshots.append(snapshot)

        return snapshot

    def get_current(self) -> MemorySnapshot:
        """Get current memory usage."""
        return self.snapshot()

    def get_peak(self) -> MemorySnapshot:
        """Get peak memory usage."""
        with self._lock:
            snapshots = list(self._snapshots)

        if not snapshots:
            return self.snapshot()

        return max(snapshots, key=lambda s: s.rss_mb)

    def get_history(self, count: int = 100) -> List[MemorySnapshot]:
        """Get memory history."""
        with self._lock:
            return list(self._snapshots)[-count:]

    def check_warning(self) -> Optional[str]:
        """Check if memory usage exceeds warning threshold."""
        current = self.get_current()

        if current.rss_mb > self.config.memory_warning_threshold_mb:
            return (
                f"Memory usage ({current.rss_mb:.1f}MB) exceeds "
                f"threshold ({self.config.memory_warning_threshold_mb}MB)"
            )

        return None


# =============================================================================
# Full Profiler
# =============================================================================

class FullProfiler:
    """Full cProfile-based profiler."""

    def __init__(self):
        self._profiler: Optional[cProfile.Profile] = None
        self._stats: Optional[pstats.Stats] = None

    def start(self) -> None:
        """Start profiling."""
        self._profiler = cProfile.Profile()
        self._profiler.enable()

    def stop(self) -> None:
        """Stop profiling."""
        if self._profiler:
            self._profiler.disable()

    @contextmanager
    def profile(self):
        """Context manager for profiling."""
        self.start()
        try:
            yield
        finally:
            self.stop()

    def get_stats(self) -> Optional[pstats.Stats]:
        """Get profiling statistics."""
        if self._profiler is None:
            return None

        stream = io.StringIO()
        self._stats = pstats.Stats(self._profiler, stream=stream)
        return self._stats

    def get_report(
        self,
        sort_by: str = "cumulative",
        top_n: int = 20,
    ) -> str:
        """Get formatted report."""
        stats = self.get_stats()
        if stats is None:
            return "No profiling data"

        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats(sort_by)
        stats.print_stats(top_n)

        return stream.getvalue()

    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, str]:
        """Profile a function call."""
        with self.profile():
            result = func(*args, **kwargs)

        report = self.get_report()
        return result, report


# =============================================================================
# Pipeline Profiler
# =============================================================================

class PipelineProfiler:
    """Main profiler for the entire pipeline."""

    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self.stage_profiler = StageProfiler(config)
        self.frame_profiler = FrameProfiler(config)
        self.memory_profiler = MemoryProfiler(config)
        self._full_profiler: Optional[FullProfiler] = None

        if config and config.level == ProfileLevel.FULL:
            self._full_profiler = FullProfiler()

    def start_frame(self, frame_id: int) -> None:
        """Start profiling a frame."""
        self.frame_profiler.start_frame(frame_id)

    def end_frame(self, object_count: int = 0, skipped: bool = False) -> FrameProfile:
        """End profiling current frame."""
        return self.frame_profiler.end_frame(object_count, skipped)

    @contextmanager
    def stage(self, stage_name: str):
        """Profile a pipeline stage."""
        with self.frame_profiler.stage(stage_name):
            yield

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Record stage timing directly."""
        self.stage_profiler.record(stage_name, duration_ms)

    def snapshot_memory(self) -> MemorySnapshot:
        """Take memory snapshot."""
        return self.memory_profiler.snapshot()

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        stage_stats = self.stage_profiler.get_all_stats()
        memory = self.memory_profiler.get_current()

        return {
            "fps": self.frame_profiler.get_fps(),
            "avg_frame_time_ms": self.frame_profiler.get_average_frame_time(),
            "stage_breakdown": self.frame_profiler.get_stage_breakdown(),
            "stage_stats": {
                name: {
                    "count": stats.count,
                    "mean_ms": stats.mean_ms,
                    "p95_ms": stats.p95_ms,
                    "max_ms": stats.max_ms,
                }
                for name, stats in stage_stats.items()
            },
            "memory_mb": memory.rss_mb,
            "memory_percent": memory.percent,
        }

    def get_report(self) -> str:
        """Get formatted report."""
        summary = self.get_summary()

        lines = [
            "=" * 60,
            "PIPELINE PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"FPS: {summary['fps']:.1f}",
            f"Average Frame Time: {summary['avg_frame_time_ms']:.1f}ms",
            f"Memory Usage: {summary['memory_mb']:.1f}MB ({summary['memory_percent']:.1f}%)",
            "",
            "Stage Breakdown:",
            "-" * 40,
        ]

        for stage, ms in sorted(
            summary["stage_breakdown"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"  {stage}: {ms:.1f}ms")

        lines.extend([
            "",
            "Stage Statistics:",
            "-" * 40,
        ])

        for stage, stats in summary["stage_stats"].items():
            lines.append(
                f"  {stage}: mean={stats['mean_ms']:.1f}ms, "
                f"p95={stats['p95_ms']:.1f}ms, max={stats['max_ms']:.1f}ms"
            )

        slow_ops = self.stage_profiler.get_slow_operations()[:5]
        if slow_ops:
            lines.extend([
                "",
                "Slow Operations (top 5):",
                "-" * 40,
            ])
            for op in slow_ops:
                lines.append(f"  {op.name}: {op.duration_ms:.1f}ms")

        lines.append("=" * 60)

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all profiling data."""
        self.stage_profiler.clear()


# =============================================================================
# Profiling Decorators
# =============================================================================

_default_profiler: Optional[PipelineProfiler] = None


def get_default_profiler() -> PipelineProfiler:
    """Get or create default profiler."""
    global _default_profiler
    if _default_profiler is None:
        _default_profiler = PipelineProfiler()
    return _default_profiler


def set_default_profiler(profiler: PipelineProfiler) -> None:
    """Set default profiler."""
    global _default_profiler
    _default_profiler = profiler


def profile_stage(stage_name: str):
    """Decorator to profile a function as a stage."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            profiler = get_default_profiler()
            with profiler.stage_profiler.profile(stage_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_function(func: Callable) -> Callable:
    """Decorator to profile a function."""
    def wrapper(*args, **kwargs):
        profiler = get_default_profiler()
        with profiler.stage_profiler.profile(func.__name__):
            return func(*args, **kwargs)
    return wrapper

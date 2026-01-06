"""Metrics panel component for PERCEPT UI.

Collects, aggregates, and formats performance metrics
for display in the dashboard.
"""

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil


# =============================================================================
# Metric Types
# =============================================================================

@dataclass
class MetricSample:
    """Single metric sample."""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetricStats:
    """Statistics for a metric over a time window."""
    name: str
    current: float
    mean: float
    min_val: float
    max_val: float
    std_dev: float
    samples: int
    unit: str = ""


@dataclass
class SystemMetrics:
    """Current system resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    temperature: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineMetrics:
    """Pipeline processing metrics."""
    fps: float
    frame_count: int
    objects_detected: int
    latency_ms: float
    segmentation_ms: float
    tracking_ms: float
    reid_ms: float
    classification_ms: float
    database_ms: float
    queue_depth: int
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Metric Collector
# =============================================================================

class MetricCollector:
    """Collect and aggregate metrics over time."""

    def __init__(self, window_size: int = 100, max_age_seconds: float = 300):
        self.window_size = window_size
        self.max_age_seconds = max_age_seconds
        self._metrics: Dict[str, deque] = {}
        self._units: Dict[str, str] = {}

    def record(self, name: str, value: float, unit: str = ""):
        """Record a metric value."""
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=self.window_size)
            self._units[name] = unit

        self._metrics[name].append(MetricSample(value=value))

    def get_stats(self, name: str) -> Optional[MetricStats]:
        """Get statistics for a metric."""
        if name not in self._metrics or not self._metrics[name]:
            return None

        samples = list(self._metrics[name])

        # Filter out old samples
        cutoff = datetime.now() - timedelta(seconds=self.max_age_seconds)
        samples = [s for s in samples if s.timestamp > cutoff]

        if not samples:
            return None

        values = [s.value for s in samples]

        return MetricStats(
            name=name,
            current=values[-1],
            mean=statistics.mean(values),
            min_val=min(values),
            max_val=max(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            samples=len(values),
            unit=self._units.get(name, ""),
        )

    def get_history(
        self,
        name: str,
        max_samples: int = None,
    ) -> List[MetricSample]:
        """Get metric history."""
        if name not in self._metrics:
            return []

        samples = list(self._metrics[name])
        if max_samples:
            samples = samples[-max_samples:]

        return samples

    def get_all_names(self) -> List[str]:
        """Get all metric names."""
        return list(self._metrics.keys())

    def clear(self, name: str = None):
        """Clear metrics."""
        if name:
            if name in self._metrics:
                self._metrics[name].clear()
        else:
            for deq in self._metrics.values():
                deq.clear()


# =============================================================================
# System Monitor
# =============================================================================

class SystemMonitor:
    """Monitor system resources."""

    def __init__(self):
        self._last_cpu_times = None

    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # Disk usage
        try:
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
        except Exception:
            disk_percent = 0.0

        # Temperature (Linux)
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        temperature = entries[0].current
                        break
        except Exception:
            pass

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_usage_percent=disk_percent,
            temperature=temperature,
        )

    def get_cpu_per_core(self) -> List[float]:
        """Get CPU usage per core."""
        return psutil.cpu_percent(percpu=True)

    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        process = psutil.Process()

        return {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
        }


# =============================================================================
# FPS Counter
# =============================================================================

class FPSCounter:
    """Calculate frames per second."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._timestamps: deque = deque(maxlen=window_size)
        self._frame_count = 0

    def tick(self):
        """Record a frame."""
        self._timestamps.append(time.perf_counter())
        self._frame_count += 1

    @property
    def fps(self) -> float:
        """Get current FPS."""
        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed

    @property
    def frame_count(self) -> int:
        """Get total frame count."""
        return self._frame_count

    def reset(self):
        """Reset counter."""
        self._timestamps.clear()
        self._frame_count = 0


# =============================================================================
# Latency Tracker
# =============================================================================

class LatencyTracker:
    """Track operation latencies."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._operations: Dict[str, deque] = {}

    def start(self, operation: str) -> "LatencyContext":
        """Start tracking an operation."""
        return LatencyContext(self, operation)

    def record(self, operation: str, latency_ms: float):
        """Record a latency value."""
        if operation not in self._operations:
            self._operations[operation] = deque(maxlen=self.window_size)

        self._operations[operation].append(latency_ms)

    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get latency statistics for an operation."""
        if operation not in self._operations or not self._operations[operation]:
            return None

        values = list(self._operations[operation])

        return {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "p50": statistics.median(values),
            "p95": sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max(values),
            "p99": sorted(values)[int(len(values) * 0.99)] if len(values) >= 100 else max(values),
            "samples": len(values),
        }

    def get_all_operations(self) -> List[str]:
        """Get all tracked operations."""
        return list(self._operations.keys())


class LatencyContext:
    """Context manager for timing operations."""

    def __init__(self, tracker: LatencyTracker, operation: str):
        self.tracker = tracker
        self.operation = operation
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self._start) * 1000  # ms
        self.tracker.record(self.operation, elapsed)


# =============================================================================
# Metrics Aggregator
# =============================================================================

class MetricsAggregator:
    """Aggregate all metrics for dashboard display."""

    def __init__(self):
        self.collector = MetricCollector()
        self.system_monitor = SystemMonitor()
        self.fps_counter = FPSCounter()
        self.latency_tracker = LatencyTracker()

    def record_frame(
        self,
        objects_detected: int = 0,
        latencies: Dict[str, float] = None,
    ):
        """Record frame processing metrics."""
        self.fps_counter.tick()
        self.collector.record("objects_detected", objects_detected)

        if latencies:
            for name, value in latencies.items():
                self.latency_tracker.record(name, value)
                self.collector.record(f"latency_{name}", value, "ms")

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display."""
        system = self.system_monitor.get_metrics()

        # Build latency summary
        latency_summary = {}
        for op in self.latency_tracker.get_all_operations():
            stats = self.latency_tracker.get_stats(op)
            if stats:
                latency_summary[op] = stats["mean"]

        return {
            "system": {
                "cpu_percent": system.cpu_percent,
                "memory_percent": system.memory_percent,
                "memory_used_mb": system.memory_used_mb,
                "temperature": system.temperature,
                "disk_percent": system.disk_usage_percent,
            },
            "processing": {
                "fps": self.fps_counter.fps,
                "frame_count": self.fps_counter.frame_count,
            },
            "latency": latency_summary,
            "timestamp": datetime.now().isoformat(),
        }

    def get_history(
        self,
        metric_name: str,
        max_samples: int = 60,
    ) -> List[Dict[str, Any]]:
        """Get metric history for charting."""
        samples = self.collector.get_history(metric_name, max_samples)
        return [
            {"value": s.value, "timestamp": s.timestamp.isoformat()}
            for s in samples
        ]


# =============================================================================
# Metrics Formatter
# =============================================================================

class MetricsFormatter:
    """Format metrics for display."""

    @staticmethod
    def format_bytes(value: float) -> str:
        """Format bytes to human readable."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(value) < 1024:
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} PB"

    @staticmethod
    def format_latency(ms: float) -> str:
        """Format latency value."""
        if ms < 1:
            return f"{ms * 1000:.1f} us"
        elif ms < 1000:
            return f"{ms:.1f} ms"
        else:
            return f"{ms / 1000:.2f} s"

    @staticmethod
    def format_fps(fps: float) -> str:
        """Format FPS value."""
        return f"{fps:.1f} FPS"

    @staticmethod
    def format_percent(percent: float) -> str:
        """Format percentage."""
        return f"{percent:.1f}%"

    @staticmethod
    def format_temperature(celsius: float) -> str:
        """Format temperature."""
        return f"{celsius:.1f}C"

"""Tests for PERCEPT UI components.

Tests verify UI components work correctly.
"""

import numpy as np
import pytest
from datetime import datetime

from ui.models import (
    AlertLevel,
    CameraStatus,
    DashboardData,
    EventType,
    MetricsData,
    NodeType,
    ObjectDetail,
    ObjectSummary,
    PaginatedResponse,
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
    ReviewItem,
    StageOutput,
    ValidationResult,
    WebSocketEvent,
)


# =============================================================================
# Model Tests
# =============================================================================

class TestDashboardData:
    """Test DashboardData model."""

    def test_create_default(self):
        """Create with defaults."""
        data = DashboardData()

        assert data.fps == 0.0
        assert data.cpu_usage == 0.0
        assert len(data.cameras) == 0

    def test_create_with_values(self):
        """Create with specified values."""
        data = DashboardData(
            fps=15.5,
            cpu_usage=45.2,
            memory_usage=60.0,
            total_objects=100,
            pending_review=5,
        )

        assert data.fps == 15.5
        assert data.cpu_usage == 45.2
        assert data.total_objects == 100

    def test_with_cameras(self):
        """Create with camera list."""
        cameras = [
            CameraStatus(id="cam1", name="Front", connected=True, fps=30.0),
            CameraStatus(id="cam2", name="Rear", connected=True, fps=30.0),
        ]
        data = DashboardData(cameras=cameras)

        assert len(data.cameras) == 2
        assert data.cameras[0].id == "cam1"


class TestPipelineGraph:
    """Test PipelineGraph model."""

    def test_create_graph(self):
        """Create pipeline graph."""
        nodes = [
            PipelineNode(id="input", type=NodeType.INPUT, label="Input"),
            PipelineNode(id="process", type=NodeType.PROCESS, label="Process"),
            PipelineNode(id="output", type=NodeType.OUTPUT, label="Output"),
        ]
        edges = [
            PipelineEdge(source="input", target="process"),
            PipelineEdge(source="process", target="output"),
        ]

        graph = PipelineGraph(name="test", nodes=nodes, edges=edges)

        assert graph.name == "test"
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_node_types(self):
        """Test node type enum."""
        assert NodeType.INPUT.value == "input"
        assert NodeType.PROCESS.value == "process"
        assert NodeType.OUTPUT.value == "output"


class TestMetricsData:
    """Test MetricsData model."""

    def test_create_default(self):
        """Create with defaults."""
        data = MetricsData()

        assert data.fps_history == []
        assert data.frames_processed == 0

    def test_with_history(self):
        """Create with history."""
        data = MetricsData(
            fps_history=[10.0, 15.0, 14.5, 15.2],
            latency_history=[58.0, 62.0, 55.0],
        )

        assert len(data.fps_history) == 4
        assert len(data.latency_history) == 3


class TestObjectSummary:
    """Test ObjectSummary model."""

    def test_create(self):
        """Create object summary."""
        obj = ObjectSummary(
            id="obj-001",
            primary_class="person",
            confidence=0.92,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )

        assert obj.id == "obj-001"
        assert obj.primary_class == "person"
        assert obj.confidence == 0.92


class TestReviewItem:
    """Test ReviewItem model."""

    def test_create(self):
        """Create review item."""
        item = ReviewItem(
            id="review-001",
            object_id="obj-001",
            primary_class="person",
            confidence=0.4,
            reason="low_confidence",
            created_at=datetime.now(),
        )

        assert item.object_id == "obj-001"
        assert item.confidence == 0.4


class TestWebSocketEvent:
    """Test WebSocketEvent model."""

    def test_create_event(self):
        """Create WebSocket event."""
        event = WebSocketEvent(
            event=EventType.OBJECT_DETECTED,
            data={"object_id": "obj-001", "class": "person"},
        )

        assert event.event == EventType.OBJECT_DETECTED
        assert event.data["object_id"] == "obj-001"

    def test_event_types(self):
        """Test event type enum values."""
        assert EventType.FRAME_PROCESSED.value == "frame_processed"
        assert EventType.OBJECT_DETECTED.value == "object_detected"
        assert EventType.REVIEW_NEEDED.value == "review_needed"


class TestValidationResult:
    """Test ValidationResult model."""

    def test_valid_config(self):
        """Test valid configuration."""
        result = ValidationResult(valid=True)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_invalid_config(self):
        """Test invalid configuration."""
        from ui.models import ValidationError

        result = ValidationResult(
            valid=False,
            errors=[
                ValidationError(path="cameras", message="Required field missing"),
            ],
        )

        assert result.valid is False
        assert len(result.errors) == 1


# =============================================================================
# Pipeline Graph Component Tests
# =============================================================================

class TestPipelineGraphBuilder:
    """Test pipeline graph builder."""

    def test_build_simple_graph(self):
        """Build simple pipeline graph."""
        from ui.components.pipeline_graph import PipelineGraphBuilder

        graph = (
            PipelineGraphBuilder("test")
            .add_node("a", NodeType.INPUT, "Input")
            .add_node("b", NodeType.PROCESS, "Process")
            .add_node("c", NodeType.OUTPUT, "Output")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .build()
        )

        assert graph.name == "test"
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_default_pipeline_graph(self):
        """Get default pipeline graph."""
        from ui.components.pipeline_graph import get_default_pipeline_graph

        graph = get_default_pipeline_graph()

        assert graph.name == "main"
        assert len(graph.nodes) >= 5
        assert len(graph.edges) >= 5


class TestPipelineAnalyzer:
    """Test pipeline analyzer."""

    def test_topological_sort(self):
        """Test topological sort."""
        from ui.components.pipeline_graph import (
            PipelineAnalyzer,
            PipelineGraphBuilder,
        )

        graph = (
            PipelineGraphBuilder("test")
            .add_node("a", NodeType.INPUT, "A")
            .add_node("b", NodeType.PROCESS, "B")
            .add_node("c", NodeType.OUTPUT, "C")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .build()
        )

        analyzer = PipelineAnalyzer(graph)
        order = analyzer.topological_sort()

        # A must come before B, B before C
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_get_successors(self):
        """Test getting successors."""
        from ui.components.pipeline_graph import (
            PipelineAnalyzer,
            PipelineGraphBuilder,
        )

        graph = (
            PipelineGraphBuilder("test")
            .add_node("a", NodeType.INPUT, "A")
            .add_node("b", NodeType.PROCESS, "B")
            .add_node("c", NodeType.OUTPUT, "C")
            .add_edge("a", "b")
            .add_edge("a", "c")
            .build()
        )

        analyzer = PipelineAnalyzer(graph)
        successors = analyzer.get_successors("a")

        assert "b" in successors
        assert "c" in successors


class TestLayoutEngine:
    """Test layout engine."""

    def test_calculate_layout(self):
        """Test layout calculation."""
        from ui.components.pipeline_graph import (
            LayoutEngine,
            PipelineGraphBuilder,
        )

        graph = (
            PipelineGraphBuilder("test")
            .add_node("a", NodeType.INPUT, "A")
            .add_node("b", NodeType.PROCESS, "B")
            .add_edge("a", "b")
            .build()
        )

        engine = LayoutEngine(graph)
        layout = engine.calculate_layout()

        assert "a" in layout.positions
        assert "b" in layout.positions
        assert layout.width > 0
        assert layout.height > 0

    def test_to_json(self):
        """Test JSON export."""
        from ui.components.pipeline_graph import (
            LayoutEngine,
            PipelineGraphBuilder,
        )

        graph = (
            PipelineGraphBuilder("test")
            .add_node("a", NodeType.INPUT, "A")
            .add_edge("a", "a")  # Self-loop (not typical but tests export)
            .build()
        )

        engine = LayoutEngine(graph)
        json_data = engine.to_json()

        assert "nodes" in json_data
        assert "edges" in json_data
        assert "width" in json_data


# =============================================================================
# Frame Viewer Component Tests
# =============================================================================

class TestFrameAnnotator:
    """Test frame annotator."""

    def test_create_annotator(self):
        """Create frame annotator."""
        from ui.components.frame_viewer import FrameAnnotator

        annotator = FrameAnnotator()

        assert annotator.font_scale == 0.5
        assert annotator.line_thickness == 2

    def test_annotate_empty(self):
        """Annotate with no detections."""
        from ui.components.frame_viewer import Annotation, FrameAnnotator

        annotator = FrameAnnotator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotation = Annotation(boxes=[])

        result = annotator.annotate(frame, annotation)

        assert result.shape == (480, 640, 3)

    def test_class_colors(self):
        """Test class color palette."""
        from ui.components.frame_viewer import FrameAnnotator

        annotator = FrameAnnotator()

        assert "person" in annotator.class_colors
        assert "car" in annotator.class_colors
        assert "default" in annotator.class_colors


class TestFrameEncoder:
    """Test frame encoder."""

    def test_encode_jpeg(self):
        """Encode frame as JPEG."""
        pytest.importorskip("cv2")
        from ui.components.frame_viewer import FrameEncoder

        encoder = FrameEncoder(quality=80)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        jpeg_bytes = encoder.encode_jpeg(frame)

        assert len(jpeg_bytes) > 0
        # JPEG magic bytes
        assert jpeg_bytes[:2] == b'\xff\xd8'

    def test_encode_base64(self):
        """Encode frame as base64."""
        pytest.importorskip("cv2")
        from ui.components.frame_viewer import FrameEncoder

        encoder = FrameEncoder()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        b64 = encoder.encode_base64(frame)

        assert isinstance(b64, str)
        assert len(b64) > 0


class TestDepthVisualizer:
    """Test depth visualizer."""

    def test_colorize(self):
        """Colorize depth image."""
        pytest.importorskip("cv2")
        from ui.components.frame_viewer import DepthVisualizer

        viz = DepthVisualizer(min_depth=0.3, max_depth=5.0)
        depth = np.random.uniform(0.5, 4.0, (480, 640)).astype(np.float32)

        colored = viz.colorize(depth)

        assert colored.shape == (480, 640, 3)
        assert colored.dtype == np.uint8


# =============================================================================
# Metrics Panel Component Tests
# =============================================================================

class TestMetricCollector:
    """Test metric collector."""

    def test_record_and_get(self):
        """Record and retrieve metrics."""
        from ui.components.metrics_panel import MetricCollector

        collector = MetricCollector()
        collector.record("fps", 15.0)
        collector.record("fps", 16.0)
        collector.record("fps", 14.5)

        stats = collector.get_stats("fps")

        assert stats is not None
        assert stats.samples == 3
        assert 14.5 <= stats.mean <= 16.0

    def test_get_nonexistent(self):
        """Get stats for nonexistent metric."""
        from ui.components.metrics_panel import MetricCollector

        collector = MetricCollector()
        stats = collector.get_stats("nonexistent")

        assert stats is None


class TestFPSCounter:
    """Test FPS counter."""

    def test_count_frames(self):
        """Count frames."""
        from ui.components.metrics_panel import FPSCounter
        import time

        counter = FPSCounter()

        # Simulate frames
        for _ in range(10):
            counter.tick()
            time.sleep(0.01)

        assert counter.frame_count == 10
        assert counter.fps > 0

    def test_reset(self):
        """Reset counter."""
        from ui.components.metrics_panel import FPSCounter

        counter = FPSCounter()
        counter.tick()
        counter.tick()
        counter.reset()

        assert counter.frame_count == 0


class TestLatencyTracker:
    """Test latency tracker."""

    def test_record_latency(self):
        """Record operation latency."""
        from ui.components.metrics_panel import LatencyTracker

        tracker = LatencyTracker()
        tracker.record("segmentation", 30.5)
        tracker.record("segmentation", 32.0)
        tracker.record("segmentation", 29.5)

        stats = tracker.get_stats("segmentation")

        assert stats is not None
        assert stats["samples"] == 3
        assert 29.5 <= stats["mean"] <= 32.0

    def test_context_manager(self):
        """Use context manager for timing."""
        from ui.components.metrics_panel import LatencyTracker
        import time

        tracker = LatencyTracker()

        with tracker.start("test_op"):
            time.sleep(0.01)

        stats = tracker.get_stats("test_op")

        assert stats is not None
        assert stats["mean"] >= 10  # At least 10ms


class TestSystemMonitor:
    """Test system monitor."""

    def test_get_metrics(self):
        """Get system metrics."""
        from ui.components.metrics_panel import SystemMonitor

        monitor = SystemMonitor()
        metrics = monitor.get_metrics()

        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert metrics.memory_used_mb > 0

    def test_get_cpu_per_core(self):
        """Get per-core CPU usage."""
        from ui.components.metrics_panel import SystemMonitor

        monitor = SystemMonitor()
        per_core = monitor.get_cpu_per_core()

        assert isinstance(per_core, list)
        assert len(per_core) > 0


class TestMetricsAggregator:
    """Test metrics aggregator."""

    def test_record_frame(self):
        """Record frame metrics."""
        from ui.components.metrics_panel import MetricsAggregator

        aggregator = MetricsAggregator()
        aggregator.record_frame(
            objects_detected=5,
            latencies={"segmentation": 30.0, "tracking": 3.0},
        )

        metrics = aggregator.get_dashboard_metrics()

        assert "system" in metrics
        assert "processing" in metrics
        assert "latency" in metrics

    def test_get_history(self):
        """Get metric history."""
        from ui.components.metrics_panel import MetricsAggregator

        aggregator = MetricsAggregator()
        aggregator.collector.record("test", 1.0)
        aggregator.collector.record("test", 2.0)

        history = aggregator.get_history("test")

        assert len(history) == 2


class TestMetricsFormatter:
    """Test metrics formatter."""

    def test_format_bytes(self):
        """Format bytes."""
        from ui.components.metrics_panel import MetricsFormatter

        assert "1.0 KB" == MetricsFormatter.format_bytes(1024)
        assert "1.0 MB" == MetricsFormatter.format_bytes(1024 * 1024)

    def test_format_latency(self):
        """Format latency."""
        from ui.components.metrics_panel import MetricsFormatter

        assert "ms" in MetricsFormatter.format_latency(50.0)
        assert "s" in MetricsFormatter.format_latency(1500.0)

    def test_format_fps(self):
        """Format FPS."""
        from ui.components.metrics_panel import MetricsFormatter

        assert "15.0 FPS" == MetricsFormatter.format_fps(15.0)

    def test_format_percent(self):
        """Format percentage."""
        from ui.components.metrics_panel import MetricsFormatter

        assert "45.5%" == MetricsFormatter.format_percent(45.5)

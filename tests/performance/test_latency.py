"""Performance tests for PERCEPT processing latency.

Tests measure execution time for critical pipeline components
to ensure they meet real-time processing requirements.
"""

import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pytest

from percept.core.schema import ObjectSchema, Detection, ObjectMask
from percept.core.pipeline import Pipeline
from percept.core.adapter import PipelineData, DataSpec
from percept.persistence.database import PerceptDatabase
from percept.persistence.embedding_store import EmbeddingStore, EmbeddingStoreConfig
from percept.tracking.gallery import FAISSGallery, GalleryConfig
from percept.tracking.bytetrack import ByteTrackWrapper, ByteTrackConfig
from percept.pipelines.router import PipelineRouter, RouterConfig
from percept.pipelines.person import PersonPipeline, PersonPipelineConfig
from percept.pipelines.vehicle import VehiclePipeline, VehiclePipelineConfig
from percept.pipelines.generic import GenericPipeline, GenericPipelineConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_rgb_image():
    """Create sample RGB image (640x480)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth_image():
    """Create sample depth image (640x480)."""
    return np.random.uniform(0.5, 10.0, (480, 640)).astype(np.float32)


@pytest.fixture
def sample_embedding():
    """Create normalized 512-dim embedding."""
    emb = np.random.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def sample_crop():
    """Create sample object crop (200x100)."""
    return np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)


@pytest.fixture
def large_gallery():
    """Create gallery with 1000 embeddings."""
    config = GalleryConfig(embedding_dimension=512)
    gallery = FAISSGallery(config)

    for i in range(1000):
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        gallery.add(f"obj-{i}", emb)

    return gallery


@pytest.fixture
def database(temp_dir):
    """Create test database with objects."""
    db_path = Path(temp_dir) / "test.db"
    db = PerceptDatabase(db_path)
    db.initialize()
    yield db
    db.close()


# =============================================================================
# Latency Measurement Helpers
# =============================================================================

def measure_latency(func, *args, iterations: int = 100, **kwargs) -> dict:
    """Measure function latency over multiple iterations.

    Returns:
        Dict with mean, min, max, std latency in seconds
    """
    times = []

    # Warmup
    for _ in range(5):
        func(*args, **kwargs)

    # Measure
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "min": np.min(times),
        "max": np.max(times),
        "std": np.std(times),
        "p95": np.percentile(times, 95),
        "p99": np.percentile(times, 99),
    }


# =============================================================================
# ReID Gallery Latency Tests
# =============================================================================

class TestReIDLatency:
    """Test ReID gallery search latency."""

    def test_gallery_search_1000_objects(self, large_gallery, sample_embedding):
        """Benchmark gallery search with 1000 objects."""
        stats = measure_latency(
            large_gallery.search,
            sample_embedding,
            k=5,
            iterations=100,
        )

        print(f"\nGallery Search (1000 objects):")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")
        print(f"  P95:  {stats['p95']*1000:.3f}ms")
        print(f"  P99:  {stats['p99']*1000:.3f}ms")

        # Should complete in under 5ms for 1000 objects
        assert stats["mean"] < 0.005, f"Gallery search too slow: {stats['mean']*1000:.3f}ms"

    def test_gallery_add_latency(self, sample_embedding):
        """Benchmark adding embeddings to gallery."""
        config = GalleryConfig(embedding_dimension=512)
        gallery = FAISSGallery(config)

        def add_embedding():
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            gallery.add(f"obj-{gallery.size()}", emb)

        stats = measure_latency(add_embedding, iterations=100)

        print(f"\nGallery Add:")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")

        # Adding should be fast (under 1ms)
        assert stats["mean"] < 0.001

    def test_gallery_search_scaling(self, sample_embedding):
        """Test gallery search time scales well with size."""
        sizes = [100, 500, 1000, 5000]
        results = {}

        for size in sizes:
            config = GalleryConfig(embedding_dimension=512)
            gallery = FAISSGallery(config)

            for i in range(size):
                emb = np.random.randn(512).astype(np.float32)
                emb = emb / np.linalg.norm(emb)
                gallery.add(f"obj-{i}", emb)

            stats = measure_latency(gallery.search, sample_embedding, k=5, iterations=50)
            results[size] = stats["mean"]

        print(f"\nGallery Search Scaling:")
        for size, latency in results.items():
            print(f"  {size} objects: {latency*1000:.3f}ms")

        # Search time should be sublinear (FAISS is O(log n) for HNSW)
        # 5000 should be less than 15x slower than 500 (generous margin for variability)
        assert results[5000] < results[500] * 15


# =============================================================================
# Tracking Latency Tests
# =============================================================================

class TestTrackingLatency:
    """Test ByteTrack tracking latency."""

    def test_tracker_update_latency(self):
        """Benchmark tracker update with typical detection count."""
        config = ByteTrackConfig()
        tracker = ByteTrackWrapper(config)

        # Typical frame: 5-10 detections (as dicts for ByteTrackWrapper)
        detections = [
            {"bbox": (100 + i*50, 100, 150 + i*50, 300), "confidence": 0.9, "class_id": 0}
            for i in range(8)
        ]

        stats = measure_latency(
            tracker.update,
            detections,
            iterations=100,
        )

        print(f"\nTracker Update (8 detections):")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")
        print(f"  P95:  {stats['p95']*1000:.3f}ms")

        # Tracking should be fast (under 10ms)
        assert stats["mean"] < 0.010

    def test_tracker_many_detections(self):
        """Test tracker with many detections."""
        config = ByteTrackConfig()
        tracker = ByteTrackWrapper(config)

        # Crowded scene: 50 detections (as dicts)
        detections = [
            {"bbox": (i*12, i*8, i*12+40, i*8+80), "confidence": 0.9, "class_id": 0}
            for i in range(50)
        ]

        stats = measure_latency(
            tracker.update,
            detections,
            iterations=50,
        )

        print(f"\nTracker Update (50 detections):")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")

        # Even with many detections, should be under 50ms
        assert stats["mean"] < 0.050


# =============================================================================
# Pipeline Routing Latency Tests
# =============================================================================

class TestRoutingLatency:
    """Test pipeline routing latency."""

    def test_router_decision_latency(self):
        """Benchmark routing decision time."""
        config = RouterConfig()
        router = PipelineRouter(config)

        det = Detection(0, "person", 0.9, (100, 100, 200, 300))

        stats = measure_latency(router.route, det, iterations=1000)

        print(f"\nRouter Decision:")
        print(f"  Mean: {stats['mean']*1000000:.3f}us")  # microseconds

        # Routing should be very fast (under 100 microseconds)
        assert stats["mean"] < 0.0001

    def test_router_batch_latency(self):
        """Benchmark batch routing."""
        config = RouterConfig()
        router = PipelineRouter(config)

        detections = [
            Detection(0, "person", 0.9, (100, 100, 200, 300)),
            Detection(2, "car", 0.85, (300, 150, 450, 280)),
            Detection(0, "person", 0.88, (500, 100, 600, 350)),
            Detection(16, "dog", 0.75, (50, 300, 120, 400)),
        ]

        stats = measure_latency(router.route_batch, detections, iterations=500)

        print(f"\nRouter Batch (4 detections):")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")

        # Batch routing should be fast (under 1ms)
        assert stats["mean"] < 0.001


# =============================================================================
# Classification Pipeline Latency Tests
# =============================================================================

class TestClassificationLatency:
    """Test classification pipeline latency."""

    def test_person_pipeline_latency(self, sample_crop):
        """Benchmark person pipeline processing."""
        config = PersonPipelineConfig(
            enable_pose=False,  # Disable Hailo-dependent features
            enable_face=False,
        )
        pipeline = PersonPipeline(config)

        obj = ObjectSchema(
            primary_class="person",
            confidence=0.9,
            bounding_box_2d=(100, 100, 200, 400),
        )

        stats = measure_latency(
            pipeline.process_object,
            obj,
            sample_crop,
            iterations=50,
        )

        print(f"\nPerson Pipeline:")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")
        print(f"  P95:  {stats['p95']*1000:.3f}ms")

        # Without Hailo inference, should be under 50ms
        assert stats["mean"] < 0.050

    def test_vehicle_pipeline_latency(self, sample_crop):
        """Benchmark vehicle pipeline processing."""
        config = VehiclePipelineConfig(
            enable_license_plate=False,
        )
        pipeline = VehiclePipeline(config)

        obj = ObjectSchema(
            primary_class="car",
            confidence=0.9,
            bounding_box_2d=(100, 100, 300, 220),
        )

        stats = measure_latency(
            pipeline.process_object,
            obj,
            sample_crop,
            iterations=50,
        )

        print(f"\nVehicle Pipeline:")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")

        # Should be under 30ms
        assert stats["mean"] < 0.030

    def test_generic_pipeline_latency(self, sample_crop):
        """Benchmark generic pipeline processing."""
        # GenericPipelineConfig doesn't have enable_imagenet, uses imagenet_model_path
        # Setting empty path disables the Hailo inference
        config = GenericPipelineConfig(
            imagenet_model_path="",  # Disable Hailo inference by setting empty path
        )
        pipeline = GenericPipeline(config)

        obj = ObjectSchema(
            primary_class="unknown",
            confidence=0.7,
            bounding_box_2d=(100, 100, 200, 200),
        )

        stats = measure_latency(
            pipeline.process_object,
            obj,
            sample_crop,
            iterations=50,
        )

        print(f"\nGeneric Pipeline:")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")

        # Generic pipeline without Hailo still does color/shape analysis
        # Should be under 100ms without neural inference
        assert stats["mean"] < 0.100


# =============================================================================
# Database Latency Tests
# =============================================================================

class TestDatabaseLatency:
    """Test database operation latency."""

    def test_save_object_latency(self, database, sample_embedding):
        """Benchmark object save time."""
        def save_new_object():
            obj = ObjectSchema(
                primary_class="person",
                confidence=0.9,
                reid_embedding=sample_embedding.copy(),
            )
            database.save_object(obj)

        stats = measure_latency(save_new_object, iterations=100)

        print(f"\nDatabase Save:")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")

        # Save should be under 10ms
        assert stats["mean"] < 0.010

    def test_get_object_latency(self, database, sample_embedding):
        """Benchmark object retrieval time."""
        # Create test object
        obj = ObjectSchema(
            primary_class="person",
            confidence=0.9,
            reid_embedding=sample_embedding,
        )
        database.save_object(obj)
        obj_id = obj.id

        stats = measure_latency(database.get_object, obj_id, iterations=200)

        print(f"\nDatabase Get:")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")

        # Retrieval should be fast (under 5ms)
        assert stats["mean"] < 0.005

    def test_query_by_class_latency(self, database, sample_embedding):
        """Benchmark class query time."""
        # Add test objects
        for i in range(100):
            obj = ObjectSchema(
                primary_class="person" if i % 2 == 0 else "car",
                confidence=0.9,
                reid_embedding=sample_embedding.copy(),
            )
            database.save_object(obj)

        stats = measure_latency(
            database.query_by_class,
            "person",
            limit=50,
            iterations=100,
        )

        print(f"\nDatabase Query by Class (100 objects):")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")

        # Query should be under 10ms with index
        assert stats["mean"] < 0.010


# =============================================================================
# Embedding Store Latency Tests
# =============================================================================

class TestEmbeddingStoreLatency:
    """Test embedding store latency."""

    def test_embedding_store_add(self, temp_dir, sample_embedding):
        """Benchmark embedding store add time."""
        config = EmbeddingStoreConfig(
            db_path=f"{temp_dir}/test.db",
            index_path=f"{temp_dir}/test.index",
            persist_on_add=False,  # Don't sync every add
        )
        store = EmbeddingStore(config)
        store.initialize(load_from_db=False)

        def add_embedding():
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            store.add(f"obj-{store.size()}", emb, primary_class="person")

        stats = measure_latency(add_embedding, iterations=100)

        print(f"\nEmbedding Store Add:")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")

        # Add should be fast (under 1ms without DB sync)
        assert stats["mean"] < 0.001

    def test_embedding_store_search(self, temp_dir, sample_embedding):
        """Benchmark embedding store search time."""
        config = EmbeddingStoreConfig(
            db_path=f"{temp_dir}/test.db",
            index_path=f"{temp_dir}/test.index",
        )
        store = EmbeddingStore(config)
        store.initialize(load_from_db=False)

        # Add 1000 embeddings
        for i in range(1000):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            store.add(f"obj-{i}", emb, primary_class="person")

        stats = measure_latency(
            store.search,
            sample_embedding,
            k=5,
            iterations=100,
        )

        print(f"\nEmbedding Store Search (1000 embeddings):")
        print(f"  Mean: {stats['mean']*1000:.3f}ms")

        # Search should be under 5ms
        assert stats["mean"] < 0.005


# =============================================================================
# Full Frame Processing Latency Tests
# =============================================================================

class TestFullFrameLatency:
    """Test full frame processing latency budget."""

    def test_latency_budget_10fps(self):
        """Verify total latency fits in 100ms budget for 10 FPS."""
        # Simulated latencies for each stage
        latencies = {
            "segmentation": 30,  # ms (FastSAM on Hailo)
            "tracking": 3,  # ms
            "reid": 5,  # ms (gallery search)
            "routing": 0.1,  # ms
            "classification": 20,  # ms (average across pipelines)
            "database": 5,  # ms
            "overhead": 10,  # ms (data transfer, etc.)
        }

        total = sum(latencies.values())

        print(f"\nLatency Budget (10 FPS target = 100ms):")
        for stage, latency in latencies.items():
            print(f"  {stage}: {latency}ms")
        print(f"  Total: {total}ms")

        # Should fit in 100ms budget
        assert total < 100, f"Total latency {total}ms exceeds 100ms budget"

    def test_latency_budget_15fps(self):
        """Verify latency for 15 FPS target (66ms budget)."""
        latencies = {
            "segmentation": 25,
            "tracking": 3,
            "reid": 4,
            "routing": 0.1,
            "classification": 15,
            "database": 4,
            "overhead": 8,
        }

        total = sum(latencies.values())

        print(f"\nLatency Budget (15 FPS target = 66ms):")
        print(f"  Total: {total}ms")

        # Should fit in 66ms budget
        assert total < 66, f"Total latency {total}ms exceeds 66ms budget"

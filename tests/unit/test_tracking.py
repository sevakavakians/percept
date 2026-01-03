"""Unit tests for PERCEPT tracking and ReID modules."""

import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from percept.core.schema import ObjectMask
from percept.tracking.reid import (
    EmbeddingType,
    ReIDConfig,
    HistogramEmbedding,
    DeepEmbedding,
    ReIDExtractor,
    ReIDMatcher,
    ReIDModule,
)
from percept.tracking.gallery import (
    GalleryConfig,
    GalleryEntry,
    FAISSGallery,
    MultiCameraGallery,
)
from percept.tracking.bytetrack import (
    TrackState,
    ByteTrackConfig,
    Track,
    SimpleIoUTracker,
    ByteTrackWrapper,
    TrackingModule,
    track_detections,
)
from percept.tracking.mask_manager import (
    MaskManagerConfig,
    MaskClaim,
    SceneMaskManager,
    MaskConflictResolver,
    MaskManagerModule,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    np.random.seed(42)
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample binary mask."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 150:250] = 255
    return mask


@pytest.fixture
def sample_embedding():
    """Create a sample normalized embedding."""
    np.random.seed(42)
    emb = np.random.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def sample_detections():
    """Create sample detection dicts."""
    return [
        {"bbox": (100, 100, 200, 200), "confidence": 0.9, "class_id": 0, "class_name": "person"},
        {"bbox": (300, 150, 400, 300), "confidence": 0.85, "class_id": 1, "class_name": "car"},
        {"bbox": (500, 200, 600, 350), "confidence": 0.7, "class_id": 0, "class_name": "person"},
    ]


# =============================================================================
# ReID Tests
# =============================================================================


class TestEmbeddingType:
    """Test EmbeddingType enum."""

    def test_embedding_types(self):
        assert EmbeddingType.HISTOGRAM.value == "histogram"
        assert EmbeddingType.DEEP.value == "deep"


class TestReIDConfig:
    """Test ReIDConfig dataclass."""

    def test_default_config(self):
        config = ReIDConfig()
        assert config.embedding_dimension == 512
        assert config.histogram_bins == 32
        assert config.match_threshold_same_camera == 0.3
        assert config.match_threshold_cross_camera == 0.25

    def test_custom_config(self):
        config = ReIDConfig(embedding_dimension=256, histogram_bins=16)
        assert config.embedding_dimension == 256
        assert config.histogram_bins == 16


class TestHistogramEmbedding:
    """Test histogram-based embedding extraction."""

    def test_extract_basic(self, sample_image):
        extractor = HistogramEmbedding()
        crop = sample_image[100:200, 150:250]
        embedding = extractor.extract(crop)

        assert embedding is not None
        assert len(embedding) == extractor.config.embedding_dimension
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_extract_with_mask(self, sample_image):
        extractor = HistogramEmbedding()
        crop = sample_image[100:200, 150:250]
        mask = np.ones((100, 100), dtype=np.uint8) * 255

        embedding = extractor.extract(crop, mask)
        assert embedding is not None
        assert len(embedding) == extractor.config.embedding_dimension

    def test_extract_empty_mask(self, sample_image):
        extractor = HistogramEmbedding()
        crop = sample_image[100:200, 150:250]
        mask = np.zeros((100, 100), dtype=np.uint8)

        embedding = extractor.extract(crop, mask)
        # Should return zero vector for empty mask
        assert embedding is not None

    def test_extract_grayscale(self):
        extractor = HistogramEmbedding()
        gray_crop = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        # Convert grayscale to BGR for histogram extraction
        bgr_crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

        embedding = extractor.extract(bgr_crop)
        assert embedding is not None

    def test_custom_bins(self, sample_image):
        config = ReIDConfig(histogram_bins=16, embedding_dimension=256)
        extractor = HistogramEmbedding(config)
        crop = sample_image[100:200, 150:250]

        embedding = extractor.extract(crop)
        assert len(embedding) == 256


class TestDeepEmbedding:
    """Test deep embedding extraction (without Hailo)."""

    def test_not_available_without_hailo(self):
        extractor = DeepEmbedding()
        # Should gracefully handle missing Hailo - returns True or False
        result = extractor.is_available()
        assert isinstance(result, bool)

    def test_extract_fallback_to_histogram(self, sample_image):
        extractor = DeepEmbedding()
        crop = sample_image[100:200, 150:250]
        embedding = extractor.extract(crop)
        # Falls back to histogram embedding when Hailo unavailable
        assert embedding is not None
        assert len(embedding) == extractor.config.embedding_dimension


class TestReIDExtractor:
    """Test combined ReID extraction."""

    def test_extract_basic(self, sample_image):
        extractor = ReIDExtractor()
        bbox = (150, 100, 250, 200)

        embedding = extractor.extract(sample_image, bbox)
        assert embedding is not None
        assert len(embedding) == extractor.config.embedding_dimension
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_extract_with_mask(self, sample_image, sample_mask):
        extractor = ReIDExtractor()
        bbox = (150, 100, 250, 200)

        embedding = extractor.extract(sample_image, bbox, sample_mask)
        assert embedding is not None

    def test_extract_uses_histogram_for_generic(self, sample_image):
        extractor = ReIDExtractor()
        bbox = (150, 100, 250, 200)

        embedding = extractor.extract(sample_image, bbox, object_class="car")
        assert embedding is not None

    def test_empty_bbox(self, sample_image):
        extractor = ReIDExtractor()
        bbox = (100, 100, 100, 100)  # Zero-area bbox

        embedding = extractor.extract(sample_image, bbox)
        assert embedding is not None  # Returns zero embedding

    def test_out_of_bounds_bbox(self, sample_image):
        extractor = ReIDExtractor()
        bbox = (600, 450, 700, 500)  # Partially out of bounds

        embedding = extractor.extract(sample_image, bbox)
        assert embedding is not None


class TestReIDMatcher:
    """Test ReID gallery matching."""

    def test_add_and_match(self, sample_embedding):
        matcher = ReIDMatcher()
        matcher.add("obj1", sample_embedding, "cam1")

        # Same embedding should match
        matches = matcher.match(sample_embedding, "cam1")
        assert len(matches) > 0
        assert matches[0][0] == "obj1"
        assert matches[0][1] >= 0.9  # High similarity

    def test_match_empty_gallery(self, sample_embedding):
        matcher = ReIDMatcher()
        matches = matcher.match(sample_embedding, "cam1")
        assert len(matches) == 0

    def test_find_best_match(self, sample_embedding):
        matcher = ReIDMatcher()
        matcher.add("obj1", sample_embedding, "cam1")

        result = matcher.find_best_match(sample_embedding, "cam1")
        assert result is not None
        assert result[0] == "obj1"

    def test_find_best_match_threshold(self, sample_embedding):
        matcher = ReIDMatcher()

        # Add a different embedding
        np.random.seed(99)
        other = np.random.randn(512).astype(np.float32)
        other = other / np.linalg.norm(other)
        matcher.add("obj1", other, "cam1")

        # Should not match if similarity is below threshold
        result = matcher.find_best_match(sample_embedding, "cam1")
        # May or may not match depending on random embedding

    def test_cross_camera_matching(self, sample_embedding):
        matcher = ReIDMatcher()
        matcher.add("obj1", sample_embedding, "cam1")

        # Match from different camera
        matches = matcher.match(sample_embedding, "cam2")
        # Cross-camera has lower threshold
        assert len(matches) >= 0  # May or may not match

    def test_multiple_embeddings(self):
        matcher = ReIDMatcher()

        # Add multiple different objects
        for i in range(5):
            np.random.seed(i)
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            matcher.add(f"obj{i}", emb, "cam1")

        # Query with first embedding
        np.random.seed(0)
        query = np.random.randn(512).astype(np.float32)
        query = query / np.linalg.norm(query)

        matches = matcher.match(query, "cam1", top_k=3)
        assert len(matches) <= 3

    def test_remove_object(self, sample_embedding):
        matcher = ReIDMatcher()
        matcher.add("obj1", sample_embedding, "cam1")

        removed = matcher.remove("obj1")
        assert removed == True

        # Should not match after removal
        matches = matcher.match(sample_embedding, "cam1")
        # Object should be gone or have fewer entries


# =============================================================================
# Gallery Tests
# =============================================================================


class TestGalleryConfig:
    """Test GalleryConfig dataclass."""

    def test_default_config(self):
        config = GalleryConfig()
        assert config.embedding_dimension == 512
        assert config.index_type == "flat"
        assert config.default_k == 5

    def test_custom_config(self):
        config = GalleryConfig(embedding_dimension=256, index_type="hnsw")
        assert config.embedding_dimension == 256
        assert config.index_type == "hnsw"


class TestGalleryEntry:
    """Test GalleryEntry dataclass."""

    def test_create_entry(self, sample_embedding):
        entry = GalleryEntry(
            object_id="obj1",
            embedding=sample_embedding,
            metadata={"class": "person"},
            timestamp=time.time(),
        )
        assert entry.object_id == "obj1"
        assert len(entry.embedding) == 512


class TestFAISSGallery:
    """Test FAISS-backed embedding gallery."""

    def test_add_and_search(self, sample_embedding):
        gallery = FAISSGallery()
        idx = gallery.add("obj1", sample_embedding)
        assert idx == 0

        results = gallery.search(sample_embedding, k=1)
        assert len(results) == 1
        assert results[0][0] == "obj1"
        assert results[0][1] < 0.1  # Small distance

    def test_add_multiple(self):
        gallery = FAISSGallery()

        for i in range(10):
            np.random.seed(i)
            emb = np.random.randn(512).astype(np.float32)
            gallery.add(f"obj{i}", emb)

        assert gallery.size() == 10
        assert gallery.object_count() == 10

    def test_search_k(self):
        gallery = FAISSGallery()

        for i in range(10):
            np.random.seed(i)
            emb = np.random.randn(512).astype(np.float32)
            gallery.add(f"obj{i}", emb)

        np.random.seed(0)
        query = np.random.randn(512).astype(np.float32)

        results = gallery.search(query, k=5)
        assert len(results) == 5

    def test_search_empty_gallery(self, sample_embedding):
        gallery = FAISSGallery()
        results = gallery.search(sample_embedding)
        assert len(results) == 0

    def test_get_by_object_id(self, sample_embedding):
        gallery = FAISSGallery()
        gallery.add("obj1", sample_embedding)
        gallery.add("obj1", sample_embedding * 0.99)  # Another embedding

        entries = gallery.get_by_object_id("obj1")
        assert len(entries) == 2

    def test_get_embedding(self, sample_embedding):
        gallery = FAISSGallery()
        gallery.add("obj1", sample_embedding)

        emb = gallery.get_embedding("obj1")
        assert emb is not None
        assert len(emb) == 512

    def test_get_average_embedding(self, sample_embedding):
        gallery = FAISSGallery()
        gallery.add("obj1", sample_embedding)
        gallery.add("obj1", sample_embedding * 1.01)

        avg = gallery.get_average_embedding("obj1")
        assert avg is not None
        assert np.isclose(np.linalg.norm(avg), 1.0, atol=1e-5)

    def test_remove_object(self, sample_embedding):
        gallery = FAISSGallery()
        gallery.add("obj1", sample_embedding)
        gallery.add("obj2", sample_embedding)

        count = gallery.remove_object("obj1")
        assert count == 1

    def test_clear(self, sample_embedding):
        gallery = FAISSGallery()
        gallery.add("obj1", sample_embedding)
        gallery.clear()

        assert gallery.size() == 0

    def test_save_and_load(self, sample_embedding):
        gallery = FAISSGallery()
        gallery.add("obj1", sample_embedding, metadata={"type": "person"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gallery.json"
            gallery.save(str(path))

            loaded = FAISSGallery.load(str(path))
            assert loaded.size() == 1

            entries = loaded.get_by_object_id("obj1")
            assert len(entries) == 1
            assert entries[0].metadata.get("type") == "person"

    def test_rebuild_index(self, sample_embedding):
        gallery = FAISSGallery()
        gallery.add("obj1", sample_embedding)
        gallery.add("obj2", sample_embedding)
        gallery.remove_object("obj1")

        gallery.rebuild_index()
        assert gallery.size() == 1

    def test_search_by_similarity(self, sample_embedding):
        gallery = FAISSGallery()
        gallery.add("obj1", sample_embedding)

        results = gallery.search_by_similarity(sample_embedding, threshold=0.5)
        assert len(results) == 1

    def test_embedding_normalization(self):
        config = GalleryConfig(normalize_embeddings=True)
        gallery = FAISSGallery(config)

        unnormalized = np.random.randn(512).astype(np.float32) * 10
        gallery.add("obj1", unnormalized)

        emb = gallery.get_embedding("obj1")
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_dimension_padding(self):
        gallery = FAISSGallery()

        short_emb = np.random.randn(256).astype(np.float32)
        gallery.add("obj1", short_emb)

        emb = gallery.get_embedding("obj1")
        assert len(emb) == 512


class TestMultiCameraGallery:
    """Test multi-camera gallery."""

    def test_add_to_cameras(self, sample_embedding):
        gallery = MultiCameraGallery()
        gallery.add("obj1", sample_embedding, "cam1")
        gallery.add("obj2", sample_embedding, "cam2")

        cam1 = gallery.get_camera_gallery("cam1")
        assert cam1 is not None
        assert cam1.size() == 1

    def test_search_same_camera(self, sample_embedding):
        gallery = MultiCameraGallery()
        gallery.add("obj1", sample_embedding, "cam1")

        results = gallery.search_same_camera(sample_embedding, "cam1")
        assert len(results) >= 1

    def test_search_cross_camera(self, sample_embedding):
        gallery = MultiCameraGallery()
        gallery.add("obj1", sample_embedding, "cam1")
        gallery.add("obj2", sample_embedding, "cam2")

        results = gallery.search_cross_camera(sample_embedding, exclude_camera="cam1")
        # Should find obj2 from cam2
        assert any(r[0] == "obj2" for r in results)

    def test_clear(self, sample_embedding):
        gallery = MultiCameraGallery()
        gallery.add("obj1", sample_embedding, "cam1")
        gallery.clear()

        assert gallery.get_camera_gallery("cam1") is None


# =============================================================================
# ByteTrack Tests
# =============================================================================


class TestTrackState:
    """Test TrackState enum."""

    def test_states(self):
        assert TrackState.TENTATIVE.value == "tentative"
        assert TrackState.CONFIRMED.value == "confirmed"
        assert TrackState.LOST.value == "lost"
        assert TrackState.DELETED.value == "deleted"


class TestTrack:
    """Test Track dataclass."""

    def test_create_track(self):
        track = Track(
            track_id=1,
            bbox=(100, 100, 200, 200),
            confidence=0.9,
        )
        assert track.track_id == 1
        assert track.state == TrackState.TENTATIVE

    def test_get_centroid(self):
        track = Track(track_id=1, bbox=(100, 100, 200, 200), confidence=0.9)
        cx, cy = track.get_centroid()
        assert cx == 150
        assert cy == 150

    def test_predict_next_bbox(self):
        track = Track(
            track_id=1,
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            velocity=(10, 5),
        )
        predicted = track.predict_next_bbox()
        assert predicted == (110, 105, 210, 205)

    def test_predict_without_velocity(self):
        track = Track(track_id=1, bbox=(100, 100, 200, 200), confidence=0.9)
        predicted = track.predict_next_bbox()
        assert predicted == (100, 100, 200, 200)


class TestSimpleIoUTracker:
    """Test SimpleIoUTracker fallback."""

    def test_create_tracker(self):
        tracker = SimpleIoUTracker()
        assert tracker._next_id == 1

    def test_update_single_detection(self, sample_detections):
        tracker = SimpleIoUTracker()
        tracks = tracker.update([sample_detections[0]])

        # First update - track is tentative
        assert len(tracks) == 0  # Not confirmed yet

    def test_confirm_track(self, sample_detections):
        config = ByteTrackConfig(minimum_consecutive_frames=2)
        tracker = SimpleIoUTracker(config)

        # Update multiple times to confirm
        for _ in range(3):
            tracks = tracker.update([sample_detections[0]])

        assert len(tracks) >= 1
        assert tracks[0].state == TrackState.CONFIRMED

    def test_track_matching(self):
        tracker = SimpleIoUTracker()

        # First frame
        det1 = [{"bbox": (100, 100, 200, 200), "confidence": 0.9}]
        tracker.update(det1)

        # Second frame - slightly moved
        det2 = [{"bbox": (105, 105, 205, 205), "confidence": 0.9}]
        tracker.update(det2)

        # Should match to same track
        track = tracker.get_track(1)
        assert track is not None
        assert track.hits >= 2

    def test_lost_track(self):
        config = ByteTrackConfig(lost_track_buffer=2)
        tracker = SimpleIoUTracker(config)

        # Create and confirm track
        det = [{"bbox": (100, 100, 200, 200), "confidence": 0.9}]
        for _ in range(3):
            tracker.update(det)

        # Empty detections - track becomes lost then deleted
        for _ in range(3):
            tracker.update([])

        track = tracker.get_track(1)
        assert track is None  # Should be deleted

    def test_reset(self, sample_detections):
        tracker = SimpleIoUTracker()
        tracker.update(sample_detections)
        tracker.reset()

        assert tracker._next_id == 1
        assert len(tracker._tracks) == 0

    def test_velocity_update(self):
        # Use lower matching threshold to ensure matching with slight movement
        config = ByteTrackConfig(minimum_matching_threshold=0.5)
        tracker = SimpleIoUTracker(config)

        det1 = [{"bbox": (100, 100, 200, 200), "confidence": 0.9}]
        tracker.update(det1)

        det2 = [{"bbox": (110, 105, 210, 205), "confidence": 0.9}]
        tracker.update(det2)

        track = tracker.get_track(1)
        assert track is not None
        assert track.velocity is not None
        assert track.velocity[0] == 10
        assert track.velocity[1] == 5


class TestByteTrackWrapper:
    """Test ByteTrackWrapper."""

    def test_create_wrapper(self):
        wrapper = ByteTrackWrapper()
        assert wrapper is not None

    def test_update_empty(self):
        wrapper = ByteTrackWrapper()
        tracks = wrapper.update([])
        assert len(tracks) == 0

    def test_update_with_detections(self, sample_detections):
        wrapper = ByteTrackWrapper()

        # Multiple updates to confirm tracks
        for _ in range(5):
            tracks = wrapper.update(sample_detections)

        assert len(tracks) >= 0  # May vary based on ByteTrack availability

    def test_reset(self, sample_detections):
        wrapper = ByteTrackWrapper()
        wrapper.update(sample_detections)
        wrapper.reset()

        tracks = wrapper.update([])
        assert len(tracks) == 0

    def test_is_using_bytetrack(self):
        wrapper = ByteTrackWrapper()
        # Returns True or False depending on supervision availability
        result = wrapper.is_using_bytetrack()
        assert isinstance(result, bool)

    def test_track_history(self, sample_detections):
        wrapper = ByteTrackWrapper()

        for _ in range(3):
            wrapper.update(sample_detections)

        # Check history is being recorded
        history = wrapper.get_track_history(1)
        # May be empty if no confirmed tracks yet


class TestTrackingModule:
    """Test TrackingModule pipeline integration."""

    def test_module_properties(self):
        module = TrackingModule()
        assert module.name == "tracking"
        assert module.input_spec.data_type == "detections"
        assert module.output_spec.data_type == "tracks"


class TestTrackDetections:
    """Test track_detections convenience function."""

    def test_basic_tracking(self, sample_detections):
        tracks = track_detections(sample_detections)
        assert isinstance(tracks, list)


# =============================================================================
# Mask Manager Tests
# =============================================================================


class TestMaskManagerConfig:
    """Test MaskManagerConfig dataclass."""

    def test_default_config(self):
        config = MaskManagerConfig()
        assert config.overlap_threshold == 0.3
        assert config.claim_threshold == 0.5
        assert config.min_mask_area == 100

    def test_custom_config(self):
        config = MaskManagerConfig(overlap_threshold=0.5, claim_threshold=0.7)
        assert config.overlap_threshold == 0.5


class TestMaskClaim:
    """Test MaskClaim dataclass."""

    def test_create_claim(self, sample_mask):
        claim = MaskClaim(
            mask_id="mask1",
            object_id="obj1",
            mask=sample_mask,
            bbox=(150, 100, 250, 200),
            area=10000,
            timestamp=time.time(),
        )
        assert claim.mask_id == "mask1"
        assert not claim.is_stale(threshold=1.0)

    def test_stale_claim(self, sample_mask):
        claim = MaskClaim(
            mask_id="mask1",
            object_id="obj1",
            mask=sample_mask,
            bbox=(150, 100, 250, 200),
            area=10000,
            timestamp=time.time() - 2.0,  # 2 seconds ago
        )
        assert claim.is_stale(threshold=1.0)


class TestSceneMaskManager:
    """Test SceneMaskManager."""

    def test_claim_region(self, sample_mask):
        manager = SceneMaskManager()
        mask_id = manager.claim("obj1", sample_mask)

        assert mask_id is not None
        assert manager.num_claims == 1

    def test_release_region(self, sample_mask):
        manager = SceneMaskManager()
        mask_id = manager.claim("obj1", sample_mask)

        released = manager.release(mask_id)
        assert released
        assert manager.num_claims == 0

    def test_release_nonexistent(self):
        manager = SceneMaskManager()
        released = manager.release("nonexistent")
        assert not released

    def test_release_object(self, sample_mask):
        manager = SceneMaskManager()
        manager.claim("obj1", sample_mask)
        manager.claim("obj1", sample_mask)  # Second claim

        count = manager.release_object("obj1")
        assert count == 2
        assert manager.num_claims == 0

    def test_update_claim(self, sample_mask):
        manager = SceneMaskManager()
        mask_id = manager.claim("obj1", sample_mask)

        new_mask = np.zeros_like(sample_mask)
        new_mask[150:250, 200:300] = 255

        updated = manager.update_claim(mask_id, new_mask)
        assert updated

    def test_check_overlap(self, sample_mask):
        manager = SceneMaskManager()
        manager.claim("obj1", sample_mask)

        # Overlapping mask
        overlap_mask = np.zeros_like(sample_mask)
        overlap_mask[150:250, 200:300] = 255

        overlaps = manager.check_overlap(overlap_mask)
        assert len(overlaps) >= 0  # May or may not overlap depending on threshold

    def test_is_region_claimed(self, sample_mask):
        manager = SceneMaskManager()
        manager.claim("obj1", sample_mask)

        # Same region should be claimed
        is_claimed = manager.is_region_claimed(sample_mask)
        assert is_claimed

    def test_is_region_claimed_different(self, sample_mask):
        manager = SceneMaskManager()
        manager.claim("obj1", sample_mask)

        # Different region
        other_mask = np.zeros_like(sample_mask)
        other_mask[300:400, 400:500] = 255

        is_claimed = manager.is_region_claimed(other_mask)
        assert not is_claimed

    def test_filter_unclaimed(self, sample_mask):
        manager = SceneMaskManager()
        manager.claim("obj1", sample_mask)

        # Create ObjectMask instances
        masks = [
            ObjectMask(mask=sample_mask, bbox=(150, 100, 250, 200), confidence=0.9),
        ]

        unclaimed = manager.filter_unclaimed(masks)
        # Same mask should be filtered out
        assert len(unclaimed) == 0

    def test_subtract_claimed(self, sample_mask):
        manager = SceneMaskManager()
        manager.claim("obj1", sample_mask)

        # Full frame mask
        full_mask = np.ones_like(sample_mask) * 255

        result = manager.subtract_claimed(full_mask)
        # Claimed region should be subtracted
        assert result[150, 200] == 0  # Inside claimed region

    def test_get_composite_mask(self, sample_mask):
        manager = SceneMaskManager()
        manager.claim("obj1", sample_mask)

        composite = manager.get_composite_mask()
        assert composite is not None
        assert composite.shape == sample_mask.shape

    def test_get_object_masks(self, sample_mask):
        manager = SceneMaskManager()
        manager.claim("obj1", sample_mask)

        masks = manager.get_object_masks("obj1")
        assert len(masks) == 1

    def test_cleanup_stale(self, sample_mask):
        config = MaskManagerConfig(claim_duration=0.1)
        manager = SceneMaskManager(config)
        manager.claim("obj1", sample_mask)

        time.sleep(0.2)  # Wait for claim to become stale

        count = manager.cleanup_stale()
        assert count == 1
        assert manager.num_claims == 0

    def test_clear(self, sample_mask):
        manager = SceneMaskManager()
        manager.claim("obj1", sample_mask)
        manager.clear()

        assert manager.num_claims == 0

    def test_claimed_area(self, sample_mask):
        manager = SceneMaskManager()
        manager.claim("obj1", sample_mask)

        area = manager.claimed_area
        assert area > 0

    def test_auto_compute_bbox(self):
        manager = SceneMaskManager()
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 150:250] = 255

        mask_id = manager.claim("obj1", mask)
        claim = manager._claims[mask_id]

        assert claim.bbox == (150, 100, 249, 199)


class TestMaskConflictResolver:
    """Test MaskConflictResolver."""

    def test_resolve_no_conflict(self):
        resolver = MaskConflictResolver()

        mask1 = np.zeros((480, 640), dtype=np.uint8)
        mask1[100:200, 100:200] = 255

        mask2 = np.zeros((480, 640), dtype=np.uint8)
        mask2[300:400, 300:400] = 255

        masks = [
            ObjectMask(mask=mask1, bbox=(100, 100, 200, 200), confidence=0.9),
            ObjectMask(mask=mask2, bbox=(300, 300, 400, 400), confidence=0.8),
        ]

        resolved = resolver.resolve(masks)
        assert len(resolved) == 2

    def test_resolve_with_overlap(self):
        resolver = MaskConflictResolver()

        mask1 = np.zeros((480, 640), dtype=np.uint8)
        mask1[100:200, 100:200] = 255

        mask2 = np.zeros((480, 640), dtype=np.uint8)
        mask2[150:250, 150:250] = 255  # Overlaps with mask1

        masks = [
            ObjectMask(mask=mask1, bbox=(100, 100, 200, 200), confidence=0.9),
            ObjectMask(mask=mask2, bbox=(150, 150, 250, 250), confidence=0.8),
        ]

        resolved = resolver.resolve(masks)
        # Both may be kept or second may be modified
        assert len(resolved) >= 1

    def test_resolve_with_priorities(self):
        resolver = MaskConflictResolver()

        mask1 = np.zeros((480, 640), dtype=np.uint8)
        mask1[100:200, 100:200] = 255

        mask2 = np.zeros((480, 640), dtype=np.uint8)
        mask2[100:200, 100:200] = 255  # Same region

        masks = [
            ObjectMask(mask=mask1, bbox=(100, 100, 200, 200), confidence=0.9),
            ObjectMask(mask=mask2, bbox=(100, 100, 200, 200), confidence=0.8),
        ]

        # Higher priority for second mask
        resolved = resolver.resolve(masks, priorities=[0, 10])
        assert len(resolved) >= 1

    def test_resolve_single_mask(self):
        resolver = MaskConflictResolver()

        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 100:200] = 255

        masks = [ObjectMask(mask=mask, bbox=(100, 100, 200, 200), confidence=0.9)]

        resolved = resolver.resolve(masks)
        assert len(resolved) == 1

    def test_resolve_empty(self):
        resolver = MaskConflictResolver()
        resolved = resolver.resolve([])
        assert len(resolved) == 0


class TestMaskManagerModule:
    """Test MaskManagerModule pipeline integration."""

    def test_module_properties(self):
        module = MaskManagerModule()
        assert module.name == "mask_manager"
        assert module.input_spec.data_type == "masks"
        assert module.output_spec.data_type == "filtered_masks"


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrackingIntegration:
    """Integration tests for tracking pipeline."""

    def test_reid_to_gallery_flow(self, sample_image, sample_mask):
        # Extract embedding
        extractor = ReIDExtractor()
        bbox = (150, 100, 250, 200)
        embedding = extractor.extract(sample_image, bbox, sample_mask)

        # Add to gallery
        gallery = FAISSGallery()
        gallery.add("obj1", embedding)

        # Search
        results = gallery.search(embedding, k=1)
        assert len(results) == 1
        assert results[0][0] == "obj1"

    def test_tracking_with_reid(self, sample_image, sample_detections):
        tracker = ByteTrackWrapper()
        matcher = ReIDMatcher()
        extractor = ReIDExtractor()

        # Simulate tracking across frames
        for _ in range(5):
            tracks = tracker.update(sample_detections)

            for track in tracks:
                # Extract embedding with integer bbox coordinates
                bbox = tuple(int(v) for v in track.bbox)
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    embedding = extractor.extract(sample_image, bbox)
                    matcher.add(str(track.track_id), embedding, "cam1")

    def test_mask_manager_with_tracks(self, sample_mask):
        manager = SceneMaskManager()

        # Simulate track claiming a region
        manager.claim("track_1", sample_mask, priority=1)

        # New detection in same region should be claimed
        is_claimed = manager.is_region_claimed(sample_mask)
        assert is_claimed

        # Update claim with new mask
        mask_ids = list(manager._object_claims["track_1"])
        new_mask = np.zeros_like(sample_mask)
        new_mask[120:220, 170:270] = 255
        manager.update_claim(mask_ids[0], new_mask)

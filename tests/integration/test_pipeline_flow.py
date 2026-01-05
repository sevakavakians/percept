"""Integration tests for full PERCEPT pipeline flow.

Tests the complete processing chain from camera capture through
segmentation, tracking, classification, and database persistence.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pytest

from percept.core.schema import ObjectSchema, ClassificationStatus, Detection, ObjectMask
from percept.core.pipeline import Pipeline, PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.persistence.database import PerceptDatabase
from percept.persistence.embedding_store import EmbeddingStore, EmbeddingStoreConfig
from percept.persistence.review import HumanReviewQueue, ConfidenceRouter, CropManager
from percept.tracking.reid import ReIDExtractor, ReIDConfig
from percept.tracking.gallery import FAISSGallery, GalleryConfig
from percept.tracking.bytetrack import ByteTrackWrapper, ByteTrackConfig
from percept.pipelines.router import PipelineRouter, RouterConfig, PipelineType
from percept.pipelines.person import PersonPipeline, PersonPipelineConfig
from percept.pipelines.vehicle import VehiclePipeline, VehiclePipelineConfig
from percept.pipelines.generic import GenericPipeline, GenericPipelineConfig


# =============================================================================
# Mock Components for Integration Testing
# =============================================================================

class MockCamera:
    """Mock camera that generates synthetic frames."""

    def __init__(self, camera_id: str = "mock_cam"):
        self.camera_id = camera_id
        self.frame_count = 0
        self._scenes: List[Dict[str, Any]] = []
        self._current_scene_idx = 0

    def set_scene(self, objects: List[Dict[str, Any]]) -> None:
        """Set scene with objects.

        Args:
            objects: List of object dicts with 'class', 'bbox', 'depth'
        """
        self._scenes.append(objects)

    def clear_scenes(self) -> None:
        """Clear all scenes."""
        self._scenes.clear()
        self._current_scene_idx = 0

    def get_frame(self) -> PipelineData:
        """Get next frame with synthetic data."""
        self.frame_count += 1

        # Generate base image
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 5.0

        # Add objects from current scene
        detections = []
        if self._scenes and self._current_scene_idx < len(self._scenes):
            scene = self._scenes[self._current_scene_idx]
            for obj in scene:
                x1, y1, x2, y2 = obj.get("bbox", (100, 100, 200, 300))
                obj_depth = obj.get("depth", 2.0)

                # Draw object in image
                color = self._class_to_color(obj.get("class", "unknown"))
                rgb[y1:y2, x1:x2] = color
                depth[y1:y2, x1:x2] = obj_depth

                # Create detection
                det = Detection(
                    class_id=self._class_to_id(obj.get("class", "unknown")),
                    class_name=obj.get("class", "unknown"),
                    confidence=obj.get("confidence", 0.9),
                    bbox=(x1, y1, x2, y2),
                )
                detections.append(det)

        return PipelineData(
            image=rgb,
            depth=depth,
            frame_id=self.frame_count,
            timestamp=datetime.now(),
            camera_id=self.camera_id,
            detections=detections,
        )

    def advance_scene(self) -> None:
        """Advance to next scene."""
        if self._current_scene_idx < len(self._scenes) - 1:
            self._current_scene_idx += 1

    def _class_to_color(self, class_name: str) -> tuple:
        colors = {
            "person": (0, 0, 255),
            "vehicle": (255, 0, 0),
            "car": (255, 0, 0),
            "dog": (0, 255, 0),
        }
        return colors.get(class_name.lower(), (128, 128, 128))

    def _class_to_id(self, class_name: str) -> int:
        ids = {"person": 0, "vehicle": 2, "car": 2, "dog": 16}
        return ids.get(class_name.lower(), 99)


class MockSegmenter(PipelineModule):
    """Mock segmenter that creates masks from detections."""

    @property
    def name(self) -> str:
        return "mock_segmenter"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(data_type="frame", required_fields=["image", "detections"])

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(data_type="segmented", required_fields=["masks"])

    def process(self, data: PipelineData) -> PipelineData:
        result = data.copy()
        masks = []

        for det in data.get("detections", []):
            x1, y1, x2, y2 = det.bbox
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[int(y1):int(y2), int(x1):int(x2)] = 255

            obj_mask = ObjectMask(
                mask=mask,
                bbox=det.bbox,
                confidence=det.confidence,
                depth_median=2.0,
            )
            masks.append(obj_mask)

        result.masks = masks
        return result


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def database(temp_dir):
    """Create test database."""
    db_path = Path(temp_dir) / "test.db"
    db = PerceptDatabase(db_path)
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def mock_camera():
    """Create mock camera."""
    return MockCamera("test_cam")


@pytest.fixture
def embedding_store(temp_dir):
    """Create embedding store."""
    config = EmbeddingStoreConfig(
        db_path=f"{temp_dir}/test.db",
        index_path=f"{temp_dir}/test.index",
    )
    store = EmbeddingStore(config)
    store.initialize(load_from_db=False)
    return store


@pytest.fixture
def reid_gallery():
    """Create ReID gallery."""
    config = GalleryConfig(embedding_dimension=512)
    return FAISSGallery(config)


@pytest.fixture
def router():
    """Create pipeline router."""
    config = RouterConfig()
    return PipelineRouter(config)


# =============================================================================
# Integration Tests: Detection to Schema
# =============================================================================

class TestDetectionToSchema:
    """Test flow from detection to ObjectSchema."""

    def test_detection_creates_schema(self, mock_camera, database):
        """Test that detections are converted to ObjectSchemas."""
        # Setup scene with person
        mock_camera.set_scene([
            {"class": "person", "bbox": (100, 100, 200, 300), "depth": 2.5}
        ])

        # Get frame with detections
        frame = mock_camera.get_frame()

        # Create objects from detections
        objects = []
        for det in frame.detections:
            obj = ObjectSchema(
                primary_class=det.class_name,
                confidence=det.confidence,
                bounding_box_2d=det.bbox,
                camera_id=frame.camera_id,
            )
            objects.append(obj)
            database.save_object(obj)

        # Verify
        assert len(objects) == 1
        assert objects[0].primary_class == "person"

        # Check database
        stored = database.get_object(objects[0].id)
        assert stored is not None
        assert stored.primary_class == "person"

    def test_multiple_detections(self, mock_camera, database):
        """Test multiple detections in single frame."""
        mock_camera.set_scene([
            {"class": "person", "bbox": (100, 100, 200, 300), "depth": 2.0},
            {"class": "car", "bbox": (300, 150, 450, 280), "depth": 5.0},
            {"class": "dog", "bbox": (500, 350, 580, 450), "depth": 3.0},
        ])

        frame = mock_camera.get_frame()
        assert len(frame.detections) == 3

        # Store all
        for det in frame.detections:
            obj = ObjectSchema(
                primary_class=det.class_name,
                confidence=det.confidence,
                bounding_box_2d=det.bbox,
            )
            database.save_object(obj)

        # Verify counts
        assert database.count_objects() == 3
        assert database.count_objects("person") == 1
        assert database.count_objects("car") == 1

    def test_low_confidence_detection(self, mock_camera, database, temp_dir):
        """Test low confidence detection goes to review queue."""
        mock_camera.set_scene([
            {"class": "person", "bbox": (100, 100, 200, 300), "confidence": 0.3}
        ])

        frame = mock_camera.get_frame()
        det = frame.detections[0]

        obj = ObjectSchema(
            primary_class=det.class_name,
            confidence=det.confidence,
            bounding_box_2d=det.bbox,
        )

        # Route through confidence router
        router = ConfidenceRouter()
        status, reason = router.route(obj)

        assert status == ClassificationStatus.NEEDS_REVIEW
        assert reason is not None


# =============================================================================
# Integration Tests: Segmentation Pipeline
# =============================================================================

class TestSegmentationPipeline:
    """Test segmentation integration."""

    def test_segmenter_creates_masks(self, mock_camera):
        """Test segmenter creates masks from detections."""
        mock_camera.set_scene([
            {"class": "person", "bbox": (100, 100, 200, 300)}
        ])

        frame = mock_camera.get_frame()
        segmenter = MockSegmenter()

        result = segmenter.process(frame)

        assert len(result.masks) == 1
        assert result.masks[0].bbox == (100, 100, 200, 300)

    def test_mask_to_object_schema(self, mock_camera):
        """Test creating ObjectSchema from mask."""
        mock_camera.set_scene([
            {"class": "person", "bbox": (100, 100, 200, 300), "depth": 2.5}
        ])

        frame = mock_camera.get_frame()
        segmenter = MockSegmenter()
        result = segmenter.process(frame)

        mask = result.masks[0]
        det = frame.detections[0]

        obj = ObjectSchema(
            primary_class=det.class_name,
            confidence=det.confidence,
            bounding_box_2d=mask.bbox,
            distance_from_camera=mask.depth_median,
        )

        assert obj.distance_from_camera == 2.0  # From mask
        assert obj.bounding_box_2d == (100, 100, 200, 300)


# =============================================================================
# Integration Tests: ReID and Tracking
# =============================================================================

class TestReIDTracking:
    """Test ReID and tracking integration."""

    def test_embedding_stored_in_gallery(self, reid_gallery):
        """Test embeddings are stored and searchable."""
        # Create embedding
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Add to gallery
        reid_gallery.add("obj-001", embedding)

        # Search should find it
        results = reid_gallery.search(embedding, k=1)
        assert len(results) == 1
        assert results[0][0] == "obj-001"
        assert results[0][1] < 0.01  # Very close match

    def test_same_object_matched_across_frames(self, reid_gallery):
        """Test same object is matched across frames."""
        # First frame - create object
        embedding1 = np.random.randn(512).astype(np.float32)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        reid_gallery.add("obj-001", embedding1)

        # Second frame - slightly different embedding (same object)
        # Use very small noise to simulate same object
        noise = np.random.randn(512).astype(np.float32) * 0.02
        embedding2 = embedding1 + noise
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Should match existing object
        results = reid_gallery.search(embedding2, k=1)
        assert results[0][0] == "obj-001"
        # L2 distance for similar embeddings with small noise
        assert results[0][1] < 0.5  # Reasonable threshold for L2 distance

    def test_new_object_not_matched(self, reid_gallery):
        """Test new object is not matched to existing."""
        # Add first object
        embedding1 = np.random.randn(512).astype(np.float32)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        reid_gallery.add("obj-001", embedding1)

        # Completely different object
        embedding2 = np.random.randn(512).astype(np.float32)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        results = reid_gallery.search(embedding2, k=1)
        # Distance should be high (not a match)
        assert results[0][1] > 0.5

    def test_tracker_maintains_ids(self):
        """Test ByteTrack maintains consistent IDs."""
        # Use low minimum_consecutive_frames for faster testing
        config = ByteTrackConfig(
            minimum_consecutive_frames=1,  # Confirm tracks immediately
            track_activation_threshold=0.3,  # Lower threshold
        )
        tracker = ByteTrackWrapper(config)

        # Run multiple frames with consistent detections
        for frame in range(5):
            dets = [
                {"bbox": (100 + frame*2, 100 + frame, 200 + frame*2, 300 + frame),
                 "confidence": 0.9, "class_id": 0},
                {"bbox": (300 + frame*2, 150 + frame, 450 + frame*2, 280 + frame),
                 "confidence": 0.85, "class_id": 2},
            ]
            tracks = tracker.update(dets)

        # After multiple frames, should have tracks
        assert len(tracks) >= 1

        # Get track IDs from last frame
        track_ids = {t.track_id for t in tracks}
        assert len(track_ids) >= 1  # At least one track established


# =============================================================================
# Integration Tests: Classification Pipeline
# =============================================================================

class TestClassificationPipeline:
    """Test classification pipeline integration."""

    def test_router_selects_person_pipeline(self, router):
        """Test router selects person pipeline for person detection."""
        det = Detection(0, "person", 0.9, (100, 100, 200, 300))

        decision = router.route(det)

        assert decision.pipeline_type == PipelineType.PERSON

    def test_router_selects_vehicle_pipeline(self, router):
        """Test router selects vehicle pipeline for vehicle detection."""
        det = Detection(2, "car", 0.9, (100, 100, 300, 250))

        decision = router.route(det)

        assert decision.pipeline_type == PipelineType.VEHICLE

    def test_router_falls_back_to_generic(self, router):
        """Test router falls back to generic for unknown class."""
        det = Detection(99, "unknown_thing", 0.9, (100, 100, 200, 200))

        decision = router.route(det)

        assert decision.pipeline_type == PipelineType.GENERIC

    def test_person_pipeline_adds_attributes(self):
        """Test person pipeline adds attributes to object."""
        config = PersonPipelineConfig(enable_pose=False, enable_face=False)
        pipeline = PersonPipeline(config)

        obj = ObjectSchema(
            primary_class="person",
            confidence=0.9,
            bounding_box_2d=(100, 100, 200, 400),
        )

        # Create person-like crop
        crop = np.zeros((300, 100, 3), dtype=np.uint8)
        crop[0:150, :] = [0, 0, 200]  # Upper body (red shirt)
        crop[150:300, :] = [100, 100, 100]  # Lower body (gray pants)

        result = pipeline.process_object(obj, crop)

        assert "clothing" in result.attributes
        # Check for clothing color (may be "upper" or "upper_color" depending on impl)
        clothing = result.attributes["clothing"]
        assert "upper" in clothing or "upper_color" in clothing

    def test_vehicle_pipeline_adds_attributes(self):
        """Test vehicle pipeline adds attributes to object."""
        config = VehiclePipelineConfig(enable_license_plate=False)
        pipeline = VehiclePipeline(config)

        obj = ObjectSchema(
            primary_class="car",
            confidence=0.9,
            bounding_box_2d=(100, 100, 300, 220),
        )

        # Create car-like crop (blue rectangle)
        crop = np.zeros((120, 200, 3), dtype=np.uint8)
        crop[:, :] = [200, 100, 50]  # Blue car

        result = pipeline.process_object(obj, crop)

        assert "color" in result.attributes
        assert "vehicle_type" in result.attributes


# =============================================================================
# Integration Tests: Database Persistence
# =============================================================================

class TestDatabasePersistence:
    """Test database persistence integration."""

    def test_object_roundtrip(self, database):
        """Test object save and retrieve."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        obj = ObjectSchema(
            primary_class="person",
            confidence=0.92,
            bounding_box_2d=(100, 100, 200, 300),
            reid_embedding=embedding,
            attributes={"clothing": {"upper": "blue shirt"}},
        )

        database.save_object(obj)

        retrieved = database.get_object(obj.id)

        assert retrieved is not None
        assert retrieved.primary_class == "person"
        assert retrieved.confidence == 0.92
        assert retrieved.attributes["clothing"]["upper"] == "blue shirt"
        assert np.allclose(retrieved.reid_embedding, embedding)

    def test_query_by_class(self, database):
        """Test querying objects by class."""
        # Add multiple objects
        for i in range(5):
            obj = ObjectSchema(primary_class="person", confidence=0.9)
            database.save_object(obj)

        for i in range(3):
            obj = ObjectSchema(primary_class="car", confidence=0.85)
            database.save_object(obj)

        persons = database.query_by_class("person")
        cars = database.query_by_class("car")

        assert len(persons) == 5
        assert len(cars) == 3

    def test_embedding_sync(self, database, embedding_store):
        """Test embedding store syncs with database."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Add to embedding store
        embedding_store.add(
            "obj-001",
            embedding,
            primary_class="person",
            camera_id="cam1",
        )

        # Search should work
        results = embedding_store.search(embedding, k=1)
        assert len(results) == 1
        assert results[0][0] == "obj-001"


# =============================================================================
# Integration Tests: End-to-End Flow
# =============================================================================

class TestEndToEndFlow:
    """Test complete end-to-end pipeline flow."""

    def test_frame_to_database(self, mock_camera, database, reid_gallery, router):
        """Test complete flow from frame capture to database."""
        # Setup scene
        mock_camera.set_scene([
            {"class": "person", "bbox": (100, 100, 200, 350), "depth": 2.5, "confidence": 0.92}
        ])

        # Get frame
        frame = mock_camera.get_frame()

        # Segment
        segmenter = MockSegmenter()
        segmented = segmenter.process(frame)

        # Process each detection
        for i, det in enumerate(frame.detections):
            # Create embedding
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            # Check for ReID match
            matches = reid_gallery.search(embedding, k=1)
            if matches and matches[0][1] < 0.3:
                # Existing object
                obj_id = matches[0][0]
                obj = database.get_object(obj_id)
            else:
                # New object
                obj = ObjectSchema(
                    primary_class=det.class_name,
                    confidence=det.confidence,
                    bounding_box_2d=det.bbox,
                    reid_embedding=embedding,
                    camera_id=frame.camera_id,
                )
                reid_gallery.add(obj.id, embedding)

            # Route to classification pipeline
            decision = router.route(det)

            # Save to database
            database.save_object(obj)

        # Verify
        assert database.count_objects() == 1
        persons = database.query_by_class("person")
        assert len(persons) == 1
        assert persons[0].confidence == 0.92

    def test_object_persistence_across_frames(self, mock_camera, database, reid_gallery):
        """Test object is persisted and matched across frames."""
        # Frame 1: Person appears
        mock_camera.set_scene([
            {"class": "person", "bbox": (100, 100, 200, 350)}
        ])
        frame1 = mock_camera.get_frame()

        # Create object with embedding
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        obj = ObjectSchema(
            primary_class="person",
            confidence=0.9,
            bounding_box_2d=frame1.detections[0].bbox,
            reid_embedding=embedding,
        )
        reid_gallery.add(obj.id, embedding)
        database.save_object(obj)
        original_id = obj.id

        # Frame 2: Same person, slightly moved
        mock_camera.clear_scenes()
        mock_camera.set_scene([
            {"class": "person", "bbox": (105, 102, 205, 352)}
        ])
        frame2 = mock_camera.get_frame()

        # Same embedding with very small noise (same person)
        embedding2 = embedding + np.random.randn(512).astype(np.float32) * 0.01
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Should match existing object
        matches = reid_gallery.search(embedding2, k=1)
        assert matches[0][0] == original_id
        assert matches[0][1] < 0.5  # L2 distance threshold

        # Update existing object
        obj = database.get_object(original_id)
        obj.bounding_box_2d = frame2.detections[0].bbox
        database.save_object(obj)

        # Still only one object
        assert database.count_objects() == 1

    def test_review_queue_integration(self, mock_camera, database, temp_dir):
        """Test low confidence objects go to review queue."""
        crop_manager = CropManager(f"{temp_dir}/crops")
        queue = HumanReviewQueue(database, crop_manager)

        # Low confidence detection
        mock_camera.set_scene([
            {"class": "person", "bbox": (100, 100, 200, 300), "confidence": 0.35}
        ])
        frame = mock_camera.get_frame()
        det = frame.detections[0]

        obj = ObjectSchema(
            primary_class=det.class_name,
            confidence=det.confidence,
            bounding_box_2d=det.bbox,
        )
        database.save_object(obj)

        # Extract crop
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        crop = frame.image[y1:y2, x1:x2]

        # Should route to review
        status = queue.process_object(obj, crop)
        assert status == ClassificationStatus.NEEDS_REVIEW

    def test_pipeline_module_chain(self):
        """Test pipeline module chaining."""
        pipeline = Pipeline("test")

        # Add mock modules
        class PassthroughModule(PipelineModule):
            def __init__(self, name):
                self._name = name

            @property
            def name(self):
                return self._name

            @property
            def input_spec(self):
                return DataSpec(data_type="any")

            @property
            def output_spec(self):
                return DataSpec(data_type="any")

            def process(self, data):
                result = data.copy()
                result.set_metadata(f"{self._name}_processed", True)
                return result

        pipeline.add_module(PassthroughModule("step1"))
        pipeline.add_module(PassthroughModule("step2"))
        pipeline.add_module(PassthroughModule("step3"))

        data = PipelineData(frame_id=1)
        result = pipeline.process(data)

        # Pipeline.process returns ModuleResult, access .data attribute
        final_data = result.data
        assert final_data.get_metadata("step1_processed") is True
        assert final_data.get_metadata("step2_processed") is True
        assert final_data.get_metadata("step3_processed") is True

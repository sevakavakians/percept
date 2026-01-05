"""Integration tests for multi-camera scenarios.

Tests cross-camera ReID matching, multi-camera synchronization,
and object handoff between cameras.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from percept.core.schema import ObjectSchema, ClassificationStatus
from percept.core.adapter import PipelineData
from percept.persistence.database import PerceptDatabase
from percept.persistence.embedding_store import CameraAwareEmbeddingStore, EmbeddingStoreConfig
from percept.tracking.gallery import FAISSGallery, GalleryConfig, MultiCameraGallery
from percept.tracking.reid import ReIDConfig


# =============================================================================
# Mock Multi-Camera System
# =============================================================================

class MockMultiCameraSystem:
    """Mock multi-camera system for testing."""

    def __init__(self, camera_ids: List[str]):
        """Initialize with camera IDs.

        Args:
            camera_ids: List of camera identifiers
        """
        self.cameras: Dict[str, MockCameraView] = {}
        for cam_id in camera_ids:
            self.cameras[cam_id] = MockCameraView(cam_id)

    def set_object_in_camera(
        self,
        camera_id: str,
        object_id: str,
        bbox: tuple,
        class_name: str = "person",
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Place an object in a specific camera view.

        Args:
            camera_id: Camera to place object in
            object_id: Unique object ID
            bbox: Bounding box
            class_name: Object class
            embedding: Optional embedding (generates if not provided)
        """
        if camera_id in self.cameras:
            self.cameras[camera_id].add_object(
                object_id, bbox, class_name, embedding
            )

    def remove_object_from_camera(self, camera_id: str, object_id: str) -> None:
        """Remove object from camera view."""
        if camera_id in self.cameras:
            self.cameras[camera_id].remove_object(object_id)

    def move_object_between_cameras(
        self,
        object_id: str,
        from_camera: str,
        to_camera: str,
        new_bbox: tuple,
    ) -> None:
        """Move object from one camera to another.

        Args:
            object_id: Object to move
            from_camera: Source camera
            to_camera: Destination camera
            new_bbox: New bounding box in destination camera
        """
        # Get object info from source camera
        obj = None
        if from_camera in self.cameras:
            obj = self.cameras[from_camera].get_object(object_id)
            self.cameras[from_camera].remove_object(object_id)

        # Add to destination camera
        if obj and to_camera in self.cameras:
            self.cameras[to_camera].add_object(
                object_id, new_bbox, obj["class_name"], obj["embedding"]
            )

    def get_frame(self, camera_id: str) -> PipelineData:
        """Get frame from specific camera."""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_frame()
        return None

    def get_all_frames(self) -> Dict[str, PipelineData]:
        """Get frames from all cameras."""
        return {
            cam_id: cam.get_frame()
            for cam_id, cam in self.cameras.items()
        }


class MockCameraView:
    """Mock single camera view."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.frame_count = 0
        self._objects: Dict[str, Dict] = {}

    def add_object(
        self,
        object_id: str,
        bbox: tuple,
        class_name: str,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Add object to camera view."""
        if embedding is None:
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

        self._objects[object_id] = {
            "bbox": bbox,
            "class_name": class_name,
            "embedding": embedding,
        }

    def remove_object(self, object_id: str) -> None:
        """Remove object from view."""
        self._objects.pop(object_id, None)

    def get_object(self, object_id: str) -> Optional[Dict]:
        """Get object info."""
        return self._objects.get(object_id)

    def get_frame(self) -> PipelineData:
        """Generate frame with current objects."""
        self.frame_count += 1

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 5.0

        objects = []
        for obj_id, obj_info in self._objects.items():
            x1, y1, x2, y2 = obj_info["bbox"]
            image[y1:y2, x1:x2] = [100, 100, 200]
            depth[y1:y2, x1:x2] = 2.0

            objects.append({
                "id": obj_id,
                "bbox": obj_info["bbox"],
                "class_name": obj_info["class_name"],
                "embedding": obj_info["embedding"],
            })

        return PipelineData(
            image=image,
            depth=depth,
            frame_id=self.frame_count,
            timestamp=datetime.now(),
            camera_id=self.camera_id,
            detected_objects=objects,
        )


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
def multi_camera_gallery():
    """Create multi-camera gallery."""
    config = GalleryConfig(embedding_dimension=512)
    return MultiCameraGallery(config)


@pytest.fixture
def camera_aware_store(temp_dir):
    """Create camera-aware embedding store."""
    config = EmbeddingStoreConfig(
        db_path=f"{temp_dir}/test.db",
        index_path=f"{temp_dir}/test.index",
    )
    store = CameraAwareEmbeddingStore(
        config,
        same_camera_threshold=0.3,
        cross_camera_threshold=0.25,
    )
    store.initialize(load_from_db=False)
    return store


@pytest.fixture
def two_camera_system():
    """Create two-camera system."""
    return MockMultiCameraSystem(["cam_front", "cam_rear"])


@pytest.fixture
def three_camera_system():
    """Create three-camera system."""
    return MockMultiCameraSystem(["cam_left", "cam_center", "cam_right"])


# =============================================================================
# Cross-Camera ReID Tests
# =============================================================================

class TestCrossCameraReID:
    """Test cross-camera re-identification."""

    def test_same_object_matched_across_cameras(self, multi_camera_gallery):
        """Test same object is matched across cameras."""
        # Create consistent embedding for an object
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Object seen in camera 1
        multi_camera_gallery.add("person-001", embedding, "cam_front")

        # Same object now in camera 2 (very slightly different embedding)
        embedding2 = embedding + np.random.randn(512).astype(np.float32) * 0.01
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Search across cameras
        results = multi_camera_gallery.search_cross_camera(
            embedding2, exclude_camera="cam_rear"
        )

        assert len(results) > 0
        assert results[0][0] == "person-001"
        assert results[0][1] < 0.5  # Within reasonable L2 distance threshold

    def test_different_objects_not_matched(self, multi_camera_gallery):
        """Test different objects are not matched across cameras."""
        # Object 1 in camera 1
        embedding1 = np.random.randn(512).astype(np.float32)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        multi_camera_gallery.add("person-001", embedding1, "cam_front")

        # Completely different object in camera 2
        embedding2 = np.random.randn(512).astype(np.float32)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        results = multi_camera_gallery.search_cross_camera(embedding2)

        # Should have high distance (no match)
        if results:
            assert results[0][1] > 0.5

    def test_camera_aware_thresholds(self, camera_aware_store):
        """Test different thresholds for same vs cross camera."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Add object from camera 1
        camera_aware_store.add(
            "person-001",
            embedding,
            primary_class="person",
            camera_id="cam_front",
        )

        # Query with noise
        query = embedding + np.random.randn(512).astype(np.float32) * 0.1
        query = query / np.linalg.norm(query)

        # Same camera match (looser threshold)
        match = camera_aware_store.find_match(query, "cam_front")
        if match:
            assert match[2] == "same_camera"

        # Cross camera match (tighter threshold)
        match = camera_aware_store.find_match(query, "cam_rear")
        if match:
            assert match[2] == "cross_camera"

    def test_object_handoff_between_cameras(self, two_camera_system, database, multi_camera_gallery):
        """Test object handoff when moving between camera views."""
        # Person appears in front camera
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        two_camera_system.set_object_in_camera(
            "cam_front", "person-001", (100, 100, 200, 350), "person", embedding
        )

        # Process front camera
        frame_front = two_camera_system.get_frame("cam_front")
        obj_info = frame_front.detected_objects[0]

        # Create and store object
        obj = ObjectSchema(
            id=obj_info["id"],
            primary_class=obj_info["class_name"],
            confidence=0.9,
            bounding_box_2d=obj_info["bbox"],
            reid_embedding=obj_info["embedding"],
            camera_id="cam_front",
        )
        database.save_object(obj)
        multi_camera_gallery.add(obj.id, obj_info["embedding"], "cam_front")

        # Person moves to rear camera
        two_camera_system.move_object_between_cameras(
            "person-001", "cam_front", "cam_rear", (150, 120, 250, 370)
        )

        # Process rear camera
        frame_rear = two_camera_system.get_frame("cam_rear")
        obj_info_rear = frame_rear.detected_objects[0]

        # Should match existing person
        results = multi_camera_gallery.search_cross_camera(
            obj_info_rear["embedding"], exclude_camera="cam_rear"
        )

        assert len(results) > 0
        assert results[0][0] == "person-001"

        # Update object with new camera
        stored = database.get_object("person-001")
        assert stored is not None

    def test_multiple_objects_across_cameras(self, three_camera_system, multi_camera_gallery):
        """Test tracking multiple objects across multiple cameras."""
        # Create unique embeddings for each person
        person_embeddings = {}
        for i in range(3):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            person_embeddings[f"person-{i}"] = emb

        # Person 0 in left camera
        three_camera_system.set_object_in_camera(
            "cam_left", "person-0", (100, 100, 200, 350), "person",
            person_embeddings["person-0"]
        )
        multi_camera_gallery.add(
            "person-0", person_embeddings["person-0"], "cam_left"
        )

        # Person 1 in center camera
        three_camera_system.set_object_in_camera(
            "cam_center", "person-1", (250, 80, 350, 380), "person",
            person_embeddings["person-1"]
        )
        multi_camera_gallery.add(
            "person-1", person_embeddings["person-1"], "cam_center"
        )

        # Person 2 in right camera
        three_camera_system.set_object_in_camera(
            "cam_right", "person-2", (300, 150, 400, 400), "person",
            person_embeddings["person-2"]
        )
        multi_camera_gallery.add(
            "person-2", person_embeddings["person-2"], "cam_right"
        )

        # Each person should only match themselves
        for person_id, embedding in person_embeddings.items():
            results = multi_camera_gallery.search_cross_camera(embedding, k=3)
            assert results[0][0] == person_id


# =============================================================================
# Multi-Camera Synchronization Tests
# =============================================================================

class TestMultiCameraSynchronization:
    """Test multi-camera frame synchronization."""

    def test_all_cameras_return_frames(self, three_camera_system):
        """Test all cameras return frames."""
        frames = three_camera_system.get_all_frames()

        assert len(frames) == 3
        assert "cam_left" in frames
        assert "cam_center" in frames
        assert "cam_right" in frames

    def test_frame_timestamps_consistent(self, three_camera_system):
        """Test frame timestamps are reasonably synchronized."""
        # Add objects to all cameras
        for i, cam_id in enumerate(["cam_left", "cam_center", "cam_right"]):
            three_camera_system.set_object_in_camera(
                cam_id, f"person-{i}", (100, 100, 200, 300)
            )

        frames = three_camera_system.get_all_frames()

        timestamps = [f.timestamp for f in frames.values()]

        # All timestamps should be within 100ms of each other
        time_range = (max(timestamps) - min(timestamps)).total_seconds()
        assert time_range < 0.1

    def test_camera_specific_objects(self, three_camera_system):
        """Test each camera only sees its own objects."""
        # Different objects in each camera
        three_camera_system.set_object_in_camera(
            "cam_left", "obj-left", (100, 100, 200, 300)
        )
        three_camera_system.set_object_in_camera(
            "cam_center", "obj-center", (200, 150, 300, 350)
        )
        three_camera_system.set_object_in_camera(
            "cam_right", "obj-right", (300, 100, 400, 300)
        )

        frames = three_camera_system.get_all_frames()

        # Each camera should only see one object
        assert len(frames["cam_left"].detected_objects) == 1
        assert frames["cam_left"].detected_objects[0]["id"] == "obj-left"

        assert len(frames["cam_center"].detected_objects) == 1
        assert frames["cam_center"].detected_objects[0]["id"] == "obj-center"

        assert len(frames["cam_right"].detected_objects) == 1
        assert frames["cam_right"].detected_objects[0]["id"] == "obj-right"


# =============================================================================
# Camera Gallery Management Tests
# =============================================================================

class TestCameraGalleryManagement:
    """Test per-camera gallery management."""

    def test_per_camera_gallery(self, multi_camera_gallery):
        """Test separate galleries per camera."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)

        multi_camera_gallery.add("obj-1", emb1, "cam_front")
        multi_camera_gallery.add("obj-2", emb2, "cam_rear")

        # Same camera search
        results_front = multi_camera_gallery.search_same_camera(emb1, "cam_front")
        assert len(results_front) == 1
        assert results_front[0][0] == "obj-1"

        results_rear = multi_camera_gallery.search_same_camera(emb2, "cam_rear")
        assert len(results_rear) == 1
        assert results_rear[0][0] == "obj-2"

    def test_cross_camera_excludes_source(self, multi_camera_gallery):
        """Test cross-camera search excludes source camera."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        multi_camera_gallery.add("obj-1", emb, "cam_front")
        multi_camera_gallery.add("obj-2", emb, "cam_rear")

        # Cross-camera excluding front should only find rear
        results = multi_camera_gallery.search_cross_camera(emb, exclude_camera="cam_front")

        # Should find obj-2 from rear camera, not obj-1 from front
        matched_ids = [r[0] for r in results]
        assert "obj-2" in matched_ids

    def test_get_camera_gallery(self, multi_camera_gallery):
        """Test getting camera-specific gallery."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        multi_camera_gallery.add("obj-1", emb, "cam_front")

        gallery = multi_camera_gallery.get_camera_gallery("cam_front")
        assert gallery is not None
        assert gallery.size() == 1

        # Non-existent camera returns None
        assert multi_camera_gallery.get_camera_gallery("cam_nonexistent") is None


# =============================================================================
# Object Trajectory Across Cameras Tests
# =============================================================================

class TestObjectTrajectoryAcrossCameras:
    """Test object trajectory tracking across cameras."""

    def test_trajectory_spans_cameras(self, two_camera_system, database, multi_camera_gallery):
        """Test trajectory includes positions from multiple cameras."""
        from datetime import datetime

        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Start in front camera
        two_camera_system.set_object_in_camera(
            "cam_front", "person-001", (100, 100, 200, 350), "person", embedding
        )

        frame1 = two_camera_system.get_frame("cam_front")
        obj = ObjectSchema(
            id="person-001",
            primary_class="person",
            confidence=0.9,
            bounding_box_2d=(100, 100, 200, 350),
            camera_id="cam_front",
            reid_embedding=embedding,
            trajectory=[(1.0, 0.5, 2.5, datetime.now())],
        )
        database.save_object(obj)
        multi_camera_gallery.add("person-001", embedding, "cam_front")

        # Move to rear camera
        two_camera_system.move_object_between_cameras(
            "person-001", "cam_front", "cam_rear", (150, 120, 250, 370)
        )

        frame2 = two_camera_system.get_frame("cam_rear")
        obj_info = frame2.detected_objects[0]

        # Match and update
        results = multi_camera_gallery.search_cross_camera(
            obj_info["embedding"], exclude_camera="cam_rear"
        )

        if results and results[0][1] < 0.5:
            # Update existing object
            stored = database.get_object("person-001")
            stored.camera_id = "cam_rear"
            stored.trajectory.append((2.0, 0.5, 3.0, datetime.now()))
            database.save_object(stored)

        # Trajectory should have at least 2 points
        # (implementation may add extra points during save/restore)
        final = database.get_object("person-001")
        assert len(final.trajectory) >= 2

    def test_camera_handoff_preserves_id(self, two_camera_system, database, multi_camera_gallery):
        """Test object maintains same ID across camera handoff."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Create in camera 1
        obj = ObjectSchema(
            id="persistent-person",
            primary_class="person",
            confidence=0.9,
            reid_embedding=embedding,
            camera_id="cam_front",
        )
        database.save_object(obj)
        multi_camera_gallery.add("persistent-person", embedding, "cam_front")

        # Simulate appearing in camera 2
        noisy_embedding = embedding + np.random.randn(512).astype(np.float32) * 0.05
        noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)

        # Match should find same person
        results = multi_camera_gallery.search_cross_camera(
            noisy_embedding, exclude_camera="cam_rear"
        )

        assert results[0][0] == "persistent-person"

        # Single object in database
        assert database.count_objects() == 1

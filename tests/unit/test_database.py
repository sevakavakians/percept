"""Unit tests for database persistence layer."""

import json
import numpy as np
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from percept.core.schema import ObjectSchema, ClassificationStatus
from percept.persistence.database import PerceptDatabase


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_percept.db"
    db = PerceptDatabase(db_path)
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def sample_schema():
    """Create a sample ObjectSchema for testing."""
    return ObjectSchema(
        id="test-object-001",
        reid_embedding=np.random.randn(512).astype(np.float32),
        position_3d=(1.0, 0.5, 2.5),
        bounding_box_2d=(150, 100, 250, 300),
        dimensions=(0.5, 1.8, 0.3),
        distance_from_camera=2.5,
        primary_class="person",
        subclass="adult",
        confidence=0.92,
        classification_status=ClassificationStatus.CONFIRMED,
        attributes={"clothing": {"upper": "blue shirt"}},
        first_seen=datetime.now(),
        last_seen=datetime.now(),
        camera_id="cam_front",
        trajectory=[],
        pipelines_completed=["segmentation", "person"],
        processing_time_ms=45.2,
        source_frame_ids=[1, 2, 3],
    )


class TestDatabaseInitialization:
    """Tests for database initialization."""

    def test_create_database(self, tmp_path):
        """Test creating a new database."""
        db_path = tmp_path / "new_db.db"
        db = PerceptDatabase(db_path)
        db.initialize()

        assert db_path.exists()
        db.close()

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        db_path = tmp_path / "subdir" / "nested" / "db.db"
        db = PerceptDatabase(db_path)
        db.initialize()

        assert db_path.exists()
        db.close()

    def test_initialize_is_idempotent(self, temp_db):
        """Test that initialize can be called multiple times."""
        temp_db.initialize()
        temp_db.initialize()
        # Should not raise


class TestObjectCRUD:
    """Tests for Object CRUD operations."""

    def test_save_and_get_object(self, temp_db, sample_schema):
        """Test saving and retrieving an object."""
        temp_db.save_object(sample_schema)

        retrieved = temp_db.get_object(sample_schema.id)

        assert retrieved is not None
        assert retrieved.id == sample_schema.id
        assert retrieved.primary_class == sample_schema.primary_class
        assert retrieved.confidence == sample_schema.confidence

    def test_save_preserves_embedding(self, temp_db, sample_schema):
        """Test that embeddings are correctly saved and restored."""
        temp_db.save_object(sample_schema)

        retrieved = temp_db.get_object(sample_schema.id)

        assert retrieved.reid_embedding is not None
        np.testing.assert_array_almost_equal(
            retrieved.reid_embedding, sample_schema.reid_embedding, decimal=5
        )

    def test_save_preserves_attributes(self, temp_db, sample_schema):
        """Test that attributes JSON is correctly saved and restored."""
        temp_db.save_object(sample_schema)

        retrieved = temp_db.get_object(sample_schema.id)

        assert retrieved.attributes == sample_schema.attributes

    def test_get_nonexistent_object(self, temp_db):
        """Test getting an object that doesn't exist."""
        result = temp_db.get_object("nonexistent-id")
        assert result is None

    def test_update_object(self, temp_db, sample_schema):
        """Test updating an existing object."""
        temp_db.save_object(sample_schema)

        # Modify and save again
        sample_schema.primary_class = "vehicle"
        sample_schema.confidence = 0.85
        temp_db.save_object(sample_schema)

        retrieved = temp_db.get_object(sample_schema.id)

        assert retrieved.primary_class == "vehicle"
        assert retrieved.confidence == 0.85

    def test_delete_object(self, temp_db, sample_schema):
        """Test deleting an object."""
        temp_db.save_object(sample_schema)

        result = temp_db.delete_object(sample_schema.id)

        assert result is True
        assert temp_db.get_object(sample_schema.id) is None

    def test_delete_nonexistent_object(self, temp_db):
        """Test deleting an object that doesn't exist."""
        result = temp_db.delete_object("nonexistent-id")
        assert result is False


class TestObjectQueries:
    """Tests for object query methods."""

    def test_get_all_objects(self, temp_db, sample_schema):
        """Test getting all objects."""
        # Save multiple objects
        for i in range(5):
            obj = ObjectSchema(
                id=f"test-object-{i:03d}",
                reid_embedding=np.random.randn(512).astype(np.float32),
                primary_class="person",
                classification_status=ClassificationStatus.CONFIRMED,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
            temp_db.save_object(obj)

        all_objects = temp_db.get_all_objects()

        assert len(all_objects) == 5

    def test_get_all_objects_pagination(self, temp_db):
        """Test pagination in get_all_objects."""
        for i in range(10):
            obj = ObjectSchema(
                id=f"test-object-{i:03d}",
                primary_class="person",
                classification_status=ClassificationStatus.CONFIRMED,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
            temp_db.save_object(obj)

        page1 = temp_db.get_all_objects(limit=5, offset=0)
        page2 = temp_db.get_all_objects(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].id != page2[0].id

    def test_query_by_class(self, temp_db):
        """Test querying by primary class."""
        for cls, count in [("person", 3), ("vehicle", 2), ("unknown", 1)]:
            for i in range(count):
                obj = ObjectSchema(
                    id=f"{cls}-{i}",
                    primary_class=cls,
                    classification_status=ClassificationStatus.CONFIRMED,
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                )
                temp_db.save_object(obj)

        persons = temp_db.query_by_class("person")
        vehicles = temp_db.query_by_class("vehicle")

        assert len(persons) == 3
        assert len(vehicles) == 2

    def test_query_by_camera(self, temp_db):
        """Test querying by camera ID."""
        for cam, count in [("cam_front", 3), ("cam_rear", 2)]:
            for i in range(count):
                obj = ObjectSchema(
                    id=f"{cam}-{i}",
                    primary_class="person",
                    camera_id=cam,
                    classification_status=ClassificationStatus.CONFIRMED,
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                )
                temp_db.save_object(obj)

        front = temp_db.query_by_camera("cam_front")
        rear = temp_db.query_by_camera("cam_rear")

        assert len(front) == 3
        assert len(rear) == 2

    def test_query_needs_review(self, temp_db):
        """Test querying objects needing review."""
        for i, status in enumerate([
            ClassificationStatus.CONFIRMED,
            ClassificationStatus.NEEDS_REVIEW,
            ClassificationStatus.PROVISIONAL,
            ClassificationStatus.NEEDS_REVIEW,
        ]):
            obj = ObjectSchema(
                id=f"test-{i}",
                primary_class="unknown",
                classification_status=status,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
            temp_db.save_object(obj)

        needs_review = temp_db.query_needs_review()

        assert len(needs_review) == 2
        for obj in needs_review:
            assert obj.classification_status == ClassificationStatus.NEEDS_REVIEW

    def test_count_objects(self, temp_db):
        """Test counting objects."""
        for cls in ["person", "person", "vehicle"]:
            obj = ObjectSchema(
                id=f"test-{cls}-{np.random.randint(1000)}",
                primary_class=cls,
                classification_status=ClassificationStatus.CONFIRMED,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
            temp_db.save_object(obj)

        total = temp_db.count_objects()
        persons = temp_db.count_objects("person")
        vehicles = temp_db.count_objects("vehicle")

        assert total == 3
        assert persons == 2
        assert vehicles == 1


class TestTrajectory:
    """Tests for trajectory storage."""

    def test_save_and_get_trajectory(self, temp_db, sample_schema):
        """Test saving and retrieving trajectory points."""
        # Add trajectory points
        sample_schema.trajectory = [
            (1.0, 0.5, 2.5, datetime.now() - timedelta(seconds=2)),
            (1.1, 0.5, 2.4, datetime.now() - timedelta(seconds=1)),
            (1.2, 0.5, 2.3, datetime.now()),
        ]

        temp_db.save_object(sample_schema)

        trajectory = temp_db.get_trajectory(sample_schema.id)

        assert len(trajectory) == 3
        assert trajectory[0][0] == 1.0  # x coordinate
        assert trajectory[2][0] == 1.2  # x coordinate of last point


class TestReviewQueue:
    """Tests for human review queue."""

    def test_add_to_review_queue(self, temp_db, sample_schema):
        """Test adding an object to review queue."""
        sample_schema.classification_status = ClassificationStatus.NEEDS_REVIEW
        temp_db.save_object(sample_schema)

        review_id = temp_db.add_to_review_queue(
            sample_schema,
            image_path="/path/to/crop.jpg",
            reason="Low confidence classification",
        )

        assert review_id is not None
        assert review_id > 0

    def test_get_pending_reviews(self, temp_db, sample_schema):
        """Test getting pending review items."""
        sample_schema.classification_status = ClassificationStatus.NEEDS_REVIEW
        temp_db.save_object(sample_schema)

        temp_db.add_to_review_queue(
            sample_schema,
            image_path="/path/to/crop.jpg",
            reason="Low confidence",
        )

        pending = temp_db.get_pending_reviews()

        assert len(pending) == 1
        assert pending[0]["object_id"] == sample_schema.id
        assert pending[0]["reason"] == "Low confidence"

    def test_submit_review(self, temp_db, sample_schema):
        """Test submitting a review."""
        sample_schema.classification_status = ClassificationStatus.NEEDS_REVIEW
        sample_schema.primary_class = "unknown"
        temp_db.save_object(sample_schema)

        review_id = temp_db.add_to_review_queue(
            sample_schema,
            image_path="/path/to/crop.jpg",
            reason="Unknown object",
        )

        temp_db.submit_review(
            review_id,
            human_class="chair",
            reviewer="test_user",
            human_attributes={"material": "wood"},
        )

        # Check object was updated
        obj = temp_db.get_object(sample_schema.id)
        assert obj.primary_class == "chair"
        assert obj.classification_status == ClassificationStatus.CONFIRMED

        # Check review queue
        pending = temp_db.get_pending_reviews()
        assert len(pending) == 0

    def test_skip_review(self, temp_db, sample_schema):
        """Test skipping a review."""
        sample_schema.classification_status = ClassificationStatus.NEEDS_REVIEW
        temp_db.save_object(sample_schema)

        review_id = temp_db.add_to_review_queue(
            sample_schema,
            image_path="/path/to/crop.jpg",
            reason="Unclear",
        )

        temp_db.skip_review(review_id)

        pending = temp_db.get_pending_reviews()
        assert len(pending) == 0

    def test_get_review_stats(self, temp_db, sample_schema):
        """Test getting review statistics."""
        sample_schema.classification_status = ClassificationStatus.NEEDS_REVIEW
        temp_db.save_object(sample_schema)

        # Add multiple reviews with different statuses
        for i in range(3):
            obj = ObjectSchema(
                id=f"review-test-{i}",
                primary_class="unknown",
                classification_status=ClassificationStatus.NEEDS_REVIEW,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
            temp_db.save_object(obj)
            temp_db.add_to_review_queue(obj, f"/path/{i}.jpg", "test")

        # Submit one, skip one
        pending = temp_db.get_pending_reviews()
        temp_db.submit_review(pending[0]["id"], "chair", "user1")
        temp_db.skip_review(pending[1]["id"])

        stats = temp_db.get_review_stats()

        assert stats["reviewed"] == 1
        assert stats["skipped"] == 1
        assert stats["pending"] == 1


class TestEmbeddingOperations:
    """Tests for embedding-specific operations."""

    def test_get_embedding(self, temp_db, sample_schema):
        """Test getting just the embedding."""
        temp_db.save_object(sample_schema)

        embedding = temp_db.get_embedding(sample_schema.id)

        assert embedding is not None
        np.testing.assert_array_almost_equal(
            embedding, sample_schema.reid_embedding, decimal=5
        )

    def test_get_embedding_nonexistent(self, temp_db):
        """Test getting embedding for nonexistent object."""
        embedding = temp_db.get_embedding("nonexistent")
        assert embedding is None

    def test_get_all_embeddings(self, temp_db):
        """Test getting all embeddings for FAISS."""
        for i in range(5):
            obj = ObjectSchema(
                id=f"embed-test-{i}",
                reid_embedding=np.random.randn(512).astype(np.float32),
                primary_class="person",
                classification_status=ClassificationStatus.CONFIRMED,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
            temp_db.save_object(obj)

        embeddings = temp_db.get_all_embeddings()

        assert len(embeddings) == 5
        for obj_id, emb in embeddings:
            assert obj_id.startswith("embed-test-")
            assert emb.shape == (512,)


class TestUpdateClassification:
    """Tests for classification update method."""

    def test_update_classification(self, temp_db, sample_schema):
        """Test updating classification."""
        temp_db.save_object(sample_schema)

        result = temp_db.update_classification(
            sample_schema.id,
            primary_class="vehicle",
            confidence=0.95,
            status=ClassificationStatus.CONFIRMED,
            subclass="car",
        )

        assert result is True

        obj = temp_db.get_object(sample_schema.id)
        assert obj.primary_class == "vehicle"
        assert obj.subclass == "car"
        assert obj.confidence == 0.95

    def test_update_classification_nonexistent(self, temp_db):
        """Test updating nonexistent object."""
        result = temp_db.update_classification(
            "nonexistent",
            primary_class="test",
            confidence=0.5,
            status=ClassificationStatus.PROVISIONAL,
        )
        assert result is False


class TestContextManager:
    """Tests for context manager usage."""

    def test_context_manager(self, tmp_path):
        """Test using database as context manager."""
        db_path = tmp_path / "context_test.db"

        with PerceptDatabase(db_path) as db:
            db.initialize()
            obj = ObjectSchema(
                id="context-test",
                primary_class="test",
                classification_status=ClassificationStatus.CONFIRMED,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )
            db.save_object(obj)

        # Connection should be closed, but we can reopen
        db2 = PerceptDatabase(db_path)
        result = db2.get_object("context-test")
        assert result is not None
        db2.close()

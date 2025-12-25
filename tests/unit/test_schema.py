"""Unit tests for ObjectSchema and related classes."""

import json
import numpy as np
import pytest
from datetime import datetime, timedelta

from percept.core.schema import (
    ObjectSchema,
    ClassificationStatus,
    Detection,
    ObjectMask,
)


class TestClassificationStatus:
    """Tests for ClassificationStatus enum."""

    def test_status_values(self):
        """Test that all expected statuses exist."""
        assert ClassificationStatus.CONFIRMED.value == "confirmed"
        assert ClassificationStatus.PROVISIONAL.value == "provisional"
        assert ClassificationStatus.NEEDS_REVIEW.value == "needs_review"
        assert ClassificationStatus.UNCLASSIFIED.value == "unclassified"

    def test_status_from_string(self):
        """Test creating status from string value."""
        status = ClassificationStatus("confirmed")
        assert status == ClassificationStatus.CONFIRMED


class TestObjectSchema:
    """Tests for ObjectSchema dataclass."""

    def test_create_minimal_schema(self):
        """Test creating schema with minimal parameters."""
        schema = ObjectSchema()
        assert schema.id is not None
        assert len(schema.id) == 36  # UUID format
        assert schema.primary_class == "unknown"
        assert schema.classification_status == ClassificationStatus.UNCLASSIFIED

    def test_create_with_embedding(self, sample_embedding):
        """Test creating schema with embedding (should normalize)."""
        schema = ObjectSchema(reid_embedding=sample_embedding)

        # Embedding should be L2 normalized
        norm = np.linalg.norm(schema.reid_embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_embedding_normalization(self):
        """Test that embeddings are automatically L2 normalized."""
        # Create non-normalized embedding (3-4-5 triangle pattern)
        embedding = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32)
        schema = ObjectSchema(reid_embedding=embedding)

        norm = np.linalg.norm(schema.reid_embedding)
        assert abs(norm - 1.0) < 1e-6
        assert abs(schema.reid_embedding[0] - 0.6) < 1e-6
        assert abs(schema.reid_embedding[1] - 0.8) < 1e-6

    def test_set_embedding(self, sample_object_schema):
        """Test set_embedding method normalizes."""
        new_embedding = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32)
        sample_object_schema.set_embedding(new_embedding)

        norm = np.linalg.norm(sample_object_schema.reid_embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_update_position(self, sample_object_schema):
        """Test updating position adds to trajectory."""
        original_len = len(sample_object_schema.trajectory)
        new_pos = (2.0, 1.0, 3.5)

        sample_object_schema.update_position(new_pos)

        assert sample_object_schema.position_3d == new_pos
        assert len(sample_object_schema.trajectory) == original_len + 1
        assert sample_object_schema.trajectory[-1][:3] == new_pos

    def test_mark_pipeline_complete(self, sample_object_schema):
        """Test marking pipeline as complete."""
        original_time = sample_object_schema.processing_time_ms
        sample_object_schema.mark_pipeline_complete("tracking", 10.5)

        assert "tracking" in sample_object_schema.pipelines_completed
        assert sample_object_schema.processing_time_ms == original_time + 10.5

    def test_mark_pipeline_complete_no_duplicate(self, sample_object_schema):
        """Test marking same pipeline twice doesn't duplicate."""
        sample_object_schema.mark_pipeline_complete("segmentation", 5.0)
        count = sample_object_schema.pipelines_completed.count("segmentation")
        assert count == 1

    def test_set_classification_high_confidence(self):
        """Test setting classification with high confidence."""
        schema = ObjectSchema()
        schema.set_classification("person", 0.95, subclass="adult")

        assert schema.primary_class == "person"
        assert schema.subclass == "adult"
        assert schema.confidence == 0.95
        assert schema.classification_status == ClassificationStatus.CONFIRMED

    def test_set_classification_medium_confidence(self):
        """Test setting classification with medium confidence."""
        schema = ObjectSchema()
        schema.set_classification("vehicle", 0.7)

        assert schema.primary_class == "vehicle"
        assert schema.classification_status == ClassificationStatus.PROVISIONAL

    def test_set_classification_low_confidence(self):
        """Test setting classification with low confidence."""
        schema = ObjectSchema()
        schema.set_classification("unknown_object", 0.3)

        assert schema.classification_status == ClassificationStatus.NEEDS_REVIEW

    def test_needs_review_property(self):
        """Test needs_review property."""
        schema = ObjectSchema()
        schema.classification_status = ClassificationStatus.NEEDS_REVIEW
        assert schema.needs_review is True

        schema.classification_status = ClassificationStatus.CONFIRMED
        assert schema.needs_review is False

    def test_is_confirmed_property(self):
        """Test is_confirmed property."""
        schema = ObjectSchema()
        schema.classification_status = ClassificationStatus.CONFIRMED
        assert schema.is_confirmed is True

        schema.classification_status = ClassificationStatus.PROVISIONAL
        assert schema.is_confirmed is False

    def test_to_dict(self, sample_object_schema):
        """Test conversion to dictionary."""
        data = sample_object_schema.to_dict()

        assert data["id"] == sample_object_schema.id
        assert data["primary_class"] == "person"
        assert data["classification_status"] == "confirmed"
        assert isinstance(data["reid_embedding"], list)
        assert isinstance(data["first_seen"], str)

    def test_to_json(self, sample_object_schema):
        """Test JSON serialization."""
        json_str = sample_object_schema.to_json()
        data = json.loads(json_str)

        assert data["id"] == sample_object_schema.id
        assert data["primary_class"] == "person"

    def test_from_dict(self, sample_object_schema):
        """Test creating from dictionary."""
        data = sample_object_schema.to_dict()
        restored = ObjectSchema.from_dict(data)

        assert restored.id == sample_object_schema.id
        assert restored.primary_class == sample_object_schema.primary_class
        assert restored.confidence == sample_object_schema.confidence
        assert np.allclose(restored.reid_embedding, sample_object_schema.reid_embedding)

    def test_from_json(self, sample_object_schema):
        """Test JSON deserialization."""
        json_str = sample_object_schema.to_json()
        restored = ObjectSchema.from_json(json_str)

        assert restored.id == sample_object_schema.id
        assert restored.primary_class == sample_object_schema.primary_class

    def test_roundtrip_serialization(self, sample_object_schema):
        """Test full roundtrip: object -> JSON -> object."""
        json_str = sample_object_schema.to_json()
        restored = ObjectSchema.from_json(json_str)

        # Re-serialize and compare
        json_str2 = restored.to_json()
        assert json.loads(json_str) == json.loads(json_str2)


class TestDetection:
    """Tests for Detection dataclass."""

    def test_create_detection(self):
        """Test creating a Detection."""
        detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.92,
            bbox=(100, 50, 200, 300),
        )
        assert detection.class_name == "person"
        assert detection.bbox == (100, 50, 200, 300)

    def test_detection_area(self, sample_detection_obj):
        """Test bounding box area calculation."""
        x1, y1, x2, y2 = sample_detection_obj.bbox
        expected_area = (x2 - x1) * (y2 - y1)
        assert sample_detection_obj.area == expected_area

    def test_detection_center(self, sample_detection_obj):
        """Test center point calculation."""
        x1, y1, x2, y2 = sample_detection_obj.bbox
        expected_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        assert sample_detection_obj.center == expected_center


class TestObjectMask:
    """Tests for ObjectMask dataclass."""

    def test_create_object_mask(self, sample_mask):
        """Test creating an ObjectMask."""
        mask = ObjectMask(
            mask=sample_mask,
            bbox=(150, 100, 250, 300),
            confidence=0.95,
        )
        assert mask.bbox == (150, 100, 250, 300)
        assert mask.confidence == 0.95

    def test_mask_area(self, sample_object_mask):
        """Test mask area calculation."""
        expected_area = np.sum(sample_object_mask.mask > 0)
        assert sample_object_mask.area == expected_area

    def test_extract_crop(self, sample_object_mask, sample_rgb_image):
        """Test extracting masked crop from image."""
        crop = sample_object_mask.extract_crop(sample_rgb_image)

        x1, y1, x2, y2 = sample_object_mask.bbox
        assert crop.shape == (y2 - y1, x2 - x1, 3)

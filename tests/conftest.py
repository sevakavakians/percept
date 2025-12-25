"""
Shared pytest fixtures for PERCEPT tests.
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from percept.core.schema import ObjectSchema, ClassificationStatus, Detection, ObjectMask
from percept.core.config import PerceptConfig, CameraConfig
from percept.core.pipeline import Pipeline, PipelineModule
from percept.core.adapter import DataSpec, DataAdapter, PipelineData

# Fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    # Generate a synthetic test image (640x480 BGR)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some colored rectangles to simulate objects
    image[100:300, 150:250] = [0, 0, 255]  # Red rectangle (person-like)
    image[200:350, 400:500] = [255, 0, 0]  # Blue rectangle (object)
    return image


@pytest.fixture
def sample_depth_image():
    """Create a sample depth image for testing (meters)."""
    # Generate synthetic depth (640x480 float32, meters)
    depth = np.ones((480, 640), dtype=np.float32) * 5.0  # Background at 5m
    # Add objects at different depths
    depth[100:300, 150:250] = 2.0  # Object 1 at 2m
    depth[200:350, 400:500] = 3.0  # Object 2 at 3m
    return depth


@pytest.fixture
def sample_mask():
    """Create a sample binary mask."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:300, 150:250] = 255
    return mask


@pytest.fixture
def sample_embedding():
    """Create a sample 512-dim embedding vector."""
    embedding = np.random.randn(512).astype(np.float32)
    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def sample_detection():
    """Create a sample detection dictionary."""
    return {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.92,
        "bbox": (150, 100, 250, 300),  # x1, y1, x2, y2
    }


@pytest.fixture
def temp_database(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_percept.db"
    # Return path - actual DB creation will be done by the module under test
    return str(db_path)


@pytest.fixture
def mock_camera_intrinsics():
    """Mock RealSense camera intrinsics."""
    return {
        "width": 640,
        "height": 480,
        "fx": 615.0,
        "fy": 615.0,
        "ppx": 320.0,
        "ppy": 240.0,
    }


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        "framework": {"name": "PERCEPT", "version": "0.1.0"},
        "cameras": [
            {
                "id": "test_cam",
                "type": "mock",
                "resolution": [640, 480],
                "fps": 30,
            }
        ],
        "segmentation": {
            "primary_method": "fastsam",
            "min_object_pixels": 500,
        },
        "reid": {
            "match_threshold_same_camera": 0.3,
            "embedding_dimension": 512,
        },
        "classification": {
            "confidence_confirmed": 0.85,
            "confidence_provisional": 0.5,
        },
    }


# Core module fixtures

@pytest.fixture
def sample_object_schema(sample_embedding):
    """Create a sample ObjectSchema for testing."""
    return ObjectSchema(
        id="test-object-001",
        reid_embedding=sample_embedding,
        position_3d=(1.0, 0.5, 2.5),
        bounding_box_2d=(150, 100, 250, 300),
        dimensions=(0.5, 1.8, 0.3),
        distance_from_camera=2.5,
        primary_class="person",
        subclass="adult",
        confidence=0.92,
        classification_status=ClassificationStatus.CONFIRMED,
        attributes={"clothing": {"upper": "blue shirt", "lower": "jeans"}},
        first_seen=datetime.now(),
        last_seen=datetime.now(),
        camera_id="cam_front",
        trajectory=[],
        pipelines_completed=["segmentation", "person"],
        processing_time_ms=45.2,
        source_frame_ids=[1, 2, 3],
    )


@pytest.fixture
def sample_detection_obj():
    """Create a sample Detection object for testing."""
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=(150, 100, 250, 300),
        mask=None,
    )


@pytest.fixture
def sample_object_mask(sample_mask):
    """Create a sample ObjectMask for testing."""
    return ObjectMask(
        mask=sample_mask,
        bbox=(150, 100, 250, 300),
        confidence=0.95,
        depth_median=2.0,
        point_count=5000,
    )


@pytest.fixture
def percept_config(sample_config, tmp_path):
    """Create a PerceptConfig from sample config."""
    return PerceptConfig.from_dict(sample_config)


@pytest.fixture
def data_adapter():
    """Create a DataAdapter instance."""
    return DataAdapter()


@pytest.fixture
def sample_pipeline_data(sample_rgb_image, sample_depth_image):
    """Create sample PipelineData with image and depth."""
    return PipelineData(
        image=sample_rgb_image,
        depth=sample_depth_image,
        frame_id=1,
        timestamp=datetime.now(),
    )


@pytest.fixture
def image_data_spec():
    """Create a sample DataSpec for images."""
    return DataSpec(
        data_type="image",
        shape=(480, 640, 3),
        dtype="uint8",
        color_space="BGR",
    )


class MockPipelineModule(PipelineModule):
    """Mock pipeline module for testing."""

    def __init__(self, name: str = "mock_module"):
        self._name = name
        self._process_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(data_type="image")

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(data_type="image")

    def process(self, data: PipelineData) -> PipelineData:
        self._process_count += 1
        result = data.copy()
        result.set_metadata("processed_by", self._name)
        return result


@pytest.fixture
def mock_module():
    """Create a mock pipeline module."""
    return MockPipelineModule()


@pytest.fixture
def sample_pipeline(mock_module):
    """Create a sample pipeline with one module."""
    pipeline = Pipeline("test_pipeline", store_intermediates=True)
    pipeline.add_module(mock_module)
    return pipeline

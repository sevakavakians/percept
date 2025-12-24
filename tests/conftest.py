"""
Shared pytest fixtures for PERCEPT tests.
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime

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

"""Unit tests for configuration system."""

import pytest
import yaml
from pathlib import Path

from percept.core.config import (
    PerceptConfig,
    CameraConfig,
    SegmentationConfig,
    ReIDConfig,
    TrackingConfig,
    ClassificationConfig,
    NormalizationConfig,
    DatabaseConfig,
    PerformanceConfig,
    get_default_config,
)


class TestCameraConfig:
    """Tests for CameraConfig."""

    def test_create_with_defaults(self):
        """Test creating camera config with defaults."""
        config = CameraConfig(id="test_cam")
        assert config.id == "test_cam"
        assert config.type == "realsense_d455"
        assert config.resolution == (640, 480)
        assert config.fps == 30
        assert config.enabled is True

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "id": "cam_front",
            "type": "realsense_d415",
            "resolution": [1280, 720],
            "fps": 15,
            "enabled": False,
        }
        config = CameraConfig.from_dict(data)
        assert config.id == "cam_front"
        assert config.type == "realsense_d415"
        assert config.resolution == (1280, 720)
        assert config.fps == 15
        assert config.enabled is False


class TestSegmentationConfig:
    """Tests for SegmentationConfig."""

    def test_defaults(self):
        """Test default values."""
        config = SegmentationConfig()
        assert config.primary_method == "fastsam"
        assert config.fusion_enabled is True
        assert config.min_object_pixels == 500
        assert config.max_objects_per_frame == 50


class TestReIDConfig:
    """Tests for ReIDConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ReIDConfig()
        assert config.embedding_dimension == 512
        assert config.match_threshold_same_camera == 0.3
        assert config.match_threshold_cross_camera == 0.25


class TestPerceptConfig:
    """Tests for main PerceptConfig."""

    def test_from_dict(self, sample_config):
        """Test creating from dictionary."""
        config = PerceptConfig.from_dict(sample_config)

        assert config.name == "PERCEPT"
        assert config.version == "0.1.0"
        assert len(config.cameras) == 1
        assert config.cameras[0].id == "test_cam"

    def test_from_dict_with_defaults(self):
        """Test that missing sections use defaults."""
        minimal_config = {
            "framework": {"name": "TEST", "version": "0.0.1"},
            "cameras": [{"id": "cam1"}],
        }
        config = PerceptConfig.from_dict(minimal_config)

        assert config.name == "TEST"
        assert config.segmentation.primary_method == "fastsam"  # Default
        assert config.reid.embedding_dimension == 512  # Default

    def test_to_dict(self, percept_config):
        """Test conversion to dictionary."""
        data = percept_config.to_dict()

        assert data["framework"]["name"] == "PERCEPT"
        assert "cameras" in data
        assert "segmentation" in data
        assert "reid" in data

    def test_load_from_file(self, tmp_path):
        """Test loading from YAML file."""
        config_data = {
            "framework": {"name": "TEST", "version": "1.0"},
            "cameras": [{"id": "test_cam"}],
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        config = PerceptConfig.load(config_file)
        assert config.name == "TEST"
        assert config.version == "1.0"

    def test_load_missing_file(self, tmp_path):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            PerceptConfig.load(tmp_path / "nonexistent.yaml")

    def test_save(self, percept_config, tmp_path):
        """Test saving to file."""
        save_path = tmp_path / "saved_config.yaml"
        percept_config._config_path = save_path  # Set path for save

        percept_config.save()

        assert save_path.exists()
        with open(save_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["framework"]["name"] == "PERCEPT"

    def test_save_to_specific_path(self, percept_config, tmp_path):
        """Test saving to specific path."""
        save_path = tmp_path / "specific_config.yaml"
        percept_config.save(save_path)

        assert save_path.exists()

    def test_get_camera(self, percept_config):
        """Test getting camera by ID."""
        camera = percept_config.get_camera("test_cam")
        assert camera is not None
        assert camera.id == "test_cam"

        # Non-existent camera
        assert percept_config.get_camera("nonexistent") is None

    def test_get_enabled_cameras(self):
        """Test getting only enabled cameras."""
        config = PerceptConfig(
            cameras=[
                CameraConfig(id="cam1", enabled=True),
                CameraConfig(id="cam2", enabled=False),
                CameraConfig(id="cam3", enabled=True),
            ]
        )

        enabled = config.get_enabled_cameras()
        assert len(enabled) == 2
        assert all(c.enabled for c in enabled)

    def test_validate_success(self, percept_config):
        """Test validation passes for valid config."""
        errors = percept_config.validate()
        assert len(errors) == 0

    def test_validate_no_cameras(self):
        """Test validation catches missing cameras."""
        config = PerceptConfig(cameras=[])
        errors = config.validate()
        assert any("No cameras" in e for e in errors)

    def test_validate_duplicate_camera_ids(self):
        """Test validation catches duplicate camera IDs."""
        config = PerceptConfig(
            cameras=[
                CameraConfig(id="cam1"),
                CameraConfig(id="cam1"),  # Duplicate
            ]
        )
        errors = config.validate()
        assert any("Duplicate" in e for e in errors)

    def test_validate_invalid_threshold(self):
        """Test validation catches invalid thresholds."""
        config = PerceptConfig(
            cameras=[CameraConfig(id="cam1")],
            classification=ClassificationConfig(
                confidence_confirmed=0.4,  # Less than provisional
                confidence_provisional=0.5,
            ),
        )
        errors = config.validate()
        assert any("threshold" in e.lower() for e in errors)


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_get_absolute_path_relative(self, tmp_path):
        """Test getting absolute path from relative path."""
        config = DatabaseConfig(path="data/test.db")
        abs_path = config.get_absolute_path(tmp_path)
        assert abs_path == tmp_path / "data" / "test.db"

    def test_get_absolute_path_absolute(self, tmp_path):
        """Test getting absolute path when already absolute."""
        abs_input = tmp_path / "absolute.db"
        config = DatabaseConfig(path=str(abs_input))
        abs_path = config.get_absolute_path()
        assert abs_path == abs_input


class TestGetDefaultConfig:
    """Tests for get_default_config helper."""

    def test_returns_valid_config(self):
        """Test that default config is valid."""
        config = get_default_config()
        assert isinstance(config, PerceptConfig)
        assert len(config.cameras) > 0
        errors = config.validate()
        assert len(errors) == 0

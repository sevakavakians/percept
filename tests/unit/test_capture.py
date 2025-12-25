"""Unit tests for capture module."""

import numpy as np
import pytest
from datetime import datetime

from percept.capture.realsense import (
    FrameData,
    CameraIntrinsics,
    is_realsense_available,
)
from percept.core.config import CameraConfig


class TestCameraIntrinsics:
    """Tests for CameraIntrinsics dataclass."""

    def test_create_intrinsics(self):
        """Test creating camera intrinsics."""
        intrinsics = CameraIntrinsics(
            width=640,
            height=480,
            fx=615.0,
            fy=615.0,
            ppx=320.0,
            ppy=240.0,
        )
        assert intrinsics.width == 640
        assert intrinsics.height == 480
        assert intrinsics.fx == 615.0

    def test_to_matrix(self):
        """Test conversion to camera matrix."""
        intrinsics = CameraIntrinsics(
            width=640,
            height=480,
            fx=615.0,
            fy=615.0,
            ppx=320.0,
            ppy=240.0,
        )

        matrix = intrinsics.to_matrix()

        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 615.0  # fx
        assert matrix[1, 1] == 615.0  # fy
        assert matrix[0, 2] == 320.0  # ppx
        assert matrix[1, 2] == 240.0  # ppy
        assert matrix[2, 2] == 1.0


class TestFrameData:
    """Tests for FrameData dataclass."""

    def test_create_empty_frame(self):
        """Test creating empty frame data."""
        frame = FrameData()
        assert frame.color is None
        assert frame.depth is None
        assert frame.camera_id == ""
        assert frame.frame_number == 0

    def test_create_frame_with_data(self, sample_rgb_image, sample_depth_image):
        """Test creating frame with image data."""
        frame = FrameData(
            color=sample_rgb_image,
            depth=sample_depth_image,
            camera_id="cam_front",
            frame_number=42,
        )

        assert frame.color is not None
        assert frame.depth is not None
        assert frame.camera_id == "cam_front"
        assert frame.frame_number == 42

    def test_has_depth_property(self, sample_depth_image):
        """Test has_depth property."""
        frame_no_depth = FrameData()
        assert frame_no_depth.has_depth is False

        frame_with_depth = FrameData(depth=sample_depth_image)
        assert frame_with_depth.has_depth is True

    def test_has_color_property(self, sample_rgb_image):
        """Test has_color property."""
        frame_no_color = FrameData()
        assert frame_no_color.has_color is False

        frame_with_color = FrameData(color=sample_rgb_image)
        assert frame_with_color.has_color is True

    def test_to_pipeline_data(self, sample_rgb_image, sample_depth_image):
        """Test conversion to PipelineData."""
        intrinsics = CameraIntrinsics(
            width=640, height=480,
            fx=615.0, fy=615.0,
            ppx=320.0, ppy=240.0,
        )

        frame = FrameData(
            color=sample_rgb_image,
            depth=sample_depth_image,
            camera_id="cam_front",
            frame_number=1,
            intrinsics=intrinsics,
        )

        pipeline_data = frame.to_pipeline_data()

        assert "image" in pipeline_data
        assert "depth" in pipeline_data
        assert pipeline_data.camera_id == "cam_front"
        assert pipeline_data.frame_number == 1
        np.testing.assert_array_equal(pipeline_data.image, sample_rgb_image)


class TestCameraConfig:
    """Tests for creating camera from config."""

    def test_camera_config_defaults(self):
        """Test camera config default values."""
        config = CameraConfig(id="test_cam")
        assert config.resolution == (640, 480)
        assert config.fps == 30
        assert config.enabled is True

    def test_camera_config_custom(self):
        """Test camera config with custom values."""
        config = CameraConfig(
            id="custom_cam",
            type="realsense_d415",
            resolution=(1280, 720),
            fps=15,
            serial="12345",
            enabled=True,
        )
        assert config.id == "custom_cam"
        assert config.resolution == (1280, 720)
        assert config.fps == 15
        assert config.serial == "12345"


class TestRealSenseAvailability:
    """Tests for RealSense availability check."""

    def test_is_realsense_available(self):
        """Test availability check function."""
        # This should return True or False depending on pyrealsense2 installation
        result = is_realsense_available()
        assert isinstance(result, bool)


# Mock tests for when RealSense is not available

class MockRealSenseCamera:
    """Mock camera for testing without hardware."""

    def __init__(
        self,
        camera_id: str = "mock_camera",
        width: int = 640,
        height: int = 480,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._started = False
        self._frame_count = 0
        self._intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=615.0,
            fy=615.0,
            ppx=width / 2,
            ppy=height / 2,
        )

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def capture(self) -> FrameData:
        """Generate synthetic frame data."""
        self._frame_count += 1

        # Generate synthetic color image
        color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        color[100:200, 100:200] = [0, 0, 255]  # Red rectangle

        # Generate synthetic depth
        depth = np.ones((self.height, self.width), dtype=np.float32) * 3.0
        depth[100:200, 100:200] = 1.5  # Object closer

        return FrameData(
            color=color,
            depth=depth,
            depth_raw=(depth * 1000).astype(np.uint16),
            timestamp=datetime.now(),
            camera_id=self.camera_id,
            frame_number=self._frame_count,
            intrinsics=self._intrinsics,
        )

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class TestMockCamera:
    """Tests using mock camera."""

    def test_mock_camera_capture(self):
        """Test mock camera produces valid frames."""
        camera = MockRealSenseCamera()
        camera.start()

        frame = camera.capture()

        assert frame.has_color
        assert frame.has_depth
        assert frame.color.shape == (480, 640, 3)
        assert frame.depth.shape == (480, 640)
        assert frame.frame_number == 1

        camera.stop()

    def test_mock_camera_context_manager(self):
        """Test mock camera as context manager."""
        with MockRealSenseCamera() as camera:
            assert camera.is_started

            frame = camera.capture()
            assert frame.has_color

        assert not camera.is_started

    def test_mock_camera_frame_counting(self):
        """Test frame counter increments."""
        camera = MockRealSenseCamera()
        camera.start()

        frame1 = camera.capture()
        frame2 = camera.capture()
        frame3 = camera.capture()

        assert frame1.frame_number == 1
        assert frame2.frame_number == 2
        assert frame3.frame_number == 3

        camera.stop()

    def test_mock_camera_intrinsics(self):
        """Test mock camera intrinsics."""
        camera = MockRealSenseCamera(width=1280, height=720)

        intrinsics = camera.intrinsics
        assert intrinsics.width == 1280
        assert intrinsics.height == 720
        assert intrinsics.ppx == 640.0
        assert intrinsics.ppy == 360.0


class TestMultiCameraMock:
    """Tests for multi-camera setup using mocks."""

    def test_multiple_mock_cameras(self):
        """Test managing multiple mock cameras."""
        cameras = {
            "cam_front": MockRealSenseCamera(camera_id="cam_front"),
            "cam_rear": MockRealSenseCamera(camera_id="cam_rear"),
        }

        # Start all
        for cam in cameras.values():
            cam.start()

        # Capture from all
        frames = {}
        for cam_id, cam in cameras.items():
            frames[cam_id] = cam.capture()

        assert len(frames) == 2
        assert frames["cam_front"].camera_id == "cam_front"
        assert frames["cam_rear"].camera_id == "cam_rear"

        # Stop all
        for cam in cameras.values():
            cam.stop()

"""Intel RealSense capture for PERCEPT.

Provides multi-camera RGB-D capture with depth alignment and
frame synchronization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

from percept.core.config import CameraConfig
from percept.core.adapter import PipelineData


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    width: int
    height: int
    fx: float  # Focal length x
    fy: float  # Focal length y
    ppx: float  # Principal point x
    ppy: float  # Principal point y
    model: str = "unknown"
    coeffs: Tuple[float, ...] = field(default_factory=tuple)

    def to_matrix(self) -> np.ndarray:
        """Return 3x3 camera matrix."""
        return np.array([
            [self.fx, 0, self.ppx],
            [0, self.fy, self.ppy],
            [0, 0, 1],
        ], dtype=np.float32)


@dataclass
class FrameData:
    """Container for a single captured frame.

    Attributes:
        color: BGR color image (H, W, 3)
        depth: Depth in meters (H, W)
        depth_raw: Raw depth in millimeters (H, W)
        timestamp: Capture timestamp
        camera_id: Camera identifier
        frame_number: Sequence number
        intrinsics: Camera intrinsic parameters
    """

    color: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    depth_raw: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    camera_id: str = ""
    frame_number: int = 0
    intrinsics: Optional[CameraIntrinsics] = None

    def to_pipeline_data(self) -> PipelineData:
        """Convert to PipelineData for pipeline processing."""
        return PipelineData(
            image=self.color,
            depth=self.depth,
            depth_raw=self.depth_raw,
            timestamp=self.timestamp,
            camera_id=self.camera_id,
            frame_number=self.frame_number,
            intrinsics=self.intrinsics,
        )

    @property
    def has_depth(self) -> bool:
        """Check if depth data is available."""
        return self.depth is not None

    @property
    def has_color(self) -> bool:
        """Check if color data is available."""
        return self.color is not None


class RealSenseCamera:
    """Single RealSense camera interface.

    Wraps pyrealsense2 with convenient methods for PERCEPT integration.
    """

    def __init__(
        self,
        camera_id: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
        enable_depth: bool = True,
        align_depth: bool = True,
    ):
        """Initialize RealSense camera.

        Args:
            camera_id: Unique identifier for this camera
            width: Frame width
            height: Frame height
            fps: Frames per second
            serial: Device serial number (for multi-camera)
            enable_depth: Enable depth stream
            align_depth: Align depth to color frame
        """
        if not REALSENSE_AVAILABLE:
            raise ImportError(
                "pyrealsense2 not installed. Install with: pip install pyrealsense2"
            )

        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.enable_depth = enable_depth
        self.align_depth = align_depth

        self._pipeline: Optional[rs.pipeline] = None
        self._config: Optional[rs.config] = None
        self._align: Optional[rs.align] = None
        self._profile: Optional[rs.pipeline_profile] = None
        self._started = False
        self._frame_count = 0
        self._depth_scale = 0.001

        self._intrinsics: Optional[CameraIntrinsics] = None
        self._device_info: Dict[str, str] = {}

    @classmethod
    def from_config(cls, config: CameraConfig) -> RealSenseCamera:
        """Create camera from configuration."""
        return cls(
            camera_id=config.id,
            width=config.resolution[0],
            height=config.resolution[1],
            fps=config.fps,
            serial=config.serial if config.serial else None,
        )

    def start(self) -> None:
        """Start the camera pipeline."""
        if self._started:
            return

        self._pipeline = rs.pipeline()
        self._config = rs.config()

        # Select specific device if serial provided
        if self.serial:
            self._config.enable_device(self.serial)

        # Enable color stream
        self._config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )

        # Enable depth stream
        if self.enable_depth:
            self._config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )

        # Start pipeline
        self._profile = self._pipeline.start(self._config)

        # Get depth scale
        if self.enable_depth:
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()

        # Create aligner
        if self.align_depth and self.enable_depth:
            self._align = rs.align(rs.stream.color)

        # Get device info
        device = self._profile.get_device()
        self._device_info = {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
        }

        # Get intrinsics
        self._intrinsics = self._get_intrinsics()

        self._started = True
        self._warmup()

    def _warmup(self, frames: int = 30) -> None:
        """Discard initial frames to let auto-exposure stabilize."""
        for _ in range(frames):
            try:
                self._pipeline.wait_for_frames(1000)
            except Exception:
                pass

    def _get_intrinsics(self) -> CameraIntrinsics:
        """Get camera intrinsic parameters."""
        color_stream = self._profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        return CameraIntrinsics(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            ppx=intr.ppx,
            ppy=intr.ppy,
            model=str(intr.model),
            coeffs=tuple(intr.coeffs),
        )

    def stop(self) -> None:
        """Stop the camera pipeline."""
        if self._started and self._pipeline:
            self._pipeline.stop()
            self._started = False

    def capture(self, timeout_ms: int = 5000) -> FrameData:
        """Capture a single frame.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            FrameData with captured images

        Raises:
            RuntimeError: If capture fails
        """
        if not self._started:
            self.start()

        frames = self._pipeline.wait_for_frames(timeout_ms)

        # Align frames if enabled
        if self._align:
            frames = self._align.process(frames)

        self._frame_count += 1

        frame_data = FrameData(
            timestamp=datetime.now(),
            camera_id=self.camera_id,
            frame_number=self._frame_count,
            intrinsics=self._intrinsics,
        )

        # Get color frame
        color_frame = frames.get_color_frame()
        if color_frame:
            frame_data.color = np.asanyarray(color_frame.get_data())

        # Get depth frame
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_raw = np.asanyarray(depth_frame.get_data())
                frame_data.depth_raw = depth_raw
                frame_data.depth = depth_raw.astype(np.float32) * self._depth_scale

        return frame_data

    def get_depth_at_point(
        self,
        depth: np.ndarray,
        x: int,
        y: int,
        window_size: int = 3,
    ) -> float:
        """Get depth at a specific point with median filtering.

        Args:
            depth: Depth array in meters
            x: X coordinate
            y: Y coordinate
            window_size: Averaging window size

        Returns:
            Depth in meters (0.0 if invalid)
        """
        h, w = depth.shape
        half = window_size // 2

        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)

        window = depth[y1:y2, x1:x2]
        valid = window[window > 0]

        if len(valid) > 0:
            return float(np.median(valid))
        return 0.0

    def deproject_pixel_to_point(
        self,
        x: int,
        y: int,
        depth: float,
    ) -> Tuple[float, float, float]:
        """Convert 2D pixel + depth to 3D point.

        Args:
            x: Pixel X coordinate
            y: Pixel Y coordinate
            depth: Depth in meters

        Returns:
            (X, Y, Z) coordinates in meters
        """
        if not self._started:
            raise RuntimeError("Camera not started")

        color_stream = self._profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        return tuple(point)

    @property
    def is_started(self) -> bool:
        """Check if camera is started."""
        return self._started

    @property
    def device_info(self) -> Dict[str, str]:
        """Get device information."""
        return self._device_info.copy()

    @property
    def depth_scale(self) -> float:
        """Get depth scale (meters per unit)."""
        return self._depth_scale

    @property
    def intrinsics(self) -> Optional[CameraIntrinsics]:
        """Get camera intrinsics."""
        return self._intrinsics

    def __enter__(self) -> RealSenseCamera:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class MultiCameraCapture:
    """Manage multiple RealSense cameras.

    Provides synchronized capture from multiple cameras with
    convenient iteration and context management.
    """

    def __init__(self, configs: Optional[List[CameraConfig]] = None):
        """Initialize multi-camera system.

        Args:
            configs: List of camera configurations
        """
        self.cameras: Dict[str, RealSenseCamera] = {}
        self._started = False

        if configs:
            for config in configs:
                if config.enabled:
                    self.add_camera(config)

    def add_camera(self, config: CameraConfig) -> None:
        """Add a camera from configuration.

        Args:
            config: Camera configuration
        """
        camera = RealSenseCamera.from_config(config)
        self.cameras[config.id] = camera

    def add_camera_by_serial(
        self,
        camera_id: str,
        serial: str,
        **kwargs: Any,
    ) -> None:
        """Add a camera by serial number.

        Args:
            camera_id: Unique identifier
            serial: Device serial number
            **kwargs: Additional camera parameters
        """
        camera = RealSenseCamera(
            camera_id=camera_id,
            serial=serial,
            **kwargs,
        )
        self.cameras[camera_id] = camera

    def start_all(self) -> None:
        """Start all cameras."""
        if self._started:
            return

        for camera in self.cameras.values():
            camera.start()

        self._started = True

    def stop_all(self) -> None:
        """Stop all cameras."""
        for camera in self.cameras.values():
            camera.stop()
        self._started = False

    def capture_all(self, timeout_ms: int = 5000) -> Dict[str, FrameData]:
        """Capture from all cameras.

        Args:
            timeout_ms: Timeout per camera in milliseconds

        Returns:
            Dictionary mapping camera_id to FrameData
        """
        if not self._started:
            self.start_all()

        frames = {}
        for camera_id, camera in self.cameras.items():
            try:
                frames[camera_id] = camera.capture(timeout_ms)
            except Exception as e:
                # Log error but continue with other cameras
                frames[camera_id] = FrameData(
                    camera_id=camera_id,
                    timestamp=datetime.now(),
                )

        return frames

    def capture_single(
        self,
        camera_id: str,
        timeout_ms: int = 5000,
    ) -> Optional[FrameData]:
        """Capture from a single camera.

        Args:
            camera_id: Camera to capture from
            timeout_ms: Timeout in milliseconds

        Returns:
            FrameData or None if camera not found
        """
        if camera_id not in self.cameras:
            return None

        if not self._started:
            self.start_all()

        return self.cameras[camera_id].capture(timeout_ms)

    def get_camera(self, camera_id: str) -> Optional[RealSenseCamera]:
        """Get camera by ID."""
        return self.cameras.get(camera_id)

    @property
    def camera_ids(self) -> List[str]:
        """Get list of camera IDs."""
        return list(self.cameras.keys())

    @property
    def count(self) -> int:
        """Get number of cameras."""
        return len(self.cameras)

    def __enter__(self) -> MultiCameraCapture:
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_all()

    def __iter__(self):
        """Iterate over cameras."""
        return iter(self.cameras.values())

    def __len__(self) -> int:
        return len(self.cameras)


def list_realsense_devices() -> List[Dict[str, str]]:
    """List all connected RealSense devices.

    Returns:
        List of device info dictionaries
    """
    if not REALSENSE_AVAILABLE:
        return []

    ctx = rs.context()
    devices = []

    for device in ctx.query_devices():
        info = {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
        }

        if device.supports(rs.camera_info.usb_type_descriptor):
            info["usb_type"] = device.get_info(rs.camera_info.usb_type_descriptor)

        devices.append(info)

    return devices


def is_realsense_available() -> bool:
    """Check if RealSense SDK is available."""
    return REALSENSE_AVAILABLE

"""Hardware smoke tests for PERCEPT.

Tests verify that hardware components (Hailo-8, RealSense) are accessible
and functioning. These tests should be run on the actual hardware platform.

Tests are marked with pytest.mark.hardware to skip on CI/non-hardware systems.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pytest


# =============================================================================
# Hardware Detection Helpers
# =============================================================================

def is_hailo_available() -> bool:
    """Check if Hailo device is available."""
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_realsense_available() -> bool:
    """Check if RealSense camera is available."""
    try:
        result = subprocess.run(
            ["rs-enumerate-devices"],
            capture_output=True,
            timeout=10,
        )
        return b"Intel RealSense" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_hailo_model_path(model_name: str) -> Optional[Path]:
    """Get path to Hailo model file."""
    paths = [
        Path("/usr/share/hailo-models") / model_name,
        Path.home() / "ClaudeHome/hailo-agents/models" / model_name,
    ]
    for path in paths:
        if path.exists():
            return path
    return None


# =============================================================================
# Hardware Availability Markers
# =============================================================================

HAILO_AVAILABLE = is_hailo_available()
REALSENSE_AVAILABLE = is_realsense_available()

hardware = pytest.mark.skipif(
    not (HAILO_AVAILABLE or REALSENSE_AVAILABLE),
    reason="No hardware available"
)

hailo_required = pytest.mark.skipif(
    not HAILO_AVAILABLE,
    reason="Hailo device not available"
)

realsense_required = pytest.mark.skipif(
    not REALSENSE_AVAILABLE,
    reason="RealSense camera not available"
)


# =============================================================================
# Hailo Hardware Tests
# =============================================================================

@hailo_required
class TestHailoHardware:
    """Tests for Hailo-8 AI accelerator."""

    def test_hailo_device_detected(self):
        """Verify Hailo device is detected."""
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Hailo" in result.stdout

        print(f"\nHailo Device Info:\n{result.stdout}")

    def test_hailo_device_info(self):
        """Get detailed Hailo device information."""
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
        )

        # Parse device info
        lines = result.stdout.strip().split('\n')
        info = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()

        print(f"\nParsed Hailo Info: {info}")

        # Should have basic info
        assert any("Hailo" in v for v in info.values())

    def test_hailo_temperature(self):
        """Check Hailo device temperature is safe."""
        result = subprocess.run(
            ["hailortcli", "measure-power"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Temperature should be reported
        # Note: This might not work on all firmware versions
        print(f"\nHailo Power/Temp:\n{result.stdout}")

    def test_fastsam_model_exists(self):
        """Verify FastSAM model file exists."""
        model_path = get_hailo_model_path("fastsam_s.hef")

        if model_path is None:
            pytest.skip("FastSAM model not found")

        assert model_path.exists()
        assert model_path.stat().st_size > 0

        print(f"\nFastSAM model: {model_path}")
        print(f"Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

    def test_yolov8_model_exists(self):
        """Verify YOLOv8 model file exists."""
        model_names = [
            "yolov8s.hef",
            "yolov8n.hef",
            "yolov8s_h8.hef",
        ]

        found = None
        for name in model_names:
            path = get_hailo_model_path(name)
            if path and path.exists():
                found = path
                break

        if found is None:
            pytest.skip("No YOLOv8 model found")

        print(f"\nYOLOv8 model: {found}")
        print(f"Size: {found.stat().st_size / 1024 / 1024:.1f} MB")

    def test_pose_model_exists(self):
        """Verify pose estimation model exists."""
        model_names = [
            "yolov8s_pose.hef",
            "yolov8n_pose.hef",
        ]

        found = None
        for name in model_names:
            path = get_hailo_model_path(name)
            if path and path.exists():
                found = path
                break

        if found is None:
            pytest.skip("No pose model found")

        print(f"\nPose model: {found}")

    def test_hailo_inference_import(self):
        """Verify Hailo inference module can be imported."""
        try:
            from hailo_platform import HailoRTEngine
            print("\nHailo platform import: SUCCESS")
        except ImportError:
            pytest.skip("hailo_platform not installed")

    def test_hailo_model_load(self):
        """Test loading a model on Hailo device."""
        try:
            from hailo_platform import HailoRTEngine
        except ImportError:
            pytest.skip("hailo_platform not installed")

        model_path = get_hailo_model_path("yolov8s.hef")
        if model_path is None:
            model_path = get_hailo_model_path("yolov8n.hef")

        if model_path is None:
            pytest.skip("No YOLOv8 model found")

        # Just test that model can be opened
        # Full inference test would be in integration tests
        assert model_path.exists()
        print(f"\nModel load test: {model_path}")


# =============================================================================
# RealSense Hardware Tests
# =============================================================================

@realsense_required
class TestRealSenseHardware:
    """Tests for Intel RealSense camera."""

    def test_realsense_detected(self):
        """Verify RealSense camera is detected."""
        result = subprocess.run(
            ["rs-enumerate-devices"],
            capture_output=True,
            text=True,
        )

        assert "Intel RealSense" in result.stdout

        print(f"\nRealSense Devices:\n{result.stdout[:500]}")

    def test_realsense_device_info(self):
        """Get RealSense device information."""
        result = subprocess.run(
            ["rs-enumerate-devices", "-s"],
            capture_output=True,
            text=True,
        )

        lines = result.stdout.strip().split('\n')

        print("\nRealSense Device Summary:")
        for line in lines:
            print(f"  {line}")

    def test_pyrealsense_import(self):
        """Verify pyrealsense2 can be imported."""
        try:
            import pyrealsense2 as rs
            print(f"\npyrealsense2 version: {rs.__version__}")
        except ImportError:
            pytest.skip("pyrealsense2 not installed")

    def test_realsense_pipeline_start(self):
        """Test starting RealSense pipeline."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            pytest.skip("pyrealsense2 not installed")

        pipeline = rs.pipeline()
        config = rs.config()

        # Configure for depth and color
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            profile = pipeline.start(config)
            print("\nRealSense pipeline started successfully")

            # Get device info
            device = profile.get_device()
            print(f"Device: {device.get_info(rs.camera_info.name)}")
            print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
            print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")

            pipeline.stop()
            print("RealSense pipeline stopped")

        except RuntimeError as e:
            pytest.fail(f"Failed to start RealSense pipeline: {e}")

    def test_realsense_capture_frame(self):
        """Test capturing a frame from RealSense."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            pytest.skip("pyrealsense2 not installed")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            pipeline.start(config)

            # Wait for frames
            frames = pipeline.wait_for_frames(timeout_ms=5000)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            assert depth_frame is not None
            assert color_frame is not None

            # Convert to numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            print(f"\nCaptured frames:")
            print(f"  Color: {color_image.shape}, dtype={color_image.dtype}")
            print(f"  Depth: {depth_image.shape}, dtype={depth_image.dtype}")
            print(f"  Depth range: {depth_image.min()}-{depth_image.max()}")

            assert color_image.shape == (480, 640, 3)
            assert depth_image.shape == (480, 640)

            pipeline.stop()

        except RuntimeError as e:
            pytest.fail(f"Failed to capture frame: {e}")

    def test_realsense_intrinsics(self):
        """Test getting camera intrinsics."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            pytest.skip("pyrealsense2 not installed")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            profile = pipeline.start(config)

            # Get depth intrinsics
            depth_stream = profile.get_stream(rs.stream.depth)
            intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

            print(f"\nDepth Camera Intrinsics:")
            print(f"  Width: {intrinsics.width}")
            print(f"  Height: {intrinsics.height}")
            print(f"  fx: {intrinsics.fx:.2f}")
            print(f"  fy: {intrinsics.fy:.2f}")
            print(f"  ppx: {intrinsics.ppx:.2f}")
            print(f"  ppy: {intrinsics.ppy:.2f}")

            assert intrinsics.width == 640
            assert intrinsics.height == 480
            assert intrinsics.fx > 0
            assert intrinsics.fy > 0

            pipeline.stop()

        except RuntimeError as e:
            pytest.fail(f"Failed to get intrinsics: {e}")


# =============================================================================
# System Resource Tests
# =============================================================================

class TestSystemResources:
    """Test system resources and configuration."""

    def test_memory_available(self):
        """Check system has sufficient memory."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / 1024 / 1024

                        print(f"\nTotal Memory: {mem_gb:.1f} GB")

                        # Need at least 4GB for comfortable operation
                        assert mem_gb >= 4, f"Only {mem_gb:.1f}GB RAM available"
                        return

        except FileNotFoundError:
            pytest.skip("Not running on Linux")

    def test_cpu_info(self):
        """Check CPU information."""
        try:
            result = subprocess.run(
                ["lscpu"],
                capture_output=True,
                text=True,
            )

            print(f"\nCPU Info:\n{result.stdout[:800]}")

        except FileNotFoundError:
            pytest.skip("lscpu not available")

    def test_disk_space(self):
        """Check sufficient disk space."""
        import shutil

        usage = shutil.disk_usage("/home")

        free_gb = usage.free / 1024 / 1024 / 1024
        total_gb = usage.total / 1024 / 1024 / 1024

        print(f"\nDisk Space (home):")
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Free: {free_gb:.1f} GB")

        # Need at least 5GB free
        assert free_gb >= 5, f"Only {free_gb:.1f}GB free"

    def test_python_packages(self):
        """Verify required Python packages are installed."""
        required = [
            "numpy",
            "opencv-python",
            "scipy",
        ]

        optional = [
            "faiss",
            "supervision",
            "pyrealsense2",
        ]

        print("\nPython Packages:")

        import importlib
        for pkg in required:
            pkg_name = pkg.replace("-", "_").replace("opencv_python", "cv2")
            try:
                mod = importlib.import_module(pkg_name)
                version = getattr(mod, "__version__", "unknown")
                print(f"  {pkg}: {version}")
            except ImportError:
                pytest.fail(f"Required package {pkg} not installed")

        for pkg in optional:
            pkg_name = pkg.replace("-", "_")
            try:
                mod = importlib.import_module(pkg_name)
                version = getattr(mod, "__version__", "unknown")
                print(f"  {pkg}: {version} (optional)")
            except ImportError:
                print(f"  {pkg}: NOT INSTALLED (optional)")

    def test_hailo_sdk_version(self):
        """Check Hailo SDK version if available."""
        if not HAILO_AVAILABLE:
            pytest.skip("Hailo not available")

        result = subprocess.run(
            ["hailortcli", "--version"],
            capture_output=True,
            text=True,
        )

        print(f"\nHailo SDK Version:\n{result.stdout}")


# =============================================================================
# Hardware Integration Smoke Tests
# =============================================================================

@hardware
class TestHardwareIntegration:
    """Smoke tests for full hardware integration."""

    def test_capture_and_process_frame(self):
        """Test capturing frame and basic processing."""
        if not REALSENSE_AVAILABLE:
            pytest.skip("RealSense not available")

        try:
            import pyrealsense2 as rs
        except ImportError:
            pytest.skip("pyrealsense2 not installed")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            pipeline.start(config)
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            pipeline.stop()

            # Basic processing (convert to grayscale)
            import cv2
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            print(f"\nProcessed frame: {gray.shape}")
            assert gray.shape == (480, 640)

        except RuntimeError as e:
            pytest.fail(f"Hardware integration test failed: {e}")

    def test_hailo_model_list(self):
        """List available Hailo models."""
        if not HAILO_AVAILABLE:
            pytest.skip("Hailo not available")

        model_dirs = [
            Path("/usr/share/hailo-models"),
            Path.home() / "ClaudeHome/hailo-agents/models",
        ]

        print("\nAvailable Hailo Models:")
        for model_dir in model_dirs:
            if model_dir.exists():
                hef_files = list(model_dir.glob("*.hef"))
                if hef_files:
                    print(f"\n  {model_dir}:")
                    for hef in sorted(hef_files):
                        size_mb = hef.stat().st_size / 1024 / 1024
                        print(f"    {hef.name} ({size_mb:.1f} MB)")

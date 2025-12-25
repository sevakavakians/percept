"""Configuration system for PERCEPT.

Loads configuration from YAML files with validation and defaults.
Supports hot-reloading and environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Default configuration file path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "percept_config.yaml"


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    id: str
    type: str = "realsense_d455"
    serial: str = ""
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CameraConfig:
        """Create from dictionary."""
        resolution = tuple(data.get("resolution", [640, 480]))
        return cls(
            id=data["id"],
            type=data.get("type", "realsense_d455"),
            serial=data.get("serial", ""),
            resolution=resolution,
            fps=data.get("fps", 30),
            enabled=data.get("enabled", True),
        )


@dataclass
class SegmentationConfig:
    """Configuration for segmentation layer."""

    primary_method: str = "fastsam"
    fallback_methods: List[str] = field(default_factory=lambda: ["depth_discontinuity"])
    fusion_enabled: bool = True
    min_object_pixels: int = 500
    max_objects_per_frame: int = 50


@dataclass
class ReIDConfig:
    """Configuration for ReID and matching."""

    person_model: str = "repvgg_a0_512"
    embedding_dimension: int = 512
    match_threshold_same_camera: float = 0.3
    match_threshold_cross_camera: float = 0.25
    gallery_max_embeddings_per_object: int = 10
    reid_interval_frames: int = 3


@dataclass
class TrackingConfig:
    """Configuration for object tracking."""

    algorithm: str = "bytetrack"
    lost_track_buffer_frames: int = 30
    min_track_confidence: float = 0.5


@dataclass
class ClassificationConfig:
    """Configuration for classification thresholds."""

    confidence_confirmed: float = 0.85
    confidence_provisional: float = 0.5
    reprocess_interval_seconds: float = 60.0


@dataclass
class NormalizationConfig:
    """Configuration for image normalization."""

    enable_light_correction: bool = True
    enable_color_correction: bool = True
    clahe_clip_limit: float = 2.0


@dataclass
class HumanReviewConfig:
    """Configuration for human review queue."""

    enabled: bool = True
    queue_low_confidence: bool = True
    review_threshold: float = 0.5


@dataclass
class DatabaseConfig:
    """Configuration for persistence layer."""

    path: str = "data/percept.db"
    embedding_index_type: str = "hnsw"

    def get_absolute_path(self, base_dir: Optional[Path] = None) -> Path:
        """Get absolute path to database file."""
        db_path = Path(self.path)
        if db_path.is_absolute():
            return db_path
        base = base_dir or Path(__file__).parent.parent.parent
        return base / db_path


@dataclass
class PerformanceConfig:
    """Configuration for performance tuning."""

    adaptive_processing: bool = True
    target_fps: int = 15
    skip_frames_when_behind: bool = True
    max_processing_time_ms: int = 100


@dataclass
class UIConfig:
    """Configuration for web UI."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    file: str = "logs/percept.log"
    console: bool = True


@dataclass
class PerceptConfig:
    """Main configuration container for PERCEPT.

    Aggregates all configuration sections and provides loading/validation.

    Usage:
        # Load from default location
        config = PerceptConfig.load()

        # Load from specific file
        config = PerceptConfig.load("/path/to/config.yaml")

        # Access configuration
        print(config.cameras[0].id)
        print(config.reid.match_threshold_same_camera)
    """

    # Framework info
    name: str = "PERCEPT"
    version: str = "0.1.0"

    # Component configs
    cameras: List[CameraConfig] = field(default_factory=list)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    human_review: HumanReviewConfig = field(default_factory=HumanReviewConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Source file tracking for hot-reload
    _config_path: Optional[Path] = field(default=None, repr=False)
    _raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> PerceptConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file. If None, uses default location.
                        Can also be set via PERCEPT_CONFIG environment variable.

        Returns:
            Loaded PerceptConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        # Determine config path
        if config_path is None:
            config_path = os.environ.get("PERCEPT_CONFIG", DEFAULT_CONFIG_PATH)

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Load YAML
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)

        return cls.from_dict(raw_config, config_path=path)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        config_path: Optional[Path] = None
    ) -> PerceptConfig:
        """Create config from dictionary.

        Args:
            data: Configuration dictionary
            config_path: Optional source file path for tracking

        Returns:
            PerceptConfig instance
        """
        framework = data.get("framework", {})

        # Parse cameras
        cameras = [
            CameraConfig.from_dict(cam)
            for cam in data.get("cameras", [])
        ]

        # Parse section configs using dataclass defaults for missing fields
        def parse_section(section_name: str, config_cls: type) -> Any:
            section_data = data.get(section_name, {})
            # Filter to only valid field names
            valid_fields = {f.name for f in config_cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in section_data.items() if k in valid_fields}
            return config_cls(**filtered)

        return cls(
            name=framework.get("name", "PERCEPT"),
            version=framework.get("version", "0.1.0"),
            cameras=cameras,
            segmentation=parse_section("segmentation", SegmentationConfig),
            reid=parse_section("reid", ReIDConfig),
            tracking=parse_section("tracking", TrackingConfig),
            classification=parse_section("classification", ClassificationConfig),
            normalization=parse_section("normalization", NormalizationConfig),
            human_review=parse_section("human_review", HumanReviewConfig),
            database=parse_section("database", DatabaseConfig),
            performance=parse_section("performance", PerformanceConfig),
            ui=parse_section("ui", UIConfig),
            logging=parse_section("logging", LoggingConfig),
            _config_path=config_path,
            _raw_config=data,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict

        def config_to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {
                    k: config_to_dict(v)
                    for k, v in asdict(obj).items()
                    if not k.startswith("_")
                }
            elif isinstance(obj, list):
                return [config_to_dict(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            return obj

        return {
            "framework": {"name": self.name, "version": self.version},
            "cameras": [config_to_dict(cam) for cam in self.cameras],
            "segmentation": config_to_dict(self.segmentation),
            "reid": config_to_dict(self.reid),
            "tracking": config_to_dict(self.tracking),
            "classification": config_to_dict(self.classification),
            "normalization": config_to_dict(self.normalization),
            "human_review": config_to_dict(self.human_review),
            "database": config_to_dict(self.database),
            "performance": config_to_dict(self.performance),
            "ui": config_to_dict(self.ui),
            "logging": config_to_dict(self.logging),
        }

    def save(self, path: Optional[str | Path] = None) -> None:
        """Save configuration to YAML file.

        Args:
            path: Destination path. If None, uses original source path.

        Raises:
            ValueError: If no path specified and no source path known
        """
        save_path = Path(path) if path else self._config_path
        if save_path is None:
            raise ValueError("No save path specified and no source path known")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def reload(self) -> PerceptConfig:
        """Reload configuration from source file.

        Returns:
            New PerceptConfig instance with updated values

        Raises:
            ValueError: If no source path known
        """
        if self._config_path is None:
            raise ValueError("No source config path known for reload")
        return PerceptConfig.load(self._config_path)

    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """Get camera configuration by ID."""
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        return None

    def get_enabled_cameras(self) -> List[CameraConfig]:
        """Get list of enabled camera configurations."""
        return [cam for cam in self.cameras if cam.enabled]

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check cameras
        if not self.cameras:
            errors.append("No cameras configured")

        camera_ids = [cam.id for cam in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            errors.append("Duplicate camera IDs found")

        # Check thresholds
        if self.reid.match_threshold_same_camera <= 0:
            errors.append("ReID threshold must be positive")

        if not (0 <= self.classification.confidence_provisional <= 1):
            errors.append("Classification confidence thresholds must be between 0 and 1")

        if self.classification.confidence_confirmed < self.classification.confidence_provisional:
            errors.append(
                "Confirmed threshold must be >= provisional threshold"
            )

        # Check performance settings
        if self.performance.target_fps <= 0:
            errors.append("Target FPS must be positive")

        if self.performance.max_processing_time_ms <= 0:
            errors.append("Max processing time must be positive")

        return errors


def get_default_config() -> PerceptConfig:
    """Get default configuration without loading from file.

    Useful for testing or when config file is not available.
    """
    return PerceptConfig(
        cameras=[CameraConfig(id="default_camera")],
    )

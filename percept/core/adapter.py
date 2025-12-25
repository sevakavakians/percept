"""Data adaptation framework for PERCEPT.

Provides automatic data conversion between pipeline modules with incompatible
input/output specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class DataSpec:
    """Specification for data flowing through pipelines.

    Describes the expected format of data at a pipeline stage,
    enabling automatic adaptation between incompatible modules.

    Attributes:
        data_type: Primary type identifier ("image", "mask", "embedding", etc.)
        shape: Expected dimensions (None values mean "any size")
        dtype: NumPy dtype string ("uint8", "float32", etc.)
        color_space: For images: "BGR", "RGB", "GRAY", "LAB"
        required_fields: Required dictionary keys if data is dict-like
        optional_fields: Optional dictionary keys
        value_range: Expected (min, max) for numerical data
    """

    data_type: str
    shape: Optional[Tuple[Optional[int], ...]] = None
    dtype: Optional[str] = None
    color_space: Optional[str] = None
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    value_range: Optional[Tuple[float, float]] = None

    def matches(self, other: DataSpec) -> bool:
        """Check if this spec matches another (for compatibility)."""
        # Must match data type
        if self.data_type != other.data_type:
            return False

        # Check shape compatibility (None matches any)
        if self.shape is not None and other.shape is not None:
            if len(self.shape) != len(other.shape):
                return False
            for s1, s2 in zip(self.shape, other.shape):
                if s1 is not None and s2 is not None and s1 != s2:
                    return False

        # Check dtype
        if self.dtype is not None and other.dtype is not None:
            if self.dtype != other.dtype:
                return False

        # Check color space
        if self.color_space is not None and other.color_space is not None:
            if self.color_space != other.color_space:
                return False

        return True

    def __repr__(self) -> str:
        parts = [f"type={self.data_type}"]
        if self.shape:
            parts.append(f"shape={self.shape}")
        if self.dtype:
            parts.append(f"dtype={self.dtype}")
        if self.color_space:
            parts.append(f"color={self.color_space}")
        return f"DataSpec({', '.join(parts)})"


class PipelineData:
    """Container for data flowing through pipelines.

    Provides a flexible interface for passing heterogeneous data
    between pipeline modules while tracking metadata.

    Attributes can be accessed as properties or through dictionary-like access.

    Example:
        data = PipelineData(image=rgb_image, depth=depth_map)
        data["mask"] = segmentation_mask
        print(data.image.shape)
    """

    def __init__(self, **kwargs: Any):
        """Initialize with arbitrary named data fields."""
        self._data: Dict[str, Any] = kwargs
        self._metadata: Dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"PipelineData has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with optional default."""
        return self._data.get(key, default)

    def keys(self) -> List[str]:
        """Get all data field names."""
        return list(self._data.keys())

    def items(self) -> List[Tuple[str, Any]]:
        """Get all (name, value) pairs."""
        return list(self._data.items())

    def update(self, other: Union[Dict[str, Any], PipelineData]) -> None:
        """Update with data from another source."""
        if isinstance(other, PipelineData):
            self._data.update(other._data)
        else:
            self._data.update(other)

    def copy(self) -> PipelineData:
        """Create a shallow copy."""
        new_data = PipelineData(**self._data.copy())
        new_data._metadata = self._metadata.copy()
        return new_data

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self._metadata.get(key, default)

    def matches_spec(self, spec: DataSpec) -> bool:
        """Check if this data matches a specification."""
        # Check required fields
        for field_name in spec.required_fields:
            if field_name not in self._data:
                return False

        # Check data type specific requirements
        if spec.data_type == "image" and "image" in self._data:
            image = self._data["image"]
            if not isinstance(image, np.ndarray):
                return False

            # Check shape
            if spec.shape is not None:
                if len(image.shape) != len(spec.shape):
                    return False
                for actual, expected in zip(image.shape, spec.shape):
                    if expected is not None and actual != expected:
                        return False

            # Check dtype
            if spec.dtype is not None:
                if str(image.dtype) != spec.dtype:
                    return False

        return True

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={type(v).__name__}" for k, v in self._data.items())
        return f"PipelineData({fields})"


# Type alias for adapter functions
AdapterFunc = Callable[[PipelineData, DataSpec, DataSpec], PipelineData]


class DataAdapter:
    """Automatically adapts data between pipeline modules.

    Handles common conversions like:
    - Image resizing
    - Color space conversion (BGR <-> RGB)
    - Value normalization (0-255 <-> 0-1)
    - Region extraction from masks
    - Data type casting

    Custom adapters can be registered for specific conversions.

    Example:
        adapter = DataAdapter()

        # Register custom adapter
        adapter.register(
            ("image", "embedding"),
            lambda data, from_spec, to_spec: extract_embedding(data)
        )

        # Automatic adaptation
        adapted = adapter.adapt(data, from_spec, to_spec)
    """

    def __init__(self):
        """Initialize with default adapters."""
        self._adapters: Dict[Tuple[str, str], AdapterFunc] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in adapters for common conversions."""
        # Image transformations
        self.register(("image", "image"), self._adapt_image)

        # Mask to image (visualization)
        self.register(("mask", "image"), self._mask_to_image)

        # Generic passthrough
        self.register(("any", "any"), self._passthrough)

    def register(
        self,
        conversion: Tuple[str, str],
        adapter: AdapterFunc
    ) -> None:
        """Register an adapter for a specific conversion.

        Args:
            conversion: Tuple of (from_type, to_type)
            adapter: Function that performs the conversion
        """
        self._adapters[conversion] = adapter

    def can_adapt(self, from_spec: DataSpec, to_spec: DataSpec) -> bool:
        """Check if adaptation is possible between specs."""
        # Same type - always possible
        if from_spec.data_type == to_spec.data_type:
            return True

        # Check for registered adapter
        key = (from_spec.data_type, to_spec.data_type)
        if key in self._adapters:
            return True

        # Check for generic adapter
        if ("any", to_spec.data_type) in self._adapters:
            return True
        if (from_spec.data_type, "any") in self._adapters:
            return True
        if ("any", "any") in self._adapters:
            return True

        return False

    def adapt(
        self,
        data: PipelineData,
        from_spec: Optional[DataSpec],
        to_spec: DataSpec
    ) -> PipelineData:
        """Adapt data from one specification to another.

        Args:
            data: Input data
            from_spec: Current data specification (inferred if None)
            to_spec: Target data specification

        Returns:
            Adapted data

        Raises:
            ValueError: If adaptation is not possible
        """
        if from_spec is None:
            from_spec = self._infer_spec(data)

        # No adaptation needed
        if from_spec.matches(to_spec):
            return data

        # Find adapter
        adapter = self._find_adapter(from_spec.data_type, to_spec.data_type)
        if adapter is None:
            raise ValueError(
                f"Cannot adapt from {from_spec.data_type} to {to_spec.data_type}"
            )

        return adapter(data, from_spec, to_spec)

    def _find_adapter(
        self,
        from_type: str,
        to_type: str
    ) -> Optional[AdapterFunc]:
        """Find the best adapter for a conversion."""
        # Exact match
        key = (from_type, to_type)
        if key in self._adapters:
            return self._adapters[key]

        # Generic source
        key = ("any", to_type)
        if key in self._adapters:
            return self._adapters[key]

        # Generic target
        key = (from_type, "any")
        if key in self._adapters:
            return self._adapters[key]

        # Fallback
        if ("any", "any") in self._adapters:
            return self._adapters[("any", "any")]

        return None

    def _infer_spec(self, data: PipelineData) -> DataSpec:
        """Infer specification from data."""
        if "image" in data:
            image = data.image
            if isinstance(image, np.ndarray):
                return DataSpec(
                    data_type="image",
                    shape=image.shape,
                    dtype=str(image.dtype),
                    color_space="BGR",  # Assume OpenCV default
                )

        if "mask" in data:
            mask = data.mask
            if isinstance(mask, np.ndarray):
                return DataSpec(
                    data_type="mask",
                    shape=mask.shape,
                    dtype=str(mask.dtype),
                )

        if "embedding" in data:
            return DataSpec(
                data_type="embedding",
                required_fields=["embedding"],
            )

        return DataSpec(data_type="unknown")

    # Built-in adapter functions

    def _passthrough(
        self,
        data: PipelineData,
        from_spec: DataSpec,
        to_spec: DataSpec
    ) -> PipelineData:
        """Pass data through unchanged."""
        return data

    def _adapt_image(
        self,
        data: PipelineData,
        from_spec: DataSpec,
        to_spec: DataSpec
    ) -> PipelineData:
        """Adapt image data (resize, color space, normalize)."""
        if "image" not in data:
            return data

        image = data.image.copy()

        # Color space conversion
        if from_spec.color_space and to_spec.color_space:
            image = self._convert_color_space(
                image, from_spec.color_space, to_spec.color_space
            )

        # Resize
        if to_spec.shape is not None:
            target_h, target_w = to_spec.shape[:2]
            if target_h is not None and target_w is not None:
                if image.shape[:2] != (target_h, target_w):
                    image = cv2.resize(image, (target_w, target_h))

        # Dtype conversion
        if to_spec.dtype is not None:
            image = self._convert_dtype(image, from_spec.dtype, to_spec.dtype)

        # Value range normalization
        if to_spec.value_range is not None:
            image = self._normalize_range(
                image,
                from_spec.value_range or (0, 255),
                to_spec.value_range
            )

        result = data.copy()
        result.image = image
        return result

    def _convert_color_space(
        self,
        image: np.ndarray,
        from_space: str,
        to_space: str
    ) -> np.ndarray:
        """Convert between color spaces."""
        if from_space == to_space:
            return image

        conversions = {
            ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
            ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
            ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
            ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
            ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
            ("GRAY", "RGB"): cv2.COLOR_GRAY2RGB,
            ("BGR", "LAB"): cv2.COLOR_BGR2LAB,
            ("LAB", "BGR"): cv2.COLOR_LAB2BGR,
            ("RGB", "LAB"): cv2.COLOR_RGB2LAB,
            ("LAB", "RGB"): cv2.COLOR_LAB2RGB,
        }

        key = (from_space.upper(), to_space.upper())
        if key in conversions:
            return cv2.cvtColor(image, conversions[key])

        raise ValueError(f"Unknown color space conversion: {from_space} -> {to_space}")

    def _convert_dtype(
        self,
        image: np.ndarray,
        from_dtype: Optional[str],
        to_dtype: str
    ) -> np.ndarray:
        """Convert image dtype with appropriate scaling."""
        if str(image.dtype) == to_dtype:
            return image

        # Handle common conversions
        if to_dtype == "float32" and image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif to_dtype == "uint8" and image.dtype == np.float32:
            return (image * 255).clip(0, 255).astype(np.uint8)

        return image.astype(to_dtype)

    def _normalize_range(
        self,
        image: np.ndarray,
        from_range: Tuple[float, float],
        to_range: Tuple[float, float]
    ) -> np.ndarray:
        """Normalize value range."""
        if from_range == to_range:
            return image

        from_min, from_max = from_range
        to_min, to_max = to_range

        # Linear mapping
        normalized = (image.astype(np.float32) - from_min) / (from_max - from_min)
        scaled = normalized * (to_max - to_min) + to_min

        return scaled.astype(image.dtype)

    def _mask_to_image(
        self,
        data: PipelineData,
        from_spec: DataSpec,
        to_spec: DataSpec
    ) -> PipelineData:
        """Convert mask to viewable image."""
        if "mask" not in data:
            return data

        mask = data.mask
        if mask.dtype == np.bool_:
            image = (mask * 255).astype(np.uint8)
        else:
            image = mask.astype(np.uint8)

        # Convert to color if needed
        if to_spec.shape and len(to_spec.shape) == 3 and to_spec.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        result = data.copy()
        result.image = image
        return result

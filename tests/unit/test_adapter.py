"""Unit tests for data adaptation framework."""

import numpy as np
import pytest

from percept.core.adapter import DataSpec, DataAdapter, PipelineData


class TestDataSpec:
    """Tests for DataSpec."""

    def test_create_minimal_spec(self):
        """Test creating spec with minimal parameters."""
        spec = DataSpec(data_type="image")
        assert spec.data_type == "image"
        assert spec.shape is None
        assert spec.dtype is None

    def test_create_full_spec(self):
        """Test creating spec with all parameters."""
        spec = DataSpec(
            data_type="image",
            shape=(480, 640, 3),
            dtype="uint8",
            color_space="BGR",
            required_fields=["image"],
            value_range=(0, 255),
        )
        assert spec.shape == (480, 640, 3)
        assert spec.color_space == "BGR"

    def test_matches_same_type(self):
        """Test matching specs with same type."""
        spec1 = DataSpec(data_type="image")
        spec2 = DataSpec(data_type="image")
        assert spec1.matches(spec2)

    def test_matches_different_type(self):
        """Test matching specs with different types."""
        spec1 = DataSpec(data_type="image")
        spec2 = DataSpec(data_type="mask")
        assert not spec1.matches(spec2)

    def test_matches_compatible_shapes(self):
        """Test matching specs with compatible shapes."""
        spec1 = DataSpec(data_type="image", shape=(480, 640, 3))
        spec2 = DataSpec(data_type="image", shape=(480, 640, 3))
        assert spec1.matches(spec2)

    def test_matches_shape_with_none(self):
        """Test matching when one shape has None (any size)."""
        spec1 = DataSpec(data_type="image", shape=(None, None, 3))
        spec2 = DataSpec(data_type="image", shape=(480, 640, 3))
        assert spec1.matches(spec2)

    def test_matches_incompatible_shapes(self):
        """Test matching specs with incompatible shapes."""
        spec1 = DataSpec(data_type="image", shape=(480, 640, 3))
        spec2 = DataSpec(data_type="image", shape=(720, 1280, 3))
        assert not spec1.matches(spec2)

    def test_repr(self):
        """Test string representation."""
        spec = DataSpec(
            data_type="image",
            shape=(480, 640, 3),
            dtype="uint8",
            color_space="BGR",
        )
        repr_str = repr(spec)
        assert "image" in repr_str
        assert "BGR" in repr_str


class TestPipelineData:
    """Tests for PipelineData container."""

    def test_create_empty(self):
        """Test creating empty data container."""
        data = PipelineData()
        assert len(data.keys()) == 0

    def test_create_with_kwargs(self, sample_rgb_image, sample_depth_image):
        """Test creating with keyword arguments."""
        data = PipelineData(image=sample_rgb_image, depth=sample_depth_image)
        assert "image" in data
        assert "depth" in data

    def test_attribute_access(self, sample_rgb_image):
        """Test accessing data as attributes."""
        data = PipelineData(image=sample_rgb_image)
        assert np.array_equal(data.image, sample_rgb_image)

    def test_attribute_set(self, sample_rgb_image, sample_mask):
        """Test setting data as attributes."""
        data = PipelineData(image=sample_rgb_image)
        data.mask = sample_mask
        assert np.array_equal(data.mask, sample_mask)

    def test_dict_access(self, sample_rgb_image):
        """Test dictionary-style access."""
        data = PipelineData(image=sample_rgb_image)
        assert np.array_equal(data["image"], sample_rgb_image)

    def test_dict_set(self, sample_rgb_image, sample_mask):
        """Test dictionary-style setting."""
        data = PipelineData(image=sample_rgb_image)
        data["mask"] = sample_mask
        assert np.array_equal(data["mask"], sample_mask)

    def test_contains(self, sample_rgb_image):
        """Test 'in' operator."""
        data = PipelineData(image=sample_rgb_image)
        assert "image" in data
        assert "mask" not in data

    def test_get_with_default(self, sample_rgb_image):
        """Test get with default value."""
        data = PipelineData(image=sample_rgb_image)
        assert data.get("image") is not None
        assert data.get("missing", "default") == "default"

    def test_keys(self, sample_rgb_image, sample_depth_image):
        """Test keys method."""
        data = PipelineData(image=sample_rgb_image, depth=sample_depth_image)
        keys = data.keys()
        assert set(keys) == {"image", "depth"}

    def test_items(self, sample_rgb_image):
        """Test items method."""
        data = PipelineData(image=sample_rgb_image)
        items = data.items()
        assert len(items) == 1
        assert items[0][0] == "image"

    def test_update_from_dict(self, sample_rgb_image, sample_mask):
        """Test update from dictionary."""
        data = PipelineData(image=sample_rgb_image)
        data.update({"mask": sample_mask})
        assert "mask" in data

    def test_update_from_pipeline_data(self, sample_rgb_image, sample_mask):
        """Test update from another PipelineData."""
        data1 = PipelineData(image=sample_rgb_image)
        data2 = PipelineData(mask=sample_mask)
        data1.update(data2)
        assert "mask" in data1

    def test_copy(self, sample_rgb_image):
        """Test shallow copy."""
        data = PipelineData(image=sample_rgb_image)
        data.set_metadata("key", "value")

        copy = data.copy()

        assert "image" in copy
        assert copy.get_metadata("key") == "value"
        # Verify it's a copy (modifying one doesn't affect other)
        copy["new_key"] = 123
        assert "new_key" not in data

    def test_metadata(self):
        """Test metadata storage."""
        data = PipelineData()
        data.set_metadata("processing_time", 25.5)
        assert data.get_metadata("processing_time") == 25.5
        assert data.get_metadata("missing", "default") == "default"

    def test_matches_spec_with_required_fields(self, sample_rgb_image):
        """Test spec matching with required fields."""
        data = PipelineData(image=sample_rgb_image)
        spec = DataSpec(data_type="image", required_fields=["image"])
        assert data.matches_spec(spec)

        spec_missing = DataSpec(data_type="image", required_fields=["mask"])
        assert not data.matches_spec(spec_missing)

    def test_attribute_error_for_missing(self):
        """Test AttributeError for missing attributes."""
        data = PipelineData()
        with pytest.raises(AttributeError):
            _ = data.nonexistent

    def test_repr(self, sample_rgb_image):
        """Test string representation."""
        data = PipelineData(image=sample_rgb_image)
        repr_str = repr(data)
        assert "image" in repr_str


class TestDataAdapter:
    """Tests for DataAdapter."""

    def test_create_adapter(self, data_adapter):
        """Test creating adapter."""
        assert data_adapter is not None

    def test_can_adapt_same_type(self, data_adapter):
        """Test can_adapt for same type."""
        spec = DataSpec(data_type="image")
        assert data_adapter.can_adapt(spec, spec)

    def test_can_adapt_registered_conversion(self, data_adapter):
        """Test can_adapt for registered conversion."""
        from_spec = DataSpec(data_type="mask")
        to_spec = DataSpec(data_type="image")
        assert data_adapter.can_adapt(from_spec, to_spec)

    def test_adapt_passthrough(self, data_adapter, sample_pipeline_data):
        """Test adaptation when no changes needed."""
        spec = DataSpec(data_type="image")
        result = data_adapter.adapt(sample_pipeline_data, spec, spec)
        assert "image" in result

    def test_adapt_image_resize(self, data_adapter, sample_rgb_image):
        """Test image resizing adaptation."""
        data = PipelineData(image=sample_rgb_image)
        from_spec = DataSpec(
            data_type="image",
            shape=(480, 640, 3),
            color_space="BGR"
        )
        to_spec = DataSpec(
            data_type="image",
            shape=(240, 320, 3),
            color_space="BGR"
        )

        result = data_adapter.adapt(data, from_spec, to_spec)
        assert result.image.shape == (240, 320, 3)

    def test_adapt_color_space_bgr_to_rgb(self, data_adapter, sample_rgb_image):
        """Test BGR to RGB conversion."""
        data = PipelineData(image=sample_rgb_image)
        from_spec = DataSpec(data_type="image", color_space="BGR")
        to_spec = DataSpec(data_type="image", color_space="RGB")

        result = data_adapter.adapt(data, from_spec, to_spec)

        # Check that channels are swapped
        # Original has red rectangle at [100:300, 150:250] with BGR [0,0,255]
        # After conversion should be RGB [255,0,0]
        assert result.image[200, 200, 0] == 255  # Red channel now first
        assert result.image[200, 200, 2] == 0    # Blue channel now last

    def test_adapt_mask_to_image(self, data_adapter, sample_mask):
        """Test mask to image conversion."""
        data = PipelineData(mask=sample_mask)
        from_spec = DataSpec(data_type="mask")
        to_spec = DataSpec(data_type="image")

        result = data_adapter.adapt(data, from_spec, to_spec)
        assert "image" in result

    def test_adapt_invalid_conversion_raises(self, data_adapter):
        """Test invalid conversion raises error."""
        data = PipelineData(custom_data="test")
        from_spec = DataSpec(data_type="custom")
        to_spec = DataSpec(data_type="incompatible")

        # Remove generic adapter to test error case
        if ("any", "any") in data_adapter._adapters:
            del data_adapter._adapters[("any", "any")]

        with pytest.raises(ValueError, match="Cannot adapt"):
            data_adapter.adapt(data, from_spec, to_spec)

    def test_register_custom_adapter(self, data_adapter, sample_pipeline_data):
        """Test registering custom adapter."""
        def custom_adapter(data, from_spec, to_spec):
            result = data.copy()
            result["custom_output"] = "processed"
            return result

        data_adapter.register(("image", "custom"), custom_adapter)

        from_spec = DataSpec(data_type="image")
        to_spec = DataSpec(data_type="custom")

        result = data_adapter.adapt(sample_pipeline_data, from_spec, to_spec)
        assert result["custom_output"] == "processed"

    def test_infer_spec_for_image(self, data_adapter, sample_rgb_image):
        """Test specification inference for images."""
        data = PipelineData(image=sample_rgb_image)
        spec = data_adapter._infer_spec(data)

        assert spec.data_type == "image"
        assert spec.shape == sample_rgb_image.shape
        assert spec.dtype == str(sample_rgb_image.dtype)

    def test_infer_spec_for_mask(self, data_adapter, sample_mask):
        """Test specification inference for masks."""
        data = PipelineData(mask=sample_mask)
        spec = data_adapter._infer_spec(data)

        assert spec.data_type == "mask"

    def test_adapt_dtype_uint8_to_float32(self, data_adapter, sample_rgb_image):
        """Test dtype conversion from uint8 to float32."""
        data = PipelineData(image=sample_rgb_image)
        from_spec = DataSpec(
            data_type="image",
            dtype="uint8",
            color_space="BGR"
        )
        to_spec = DataSpec(
            data_type="image",
            dtype="float32",
            color_space="BGR"
        )

        result = data_adapter.adapt(data, from_spec, to_spec)
        assert result.image.dtype == np.float32
        assert result.image.max() <= 1.0

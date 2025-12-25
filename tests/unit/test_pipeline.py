"""Unit tests for pipeline framework."""

import pytest
from percept.core.pipeline import (
    Pipeline,
    PipelineModule,
    ModuleResult,
    PipelineRegistry,
    get_registry,
    register_module,
)
from percept.core.adapter import DataSpec, PipelineData


class TestPipelineModule:
    """Tests for PipelineModule interface."""

    def test_mock_module_implements_interface(self, mock_module):
        """Test that mock module properly implements interface."""
        assert hasattr(mock_module, "name")
        assert hasattr(mock_module, "input_spec")
        assert hasattr(mock_module, "output_spec")
        assert hasattr(mock_module, "process")

    def test_mock_module_process(self, mock_module, sample_pipeline_data):
        """Test mock module processing."""
        result = mock_module.process(sample_pipeline_data)
        assert isinstance(result, PipelineData)
        assert mock_module._process_count == 1

    def test_can_process_default(self, mock_module, sample_pipeline_data):
        """Test default can_process implementation."""
        # Mock module expects "image" type, sample_pipeline_data has image
        assert mock_module.can_process(sample_pipeline_data) is True


class TestModuleResult:
    """Tests for ModuleResult dataclass."""

    def test_create_success_result(self, sample_pipeline_data):
        """Test creating successful result."""
        result = ModuleResult(
            data=sample_pipeline_data,
            timing_ms=25.5,
            success=True,
        )
        assert result.success is True
        assert result.timing_ms == 25.5
        assert result.error is None

    def test_create_error_result(self, sample_pipeline_data):
        """Test creating error result."""
        result = ModuleResult(
            data=sample_pipeline_data,
            timing_ms=10.0,
            success=False,
            error="Processing failed",
        )
        assert result.success is False
        assert result.error == "Processing failed"


class TestPipeline:
    """Tests for Pipeline class."""

    def test_create_pipeline(self):
        """Test creating empty pipeline."""
        pipeline = Pipeline("test")
        assert pipeline.name == "test"
        assert len(pipeline) == 0

    def test_add_module(self, mock_module):
        """Test adding module to pipeline."""
        pipeline = Pipeline("test")
        pipeline.add_module(mock_module)

        assert len(pipeline) == 1
        assert mock_module.name in pipeline.module_names

    def test_add_module_at_position(self):
        """Test adding module at specific position."""
        from tests.conftest import MockPipelineModule

        pipeline = Pipeline("test")
        module1 = MockPipelineModule("module1")
        module2 = MockPipelineModule("module2")
        module3 = MockPipelineModule("module3")

        pipeline.add_module(module1)
        pipeline.add_module(module2)
        pipeline.add_module(module3, position=1)  # Insert between 1 and 2

        assert pipeline.module_names == ["module1", "module3", "module2"]

    def test_add_duplicate_module_raises(self, mock_module):
        """Test adding duplicate module name raises error."""
        pipeline = Pipeline("test")
        pipeline.add_module(mock_module)

        from tests.conftest import MockPipelineModule
        duplicate = MockPipelineModule(mock_module.name)

        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_module(duplicate)

    def test_remove_module(self, mock_module):
        """Test removing module from pipeline."""
        pipeline = Pipeline("test")
        pipeline.add_module(mock_module)

        removed = pipeline.remove_module(mock_module.name)

        assert removed is mock_module
        assert len(pipeline) == 0

    def test_remove_nonexistent_module_raises(self):
        """Test removing non-existent module raises error."""
        pipeline = Pipeline("test")

        with pytest.raises(KeyError):
            pipeline.remove_module("nonexistent")

    def test_swap_module(self):
        """Test swapping module."""
        from tests.conftest import MockPipelineModule

        pipeline = Pipeline("test")
        original = MockPipelineModule("module1")
        replacement = MockPipelineModule("module1_v2")

        pipeline.add_module(original)
        old = pipeline.swap_module("module1", replacement)

        assert old is original
        assert pipeline.get_module("module1_v2") is replacement
        assert pipeline.get_module("module1") is None

    def test_get_module(self, sample_pipeline, mock_module):
        """Test getting module by name."""
        module = sample_pipeline.get_module(mock_module.name)
        assert module is mock_module

    def test_get_module_nonexistent(self, sample_pipeline):
        """Test getting non-existent module returns None."""
        assert sample_pipeline.get_module("nonexistent") is None

    def test_process_single_module(self, sample_pipeline, sample_pipeline_data):
        """Test processing through single module."""
        result = sample_pipeline.process(sample_pipeline_data)

        assert result.success is True
        assert result.timing_ms > 0
        assert "timings" in result.metadata

    def test_process_multiple_modules(self, sample_pipeline_data):
        """Test processing through multiple modules."""
        from tests.conftest import MockPipelineModule

        pipeline = Pipeline("test", store_intermediates=True)
        pipeline.add_module(MockPipelineModule("stage1"))
        pipeline.add_module(MockPipelineModule("stage2"))
        pipeline.add_module(MockPipelineModule("stage3"))

        result = pipeline.process(sample_pipeline_data)

        assert result.success is True
        assert len(result.metadata["stages_completed"]) == 3

    def test_store_intermediates(self, sample_pipeline, sample_pipeline_data, mock_module):
        """Test intermediate output caching."""
        sample_pipeline.process(sample_pipeline_data)

        intermediate = sample_pipeline.get_intermediate(mock_module.name)
        assert intermediate is not None

    def test_get_timings(self, sample_pipeline, sample_pipeline_data, mock_module):
        """Test timing data retrieval."""
        sample_pipeline.process(sample_pipeline_data)
        timings = sample_pipeline.get_timings()

        assert mock_module.name in timings
        assert timings[mock_module.name] > 0

    def test_on_stage_complete_callback(self, sample_pipeline, sample_pipeline_data, mock_module):
        """Test stage completion callback."""
        callback_called = []

        def callback(stage_name, result):
            callback_called.append((stage_name, result.success))

        sample_pipeline.on_stage_complete(callback)
        sample_pipeline.process(sample_pipeline_data)

        assert len(callback_called) == 1
        assert callback_called[0] == (mock_module.name, True)

    def test_on_error_callback(self, sample_pipeline_data):
        """Test error callback."""
        from tests.conftest import MockPipelineModule

        class FailingModule(MockPipelineModule):
            def process(self, data):
                raise ValueError("Intentional failure")

        pipeline = Pipeline("test")
        pipeline.add_module(FailingModule("failing"))

        errors = []
        pipeline.on_error(lambda name, err: errors.append((name, str(err))))

        result = pipeline.process(sample_pipeline_data)

        assert result.success is False
        assert "failing" in result.error
        assert len(errors) == 1

    def test_validate_connections(self):
        """Test connection validation."""
        from tests.conftest import MockPipelineModule

        pipeline = Pipeline("test")
        pipeline.add_module(MockPipelineModule("module1"))
        pipeline.add_module(MockPipelineModule("module2"))

        errors = pipeline.validate_connections()
        assert len(errors) == 0  # Same type should be compatible

    def test_cleanup(self, sample_pipeline, mock_module):
        """Test cleanup releases all modules."""
        sample_pipeline.cleanup()
        assert len(sample_pipeline) == 0

    def test_modules_property(self, sample_pipeline, mock_module):
        """Test modules property returns list copy."""
        modules = sample_pipeline.modules
        assert modules == [mock_module]
        # Should be a copy
        modules.clear()
        assert len(sample_pipeline) == 1


class TestPipelineRegistry:
    """Tests for PipelineRegistry."""

    def test_register_module(self):
        """Test registering a module type."""
        from tests.conftest import MockPipelineModule

        registry = PipelineRegistry()
        registry.register_module("mock", MockPipelineModule)

        assert "mock" in registry.get_module_types()

    def test_create_module(self):
        """Test creating module from registry."""
        from tests.conftest import MockPipelineModule

        registry = PipelineRegistry()
        registry.register_module("mock", MockPipelineModule)

        module = registry.create_module("mock")
        assert isinstance(module, MockPipelineModule)

    def test_create_unknown_module_raises(self):
        """Test creating unknown module raises error."""
        registry = PipelineRegistry()

        with pytest.raises(KeyError):
            registry.create_module("unknown")

    def test_register_pipeline_template(self):
        """Test registering pipeline template."""
        registry = PipelineRegistry()
        registry.register_pipeline_template("detection", ["segment", "classify"])

        assert "detection" in registry.get_pipeline_templates()

    def test_create_pipeline_from_template(self):
        """Test creating pipeline from template."""
        from tests.conftest import MockPipelineModule

        # Create unique module classes with different default names
        class Stage1Module(MockPipelineModule):
            def __init__(self, **kwargs):
                super().__init__(name="stage1")

        class Stage2Module(MockPipelineModule):
            def __init__(self, **kwargs):
                super().__init__(name="stage2")

        registry = PipelineRegistry()
        registry.register_module("stage1", Stage1Module)
        registry.register_module("stage2", Stage2Module)
        registry.register_pipeline_template("test_pipeline", ["stage1", "stage2"])

        pipeline = registry.create_pipeline("test_pipeline")

        assert pipeline.name == "test_pipeline"
        assert len(pipeline) == 2


class TestRegisterModuleDecorator:
    """Tests for @register_module decorator."""

    def test_decorator_registers_module(self):
        """Test that decorator registers module type."""
        from tests.conftest import MockPipelineModule

        @register_module("decorated_mock")
        class DecoratedModule(MockPipelineModule):
            pass

        registry = get_registry()
        assert "decorated_mock" in registry.get_module_types()

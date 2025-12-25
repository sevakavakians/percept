"""Pipeline framework for PERCEPT.

Defines the PipelineModule interface and Pipeline orchestration class
for building hot-swappable processing pipelines.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from percept.core.adapter import DataAdapter, DataSpec, PipelineData
from percept.core.schema import ObjectSchema


@dataclass
class ModuleResult:
    """Result returned by a pipeline module.

    Attributes:
        data: The processed data
        timing_ms: Processing time in milliseconds
        success: Whether processing succeeded
        error: Error message if failed
        metadata: Additional output metadata
    """

    data: PipelineData
    timing_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineModule(ABC):
    """Base class for all pipeline modules.

    Every algorithm module in PERCEPT implements this interface for hot-swappability.
    Modules declare their input/output specifications, enabling automatic data
    adaptation between incompatible stages.

    Example:
        class PoseEstimationModule(PipelineModule):
            @property
            def name(self) -> str:
                return "pose_estimation"

            @property
            def input_spec(self) -> DataSpec:
                return DataSpec(
                    data_type="image",
                    shape=(None, None, 3),
                    color_space="RGB"
                )

            @property
            def output_spec(self) -> DataSpec:
                return DataSpec(
                    data_type="keypoints",
                    required_fields=["keypoints", "confidence"]
                )

            def process(self, data: PipelineData) -> PipelineData:
                keypoints = self._detect_pose(data.image)
                return PipelineData(keypoints=keypoints, confidence=0.95)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique module identifier."""
        pass

    @property
    @abstractmethod
    def input_spec(self) -> DataSpec:
        """What this module expects as input."""
        pass

    @property
    @abstractmethod
    def output_spec(self) -> DataSpec:
        """What this module produces as output."""
        pass

    @abstractmethod
    def process(self, data: PipelineData) -> PipelineData:
        """Process input and return output.

        Args:
            data: Input data matching input_spec

        Returns:
            Processed data matching output_spec
        """
        pass

    def can_process(self, data: PipelineData) -> bool:
        """Check if this module can process the given data.

        Default implementation checks if data matches input_spec.
        Override for custom validation logic.
        """
        return data.matches_spec(self.input_spec)

    def initialize(self) -> None:
        """Initialize module resources (e.g., load models).

        Called once when module is added to pipeline.
        Override to perform expensive initialization.
        """
        pass

    def cleanup(self) -> None:
        """Release module resources.

        Called when module is removed from pipeline.
        Override to cleanup resources like GPU memory.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class Pipeline:
    """Orchestrates a sequence of pipeline modules.

    Manages data flow between modules, automatic data adaptation,
    timing collection, and error handling.

    Features:
        - Sequential module execution
        - Automatic data adaptation between stages
        - Hot-swapping of modules
        - Intermediate output caching for debugging
        - Timing and metrics collection

    Example:
        pipeline = Pipeline("person_detection")
        pipeline.add_module(SegmentationModule())
        pipeline.add_module(PoseEstimationModule())
        pipeline.add_module(ReIDModule())

        result = pipeline.process(input_data)
    """

    def __init__(
        self,
        name: str,
        adapter: Optional[DataAdapter] = None,
        store_intermediates: bool = False,
    ):
        """Initialize pipeline.

        Args:
            name: Unique pipeline identifier
            adapter: DataAdapter for automatic conversions (uses default if None)
            store_intermediates: If True, cache output of each stage for debugging
        """
        self.name = name
        self.adapter = adapter or DataAdapter()
        self.store_intermediates = store_intermediates

        self._modules: List[PipelineModule] = []
        self._module_index: Dict[str, int] = {}
        self._intermediates: Dict[str, PipelineData] = {}
        self._timings: Dict[str, float] = {}

        # Callbacks
        self._on_stage_complete: Optional[Callable[[str, ModuleResult], None]] = None
        self._on_error: Optional[Callable[[str, Exception], None]] = None

    def add_module(self, module: PipelineModule, position: Optional[int] = None) -> None:
        """Add a module to the pipeline.

        Args:
            module: Module to add
            position: Optional position in pipeline. If None, appends to end.

        Raises:
            ValueError: If module name already exists
        """
        if module.name in self._module_index:
            raise ValueError(f"Module '{module.name}' already exists in pipeline")

        # Initialize module
        module.initialize()

        if position is None:
            self._modules.append(module)
            self._module_index[module.name] = len(self._modules) - 1
        else:
            self._modules.insert(position, module)
            # Rebuild index
            self._module_index = {m.name: i for i, m in enumerate(self._modules)}

    def remove_module(self, name: str) -> PipelineModule:
        """Remove a module from the pipeline.

        Args:
            name: Name of module to remove

        Returns:
            The removed module

        Raises:
            KeyError: If module not found
        """
        if name not in self._module_index:
            raise KeyError(f"Module '{name}' not found in pipeline")

        idx = self._module_index[name]
        module = self._modules.pop(idx)
        module.cleanup()

        # Rebuild index
        self._module_index = {m.name: i for i, m in enumerate(self._modules)}

        return module

    def swap_module(self, name: str, new_module: PipelineModule) -> PipelineModule:
        """Replace a module with a new one.

        Args:
            name: Name of module to replace
            new_module: New module to insert

        Returns:
            The replaced module

        Raises:
            KeyError: If module not found
        """
        if name not in self._module_index:
            raise KeyError(f"Module '{name}' not found in pipeline")

        idx = self._module_index[name]
        old_module = self._modules[idx]
        old_module.cleanup()

        new_module.initialize()
        self._modules[idx] = new_module

        # Update index
        del self._module_index[name]
        self._module_index[new_module.name] = idx

        return old_module

    def get_module(self, name: str) -> Optional[PipelineModule]:
        """Get a module by name."""
        if name not in self._module_index:
            return None
        return self._modules[self._module_index[name]]

    def process(self, data: PipelineData) -> ModuleResult:
        """Process data through all pipeline stages.

        Args:
            data: Input data for first module

        Returns:
            ModuleResult containing final output and metadata
        """
        self._intermediates.clear()
        self._timings.clear()

        current_data = data
        total_time = 0.0

        for i, module in enumerate(self._modules):
            try:
                # Adapt data if needed
                if not module.can_process(current_data):
                    prev_spec = self._modules[i - 1].output_spec if i > 0 else None
                    current_data = self.adapter.adapt(
                        current_data, prev_spec, module.input_spec
                    )

                # Process
                start = time.perf_counter()
                output = module.process(current_data)
                elapsed = (time.perf_counter() - start) * 1000

                self._timings[module.name] = elapsed
                total_time += elapsed

                if self.store_intermediates:
                    self._intermediates[module.name] = output

                # Callback
                if self._on_stage_complete:
                    result = ModuleResult(
                        data=output,
                        timing_ms=elapsed,
                        success=True,
                    )
                    self._on_stage_complete(module.name, result)

                current_data = output

            except Exception as e:
                if self._on_error:
                    self._on_error(module.name, e)

                return ModuleResult(
                    data=current_data,
                    timing_ms=total_time,
                    success=False,
                    error=f"Error in module '{module.name}': {str(e)}",
                )

        return ModuleResult(
            data=current_data,
            timing_ms=total_time,
            success=True,
            metadata={
                "timings": self._timings.copy(),
                "stages_completed": [m.name for m in self._modules],
            },
        )

    def get_intermediate(self, module_name: str) -> Optional[PipelineData]:
        """Get cached intermediate output from a module.

        Only available if store_intermediates=True.
        """
        return self._intermediates.get(module_name)

    def get_timings(self) -> Dict[str, float]:
        """Get timing data from last execution."""
        return self._timings.copy()

    def on_stage_complete(
        self, callback: Callable[[str, ModuleResult], None]
    ) -> None:
        """Set callback for stage completion."""
        self._on_stage_complete = callback

    def on_error(self, callback: Callable[[str, Exception], None]) -> None:
        """Set callback for errors."""
        self._on_error = callback

    @property
    def modules(self) -> List[PipelineModule]:
        """Get list of modules in order."""
        return list(self._modules)

    @property
    def module_names(self) -> List[str]:
        """Get names of modules in order."""
        return [m.name for m in self._modules]

    def validate_connections(self) -> List[str]:
        """Validate that module connections are compatible.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        for i in range(len(self._modules) - 1):
            current = self._modules[i]
            next_module = self._modules[i + 1]

            # Check if output can be adapted to next input
            if not self.adapter.can_adapt(
                current.output_spec, next_module.input_spec
            ):
                errors.append(
                    f"Cannot adapt output of '{current.name}' "
                    f"({current.output_spec.data_type}) to input of "
                    f"'{next_module.name}' ({next_module.input_spec.data_type})"
                )

        return errors

    def cleanup(self) -> None:
        """Cleanup all modules."""
        for module in self._modules:
            module.cleanup()
        self._modules.clear()
        self._module_index.clear()

    def __len__(self) -> int:
        return len(self._modules)

    def __repr__(self) -> str:
        module_names = ", ".join(self.module_names)
        return f"Pipeline(name={self.name}, modules=[{module_names}])"


class PipelineRegistry:
    """Registry for pipeline templates and module types.

    Enables dynamic pipeline construction from configuration.
    """

    def __init__(self):
        self._module_types: Dict[str, Type[PipelineModule]] = {}
        self._pipeline_templates: Dict[str, List[str]] = {}

    def register_module(self, name: str, module_class: Type[PipelineModule]) -> None:
        """Register a module type."""
        self._module_types[name] = module_class

    def register_pipeline_template(
        self, name: str, module_names: List[str]
    ) -> None:
        """Register a pipeline template."""
        self._pipeline_templates[name] = module_names

    def create_module(self, name: str, **kwargs: Any) -> PipelineModule:
        """Create a module instance by registered name."""
        if name not in self._module_types:
            raise KeyError(f"Unknown module type: {name}")
        return self._module_types[name](**kwargs)

    def create_pipeline(
        self,
        template_name: str,
        adapter: Optional[DataAdapter] = None,
        **kwargs: Any,
    ) -> Pipeline:
        """Create a pipeline from a registered template."""
        if template_name not in self._pipeline_templates:
            raise KeyError(f"Unknown pipeline template: {template_name}")

        pipeline = Pipeline(template_name, adapter=adapter)

        for module_name in self._pipeline_templates[template_name]:
            module = self.create_module(module_name, **kwargs)
            pipeline.add_module(module)

        return pipeline

    def get_module_types(self) -> List[str]:
        """Get list of registered module type names."""
        return list(self._module_types.keys())

    def get_pipeline_templates(self) -> List[str]:
        """Get list of registered pipeline template names."""
        return list(self._pipeline_templates.keys())


# Global registry instance
_registry = PipelineRegistry()


def get_registry() -> PipelineRegistry:
    """Get the global pipeline registry."""
    return _registry


def register_module(name: str) -> Callable[[Type[PipelineModule]], Type[PipelineModule]]:
    """Decorator to register a module type.

    Example:
        @register_module("pose_estimation")
        class PoseEstimationModule(PipelineModule):
            ...
    """
    def decorator(cls: Type[PipelineModule]) -> Type[PipelineModule]:
        _registry.register_module(name, cls)
        return cls
    return decorator

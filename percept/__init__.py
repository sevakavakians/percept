"""
PERCEPT: Pipeline for Entity Recognition, Classification, Extraction, and Persistence Tracking

A modular, pipeline-based vision processing framework designed for mobile robotic platforms.
"""

__version__ = "0.1.0"
__author__ = "Sevak Avakians"

from percept.core.schema import ObjectSchema, ClassificationStatus
from percept.core.pipeline import Pipeline, PipelineModule
from percept.core.config import PerceptConfig

__all__ = [
    "ObjectSchema",
    "ClassificationStatus",
    "Pipeline",
    "PipelineModule",
    "PerceptConfig",
]

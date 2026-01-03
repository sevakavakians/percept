"""ReID and tracking layer: ByteTrack integration and FAISS-backed gallery."""

from percept.tracking.reid import (
    EmbeddingType,
    ReIDConfig,
    HistogramEmbedding,
    DeepEmbedding,
    ReIDExtractor,
    ReIDMatcher,
    ReIDModule,
)
from percept.tracking.gallery import (
    GalleryConfig,
    GalleryEntry,
    FAISSGallery,
    MultiCameraGallery,
)
from percept.tracking.bytetrack import (
    TrackState,
    ByteTrackConfig,
    Track,
    SimpleIoUTracker,
    ByteTrackWrapper,
    TrackingModule,
    track_detections,
)
from percept.tracking.mask_manager import (
    MaskManagerConfig,
    MaskClaim,
    SceneMaskManager,
    MaskConflictResolver,
    MaskManagerModule,
)

__all__ = [
    # ReID
    "EmbeddingType",
    "ReIDConfig",
    "HistogramEmbedding",
    "DeepEmbedding",
    "ReIDExtractor",
    "ReIDMatcher",
    "ReIDModule",
    # Gallery
    "GalleryConfig",
    "GalleryEntry",
    "FAISSGallery",
    "MultiCameraGallery",
    # ByteTrack
    "TrackState",
    "ByteTrackConfig",
    "Track",
    "SimpleIoUTracker",
    "ByteTrackWrapper",
    "TrackingModule",
    "track_detections",
    # Mask Manager
    "MaskManagerConfig",
    "MaskClaim",
    "SceneMaskManager",
    "MaskConflictResolver",
    "MaskManagerModule",
]

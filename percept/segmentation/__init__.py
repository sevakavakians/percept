"""Segmentation layer: multi-method object segmentation and fusion.

This module provides multiple approaches to segment objects from RGB-D frames:
- FastSAM: AI-based instance segmentation (Hailo-8 accelerated)
- Depth: Depth discontinuity edge detection and connected components
- PointCloud: Euclidean clustering in 3D space
- RANSAC: Plane detection for floor/table removal
- Fusion: Combines all methods for robust segmentation
"""

from percept.segmentation.depth_seg import (
    DepthEdgeDetector,
    DepthConnectedComponents,
    DepthSegmenter,
    DepthSegmentationConfig,
    detect_depth_discontinuities,
    depth_connected_components,
)

from percept.segmentation.ransac import (
    PlaneModel,
    RANSACConfig,
    RANSACPlaneDetector,
    PlaneRemovalModule,
    detect_floor_plane,
    filter_above_plane,
)

from percept.segmentation.pointcloud_seg import (
    PointCloud,
    PointCloudConfig,
    DepthToPointCloud,
    PointCloudFilter,
    EuclideanClusterer,
    PointCloudSegmenter,
    depth_to_pointcloud,
    cluster_pointcloud,
)

from percept.segmentation.fastsam import (
    FastSAMConfig,
    FastSAMSegmenter,
    is_hailo_available,
    segment_with_fastsam,
)

from percept.segmentation.fusion import (
    FusionConfig,
    MaskMerger,
    DepthRefiner,
    SegmentationFusion,
    fuse_segmentation_results,
)

__all__ = [
    # Depth segmentation
    "DepthEdgeDetector",
    "DepthConnectedComponents",
    "DepthSegmenter",
    "DepthSegmentationConfig",
    "detect_depth_discontinuities",
    "depth_connected_components",
    # RANSAC
    "PlaneModel",
    "RANSACConfig",
    "RANSACPlaneDetector",
    "PlaneRemovalModule",
    "detect_floor_plane",
    "filter_above_plane",
    # Point cloud
    "PointCloud",
    "PointCloudConfig",
    "DepthToPointCloud",
    "PointCloudFilter",
    "EuclideanClusterer",
    "PointCloudSegmenter",
    "depth_to_pointcloud",
    "cluster_pointcloud",
    # FastSAM
    "FastSAMConfig",
    "FastSAMSegmenter",
    "is_hailo_available",
    "segment_with_fastsam",
    # Fusion
    "FusionConfig",
    "MaskMerger",
    "DepthRefiner",
    "SegmentationFusion",
    "fuse_segmentation_results",
]

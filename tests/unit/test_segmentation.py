"""Unit tests for segmentation modules."""

import numpy as np
import pytest
from datetime import datetime

from percept.core.adapter import PipelineData
from percept.core.schema import ObjectMask


class TestDepthSegmentation:
    """Tests for depth-based segmentation."""

    def test_import(self):
        """Test module imports."""
        from percept.segmentation import (
            DepthEdgeDetector,
            DepthConnectedComponents,
            DepthSegmenter,
            DepthSegmentationConfig,
        )

    def test_config_defaults(self):
        """Test default configuration values."""
        from percept.segmentation import DepthSegmentationConfig

        config = DepthSegmentationConfig()
        assert config.gradient_threshold == 0.1
        assert config.min_depth == 0.2
        assert config.max_depth == 5.0
        assert config.min_area == 500

    def test_edge_detector_create(self):
        """Test edge detector creation."""
        from percept.segmentation import DepthEdgeDetector

        detector = DepthEdgeDetector()
        assert detector.config is not None

    def test_edge_detection(self, sample_depth_image):
        """Test edge detection on depth image."""
        from percept.segmentation import DepthEdgeDetector

        detector = DepthEdgeDetector()
        edges = detector.detect_edges(sample_depth_image)

        assert edges.shape == sample_depth_image.shape
        assert edges.dtype == np.uint8
        assert edges.max() == 255 or edges.max() == 0

    def test_edge_detection_adaptive(self, sample_depth_image):
        """Test adaptive edge detection."""
        from percept.segmentation import DepthEdgeDetector

        detector = DepthEdgeDetector()
        edges = detector.detect_edges_adaptive(sample_depth_image)

        assert edges.shape == sample_depth_image.shape
        assert edges.dtype == np.uint8

    def test_connected_components_create(self):
        """Test connected components segmenter creation."""
        from percept.segmentation import DepthConnectedComponents

        segmenter = DepthConnectedComponents()
        assert segmenter.config is not None

    def test_connected_components_segment(self, sample_depth_image):
        """Test connected component segmentation."""
        from percept.segmentation import DepthConnectedComponents, DepthEdgeDetector

        edge_detector = DepthEdgeDetector()
        segmenter = DepthConnectedComponents()

        edges = edge_detector.detect_edges(sample_depth_image)
        labels, num_labels = segmenter.segment(sample_depth_image, edges)

        assert labels.shape == sample_depth_image.shape
        assert labels.dtype in [np.int32, np.int64]
        assert num_labels >= 1

    def test_extract_masks(self, sample_depth_image):
        """Test extracting masks from labels."""
        from percept.segmentation import DepthConnectedComponents, DepthEdgeDetector

        edge_detector = DepthEdgeDetector()
        segmenter = DepthConnectedComponents()

        edges = edge_detector.detect_edges(sample_depth_image)
        labels, _ = segmenter.segment(sample_depth_image, edges)
        masks = segmenter.extract_masks(labels, sample_depth_image)

        assert isinstance(masks, list)
        for mask in masks:
            assert isinstance(mask, ObjectMask)
            assert mask.mask.shape == sample_depth_image.shape

    def test_depth_segmenter_module(self, sample_depth_image):
        """Test full depth segmenter pipeline module."""
        from percept.segmentation import DepthSegmenter

        segmenter = DepthSegmenter()
        assert segmenter.name == "depth_segmenter"

        data = PipelineData(depth=sample_depth_image)
        result = segmenter.process(data)

        assert hasattr(result, 'masks')
        assert hasattr(result, 'edges')
        assert isinstance(result.masks, list)

    def test_convenience_functions(self, sample_depth_image):
        """Test convenience functions."""
        from percept.segmentation import (
            detect_depth_discontinuities,
            depth_connected_components,
        )

        edges = detect_depth_discontinuities(sample_depth_image)
        assert edges.shape == sample_depth_image.shape

        masks = depth_connected_components(sample_depth_image)
        assert isinstance(masks, list)


class TestRANSACPlaneDetection:
    """Tests for RANSAC plane detection."""

    def test_import(self):
        """Test module imports."""
        from percept.segmentation import (
            PlaneModel,
            RANSACConfig,
            RANSACPlaneDetector,
            PlaneRemovalModule,
        )

    def test_config_defaults(self):
        """Test default configuration values."""
        from percept.segmentation import RANSACConfig

        config = RANSACConfig()
        assert config.iterations == 1000
        assert config.distance_threshold == 0.01
        assert config.min_inlier_ratio == 0.1

    def test_plane_model_create(self):
        """Test plane model creation."""
        from percept.segmentation import PlaneModel

        plane = PlaneModel(
            normal=np.array([0, 0, 1]),
            point=np.array([0, 0, 0]),
            inlier_count=1000,
            inlier_ratio=0.5,
        )
        assert np.allclose(plane.normal, [0, 0, 1])

    def test_plane_distance_to_points(self):
        """Test distance calculation from plane to points."""
        from percept.segmentation import PlaneModel

        plane = PlaneModel(
            normal=np.array([0, 0, 1]),
            point=np.array([0, 0, 0]),
            inlier_count=1000,
            inlier_ratio=0.5,
        )

        points = np.array([[0, 0, 1], [0, 0, 2], [0, 0, -1]])
        distances = plane.distance_to_points(points)

        np.testing.assert_array_almost_equal(distances, [1, 2, -1])

    def test_plane_inlier_mask(self):
        """Test inlier mask generation."""
        from percept.segmentation import PlaneModel

        plane = PlaneModel(
            normal=np.array([0, 0, 1]),
            point=np.array([0, 0, 0]),
            inlier_count=1000,
            inlier_ratio=0.5,
        )

        points = np.array([[0, 0, 0.005], [0, 0, 0.02], [0, 0, 0.5]])
        mask = plane.get_inlier_mask(points, threshold=0.01)

        assert mask[0] == True  # Within threshold
        assert mask[1] == False  # Outside threshold
        assert mask[2] == False  # Far outside

    def test_detector_create(self):
        """Test detector creation."""
        from percept.segmentation import RANSACPlaneDetector

        detector = RANSACPlaneDetector()
        assert detector.config is not None

    def test_detect_plane_synthetic(self):
        """Test plane detection on synthetic data."""
        from percept.segmentation import RANSACPlaneDetector

        # Create synthetic floor plane at z=0 with some objects above
        np.random.seed(42)
        floor_points = np.random.randn(500, 3) * 0.01
        floor_points[:, 2] = 0  # Floor at z=0

        object_points = np.random.randn(100, 3) * 0.1
        object_points[:, 2] += 0.2  # Object above floor

        points = np.vstack([floor_points, object_points])

        detector = RANSACPlaneDetector()
        plane = detector.detect_plane(points)

        assert plane is not None
        # Normal should be roughly vertical
        assert abs(plane.normal[2]) > 0.9

    def test_detect_plane_too_few_points(self):
        """Test detection with too few points returns None."""
        from percept.segmentation import RANSACPlaneDetector

        points = np.random.randn(5, 3)  # Too few
        detector = RANSACPlaneDetector()
        plane = detector.detect_plane(points)

        assert plane is None

    def test_detect_multiple_planes(self):
        """Test detecting multiple planes."""
        from percept.segmentation import RANSACPlaneDetector

        np.random.seed(42)
        # Floor at z=0
        floor = np.random.randn(300, 3) * 0.01
        floor[:, 2] = 0

        # Wall at y=1
        wall = np.random.randn(300, 3) * 0.01
        wall[:, 1] = 1

        points = np.vstack([floor, wall])

        detector = RANSACPlaneDetector()
        planes = detector.detect_multiple_planes(points, max_planes=2)

        assert len(planes) >= 1

    def test_remove_plane(self):
        """Test removing plane from point cloud."""
        from percept.segmentation import RANSACPlaneDetector, PlaneModel

        # Normal pointing toward camera (negative Z means camera is at origin looking down +Z)
        # In RANSAC, normal points toward camera, so positive distance = toward camera = above plane
        plane = PlaneModel(
            normal=np.array([0, 0, -1]),  # Normal pointing toward camera
            point=np.array([0, 0, 1.0]),  # Plane at z=1.0
            inlier_count=100,
            inlier_ratio=0.5,
        )

        # Points: some on plane, some above (closer to camera = smaller z)
        points = np.array([
            [0, 0, 1.0],    # On plane
            [0, 0, 0.9],    # Slightly above (0.1m toward camera)
            [0, 0, 0.7],    # Above plane (0.3m toward camera, in valid range)
            [0, 0, 0.0],    # Too far above (1.0m toward camera)
        ])

        detector = RANSACPlaneDetector()
        filtered, mask = detector.remove_plane(
            points, plane,
            keep_above=True,
            min_height=0.02,
            max_height=0.5,
        )

        assert len(filtered) < len(points)
        # Point at z=0.7 is 0.3m above plane (toward camera), should be kept
        assert mask[2] == True

    def test_plane_removal_module(self):
        """Test PlaneRemovalModule pipeline module."""
        from percept.segmentation import PlaneRemovalModule

        np.random.seed(42)
        # Create synthetic point cloud with floor
        floor = np.random.randn(300, 3) * 0.01
        floor[:, 2] = 0
        objects = np.random.randn(100, 3) * 0.05
        objects[:, 2] += 0.1

        points = np.vstack([floor, objects])

        module = PlaneRemovalModule()
        assert module.name == "plane_removal"

        data = PipelineData(points=points)
        result = module.process(data)

        assert hasattr(result, 'plane')
        assert hasattr(result, 'plane_removed')

    def test_convenience_functions(self):
        """Test convenience functions."""
        from percept.segmentation import detect_floor_plane, filter_above_plane

        np.random.seed(42)
        points = np.random.randn(500, 3) * 0.01
        points[:, 2] = 0

        plane = detect_floor_plane(points)
        if plane is not None:
            filtered = filter_above_plane(
                np.vstack([points, np.array([[0, 0, 0.1]])]),
                plane
            )
            assert len(filtered) >= 0


class TestPointCloudSegmentation:
    """Tests for point cloud clustering."""

    def test_import(self):
        """Test module imports."""
        from percept.segmentation import (
            PointCloud,
            PointCloudConfig,
            DepthToPointCloud,
            PointCloudFilter,
            EuclideanClusterer,
        )

    def test_pointcloud_create(self):
        """Test PointCloud creation."""
        from percept.segmentation import PointCloud

        points = np.random.randn(100, 3).astype(np.float32)
        pcd = PointCloud(points=points)

        assert len(pcd) == 100
        assert pcd.colors is None

    def test_pointcloud_with_colors(self):
        """Test PointCloud with colors."""
        from percept.segmentation import PointCloud

        points = np.random.randn(100, 3).astype(np.float32)
        colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
        pcd = PointCloud(points=points, colors=colors)

        assert len(pcd) == 100
        assert pcd.colors is not None

    def test_pointcloud_copy(self):
        """Test PointCloud copy."""
        from percept.segmentation import PointCloud

        points = np.random.randn(100, 3).astype(np.float32)
        pcd = PointCloud(points=points)
        pcd_copy = pcd.copy()

        assert len(pcd_copy) == len(pcd)
        pcd_copy.points[0] = [999, 999, 999]
        assert pcd.points[0, 0] != 999  # Original unchanged

    def test_pointcloud_subset(self):
        """Test PointCloud subset extraction."""
        from percept.segmentation import PointCloud

        points = np.random.randn(100, 3).astype(np.float32)
        pcd = PointCloud(points=points)

        mask = np.zeros(100, dtype=bool)
        mask[:50] = True
        subset = pcd.subset(mask)

        assert len(subset) == 50

    def test_depth_to_pointcloud(self, sample_depth_image, mock_camera_intrinsics):
        """Test depth to point cloud conversion."""
        from percept.segmentation import DepthToPointCloud

        converter = DepthToPointCloud()
        pcd = converter.convert(sample_depth_image, mock_camera_intrinsics)

        assert len(pcd) > 0
        assert pcd.points.shape[1] == 3

    def test_depth_to_pointcloud_with_color(
        self, sample_rgb_image, sample_depth_image, mock_camera_intrinsics
    ):
        """Test conversion with color."""
        from percept.segmentation import DepthToPointCloud

        converter = DepthToPointCloud()
        pcd = converter.convert(
            sample_depth_image, mock_camera_intrinsics, color=sample_rgb_image
        )

        assert len(pcd) > 0
        assert pcd.colors is not None

    def test_pointcloud_filter_voxel(self):
        """Test voxel downsampling."""
        from percept.segmentation import PointCloudFilter, PointCloud

        points = np.random.randn(10000, 3).astype(np.float32)
        pcd = PointCloud(points=points)

        filter = PointCloudFilter()
        downsampled = filter.downsample_voxel(pcd, voxel_size=0.1)

        assert len(downsampled) < len(pcd)

    def test_pointcloud_filter_statistical(self):
        """Test statistical outlier removal."""
        from percept.segmentation import PointCloudFilter, PointCloud

        # Create points with outliers
        np.random.seed(42)
        main_points = np.random.randn(1000, 3).astype(np.float32) * 0.1
        outliers = np.random.randn(50, 3).astype(np.float32) * 10
        points = np.vstack([main_points, outliers])
        pcd = PointCloud(points=points)

        filter = PointCloudFilter()
        filtered = filter.filter_statistical(pcd)

        assert len(filtered) < len(pcd)

    def test_euclidean_clusterer(self):
        """Test Euclidean clustering."""
        from percept.segmentation import EuclideanClusterer, PointCloud

        np.random.seed(42)
        # Create two separate clusters
        cluster1 = np.random.randn(200, 3).astype(np.float32) * 0.02
        cluster2 = np.random.randn(200, 3).astype(np.float32) * 0.02
        cluster2[:, 0] += 0.5  # Offset second cluster

        points = np.vstack([cluster1, cluster2])
        pcd = PointCloud(points=points)

        clusterer = EuclideanClusterer()
        clusters = clusterer.cluster(pcd, cluster_tolerance=0.05, min_cluster_size=50)

        assert len(clusters) == 2

    def test_cluster_to_mask(self, mock_camera_intrinsics):
        """Test converting cluster to 2D mask."""
        from percept.segmentation import EuclideanClusterer, PointCloud

        # Create synthetic cluster
        points = np.array([
            [0, 0, 1.0],
            [0.01, 0, 1.0],
            [0, 0.01, 1.0],
        ], dtype=np.float32)
        pcd = PointCloud(points=points)

        clusterer = EuclideanClusterer()
        mask = clusterer.cluster_to_mask(pcd, (480, 640), mock_camera_intrinsics)

        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8

    def test_pointcloud_segmenter_module(
        self, sample_depth_image, mock_camera_intrinsics
    ):
        """Test full point cloud segmenter pipeline module."""
        from percept.segmentation import PointCloudSegmenter

        segmenter = PointCloudSegmenter()
        assert segmenter.name == "pointcloud_segmenter"

        data = PipelineData(depth=sample_depth_image, intrinsics=mock_camera_intrinsics)
        result = segmenter.process(data)

        assert hasattr(result, 'masks')
        assert hasattr(result, 'clusters')

    def test_convenience_functions(self, sample_depth_image, mock_camera_intrinsics):
        """Test convenience functions."""
        from percept.segmentation import depth_to_pointcloud, cluster_pointcloud

        pcd = depth_to_pointcloud(sample_depth_image, mock_camera_intrinsics)
        assert len(pcd) > 0


class TestFastSAMSegmentation:
    """Tests for FastSAM segmentation."""

    def test_import(self):
        """Test module imports."""
        from percept.segmentation import (
            FastSAMConfig,
            FastSAMSegmenter,
            is_hailo_available,
        )

    def test_hailo_availability_check(self):
        """Test Hailo availability check."""
        from percept.segmentation import is_hailo_available

        result = is_hailo_available()
        assert isinstance(result, bool)

    def test_config_defaults(self):
        """Test default configuration values."""
        from percept.segmentation import FastSAMConfig

        config = FastSAMConfig()
        assert config.conf_threshold == 0.25
        assert config.iou_threshold == 0.5
        assert config.max_detections == 20

    def test_segmenter_create(self):
        """Test FastSAM segmenter creation (works without hardware)."""
        from percept.segmentation import FastSAMSegmenter

        segmenter = FastSAMSegmenter()
        assert segmenter.name == "fastsam_segmenter"

    def test_segmenter_process_no_hardware(self, sample_rgb_image):
        """Test processing without Hailo hardware returns empty."""
        from percept.segmentation import FastSAMSegmenter

        segmenter = FastSAMSegmenter()

        data = PipelineData(image=sample_rgb_image)
        result = segmenter.process(data)

        # Without hardware, should return empty masks
        assert hasattr(result, 'masks')
        if not segmenter.is_available():
            assert result.masks == []


class TestSegmentationFusion:
    """Tests for segmentation fusion."""

    def test_import(self):
        """Test module imports."""
        from percept.segmentation import (
            FusionConfig,
            MaskMerger,
            DepthRefiner,
            SegmentationFusion,
        )

    def test_config_defaults(self):
        """Test default configuration values."""
        from percept.segmentation import FusionConfig

        config = FusionConfig()
        assert config.merge_iou_threshold == 0.5
        assert config.agreement_iou_threshold == 0.3
        assert config.agreement_boost == 0.1

    def test_compute_iou(self, sample_mask):
        """Test IoU computation."""
        from percept.segmentation import MaskMerger

        merger = MaskMerger()

        # Same mask should have IoU of 1
        iou = merger.compute_iou(sample_mask, sample_mask)
        assert iou == 1.0

        # Non-overlapping masks should have IoU of 0
        empty_mask = np.zeros_like(sample_mask)
        iou = merger.compute_iou(sample_mask, empty_mask)
        assert iou == 0.0

    def test_compute_overlap(self, sample_mask):
        """Test overlap computation."""
        from percept.segmentation import MaskMerger

        merger = MaskMerger()

        # Same mask
        overlap = merger.compute_overlap(sample_mask, sample_mask)
        assert overlap == 1.0

        # Partial overlap
        partial_mask = np.zeros_like(sample_mask)
        partial_mask[100:200, 150:200] = 255  # Overlap with sample_mask
        overlap = merger.compute_overlap(sample_mask, partial_mask)
        assert 0 < overlap <= 1.0

    def test_merge_masks_union(self, sample_mask):
        """Test mask merging with union mode."""
        from percept.segmentation import MaskMerger

        mask1 = ObjectMask(
            mask=sample_mask,
            bbox=(150, 100, 250, 300),
            confidence=0.9,
            depth_median=2.0,
            point_count=1000,
        )

        # Create second mask with slight offset
        mask2_data = np.zeros_like(sample_mask)
        mask2_data[150:350, 200:300] = 255
        mask2 = ObjectMask(
            mask=mask2_data,
            bbox=(200, 150, 300, 350),
            confidence=0.8,
            depth_median=2.1,
            point_count=900,
        )

        merger = MaskMerger()
        merged = merger.merge_masks([mask1, mask2], mode='union')

        assert merged.mask.sum() >= mask1.mask.sum()
        assert merged.confidence == 0.9  # Max for union

    def test_merge_masks_intersection(self, sample_mask):
        """Test mask merging with intersection mode."""
        from percept.segmentation import MaskMerger

        mask1 = ObjectMask(
            mask=sample_mask,
            bbox=(150, 100, 250, 300),
            confidence=0.9,
            depth_median=2.0,
            point_count=1000,
        )

        # Overlapping mask
        mask2_data = np.zeros_like(sample_mask)
        mask2_data[150:250, 175:225] = 255
        mask2 = ObjectMask(
            mask=mask2_data,
            bbox=(175, 150, 225, 250),
            confidence=0.8,
            depth_median=2.0,
            point_count=500,
        )

        merger = MaskMerger()
        merged = merger.merge_masks([mask1, mask2], mode='intersection')

        assert merged.mask.sum() <= mask1.mask.sum()
        assert merged.confidence == 0.8  # Min for intersection

    def test_find_matching_masks(self, sample_mask):
        """Test finding matching masks."""
        from percept.segmentation import MaskMerger

        query = ObjectMask(
            mask=sample_mask,
            bbox=(150, 100, 250, 300),
            confidence=0.9,
            depth_median=2.0,
            point_count=1000,
        )

        # Matching mask
        candidate1 = ObjectMask(
            mask=sample_mask.copy(),
            bbox=(150, 100, 250, 300),
            confidence=0.8,
            depth_median=2.0,
            point_count=1000,
        )

        # Non-matching mask
        other_mask = np.zeros_like(sample_mask)
        other_mask[0:50, 0:50] = 255
        candidate2 = ObjectMask(
            mask=other_mask,
            bbox=(0, 0, 50, 50),
            confidence=0.7,
            depth_median=3.0,
            point_count=500,
        )

        merger = MaskMerger()
        matches = merger.find_matching_masks(query, [candidate1, candidate2], iou_threshold=0.5)

        assert len(matches) == 1
        assert matches[0][0] == 0  # Index of candidate1
        assert matches[0][1] == 1.0  # IoU

    def test_depth_refiner(self, sample_mask, sample_depth_image):
        """Test depth-based mask refinement."""
        from percept.segmentation import DepthRefiner

        mask = ObjectMask(
            mask=sample_mask,
            bbox=(150, 100, 250, 300),
            confidence=0.9,
            depth_median=2.0,
            point_count=1000,
        )

        refiner = DepthRefiner()
        refined = refiner.refine_mask(mask, sample_depth_image)

        assert isinstance(refined, ObjectMask)
        assert refined.mask.shape == mask.mask.shape

    def test_segmentation_fusion_module(self, sample_mask, sample_depth_image):
        """Test SegmentationFusion pipeline module."""
        from percept.segmentation import SegmentationFusion

        fusion = SegmentationFusion()
        assert fusion.name == "segmentation_fusion"

        # Create test masks from different methods
        fastsam_masks = [ObjectMask(
            mask=sample_mask,
            bbox=(150, 100, 250, 300),
            confidence=0.9,
            depth_median=2.0,
            point_count=1000,
        )]

        depth_masks = [ObjectMask(
            mask=sample_mask.copy(),
            bbox=(150, 100, 250, 300),
            confidence=0.7,
            depth_median=2.0,
            point_count=1000,
        )]

        data = PipelineData(
            masks_fastsam=fastsam_masks,
            masks_depth=depth_masks,
            masks_pointcloud=[],
            depth=sample_depth_image,
        )

        result = fusion.process(data)

        assert hasattr(result, 'masks')
        assert isinstance(result.masks, list)
        # Should fuse the overlapping masks
        assert len(result.masks) >= 1

    def test_convenience_function(self, sample_mask, sample_depth_image):
        """Test convenience fusion function."""
        from percept.segmentation import fuse_segmentation_results

        fastsam_masks = [ObjectMask(
            mask=sample_mask,
            bbox=(150, 100, 250, 300),
            confidence=0.9,
            depth_median=2.0,
            point_count=1000,
        )]

        fused = fuse_segmentation_results(
            fastsam_masks=fastsam_masks,
            depth_masks=[],
            pointcloud_masks=[],
            depth=sample_depth_image,
        )

        assert isinstance(fused, list)


class TestSegmentationModuleExports:
    """Test that all expected classes are exported from __init__.py."""

    def test_all_exports(self):
        """Verify all expected exports are available."""
        from percept.segmentation import (
            # Depth
            DepthEdgeDetector,
            DepthConnectedComponents,
            DepthSegmenter,
            DepthSegmentationConfig,
            detect_depth_discontinuities,
            depth_connected_components,
            # RANSAC
            PlaneModel,
            RANSACConfig,
            RANSACPlaneDetector,
            PlaneRemovalModule,
            detect_floor_plane,
            filter_above_plane,
            # Point cloud
            PointCloud,
            PointCloudConfig,
            DepthToPointCloud,
            PointCloudFilter,
            EuclideanClusterer,
            PointCloudSegmenter,
            depth_to_pointcloud,
            cluster_pointcloud,
            # FastSAM
            FastSAMConfig,
            FastSAMSegmenter,
            is_hailo_available,
            segment_with_fastsam,
            # Fusion
            FusionConfig,
            MaskMerger,
            DepthRefiner,
            SegmentationFusion,
            fuse_segmentation_results,
        )
        # If we get here, all imports succeeded
        assert True

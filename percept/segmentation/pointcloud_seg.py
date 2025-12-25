"""Point cloud segmentation for PERCEPT.

Euclidean clustering for object segmentation in 3D space.
Uses scipy KDTree for ARM64 compatibility (no Open3D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.core.schema import ObjectMask


@dataclass
class PointCloud:
    """Simple point cloud representation.

    Attributes:
        points: (N, 3) XYZ coordinates in meters
        colors: Optional (N, 3) RGB values 0-255
        normals: Optional (N, 3) normal vectors
        indices: Optional (N,) original pixel indices for back-projection
    """

    points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.points)

    def __post_init__(self):
        """Validate arrays have consistent shapes."""
        n = len(self.points)
        if self.colors is not None and len(self.colors) != n:
            raise ValueError(f"colors length {len(self.colors)} != points {n}")
        if self.normals is not None and len(self.normals) != n:
            raise ValueError(f"normals length {len(self.normals)} != points {n}")

    def copy(self) -> PointCloud:
        """Create a deep copy."""
        return PointCloud(
            points=self.points.copy(),
            colors=self.colors.copy() if self.colors is not None else None,
            normals=self.normals.copy() if self.normals is not None else None,
            indices=self.indices.copy() if self.indices is not None else None,
        )

    def subset(self, mask: np.ndarray) -> PointCloud:
        """Extract subset using boolean mask."""
        return PointCloud(
            points=self.points[mask],
            colors=self.colors[mask] if self.colors is not None else None,
            normals=self.normals[mask] if self.normals is not None else None,
            indices=self.indices[mask] if self.indices is not None else None,
        )


@dataclass
class PointCloudConfig:
    """Configuration for point cloud operations."""

    # Depth to point cloud
    min_depth: float = 0.2  # Minimum valid depth (meters)
    max_depth: float = 5.0  # Maximum valid depth (meters)
    max_points: int = 100000  # Maximum points to keep

    # Voxel downsampling
    voxel_size: float = 0.005  # 5mm voxel for downsampling

    # Clustering
    cluster_tolerance: float = 0.02  # 2cm between neighbors
    min_cluster_size: int = 100  # Minimum points per cluster
    max_cluster_size: int = 50000  # Maximum points per cluster

    # Statistical outlier removal
    outlier_neighbors: int = 20  # Neighbors for outlier detection
    outlier_std_ratio: float = 2.0  # Std dev ratio threshold


class DepthToPointCloud:
    """Convert RGB-D frames to point clouds.

    Uses camera intrinsics to back-project depth pixels to 3D.
    """

    def __init__(self, config: Optional[PointCloudConfig] = None):
        """Initialize converter.

        Args:
            config: Configuration options
        """
        self.config = config or PointCloudConfig()

    def convert(
        self,
        depth: np.ndarray,
        intrinsics: Dict[str, float],
        color: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> PointCloud:
        """Convert depth image to point cloud.

        Args:
            depth: Depth image in meters (H, W), float32
            intrinsics: Camera intrinsics with fx, fy, ppx, ppy
            color: Optional RGB image (H, W, 3), uint8
            mask: Optional binary mask (H, W), only convert where mask > 0

        Returns:
            PointCloud with XYZ points and optional colors
        """
        h, w = depth.shape

        # Get intrinsics
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics.get('ppx', w / 2)
        cy = intrinsics.get('ppy', h / 2)

        # Create pixel coordinate grids
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)

        # Valid depth mask
        valid = (depth > self.config.min_depth) & (depth < self.config.max_depth)
        if mask is not None:
            valid = valid & (mask > 0)

        # Get valid pixels
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = depth[valid]

        # Back-project to 3D (pinhole camera model)
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        z = z_valid

        points = np.stack([x, y, z], axis=-1).astype(np.float32)

        # Get colors if provided
        colors = None
        if color is not None:
            if color.shape[2] == 3:
                # Assume BGR, convert to RGB
                colors = color[valid][:, ::-1].copy()
            else:
                colors = color[valid][:, :3].copy()

        # Store original pixel indices for back-projection
        indices = np.where(valid.flatten())[0]

        # Limit points if too many
        if len(points) > self.config.max_points:
            sample_idx = np.random.choice(
                len(points), self.config.max_points, replace=False
            )
            points = points[sample_idx]
            if colors is not None:
                colors = colors[sample_idx]
            indices = indices[sample_idx]

        return PointCloud(points=points, colors=colors, indices=indices)


class PointCloudFilter:
    """Filter point clouds to remove noise and downsample."""

    def __init__(self, config: Optional[PointCloudConfig] = None):
        """Initialize filter.

        Args:
            config: Configuration options
        """
        self.config = config or PointCloudConfig()

    def downsample_voxel(
        self,
        pcd: PointCloud,
        voxel_size: Optional[float] = None,
    ) -> PointCloud:
        """Voxel grid downsampling.

        Groups points into voxels and keeps centroid of each.

        Args:
            pcd: Input point cloud
            voxel_size: Voxel size in meters (default from config)

        Returns:
            Downsampled point cloud
        """
        if voxel_size is None:
            voxel_size = self.config.voxel_size

        if len(pcd) == 0:
            return pcd

        # Compute voxel indices
        voxel_indices = np.floor(pcd.points / voxel_size).astype(np.int32)

        # Offset to handle negative indices
        min_idx = voxel_indices.min(axis=0)
        voxel_indices = voxel_indices - min_idx

        # Create linear index
        max_idx = voxel_indices.max(axis=0) + 1
        linear_idx = (
            voxel_indices[:, 0] * max_idx[1] * max_idx[2] +
            voxel_indices[:, 1] * max_idx[2] +
            voxel_indices[:, 2]
        )

        # Find unique voxels and compute centroids
        unique_voxels, inverse, counts = np.unique(
            linear_idx, return_inverse=True, return_counts=True
        )

        # Average points per voxel
        new_points = np.zeros((len(unique_voxels), 3), dtype=np.float32)
        np.add.at(new_points, inverse, pcd.points)
        new_points /= counts[:, np.newaxis]

        # Average colors if present
        new_colors = None
        if pcd.colors is not None:
            new_colors = np.zeros((len(unique_voxels), 3), dtype=np.float32)
            np.add.at(new_colors, inverse, pcd.colors.astype(np.float32))
            new_colors /= counts[:, np.newaxis]
            new_colors = new_colors.astype(np.uint8)

        return PointCloud(points=new_points, colors=new_colors)

    def filter_statistical(
        self,
        pcd: PointCloud,
        nb_neighbors: Optional[int] = None,
        std_ratio: Optional[float] = None,
    ) -> PointCloud:
        """Remove statistical outliers.

        Points whose mean distance to neighbors exceeds threshold are removed.

        Args:
            pcd: Input point cloud
            nb_neighbors: Number of neighbors to consider
            std_ratio: Standard deviation threshold ratio

        Returns:
            Filtered point cloud
        """
        if nb_neighbors is None:
            nb_neighbors = self.config.outlier_neighbors
        if std_ratio is None:
            std_ratio = self.config.outlier_std_ratio

        if len(pcd) < nb_neighbors:
            return pcd

        # Build KD-tree
        tree = cKDTree(pcd.points)

        # Query k nearest neighbors for each point
        distances, _ = tree.query(pcd.points, k=nb_neighbors + 1)

        # Mean distance to neighbors (excluding self at index 0)
        mean_dists = distances[:, 1:].mean(axis=1)

        # Compute threshold
        global_mean = mean_dists.mean()
        global_std = mean_dists.std()
        threshold = global_mean + std_ratio * global_std

        # Filter
        mask = mean_dists < threshold

        return pcd.subset(mask)


class EuclideanClusterer:
    """Segment point cloud into object clusters.

    Uses BFS flood-fill with KDTree for efficient neighbor queries.
    """

    def __init__(self, config: Optional[PointCloudConfig] = None):
        """Initialize clusterer.

        Args:
            config: Configuration options
        """
        self.config = config or PointCloudConfig()

    def cluster(
        self,
        pcd: PointCloud,
        cluster_tolerance: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
        max_cluster_size: Optional[int] = None,
    ) -> List[PointCloud]:
        """Segment point cloud into distinct clusters.

        Args:
            pcd: Input point cloud
            cluster_tolerance: Max distance between neighbors in cluster
            min_cluster_size: Minimum points per cluster
            max_cluster_size: Maximum points per cluster

        Returns:
            List of point clouds, one per cluster, sorted by size (largest first)
        """
        if cluster_tolerance is None:
            cluster_tolerance = self.config.cluster_tolerance
        if min_cluster_size is None:
            min_cluster_size = self.config.min_cluster_size
        if max_cluster_size is None:
            max_cluster_size = self.config.max_cluster_size

        points = pcd.points
        n_points = len(points)

        if n_points < min_cluster_size:
            return []

        # Build KD-tree for efficient neighbor queries
        tree = cKDTree(points)
        visited = np.zeros(n_points, dtype=bool)
        clusters = []

        for seed_idx in range(n_points):
            if visited[seed_idx]:
                continue

            # BFS flood-fill to grow cluster
            cluster_indices = []
            queue = [seed_idx]

            while queue:
                idx = queue.pop(0)
                if visited[idx]:
                    continue

                visited[idx] = True
                cluster_indices.append(idx)

                # Early termination if cluster too large
                if len(cluster_indices) > max_cluster_size:
                    break

                # Find neighbors within tolerance
                neighbors = tree.query_ball_point(points[idx], cluster_tolerance)
                for n_idx in neighbors:
                    if not visited[n_idx]:
                        queue.append(n_idx)

            # Check cluster size
            if min_cluster_size <= len(cluster_indices) <= max_cluster_size:
                idx_array = np.array(cluster_indices)
                clusters.append(PointCloud(
                    points=points[idx_array].copy(),
                    colors=pcd.colors[idx_array].copy() if pcd.colors is not None else None,
                    indices=pcd.indices[idx_array].copy() if pcd.indices is not None else None,
                ))

        # Sort by size (largest first)
        clusters.sort(key=lambda c: len(c), reverse=True)
        return clusters

    def cluster_to_mask(
        self,
        cluster: PointCloud,
        depth_shape: Tuple[int, int],
        intrinsics: Dict[str, float],
    ) -> np.ndarray:
        """Convert 3D cluster back to 2D binary mask.

        Projects 3D points onto image plane and fills the region.

        Args:
            cluster: Point cloud cluster
            depth_shape: (height, width) of depth image
            intrinsics: Camera intrinsics with fx, fy, ppx, ppy

        Returns:
            Binary mask (H, W) with 255 for object pixels
        """
        h, w = depth_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(cluster) == 0:
            return mask

        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics.get('ppx', w / 2)
        cy = intrinsics.get('ppy', h / 2)

        # Project 3D points to 2D
        points = cluster.points
        z = points[:, 2]
        valid = z > 0.01

        if not np.any(valid):
            return mask

        u = (points[valid, 0] * fx / z[valid] + cx).astype(np.int32)
        v = (points[valid, 1] * fy / z[valid] + cy).astype(np.int32)

        # Clip to image bounds
        u = np.clip(u, 0, w - 1)
        v = np.clip(v, 0, h - 1)

        # Set mask pixels
        mask[v, u] = 255

        # Morphological closing to fill gaps
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=2)

        # Fill contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, -1)

        return mask


class PointCloudSegmenter(PipelineModule):
    """Pipeline module for point cloud-based object segmentation.

    Converts depth to point cloud, removes floor plane, clusters objects,
    and outputs masks.
    """

    def __init__(self, config: Optional[PointCloudConfig] = None):
        """Initialize segmenter.

        Args:
            config: Configuration options
        """
        self.config = config or PointCloudConfig()
        self.converter = DepthToPointCloud(self.config)
        self.filter = PointCloudFilter(self.config)
        self.clusterer = EuclideanClusterer(self.config)

    @property
    def name(self) -> str:
        return "pointcloud_segmenter"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="rgbd",
            required_fields=["depth", "intrinsics"],
            optional_fields=["image", "points", "plane"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="masks",
            required_fields=["masks", "clusters"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Segment objects using point cloud clustering.

        Args:
            data: PipelineData with depth, intrinsics, optional plane

        Returns:
            PipelineData with masks and cluster info
        """
        depth = data.depth
        intrinsics = data.intrinsics
        color = data.get("image")

        # Convert intrinsics to dict if needed
        if hasattr(intrinsics, '__dict__'):
            intrinsics_dict = {
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
            }
        elif isinstance(intrinsics, dict):
            intrinsics_dict = intrinsics
        else:
            # Assume it's a matrix, extract parameters
            intrinsics_dict = {
                'fx': float(intrinsics[0, 0]),
                'fy': float(intrinsics[1, 1]),
                'ppx': float(intrinsics[0, 2]),
                'ppy': float(intrinsics[1, 2]),
            }

        # Use existing point cloud if available, otherwise convert
        if hasattr(data, 'points') and data.points is not None:
            pcd = PointCloud(points=data.points)
            if hasattr(data, 'colors') and data.colors is not None:
                pcd.colors = data.colors
        else:
            pcd = self.converter.convert(depth, intrinsics_dict, color)

        # Optional: use plane mask from RANSAC to filter points
        plane = data.get("plane")
        if plane is not None and hasattr(plane, 'get_inlier_mask'):
            # Remove plane points
            plane_mask = plane.get_inlier_mask(pcd.points, threshold=0.02)
            pcd = pcd.subset(~plane_mask)

        # Filter outliers
        pcd = self.filter.filter_statistical(pcd)

        # Optionally downsample for faster clustering
        if len(pcd) > 50000:
            pcd = self.filter.downsample_voxel(pcd)

        # Cluster into objects
        clusters = self.clusterer.cluster(pcd)

        # Convert clusters to masks
        depth_shape = depth.shape[:2]
        masks = []

        for i, cluster in enumerate(clusters):
            mask_img = self.clusterer.cluster_to_mask(
                cluster, depth_shape, intrinsics_dict
            )

            # Compute bounding box
            ys, xs = np.where(mask_img > 0)
            if len(xs) == 0:
                continue

            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            area = mask_img.sum() // 255

            # Compute median depth
            cluster_depths = cluster.points[:, 2]
            depth_median = float(np.median(cluster_depths))

            masks.append(ObjectMask(
                mask=mask_img,
                bbox=bbox,
                confidence=0.7,  # Lower confidence for geometric methods
                depth_median=depth_median,
                point_count=len(cluster),
            ))

        # Build result
        result = data.copy()
        result.masks = masks
        result.clusters = clusters
        result.pointcloud = pcd

        return result


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: Dict[str, float],
    color: Optional[np.ndarray] = None,
    min_depth: float = 0.2,
    max_depth: float = 5.0,
) -> PointCloud:
    """Convenience function to convert depth to point cloud.

    Args:
        depth: Depth image in meters (H, W)
        intrinsics: Camera intrinsics dict
        color: Optional RGB image
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth

    Returns:
        PointCloud instance
    """
    config = PointCloudConfig(min_depth=min_depth, max_depth=max_depth)
    converter = DepthToPointCloud(config)
    return converter.convert(depth, intrinsics, color)


def cluster_pointcloud(
    pcd: PointCloud,
    tolerance: float = 0.02,
    min_size: int = 100,
    max_size: int = 50000,
) -> List[PointCloud]:
    """Convenience function for point cloud clustering.

    Args:
        pcd: Input point cloud
        tolerance: Cluster tolerance in meters
        min_size: Minimum cluster size
        max_size: Maximum cluster size

    Returns:
        List of cluster point clouds
    """
    config = PointCloudConfig(
        cluster_tolerance=tolerance,
        min_cluster_size=min_size,
        max_cluster_size=max_size,
    )
    clusterer = EuclideanClusterer(config)
    return clusterer.cluster(pcd)

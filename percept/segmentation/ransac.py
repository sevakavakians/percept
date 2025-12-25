"""RANSAC plane detection for PERCEPT.

Detects dominant planes (floor, table, walls) in point clouds for removal
before object segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData


@dataclass
class PlaneModel:
    """Represents a detected plane.

    Attributes:
        normal: Unit normal vector (3,)
        point: A point on the plane (3,)
        inlier_count: Number of inlier points
        inlier_ratio: Ratio of inliers to total points
    """

    normal: np.ndarray
    point: np.ndarray
    inlier_count: int
    inlier_ratio: float

    def distance_to_points(self, points: np.ndarray) -> np.ndarray:
        """Compute signed distance from points to plane.

        Args:
            points: (N, 3) array of points

        Returns:
            (N,) array of signed distances
        """
        return np.dot(points - self.point, self.normal)

    def get_inlier_mask(
        self,
        points: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Get mask of points within threshold of plane.

        Args:
            points: (N, 3) array of points
            threshold: Maximum distance to be considered inlier

        Returns:
            (N,) boolean mask
        """
        distances = np.abs(self.distance_to_points(points))
        return distances < threshold


@dataclass
class RANSACConfig:
    """Configuration for RANSAC plane detection."""

    iterations: int = 1000
    distance_threshold: float = 0.01  # 1cm
    min_inlier_ratio: float = 0.1  # Minimum 10% inliers
    require_horizontal: bool = False  # Only detect horizontal planes
    horizontal_threshold: float = 0.7  # |normal.z| > 0.7 for horizontal


class RANSACPlaneDetector:
    """Detect planes in point clouds using RANSAC.

    RANSAC (Random Sample Consensus) finds the dominant plane by:
    1. Randomly sampling 3 points
    2. Fitting a plane through them
    3. Counting points within threshold distance
    4. Keeping the plane with most inliers
    """

    def __init__(self, config: Optional[RANSACConfig] = None):
        """Initialize detector.

        Args:
            config: Configuration options
        """
        self.config = config or RANSACConfig()

    def detect_plane(
        self,
        points: np.ndarray,
    ) -> Optional[PlaneModel]:
        """Detect the dominant plane in a point cloud.

        Args:
            points: (N, 3) array of XYZ points

        Returns:
            PlaneModel if found, None otherwise
        """
        n_points = len(points)
        if n_points < 10:
            return None

        best_normal = None
        best_point = None
        best_inliers = 0

        for _ in range(self.config.iterations):
            # Sample 3 random points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]

            # Compute plane normal via cross product
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)

            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-6:
                continue  # Degenerate triangle

            normal = normal / norm_len

            # Optional: reject non-horizontal planes
            if self.config.require_horizontal:
                if abs(normal[2]) < self.config.horizontal_threshold:
                    continue

            # Orient normal toward camera (negative Z direction)
            if normal[2] > 0:
                normal = -normal

            # Count inliers
            distances = np.abs(np.dot(points - p1, normal))
            inliers = np.sum(distances < self.config.distance_threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal.copy()
                best_point = p1.copy()

        if best_normal is None:
            return None

        inlier_ratio = best_inliers / n_points

        if inlier_ratio < self.config.min_inlier_ratio:
            return None

        return PlaneModel(
            normal=best_normal,
            point=best_point,
            inlier_count=best_inliers,
            inlier_ratio=inlier_ratio,
        )

    def detect_multiple_planes(
        self,
        points: np.ndarray,
        max_planes: int = 3,
    ) -> List[PlaneModel]:
        """Detect multiple planes by iteratively removing inliers.

        Args:
            points: (N, 3) array of XYZ points
            max_planes: Maximum number of planes to detect

        Returns:
            List of PlaneModel instances
        """
        planes = []
        remaining_points = points.copy()
        remaining_indices = np.arange(len(points))

        for _ in range(max_planes):
            if len(remaining_points) < 100:
                break

            plane = self.detect_plane(remaining_points)
            if plane is None:
                break

            planes.append(plane)

            # Remove inliers
            mask = ~plane.get_inlier_mask(
                remaining_points,
                self.config.distance_threshold
            )
            remaining_points = remaining_points[mask]
            remaining_indices = remaining_indices[mask]

        return planes

    def remove_plane(
        self,
        points: np.ndarray,
        plane: PlaneModel,
        keep_above: bool = True,
        min_height: float = 0.02,
        max_height: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove points on/below a plane, keep points above.

        Args:
            points: (N, 3) array of XYZ points
            plane: Detected plane to remove
            keep_above: If True, keep points above plane
            min_height: Minimum distance from plane (meters)
            max_height: Maximum distance from plane (meters)

        Returns:
            (filtered_points, mask) where mask indicates kept points
        """
        # Signed distance (positive = toward camera = above plane)
        distances = plane.distance_to_points(points)

        if keep_above:
            mask = (distances > min_height) & (distances < max_height)
        else:
            mask = (distances < -min_height) | (np.abs(distances) < self.config.distance_threshold)

        return points[mask], mask


class PlaneRemovalModule(PipelineModule):
    """Pipeline module for removing floor/table planes.

    Detects dominant horizontal plane and removes points on/below it,
    keeping only objects above the surface.
    """

    def __init__(
        self,
        config: Optional[RANSACConfig] = None,
        min_height: float = 0.02,
        max_height: float = 0.5,
    ):
        """Initialize module.

        Args:
            config: RANSAC configuration
            min_height: Minimum height above plane to keep
            max_height: Maximum height above plane to keep
        """
        self.config = config or RANSACConfig(require_horizontal=True)
        self.detector = RANSACPlaneDetector(self.config)
        self.min_height = min_height
        self.max_height = max_height

    @property
    def name(self) -> str:
        return "plane_removal"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="pointcloud",
            required_fields=["points"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="pointcloud",
            required_fields=["points", "plane"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Detect and remove dominant plane.

        Args:
            data: PipelineData with 'points' field

        Returns:
            PipelineData with filtered points and plane info
        """
        points = data.points
        colors = data.get("colors")

        # Detect plane
        plane = self.detector.detect_plane(points)

        result = data.copy()

        if plane is None:
            result.plane = None
            result.plane_removed = False
            return result

        # Remove plane
        filtered_points, mask = self.detector.remove_plane(
            points,
            plane,
            keep_above=True,
            min_height=self.min_height,
            max_height=self.max_height,
        )

        result.points = filtered_points
        result.plane = plane
        result.plane_removed = True
        result.plane_mask = mask

        if colors is not None:
            result.colors = colors[mask]

        return result


def detect_floor_plane(
    points: np.ndarray,
    distance_threshold: float = 0.01,
) -> Optional[PlaneModel]:
    """Convenience function to detect floor plane.

    Args:
        points: (N, 3) point cloud
        distance_threshold: RANSAC inlier threshold

    Returns:
        PlaneModel for floor, or None
    """
    config = RANSACConfig(
        distance_threshold=distance_threshold,
        require_horizontal=True,
    )
    detector = RANSACPlaneDetector(config)
    return detector.detect_plane(points)


def filter_above_plane(
    points: np.ndarray,
    plane: PlaneModel,
    min_height: float = 0.02,
    max_height: float = 0.5,
) -> np.ndarray:
    """Convenience function to filter points above a plane.

    Args:
        points: (N, 3) point cloud
        plane: Detected plane
        min_height: Minimum height above plane
        max_height: Maximum height above plane

    Returns:
        Filtered point cloud
    """
    detector = RANSACPlaneDetector()
    filtered, _ = detector.remove_plane(
        points, plane,
        keep_above=True,
        min_height=min_height,
        max_height=max_height,
    )
    return filtered

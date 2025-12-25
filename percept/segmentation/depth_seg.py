"""Depth-based segmentation for PERCEPT.

Implements depth discontinuity edge detection and connected component
segmentation - fast CPU methods that complement AI-based segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.core.schema import ObjectMask


@dataclass
class DepthSegmentationConfig:
    """Configuration for depth-based segmentation."""

    # Edge detection
    gradient_threshold: float = 0.1  # Depth gradient threshold (meters/pixel)
    edge_kernel_size: int = 3  # Sobel kernel size

    # Connected components
    min_area: int = 500  # Minimum region area in pixels
    max_area: int = 500000  # Maximum region area
    depth_similarity_threshold: float = 0.1  # Max depth variance within region

    # Filtering
    min_depth: float = 0.2  # Minimum valid depth (meters)
    max_depth: float = 5.0  # Maximum valid depth (meters)


class DepthEdgeDetector:
    """Detect depth discontinuity edges.

    Uses Sobel gradients on depth image to find object boundaries
    where there are sharp depth changes.
    """

    def __init__(self, config: Optional[DepthSegmentationConfig] = None):
        """Initialize detector.

        Args:
            config: Configuration options
        """
        self.config = config or DepthSegmentationConfig()

    def detect_edges(self, depth: np.ndarray) -> np.ndarray:
        """Detect depth discontinuity edges.

        Args:
            depth: Depth image in meters (H, W), float32

        Returns:
            Binary edge mask (H, W), uint8 with 255 for edges
        """
        # Handle invalid depth
        valid_mask = (depth > self.config.min_depth) & (depth < self.config.max_depth)
        depth_filtered = np.where(valid_mask, depth, 0)

        # Compute gradients using Sobel
        ksize = self.config.edge_kernel_size
        grad_x = cv2.Sobel(depth_filtered, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(depth_filtered, cv2.CV_32F, 0, 1, ksize=ksize)

        # Gradient magnitude
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold to get edges
        edges = (gradient_mag > self.config.gradient_threshold).astype(np.uint8) * 255

        # Mask out invalid regions
        edges = np.where(valid_mask, edges, 0).astype(np.uint8)

        return edges

    def detect_edges_adaptive(
        self,
        depth: np.ndarray,
        relative_threshold: float = 0.1
    ) -> np.ndarray:
        """Detect edges with depth-relative threshold.

        Uses a threshold that scales with depth - objects further away
        need smaller absolute depth changes to be detected.

        Args:
            depth: Depth image in meters
            relative_threshold: Threshold as fraction of depth

        Returns:
            Binary edge mask
        """
        valid_mask = (depth > self.config.min_depth) & (depth < self.config.max_depth)
        depth_filtered = np.where(valid_mask, depth, 0)

        # Compute gradients
        ksize = self.config.edge_kernel_size
        grad_x = cv2.Sobel(depth_filtered, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(depth_filtered, cv2.CV_32F, 0, 1, ksize=ksize)

        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Adaptive threshold based on depth
        threshold = np.where(
            valid_mask,
            depth_filtered * relative_threshold,
            float('inf')
        )

        edges = (gradient_mag > threshold).astype(np.uint8) * 255

        return edges


class DepthConnectedComponents:
    """Segment depth image into connected regions.

    Groups pixels with similar depth values that are spatially connected.
    """

    def __init__(self, config: Optional[DepthSegmentationConfig] = None):
        """Initialize segmenter.

        Args:
            config: Configuration options
        """
        self.config = config or DepthSegmentationConfig()

    def segment(
        self,
        depth: np.ndarray,
        edges: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """Segment depth into connected components.

        Args:
            depth: Depth image in meters (H, W)
            edges: Optional edge mask to use as boundaries

        Returns:
            (labels, num_labels) where labels is (H, W) with region IDs
        """
        # Create valid depth mask
        valid_mask = (depth > self.config.min_depth) & (depth < self.config.max_depth)

        # If edges provided, use them as boundaries
        if edges is not None:
            valid_mask = valid_mask & (edges == 0)

        # Convert to uint8 for connected components
        binary = (valid_mask * 255).astype(np.uint8)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

        return labels, num_labels

    def segment_by_depth_similarity(
        self,
        depth: np.ndarray,
        initial_labels: np.ndarray,
    ) -> np.ndarray:
        """Refine segmentation by splitting regions with varying depth.

        Args:
            depth: Depth image in meters
            initial_labels: Initial label map

        Returns:
            Refined label map
        """
        refined = initial_labels.copy()
        max_label = initial_labels.max()

        for label_id in range(1, initial_labels.max() + 1):
            mask = initial_labels == label_id
            if mask.sum() == 0:
                continue

            region_depth = depth[mask]
            valid = region_depth > 0

            if valid.sum() == 0:
                continue

            depth_std = region_depth[valid].std()

            # If depth varies too much, try to split
            if depth_std > self.config.depth_similarity_threshold:
                # Use k-means on depth values to split
                region_depth_valid = region_depth[valid].reshape(-1, 1)

                if len(region_depth_valid) < 10:
                    continue

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5)
                _, sub_labels, _ = cv2.kmeans(
                    region_depth_valid.astype(np.float32),
                    2,  # Split into 2 clusters
                    None,
                    criteria,
                    3,
                    cv2.KMEANS_PP_CENTERS
                )

                # Apply sub-labels
                full_mask_indices = np.where(mask.flatten())[0]
                valid_indices = full_mask_indices[valid]

                for sub_idx, new_label in enumerate(sub_labels.flatten()):
                    if new_label == 1:  # Keep label 0 as original
                        max_label += 1
                        flat_refined = refined.flatten()
                        flat_refined[valid_indices[sub_idx]] = max_label
                        refined = flat_refined.reshape(refined.shape)

        return refined

    def extract_masks(
        self,
        labels: np.ndarray,
        depth: np.ndarray,
    ) -> List[ObjectMask]:
        """Extract ObjectMask instances from label map.

        Args:
            labels: Label map (H, W)
            depth: Depth image for metadata

        Returns:
            List of ObjectMask instances
        """
        masks = []

        for label_id in range(1, labels.max() + 1):
            mask = (labels == label_id).astype(np.uint8)
            area = mask.sum()

            # Filter by area
            if area < self.config.min_area or area > self.config.max_area:
                continue

            # Compute bounding box
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

            # Compute median depth
            region_depth = depth[mask > 0]
            valid_depth = region_depth[region_depth > 0]
            depth_median = float(np.median(valid_depth)) if len(valid_depth) > 0 else 0.0

            masks.append(ObjectMask(
                mask=mask * 255,
                bbox=bbox,
                confidence=0.8,  # Lower confidence than AI methods
                depth_median=depth_median,
                point_count=area,
            ))

        return masks


class DepthSegmenter(PipelineModule):
    """Pipeline module for depth-based segmentation.

    Combines edge detection and connected components to segment
    objects from depth images without using AI models.
    """

    def __init__(self, config: Optional[DepthSegmentationConfig] = None):
        """Initialize segmenter.

        Args:
            config: Configuration options
        """
        self.config = config or DepthSegmentationConfig()
        self.edge_detector = DepthEdgeDetector(self.config)
        self.component_segmenter = DepthConnectedComponents(self.config)

    @property
    def name(self) -> str:
        return "depth_segmenter"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="depth",
            dtype="float32",
            required_fields=["depth"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="masks",
            required_fields=["masks", "edges"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Process depth image to extract object masks.

        Args:
            data: PipelineData with 'depth' field

        Returns:
            PipelineData with 'masks' and 'edges' fields
        """
        depth = data.depth

        # Detect edges
        edges = self.edge_detector.detect_edges(depth)

        # Segment into regions
        labels, _ = self.component_segmenter.segment(depth, edges)

        # Optionally refine by depth similarity
        labels = self.component_segmenter.segment_by_depth_similarity(depth, labels)

        # Extract masks
        masks = self.component_segmenter.extract_masks(labels, depth)

        # Sort by area (largest first)
        masks.sort(key=lambda m: m.area, reverse=True)

        result = data.copy()
        result.masks = masks
        result.edges = edges
        result.labels = labels

        return result


def detect_depth_discontinuities(
    depth: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Convenience function to detect depth edges.

    Args:
        depth: Depth image in meters
        threshold: Gradient threshold

    Returns:
        Binary edge mask
    """
    config = DepthSegmentationConfig(gradient_threshold=threshold)
    detector = DepthEdgeDetector(config)
    return detector.detect_edges(depth)


def depth_connected_components(
    depth: np.ndarray,
    threshold: float = 0.3,
    min_area: int = 500,
) -> List[ObjectMask]:
    """Convenience function for depth-based segmentation.

    Args:
        depth: Depth image in meters
        threshold: Depth similarity threshold
        min_area: Minimum region area

    Returns:
        List of ObjectMask instances
    """
    config = DepthSegmentationConfig(
        depth_similarity_threshold=threshold,
        min_area=min_area,
    )
    segmenter = DepthConnectedComponents(config)
    edge_detector = DepthEdgeDetector(config)

    edges = edge_detector.detect_edges(depth)
    labels, _ = segmenter.segment(depth, edges)
    masks = segmenter.extract_masks(labels, depth)

    return masks

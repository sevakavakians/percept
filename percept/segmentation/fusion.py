"""Segmentation fusion for PERCEPT.

Combines masks from multiple segmentation methods (FastSAM, depth edges,
point cloud clustering) to produce robust final masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.core.schema import ObjectMask


@dataclass
class FusionConfig:
    """Configuration for segmentation fusion."""

    # IoU thresholds
    merge_iou_threshold: float = 0.5  # Merge masks with IoU above this
    agreement_iou_threshold: float = 0.3  # Consider masks agreeing above this

    # Confidence adjustment
    agreement_boost: float = 0.1  # Boost confidence for agreeing masks
    disagreement_penalty: float = 0.05  # Penalize masks with no agreement

    # Size filtering
    min_area: int = 500  # Minimum mask area
    max_area: int = 500000  # Maximum mask area

    # Depth consistency
    depth_std_threshold: float = 0.1  # Max depth std within mask
    use_depth_refinement: bool = True

    # Method weights (for confidence combination)
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        'fastsam': 1.0,
        'depth': 0.7,
        'pointcloud': 0.8,
    })


class MaskMerger:
    """Merge overlapping masks from different sources."""

    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialize merger.

        Args:
            config: Configuration options
        """
        self.config = config or FusionConfig()

    def compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union between two masks.

        Args:
            mask1: Binary mask (H, W)
            mask2: Binary mask (H, W)

        Returns:
            IoU score in [0, 1]
        """
        m1 = mask1 > 0
        m2 = mask2 > 0

        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def compute_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute overlap ratio (intersection / smaller mask area).

        Args:
            mask1: Binary mask (H, W)
            mask2: Binary mask (H, W)

        Returns:
            Overlap ratio in [0, 1]
        """
        m1 = mask1 > 0
        m2 = mask2 > 0

        intersection = np.logical_and(m1, m2).sum()
        min_area = min(m1.sum(), m2.sum())

        if min_area == 0:
            return 0.0

        return intersection / min_area

    def merge_masks(
        self,
        masks: List[ObjectMask],
        mode: str = 'union',
    ) -> ObjectMask:
        """Merge multiple masks into one.

        Args:
            masks: List of ObjectMask instances to merge
            mode: 'union', 'intersection', or 'weighted'

        Returns:
            Merged ObjectMask
        """
        if len(masks) == 0:
            raise ValueError("Cannot merge empty list of masks")

        if len(masks) == 1:
            return masks[0]

        # Get reference shape from first mask
        h, w = masks[0].mask.shape[:2]

        if mode == 'union':
            merged = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                merged = np.maximum(merged, m.mask)

        elif mode == 'intersection':
            merged = np.ones((h, w), dtype=np.uint8) * 255
            for m in masks:
                merged = np.minimum(merged, m.mask)

        elif mode == 'weighted':
            # Weight by confidence
            weights = np.array([m.confidence for m in masks])
            weights = weights / weights.sum()

            merged_float = np.zeros((h, w), dtype=np.float32)
            for m, weight in zip(masks, weights):
                merged_float += (m.mask > 0).astype(np.float32) * weight

            merged = (merged_float > 0.5).astype(np.uint8) * 255

        else:
            raise ValueError(f"Unknown merge mode: {mode}")

        # Compute combined properties
        ys, xs = np.where(merged > 0)
        if len(xs) == 0:
            # Return first mask if merge produces empty result
            return masks[0]

        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        area = (merged > 0).sum()

        # Combined confidence (max for union, min for intersection)
        if mode == 'union':
            confidence = max(m.confidence for m in masks)
        elif mode == 'intersection':
            confidence = min(m.confidence for m in masks)
        else:
            confidence = sum(m.confidence for m in masks) / len(masks)

        # Combined depth (weighted average)
        valid_depths = [m.depth_median for m in masks if m.depth_median > 0]
        depth_median = np.mean(valid_depths) if valid_depths else 0.0

        return ObjectMask(
            mask=merged,
            bbox=bbox,
            confidence=confidence,
            depth_median=float(depth_median),
            point_count=area,
        )

    def find_matching_masks(
        self,
        query_mask: ObjectMask,
        candidate_masks: List[ObjectMask],
        iou_threshold: float,
    ) -> List[Tuple[int, float]]:
        """Find masks matching query above IoU threshold.

        Args:
            query_mask: Query ObjectMask
            candidate_masks: List of candidate ObjectMasks
            iou_threshold: Minimum IoU for match

        Returns:
            List of (index, iou) tuples for matches
        """
        matches = []

        for i, candidate in enumerate(candidate_masks):
            iou = self.compute_iou(query_mask.mask, candidate.mask)
            if iou >= iou_threshold:
                matches.append((i, iou))

        # Sort by IoU descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches


class DepthRefiner:
    """Refine masks using depth information."""

    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialize refiner.

        Args:
            config: Configuration options
        """
        self.config = config or FusionConfig()

    def refine_mask(
        self,
        mask: ObjectMask,
        depth: np.ndarray,
    ) -> ObjectMask:
        """Refine mask by removing depth-inconsistent regions.

        Args:
            mask: Input ObjectMask
            depth: Depth image (H, W) in meters

        Returns:
            Refined ObjectMask
        """
        mask_binary = mask.mask > 0
        region_depth = depth[mask_binary]

        # Filter invalid depth
        valid = region_depth > 0
        if valid.sum() < 10:
            return mask

        valid_depth = region_depth[valid]
        median_depth = np.median(valid_depth)
        std_depth = np.std(valid_depth)

        # If depth is consistent, no refinement needed
        if std_depth < self.config.depth_std_threshold:
            return mask

        # Create refined mask keeping only depth-consistent pixels
        depth_diff = np.abs(depth - median_depth)
        depth_consistent = depth_diff < (2 * self.config.depth_std_threshold)

        refined_mask = (mask_binary & depth_consistent).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

        # Update bounding box and area
        ys, xs = np.where(refined_mask > 0)
        if len(xs) < 100:
            # Refinement too aggressive, keep original
            return mask

        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        area = (refined_mask > 0).sum()

        # Recalculate median depth on refined region
        new_region_depth = depth[refined_mask > 0]
        new_valid = new_region_depth > 0
        new_median = float(np.median(new_region_depth[new_valid])) if new_valid.sum() > 0 else median_depth

        return ObjectMask(
            mask=refined_mask,
            bbox=bbox,
            confidence=mask.confidence,
            depth_median=new_median,
            point_count=area,
        )

    def split_by_depth(
        self,
        mask: ObjectMask,
        depth: np.ndarray,
        max_objects: int = 3,
    ) -> List[ObjectMask]:
        """Split mask into multiple objects based on depth discontinuities.

        Args:
            mask: Input ObjectMask
            depth: Depth image (H, W)
            max_objects: Maximum objects to split into

        Returns:
            List of ObjectMasks (may be single element if no split)
        """
        mask_binary = mask.mask > 0
        region_depth = depth[mask_binary]

        valid = region_depth > 0
        if valid.sum() < 100:
            return [mask]

        valid_depth = region_depth[valid]
        std_depth = np.std(valid_depth)

        # If depth is consistent, no split needed
        if std_depth < 2 * self.config.depth_std_threshold:
            return [mask]

        # Get coordinates of valid pixels
        ys, xs = np.where(mask_binary)
        depths = depth[mask_binary]

        # Combine position and depth for clustering
        valid_mask = depths > 0
        valid_ys = ys[valid_mask]
        valid_xs = xs[valid_mask]
        valid_depths = depths[valid_mask]

        if len(valid_depths) < 100:
            return [mask]

        # Use k-means on depth values
        depth_data = valid_depths.reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5)
        n_clusters = min(max_objects, max(2, int(std_depth / self.config.depth_std_threshold)))

        try:
            _, labels, centers = cv2.kmeans(
                depth_data, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
        except cv2.error:
            return [mask]

        labels = labels.flatten()

        # Create separate masks for each cluster
        result_masks = []
        h, w = mask.mask.shape[:2]

        for cluster_id in range(n_clusters):
            cluster_mask = np.zeros((h, w), dtype=np.uint8)
            cluster_indices = labels == cluster_id

            cluster_ys = valid_ys[cluster_indices]
            cluster_xs = valid_xs[cluster_indices]

            if len(cluster_ys) < 100:
                continue

            cluster_mask[cluster_ys, cluster_xs] = 255

            # Morphological cleanup
            kernel = np.ones((5, 5), np.uint8)
            cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel)

            # Get bounding box
            cys, cxs = np.where(cluster_mask > 0)
            if len(cxs) == 0:
                continue

            bbox = (int(cxs.min()), int(cys.min()), int(cxs.max()), int(cys.max()))
            area = (cluster_mask > 0).sum()

            # Get median depth for this cluster
            cluster_depths = valid_depths[cluster_indices]
            depth_median = float(np.median(cluster_depths))

            result_masks.append(ObjectMask(
                mask=cluster_mask,
                bbox=bbox,
                confidence=mask.confidence * 0.9,  # Slight penalty for split
                depth_median=depth_median,
                point_count=area,
            ))

        if len(result_masks) == 0:
            return [mask]

        return result_masks


class SegmentationFusion(PipelineModule):
    """Pipeline module for fusing multiple segmentation methods.

    Combines masks from FastSAM, depth segmentation, and point cloud
    clustering to produce robust final segmentation.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialize fusion module.

        Args:
            config: Configuration options
        """
        self.config = config or FusionConfig()
        self.merger = MaskMerger(self.config)
        self.refiner = DepthRefiner(self.config)

    @property
    def name(self) -> str:
        return "segmentation_fusion"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="multi_masks",
            required_fields=["masks_fastsam", "masks_depth", "masks_pointcloud"],
            optional_fields=["depth"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="masks",
            required_fields=["masks"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Fuse masks from multiple segmentation methods.

        Args:
            data: PipelineData with mask lists from different methods

        Returns:
            PipelineData with fused 'masks' field
        """
        # Get masks from different methods (may be empty lists)
        fastsam_masks = data.get("masks_fastsam", [])
        depth_masks = data.get("masks_depth", [])
        pc_masks = data.get("masks_pointcloud", [])
        depth = data.get("depth")

        # Tag masks with their source
        all_masks = []

        for m in fastsam_masks:
            all_masks.append((m, 'fastsam'))
        for m in depth_masks:
            all_masks.append((m, 'depth'))
        for m in pc_masks:
            all_masks.append((m, 'pointcloud'))

        if len(all_masks) == 0:
            result = data.copy()
            result.masks = []
            return result

        # Fuse masks
        fused_masks = self._fuse_masks(all_masks, depth)

        # Filter by size
        fused_masks = [
            m for m in fused_masks
            if self.config.min_area <= m.area <= self.config.max_area
        ]

        # Sort by area (largest first)
        fused_masks.sort(key=lambda m: m.area, reverse=True)

        result = data.copy()
        result.masks = fused_masks

        return result

    def _fuse_masks(
        self,
        tagged_masks: List[Tuple[ObjectMask, str]],
        depth: Optional[np.ndarray],
    ) -> List[ObjectMask]:
        """Fuse masks from multiple sources.

        Args:
            tagged_masks: List of (mask, source_name) tuples
            depth: Optional depth image for refinement

        Returns:
            List of fused ObjectMasks
        """
        if len(tagged_masks) == 0:
            return []

        # Group masks by approximate location (using IoU matching)
        groups = []  # List of lists of (mask, source)
        used = [False] * len(tagged_masks)

        for i, (mask_i, source_i) in enumerate(tagged_masks):
            if used[i]:
                continue

            group = [(mask_i, source_i)]
            used[i] = True

            # Find all masks that overlap with this one
            for j, (mask_j, source_j) in enumerate(tagged_masks):
                if used[j]:
                    continue

                iou = self.merger.compute_iou(mask_i.mask, mask_j.mask)
                if iou >= self.config.agreement_iou_threshold:
                    group.append((mask_j, source_j))
                    used[j] = True

            groups.append(group)

        # Process each group
        fused = []

        for group in groups:
            if len(group) == 1:
                # Single mask - apply disagreement penalty
                mask, source = group[0]
                new_conf = max(0.1, mask.confidence - self.config.disagreement_penalty)
                fused_mask = ObjectMask(
                    mask=mask.mask,
                    bbox=mask.bbox,
                    confidence=new_conf,
                    depth_median=mask.depth_median,
                    point_count=mask.point_count,
                )
            else:
                # Multiple masks agree - merge and boost confidence
                masks = [m for m, _ in group]
                sources = [s for _, s in group]

                merged = self.merger.merge_masks(masks, mode='weighted')

                # Boost confidence based on agreement
                agreement_factor = len(set(sources)) / 3.0  # Max 3 methods
                new_conf = min(1.0, merged.confidence + self.config.agreement_boost * agreement_factor)

                fused_mask = ObjectMask(
                    mask=merged.mask,
                    bbox=merged.bbox,
                    confidence=new_conf,
                    depth_median=merged.depth_median,
                    point_count=merged.point_count,
                )

            # Optional depth refinement
            if depth is not None and self.config.use_depth_refinement:
                fused_mask = self.refiner.refine_mask(fused_mask, depth)

            fused.append(fused_mask)

        return fused


def fuse_segmentation_results(
    fastsam_masks: List[ObjectMask],
    depth_masks: List[ObjectMask],
    pointcloud_masks: List[ObjectMask],
    depth: Optional[np.ndarray] = None,
) -> List[ObjectMask]:
    """Convenience function for segmentation fusion.

    Args:
        fastsam_masks: Masks from FastSAM
        depth_masks: Masks from depth segmentation
        pointcloud_masks: Masks from point cloud clustering
        depth: Optional depth image for refinement

    Returns:
        List of fused ObjectMasks
    """
    config = FusionConfig()
    fusion = SegmentationFusion(config)

    data = PipelineData()
    data.masks_fastsam = fastsam_masks
    data.masks_depth = depth_masks
    data.masks_pointcloud = pointcloud_masks
    if depth is not None:
        data.depth = depth

    result = fusion.process(data)
    return result.masks

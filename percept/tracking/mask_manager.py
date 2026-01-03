"""Scene mask management for PERCEPT.

Manages object masks within a scene to:
- Prevent duplicate processing of claimed regions
- Handle overlapping detections
- Maintain spatial consistency across frames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import time

import cv2
import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.core.schema import ObjectMask


@dataclass
class MaskManagerConfig:
    """Configuration for mask management."""

    # Overlap thresholds
    overlap_threshold: float = 0.3  # Min overlap to consider masks conflicting
    claim_threshold: float = 0.5  # Min overlap to claim region

    # Mask processing
    min_mask_area: int = 100  # Minimum pixels for valid mask
    max_masks_per_frame: int = 50  # Maximum masks to process

    # Temporal settings
    claim_duration: float = 1.0  # Seconds to maintain claim
    stale_threshold: float = 0.5  # Seconds before claim becomes stale


@dataclass
class MaskClaim:
    """A claimed mask region."""
    mask_id: str
    object_id: str
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: int
    timestamp: float
    priority: int = 0  # Higher = stronger claim

    def is_stale(self, threshold: float) -> bool:
        """Check if claim is stale."""
        return (time.time() - self.timestamp) > threshold


class SceneMaskManager:
    """Manages mask claims and mutual exclusion in a scene.

    Prevents duplicate processing by tracking which regions have been
    claimed by tracked objects. New detections in claimed regions are
    either merged or filtered.
    """

    def __init__(self, config: Optional[MaskManagerConfig] = None):
        """Initialize manager.

        Args:
            config: Configuration options
        """
        self.config = config or MaskManagerConfig()
        self._claims: Dict[str, MaskClaim] = {}  # mask_id -> claim
        self._object_claims: Dict[str, Set[str]] = {}  # object_id -> set of mask_ids
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._composite_mask: Optional[np.ndarray] = None

    def claim(
        self,
        object_id: str,
        mask: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        priority: int = 0,
    ) -> str:
        """Claim a mask region for an object.

        Args:
            object_id: Object claiming the region
            mask: Binary mask (H, W)
            bbox: Optional bounding box (auto-computed if None)
            priority: Claim priority (higher wins conflicts)

        Returns:
            Unique mask_id for this claim
        """
        # Update frame shape
        if self._frame_shape is None or self._frame_shape != mask.shape:
            self._frame_shape = mask.shape
            self._composite_mask = np.zeros(mask.shape, dtype=np.uint8)

        # Generate mask ID
        import uuid
        mask_id = str(uuid.uuid4())[:8]

        # Compute bbox if not provided
        if bbox is None:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            else:
                bbox = (0, 0, 0, 0)

        area = int((mask > 0).sum())

        claim = MaskClaim(
            mask_id=mask_id,
            object_id=object_id,
            mask=mask.copy(),
            bbox=bbox,
            area=area,
            timestamp=time.time(),
            priority=priority,
        )

        self._claims[mask_id] = claim

        # Track claims per object
        if object_id not in self._object_claims:
            self._object_claims[object_id] = set()
        self._object_claims[object_id].add(mask_id)

        # Update composite mask
        self._update_composite()

        return mask_id

    def release(self, mask_id: str) -> bool:
        """Release a mask claim.

        Args:
            mask_id: Mask to release

        Returns:
            True if released
        """
        if mask_id not in self._claims:
            return False

        claim = self._claims[mask_id]
        object_id = claim.object_id

        del self._claims[mask_id]

        if object_id in self._object_claims:
            self._object_claims[object_id].discard(mask_id)
            if not self._object_claims[object_id]:
                del self._object_claims[object_id]

        self._update_composite()
        return True

    def release_object(self, object_id: str) -> int:
        """Release all claims for an object.

        Args:
            object_id: Object whose claims to release

        Returns:
            Number of claims released
        """
        if object_id not in self._object_claims:
            return 0

        mask_ids = list(self._object_claims[object_id])
        for mask_id in mask_ids:
            if mask_id in self._claims:
                del self._claims[mask_id]

        del self._object_claims[object_id]
        self._update_composite()

        return len(mask_ids)

    def update_claim(
        self,
        mask_id: str,
        mask: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """Update an existing claim with new mask.

        Args:
            mask_id: Claim to update
            mask: New mask
            bbox: New bounding box

        Returns:
            True if updated
        """
        if mask_id not in self._claims:
            return False

        claim = self._claims[mask_id]

        if bbox is None:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            else:
                bbox = (0, 0, 0, 0)

        claim.mask = mask.copy()
        claim.bbox = bbox
        claim.area = int((mask > 0).sum())
        claim.timestamp = time.time()

        self._update_composite()
        return True

    def check_overlap(
        self,
        mask: np.ndarray,
        exclude_object: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """Check if mask overlaps with existing claims.

        Args:
            mask: Mask to check
            exclude_object: Object to exclude from check

        Returns:
            List of (mask_id, object_id, overlap_ratio) for overlapping claims
        """
        overlaps = []

        mask_bool = mask > 0
        mask_area = mask_bool.sum()
        if mask_area == 0:
            return overlaps

        for mask_id, claim in self._claims.items():
            if exclude_object and claim.object_id == exclude_object:
                continue

            if claim.is_stale(self.config.stale_threshold):
                continue

            claim_bool = claim.mask > 0
            intersection = np.logical_and(mask_bool, claim_bool).sum()

            if intersection > 0:
                # Overlap ratio relative to smaller mask
                min_area = min(mask_area, claim.area)
                overlap_ratio = intersection / min_area

                if overlap_ratio >= self.config.overlap_threshold:
                    overlaps.append((mask_id, claim.object_id, float(overlap_ratio)))

        # Sort by overlap ratio
        overlaps.sort(key=lambda x: x[2], reverse=True)
        return overlaps

    def is_region_claimed(
        self,
        mask: np.ndarray,
        exclude_object: Optional[str] = None,
    ) -> bool:
        """Check if region is significantly claimed.

        Args:
            mask: Mask to check
            exclude_object: Object to exclude

        Returns:
            True if region is mostly claimed by other objects
        """
        overlaps = self.check_overlap(mask, exclude_object)
        if not overlaps:
            return False

        # Check if any overlap exceeds claim threshold
        return any(ratio >= self.config.claim_threshold for _, _, ratio in overlaps)

    def filter_unclaimed(
        self,
        masks: List[ObjectMask],
        exclude_object: Optional[str] = None,
    ) -> List[ObjectMask]:
        """Filter masks to only unclaimed regions.

        Args:
            masks: List of ObjectMasks to filter
            exclude_object: Object to exclude from claims

        Returns:
            Filtered list of unclaimed masks
        """
        unclaimed = []

        for mask_obj in masks:
            if not self.is_region_claimed(mask_obj.mask, exclude_object):
                unclaimed.append(mask_obj)

        return unclaimed

    def subtract_claimed(
        self,
        mask: np.ndarray,
        exclude_object: Optional[str] = None,
    ) -> np.ndarray:
        """Subtract claimed regions from mask.

        Args:
            mask: Input mask
            exclude_object: Object to exclude

        Returns:
            Mask with claimed regions removed
        """
        result = mask.copy()

        for claim in self._claims.values():
            if exclude_object and claim.object_id == exclude_object:
                continue

            if claim.is_stale(self.config.stale_threshold):
                continue

            # Subtract claim from mask
            result = np.where(claim.mask > 0, 0, result)

        return result

    def get_composite_mask(self) -> Optional[np.ndarray]:
        """Get composite of all current claims.

        Returns:
            Combined mask or None if no claims
        """
        return self._composite_mask

    def get_object_masks(self, object_id: str) -> List[np.ndarray]:
        """Get all masks claimed by an object.

        Args:
            object_id: Object to query

        Returns:
            List of claimed masks
        """
        if object_id not in self._object_claims:
            return []

        masks = []
        for mask_id in self._object_claims[object_id]:
            if mask_id in self._claims:
                masks.append(self._claims[mask_id].mask)

        return masks

    def cleanup_stale(self) -> int:
        """Remove stale claims.

        Returns:
            Number of claims removed
        """
        stale_ids = [
            mask_id for mask_id, claim in self._claims.items()
            if claim.is_stale(self.config.claim_duration)
        ]

        for mask_id in stale_ids:
            self.release(mask_id)

        return len(stale_ids)

    def clear(self):
        """Clear all claims."""
        self._claims.clear()
        self._object_claims.clear()
        if self._frame_shape:
            self._composite_mask = np.zeros(self._frame_shape, dtype=np.uint8)

    def _update_composite(self):
        """Update composite mask from all claims."""
        if self._frame_shape is None:
            return

        self._composite_mask = np.zeros(self._frame_shape, dtype=np.uint8)

        for claim in self._claims.values():
            if not claim.is_stale(self.config.stale_threshold):
                self._composite_mask = np.maximum(self._composite_mask, claim.mask)

    @property
    def num_claims(self) -> int:
        """Get number of active claims."""
        return len(self._claims)

    @property
    def claimed_area(self) -> int:
        """Get total claimed area in pixels."""
        if self._composite_mask is None:
            return 0
        return int((self._composite_mask > 0).sum())


class MaskConflictResolver:
    """Resolve conflicts between overlapping masks."""

    def __init__(self, config: Optional[MaskManagerConfig] = None):
        """Initialize resolver.

        Args:
            config: Configuration options
        """
        self.config = config or MaskManagerConfig()

    def resolve(
        self,
        masks: List[ObjectMask],
        priorities: Optional[List[int]] = None,
    ) -> List[ObjectMask]:
        """Resolve conflicts between overlapping masks.

        Higher priority masks take precedence. For equal priority,
        larger area wins.

        Args:
            masks: List of potentially overlapping masks
            priorities: Optional priority per mask (default: all 0)

        Returns:
            List of non-overlapping masks
        """
        if len(masks) <= 1:
            return masks

        if priorities is None:
            priorities = [0] * len(masks)

        # Sort by priority (desc), then area (desc)
        sorted_masks = sorted(
            zip(masks, priorities),
            key=lambda x: (x[1], x[0].area),
            reverse=True,
        )

        resolved = []
        claimed = None

        for mask_obj, priority in sorted_masks:
            if claimed is None:
                claimed = np.zeros(mask_obj.mask.shape, dtype=np.uint8)

            # Check overlap with already-resolved masks
            mask_bool = mask_obj.mask > 0
            overlap = np.logical_and(mask_bool, claimed > 0).sum()
            overlap_ratio = overlap / max(mask_bool.sum(), 1)

            if overlap_ratio < self.config.claim_threshold:
                # Accept this mask
                resolved.append(mask_obj)
                claimed = np.maximum(claimed, mask_obj.mask)
            else:
                # Subtract claimed regions
                new_mask = np.where(claimed > 0, 0, mask_obj.mask)
                new_area = (new_mask > 0).sum()

                if new_area >= self.config.min_mask_area:
                    # Update mask with subtracted region
                    ys, xs = np.where(new_mask > 0)
                    if len(xs) > 0:
                        new_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

                        modified_mask = ObjectMask(
                            mask=new_mask.astype(np.uint8) * 255,
                            bbox=new_bbox,
                            confidence=mask_obj.confidence * 0.9,  # Reduce confidence
                            depth_median=mask_obj.depth_median,
                            point_count=int(new_area),
                        )
                        resolved.append(modified_mask)
                        claimed = np.maximum(claimed, new_mask)

        return resolved


class MaskManagerModule(PipelineModule):
    """Pipeline module for mask management.

    Filters new detections against claimed regions and manages
    mask claims for tracked objects.
    """

    def __init__(self, config: Optional[MaskManagerConfig] = None):
        """Initialize module.

        Args:
            config: Configuration options
        """
        self.config = config or MaskManagerConfig()
        self.manager = SceneMaskManager(self.config)
        self.resolver = MaskConflictResolver(self.config)

    @property
    def name(self) -> str:
        return "mask_manager"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="masks",
            required_fields=["masks"],
            optional_fields=["tracks"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="filtered_masks",
            required_fields=["masks"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Process masks through manager.

        Args:
            data: PipelineData with masks

        Returns:
            PipelineData with filtered masks
        """
        masks = data.get("masks", [])
        tracks = data.get("tracks", [])

        # Cleanup stale claims
        self.manager.cleanup_stale()

        # Update claims from tracks
        for track in tracks:
            if hasattr(track, 'track_id') and hasattr(track, 'current_mask'):
                if track.current_mask is not None:
                    object_id = str(track.track_id)

                    # Check if object already has a claim
                    if object_id in self.manager._object_claims:
                        # Update existing claim
                        mask_ids = list(self.manager._object_claims[object_id])
                        if mask_ids:
                            self.manager.update_claim(mask_ids[0], track.current_mask)
                    else:
                        # Create new claim
                        self.manager.claim(object_id, track.current_mask)

        # Filter new masks
        filtered_masks = self.manager.filter_unclaimed(masks)

        # Resolve any remaining conflicts
        resolved_masks = self.resolver.resolve(filtered_masks)

        result = data.copy()
        result.masks = resolved_masks
        result.claimed_mask = self.manager.get_composite_mask()

        return result

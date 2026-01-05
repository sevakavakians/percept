"""Human review system for PERCEPT.

Provides confidence-based routing, review queue management,
and image crop handling for human verification of classifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import hashlib
import shutil

import numpy as np

from percept.core.schema import ObjectSchema, ClassificationStatus
from percept.persistence.database import PerceptDatabase


class ReviewStatus(Enum):
    """Status of a review item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEWED = "reviewed"
    SKIPPED = "skipped"
    EXPIRED = "expired"


class ReviewReason(Enum):
    """Reason for requiring review."""
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS_CLASS = "ambiguous_class"
    NEW_OBJECT_TYPE = "new_object_type"
    CONFLICTING_ATTRIBUTES = "conflicting_attributes"
    PERIODIC_VERIFICATION = "periodic_verification"
    USER_FLAGGED = "user_flagged"
    MODEL_UNCERTAINTY = "model_uncertainty"


class ReviewPriority(Enum):
    """Priority level for review items."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class ConfidenceConfig:
    """Configuration for confidence-based routing."""

    # Classification thresholds
    confirmed_threshold: float = 0.85  # Auto-confirm above this
    provisional_threshold: float = 0.5  # Provisional between this and confirmed
    review_threshold: float = 0.5  # Queue for review below this

    # ReID thresholds
    reid_match_threshold: float = 0.3  # Cosine distance for same object
    reid_new_object_threshold: float = 0.5  # Above this = definitely new

    # Reprocessing
    reprocess_interval_seconds: float = 60.0  # Re-check provisional objects

    # Review settings
    max_pending_reviews: int = 1000  # Maximum pending items in queue
    auto_skip_after_hours: float = 24.0  # Auto-skip old items


@dataclass
class ReviewItem:
    """An item in the human review queue."""
    review_id: int
    object_id: str
    schema: ObjectSchema
    image_path: str
    reason: ReviewReason
    priority: ReviewPriority
    provisional_class: str
    confidence: float
    created_at: datetime
    status: ReviewStatus = ReviewStatus.PENDING
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    # Review results (filled after review)
    reviewed_at: Optional[datetime] = None
    reviewer: Optional[str] = None
    human_class: Optional[str] = None
    human_attributes: Optional[Dict[str, Any]] = None
    correction_notes: Optional[str] = None


@dataclass
class ReviewResult:
    """Result of a human review."""
    object_id: str
    human_class: str
    human_attributes: Dict[str, Any]
    reviewer: str
    reviewed_at: datetime
    correction_notes: str = ""
    was_correct: bool = True  # Whether original classification was correct
    original_class: str = ""
    original_confidence: float = 0.0


class CropManager:
    """Manages image crop storage for review items."""

    def __init__(self, storage_path: str = "data/review_crops"):
        """Initialize crop manager.

        Args:
            storage_path: Directory for storing crops
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_crop(
        self,
        object_id: str,
        crop: np.ndarray,
        format: str = "jpg",
    ) -> str:
        """Save crop image.

        Args:
            object_id: Object identifier
            crop: Image crop array (BGR format)
            format: Image format (jpg, png)

        Returns:
            Path to saved image
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{object_id}_{timestamp}.{format}"
        filepath = self.storage_path / filename

        # Save image
        try:
            import cv2
            if format == "jpg":
                cv2.imwrite(str(filepath), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(filepath), crop)
        except ImportError:
            # Fallback: save as numpy array
            filepath = self.storage_path / f"{object_id}_{timestamp}.npy"
            np.save(str(filepath), crop)

        return str(filepath)

    def load_crop(self, image_path: str) -> Optional[np.ndarray]:
        """Load crop image.

        Args:
            image_path: Path to image

        Returns:
            Image array or None
        """
        path = Path(image_path)
        if not path.exists():
            return None

        if path.suffix == ".npy":
            return np.load(str(path))

        try:
            import cv2
            return cv2.imread(str(path))
        except ImportError:
            return None

    def delete_crop(self, image_path: str) -> bool:
        """Delete crop image.

        Args:
            image_path: Path to image

        Returns:
            True if deleted
        """
        path = Path(image_path)
        if path.exists():
            path.unlink()
            return True
        return False

    def cleanup_old_crops(self, max_age_hours: float = 48.0) -> int:
        """Remove old crop images.

        Args:
            max_age_hours: Maximum age before deletion

        Returns:
            Number of files deleted
        """
        import time

        cutoff = time.time() - (max_age_hours * 3600)
        deleted = 0

        for path in self.storage_path.iterdir():
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1

        return deleted

    def get_storage_size(self) -> int:
        """Get total storage size in bytes."""
        total = 0
        for path in self.storage_path.iterdir():
            total += path.stat().st_size
        return total


class ConfidenceRouter:
    """Routes objects based on classification confidence.

    Determines whether objects should be:
    - Auto-confirmed (high confidence)
    - Marked provisional (medium confidence)
    - Queued for review (low confidence)
    """

    def __init__(self, config: Optional[ConfidenceConfig] = None):
        """Initialize router.

        Args:
            config: Configuration options
        """
        self.config = config or ConfidenceConfig()

    def route(
        self,
        obj: ObjectSchema,
        alternatives: Optional[List[Tuple[str, float]]] = None,
    ) -> Tuple[ClassificationStatus, Optional[ReviewReason]]:
        """Determine classification status and review need.

        Args:
            obj: ObjectSchema to route
            alternatives: Alternative classifications [(class, confidence), ...]

        Returns:
            Tuple of (ClassificationStatus, ReviewReason or None)
        """
        confidence = obj.confidence

        # Check for ambiguous classification
        if alternatives and len(alternatives) >= 2:
            top_conf = alternatives[0][1]
            second_conf = alternatives[1][1]

            # If top two are close, it's ambiguous
            if top_conf - second_conf < 0.1:
                return ClassificationStatus.NEEDS_REVIEW, ReviewReason.AMBIGUOUS_CLASS

        # High confidence - auto-confirm
        if confidence >= self.config.confirmed_threshold:
            return ClassificationStatus.CONFIRMED, None

        # Low confidence - needs review
        if confidence < self.config.review_threshold:
            return ClassificationStatus.NEEDS_REVIEW, ReviewReason.LOW_CONFIDENCE

        # Medium confidence - provisional
        return ClassificationStatus.PROVISIONAL, None

    def should_reprocess(self, obj: ObjectSchema) -> bool:
        """Check if provisional object should be reprocessed.

        Args:
            obj: ObjectSchema to check

        Returns:
            True if should reprocess
        """
        if obj.classification_status != ClassificationStatus.PROVISIONAL:
            return False

        age = (datetime.now() - obj.last_seen).total_seconds()
        return age >= self.config.reprocess_interval_seconds

    def calculate_priority(
        self,
        obj: ObjectSchema,
        reason: ReviewReason,
    ) -> ReviewPriority:
        """Calculate review priority.

        Args:
            obj: Object to review
            reason: Reason for review

        Returns:
            Priority level
        """
        # User-flagged items are urgent
        if reason == ReviewReason.USER_FLAGGED:
            return ReviewPriority.URGENT

        # Very low confidence is high priority
        if obj.confidence < 0.2:
            return ReviewPriority.HIGH

        # New object types need attention
        if reason == ReviewReason.NEW_OBJECT_TYPE:
            return ReviewPriority.HIGH

        # Ambiguous classification is normal priority
        if reason == ReviewReason.AMBIGUOUS_CLASS:
            return ReviewPriority.NORMAL

        return ReviewPriority.LOW


class HumanReviewQueue:
    """Manages the human review queue.

    Stores uncertain objects for async human verification,
    tracks review results, and integrates with the database.
    """

    def __init__(
        self,
        db: PerceptDatabase,
        crop_manager: Optional[CropManager] = None,
        confidence_config: Optional[ConfidenceConfig] = None,
    ):
        """Initialize review queue.

        Args:
            db: Database connection
            crop_manager: Optional crop manager
            confidence_config: Optional confidence configuration
        """
        self.db = db
        self.crop_manager = crop_manager or CropManager()
        self.config = confidence_config or ConfidenceConfig()
        self.router = ConfidenceRouter(self.config)

        # Callbacks for review events
        self._on_review_complete: List[Callable[[ReviewResult], None]] = []

    def add_for_review(
        self,
        obj: ObjectSchema,
        crop: np.ndarray,
        reason: ReviewReason,
        alternatives: Optional[List[Tuple[str, float]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add object to review queue.

        Args:
            obj: ObjectSchema to review
            crop: Image crop
            reason: Reason for review
            alternatives: Alternative classifications
            context: Additional context

        Returns:
            Review queue item ID
        """
        # Save crop image
        image_path = self.crop_manager.save_crop(obj.id, crop)

        # Calculate priority
        priority = self.router.calculate_priority(obj, reason)

        # Update object status
        obj.classification_status = ClassificationStatus.NEEDS_REVIEW
        self.db.save_object(obj)

        # Add to queue with extended fields
        review_id = self.db.add_to_review_queue(obj, image_path, reason.value)

        return review_id

    def get_pending(
        self,
        limit: int = 50,
        priority_filter: Optional[ReviewPriority] = None,
        class_filter: Optional[str] = None,
    ) -> List[ReviewItem]:
        """Get pending review items.

        Args:
            limit: Maximum items to return
            priority_filter: Optional priority filter
            class_filter: Optional class filter

        Returns:
            List of ReviewItem instances
        """
        raw_items = self.db.get_pending_reviews(limit * 2)  # Get more for filtering

        items = []
        for raw in raw_items:
            try:
                schema = ObjectSchema.from_json(raw["schema_json"])

                # Parse reason
                try:
                    reason = ReviewReason(raw["reason"])
                except ValueError:
                    reason = ReviewReason.LOW_CONFIDENCE

                item = ReviewItem(
                    review_id=raw["id"],
                    object_id=raw["object_id"],
                    schema=schema,
                    image_path=raw["image_path"],
                    reason=reason,
                    priority=ReviewPriority.NORMAL,  # Could be stored in DB
                    provisional_class=raw["provisional_class"],
                    confidence=raw["confidence"],
                    created_at=datetime.fromisoformat(raw["created_at"]),
                )

                # Apply filters
                if class_filter and item.provisional_class != class_filter:
                    continue

                items.append(item)

                if len(items) >= limit:
                    break

            except (json.JSONDecodeError, KeyError):
                continue

        return items

    def get_review_item(self, review_id: int) -> Optional[ReviewItem]:
        """Get specific review item.

        Args:
            review_id: Review queue ID

        Returns:
            ReviewItem or None
        """
        items = self.get_pending(limit=1000)
        for item in items:
            if item.review_id == review_id:
                return item
        return None

    def submit_review(
        self,
        review_id: int,
        human_class: str,
        reviewer: str,
        human_attributes: Optional[Dict[str, Any]] = None,
        correction_notes: str = "",
    ) -> ReviewResult:
        """Submit a human review.

        Args:
            review_id: Review queue item ID
            human_class: Human-assigned class
            reviewer: Reviewer identifier
            human_attributes: Additional attributes
            correction_notes: Notes about the correction

        Returns:
            ReviewResult with details
        """
        # Get the review item first
        item = self.get_review_item(review_id)
        original_class = item.provisional_class if item else ""
        original_confidence = item.confidence if item else 0.0

        # Submit to database
        self.db.submit_review(review_id, human_class, reviewer, human_attributes)

        # Create result
        result = ReviewResult(
            object_id=item.object_id if item else "",
            human_class=human_class,
            human_attributes=human_attributes or {},
            reviewer=reviewer,
            reviewed_at=datetime.now(),
            correction_notes=correction_notes,
            was_correct=(human_class == original_class),
            original_class=original_class,
            original_confidence=original_confidence,
        )

        # Notify listeners
        for callback in self._on_review_complete:
            try:
                callback(result)
            except Exception:
                pass

        return result

    def skip_review(self, review_id: int, reason: str = "") -> None:
        """Skip a review item.

        Args:
            review_id: Review queue item ID
            reason: Optional skip reason
        """
        self.db.skip_review(review_id)

    def flag_for_review(
        self,
        obj: ObjectSchema,
        crop: np.ndarray,
        flagged_by: str = "user",
        notes: str = "",
    ) -> int:
        """Flag an object for review.

        Args:
            obj: Object to flag
            crop: Image crop
            flagged_by: Who flagged it
            notes: Optional notes

        Returns:
            Review queue item ID
        """
        return self.add_for_review(
            obj=obj,
            crop=crop,
            reason=ReviewReason.USER_FLAGGED,
            context={"flagged_by": flagged_by, "notes": notes},
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get review queue statistics.

        Returns:
            Dictionary of statistics
        """
        base_stats = self.db.get_review_stats()

        return {
            **base_stats,
            "storage_size_bytes": self.crop_manager.get_storage_size(),
        }

    def cleanup_old_reviews(self, max_age_hours: float = 48.0) -> int:
        """Clean up old review items and crops.

        Args:
            max_age_hours: Maximum age

        Returns:
            Number of items cleaned
        """
        return self.crop_manager.cleanup_old_crops(max_age_hours)

    def on_review_complete(self, callback: Callable[[ReviewResult], None]) -> None:
        """Register callback for review completion.

        Args:
            callback: Function to call with ReviewResult
        """
        self._on_review_complete.append(callback)

    def process_object(
        self,
        obj: ObjectSchema,
        crop: np.ndarray,
        alternatives: Optional[List[Tuple[str, float]]] = None,
    ) -> ClassificationStatus:
        """Process object through confidence routing.

        Automatically routes to confirm, provisional, or review.

        Args:
            obj: Object to process
            crop: Image crop
            alternatives: Alternative classifications

        Returns:
            Final classification status
        """
        status, reason = self.router.route(obj, alternatives)

        if status == ClassificationStatus.NEEDS_REVIEW and reason is not None:
            self.add_for_review(obj, crop, reason, alternatives)

        obj.classification_status = status
        self.db.save_object(obj)

        return status


class BatchReviewer:
    """Supports batch review operations for efficiency."""

    def __init__(self, queue: HumanReviewQueue):
        """Initialize batch reviewer.

        Args:
            queue: Review queue to use
        """
        self.queue = queue

    def get_batch(
        self,
        batch_size: int = 20,
        same_class: bool = True,
    ) -> List[ReviewItem]:
        """Get a batch of similar items for efficient review.

        Args:
            batch_size: Number of items in batch
            same_class: Group by provisional class

        Returns:
            List of ReviewItems
        """
        all_items = self.queue.get_pending(limit=batch_size * 3)

        if not same_class:
            return all_items[:batch_size]

        # Group by class
        by_class: Dict[str, List[ReviewItem]] = {}
        for item in all_items:
            cls = item.provisional_class
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(item)

        # Return largest group
        if by_class:
            largest = max(by_class.values(), key=len)
            return largest[:batch_size]

        return []

    def submit_batch(
        self,
        items: List[ReviewItem],
        human_class: str,
        reviewer: str,
    ) -> List[ReviewResult]:
        """Submit batch of reviews with same classification.

        Args:
            items: Items to review
            human_class: Class to assign to all
            reviewer: Reviewer identifier

        Returns:
            List of ReviewResults
        """
        results = []
        for item in items:
            result = self.queue.submit_review(
                review_id=item.review_id,
                human_class=human_class,
                reviewer=reviewer,
            )
            results.append(result)

        return results

    def skip_batch(
        self,
        items: List[ReviewItem],
        reason: str = "batch_skip",
    ) -> int:
        """Skip a batch of reviews.

        Args:
            items: Items to skip
            reason: Skip reason

        Returns:
            Number of items skipped
        """
        for item in items:
            self.queue.skip_review(item.review_id, reason)

        return len(items)

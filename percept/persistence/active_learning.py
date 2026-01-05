"""Active learning system for PERCEPT.

Collects feedback from human reviews, tracks model accuracy,
and prepares training data for model improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import random

import numpy as np

from percept.persistence.database import PerceptDatabase
from percept.persistence.review import ReviewResult, HumanReviewQueue


@dataclass
class FeedbackEntry:
    """A single feedback entry from human review."""
    object_id: str
    original_class: str
    original_confidence: float
    corrected_class: str
    was_correct: bool
    reviewer: str
    timestamp: datetime
    image_path: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    pipeline: str = ""  # Which pipeline made the prediction


@dataclass
class AccuracyMetrics:
    """Model accuracy metrics."""
    total_reviewed: int = 0
    correct: int = 0
    incorrect: int = 0
    accuracy: float = 0.0
    precision_by_class: Dict[str, float] = field(default_factory=dict)
    recall_by_class: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def calculate(self) -> None:
        """Calculate accuracy from counts."""
        if self.total_reviewed > 0:
            self.accuracy = self.correct / self.total_reviewed
        else:
            self.accuracy = 0.0


@dataclass
class TrainingExample:
    """A training example for model improvement."""
    image_path: str
    label: str
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0  # Human-verified = high confidence
    source: str = "human_review"
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning system."""

    # Feedback collection
    feedback_storage_path: str = "data/feedback"
    max_feedback_entries: int = 100000

    # Training data
    training_data_path: str = "data/training"
    min_examples_per_class: int = 10
    max_examples_per_class: int = 1000

    # Model tracking
    track_accuracy_window_hours: float = 24.0
    accuracy_alert_threshold: float = 0.7  # Alert if accuracy drops below

    # Active learning
    uncertainty_sampling: bool = True  # Prioritize uncertain examples
    diversity_sampling: bool = True  # Ensure class diversity
    hard_negative_mining: bool = True  # Focus on misclassified examples

    # Export settings
    export_format: str = "json"  # "json", "csv", "tfrecord"


class FeedbackCollector:
    """Collects and stores feedback from human reviews."""

    def __init__(
        self,
        config: Optional[ActiveLearningConfig] = None,
    ):
        """Initialize feedback collector.

        Args:
            config: Configuration options
        """
        self.config = config or ActiveLearningConfig()
        self._feedback: List[FeedbackEntry] = []
        self._feedback_path = Path(self.config.feedback_storage_path)
        self._feedback_path.mkdir(parents=True, exist_ok=True)

    def add_feedback(self, result: ReviewResult, image_path: str = "") -> None:
        """Add feedback from review result.

        Args:
            result: Review result
            image_path: Path to image crop
        """
        entry = FeedbackEntry(
            object_id=result.object_id,
            original_class=result.original_class,
            original_confidence=result.original_confidence,
            corrected_class=result.human_class,
            was_correct=result.was_correct,
            reviewer=result.reviewer,
            timestamp=result.reviewed_at,
            image_path=image_path,
            attributes=result.human_attributes,
        )

        self._feedback.append(entry)

        # Persist if at capacity
        if len(self._feedback) >= self.config.max_feedback_entries:
            self.save()
            self._feedback = self._feedback[-1000:]  # Keep recent

    def get_feedback(
        self,
        since: Optional[datetime] = None,
        class_filter: Optional[str] = None,
        only_corrections: bool = False,
    ) -> List[FeedbackEntry]:
        """Get feedback entries.

        Args:
            since: Only entries after this time
            class_filter: Filter by corrected class
            only_corrections: Only include corrections (was_correct=False)

        Returns:
            List of FeedbackEntry
        """
        results = []

        for entry in self._feedback:
            if since and entry.timestamp < since:
                continue
            if class_filter and entry.corrected_class != class_filter:
                continue
            if only_corrections and entry.was_correct:
                continue
            results.append(entry)

        return results

    def get_corrections(self) -> List[FeedbackEntry]:
        """Get all corrections (where human changed the class).

        Returns:
            List of correction entries
        """
        return self.get_feedback(only_corrections=True)

    def save(self, path: Optional[str] = None) -> None:
        """Save feedback to disk.

        Args:
            path: Optional override path
        """
        save_path = Path(path) if path else self._feedback_path / "feedback.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for entry in self._feedback:
            data.append({
                "object_id": entry.object_id,
                "original_class": entry.original_class,
                "original_confidence": entry.original_confidence,
                "corrected_class": entry.corrected_class,
                "was_correct": entry.was_correct,
                "reviewer": entry.reviewer,
                "timestamp": entry.timestamp.isoformat(),
                "image_path": entry.image_path,
                "attributes": entry.attributes,
                "pipeline": entry.pipeline,
            })

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> int:
        """Load feedback from disk.

        Args:
            path: Optional override path

        Returns:
            Number of entries loaded
        """
        load_path = Path(path) if path else self._feedback_path / "feedback.json"

        if not load_path.exists():
            return 0

        with open(load_path, 'r') as f:
            data = json.load(f)

        for item in data:
            entry = FeedbackEntry(
                object_id=item["object_id"],
                original_class=item["original_class"],
                original_confidence=item.get("original_confidence", 0.0),
                corrected_class=item["corrected_class"],
                was_correct=item["was_correct"],
                reviewer=item["reviewer"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
                image_path=item.get("image_path"),
                attributes=item.get("attributes", {}),
                pipeline=item.get("pipeline", ""),
            )
            self._feedback.append(entry)

        return len(data)

    def count(self) -> int:
        """Get total feedback count."""
        return len(self._feedback)


class AccuracyTracker:
    """Tracks model accuracy over time."""

    def __init__(
        self,
        config: Optional[ActiveLearningConfig] = None,
    ):
        """Initialize accuracy tracker.

        Args:
            config: Configuration options
        """
        self.config = config or ActiveLearningConfig()
        self._predictions: Dict[str, Dict[str, int]] = {}  # class -> {correct, incorrect}
        self._confusion: Dict[str, Dict[str, int]] = {}  # actual -> predicted -> count
        self._timeline: List[Tuple[datetime, bool]] = []  # (time, was_correct)

    def record(
        self,
        predicted_class: str,
        actual_class: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a prediction result.

        Args:
            predicted_class: Model's prediction
            actual_class: Human-verified class
            timestamp: When prediction was made
        """
        ts = timestamp or datetime.now()
        was_correct = (predicted_class == actual_class)

        # Update prediction counts
        if predicted_class not in self._predictions:
            self._predictions[predicted_class] = {"correct": 0, "incorrect": 0}

        if was_correct:
            self._predictions[predicted_class]["correct"] += 1
        else:
            self._predictions[predicted_class]["incorrect"] += 1

        # Update confusion matrix
        if actual_class not in self._confusion:
            self._confusion[actual_class] = {}
        if predicted_class not in self._confusion[actual_class]:
            self._confusion[actual_class][predicted_class] = 0
        self._confusion[actual_class][predicted_class] += 1

        # Update timeline
        self._timeline.append((ts, was_correct))

        # Trim old entries
        self._trim_timeline()

    def _trim_timeline(self) -> None:
        """Remove old timeline entries."""
        cutoff = datetime.now() - timedelta(hours=self.config.track_accuracy_window_hours * 2)
        self._timeline = [(ts, c) for ts, c in self._timeline if ts >= cutoff]

    def get_metrics(self, time_window_hours: Optional[float] = None) -> AccuracyMetrics:
        """Get accuracy metrics.

        Args:
            time_window_hours: Only include recent entries

        Returns:
            AccuracyMetrics
        """
        metrics = AccuracyMetrics()

        # Filter by time if specified
        if time_window_hours:
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            recent = [(ts, c) for ts, c in self._timeline if ts >= cutoff]
        else:
            recent = self._timeline

        # Calculate overall accuracy
        metrics.total_reviewed = len(recent)
        metrics.correct = sum(1 for _, c in recent if c)
        metrics.incorrect = metrics.total_reviewed - metrics.correct
        metrics.calculate()

        # Calculate per-class metrics
        for cls, counts in self._predictions.items():
            total = counts["correct"] + counts["incorrect"]
            if total > 0:
                metrics.precision_by_class[cls] = counts["correct"] / total

        # Calculate recall (from confusion matrix)
        for actual_class, predictions in self._confusion.items():
            total_actual = sum(predictions.values())
            if total_actual > 0:
                correct = predictions.get(actual_class, 0)
                metrics.recall_by_class[actual_class] = correct / total_actual

        metrics.confusion_matrix = self._confusion

        return metrics

    def get_recent_accuracy(self) -> float:
        """Get accuracy in configured time window.

        Returns:
            Accuracy as float (0-1)
        """
        metrics = self.get_metrics(self.config.track_accuracy_window_hours)
        return metrics.accuracy

    def check_accuracy_alert(self) -> bool:
        """Check if accuracy has dropped below threshold.

        Returns:
            True if accuracy is below threshold
        """
        return self.get_recent_accuracy() < self.config.accuracy_alert_threshold

    def get_worst_classes(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get classes with lowest accuracy.

        Args:
            n: Number of classes to return

        Returns:
            List of (class, accuracy) tuples
        """
        class_accuracy = []

        for cls, counts in self._predictions.items():
            total = counts["correct"] + counts["incorrect"]
            if total >= 5:  # Minimum samples
                acc = counts["correct"] / total
                class_accuracy.append((cls, acc))

        class_accuracy.sort(key=lambda x: x[1])
        return class_accuracy[:n]

    def get_most_confused(self, n: int = 5) -> List[Tuple[str, str, int]]:
        """Get most common confusion pairs.

        Args:
            n: Number of pairs to return

        Returns:
            List of (actual, predicted, count) tuples
        """
        confusions = []

        for actual, predictions in self._confusion.items():
            for predicted, count in predictions.items():
                if actual != predicted:  # Only misclassifications
                    confusions.append((actual, predicted, count))

        confusions.sort(key=lambda x: x[2], reverse=True)
        return confusions[:n]


class TrainingDataExporter:
    """Exports training data from feedback for model improvement."""

    def __init__(
        self,
        config: Optional[ActiveLearningConfig] = None,
    ):
        """Initialize exporter.

        Args:
            config: Configuration options
        """
        self.config = config or ActiveLearningConfig()
        self._export_path = Path(self.config.training_data_path)
        self._export_path.mkdir(parents=True, exist_ok=True)

    def create_examples(
        self,
        feedback: List[FeedbackEntry],
        include_embeddings: bool = False,
    ) -> List[TrainingExample]:
        """Create training examples from feedback.

        Args:
            feedback: Feedback entries
            include_embeddings: Include embeddings in examples

        Returns:
            List of TrainingExample
        """
        examples = []

        for entry in feedback:
            if not entry.image_path:
                continue

            example = TrainingExample(
                image_path=entry.image_path,
                label=entry.corrected_class,
                embedding=entry.embedding if include_embeddings else None,
                confidence=1.0,  # Human verified
                source="human_review",
                attributes=entry.attributes,
            )
            examples.append(example)

        return examples

    def balance_classes(
        self,
        examples: List[TrainingExample],
    ) -> List[TrainingExample]:
        """Balance training examples across classes.

        Args:
            examples: Training examples

        Returns:
            Balanced list of examples
        """
        # Group by class
        by_class: Dict[str, List[TrainingExample]] = {}
        for ex in examples:
            if ex.label not in by_class:
                by_class[ex.label] = []
            by_class[ex.label].append(ex)

        # Determine target count
        counts = [len(v) for v in by_class.values()]
        if not counts:
            return []

        target = min(
            max(counts),
            self.config.max_examples_per_class
        )

        # Balance
        balanced = []
        for cls, cls_examples in by_class.items():
            if len(cls_examples) >= self.config.min_examples_per_class:
                # Oversample if needed
                if len(cls_examples) < target:
                    sampled = cls_examples * (target // len(cls_examples) + 1)
                    balanced.extend(sampled[:target])
                else:
                    balanced.extend(random.sample(cls_examples, min(len(cls_examples), target)))

        return balanced

    def export_json(
        self,
        examples: List[TrainingExample],
        filename: str = "training_data.json",
    ) -> str:
        """Export examples to JSON format.

        Args:
            examples: Training examples
            filename: Output filename

        Returns:
            Path to exported file
        """
        output_path = self._export_path / filename

        data = []
        for ex in examples:
            item = {
                "image_path": ex.image_path,
                "label": ex.label,
                "confidence": ex.confidence,
                "source": ex.source,
                "attributes": ex.attributes,
            }
            if ex.embedding is not None:
                item["embedding"] = ex.embedding.tolist()
            data.append(item)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return str(output_path)

    def export_csv(
        self,
        examples: List[TrainingExample],
        filename: str = "training_data.csv",
    ) -> str:
        """Export examples to CSV format.

        Args:
            examples: Training examples
            filename: Output filename

        Returns:
            Path to exported file
        """
        import csv

        output_path = self._export_path / filename

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label", "confidence", "source"])

            for ex in examples:
                writer.writerow([
                    ex.image_path,
                    ex.label,
                    ex.confidence,
                    ex.source,
                ])

        return str(output_path)

    def get_class_distribution(
        self,
        examples: List[TrainingExample],
    ) -> Dict[str, int]:
        """Get distribution of classes in examples.

        Args:
            examples: Training examples

        Returns:
            Dictionary of class -> count
        """
        distribution: Dict[str, int] = {}
        for ex in examples:
            distribution[ex.label] = distribution.get(ex.label, 0) + 1
        return distribution


class ActiveLearningManager:
    """Manages the complete active learning pipeline.

    Coordinates feedback collection, accuracy tracking, and training data export.
    """

    def __init__(
        self,
        db: PerceptDatabase,
        config: Optional[ActiveLearningConfig] = None,
    ):
        """Initialize active learning manager.

        Args:
            db: Database connection
            config: Configuration options
        """
        self.config = config or ActiveLearningConfig()
        self.db = db

        self.feedback_collector = FeedbackCollector(self.config)
        self.accuracy_tracker = AccuracyTracker(self.config)
        self.exporter = TrainingDataExporter(self.config)

        # Load existing feedback
        self.feedback_collector.load()

    def process_review(
        self,
        result: ReviewResult,
        image_path: str = "",
    ) -> None:
        """Process a completed review.

        Args:
            result: Review result
            image_path: Path to image crop
        """
        # Collect feedback
        self.feedback_collector.add_feedback(result, image_path)

        # Track accuracy
        self.accuracy_tracker.record(
            predicted_class=result.original_class,
            actual_class=result.human_class,
            timestamp=result.reviewed_at,
        )

    def get_accuracy_report(self) -> Dict[str, Any]:
        """Get comprehensive accuracy report.

        Returns:
            Dictionary with accuracy information
        """
        metrics = self.accuracy_tracker.get_metrics()

        return {
            "overall_accuracy": metrics.accuracy,
            "total_reviewed": metrics.total_reviewed,
            "correct": metrics.correct,
            "incorrect": metrics.incorrect,
            "precision_by_class": metrics.precision_by_class,
            "recall_by_class": metrics.recall_by_class,
            "worst_classes": self.accuracy_tracker.get_worst_classes(),
            "most_confused": self.accuracy_tracker.get_most_confused(),
            "accuracy_alert": self.accuracy_tracker.check_accuracy_alert(),
        }

    def export_training_data(
        self,
        since: Optional[datetime] = None,
        balance: bool = True,
        format: str = "json",
    ) -> Tuple[str, int]:
        """Export training data from feedback.

        Args:
            since: Only include feedback after this time
            balance: Balance classes
            format: Export format ("json" or "csv")

        Returns:
            Tuple of (output_path, num_examples)
        """
        # Get feedback
        feedback = self.feedback_collector.get_feedback(since=since)

        # Create examples
        examples = self.exporter.create_examples(feedback)

        # Balance if requested
        if balance:
            examples = self.exporter.balance_classes(examples)

        # Export
        if format == "csv":
            path = self.exporter.export_csv(examples)
        else:
            path = self.exporter.export_json(examples)

        return path, len(examples)

    def get_sampling_priorities(
        self,
        candidates: List[Tuple[str, str, float]],  # (object_id, class, confidence)
    ) -> List[Tuple[str, float]]:
        """Get sampling priorities for candidate objects.

        Implements uncertainty and diversity sampling for active learning.

        Args:
            candidates: List of (object_id, class, confidence) tuples

        Returns:
            List of (object_id, priority) tuples, sorted by priority
        """
        priorities = []

        # Get class distribution in feedback
        class_counts = {}
        for entry in self.feedback_collector.get_feedback():
            cls = entry.corrected_class
            class_counts[cls] = class_counts.get(cls, 0) + 1

        # Get worst performing classes
        worst_classes = {cls for cls, _ in self.accuracy_tracker.get_worst_classes()}

        for obj_id, cls, confidence in candidates:
            priority = 0.0

            # Uncertainty sampling: prioritize low confidence
            if self.config.uncertainty_sampling:
                priority += (1.0 - confidence)

            # Diversity sampling: prioritize underrepresented classes
            if self.config.diversity_sampling:
                class_count = class_counts.get(cls, 0)
                if class_count < self.config.min_examples_per_class:
                    priority += 0.5

            # Hard negative mining: prioritize classes with low accuracy
            if self.config.hard_negative_mining:
                if cls in worst_classes:
                    priority += 0.3

            priorities.append((obj_id, priority))

        # Sort by priority (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)

        return priorities

    def suggest_review_candidates(
        self,
        n: int = 20,
    ) -> List[str]:
        """Suggest objects for human review based on active learning.

        Args:
            n: Number of suggestions

        Returns:
            List of object IDs to review
        """
        # Get objects needing review from DB
        candidates = self.db.query_needs_review(limit=n * 3)

        # Convert to priority format
        candidate_tuples = [
            (obj.id, obj.primary_class, obj.confidence)
            for obj in candidates
        ]

        # Get priorities
        priorities = self.get_sampling_priorities(candidate_tuples)

        # Return top N
        return [obj_id for obj_id, _ in priorities[:n]]

    def save(self) -> None:
        """Save all state to disk."""
        self.feedback_collector.save()

    def get_stats(self) -> Dict[str, Any]:
        """Get active learning statistics.

        Returns:
            Dictionary of statistics
        """
        feedback = self.feedback_collector.get_feedback()
        corrections = self.feedback_collector.get_corrections()

        return {
            "total_feedback": len(feedback),
            "total_corrections": len(corrections),
            "accuracy_report": self.get_accuracy_report(),
            "class_distribution": self.exporter.get_class_distribution(
                self.exporter.create_examples(feedback)
            ),
        }

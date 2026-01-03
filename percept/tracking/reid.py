"""ReID embedding extraction and matching for PERCEPT.

Provides object re-identification using:
- Deep embeddings (RepVGG) for persons via Hailo-8
- Color/texture histograms for generic objects
- FAISS-backed gallery matching
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.core.schema import ObjectSchema


class EmbeddingType(Enum):
    """Type of embedding used for ReID."""
    DEEP = "deep"  # Deep learning embedding (RepVGG)
    HISTOGRAM = "histogram"  # Color/texture histogram
    HYBRID = "hybrid"  # Combination of both


@dataclass
class ReIDConfig:
    """Configuration for ReID system."""

    # Embedding settings
    embedding_dimension: int = 512
    use_deep_embeddings: bool = True  # Use Hailo-8 when available
    histogram_bins: int = 32  # Bins per channel for histogram embeddings

    # Matching thresholds (cosine distance)
    match_threshold_same_camera: float = 0.3
    match_threshold_cross_camera: float = 0.25
    new_object_threshold: float = 0.5  # Above this = definitely new

    # Gallery settings
    max_embeddings_per_object: int = 10
    embedding_update_interval: int = 3  # Frames between embedding updates

    # Model paths
    reid_model_path: str = "/usr/share/hailo-models/repvgg_a0_person_reid_512.hef"


def is_faiss_available() -> bool:
    """Check if FAISS is available."""
    try:
        import faiss
        return True
    except ImportError:
        return False


def is_hailo_available() -> bool:
    """Check if Hailo runtime is available."""
    try:
        from hailo_platform import HEF, VDevice
        return True
    except ImportError:
        return False


FAISS_AVAILABLE = is_faiss_available()
HAILO_AVAILABLE = is_hailo_available()


class HistogramEmbedding:
    """Generate histogram-based embeddings for generic objects.

    Combines color histogram and texture features into a fixed-size vector.
    """

    def __init__(self, config: Optional[ReIDConfig] = None):
        """Initialize histogram embedding generator.

        Args:
            config: Configuration options
        """
        self.config = config or ReIDConfig()
        self.bins = self.config.histogram_bins

    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract histogram embedding from image region.

        Args:
            image: BGR image crop (H, W, 3)
            mask: Optional binary mask for the object

        Returns:
            L2-normalized embedding vector
        """
        if mask is not None:
            mask_bool = mask > 0
        else:
            mask_bool = np.ones(image.shape[:2], dtype=bool)

        # Color histogram in HSV space (more robust to lighting)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        color_features = []
        for channel in range(3):
            channel_data = hsv[:, :, channel][mask_bool]
            if len(channel_data) == 0:
                hist = np.zeros(self.bins)
            else:
                max_val = 180 if channel == 0 else 256  # H is 0-180, S/V are 0-255
                hist, _ = np.histogram(channel_data, bins=self.bins, range=(0, max_val))
                hist = hist.astype(np.float32)
                if hist.sum() > 0:
                    hist = hist / hist.sum()
            color_features.append(hist)

        # Texture features using Local Binary Pattern approximation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture_features = self._compute_texture_features(gray, mask_bool)

        # Combine features
        embedding = np.concatenate([
            np.concatenate(color_features),  # 3 * bins = 96
            texture_features,  # Additional texture features
        ])

        # Pad or truncate to target dimension
        target_dim = self.config.embedding_dimension
        if len(embedding) < target_dim:
            embedding = np.pad(embedding, (0, target_dim - len(embedding)))
        else:
            embedding = embedding[:target_dim]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def _compute_texture_features(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Compute texture features from grayscale image."""
        # Gradient magnitude histogram
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        mag_values = magnitude[mask]
        if len(mag_values) > 0:
            mag_hist, _ = np.histogram(mag_values, bins=self.bins, range=(0, 255))
            mag_hist = mag_hist.astype(np.float32)
            if mag_hist.sum() > 0:
                mag_hist = mag_hist / mag_hist.sum()
        else:
            mag_hist = np.zeros(self.bins, dtype=np.float32)

        # Gradient direction histogram
        direction = np.arctan2(grad_y, grad_x)
        dir_values = direction[mask]
        if len(dir_values) > 0:
            dir_hist, _ = np.histogram(dir_values, bins=self.bins, range=(-np.pi, np.pi))
            dir_hist = dir_hist.astype(np.float32)
            if dir_hist.sum() > 0:
                dir_hist = dir_hist / dir_hist.sum()
        else:
            dir_hist = np.zeros(self.bins, dtype=np.float32)

        return np.concatenate([mag_hist, dir_hist])


class DeepEmbedding:
    """Generate deep embeddings using Hailo-8 accelerated model.

    Uses RepVGG architecture trained for person re-identification.
    Falls back to histogram embeddings when hardware unavailable.
    """

    def __init__(self, config: Optional[ReIDConfig] = None):
        """Initialize deep embedding generator.

        Args:
            config: Configuration options
        """
        self.config = config or ReIDConfig()
        self._inference = None
        self._configured = False
        self._fallback = HistogramEmbedding(config)

        if HAILO_AVAILABLE:
            self._try_load_model()

    def _try_load_model(self):
        """Try to load the Hailo model."""
        try:
            from hailo_platform import HEF, VDevice
            from hailo_platform import InputVStreamParams, OutputVStreamParams

            model_path = Path(self.config.reid_model_path)
            if not model_path.exists():
                return

            self.hef = HEF(str(model_path))
            self.vdevice = None
            self.network_group = None

            # Get model info
            self.input_vstream_info = self.hef.get_input_vstream_infos()
            if self.input_vstream_info:
                info = self.input_vstream_info[0]
                self.input_shape = info.shape
                self.input_name = info.name

            self._inference = True
        except Exception:
            self._inference = None

    def _configure(self):
        """Configure Hailo device."""
        if self._configured or self._inference is None:
            return

        try:
            from hailo_platform import VDevice
            from hailo_platform import InputVStreamParams, OutputVStreamParams

            self.vdevice = VDevice()
            self.network_group = self.vdevice.configure(self.hef)[0]

            self._input_vstreams_params = InputVStreamParams.make(self.network_group)
            self._output_vstreams_params = OutputVStreamParams.make(self.network_group)

            self._configured = True
        except Exception:
            self._inference = None

    def is_available(self) -> bool:
        """Check if deep embeddings are available."""
        return self._inference is not None

    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract deep embedding from image region.

        Args:
            image: BGR image crop (H, W, 3)
            mask: Optional binary mask (unused for deep model)

        Returns:
            L2-normalized embedding vector
        """
        if not self.is_available():
            return self._fallback.extract(image, mask)

        if not self._configured:
            self._configure()
            if not self._configured:
                return self._fallback.extract(image, mask)

        try:
            from hailo_platform import InferVStreams

            # Preprocess
            preprocessed = self._preprocess(image)

            # Add batch dimension
            input_data = {self.input_name: np.expand_dims(preprocessed, 0)}

            # Run inference
            with InferVStreams(
                self.network_group,
                self._input_vstreams_params,
                self._output_vstreams_params
            ) as infer_pipeline:
                with self.network_group.activate():
                    results = infer_pipeline.infer(input_data)

            # Extract embedding from output
            embedding = None
            for name, output in results.items():
                embedding = output.flatten().astype(np.float32)
                break

            if embedding is None:
                return self._fallback.extract(image, mask)

            # Resize to target dimension if needed
            if len(embedding) != self.config.embedding_dimension:
                if len(embedding) > self.config.embedding_dimension:
                    embedding = embedding[:self.config.embedding_dimension]
                else:
                    embedding = np.pad(embedding, (0, self.config.embedding_dimension - len(embedding)))

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception:
            return self._fallback.extract(image, mask)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        target_h, target_w = self.input_shape[0], self.input_shape[1]

        # Resize with letterboxing
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to target size
        padded = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        return rgb


class ReIDExtractor:
    """Extract ReID embeddings from object crops.

    Automatically selects appropriate embedding method based on
    object class and hardware availability.
    """

    def __init__(self, config: Optional[ReIDConfig] = None):
        """Initialize extractor.

        Args:
            config: Configuration options
        """
        self.config = config or ReIDConfig()
        self.deep_extractor = DeepEmbedding(self.config) if self.config.use_deep_embeddings else None
        self.histogram_extractor = HistogramEmbedding(self.config)

    def extract(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None,
        object_class: str = "unknown",
    ) -> np.ndarray:
        """Extract ReID embedding for an object.

        Args:
            image: Full BGR image
            bbox: Bounding box (x1, y1, x2, y2)
            mask: Optional binary mask
            object_class: Object class for method selection

        Returns:
            L2-normalized embedding vector
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            # Invalid bbox, return zero embedding
            return np.zeros(self.config.embedding_dimension, dtype=np.float32)

        crop = image[y1:y2, x1:x2]

        # Crop mask if provided
        crop_mask = None
        if mask is not None:
            crop_mask = mask[y1:y2, x1:x2]

        # Use deep embeddings for persons if available
        if object_class == "person" and self.deep_extractor and self.deep_extractor.is_available():
            return self.deep_extractor.extract(crop, crop_mask)

        # Use histogram for other objects or as fallback
        return self.histogram_extractor.extract(crop, crop_mask)

    def extract_batch(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[np.ndarray]:
        """Extract embeddings for multiple detections.

        Args:
            image: Full BGR image
            detections: List of detection dicts with bbox, mask, class

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for det in detections:
            bbox = det.get("bbox", (0, 0, 0, 0))
            mask = det.get("mask")
            obj_class = det.get("class_name", "unknown")
            emb = self.extract(image, bbox, mask, obj_class)
            embeddings.append(emb)
        return embeddings


class ReIDMatcher:
    """Match ReID embeddings against a gallery of known objects.

    Uses cosine similarity for matching with configurable thresholds.
    """

    def __init__(self, config: Optional[ReIDConfig] = None):
        """Initialize matcher.

        Args:
            config: Configuration options
        """
        self.config = config or ReIDConfig()
        self._embeddings: Dict[str, List[np.ndarray]] = {}  # object_id -> list of embeddings
        self._metadata: Dict[str, Dict[str, Any]] = {}  # object_id -> metadata

    def add(
        self,
        object_id: str,
        embedding: np.ndarray,
        camera_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add embedding to gallery.

        Args:
            object_id: Unique object identifier
            embedding: L2-normalized embedding vector
            camera_id: Camera that captured this embedding
            metadata: Additional metadata
        """
        if object_id not in self._embeddings:
            self._embeddings[object_id] = []
            self._metadata[object_id] = metadata or {}

        embeddings_list = self._embeddings[object_id]
        embeddings_list.append(embedding.copy())

        # Limit embeddings per object
        max_emb = self.config.max_embeddings_per_object
        if len(embeddings_list) > max_emb:
            self._embeddings[object_id] = embeddings_list[-max_emb:]

        # Update metadata
        if metadata:
            self._metadata[object_id].update(metadata)
        if camera_id:
            self._metadata[object_id]["last_camera_id"] = camera_id

    def match(
        self,
        embedding: np.ndarray,
        camera_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find matching objects in gallery.

        Args:
            embedding: Query embedding
            camera_id: Camera that captured query (affects threshold)
            top_k: Maximum matches to return

        Returns:
            List of (object_id, similarity) tuples, sorted by similarity
        """
        if len(self._embeddings) == 0:
            return []

        results = []

        for obj_id, emb_list in self._embeddings.items():
            # Compute similarity to all embeddings for this object
            similarities = []
            for emb in emb_list:
                sim = self._cosine_similarity(embedding, emb)
                similarities.append(sim)

            # Use max similarity (best match across all embeddings)
            best_sim = max(similarities) if similarities else 0.0

            # Apply threshold based on camera
            obj_camera = self._metadata.get(obj_id, {}).get("last_camera_id")
            threshold = (self.config.match_threshold_same_camera
                        if camera_id and obj_camera == camera_id
                        else self.config.match_threshold_cross_camera)

            # Convert to distance (lower = better match)
            distance = 1.0 - best_sim
            if distance < threshold:
                results.append((obj_id, best_sim))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def find_best_match(
        self,
        embedding: np.ndarray,
        camera_id: Optional[str] = None,
    ) -> Optional[Tuple[str, float]]:
        """Find single best matching object.

        Args:
            embedding: Query embedding
            camera_id: Camera that captured query

        Returns:
            (object_id, similarity) or None if no match
        """
        matches = self.match(embedding, camera_id, top_k=1)
        return matches[0] if matches else None

    def is_new_object(
        self,
        embedding: np.ndarray,
        camera_id: Optional[str] = None,
    ) -> bool:
        """Determine if embedding represents a new object.

        Args:
            embedding: Query embedding
            camera_id: Camera that captured query

        Returns:
            True if definitely a new object
        """
        if len(self._embeddings) == 0:
            return True

        matches = self.match(embedding, camera_id, top_k=1)
        if not matches:
            return True

        # Check if best match is above new object threshold
        _, best_sim = matches[0]
        distance = 1.0 - best_sim
        return distance > self.config.new_object_threshold

    def remove(self, object_id: str) -> bool:
        """Remove object from gallery.

        Args:
            object_id: Object to remove

        Returns:
            True if object was removed
        """
        if object_id in self._embeddings:
            del self._embeddings[object_id]
            del self._metadata[object_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all objects from gallery."""
        self._embeddings.clear()
        self._metadata.clear()

    def size(self) -> int:
        """Get number of objects in gallery."""
        return len(self._embeddings)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


class ReIDModule(PipelineModule):
    """Pipeline module for ReID embedding extraction and matching.

    Extracts embeddings from detected objects and matches against gallery.
    """

    def __init__(self, config: Optional[ReIDConfig] = None):
        """Initialize module.

        Args:
            config: Configuration options
        """
        self.config = config or ReIDConfig()
        self.extractor = ReIDExtractor(self.config)
        self.matcher = ReIDMatcher(self.config)
        self._frame_count = 0

    @property
    def name(self) -> str:
        return "reid"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="detections",
            required_fields=["image", "objects"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="tracked_objects",
            required_fields=["objects"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Process detections through ReID.

        Args:
            data: PipelineData with image and objects

        Returns:
            PipelineData with updated objects (matched or new)
        """
        image = data.image
        objects = data.get("objects", [])
        camera_id = data.get("camera_id")

        self._frame_count += 1

        result = data.copy()
        processed_objects = []

        for obj in objects:
            # Extract embedding
            if isinstance(obj, ObjectSchema):
                bbox = obj.bounding_box_2d
                mask = getattr(obj, 'mask', None)
                obj_class = obj.primary_class
            elif isinstance(obj, dict):
                bbox = obj.get("bbox", (0, 0, 0, 0))
                mask = obj.get("mask")
                obj_class = obj.get("class_name", "unknown")
            else:
                continue

            embedding = self.extractor.extract(image, bbox, mask, obj_class)

            # Try to match
            match = self.matcher.find_best_match(embedding, camera_id)

            if match:
                matched_id, similarity = match
                # Update existing object
                if isinstance(obj, ObjectSchema):
                    obj.id = matched_id
                    obj.reid_embedding = embedding
                elif isinstance(obj, dict):
                    obj["id"] = matched_id
                    obj["reid_embedding"] = embedding
                    obj["reid_similarity"] = similarity
            else:
                # New object - add to gallery
                if isinstance(obj, ObjectSchema):
                    self.matcher.add(obj.id, embedding, camera_id)
                    obj.reid_embedding = embedding
                elif isinstance(obj, dict):
                    import uuid
                    new_id = str(uuid.uuid4())
                    obj["id"] = new_id
                    obj["reid_embedding"] = embedding
                    self.matcher.add(new_id, embedding, camera_id)

            processed_objects.append(obj)

        result.objects = processed_objects
        return result


def extract_reid_embedding(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mask: Optional[np.ndarray] = None,
    object_class: str = "unknown",
) -> np.ndarray:
    """Convenience function for ReID embedding extraction.

    Args:
        image: BGR image
        bbox: Bounding box (x1, y1, x2, y2)
        mask: Optional binary mask
        object_class: Object class

    Returns:
        L2-normalized embedding
    """
    extractor = ReIDExtractor()
    return extractor.extract(image, bbox, mask, object_class)


def match_embedding(
    embedding: np.ndarray,
    gallery: Dict[str, np.ndarray],
    threshold: float = 0.3,
) -> Optional[str]:
    """Convenience function for embedding matching.

    Args:
        embedding: Query embedding
        gallery: Dict of object_id -> embedding
        threshold: Match threshold (cosine distance)

    Returns:
        Matched object_id or None
    """
    matcher = ReIDMatcher()
    for obj_id, emb in gallery.items():
        matcher.add(obj_id, emb)

    match = matcher.find_best_match(embedding)
    return match[0] if match else None

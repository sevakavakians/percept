"""FAISS-backed embedding gallery for PERCEPT.

Provides efficient approximate nearest neighbor search for ReID embeddings.
Falls back to brute-force search when FAISS is not available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pickle

import numpy as np


def is_faiss_available() -> bool:
    """Check if FAISS is available."""
    try:
        import faiss
        return True
    except ImportError:
        return False


FAISS_AVAILABLE = is_faiss_available()


@dataclass
class GalleryConfig:
    """Configuration for embedding gallery."""

    # Index settings
    embedding_dimension: int = 512
    index_type: str = "flat"  # "flat", "hnsw", "ivf"
    hnsw_m: int = 16  # HNSW connectivity parameter
    nlist: int = 100  # IVF number of clusters

    # Search settings
    default_k: int = 5  # Default number of results
    normalize_embeddings: bool = True

    # Persistence
    auto_save: bool = False
    save_path: Optional[str] = None
    save_interval: int = 100  # Operations between auto-saves


@dataclass
class GalleryEntry:
    """An entry in the embedding gallery."""
    object_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class FAISSGallery:
    """FAISS-backed gallery for efficient embedding search.

    Stores embeddings with associated object IDs and metadata.
    Falls back to numpy-based brute force search if FAISS unavailable.
    """

    def __init__(self, config: Optional[GalleryConfig] = None):
        """Initialize gallery.

        Args:
            config: Configuration options
        """
        self.config = config or GalleryConfig()
        self._entries: List[GalleryEntry] = []
        self._id_to_indices: Dict[str, List[int]] = {}  # object_id -> list of indices
        self._index = None
        self._faiss_available = FAISS_AVAILABLE
        self._operations_since_save = 0

        self._build_index()

    def _build_index(self):
        """Build or rebuild the FAISS index."""
        if not self._faiss_available:
            return

        import faiss

        dim = self.config.embedding_dimension

        if self.config.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m)
        elif self.config.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, self.config.nlist)
        else:  # flat
            self._index = faiss.IndexFlatL2(dim)

    def add(
        self,
        object_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Add embedding to gallery.

        Args:
            object_id: Unique object identifier
            embedding: Embedding vector (will be normalized if configured)
            metadata: Optional metadata
            timestamp: Optional timestamp

        Returns:
            Index of added entry
        """
        import time

        # Normalize if configured
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Ensure correct dimension
        if len(embedding) != self.config.embedding_dimension:
            if len(embedding) > self.config.embedding_dimension:
                embedding = embedding[:self.config.embedding_dimension]
            else:
                embedding = np.pad(embedding, (0, self.config.embedding_dimension - len(embedding)))

        embedding = embedding.astype(np.float32)

        entry = GalleryEntry(
            object_id=object_id,
            embedding=embedding.copy(),
            metadata=metadata or {},
            timestamp=timestamp or time.time(),
        )

        idx = len(self._entries)
        self._entries.append(entry)

        # Update ID mapping
        if object_id not in self._id_to_indices:
            self._id_to_indices[object_id] = []
        self._id_to_indices[object_id].append(idx)

        # Add to FAISS index
        if self._faiss_available and self._index is not None:
            # IVF index needs training
            if hasattr(self._index, 'is_trained') and not self._index.is_trained:
                if len(self._entries) >= self.config.nlist:
                    embeddings = np.vstack([e.embedding for e in self._entries])
                    self._index.train(embeddings)
                    self._index.add(embeddings)
            else:
                self._index.add(embedding.reshape(1, -1))

        # Auto-save if configured
        self._operations_since_save += 1
        if self.config.auto_save and self._operations_since_save >= self.config.save_interval:
            self._auto_save()

        return idx

    def search(
        self,
        query: np.ndarray,
        k: Optional[int] = None,
        return_distances: bool = True,
    ) -> List[Tuple[str, float, int]]:
        """Search for nearest embeddings.

        Args:
            query: Query embedding vector
            k: Number of results (default from config)
            return_distances: Include distances in results

        Returns:
            List of (object_id, distance, index) tuples
        """
        if len(self._entries) == 0:
            return []

        if k is None:
            k = self.config.default_k
        k = min(k, len(self._entries))

        # Normalize query if configured
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

        query = query.astype(np.float32).reshape(1, -1)

        if self._faiss_available and self._index is not None and self._index.ntotal > 0:
            # FAISS search
            distances, indices = self._index.search(query, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self._entries):
                    entry = self._entries[idx]
                    results.append((entry.object_id, float(dist), int(idx)))
            return results
        else:
            # Brute force search
            return self._brute_force_search(query.flatten(), k)

    def _brute_force_search(
        self,
        query: np.ndarray,
        k: int,
    ) -> List[Tuple[str, float, int]]:
        """Brute force nearest neighbor search."""
        if len(self._entries) == 0:
            return []

        embeddings = np.vstack([e.embedding for e in self._entries])

        # Compute L2 distances
        distances = np.sum((embeddings - query) ** 2, axis=1)

        # Get top-k
        top_indices = np.argsort(distances)[:k]

        results = []
        for idx in top_indices:
            entry = self._entries[idx]
            results.append((entry.object_id, float(distances[idx]), int(idx)))

        return results

    def search_by_similarity(
        self,
        query: np.ndarray,
        threshold: float,
        max_results: int = 100,
    ) -> List[Tuple[str, float, int]]:
        """Search for embeddings within similarity threshold.

        Args:
            query: Query embedding
            threshold: Maximum L2 distance threshold
            max_results: Maximum results to return

        Returns:
            List of (object_id, distance, index) tuples within threshold
        """
        results = self.search(query, k=max_results)
        return [(obj_id, dist, idx) for obj_id, dist, idx in results if dist < threshold]

    def get_by_object_id(self, object_id: str) -> List[GalleryEntry]:
        """Get all entries for an object ID.

        Args:
            object_id: Object identifier

        Returns:
            List of GalleryEntry instances
        """
        if object_id not in self._id_to_indices:
            return []
        return [self._entries[idx] for idx in self._id_to_indices[object_id]]

    def get_embedding(self, object_id: str) -> Optional[np.ndarray]:
        """Get most recent embedding for an object.

        Args:
            object_id: Object identifier

        Returns:
            Embedding vector or None
        """
        entries = self.get_by_object_id(object_id)
        if not entries:
            return None
        # Return most recent
        return max(entries, key=lambda e: e.timestamp).embedding

    def get_average_embedding(self, object_id: str) -> Optional[np.ndarray]:
        """Get average embedding for an object.

        Args:
            object_id: Object identifier

        Returns:
            Average embedding or None
        """
        entries = self.get_by_object_id(object_id)
        if not entries:
            return None

        embeddings = np.vstack([e.embedding for e in entries])
        avg = np.mean(embeddings, axis=0)

        # Normalize
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        return avg.astype(np.float32)

    def remove_object(self, object_id: str) -> int:
        """Remove all entries for an object.

        Note: This doesn't rebuild FAISS index - entries are just marked.
        For production use, periodic rebuilding is recommended.

        Args:
            object_id: Object to remove

        Returns:
            Number of entries removed
        """
        if object_id not in self._id_to_indices:
            return 0

        indices = self._id_to_indices[object_id]
        count = len(indices)

        # Mark entries as removed (set object_id to empty)
        for idx in indices:
            self._entries[idx].object_id = ""

        del self._id_to_indices[object_id]

        return count

    def clear(self):
        """Clear all entries and rebuild index."""
        self._entries.clear()
        self._id_to_indices.clear()
        self._build_index()

    def size(self) -> int:
        """Get total number of entries."""
        return len(self._entries)

    def object_count(self) -> int:
        """Get number of unique objects."""
        return len(self._id_to_indices)

    def rebuild_index(self):
        """Rebuild FAISS index from current entries.

        Use this after many removals to reclaim memory and
        improve search performance.
        """
        # Filter out removed entries
        active_entries = [e for e in self._entries if e.object_id]

        self._entries = active_entries
        self._id_to_indices.clear()

        for idx, entry in enumerate(self._entries):
            if entry.object_id not in self._id_to_indices:
                self._id_to_indices[entry.object_id] = []
            self._id_to_indices[entry.object_id].append(idx)

        # Rebuild FAISS index
        self._build_index()
        if self._faiss_available and self._index is not None and len(self._entries) > 0:
            embeddings = np.vstack([e.embedding for e in self._entries])
            if hasattr(self._index, 'is_trained') and not self._index.is_trained:
                if len(embeddings) >= self.config.nlist:
                    self._index.train(embeddings)
            self._index.add(embeddings)

    def save(self, path: str):
        """Save gallery to disk.

        Args:
            path: File path for saving
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": {
                "embedding_dimension": self.config.embedding_dimension,
                "index_type": self.config.index_type,
                "hnsw_m": self.config.hnsw_m,
                "nlist": self.config.nlist,
                "normalize_embeddings": self.config.normalize_embeddings,
            },
            "entries": [
                {
                    "object_id": e.object_id,
                    "embedding": e.embedding.tolist(),
                    "metadata": e.metadata,
                    "timestamp": e.timestamp,
                }
                for e in self._entries if e.object_id
            ]
        }

        with open(save_path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> FAISSGallery:
        """Load gallery from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded FAISSGallery instance
        """
        with open(path, 'r') as f:
            data = json.load(f)

        config = GalleryConfig(**data["config"])
        gallery = cls(config)

        for entry_data in data["entries"]:
            gallery.add(
                object_id=entry_data["object_id"],
                embedding=np.array(entry_data["embedding"], dtype=np.float32),
                metadata=entry_data.get("metadata", {}),
                timestamp=entry_data.get("timestamp", 0.0),
            )

        return gallery

    def _auto_save(self):
        """Auto-save if configured."""
        if self.config.save_path:
            self.save(self.config.save_path)
            self._operations_since_save = 0


class MultiCameraGallery:
    """Gallery with camera-aware matching.

    Maintains separate galleries per camera with cross-camera matching support.
    """

    def __init__(self, config: Optional[GalleryConfig] = None):
        """Initialize multi-camera gallery.

        Args:
            config: Configuration options
        """
        self.config = config or GalleryConfig()
        self._cameras: Dict[str, FAISSGallery] = {}
        self._global_gallery = FAISSGallery(config)

    def add(
        self,
        object_id: str,
        embedding: np.ndarray,
        camera_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add embedding to camera-specific and global galleries.

        Args:
            object_id: Object identifier
            embedding: Embedding vector
            camera_id: Camera identifier
            metadata: Optional metadata
        """
        # Add to camera-specific gallery
        if camera_id not in self._cameras:
            self._cameras[camera_id] = FAISSGallery(self.config)

        camera_gallery = self._cameras[camera_id]
        camera_gallery.add(object_id, embedding, metadata)

        # Add to global gallery
        full_metadata = dict(metadata or {})
        full_metadata["camera_id"] = camera_id
        self._global_gallery.add(object_id, embedding, full_metadata)

    def search_same_camera(
        self,
        query: np.ndarray,
        camera_id: str,
        k: int = 5,
    ) -> List[Tuple[str, float, int]]:
        """Search within same camera gallery.

        Args:
            query: Query embedding
            camera_id: Camera to search
            k: Number of results

        Returns:
            List of (object_id, distance, index) tuples
        """
        if camera_id not in self._cameras:
            return []
        return self._cameras[camera_id].search(query, k)

    def search_cross_camera(
        self,
        query: np.ndarray,
        exclude_camera: Optional[str] = None,
        k: int = 5,
    ) -> List[Tuple[str, float, int]]:
        """Search across all cameras.

        Args:
            query: Query embedding
            exclude_camera: Camera to exclude from results
            k: Number of results

        Returns:
            List of (object_id, distance, index) tuples
        """
        results = self._global_gallery.search(query, k * 2)  # Get more, then filter

        if exclude_camera:
            filtered = []
            for obj_id, dist, idx in results:
                entry = self._global_gallery._entries[idx]
                if entry.metadata.get("camera_id") != exclude_camera:
                    filtered.append((obj_id, dist, idx))
            results = filtered[:k]

        return results[:k]

    def get_camera_gallery(self, camera_id: str) -> Optional[FAISSGallery]:
        """Get gallery for specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Camera-specific gallery or None
        """
        return self._cameras.get(camera_id)

    def clear(self):
        """Clear all galleries."""
        for gallery in self._cameras.values():
            gallery.clear()
        self._cameras.clear()
        self._global_gallery.clear()

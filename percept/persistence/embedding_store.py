"""Persistent embedding store for PERCEPT.

Provides FAISS-backed embedding storage with SQLite synchronization.
Supports both in-memory operations and disk persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import struct

import numpy as np

from percept.persistence.database import PerceptDatabase
from percept.tracking.gallery import FAISSGallery, GalleryConfig, GalleryEntry


@dataclass
class EmbeddingStoreConfig:
    """Configuration for embedding store."""

    # Storage paths
    db_path: str = "data/percept.db"
    index_path: str = "data/embeddings.index"

    # Embedding settings
    embedding_dimension: int = 512
    normalize_embeddings: bool = True

    # FAISS settings
    index_type: str = "flat"  # "flat", "hnsw", "ivf"
    hnsw_m: int = 16
    nlist: int = 100

    # Sync settings
    sync_interval: int = 50  # Operations between DB syncs
    auto_sync: bool = True
    persist_on_add: bool = False  # Write to DB on every add

    # Cache settings
    cache_size: int = 10000  # Max embeddings in memory
    eviction_strategy: str = "lru"  # "lru", "oldest"


@dataclass
class EmbeddingRecord:
    """Record of an embedding with metadata."""
    object_id: str
    embedding: np.ndarray
    primary_class: str = ""
    camera_id: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    synced: bool = False  # Whether persisted to DB


class EmbeddingStore:
    """Persistent embedding store with FAISS indexing.

    Combines FAISSGallery for fast search with SQLite for persistence.
    Supports incremental sync and background persistence.

    Usage:
        store = EmbeddingStore(EmbeddingStoreConfig(
            db_path="data/percept.db",
            index_path="data/embeddings.index",
        ))
        store.initialize()

        # Add embedding
        store.add("obj-123", embedding, camera_id="cam1")

        # Search
        matches = store.search(query_embedding, k=5)

        # Sync to disk
        store.sync()
    """

    def __init__(self, config: Optional[EmbeddingStoreConfig] = None):
        """Initialize embedding store.

        Args:
            config: Configuration options
        """
        self.config = config or EmbeddingStoreConfig()

        # Gallery for fast search
        gallery_config = GalleryConfig(
            embedding_dimension=self.config.embedding_dimension,
            index_type=self.config.index_type,
            hnsw_m=self.config.hnsw_m,
            nlist=self.config.nlist,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        self._gallery = FAISSGallery(gallery_config)

        # Database connection
        self._db: Optional[PerceptDatabase] = None

        # Tracking
        self._records: Dict[str, EmbeddingRecord] = {}
        self._pending_sync: List[str] = []  # Object IDs needing DB sync
        self._operations = 0
        self._last_sync = time.time()

        # LRU cache tracking
        self._access_times: Dict[str, float] = {}

    def initialize(self, load_from_db: bool = True) -> int:
        """Initialize store and optionally load from database.

        Args:
            load_from_db: Load existing embeddings from database

        Returns:
            Number of embeddings loaded
        """
        # Create paths
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.index_path).parent.mkdir(parents=True, exist_ok=True)

        # Open database
        self._db = PerceptDatabase(self.config.db_path)
        self._db.initialize()

        loaded = 0

        # Try to load existing FAISS index first
        if Path(self.config.index_path).exists():
            try:
                loaded = self._load_index()
            except Exception:
                # Fall back to loading from DB
                pass

        # Load from database if no index or requested
        if load_from_db and loaded == 0:
            loaded = self._load_from_database()

        return loaded

    def _load_from_database(self) -> int:
        """Load embeddings from database.

        Returns:
            Number of embeddings loaded
        """
        if self._db is None:
            return 0

        embeddings = self._db.get_all_embeddings()

        for object_id, embedding in embeddings:
            # Get object details
            obj = self._db.get_object(object_id)
            if obj is None:
                continue

            record = EmbeddingRecord(
                object_id=object_id,
                embedding=embedding,
                primary_class=obj.primary_class,
                camera_id=obj.camera_id or "",
                timestamp=obj.last_seen.timestamp(),
                synced=True,
            )

            self._records[object_id] = record
            self._gallery.add(
                object_id=object_id,
                embedding=embedding,
                metadata={
                    "primary_class": obj.primary_class,
                    "camera_id": obj.camera_id,
                },
                timestamp=record.timestamp,
            )

        return len(embeddings)

    def add(
        self,
        object_id: str,
        embedding: np.ndarray,
        primary_class: str = "",
        camera_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update embedding for an object.

        Args:
            object_id: Unique object identifier
            embedding: Embedding vector
            primary_class: Object class
            camera_id: Source camera
            metadata: Additional metadata
        """
        timestamp = time.time()

        # Create record
        record = EmbeddingRecord(
            object_id=object_id,
            embedding=embedding.copy(),
            primary_class=primary_class,
            camera_id=camera_id,
            timestamp=timestamp,
            metadata=metadata or {},
            synced=False,
        )

        # Update tracking
        self._records[object_id] = record
        self._access_times[object_id] = timestamp
        self._pending_sync.append(object_id)

        # Add to gallery
        self._gallery.add(
            object_id=object_id,
            embedding=embedding,
            metadata={
                "primary_class": primary_class,
                "camera_id": camera_id,
                **(metadata or {}),
            },
            timestamp=timestamp,
        )

        # Persist immediately if configured
        if self.config.persist_on_add:
            self._sync_object(object_id)

        # Auto-sync check
        self._operations += 1
        if self.config.auto_sync and self._operations >= self.config.sync_interval:
            self.sync()

        # Cache eviction
        self._check_cache_size()

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        camera_id: Optional[str] = None,
        class_filter: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings.

        Args:
            query: Query embedding vector
            k: Number of results
            camera_id: Optional camera filter
            class_filter: Optional class filter

        Returns:
            List of (object_id, distance, metadata) tuples
        """
        # Get more results if filtering
        search_k = k * 3 if (camera_id or class_filter) else k

        results = self._gallery.search(query, k=search_k)

        # Filter and format results
        filtered = []
        for obj_id, distance, idx in results:
            if obj_id not in self._records:
                continue

            record = self._records[obj_id]

            # Apply filters
            if camera_id and record.camera_id != camera_id:
                continue
            if class_filter and record.primary_class != class_filter:
                continue

            # Update access time
            self._access_times[obj_id] = time.time()

            filtered.append((
                obj_id,
                distance,
                {
                    "primary_class": record.primary_class,
                    "camera_id": record.camera_id,
                    "timestamp": record.timestamp,
                    **record.metadata,
                }
            ))

            if len(filtered) >= k:
                break

        return filtered

    def search_by_threshold(
        self,
        query: np.ndarray,
        threshold: float,
        max_results: int = 100,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for embeddings within distance threshold.

        Args:
            query: Query embedding
            threshold: Maximum L2 distance
            max_results: Maximum results to return

        Returns:
            List of matching (object_id, distance, metadata) tuples
        """
        results = self._gallery.search_by_similarity(query, threshold, max_results)

        formatted = []
        for obj_id, distance, idx in results:
            if obj_id in self._records:
                record = self._records[obj_id]
                formatted.append((
                    obj_id,
                    distance,
                    {
                        "primary_class": record.primary_class,
                        "camera_id": record.camera_id,
                        "timestamp": record.timestamp,
                    }
                ))

        return formatted

    def get(self, object_id: str) -> Optional[np.ndarray]:
        """Get embedding for an object.

        Args:
            object_id: Object identifier

        Returns:
            Embedding vector or None
        """
        if object_id in self._records:
            self._access_times[object_id] = time.time()
            return self._records[object_id].embedding.copy()
        return None

    def get_record(self, object_id: str) -> Optional[EmbeddingRecord]:
        """Get full embedding record.

        Args:
            object_id: Object identifier

        Returns:
            EmbeddingRecord or None
        """
        return self._records.get(object_id)

    def remove(self, object_id: str) -> bool:
        """Remove embedding for an object.

        Args:
            object_id: Object to remove

        Returns:
            True if removed
        """
        if object_id not in self._records:
            return False

        del self._records[object_id]
        self._access_times.pop(object_id, None)

        if object_id in self._pending_sync:
            self._pending_sync.remove(object_id)

        self._gallery.remove_object(object_id)

        return True

    def sync(self) -> int:
        """Sync pending changes to database.

        Returns:
            Number of objects synced
        """
        if self._db is None or not self._pending_sync:
            return 0

        synced = 0
        to_sync = list(set(self._pending_sync))

        for object_id in to_sync:
            if self._sync_object(object_id):
                synced += 1

        self._pending_sync.clear()
        self._operations = 0
        self._last_sync = time.time()

        return synced

    def _sync_object(self, object_id: str) -> bool:
        """Sync single object to database.

        Args:
            object_id: Object to sync

        Returns:
            True if synced successfully
        """
        if self._db is None:
            return False

        record = self._records.get(object_id)
        if record is None:
            return False

        # Check if object exists in DB
        existing = self._db.get_object(object_id)
        if existing is not None:
            # Update embedding in existing object
            existing.reid_embedding = record.embedding
            self._db.save_object(existing)

        record.synced = True
        return True

    def save_index(self, path: Optional[str] = None) -> None:
        """Save FAISS index to disk.

        Args:
            path: Optional override path
        """
        save_path = path or self.config.index_path

        # Save gallery
        self._gallery.save(save_path)

        # Save records metadata
        metadata_path = save_path + ".meta"
        records_data = {}
        for obj_id, record in self._records.items():
            records_data[obj_id] = {
                "primary_class": record.primary_class,
                "camera_id": record.camera_id,
                "timestamp": record.timestamp,
                "metadata": record.metadata,
                "synced": record.synced,
            }

        with open(metadata_path, 'w') as f:
            json.dump(records_data, f)

    def _load_index(self) -> int:
        """Load FAISS index from disk.

        Returns:
            Number of embeddings loaded
        """
        if not Path(self.config.index_path).exists():
            return 0

        # Load gallery
        self._gallery = FAISSGallery.load(self.config.index_path)

        # Load records metadata
        metadata_path = self.config.index_path + ".meta"
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                records_data = json.load(f)

            for obj_id, data in records_data.items():
                embedding = self._gallery.get_embedding(obj_id)
                if embedding is not None:
                    self._records[obj_id] = EmbeddingRecord(
                        object_id=obj_id,
                        embedding=embedding,
                        primary_class=data.get("primary_class", ""),
                        camera_id=data.get("camera_id", ""),
                        timestamp=data.get("timestamp", 0.0),
                        metadata=data.get("metadata", {}),
                        synced=data.get("synced", True),
                    )
                    self._access_times[obj_id] = time.time()

        return len(self._records)

    def _check_cache_size(self) -> None:
        """Evict entries if cache is too large."""
        if len(self._records) <= self.config.cache_size:
            return

        # Calculate how many to evict (10% of cache)
        evict_count = max(1, self.config.cache_size // 10)

        if self.config.eviction_strategy == "lru":
            # Evict least recently used
            sorted_ids = sorted(
                self._access_times.keys(),
                key=lambda k: self._access_times[k]
            )
        else:  # oldest
            # Evict oldest by timestamp
            sorted_ids = sorted(
                self._records.keys(),
                key=lambda k: self._records[k].timestamp
            )

        # Evict only synced entries
        evicted = 0
        for obj_id in sorted_ids:
            if evicted >= evict_count:
                break

            record = self._records.get(obj_id)
            if record and record.synced:
                # Don't fully remove - just remove from memory
                # Gallery still has the embedding for search
                del self._records[obj_id]
                self._access_times.pop(obj_id, None)
                evicted += 1

    def rebuild_index(self) -> None:
        """Rebuild FAISS index from current embeddings."""
        self._gallery.rebuild_index()

    def size(self) -> int:
        """Get number of embeddings in store."""
        return len(self._records)

    def gallery_size(self) -> int:
        """Get number of embeddings in FAISS gallery."""
        return self._gallery.size()

    def pending_count(self) -> int:
        """Get number of unsynced embeddings."""
        return len(set(self._pending_sync))

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics.

        Returns:
            Dictionary of statistics
        """
        synced = sum(1 for r in self._records.values() if r.synced)
        unsynced = len(self._records) - synced

        return {
            "total_embeddings": len(self._records),
            "gallery_size": self._gallery.size(),
            "synced": synced,
            "unsynced": unsynced,
            "pending_sync": len(set(self._pending_sync)),
            "operations_since_sync": self._operations,
            "last_sync": self._last_sync,
        }

    def close(self) -> None:
        """Close store and sync remaining changes."""
        self.sync()
        self.save_index()
        if self._db is not None:
            self._db.close()
            self._db = None


class CameraAwareEmbeddingStore(EmbeddingStore):
    """Embedding store with camera-aware matching.

    Supports different thresholds for same-camera vs cross-camera matching.
    """

    def __init__(
        self,
        config: Optional[EmbeddingStoreConfig] = None,
        same_camera_threshold: float = 0.3,
        cross_camera_threshold: float = 0.25,
    ):
        """Initialize camera-aware store.

        Args:
            config: Configuration options
            same_camera_threshold: Distance threshold for same camera
            cross_camera_threshold: Distance threshold for cross camera
        """
        super().__init__(config)
        self.same_camera_threshold = same_camera_threshold
        self.cross_camera_threshold = cross_camera_threshold

        # Per-camera indices for efficiency
        self._camera_object_ids: Dict[str, set] = {}

    def add(
        self,
        object_id: str,
        embedding: np.ndarray,
        primary_class: str = "",
        camera_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add embedding with camera tracking."""
        super().add(object_id, embedding, primary_class, camera_id, metadata)

        # Track camera membership
        if camera_id:
            if camera_id not in self._camera_object_ids:
                self._camera_object_ids[camera_id] = set()
            self._camera_object_ids[camera_id].add(object_id)

    def find_match(
        self,
        query: np.ndarray,
        query_camera: str,
    ) -> Optional[Tuple[str, float, str]]:
        """Find best matching object with camera-aware thresholds.

        Args:
            query: Query embedding
            query_camera: Camera the query is from

        Returns:
            Tuple of (object_id, distance, match_type) or None
        """
        results = self.search(query, k=10)

        for obj_id, distance, metadata in results:
            obj_camera = metadata.get("camera_id", "")

            if obj_camera == query_camera:
                # Same camera - use looser threshold
                if distance < self.same_camera_threshold:
                    return (obj_id, distance, "same_camera")
            else:
                # Cross camera - use tighter threshold
                if distance < self.cross_camera_threshold:
                    return (obj_id, distance, "cross_camera")

        return None

    def search_same_camera(
        self,
        query: np.ndarray,
        camera_id: str,
        k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search only within same camera.

        Args:
            query: Query embedding
            camera_id: Camera to search
            k: Number of results

        Returns:
            List of matches from same camera
        """
        return self.search(query, k=k, camera_id=camera_id)

    def search_cross_camera(
        self,
        query: np.ndarray,
        exclude_camera: str,
        k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search across all cameras except one.

        Args:
            query: Query embedding
            exclude_camera: Camera to exclude
            k: Number of results

        Returns:
            List of matches from other cameras
        """
        all_results = self.search(query, k=k * 2)

        filtered = [
            (obj_id, dist, meta)
            for obj_id, dist, meta in all_results
            if meta.get("camera_id") != exclude_camera
        ]

        return filtered[:k]

    def get_camera_objects(self, camera_id: str) -> List[str]:
        """Get all object IDs for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            List of object IDs
        """
        return list(self._camera_object_ids.get(camera_id, set()))

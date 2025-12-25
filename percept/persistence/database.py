"""SQLite database operations for PERCEPT.

Provides persistent storage for ObjectSchemas, trajectory data,
and human review queue.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from percept.core.schema import ObjectSchema, ClassificationStatus


class PerceptDatabase:
    """SQLite database for PERCEPT object persistence.

    Stores ObjectSchemas with their embeddings, trajectory history,
    and supports the human review queue workflow.

    Usage:
        db = PerceptDatabase("data/percept.db")
        db.initialize()

        # Store object
        db.save_object(object_schema)

        # Query
        obj = db.get_object(object_id)
        all_persons = db.query_by_class("person")
    """

    def __init__(self, db_path: str | Path):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._connection: Optional[sqlite3.Connection] = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row
        return self._connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database transactions."""
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def initialize(self) -> None:
        """Create database schema if it doesn't exist."""
        with self.transaction() as cursor:
            # Main objects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    id TEXT PRIMARY KEY,
                    reid_embedding BLOB,

                    -- Spatial
                    position_x REAL,
                    position_y REAL,
                    position_z REAL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    width_m REAL,
                    height_m REAL,
                    depth_m REAL,
                    distance_m REAL,

                    -- Classification
                    primary_class TEXT,
                    subclass TEXT,
                    confidence REAL,
                    classification_status TEXT,

                    -- Attributes (JSON)
                    attributes_json TEXT,

                    -- Tracking
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    camera_id TEXT,

                    -- Processing
                    pipelines_completed TEXT,
                    processing_time_ms REAL,
                    source_frame_ids TEXT,

                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Trajectory points
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trajectory_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id TEXT REFERENCES objects(id) ON DELETE CASCADE,
                    x REAL,
                    y REAL,
                    z REAL,
                    timestamp TIMESTAMP,
                    camera_id TEXT
                )
            """)

            # Human review queue
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id TEXT REFERENCES objects(id) ON DELETE CASCADE,
                    schema_json TEXT,
                    image_path TEXT,
                    reason TEXT,
                    provisional_class TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TIMESTAMP,
                    reviewer TEXT,
                    human_class TEXT,
                    human_attributes_json TEXT,
                    status TEXT DEFAULT 'pending'
                )
            """)

            # Indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_objects_class ON objects(primary_class)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_objects_camera ON objects(camera_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_objects_last_seen ON objects(last_seen)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_objects_status ON objects(classification_status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_trajectory_object ON trajectory_points(object_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status)"
            )

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> PerceptDatabase:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # Object CRUD operations

    def save_object(self, obj: ObjectSchema) -> None:
        """Save or update an ObjectSchema.

        Args:
            obj: ObjectSchema to save
        """
        embedding_blob = None
        if obj.reid_embedding is not None:
            embedding_blob = obj.reid_embedding.astype(np.float32).tobytes()

        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO objects (
                    id, reid_embedding,
                    position_x, position_y, position_z,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    width_m, height_m, depth_m, distance_m,
                    primary_class, subclass, confidence, classification_status,
                    attributes_json,
                    first_seen, last_seen, camera_id,
                    pipelines_completed, processing_time_ms, source_frame_ids,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obj.id,
                    embedding_blob,
                    obj.position_3d[0],
                    obj.position_3d[1],
                    obj.position_3d[2],
                    obj.bounding_box_2d[0],
                    obj.bounding_box_2d[1],
                    obj.bounding_box_2d[2],
                    obj.bounding_box_2d[3],
                    obj.dimensions[0],
                    obj.dimensions[1],
                    obj.dimensions[2],
                    obj.distance_from_camera,
                    obj.primary_class,
                    obj.subclass,
                    obj.confidence,
                    obj.classification_status.value,
                    json.dumps(obj.attributes),
                    obj.first_seen.isoformat(),
                    obj.last_seen.isoformat(),
                    obj.camera_id,
                    ",".join(obj.pipelines_completed),
                    obj.processing_time_ms,
                    ",".join(str(f) for f in obj.source_frame_ids),
                    datetime.now().isoformat(),
                ),
            )

            # Save trajectory points
            for x, y, z, ts in obj.trajectory:
                cursor.execute(
                    """
                    INSERT INTO trajectory_points (object_id, x, y, z, timestamp, camera_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (obj.id, x, y, z, ts.isoformat(), obj.camera_id),
                )

    def get_object(self, object_id: str) -> Optional[ObjectSchema]:
        """Retrieve an ObjectSchema by ID.

        Args:
            object_id: Object UUID

        Returns:
            ObjectSchema if found, None otherwise
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM objects WHERE id = ?", (object_id,))
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            return None

        return self._row_to_schema(row)

    def delete_object(self, object_id: str) -> bool:
        """Delete an object and its trajectory.

        Args:
            object_id: Object UUID

        Returns:
            True if deleted, False if not found
        """
        with self.transaction() as cursor:
            cursor.execute("DELETE FROM objects WHERE id = ?", (object_id,))
            return cursor.rowcount > 0

    def get_all_objects(
        self,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[ObjectSchema]:
        """Get all objects with pagination.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of ObjectSchemas
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM objects ORDER BY last_seen DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()
        cursor.close()

        return [self._row_to_schema(row) for row in rows]

    def query_by_class(
        self,
        primary_class: str,
        limit: int = 100,
    ) -> List[ObjectSchema]:
        """Query objects by primary class.

        Args:
            primary_class: Class to filter by
            limit: Maximum number of results

        Returns:
            List of matching ObjectSchemas
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM objects WHERE primary_class = ? ORDER BY last_seen DESC LIMIT ?",
            (primary_class, limit),
        )
        rows = cursor.fetchall()
        cursor.close()

        return [self._row_to_schema(row) for row in rows]

    def query_by_camera(
        self,
        camera_id: str,
        limit: int = 100,
    ) -> List[ObjectSchema]:
        """Query objects by camera ID.

        Args:
            camera_id: Camera to filter by
            limit: Maximum number of results

        Returns:
            List of matching ObjectSchemas
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM objects WHERE camera_id = ? ORDER BY last_seen DESC LIMIT ?",
            (camera_id, limit),
        )
        rows = cursor.fetchall()
        cursor.close()

        return [self._row_to_schema(row) for row in rows]

    def query_recent(
        self,
        seconds: float = 60.0,
        camera_id: Optional[str] = None,
    ) -> List[ObjectSchema]:
        """Query objects seen recently.

        Args:
            seconds: Time window in seconds
            camera_id: Optional camera filter

        Returns:
            List of recently seen ObjectSchemas
        """
        cutoff = datetime.now().timestamp() - seconds
        cutoff_str = datetime.fromtimestamp(cutoff).isoformat()

        cursor = self.connection.cursor()
        if camera_id:
            cursor.execute(
                "SELECT * FROM objects WHERE last_seen > ? AND camera_id = ? ORDER BY last_seen DESC",
                (cutoff_str, camera_id),
            )
        else:
            cursor.execute(
                "SELECT * FROM objects WHERE last_seen > ? ORDER BY last_seen DESC",
                (cutoff_str,),
            )
        rows = cursor.fetchall()
        cursor.close()

        return [self._row_to_schema(row) for row in rows]

    def query_needs_review(self, limit: int = 50) -> List[ObjectSchema]:
        """Query objects that need human review.

        Args:
            limit: Maximum number of results

        Returns:
            List of ObjectSchemas needing review
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM objects WHERE classification_status = ? ORDER BY created_at DESC LIMIT ?",
            (ClassificationStatus.NEEDS_REVIEW.value, limit),
        )
        rows = cursor.fetchall()
        cursor.close()

        return [self._row_to_schema(row) for row in rows]

    def count_objects(self, primary_class: Optional[str] = None) -> int:
        """Count objects in database.

        Args:
            primary_class: Optional class filter

        Returns:
            Number of objects
        """
        cursor = self.connection.cursor()
        if primary_class:
            cursor.execute(
                "SELECT COUNT(*) FROM objects WHERE primary_class = ?",
                (primary_class,),
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM objects")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def get_trajectory(
        self,
        object_id: str,
    ) -> List[Tuple[float, float, float, datetime]]:
        """Get trajectory history for an object.

        Args:
            object_id: Object UUID

        Returns:
            List of (x, y, z, timestamp) tuples
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT x, y, z, timestamp FROM trajectory_points WHERE object_id = ? ORDER BY timestamp",
            (object_id,),
        )
        rows = cursor.fetchall()
        cursor.close()

        return [
            (row["x"], row["y"], row["z"], datetime.fromisoformat(row["timestamp"]))
            for row in rows
        ]

    # Review queue operations

    def add_to_review_queue(
        self,
        obj: ObjectSchema,
        image_path: str,
        reason: str,
    ) -> int:
        """Add an object to the human review queue.

        Args:
            obj: ObjectSchema to review
            image_path: Path to crop image
            reason: Reason for review

        Returns:
            Review queue item ID
        """
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO review_queue (
                    object_id, schema_json, image_path, reason,
                    provisional_class, confidence, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    obj.id,
                    obj.to_json(),
                    image_path,
                    reason,
                    obj.primary_class,
                    obj.confidence,
                ),
            )
            return cursor.lastrowid

    def get_pending_reviews(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending review items.

        Args:
            limit: Maximum number of results

        Returns:
            List of review item dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT id, object_id, schema_json, image_path, reason,
                   provisional_class, confidence, created_at
            FROM review_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        cursor.close()

        return [dict(row) for row in rows]

    def submit_review(
        self,
        review_id: int,
        human_class: str,
        reviewer: str,
        human_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Submit a human review.

        Args:
            review_id: Review queue item ID
            human_class: Human-assigned class
            reviewer: Reviewer identifier
            human_attributes: Optional additional attributes
        """
        with self.transaction() as cursor:
            # Update review queue
            cursor.execute(
                """
                UPDATE review_queue
                SET human_class = ?, reviewer = ?, human_attributes_json = ?,
                    reviewed_at = ?, status = 'reviewed'
                WHERE id = ?
                """,
                (
                    human_class,
                    reviewer,
                    json.dumps(human_attributes) if human_attributes else None,
                    datetime.now().isoformat(),
                    review_id,
                ),
            )

            # Get object_id for this review
            cursor.execute(
                "SELECT object_id FROM review_queue WHERE id = ?",
                (review_id,),
            )
            row = cursor.fetchone()
            if row:
                # Update the object with confirmed classification
                cursor.execute(
                    """
                    UPDATE objects
                    SET primary_class = ?,
                        classification_status = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        human_class,
                        ClassificationStatus.CONFIRMED.value,
                        datetime.now().isoformat(),
                        row["object_id"],
                    ),
                )

    def skip_review(self, review_id: int) -> None:
        """Skip a review item.

        Args:
            review_id: Review queue item ID
        """
        with self.transaction() as cursor:
            cursor.execute(
                "UPDATE review_queue SET status = 'skipped' WHERE id = ?",
                (review_id,),
            )

    def get_review_stats(self) -> Dict[str, int]:
        """Get review queue statistics.

        Returns:
            Dictionary with counts by status
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT status, COUNT(*) as count FROM review_queue GROUP BY status"
        )
        rows = cursor.fetchall()
        cursor.close()

        return {row["status"]: row["count"] for row in rows}

    # Helper methods

    def _row_to_schema(self, row: sqlite3.Row) -> ObjectSchema:
        """Convert database row to ObjectSchema."""
        # Reconstruct embedding
        embedding = None
        if row["reid_embedding"]:
            embedding = np.frombuffer(row["reid_embedding"], dtype=np.float32)

        # Parse lists
        pipelines = []
        if row["pipelines_completed"]:
            pipelines = row["pipelines_completed"].split(",")

        frame_ids = []
        if row["source_frame_ids"]:
            frame_ids = [int(f) for f in row["source_frame_ids"].split(",") if f]

        # Get trajectory
        trajectory = self.get_trajectory(row["id"])

        return ObjectSchema(
            id=row["id"],
            reid_embedding=embedding,
            position_3d=(row["position_x"], row["position_y"], row["position_z"]),
            bounding_box_2d=(
                row["bbox_x1"],
                row["bbox_y1"],
                row["bbox_x2"],
                row["bbox_y2"],
            ),
            dimensions=(row["width_m"], row["height_m"], row["depth_m"]),
            distance_from_camera=row["distance_m"],
            primary_class=row["primary_class"],
            subclass=row["subclass"],
            confidence=row["confidence"],
            classification_status=ClassificationStatus(row["classification_status"]),
            attributes=json.loads(row["attributes_json"]) if row["attributes_json"] else {},
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            camera_id=row["camera_id"],
            trajectory=trajectory,
            pipelines_completed=pipelines,
            processing_time_ms=row["processing_time_ms"],
            source_frame_ids=frame_ids,
        )

    def get_embedding(self, object_id: str) -> Optional[np.ndarray]:
        """Get just the embedding for an object (efficient for FAISS sync).

        Args:
            object_id: Object UUID

        Returns:
            Embedding array if found, None otherwise
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT reid_embedding FROM objects WHERE id = ?",
            (object_id,),
        )
        row = cursor.fetchone()
        cursor.close()

        if row is None or row["reid_embedding"] is None:
            return None

        return np.frombuffer(row["reid_embedding"], dtype=np.float32)

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Get all object IDs and embeddings (for FAISS index building).

        Returns:
            List of (object_id, embedding) tuples
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT id, reid_embedding FROM objects WHERE reid_embedding IS NOT NULL"
        )
        rows = cursor.fetchall()
        cursor.close()

        return [
            (row["id"], np.frombuffer(row["reid_embedding"], dtype=np.float32))
            for row in rows
        ]

    def update_classification(
        self,
        object_id: str,
        primary_class: str,
        confidence: float,
        status: ClassificationStatus,
        subclass: Optional[str] = None,
    ) -> bool:
        """Update object classification.

        Args:
            object_id: Object UUID
            primary_class: New class
            confidence: New confidence
            status: New classification status
            subclass: Optional subclass

        Returns:
            True if updated, False if not found
        """
        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE objects
                SET primary_class = ?, subclass = ?, confidence = ?,
                    classification_status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    primary_class,
                    subclass,
                    confidence,
                    status.value,
                    datetime.now().isoformat(),
                    object_id,
                ),
            )
            return cursor.rowcount > 0

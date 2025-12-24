# PERCEPT: Pipeline for Entity Recognition, Classification, Extraction, and Persistence Tracking

**Framework Specification Document**
**Version:** 1.0 Draft
**Date:** December 23, 2025
**Platform:** Raspberry Pi 5 + Hailo-8 + Intel RealSense

---

## 1. Executive Summary

PERCEPT is a modular, pipeline-based vision processing framework designed for mobile robotic platforms. It provides:

- **Flexible pipelines** with hot-swappable algorithm modules
- **Automatic data adaptation** between pipeline stages
- **3D-aware segmentation** using RGB-D cameras
- **Zero-shot object detection** for unknown objects
- **ReID-based tracking** to avoid redundant processing
- **Progressive object schemas** that accumulate knowledge
- **Multi-camera support** with cross-camera ReID
- **Adaptive processing** based on scene complexity
- **Human review queue** for uncertain classifications

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PERCEPT FRAMEWORK                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Camera 1   │    │   Camera 2   │    │   Camera N   │                   │
│  │  (RealSense) │    │  (RealSense) │    │  (RealSense) │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         └───────────────────┼───────────────────┘                            │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     FRAME ACQUISITION LAYER                          │    │
│  │  • Multi-camera synchronization                                      │    │
│  │  • RGB + Depth + Point Cloud generation                             │    │
│  │  • Intrinsics/extrinsics management                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     SEGMENTATION LAYER                               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │  FastSAM    │  │   Depth     │  │  Point Cloud │                  │    │
│  │  │  (Hailo-8)  │  │ Discontinuity│  │  Clustering  │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  │         │                │                │                          │    │
│  │         └────────────────┼────────────────┘                          │    │
│  │                          ▼                                           │    │
│  │              Segment Fusion & Mask Generation                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     REID & TRACKING LAYER                            │    │
│  │  • Generate ReID embeddings (RepVGG for persons, histograms others) │    │
│  │  • Match against known object gallery                                │    │
│  │  • ByteTrack for frame-to-frame association                         │    │
│  │  • Cross-camera matching                                             │    │
│  │  • Decision: NEW object or EXISTING object?                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                │
│           ┌─────────────────┴─────────────────┐                             │
│           ▼                                   ▼                             │
│    [NEW OBJECT]                        [EXISTING OBJECT]                    │
│           │                                   │                             │
│           ▼                                   ▼                             │
│  ┌─────────────────────┐            ┌─────────────────────┐                 │
│  │  PIPELINE ROUTER    │            │  Schema Lookup      │                 │
│  │  Route to appropriate│            │  Return existing    │                 │
│  │  classification      │            │  object info        │                 │
│  │  pipeline            │            │  Update position    │                 │
│  └──────────┬──────────┘            └─────────────────────┘                 │
│             │                                                                │
│             ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     CLASSIFICATION PIPELINES                         │    │
│  │                                                                      │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │   Person     │  │   Vehicle    │  │   Generic    │               │    │
│  │  │   Pipeline   │  │   Pipeline   │  │   Object     │               │    │
│  │  │              │  │              │  │   Pipeline   │               │    │
│  │  │ • Pose Est.  │  │ • Type       │  │              │               │    │
│  │  │ • Clothing   │  │ • Color      │  │ • ImageNet   │               │    │
│  │  │ • Gender     │  │ • License    │  │ • LLM Desc.  │               │    │
│  │  │ • Face Det.  │  │ • Make/Model │  │ • Shape      │               │    │
│  │  │   └─Face ID  │  │              │  │ • Color      │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     NORMALIZATION LAYER                              │    │
│  │  • Size standardization (resize to canonical dimensions)            │    │
│  │  • Light intensity correction (histogram equalization)              │    │
│  │  • Color correction (white balance, color constancy)                │    │
│  │  • Orientation normalization (upright objects)                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     SCHEMA BUILDER                                   │    │
│  │  • Aggregate pipeline outputs into ObjectSchema                     │    │
│  │  • Confidence assessment                                            │    │
│  │  • Human review queue (low confidence objects)                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     PERSISTENCE LAYER                                │    │
│  │  • SQLite database for object schemas                               │    │
│  │  • FAISS index for ReID embeddings                                  │    │
│  │  • Human review queue                                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Concepts

### 3.1 Object Schema

The central data structure that accumulates knowledge about detected objects:

```python
@dataclass
class ObjectSchema:
    # Identity
    id: str                          # Unique identifier (UUID)
    reid_embedding: np.ndarray       # 512-dim feature vector

    # Spatial (camera-relative, SLAM-ready)
    position_3d: Tuple[float, float, float]  # (x, y, z) meters
    bounding_box_2d: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    dimensions: Tuple[float, float, float]  # (width, height, depth) meters
    distance_from_camera: float      # meters

    # Classification
    primary_class: str               # "person", "vehicle", "furniture", etc.
    subclass: Optional[str]          # "adult", "car", "chair"
    confidence: float                # 0.0 - 1.0
    classification_status: ClassificationStatus  # CONFIRMED, PROVISIONAL, NEEDS_REVIEW

    # Type-specific attributes (populated by specialized pipelines)
    attributes: Dict[str, Any]       # Flexible attribute storage
    # Examples:
    # Person: {"gender": "male", "clothing": {...}, "pose": [...], "face": {...}}
    # Vehicle: {"type": "sedan", "color": "blue", "license": "ABC123"}

    # Tracking
    first_seen: datetime
    last_seen: datetime
    camera_id: str
    trajectory: List[Tuple[float, float, float, datetime]]  # Position history

    # Processing metadata
    pipelines_completed: List[str]   # Which pipelines have processed this
    processing_time_ms: float
    source_frame_ids: List[int]
```

### 3.2 Pipeline Module Interface

Every algorithm module implements this interface for hot-swappability:

```python
class PipelineModule(ABC):
    """Base class for all pipeline modules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique module identifier."""
        pass

    @property
    @abstractmethod
    def input_spec(self) -> DataSpec:
        """What this module expects as input."""
        pass

    @property
    @abstractmethod
    def output_spec(self) -> DataSpec:
        """What this module produces as output."""
        pass

    @abstractmethod
    def process(self, data: PipelineData) -> PipelineData:
        """Process input and return output."""
        pass

    def can_process(self, data: PipelineData) -> bool:
        """Check if this module can process the given data."""
        return data.matches_spec(self.input_spec)

@dataclass
class DataSpec:
    """Specification for data flowing through pipelines."""
    data_type: str           # "image", "mask", "embedding", "schema", etc.
    shape: Optional[Tuple]   # Expected dimensions
    dtype: Optional[str]     # "uint8", "float32", etc.
    color_space: Optional[str]  # "BGR", "RGB", "GRAY"
    required_fields: List[str]  # Required keys if dict-like
```

### 3.3 Data Adapter

Automatically converts data between incompatible modules:

```python
class DataAdapter:
    """Automatically adapts data between pipeline modules."""

    adapters: Dict[Tuple[DataSpec, DataSpec], Callable]

    def adapt(self, data: PipelineData,
              from_spec: DataSpec,
              to_spec: DataSpec) -> PipelineData:
        """Convert data from one spec to another."""
        # Examples:
        # - Resize image (640x480 → 256x128)
        # - Convert color space (BGR → RGB)
        # - Normalize values (0-255 → 0-1)
        # - Extract region from full image using mask
```

### 3.4 Mask-Based Mutual Exclusion

When a pipeline "claims" an object region:

```python
class SceneMaskManager:
    """Manages claimed regions to prevent duplicate processing."""

    def __init__(self):
        self.claimed_mask: np.ndarray  # Binary mask of claimed regions
        self.claims: Dict[str, np.ndarray]  # object_id → mask

    def claim_region(self, object_id: str, mask: np.ndarray) -> bool:
        """Claim a region for an object. Returns False if already claimed."""
        overlap = np.logical_and(self.claimed_mask, mask)
        if np.any(overlap):
            return False  # Region already claimed
        self.claimed_mask = np.logical_or(self.claimed_mask, mask)
        self.claims[object_id] = mask
        return True

    def get_unclaimed_mask(self) -> np.ndarray:
        """Return mask of regions not yet claimed."""
        return np.logical_not(self.claimed_mask)

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """Return image with claimed regions masked out."""
        return image * self.get_unclaimed_mask()[..., np.newaxis]
```

---

## 4. Segmentation Strategy

### 4.1 Multi-Method Fusion

PERCEPT uses three complementary segmentation approaches:

| Method | Runs On | Speed | Strength | Weakness |
|--------|---------|-------|----------|----------|
| **FastSAM** | Hailo-8 | 20 FPS | Class-agnostic, accurate boundaries | May miss small objects |
| **Depth Discontinuity** | CPU | 50 FPS | Very fast, no AI needed | Fails on same-depth objects |
| **Point Cloud Clustering** | CPU | 30 FPS | True 3D separation | Requires good depth data |

### 4.2 Fusion Algorithm

```python
def segment_scene(rgb: np.ndarray, depth: np.ndarray) -> List[ObjectMask]:
    """Fuse multiple segmentation methods."""

    # Method 1: FastSAM (Hailo-8)
    fastsam_masks = fastsam.segment(rgb)  # ~50ms

    # Method 2: Depth Discontinuity (CPU)
    depth_edges = detect_depth_discontinuities(depth)  # ~10ms
    depth_masks = connected_components(depth, depth_edges)  # ~5ms

    # Method 3: Point Cloud Clustering (CPU)
    point_cloud = depth_to_pointcloud(depth, intrinsics)
    planes = ransac_detect_planes(point_cloud)  # Remove floor/walls
    clusters = euclidean_clustering(point_cloud - planes)  # ~20ms
    cluster_masks = project_clusters_to_2d(clusters, intrinsics)

    # Fusion: Use FastSAM as primary, refine with depth
    final_masks = []
    for fsam_mask in fastsam_masks:
        # Refine boundaries using depth edges
        refined = refine_mask_with_depth(fsam_mask, depth_edges)
        # Validate with point cloud (reject if not spatially coherent)
        if validate_3d_coherence(refined, point_cloud):
            final_masks.append(refined)

    # Add depth-only detections missed by FastSAM
    for depth_mask in depth_masks:
        if not overlaps_existing(depth_mask, final_masks):
            final_masks.append(depth_mask)

    return final_masks
```

### 4.3 Zero-Shot Object Handling

For objects without trained models:

1. **Segment** using FastSAM (class-agnostic)
2. **Extract 3D properties** from depth (dimensions, position)
3. **Generate description** using Moondream LLM
4. **Create temporary classification** based on shape/size
5. **Queue for human review** if confidence < threshold

---

## 5. ReID & Tracking System

### 5.1 Embedding Strategy

| Object Type | Embedding Method | Dimensions | Runs On |
|-------------|------------------|------------|---------|
| **Person** | RepVGG_A0 | 512-dim | Hailo-8 |
| **Face** | (Future: ArcFace) | 512-dim | Hailo-8 |
| **Vehicle** | Color histogram + shape | 128-dim | CPU |
| **Generic** | Color + depth + shape | 103-dim | CPU |

### 5.2 Matching Algorithm

```python
class ReIDMatcher:
    """Match objects across frames and cameras."""

    def __init__(self):
        self.gallery: Dict[str, ObjectSchema] = {}
        self.faiss_index = faiss.IndexHNSWFlat(512, 16)
        self.threshold_same_camera = 0.3   # Cosine distance
        self.threshold_cross_camera = 0.25  # Tighter for cross-camera

    def match(self, embedding: np.ndarray,
              camera_id: str) -> Optional[str]:
        """Find matching object in gallery."""

        if len(self.gallery) == 0:
            return None

        # Search FAISS index
        distances, indices = self.faiss_index.search(
            embedding.reshape(1, -1).astype('float32'), k=5
        )

        # Find best match within threshold
        for dist, idx in zip(distances[0], indices[0]):
            candidate = list(self.gallery.values())[idx]
            threshold = (self.threshold_same_camera
                        if candidate.camera_id == camera_id
                        else self.threshold_cross_camera)

            if dist < threshold:
                return candidate.id

        return None  # No match found - new object
```

### 5.3 Tracking Integration (ByteTrack)

```python
class PerceptTracker:
    """Frame-to-frame tracking with ReID integration."""

    def __init__(self):
        self.byte_tracker = ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=30,  # frames before track is lost
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        self.reid_matcher = ReIDMatcher()
        self.reid_interval = 3  # Run ReID every N frames
        self.frame_count = 0

    def update(self, detections: List[Detection],
               frame: np.ndarray) -> List[TrackedObject]:
        """Update tracking with new detections."""

        # ByteTrack for frame-to-frame association
        tracked = self.byte_tracker.update(detections)

        # ReID for identity verification (every N frames)
        if self.frame_count % self.reid_interval == 0:
            for obj in tracked:
                embedding = self.extract_embedding(obj, frame)
                match_id = self.reid_matcher.match(embedding, obj.camera_id)

                if match_id:
                    obj.schema_id = match_id  # Link to existing schema
                else:
                    obj.schema_id = self.create_new_schema(obj, embedding)

        self.frame_count += 1
        return tracked
```

---

## 6. Classification Pipelines

### 6.1 Pipeline Router

```python
class PipelineRouter:
    """Routes objects to appropriate classification pipelines."""

    pipelines = {
        "person": PersonPipeline(),
        "vehicle": VehiclePipeline(),
        "face": FacePipeline(),
        "generic": GenericPipeline(),
    }

    def route(self, detection: Detection,
              mask: np.ndarray) -> str:
        """Determine which pipeline to use."""

        # Use initial detection class if available
        if detection.class_name == "person":
            return "person"
        elif detection.class_name in ["car", "truck", "bus", "motorcycle"]:
            return "vehicle"
        else:
            return "generic"
```

### 6.2 Person Pipeline

```python
class PersonPipeline(Pipeline):
    """Extract detailed information about detected persons."""

    modules = [
        PoseEstimationModule(),      # YOLOv8s-pose → 17 keypoints
        ClothingAnalysisModule(),    # Color/type of clothing
        GenderEstimationModule(),    # Gender classification
        FaceDetectionModule(),       # SCRFD → face bbox + landmarks
        # Face triggers sub-pipeline
    ]

    def process(self, obj: ObjectSchema, crop: np.ndarray):
        """Process person through all modules."""

        # Pose estimation
        keypoints = self.pose_module.process(crop)
        obj.attributes["pose"] = keypoints
        obj.attributes["posture"] = classify_posture(keypoints)  # standing/sitting/lying

        # Clothing analysis
        clothing = self.clothing_module.process(crop, keypoints)
        obj.attributes["clothing"] = {
            "upper": clothing.upper,  # "blue shirt"
            "lower": clothing.lower,  # "dark jeans"
        }

        # Face detection → triggers face sub-pipeline
        face_detection = self.face_module.process(crop)
        if face_detection:
            face_crop = extract_face(crop, face_detection.bbox)
            face_schema = self.face_pipeline.process(face_crop)
            obj.attributes["face"] = face_schema
```

### 6.3 Generic Object Pipeline

```python
class GenericPipeline(Pipeline):
    """Handle unknown objects with zero-shot techniques."""

    modules = [
        ImageNetClassifierModule(),  # ResNet-50 → 1000 classes
        ColorAnalysisModule(),       # Dominant colors
        ShapeAnalysisModule(),       # Aspect ratio, contour features
        LLMDescriptionModule(),      # Moondream → natural language
    ]

    def process(self, obj: ObjectSchema, crop: np.ndarray):
        """Process unknown object."""

        # Try ImageNet classification
        class_probs = self.imagenet_module.process(crop)
        top_class, confidence = class_probs[0]

        if confidence > 0.7:
            obj.primary_class = top_class
            obj.confidence = confidence
            obj.classification_status = ClassificationStatus.CONFIRMED
        else:
            # Use LLM for description
            description = self.llm_module.describe(crop)
            obj.attributes["llm_description"] = description
            obj.primary_class = extract_class_from_description(description)
            obj.confidence = 0.5  # Lower confidence
            obj.classification_status = ClassificationStatus.PROVISIONAL
```

---

## 7. Normalization Layer

### 7.1 Size Standardization

```python
class SizeNormalizer(PipelineModule):
    """Resize objects to canonical dimensions."""

    canonical_sizes = {
        "person": (256, 128),   # Height x Width (portrait)
        "face": (112, 112),     # Square
        "vehicle": (224, 224),  # Square
        "generic": (224, 224),  # ImageNet standard
    }

    def process(self, crop: np.ndarray, object_type: str) -> np.ndarray:
        target_size = self.canonical_sizes.get(object_type, (224, 224))
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
```

### 7.2 Light Intensity Correction

```python
class LightCorrector(PipelineModule):
    """Correct for varying lighting conditions."""

    def process(self, image: np.ndarray) -> np.ndarray:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE on luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_corrected = clahe.apply(l)

        # Merge and convert back
        lab_corrected = cv2.merge([l_corrected, a, b])
        return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
```

### 7.3 Color Correction

```python
class ColorCorrector(PipelineModule):
    """Correct for color casts (colored lighting)."""

    def process(self, image: np.ndarray) -> np.ndarray:
        # Gray-world assumption for white balance
        b, g, r = cv2.split(image.astype(np.float32))

        avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()
        avg_gray = (avg_b + avg_g + avg_r) / 3

        # Scale channels to match average gray
        b = np.clip(b * (avg_gray / avg_b), 0, 255)
        g = np.clip(g * (avg_gray / avg_g), 0, 255)
        r = np.clip(r * (avg_gray / avg_r), 0, 255)

        return cv2.merge([b, g, r]).astype(np.uint8)
```

---

## 8. Confidence & Human Review

### 8.1 Confidence Thresholds

```python
@dataclass
class ConfidenceConfig:
    """Configurable confidence thresholds."""

    # Classification confidence
    confirmed_threshold: float = 0.85    # Auto-confirm above this
    provisional_threshold: float = 0.5   # Provisional between this and confirmed
    review_threshold: float = 0.5        # Queue for review below this

    # ReID confidence
    reid_match_threshold: float = 0.3    # Cosine distance for same object
    reid_new_object_threshold: float = 0.5  # Above this = definitely new

    # Reprocessing
    reprocess_interval_seconds: float = 60.0  # Re-check provisional objects
```

### 8.2 Human Review Queue

```python
class HumanReviewQueue:
    """Store uncertain objects for async human review."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self._create_tables()

    def add_for_review(self, obj: ObjectSchema,
                       crop: np.ndarray,
                       reason: str):
        """Add object to review queue."""

        # Save crop image
        image_path = self._save_crop(obj.id, crop)

        self.db.execute("""
            INSERT INTO review_queue
            (object_id, schema_json, image_path, reason,
             provisional_class, confidence, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
        """, (obj.id, obj.to_json(), image_path, reason,
              obj.primary_class, obj.confidence, datetime.now()))

    def get_pending(self, limit: int = 50) -> List[ReviewItem]:
        """Get pending review items."""
        pass

    def submit_review(self, object_id: str,
                      human_class: str,
                      human_attributes: Dict):
        """Submit human classification."""
        # Update object schema with human labels
        # Mark as CONFIRMED
        # Optionally: feed back to improve models (active learning)
```

---

## 9. Database Schema

```sql
-- Object schemas (main entity table)
CREATE TABLE objects (
    id TEXT PRIMARY KEY,
    reid_embedding BLOB,              -- 512-dim float32

    -- Spatial
    position_x REAL, position_y REAL, position_z REAL,
    bbox_x1 INT, bbox_y1 INT, bbox_x2 INT, bbox_y2 INT,
    width_m REAL, height_m REAL, depth_m REAL,
    distance_m REAL,

    -- Classification
    primary_class TEXT,
    subclass TEXT,
    confidence REAL,
    classification_status TEXT,       -- CONFIRMED, PROVISIONAL, NEEDS_REVIEW

    -- Attributes (JSON for flexibility)
    attributes_json TEXT,

    -- Tracking
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    camera_id TEXT,

    -- Processing
    pipelines_completed TEXT,         -- Comma-separated list
    processing_time_ms REAL,

    -- Indexes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_objects_class ON objects(primary_class);
CREATE INDEX idx_objects_camera ON objects(camera_id);
CREATE INDEX idx_objects_last_seen ON objects(last_seen);
CREATE INDEX idx_objects_status ON objects(classification_status);

-- Trajectory points (for motion history)
CREATE TABLE trajectory_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id TEXT REFERENCES objects(id),
    x REAL, y REAL, z REAL,
    timestamp TIMESTAMP,
    camera_id TEXT
);

CREATE INDEX idx_trajectory_object ON trajectory_points(object_id);

-- Human review queue
CREATE TABLE review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id TEXT REFERENCES objects(id),
    schema_json TEXT,
    image_path TEXT,
    reason TEXT,
    provisional_class TEXT,
    confidence REAL,
    created_at TIMESTAMP,
    reviewed_at TIMESTAMP,
    reviewer TEXT,
    human_class TEXT,
    human_attributes_json TEXT,
    status TEXT                       -- pending, reviewed, skipped
);

CREATE INDEX idx_review_status ON review_queue(status);
```

---

## 10. Configuration System

```yaml
# percept_config.yaml

framework:
  name: "PERCEPT"
  version: "1.0"

cameras:
  - id: "cam_front"
    type: "realsense_d455"
    serial: "123456789"
    resolution: [640, 480]
    fps: 30
  - id: "cam_rear"
    type: "realsense_d415"
    serial: "987654321"
    resolution: [640, 480]
    fps: 30

segmentation:
  primary_method: "fastsam"
  fallback_methods: ["depth_discontinuity", "point_cloud_clustering"]
  fusion_enabled: true
  min_object_pixels: 500
  max_objects_per_frame: 50

reid:
  person_model: "repvgg_a0_512"
  embedding_dimension: 512
  match_threshold_same_camera: 0.3
  match_threshold_cross_camera: 0.25
  gallery_max_embeddings_per_object: 10
  reid_interval_frames: 3

tracking:
  algorithm: "bytetrack"
  lost_track_buffer_frames: 30
  min_track_confidence: 0.5

classification:
  confidence_confirmed: 0.85
  confidence_provisional: 0.5
  reprocess_interval_seconds: 60

normalization:
  enable_light_correction: true
  enable_color_correction: true
  clahe_clip_limit: 2.0

human_review:
  enabled: true
  queue_low_confidence: true
  review_threshold: 0.5

database:
  path: "/home/sevak/apps/percept/data/percept.db"
  embedding_index_type: "hnsw"

performance:
  adaptive_processing: true
  target_fps: 15
  skip_frames_when_behind: true
  max_processing_time_ms: 100
```

---

## 11. Project Structure

```
/home/sevak/apps/percept/
├── percept/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py           # Pipeline base classes
│   │   ├── module.py             # Module interface
│   │   ├── adapter.py            # Data adapters
│   │   ├── schema.py             # ObjectSchema definition
│   │   └── config.py             # Configuration loading
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── realsense.py          # Multi-camera RealSense
│   │   └── frame_sync.py         # Frame synchronization
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── fastsam.py            # FastSAM on Hailo-8
│   │   ├── depth_seg.py          # Depth discontinuity
│   │   ├── pointcloud_seg.py     # Euclidean clustering
│   │   ├── ransac.py             # RANSAC plane detection
│   │   └── fusion.py             # Multi-method fusion
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── reid.py               # ReID embedding & matching
│   │   ├── bytetrack.py          # ByteTrack wrapper
│   │   └── gallery.py            # FAISS-backed gallery
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── router.py             # Pipeline router
│   │   ├── person.py             # Person pipeline
│   │   ├── vehicle.py            # Vehicle pipeline
│   │   ├── face.py               # Face sub-pipeline
│   │   └── generic.py            # Generic object pipeline
│   ├── normalization/
│   │   ├── __init__.py
│   │   ├── size.py               # Size normalization
│   │   ├── light.py              # Light correction
│   │   └── color.py              # Color correction
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── hailo.py              # Hailo-8 inference (reuse from hailo-agents)
│   │   └── models.py             # Model registry
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── database.py           # SQLite operations
│   │   ├── embedding_store.py    # FAISS index management
│   │   └── review_queue.py       # Human review queue
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py           # 3D math utilities
│       ├── visualization.py      # Debug visualization
│       └── metrics.py            # Performance monitoring
├── config/
│   └── percept_config.yaml
├── data/
│   ├── percept.db
│   ├── embeddings/
│   └── review_images/
├── examples/
│   ├── basic_detection.py
│   ├── person_tracking.py
│   └── multi_camera.py
├── ui/
│   ├── __init__.py
│   ├── app.py                    # Main Flask/FastAPI application
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   │   ├── dashboard.html        # Main dashboard
│   │   ├── pipeline_view.html    # Pipeline visualization
│   │   ├── config_editor.html    # Pipeline configuration
│   │   └── review_queue.html     # Human review interface
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py             # REST API endpoints
│   │   └── websocket.py          # Real-time updates
│   └── components/
│       ├── pipeline_graph.py     # Pipeline DAG visualization
│       ├── frame_viewer.py       # Live frame display
│       └── metrics_panel.py      # Performance metrics
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── unit/
│   │   ├── test_schema.py
│   │   ├── test_pipeline.py
│   │   ├── test_adapter.py
│   │   ├── test_segmentation.py
│   │   ├── test_tracking.py
│   │   └── test_database.py
│   ├── integration/
│   │   ├── test_pipeline_flow.py
│   │   ├── test_multi_camera.py
│   │   ├── test_reid_matching.py
│   │   └── test_end_to_end.py
│   ├── performance/
│   │   ├── test_latency.py
│   │   ├── test_throughput.py
│   │   └── test_memory.py
│   ├── fixtures/
│   │   ├── sample_images/
│   │   ├── sample_depths/
│   │   └── mock_detections.json
│   └── mocks/
│       ├── mock_camera.py
│       ├── mock_hailo.py
│       └── mock_database.py
└── README.md
```

---

## 12. Implementation Phases

### Phase 1: Foundation (Core Infrastructure)
- [ ] Project structure setup
- [ ] Configuration system
- [ ] Pipeline base classes and module interface
- [ ] Data adapter framework
- [ ] Multi-camera RealSense capture
- [ ] Database schema and basic operations
- [ ] Test framework setup (pytest, fixtures, mocks)

### Phase 2: Segmentation Layer
- [ ] FastSAM integration (reuse HailoInference)
- [ ] Depth discontinuity segmentation
- [ ] Point cloud generation and clustering
- [ ] RANSAC plane detection
- [ ] Segmentation fusion
- [ ] Unit tests for segmentation modules

### Phase 3: Tracking & ReID
- [ ] ByteTrack integration
- [ ] ReID embedding extraction (RepVGG)
- [ ] FAISS gallery management
- [ ] Cross-camera matching
- [ ] Mask-based mutual exclusion
- [ ] Unit tests for tracking/ReID

### Phase 4: Classification Pipelines
- [ ] Pipeline router
- [ ] Person pipeline (pose, clothing, face detection)
- [ ] Generic object pipeline (ImageNet, LLM description)
- [ ] Normalization modules
- [ ] Unit tests for pipelines

### Phase 5: Persistence & Review
- [ ] Full database operations
- [ ] Human review queue
- [ ] ObjectSchema serialization
- [ ] Trajectory storage
- [ ] Database unit tests

### Phase 6: Integration & Testing
- [ ] Integration tests for full pipeline flow
- [ ] Cross-camera ReID integration tests
- [ ] Performance benchmarks (latency, throughput, memory)
- [ ] Hardware smoke tests

### Phase 7: Pipeline Visualization UI
- [ ] FastAPI backend setup
- [ ] REST API endpoints
- [ ] WebSocket real-time streaming
- [ ] Dashboard view (system stats, camera feeds)
- [ ] Pipeline DAG visualization
- [ ] Stage intermediate output viewer
- [ ] Configuration editor with validation
- [ ] Human review interface

### Phase 8: Polish & Documentation
- [ ] Adaptive processing based on scene complexity
- [ ] Performance optimization
- [ ] CI/CD integration
- [ ] API documentation
- [ ] User guide and examples
- [ ] Deployment instructions

---

## 13. Reusable Components from hailo-agents

| Component | Path | Reuse Strategy |
|-----------|------|----------------|
| `HailoInference` | `utils/hailo_inference.py` | Import directly, extend for new models |
| `FastSAMInference` | `utils/hailo_inference.py` | Import directly |
| `RealSenseCapture` | `utils/realsense_capture.py` | Extend for multi-camera sync |
| `ObjectTracker` | `scanner/object_tracker.py` | Reference for fingerprint matching |
| `ModelDatabase` | `scanner/model_database.py` | Adapt schema pattern |
| `PointCloudProcessor` | `scanner/point_cloud_processor.py` | Reuse RANSAC, clustering |

---

## 14. Key Design Decisions

1. **Pipeline modularity over monolithic processing** - Each algorithm is a swappable module
2. **ReID-first tracking** - Identity verification prevents redundant processing
3. **3D-native** - All objects have spatial coordinates from inception
4. **Graceful degradation** - Falls back to simpler methods when AI fails
5. **Human-in-the-loop** - Uncertain classifications queued, not discarded
6. **Multi-camera from start** - Architecture supports multiple cameras natively
7. **SLAM-ready coordinates** - Camera-relative now, world coordinates when SLAM added
8. **Test-driven development** - Comprehensive tests validate each component before integration
9. **Observable pipelines** - Every stage produces inspectable intermediate outputs for debugging
10. **Configuration over code** - Pipeline composition and parameters editable without code changes

---

## 15. Pipeline Visualization UI

### 15.1 Overview

The PERCEPT UI provides real-time visualization of pipeline execution, configuration management, and human review capabilities. Built with FastAPI (backend) and a lightweight JavaScript frontend for embedded deployment.

### 15.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PERCEPT UI                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   Dashboard     │    │  Pipeline View  │    │  Config Editor  │  │
│  │                 │    │                 │    │                 │  │
│  │ • System stats  │    │ • DAG graph     │    │ • YAML editor   │  │
│  │ • Camera feeds  │    │ • Stage outputs │    │ • Module params │  │
│  │ • Object count  │    │ • Timing data   │    │ • Hot reload    │  │
│  │ • Alerts        │    │ • Zoom/pan      │    │ • Validation    │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │  Review Queue   │    │  Object Gallery │    │   Metrics       │  │
│  │                 │    │                 │    │                 │  │
│  │ • Pending items │    │ • Browse objects│    │ • FPS charts    │  │
│  │ • Quick classify│    │ • Search/filter │    │ • Latency hist  │  │
│  │ • Batch review  │    │ • View history  │    │ • Memory usage  │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                         Backend (FastAPI)                            │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  REST API    │  │  WebSocket   │  │  Static      │               │
│  │  /api/...    │  │  /ws/stream  │  │  Files       │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 15.3 Dashboard View

Main monitoring interface showing system status at a glance:

```python
@dataclass
class DashboardData:
    """Real-time dashboard data."""

    # System metrics
    fps: float                    # Current processing FPS
    cpu_usage: float              # Percent
    memory_usage: float           # Percent
    hailo_utilization: float      # NPU usage percent
    temperature: float            # CPU temperature

    # Pipeline status
    active_pipelines: List[str]   # Currently running pipelines
    queue_depth: int              # Frames waiting to process

    # Object counts
    total_objects: int            # In database
    objects_in_view: int          # Currently visible
    pending_review: int           # Awaiting human review

    # Camera status
    cameras: List[CameraStatus]   # Per-camera info

    # Alerts
    alerts: List[Alert]           # Warnings, errors
```

### 15.4 Pipeline Visualization View

Interactive DAG showing data flow through pipeline stages:

```python
class PipelineVisualizer:
    """Generate interactive pipeline visualization."""

    def get_pipeline_graph(self, pipeline_name: str) -> PipelineGraph:
        """Return DAG structure for visualization."""
        return PipelineGraph(
            nodes=[
                Node(id="capture", type="input", label="Camera Capture"),
                Node(id="segment", type="process", label="Segmentation"),
                Node(id="reid", type="process", label="ReID Matching"),
                Node(id="classify", type="process", label="Classification"),
                Node(id="persist", type="output", label="Database"),
            ],
            edges=[
                Edge(source="capture", target="segment"),
                Edge(source="segment", target="reid"),
                Edge(source="reid", target="classify"),
                Edge(source="classify", target="persist"),
            ]
        )

    def get_stage_output(self, stage_id: str,
                         frame_id: int) -> StageOutput:
        """Get intermediate output from a pipeline stage."""
        return StageOutput(
            stage_id=stage_id,
            frame_id=frame_id,
            image=self._get_stage_image(stage_id, frame_id),
            metadata=self._get_stage_metadata(stage_id, frame_id),
            timing_ms=self._get_stage_timing(stage_id, frame_id),
        )
```

**Features:**
- Click any stage node to see its output image/data
- Hover for timing information and throughput
- Color-coded status (green=ok, yellow=slow, red=error)
- Zoom/pan for complex pipelines
- Toggle to show/hide optional stages

### 15.5 Configuration Editor

Edit pipeline configuration with live validation:

```python
class ConfigEditor:
    """Pipeline configuration editor."""

    def get_config(self) -> dict:
        """Get current configuration."""
        return load_yaml_config()

    def validate_config(self, config: dict) -> ValidationResult:
        """Validate configuration before applying."""
        errors = []
        warnings = []

        # Check required fields
        if "cameras" not in config:
            errors.append("Missing 'cameras' section")

        # Check module availability
        for module in config.get("pipelines", {}).get("person", []):
            if not self._module_exists(module):
                errors.append(f"Unknown module: {module}")

        # Check threshold ranges
        if config.get("reid", {}).get("match_threshold", 0) > 1.0:
            warnings.append("ReID threshold > 1.0 may cause no matches")

        return ValidationResult(valid=len(errors) == 0,
                                errors=errors, warnings=warnings)

    def apply_config(self, config: dict) -> bool:
        """Apply configuration (hot reload)."""
        if not self.validate_config(config).valid:
            return False
        save_yaml_config(config)
        self._trigger_reload()
        return True
```

**Features:**
- Syntax-highlighted YAML editor
- Real-time validation with error highlighting
- Module documentation on hover
- Hot reload without restart
- Configuration history/rollback

### 15.6 Human Review Interface

Efficient interface for reviewing uncertain classifications:

```python
class ReviewInterface:
    """Human review queue interface."""

    def get_review_batch(self, limit: int = 20) -> List[ReviewItem]:
        """Get batch of items for review."""
        return self.review_queue.get_pending(limit)

    def submit_review(self, object_id: str,
                      classification: ReviewClassification):
        """Submit human review."""
        self.review_queue.submit_review(
            object_id=object_id,
            human_class=classification.primary_class,
            human_subclass=classification.subclass,
            human_attributes=classification.attributes,
            reviewer=classification.reviewer,
        )

        # Update object schema
        self.database.update_object(
            object_id,
            primary_class=classification.primary_class,
            classification_status=ClassificationStatus.CONFIRMED,
        )
```

**Features:**
- Grid view of pending items with thumbnails
- One-click classification for common classes
- Keyboard shortcuts for fast review
- Bulk selection for similar items
- Skip/flag options for ambiguous cases

### 15.7 REST API Endpoints

```python
# API Routes

# Dashboard
GET  /api/dashboard              # Dashboard data
GET  /api/metrics                # Detailed metrics

# Cameras
GET  /api/cameras                # List cameras
GET  /api/cameras/{id}/frame     # Current frame (JPEG)
GET  /api/cameras/{id}/depth     # Depth visualization

# Pipeline
GET  /api/pipeline/graph         # Pipeline DAG structure
GET  /api/pipeline/stages        # List all stages
GET  /api/pipeline/stages/{id}/output  # Stage intermediate output

# Configuration
GET  /api/config                 # Current config
PUT  /api/config                 # Update config
POST /api/config/validate        # Validate config

# Objects
GET  /api/objects                # List objects (paginated)
GET  /api/objects/{id}           # Object details
GET  /api/objects/{id}/trajectory  # Position history

# Review Queue
GET  /api/review                 # Pending reviews
POST /api/review/{id}            # Submit review
POST /api/review/{id}/skip       # Skip item

# WebSocket
WS   /ws/stream                  # Real-time frame stream
WS   /ws/events                  # Pipeline events
```

### 15.8 WebSocket Events

```python
# Real-time events via WebSocket

class PipelineEvent:
    """Events pushed to connected clients."""

    FRAME_PROCESSED = "frame_processed"      # New frame complete
    OBJECT_DETECTED = "object_detected"      # New object found
    OBJECT_UPDATED = "object_updated"        # Object re-identified
    REVIEW_NEEDED = "review_needed"          # Low confidence item
    ALERT = "alert"                          # Warning or error
    CONFIG_CHANGED = "config_changed"        # Config reloaded

# Example event payload
{
    "event": "object_detected",
    "timestamp": "2025-12-23T10:30:00Z",
    "data": {
        "object_id": "abc-123",
        "class": "person",
        "confidence": 0.92,
        "position": [1.2, 0.0, 3.5],
        "camera_id": "cam_front"
    }
}
```

---

## 16. Test Suite

### 16.1 Testing Strategy

PERCEPT uses a multi-layer testing approach:

| Layer | Purpose | Tools | Coverage Target |
|-------|---------|-------|-----------------|
| **Unit** | Test individual functions/classes | pytest | 80%+ |
| **Integration** | Test component interactions | pytest + fixtures | Key flows |
| **Performance** | Verify latency/throughput | pytest-benchmark | Baselines |
| **Hardware** | Test with real devices | Manual + scripts | Smoke tests |

### 16.2 Test Framework Configuration

```python
# conftest.py - Shared pytest fixtures

import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_rgb_image():
    """Load sample RGB image for testing."""
    return cv2.imread(str(FIXTURES_DIR / "sample_images" / "office_scene.jpg"))

@pytest.fixture
def sample_depth_image():
    """Load sample depth image (16-bit PNG, mm units)."""
    depth = cv2.imread(
        str(FIXTURES_DIR / "sample_depths" / "office_depth.png"),
        cv2.IMREAD_UNCHANGED
    )
    return depth.astype(np.float32) / 1000.0  # Convert to meters

@pytest.fixture
def mock_hailo():
    """Mock Hailo inference for tests without hardware."""
    return MockHailoInference()

@pytest.fixture
def mock_camera():
    """Mock camera returning test frames."""
    return MockRealSenseCamera()

@pytest.fixture
def temp_database(tmp_path):
    """Temporary SQLite database for tests."""
    db_path = tmp_path / "test.db"
    db = PerceptDatabase(str(db_path))
    db.initialize()
    yield db
    db.close()

@pytest.fixture
def sample_object_schema():
    """Sample ObjectSchema for testing."""
    return ObjectSchema(
        id="test-001",
        reid_embedding=np.random.randn(512).astype(np.float32),
        position_3d=(1.0, 0.5, 2.5),
        bounding_box_2d=(100, 100, 200, 300),
        dimensions=(0.5, 1.8, 0.3),
        distance_from_camera=2.5,
        primary_class="person",
        confidence=0.85,
        classification_status=ClassificationStatus.CONFIRMED,
        attributes={"pose": [...], "clothing": {...}},
        first_seen=datetime.now(),
        last_seen=datetime.now(),
        camera_id="cam_front",
        trajectory=[],
        pipelines_completed=["segmentation", "person"],
        processing_time_ms=45.2,
        source_frame_ids=[1, 2, 3],
    )
```

### 16.3 Unit Tests

```python
# tests/unit/test_schema.py

class TestObjectSchema:
    """Unit tests for ObjectSchema."""

    def test_create_schema(self):
        """Test ObjectSchema creation."""
        schema = ObjectSchema(
            id="test-001",
            reid_embedding=np.zeros(512),
            ...
        )
        assert schema.id == "test-001"
        assert schema.reid_embedding.shape == (512,)

    def test_to_json(self, sample_object_schema):
        """Test JSON serialization."""
        json_str = sample_object_schema.to_json()
        restored = ObjectSchema.from_json(json_str)
        assert restored.id == sample_object_schema.id
        assert restored.primary_class == sample_object_schema.primary_class

    def test_embedding_normalization(self):
        """Test that embeddings are L2 normalized."""
        embedding = np.array([3.0, 4.0] + [0.0] * 510)  # 3-4-5 triangle
        schema = ObjectSchema(reid_embedding=embedding, ...)
        norm = np.linalg.norm(schema.reid_embedding)
        assert abs(norm - 1.0) < 1e-6


# tests/unit/test_adapter.py

class TestDataAdapter:
    """Unit tests for data adaptation."""

    def test_resize_image(self):
        """Test image resizing adapter."""
        adapter = DataAdapter()
        input_spec = DataSpec(data_type="image", shape=(480, 640, 3))
        output_spec = DataSpec(data_type="image", shape=(256, 128, 3))

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = adapter.adapt(image, input_spec, output_spec)

        assert result.shape == (256, 128, 3)

    def test_color_space_conversion(self):
        """Test BGR to RGB conversion."""
        adapter = DataAdapter()
        bgr_spec = DataSpec(data_type="image", color_space="BGR")
        rgb_spec = DataSpec(data_type="image", color_space="RGB")

        bgr_image = np.array([[[255, 0, 0]]])  # Blue in BGR
        rgb_image = adapter.adapt(bgr_image, bgr_spec, rgb_spec)

        assert rgb_image[0, 0, 0] == 0    # Red channel
        assert rgb_image[0, 0, 2] == 255  # Blue channel


# tests/unit/test_segmentation.py

class TestDepthSegmentation:
    """Unit tests for depth-based segmentation."""

    def test_depth_discontinuity_detection(self, sample_depth_image):
        """Test edge detection on depth."""
        edges = detect_depth_discontinuities(sample_depth_image)

        assert edges.shape == sample_depth_image.shape
        assert edges.dtype == np.uint8
        assert np.any(edges > 0)  # Some edges detected

    def test_connected_components(self):
        """Test connected component extraction."""
        # Create synthetic depth with two objects
        depth = np.ones((100, 100)) * 2.0  # Background at 2m
        depth[20:40, 20:40] = 1.0  # Object 1 at 1m
        depth[60:80, 60:80] = 1.5  # Object 2 at 1.5m

        masks = depth_connected_components(depth, threshold=0.3)

        assert len(masks) >= 2  # At least 2 objects found
```

### 16.4 Integration Tests

```python
# tests/integration/test_pipeline_flow.py

class TestPipelineFlow:
    """Integration tests for full pipeline execution."""

    def test_detection_to_schema(self, mock_camera, mock_hailo, temp_database):
        """Test complete flow from detection to database."""
        pipeline = PerceptPipeline(
            camera=mock_camera,
            hailo=mock_hailo,
            database=temp_database,
        )

        # Process one frame
        result = pipeline.process_frame()

        # Verify objects were detected and stored
        assert len(result.objects) > 0
        for obj in result.objects:
            assert obj.id is not None
            assert obj.primary_class is not None
            stored = temp_database.get_object(obj.id)
            assert stored is not None

    def test_reid_matching(self, mock_camera, temp_database):
        """Test ReID matches same object across frames."""
        pipeline = PerceptPipeline(...)

        # Process first frame
        result1 = pipeline.process_frame()
        person1 = result1.objects[0]

        # Process second frame (same scene, person moved slightly)
        mock_camera.advance_frame()
        result2 = pipeline.process_frame()

        # Should match same person (not create new object)
        assert len(temp_database.get_all_objects()) == 1
        assert result2.objects[0].id == person1.id

    def test_pipeline_module_swap(self):
        """Test hot-swapping pipeline modules."""
        pipeline = PerceptPipeline(...)

        # Start with default segmentation
        assert pipeline.get_module("segmentation").name == "fastsam"

        # Swap to depth-only segmentation
        pipeline.swap_module("segmentation", DepthOnlySegmentation())
        assert pipeline.get_module("segmentation").name == "depth_only"

        # Verify still works
        result = pipeline.process_frame()
        assert len(result.objects) >= 0  # May find different objects


# tests/integration/test_multi_camera.py

class TestMultiCamera:
    """Integration tests for multi-camera scenarios."""

    def test_cross_camera_reid(self, temp_database):
        """Test ReID matching across cameras."""
        cam1 = MockCamera(id="cam_front")
        cam2 = MockCamera(id="cam_rear")

        pipeline = PerceptPipeline(cameras=[cam1, cam2], ...)

        # Person visible in camera 1
        cam1.set_scene(person_at=(1.0, 0, 2.0))
        result1 = pipeline.process_cameras()
        person_id = result1["cam_front"].objects[0].id

        # Same person now visible in camera 2
        cam1.clear_scene()
        cam2.set_scene(person_at=(1.0, 0, 2.0))
        result2 = pipeline.process_cameras()

        # Should recognize as same person
        assert result2["cam_rear"].objects[0].id == person_id
```

### 16.5 Performance Tests

```python
# tests/performance/test_latency.py

class TestLatency:
    """Performance tests for processing latency."""

    @pytest.mark.benchmark
    def test_segmentation_latency(self, benchmark, sample_rgb_image, mock_hailo):
        """Benchmark segmentation latency."""
        segmenter = FastSAMSegmentation(hailo=mock_hailo)

        result = benchmark(segmenter.segment, sample_rgb_image)

        # Assert latency is under threshold
        assert benchmark.stats["mean"] < 0.060  # 60ms max

    @pytest.mark.benchmark
    def test_reid_matching_latency(self, benchmark):
        """Benchmark ReID gallery matching."""
        gallery = ReIDGallery()

        # Pre-populate gallery with 1000 embeddings
        for i in range(1000):
            gallery.add(f"obj-{i}", np.random.randn(512).astype(np.float32))

        query = np.random.randn(512).astype(np.float32)
        result = benchmark(gallery.search, query, k=5)

        assert benchmark.stats["mean"] < 0.005  # 5ms max for 1000 objects

    @pytest.mark.benchmark
    def test_full_pipeline_latency(self, benchmark, mock_camera, mock_hailo):
        """Benchmark full pipeline processing."""
        pipeline = PerceptPipeline(...)

        result = benchmark(pipeline.process_frame)

        # Full pipeline should complete in 100ms for 10 FPS target
        assert benchmark.stats["mean"] < 0.100


# tests/performance/test_memory.py

class TestMemory:
    """Memory usage tests."""

    def test_gallery_memory_growth(self):
        """Test that gallery memory grows linearly."""
        import tracemalloc
        tracemalloc.start()

        gallery = ReIDGallery()

        # Add 10,000 objects
        for i in range(10000):
            gallery.add(f"obj-{i}", np.random.randn(512).astype(np.float32))

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 10K objects × 512 floats × 4 bytes = ~20MB expected
        # Allow 50MB for overhead
        assert peak < 50 * 1024 * 1024
```

### 16.6 Mock Components

```python
# tests/mocks/mock_camera.py

class MockRealSenseCamera:
    """Mock camera for testing without hardware."""

    def __init__(self, frames_dir: Path = None):
        self.frame_index = 0
        self.frames = self._load_frames(frames_dir)

    def capture(self) -> dict:
        """Return mock frame data."""
        frame = self.frames[self.frame_index % len(self.frames)]
        return {
            "color": frame["rgb"],
            "depth": frame["depth"],
            "intrinsics": self._mock_intrinsics(),
            "timestamp": time.time(),
        }

    def advance_frame(self):
        """Advance to next frame."""
        self.frame_index += 1


# tests/mocks/mock_hailo.py

class MockHailoInference:
    """Mock Hailo inference returning pre-computed results."""

    def __init__(self, results_path: Path = None):
        self.mock_results = self._load_results(results_path)

    def detect(self, image: np.ndarray,
               conf_threshold: float = 0.5) -> List[Detection]:
        """Return mock detections."""
        # Return detections based on image hash or random
        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.92,
                bbox=(100, 50, 200, 350),
            ),
        ]

    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        """Return mock segmentation masks."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[50:350, 100:200] = 255  # Mock person mask
        return [mask]
```

### 16.7 Test Fixtures

```
tests/fixtures/
├── sample_images/
│   ├── office_scene.jpg       # Indoor office with people
│   ├── outdoor_scene.jpg      # Outdoor with vehicles
│   ├── crowded_scene.jpg      # Multiple people
│   └── empty_scene.jpg        # No objects
├── sample_depths/
│   ├── office_depth.png       # Matching depth for office
│   ├── outdoor_depth.png      # Matching depth for outdoor
│   └── synthetic_depth.npy    # Programmatically generated
├── mock_detections.json       # Pre-computed detection results
├── mock_embeddings.npy        # Pre-computed ReID embeddings
└── test_config.yaml           # Test configuration
```

### 16.8 CI/CD Integration

```yaml
# .github/workflows/test.yaml (or equivalent)

name: PERCEPT Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-benchmark

      - name: Run unit tests
        run: pytest tests/unit -v --cov=percept --cov-report=xml

      - name: Run integration tests
        run: pytest tests/integration -v

      - name: Run performance benchmarks
        run: pytest tests/performance --benchmark-only --benchmark-json=benchmark.json

      - name: Check coverage threshold
        run: |
          coverage report --fail-under=80
```

### 16.9 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=percept --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Run specific test file
pytest tests/unit/test_schema.py -v

# Run tests matching pattern
pytest -k "reid" -v

# Run with verbose output
pytest tests/ -v --tb=short
```

---

*This specification document will be refined as implementation progresses.*
*Generated by Claude Code - December 23, 2025*

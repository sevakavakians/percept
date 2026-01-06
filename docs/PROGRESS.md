# PERCEPT Implementation Progress

**Last Updated:** January 6, 2026
**Total Tests:** 563 passing (9 skipped)

---

## Phase Overview

| Phase | Name | Status | Tests |
|-------|------|--------|-------|
| 1 | Foundation | ✅ COMPLETE | 154 |
| 2 | Segmentation Layer | ✅ COMPLETE | 51 |
| 3 | Tracking & ReID | ✅ COMPLETE | 92 |
| 4 | Classification Pipelines | ✅ COMPLETE | 68 |
| 5 | Persistence & Review | ✅ COMPLETE | 70 |
| 6 | Integration & Testing | ✅ COMPLETE | 63 |
| 7 | Pipeline Visualization UI | ✅ COMPLETE | 65 |
| 8 | Polish & Documentation | ⏳ Not Started | - |

---

## Phase 1: Foundation ✅ COMPLETE

### Implemented Modules

| Module | File | Description |
|--------|------|-------------|
| ObjectSchema | `percept/core/schema.py` | Central data structure for detected objects with L2-normalized embeddings, JSON serialization |
| ClassificationStatus | `percept/core/schema.py` | Enum: CONFIRMED, PROVISIONAL, NEEDS_REVIEW, UNCLASSIFIED |
| Detection | `percept/core/schema.py` | Raw detection from object detector |
| ObjectMask | `percept/core/schema.py` | Segmentation mask with bbox and depth info |
| PerceptConfig | `percept/core/config.py` | YAML configuration with typed dataclasses for all sections |
| PipelineModule | `percept/core/pipeline.py` | Abstract base class for hot-swappable modules |
| Pipeline | `percept/core/pipeline.py` | Orchestration with automatic data adaptation |
| PipelineRegistry | `percept/core/pipeline.py` | Dynamic pipeline construction from templates |
| DataSpec | `percept/core/adapter.py` | Specification for data flowing through pipelines |
| PipelineData | `percept/core/adapter.py` | Flexible container for pipeline data |
| DataAdapter | `percept/core/adapter.py` | Automatic data conversion (resize, color space, dtype) |
| PerceptDatabase | `percept/persistence/database.py` | SQLite CRUD, trajectory storage, review queue |
| RealSenseCamera | `percept/capture/realsense.py` | Single camera RGB-D capture |
| MultiCameraCapture | `percept/capture/realsense.py` | Multi-camera management |
| FrameData | `percept/capture/realsense.py` | Frame container with PipelineData conversion |
| CameraIntrinsics | `percept/capture/realsense.py` | Camera parameters with matrix generation |

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_schema.py` | 30 | ObjectSchema, Detection, ObjectMask |
| `tests/unit/test_config.py` | 20 | PerceptConfig, CameraConfig |
| `tests/unit/test_pipeline.py` | 26 | Pipeline, PipelineModule, Registry |
| `tests/unit/test_adapter.py` | 35 | DataSpec, PipelineData, DataAdapter |
| `tests/unit/test_database.py` | 28 | PerceptDatabase, Review queue |
| `tests/unit/test_capture.py` | 15 | FrameData, CameraIntrinsics, Mock camera |

### Key Design Decisions

1. **L2-normalized embeddings** - All ReID embeddings auto-normalized on set
2. **Typed configuration** - Dataclasses for each config section
3. **DataSpec matching** - Automatic compatibility checking between modules
4. **Transaction support** - Database operations wrapped in transactions
5. **Graceful degradation** - Capture module works without pyrealsense2

---

## Phase 2: Segmentation Layer ✅ COMPLETE

### Goals
- Segment objects from RGB-D frames using multiple methods
- Fuse results for robust segmentation
- Output ObjectMask instances for downstream processing

### Implemented Modules

| Module | File | Description |
|--------|------|-------------|
| DepthEdgeDetector | `percept/segmentation/depth_seg.py` | Sobel-based depth discontinuity detection |
| DepthConnectedComponents | `percept/segmentation/depth_seg.py` | Connected component segmentation with k-means refinement |
| DepthSegmenter | `percept/segmentation/depth_seg.py` | Pipeline module combining edges + components |
| PlaneModel | `percept/segmentation/ransac.py` | Plane representation with distance/inlier calculations |
| RANSACPlaneDetector | `percept/segmentation/ransac.py` | RANSAC algorithm for dominant plane detection |
| PlaneRemovalModule | `percept/segmentation/ransac.py` | Pipeline module for floor/table removal |
| PointCloud | `percept/segmentation/pointcloud_seg.py` | Point cloud container with colors, normals, indices |
| DepthToPointCloud | `percept/segmentation/pointcloud_seg.py` | RGB-D to point cloud conversion |
| PointCloudFilter | `percept/segmentation/pointcloud_seg.py` | Voxel downsampling, statistical outlier removal |
| EuclideanClusterer | `percept/segmentation/pointcloud_seg.py` | KDTree-based flood-fill clustering |
| PointCloudSegmenter | `percept/segmentation/pointcloud_seg.py` | Pipeline module for 3D clustering |
| FastSAMInference | `percept/segmentation/fastsam.py` | Hailo-8 FastSAM model wrapper |
| FastSAMSegmenter | `percept/segmentation/fastsam.py` | Pipeline module with graceful degradation |
| MaskMerger | `percept/segmentation/fusion.py` | IoU-based mask matching and merging |
| DepthRefiner | `percept/segmentation/fusion.py` | Depth-based mask refinement and splitting |
| SegmentationFusion | `percept/segmentation/fusion.py` | Multi-method fusion with confidence boosting |

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_segmentation.py` | 51 | All segmentation modules |

### Key Design Decisions

1. **No Open3D dependency** - Uses scipy KDTree for ARM64 compatibility
2. **Multiple methods** - AI (FastSAM), depth edges, 3D clustering
3. **Graceful degradation** - FastSAM works without Hailo hardware (returns empty)
4. **Fusion strategy** - IoU-based grouping, confidence boosting for agreement
5. **Depth refinement** - Split masks by depth clusters, remove inconsistent regions

### Architecture

```
RGB-D Frame
    │
    ├─→ FastSAM (Hailo-8) ────────────┐
    │                                  │
    ├─→ Depth Edges ─→ Connected      │
    │   Components                     ├─→ Fusion ─→ Final Masks
    │                                  │
    └─→ Point Cloud ─→ Euclidean      │
        Clustering ────────────────────┘
```

---

## Phase 3: Tracking & ReID ✅ COMPLETE

### Goals
- Track objects across frames using ByteTrack algorithm
- Re-identify objects using appearance embeddings
- Manage mask claims to prevent duplicate processing

### Implemented Modules

| Module | File | Description |
|--------|------|-------------|
| EmbeddingType | `percept/tracking/reid.py` | Enum: DEEP, HISTOGRAM, HYBRID |
| ReIDConfig | `percept/tracking/reid.py` | Configuration for ReID system |
| HistogramEmbedding | `percept/tracking/reid.py` | Color/texture histogram embeddings |
| DeepEmbedding | `percept/tracking/reid.py` | Hailo-8 RepVGG embeddings with fallback |
| ReIDExtractor | `percept/tracking/reid.py` | Combined embedding extraction |
| ReIDMatcher | `percept/tracking/reid.py` | Gallery-based object matching |
| ReIDModule | `percept/tracking/reid.py` | Pipeline module for ReID |
| GalleryConfig | `percept/tracking/gallery.py` | Configuration for embedding gallery |
| GalleryEntry | `percept/tracking/gallery.py` | Entry with ID, embedding, metadata |
| FAISSGallery | `percept/tracking/gallery.py` | FAISS-backed embedding index |
| MultiCameraGallery | `percept/tracking/gallery.py` | Cross-camera matching support |
| TrackState | `percept/tracking/bytetrack.py` | Enum: TENTATIVE, CONFIRMED, LOST, DELETED |
| ByteTrackConfig | `percept/tracking/bytetrack.py` | Configuration for tracking |
| Track | `percept/tracking/bytetrack.py` | Track dataclass with velocity prediction |
| SimpleIoUTracker | `percept/tracking/bytetrack.py` | IoU-based fallback tracker |
| ByteTrackWrapper | `percept/tracking/bytetrack.py` | supervision.ByteTrack with fallback |
| TrackingModule | `percept/tracking/bytetrack.py` | Pipeline module for tracking |
| MaskManagerConfig | `percept/tracking/mask_manager.py` | Configuration for mask management |
| MaskClaim | `percept/tracking/mask_manager.py` | Claimed region with priority/timestamp |
| SceneMaskManager | `percept/tracking/mask_manager.py` | Claim/release region management |
| MaskConflictResolver | `percept/tracking/mask_manager.py` | Priority-based conflict resolution |
| MaskManagerModule | `percept/tracking/mask_manager.py` | Pipeline module for mask filtering |

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_tracking.py` | 92 | All tracking modules |

### Key Design Decisions

1. **Graceful degradation** - Deep embeddings → Histogram when Hailo unavailable
2. **FAISS with fallback** - Uses brute-force numpy search when FAISS unavailable
3. **ByteTrack fallback** - Simple IoU tracker when supervision package unavailable
4. **Camera-aware matching** - Different thresholds for same-camera vs cross-camera
5. **Temporal claims** - Stale claims auto-expire after configurable duration
6. **L2 normalization** - All embeddings normalized for cosine similarity

### Architecture

```
Detections
    │
    ├─→ ByteTrack ───────→ Tracks ─┐
    │                              │
    └─→ ReID Extractor ─→ Gallery ─┼─→ Object IDs
                                   │
               Mask Manager ←──────┘
                    │
                    └─→ Filtered Masks
```

---

## Phase 4: Classification Pipelines ✅ COMPLETE

### Goals
- Route objects to specialized classification pipelines
- Extract detailed attributes from persons, vehicles, and generic objects
- Provide graceful degradation when Hailo-8 unavailable

### Implemented Modules

| Module | File | Description |
|--------|------|-------------|
| PipelineType | `percept/pipelines/router.py` | Enum: PERSON, VEHICLE, GENERIC, FACE |
| RouterConfig | `percept/pipelines/router.py` | Configuration for routing behavior |
| RoutingDecision | `percept/pipelines/router.py` | Result with pipeline type, confidence, reason |
| PipelineRouter | `percept/pipelines/router.py` | Route based on class, size, confidence |
| RouterModule | `percept/pipelines/router.py` | Pipeline module for object routing |
| Posture | `percept/pipelines/person.py` | Enum: STANDING, SITTING, LYING, CROUCHING |
| Keypoint | `percept/pipelines/person.py` | Pose keypoint with x, y, confidence |
| PoseResult | `percept/pipelines/person.py` | Pose estimation with keypoints, posture |
| ClothingResult | `percept/pipelines/person.py` | Upper/lower clothing colors and types |
| PoseEstimator | `percept/pipelines/person.py` | YOLOv8-pose with Hailo-8, posture classification |
| ClothingAnalyzer | `percept/pipelines/person.py` | HSV histogram-based color/type analysis |
| FaceDetector | `percept/pipelines/person.py` | Face bbox detection with Haar cascade fallback |
| PersonPipeline | `percept/pipelines/person.py` | Orchestrate pose, clothing, face analysis |
| PersonPipelineModule | `percept/pipelines/person.py` | Pipeline module wrapper |
| VehicleType | `percept/pipelines/vehicle.py` | Enum: SEDAN, SUV, TRUCK, etc. |
| VehicleOrientation | `percept/pipelines/vehicle.py` | Enum: FRONT, BACK, LEFT, RIGHT |
| VehicleColorResult | `percept/pipelines/vehicle.py` | Primary/secondary colors with confidence |
| VehicleTypeResult | `percept/pipelines/vehicle.py` | Type, orientation, subtype classification |
| LicensePlateResult | `percept/pipelines/vehicle.py` | Plate bbox, text (placeholder), confidence |
| VehicleColorAnalyzer | `percept/pipelines/vehicle.py` | HSV histogram with automotive color mapping |
| VehicleTypeClassifier | `percept/pipelines/vehicle.py` | Aspect ratio + shape-based classification |
| LicensePlateDetector | `percept/pipelines/vehicle.py` | Edge + contour-based plate detection |
| VehiclePipeline | `percept/pipelines/vehicle.py` | Orchestrate color, type, plate analysis |
| VehiclePipelineModule | `percept/pipelines/vehicle.py` | Pipeline module wrapper |
| ClassificationResult | `percept/pipelines/generic.py` | ImageNet class with confidence |
| ColorResult | `percept/pipelines/generic.py` | Dominant/accent colors with proportions |
| ShapeResult | `percept/pipelines/generic.py` | Solidity, extent, circularity, aspect ratio |
| ImageNetClassifier | `percept/pipelines/generic.py` | ResNet-50 on Hailo-8 with top-k results |
| ColorAnalyzer | `percept/pipelines/generic.py` | K-means dominant color extraction |
| ShapeAnalyzer | `percept/pipelines/generic.py` | Contour-based shape metrics |
| SizeEstimator | `percept/pipelines/generic.py` | 3D size from depth + intrinsics |
| GenericPipeline | `percept/pipelines/generic.py` | Orchestrate classification, color, shape, size |
| GenericPipelineModule | `percept/pipelines/generic.py` | Pipeline module wrapper |

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_pipelines.py` | 68 | Router, Person, Vehicle, Generic pipelines |

### Key Design Decisions

1. **Class-based routing** - PERSON_CLASSES and VEHICLE_CLASSES sets for flexible mapping
2. **Graceful degradation** - All Hailo models fall back to CPU alternatives when unavailable
3. **Pose-based posture** - Classify posture from keypoint geometry (hip-ankle angles)
4. **HSV color analysis** - Robust to lighting variations, maps to named colors
5. **Automotive colors** - Specialized color palette for vehicle classification
6. **Contour-based shape** - OpenCV contour analysis for shape descriptors
7. **3D size estimation** - Use depth + camera intrinsics for real-world dimensions
8. **Pipeline modules** - Each pipeline has a PipelineModule wrapper for integration

### Architecture

```
Detection
    │
    └─→ PipelineRouter ──────────────────┐
             │                           │
             ├─→ PersonPipeline          │
             │    ├─→ PoseEstimator      │
             │    ├─→ ClothingAnalyzer   │
             │    └─→ FaceDetector       │
             │                           │
             ├─→ VehiclePipeline         ├─→ ObjectSchema
             │    ├─→ ColorAnalyzer      │   (attributes)
             │    ├─→ TypeClassifier     │
             │    └─→ PlateDetector      │
             │                           │
             └─→ GenericPipeline         │
                  ├─→ ImageNetClassifier │
                  ├─→ ColorAnalyzer      │
                  ├─→ ShapeAnalyzer      │
                  └─→ SizeEstimator ─────┘
```

---

## Phase 5: Persistence & Review ✅ COMPLETE

### Goals
- Persistent embedding storage with FAISS sync
- Human review system with confidence-based routing
- Active learning for model improvement

### Implemented Modules

| Module | File | Description |
|--------|------|-------------|
| EmbeddingStoreConfig | `percept/persistence/embedding_store.py` | Configuration for embedding persistence |
| EmbeddingRecord | `percept/persistence/embedding_store.py` | Record with embedding and metadata |
| EmbeddingStore | `percept/persistence/embedding_store.py` | FAISS-backed persistent storage with DB sync |
| CameraAwareEmbeddingStore | `percept/persistence/embedding_store.py` | Camera-aware matching thresholds |
| ReviewStatus | `percept/persistence/review.py` | Enum: PENDING, IN_PROGRESS, REVIEWED, SKIPPED |
| ReviewReason | `percept/persistence/review.py` | Enum: LOW_CONFIDENCE, AMBIGUOUS_CLASS, etc. |
| ReviewPriority | `percept/persistence/review.py` | Enum: LOW, NORMAL, HIGH, URGENT |
| ConfidenceConfig | `percept/persistence/review.py` | Thresholds for auto-confirm/provisional/review |
| ReviewItem | `percept/persistence/review.py` | Complete review item with alternatives |
| ReviewResult | `percept/persistence/review.py` | Result of human review |
| CropManager | `percept/persistence/review.py` | Image crop storage management |
| ConfidenceRouter | `percept/persistence/review.py` | Routes objects based on confidence |
| HumanReviewQueue | `percept/persistence/review.py` | Queue management with callbacks |
| BatchReviewer | `percept/persistence/review.py` | Batch review operations |
| FeedbackEntry | `percept/persistence/active_learning.py` | Feedback from human review |
| AccuracyMetrics | `percept/persistence/active_learning.py` | Precision, recall, confusion matrix |
| TrainingExample | `percept/persistence/active_learning.py` | Training data for model improvement |
| FeedbackCollector | `percept/persistence/active_learning.py` | Collect and persist feedback |
| AccuracyTracker | `percept/persistence/active_learning.py` | Track model accuracy over time |
| TrainingDataExporter | `percept/persistence/active_learning.py` | Export training data (JSON/CSV) |
| ActiveLearningManager | `percept/persistence/active_learning.py` | Orchestrate feedback and training |

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_persistence.py` | 70 | Embedding store, review, active learning |

### Key Design Decisions

1. **FAISS + SQLite sync** - In-memory FAISS for fast search, SQLite for persistence
2. **Confidence routing** - Auto-confirm (>0.85), provisional (0.5-0.85), review (<0.5)
3. **Camera-aware thresholds** - Different distance thresholds for same vs cross-camera
4. **Crop management** - Saves image crops for review, auto-cleanup old files
5. **Callback architecture** - Review completion triggers active learning updates
6. **Uncertainty sampling** - Prioritize low-confidence objects for review
7. **Class balancing** - Training data export with class balancing
8. **Accuracy tracking** - Rolling window accuracy with alerts

### Architecture

```
Objects
    │
    ├─→ EmbeddingStore ─→ FAISS Index ─→ SQLite Sync
    │                          │
    └─→ ConfidenceRouter ──────┤
             │                 │
             ├─→ CONFIRMED ────┤
             ├─→ PROVISIONAL ──┤
             └─→ NEEDS_REVIEW ─┼─→ HumanReviewQueue
                               │          │
                               │          ├─→ CropManager
                               │          │
                               │          └─→ ReviewResult
                               │                  │
                               └──────────────────┼─→ ActiveLearningManager
                                                  │          │
                                                  │          ├─→ FeedbackCollector
                                                  │          ├─→ AccuracyTracker
                                                  │          └─→ TrainingDataExporter
```

---

## Phase 6: Integration & Testing ✅ COMPLETE

### Goals
- Integration tests for full pipeline flow
- Cross-camera ReID tests
- Performance benchmarks for latency verification
- Hardware smoke tests for Hailo-8 and RealSense

### Test Files

| File | Tests | Description |
|------|-------|-------------|
| `tests/integration/test_pipeline_flow.py` | 21 | Detection-to-schema, segmentation, ReID, classification, database, end-to-end |
| `tests/integration/test_multi_camera.py` | 13 | Cross-camera matching, gallery management, object trajectory |
| `tests/performance/test_latency.py` | 17 | Gallery search, tracking, routing, pipelines, database latency |
| `tests/hardware/test_smoke.py` | 21 | Hailo device, RealSense camera, system resources, hardware integration |

### Key Test Scenarios

1. **Pipeline Flow Integration**
   - Detection → ObjectSchema conversion
   - Segmenter → mask creation
   - ReID gallery → embedding storage and matching
   - ByteTrack → consistent ID tracking across frames
   - Classification router → person/vehicle/generic pipelines
   - Database → object persistence and retrieval
   - Module chaining → PipelineData flow through Pipeline

2. **Cross-Camera ReID**
   - Same object matched across cameras (embedding similarity)
   - Different objects not matched (high L2 distance)
   - Camera-aware thresholds (same-camera vs cross-camera)
   - Object handoff between cameras
   - Trajectory spans cameras
   - Per-camera gallery management

3. **Performance Benchmarks**
   - Gallery search: <5ms for 1000 objects
   - Gallery add: <1ms per embedding
   - Gallery scaling: sublinear (5000 < 15x slower than 500)
   - Tracker update: <10ms for typical frame, <50ms for crowded scene
   - Router decision: <100μs per detection
   - Pipeline processing: <50ms person, <30ms vehicle, <100ms generic
   - Database save: <10ms, get: <5ms, query: <10ms
   - Full frame budget: <100ms (10 FPS), <66ms (15 FPS)

4. **Hardware Smoke Tests**
   - Hailo-8 device detection and firmware version
   - Hailo model file existence (YOLOv8, FastSAM, pose)
   - RealSense camera detection
   - System memory (≥4GB), disk space (≥5GB free)
   - Required Python packages verification

### Hardware Test Markers

```python
@hailo_required   # Skip if Hailo-8 not available
@realsense_required  # Skip if RealSense not available
@hardware  # Skip if neither hardware available
```

### Key Design Decisions

1. **Dict-based detections** - ByteTrackWrapper uses `{"bbox": tuple, "confidence": float, "class_id": int}`
2. **Realistic thresholds** - L2 distance thresholds adjusted for normalized embeddings
3. **Frame warmup** - Performance tests include warmup iterations for consistent measurements
4. **Graceful skipping** - Hardware tests skip gracefully on CI systems
5. **Statistical metrics** - Latency tests report mean, p95, p99 for reliability

---

## Phase 7: Pipeline Visualization UI ✅ COMPLETE

### Goals
- FastAPI backend with REST API and WebSocket support
- Real-time dashboard with system metrics
- Pipeline DAG visualization
- Human review interface
- Configuration editor with validation

### Implemented Modules

| Module | File | Description |
|--------|------|-------------|
| AppState | `ui/app.py` | Application lifecycle, database connection |
| DashboardData | `ui/models.py` | Real-time metrics (FPS, CPU, memory, objects) |
| PipelineGraph | `ui/models.py` | DAG structure with nodes and edges |
| WebSocketEvent | `ui/models.py` | Event types for real-time updates |
| API Routes | `ui/api/routes.py` | REST endpoints for all resources |
| WebSocket Handlers | `ui/api/websocket.py` | Stream and event connections |
| PipelineGraphBuilder | `ui/components/pipeline_graph.py` | Graph construction and layout |
| PipelineAnalyzer | `ui/components/pipeline_graph.py` | Topological sort, path analysis |
| FrameAnnotator | `ui/components/frame_viewer.py` | Detection overlay rendering |
| FrameEncoder | `ui/components/frame_viewer.py` | JPEG/PNG encoding |
| MetricCollector | `ui/components/metrics_panel.py` | Time-windowed metric aggregation |
| SystemMonitor | `ui/components/metrics_panel.py` | CPU, memory, temperature monitoring |
| FPSCounter | `ui/components/metrics_panel.py` | Frame rate calculation |
| LatencyTracker | `ui/components/metrics_panel.py` | Operation timing with context manager |

### REST API Endpoints

```
GET  /api/dashboard          # System metrics, camera status
GET  /api/metrics            # Detailed performance data
GET  /api/cameras            # List configured cameras
GET  /api/cameras/{id}/frame # Current frame (JPEG)
GET  /api/cameras/{id}/depth # Depth visualization
GET  /api/pipeline/graph     # Pipeline DAG structure
GET  /api/pipeline/stages    # List all stages
GET  /api/objects            # Paginated object list
GET  /api/objects/{id}       # Object details
GET  /api/config             # Current configuration
PUT  /api/config             # Update configuration
POST /api/config/validate    # Validate configuration
GET  /api/review             # Pending review items
POST /api/review/{id}        # Submit review
```

### WebSocket Endpoints

```
WS /ws/events   # Pipeline events (object detected, review needed, alerts)
WS /ws/stream   # Binary frame streaming
```

### Frontend Pages

| Template | Path | Description |
|----------|------|-------------|
| dashboard.html | `/` | System stats, camera feeds, alerts |
| pipeline_view.html | `/pipeline` | Interactive DAG, timing breakdown |
| object_gallery.html | `/objects` | Grid view with filtering |
| review_queue.html | `/review` | Quick classification interface |
| config_editor.html | `/config` | YAML editor with validation |

### Test Files

| File | Tests | Description |
|------|-------|-------------|
| `tests/ui/test_api.py` | 26 | REST endpoints, templates |
| `tests/ui/test_components.py` | 39 | Models, graph builder, metrics |

### Key Design Decisions

1. **FastAPI + Jinja2** - Lightweight embedded deployment, async support
2. **Fallback templates** - Inline HTML when templates directory missing
3. **WebSocket events** - Real-time updates without polling
4. **Pydantic models** - Type-safe API responses
5. **psutil metrics** - Cross-platform system monitoring
6. **Graph layout engine** - Automatic node positioning for DAG visualization

---

## Phase 8: Polish & Documentation ⏳ NOT STARTED

- Adaptive processing
- Performance optimization
- API documentation
- User guide
- Deployment instructions

---

## Development Environment

```bash
# Activate virtual environment
cd /home/sevak/apps/percept
source venv/bin/activate

# Run tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=percept --cov-report=html

# Check imports
python -c "from percept import *; print('OK')"

# Test segmentation imports
python -c "from percept.segmentation import *; print('Segmentation OK')"
```

## Hardware Reference

- **Platform:** Raspberry Pi 5 (8GB)
- **AI Accelerator:** Hailo-8 (26 TOPS)
- **Depth Camera:** Intel RealSense D415/D455
- **Hailo Models:** `/usr/share/hailo-models/`

## Git History

```
[pending] Complete Phase 7: Add Pipeline Visualization UI
[done] Complete Phase 6: Add integration and testing layer
[done] Complete Phase 5: Add persistence and review layer
4630032 Complete Phase 4: Add classification pipelines
e27b478 Complete Phase 3: Add tracking and ReID layer
6d32cfc Complete Phase 2: Add segmentation layer
14180aa Complete Phase 1: Add RealSense multi-camera capture
94580f5 Add database persistence layer with full CRUD and review queue
fe940bf Implement Phase 1 core modules: schema, config, pipeline, adapter
3285e57 Add CLAUDE.md for session context
d29118b Initial project structure with specification
d011711 Initial commit
```

---

*This document is updated at the end of each implementation session.*

# CLAUDE.md

This file provides guidance to Claude Code when working with the PERCEPT project.

## Project Overview

**PERCEPT** (Pipeline for Entity Recognition, Classification, Extraction, and Persistence Tracking) is a modular vision processing framework for mobile robots. It runs on a Raspberry Pi 5 with Hailo-8 AI accelerator and Intel RealSense depth cameras.

## Hardware Platform

This project runs on specialized hardware. Reference documentation:
- **Hardware specs:** `~/ClaudeHome/CLAUDE.md` and `~/ClaudeHome/PIRONMAN5_MAX_HARDWARE_REPORT.md`
- **Hardware reference:** `/home/sevak/apps/HARDWARE_REFERENCE.md`
- **Hailo models:** `/usr/share/hailo-models/` (HEF files for YOLOv8, FastSAM, pose, etc.)

Key hardware:
| Component | Specification |
|-----------|---------------|
| Computer | Raspberry Pi 5 (8GB) |
| AI Accelerator | Hailo-8 (26 TOPS) via M.2 |
| Depth Camera | Intel RealSense D415/D455 |
| OS | Debian 13 (Trixie) ARM64 |

## Reusable Code from hailo-agents

The `~/ClaudeHome/hailo-agents/` project contains working implementations to reuse:

| Component | Path | Purpose |
|-----------|------|---------|
| `HailoInference` | `utils/hailo_inference.py` | Model loading, preprocessing, inference |
| `FastSAMInference` | `utils/hailo_inference.py` | Segmentation with mask decoding |
| `RealSenseCapture` | `utils/realsense_capture.py` | RGB+Depth capture, alignment |
| `PointCloudProcessor` | `scanner/point_cloud_processor.py` | RANSAC, clustering |
| `ObjectTracker` | `scanner/object_tracker.py` | Fingerprint-based tracking |

Activate hailo-agents environment: `source ~/ClaudeHome/hailo-agents/activate.sh`

## Project Structure

```
percept/
├── percept/              # Main Python package
│   ├── core/             # Pipeline base classes, ObjectSchema, config
│   ├── capture/          # Multi-camera RealSense capture
│   ├── segmentation/     # FastSAM, depth discontinuity, point cloud
│   ├── tracking/         # ReID embeddings, ByteTrack, FAISS gallery
│   ├── pipelines/        # Person, vehicle, generic classification
│   ├── normalization/    # Size, light, color correction
│   ├── inference/        # Hailo-8 model execution
│   ├── persistence/      # SQLite database, embedding storage
│   └── utils/            # Geometry, visualization, metrics
├── ui/                   # FastAPI web interface
├── tests/                # pytest test suite
├── config/               # YAML configuration
├── docs/SPECIFICATION.md # Full architecture specification
└── data/                 # Database and embeddings
```

## Key Documentation

- **Full specification:** `docs/SPECIFICATION.md` - Comprehensive architecture, data structures, algorithms
- **Implementation phases:** See Section 12 of SPECIFICATION.md for phased implementation plan

## Development Commands

```bash
# Activate virtual environment (when created)
source venv/bin/activate

# Install in development mode
pip install -e .

# Run tests
pytest tests/
pytest tests/ --cov=percept --cov-report=html

# Check Hailo device
hailortcli fw-control identify

# List RealSense cameras
rs-enumerate-devices
```

## Implementation Status

**Phase 1: Foundation** - NEARLY COMPLETE (RealSense capture pending)
- [x] Project structure setup
- [x] Configuration system (`percept/core/config.py`)
- [x] Pipeline base classes and module interface (`percept/core/pipeline.py`)
- [x] Data adapter framework (`percept/core/adapter.py`)
- [x] ObjectSchema and Classification types (`percept/core/schema.py`)
- [x] Test framework setup with fixtures (`tests/conftest.py`)
- [x] Unit tests for core modules (111 tests passing)
- [x] Database schema and operations (`percept/persistence/database.py`)
- [x] Unit tests for persistence (28 tests)
- [ ] Multi-camera RealSense capture

**Phase 2-8:** Not started

### Session Log: December 24, 2025

**Completed:**
1. Implemented `percept/core/schema.py`:
   - `ObjectSchema` dataclass with full serialization (JSON roundtrip)
   - `ClassificationStatus` enum (CONFIRMED, PROVISIONAL, NEEDS_REVIEW, UNCLASSIFIED)
   - `Detection` and `ObjectMask` helper classes
   - Automatic L2 normalization of ReID embeddings

2. Implemented `percept/core/config.py`:
   - YAML-based configuration loading
   - Typed dataclasses for all config sections
   - Config validation, save/reload, hot-reload support

3. Implemented `percept/core/pipeline.py`:
   - `PipelineModule` abstract base class for hot-swappable modules
   - `Pipeline` orchestration with automatic data adaptation
   - `PipelineRegistry` for dynamic pipeline construction
   - Timing collection and intermediate output caching

4. Implemented `percept/core/adapter.py`:
   - `DataSpec` for describing data requirements
   - `PipelineData` flexible container for pipeline data
   - `DataAdapter` with image resize, color space, dtype conversions

5. Implemented `percept/persistence/database.py`:
   - `PerceptDatabase` class with full CRUD for ObjectSchemas
   - Trajectory storage and retrieval
   - Human review queue with pending/reviewed/skipped workflow
   - Embedding-specific operations for FAISS sync

6. Set up testing infrastructure:
   - Enhanced `tests/conftest.py` with fixtures for all core types
   - `MockPipelineModule` for testing pipeline behavior
   - Unit tests: `test_schema.py`, `test_config.py`, `test_pipeline.py`, `test_adapter.py`, `test_database.py`

**Test Results:** 139 tests passing

**Next Steps:**
1. Implement `percept/capture/realsense.py` - Multi-camera RealSense capture
2. Complete Phase 1
3. Begin Phase 2: Segmentation Layer

## Architecture Summary

```
Camera(s) → Segmentation → ReID/Tracking → Classification → Schema → Database
                ↓                              ↓
         FastSAM + Depth           Person / Vehicle / Generic
            Fusion                      Pipelines
```

Key concepts:
1. **ObjectSchema** - Central data structure accumulating knowledge about objects
2. **PipelineModule** - Interface for hot-swappable algorithm modules
3. **DataAdapter** - Automatic data conversion between modules
4. **ReIDMatcher** - FAISS-backed gallery for object re-identification
5. **SceneMaskManager** - Prevents duplicate processing of claimed regions

## Testing Without Hardware

Use mock components in `tests/mocks/`:
- `MockRealSenseCamera` - Returns sample frames
- `MockHailoInference` - Returns pre-computed detections

Sample fixtures in `tests/fixtures/` (to be populated with test images).

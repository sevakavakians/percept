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

**Phase 1: Foundation** - NOT STARTED
- [ ] Project structure setup ✓ (done)
- [ ] Configuration system
- [ ] Pipeline base classes and module interface
- [ ] Data adapter framework
- [ ] Multi-camera RealSense capture
- [ ] Database schema and basic operations
- [ ] Test framework setup

Next step: Implement `percept/core/` modules (schema.py, pipeline.py, adapter.py, config.py)

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

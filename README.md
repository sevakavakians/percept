# PERCEPT

**Pipeline for Entity Recognition, Classification, Extraction, and Persistence Tracking**

A modular, pipeline-based vision processing framework designed for mobile robotic platforms running on Raspberry Pi 5 with Hailo-8 AI accelerator and Intel RealSense depth cameras.

## Features

- **Modular Pipelines** - Hot-swappable algorithm modules with automatic data adaptation
- **3D-Aware Segmentation** - Fuses FastSAM, depth discontinuity, and point cloud clustering
- **ReID-Based Tracking** - Avoids redundant processing by recognizing previously seen objects
- **Progressive Object Schemas** - Accumulates knowledge about detected entities over time
- **Multi-Camera Support** - Cross-camera re-identification and tracking
- **Human Review Queue** - Async queue for uncertain classifications
- **Pipeline Visualization UI** - Real-time monitoring and configuration interface

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **Computer** | Raspberry Pi 5 (8GB) |
| **AI Accelerator** | Hailo-8 (26 TOPS) via M.2 |
| **Depth Camera** | Intel RealSense D415/D455 |
| **OS** | Debian 13 (Trixie) ARM64 |

## Installation

```bash
# Clone the repository
git clone https://github.com/sevakavakians/percept.git
cd percept

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PERCEPT in development mode
pip install -e .
```

## Quick Start

```python
from percept import PerceptPipeline

# Initialize pipeline with default configuration
pipeline = PerceptPipeline.from_config("config/percept_config.yaml")

# Process frames
with pipeline:
    while True:
        result = pipeline.process_frame()
        for obj in result.objects:
            print(f"{obj.primary_class} at {obj.distance_from_camera:.2f}m")
```

## Project Structure

```
percept/
├── percept/              # Main Python package
│   ├── core/             # Pipeline base classes, schemas, config
│   ├── capture/          # Multi-camera RealSense capture
│   ├── segmentation/     # FastSAM, depth, point cloud segmentation
│   ├── tracking/         # ReID and ByteTrack integration
│   ├── pipelines/        # Person, vehicle, generic pipelines
│   ├── normalization/    # Size, light, color correction
│   ├── inference/        # Hailo-8 model execution
│   ├── persistence/      # SQLite and FAISS storage
│   └── utils/            # Geometry, visualization, metrics
├── ui/                   # Web-based visualization UI
├── tests/                # Unit, integration, performance tests
├── config/               # Configuration files
├── data/                 # Database and embeddings
├── examples/             # Example scripts
└── docs/                 # Documentation and specification
```

## Configuration

Edit `config/percept_config.yaml` to customize:

- Camera settings (resolution, FPS, serial numbers)
- Segmentation methods and fusion parameters
- ReID thresholds and gallery settings
- Classification confidence thresholds
- Human review queue behavior
- Performance tuning (adaptive processing, target FPS)

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=percept --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run performance benchmarks
pytest tests/performance/ --benchmark-only
```

## Documentation

- [Full Specification](docs/SPECIFICATION.md) - Detailed architecture and design
- [API Reference](docs/api/) - Module documentation (coming soon)
- [Examples](examples/) - Usage examples

## Architecture Overview

```
Camera(s) → Segmentation → ReID/Tracking → Classification → Schema → Database
                ↓                              ↓
         Multi-method              Person / Vehicle / Generic
            Fusion                      Pipelines
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

Built for Raspberry Pi 5 + Hailo-8 + Intel RealSense platform.
Leverages components from [hailo-agents](https://github.com/sevakavakians/hailo-agents).

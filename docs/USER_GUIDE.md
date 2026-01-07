# PERCEPT User Guide

A comprehensive guide to using PERCEPT for vision processing on mobile robots.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Web Dashboard](#web-dashboard)
3. [Pipeline Configuration](#pipeline-configuration)
4. [Object Tracking](#object-tracking)
5. [Human Review](#human-review)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Raspberry Pi 5 (8GB recommended)
- Hailo-8 AI accelerator (M.2 or HAT)
- Intel RealSense D415 or D455 depth camera
- Python 3.10+

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/percept.git
   cd percept
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

4. **Verify hardware:**
   ```bash
   # Check Hailo device
   hailortcli fw-control identify

   # Check RealSense camera
   rs-enumerate-devices
   ```

### Quick Start

1. **Start the system:**
   ```bash
   percept start
   ```

2. **Open the dashboard:**
   Open http://localhost:8000 in your browser.

3. **View live processing:**
   The dashboard shows real-time object detection and tracking.

---

## Web Dashboard

### Overview Panel

The dashboard provides real-time system status:

- **FPS**: Current processing frame rate
- **CPU/Memory**: System resource usage
- **Objects**: Total tracked objects
- **Pending Review**: Items needing human verification

### Live View

The live view shows:

- Camera feed with detection overlays
- Bounding boxes around detected objects
- Class labels and confidence scores
- Depth visualization (toggle with button)

### Camera Controls

- **Select Camera**: Choose active camera from dropdown
- **Pause/Resume**: Temporarily pause processing
- **Snapshot**: Capture current frame
- **Depth View**: Toggle depth visualization

---

## Pipeline Configuration

### Accessing Configuration

Navigate to `/config` in the web interface or edit `config/default.yaml`.

### Key Settings

#### Cameras
```yaml
cameras:
  - id: front
    serial: "123456789"
    resolution: [640, 480]
    fps: 30
    depth_enabled: true
```

#### Segmentation
```yaml
segmentation:
  model: fastsam
  confidence_threshold: 0.5
  use_depth: true
  depth_discontinuity_threshold: 0.3
```

#### Tracking
```yaml
tracking:
  match_threshold: 0.7
  max_age: 30
  min_hits: 3
  embedding_model: osnet
```

#### Classification
```yaml
classification:
  person:
    enabled: true
    face_detection: true
    pose_estimation: true
  vehicle:
    enabled: true
    plate_detection: true
```

### Applying Changes

1. Edit configuration in the web editor
2. Click "Validate" to check for errors
3. Click "Apply" to activate changes
4. Changes take effect on next frame

---

## Object Tracking

### How It Works

PERCEPT uses a multi-stage tracking approach:

1. **Detection**: FastSAM segments objects from the scene
2. **Embedding**: ReID model generates identity embeddings
3. **Matching**: FAISS matches embeddings to existing tracks
4. **Update**: Track states updated via ByteTrack

### Viewing Tracks

Navigate to `/objects` to see all tracked objects:

- **Object Gallery**: Thumbnails of all detected objects
- **Timeline**: When objects were seen
- **Attributes**: Extracted characteristics
- **Trajectory**: Movement path on map

### Object Details

Click an object to view:

- Full-resolution images
- Classification history
- Re-identification matches
- Attribute details

### Search and Filter

Use the filter panel to:

- Filter by class (person, vehicle, etc.)
- Filter by time range
- Filter by confidence level
- Search by attributes

---

## Human Review

### Review Queue

Navigate to `/review` for items needing verification:

- **Low Confidence**: Detections below threshold
- **Classification Conflict**: Inconsistent classifications
- **Re-ID Uncertainty**: Potential duplicate objects

### Review Actions

For each item:

1. **Confirm**: Accept the classification as correct
2. **Reject**: Mark as false positive (remove from database)
3. **Reclassify**: Change to a different class
4. **Merge**: Combine with another object (duplicates)

### Keyboard Shortcuts

- `C`: Confirm current item
- `R`: Reject current item
- `→`: Next item
- `←`: Previous item
- `1-9`: Quick reclassify to class N

---

## Performance Tuning

### Adaptive Processing

PERCEPT automatically adjusts processing based on load:

| Mode | Features | Target FPS |
|------|----------|------------|
| FULL | All features enabled | 20+ FPS |
| BALANCED | Core features only | 15+ FPS |
| FAST | Detection + tracking | 10+ FPS |
| MINIMAL | Detection only | 5+ FPS |

### Manual Optimization

#### Reduce Resolution
```yaml
cameras:
  - resolution: [640, 480]  # Instead of 1280x720
```

#### Disable Optional Features
```yaml
segmentation:
  use_depth: false  # Faster without depth fusion

classification:
  person:
    face_detection: false  # Skip face detection
    pose_estimation: false  # Skip pose estimation
```

#### Adjust Thresholds
```yaml
segmentation:
  confidence_threshold: 0.6  # Higher = fewer detections

tracking:
  max_age: 15  # Shorter track lifetime
```

### Monitoring Performance

View performance metrics at `/api/metrics`:

```json
{
  "segmentation_latency": 35.2,
  "tracking_latency": 8.5,
  "classification_latency": 12.3,
  "total_latency": 58.0
}
```

### Profiling

Enable detailed profiling:

```python
from percept.utils.profiler import PipelineProfiler

profiler = PipelineProfiler()

# After processing
print(profiler.get_report())
```

---

## Troubleshooting

### Common Issues

#### Hailo Device Not Found

```
Error: Failed to initialize Hailo device
```

**Solution:**
1. Check physical connection
2. Run `hailortcli fw-control identify`
3. Check kernel module: `lsmod | grep hailo`
4. Reload driver: `sudo modprobe hailo_pci`

#### RealSense Camera Not Detected

```
Error: No RealSense devices found
```

**Solution:**
1. Check USB connection (use USB 3.0 port)
2. Run `rs-enumerate-devices`
3. Check permissions: `sudo chmod 666 /dev/bus/usb/*/*`
4. Install udev rules: `sudo cp librealsense2/config/99-realsense-libusb.rules /etc/udev/rules.d/`

#### Low Frame Rate

**Symptoms:**
- FPS below 10
- Laggy video feed
- High CPU usage

**Solutions:**
1. Lower resolution in config
2. Disable optional features
3. Check thermal throttling: `vcgencmd measure_temp`
4. Ensure adequate cooling

#### Objects Not Tracked

**Symptoms:**
- Same object gets multiple IDs
- Tracks lost frequently

**Solutions:**
1. Lower `match_threshold` (default 0.7)
2. Increase `max_age` for longer track persistence
3. Check lighting conditions
4. Verify ReID model loaded correctly

#### High Memory Usage

**Symptoms:**
- System slowdown
- Out of memory errors

**Solutions:**
1. Reduce `gallery_size` in tracking config
2. Lower `history_size` in adaptive config
3. Enable embedding compression
4. Restart periodically to clear accumulated data

### Logs

View system logs:

```bash
# Application logs
tail -f logs/percept.log

# Hailo runtime logs
dmesg | grep hailo

# RealSense logs
export RS2_LOG_LEVEL=debug
```

### Debug Mode

Enable debug mode for verbose output:

```bash
PERCEPT_DEBUG=1 percept start
```

Or in config:
```yaml
logging:
  level: DEBUG
  console: true
  file: logs/debug.log
```

### Getting Help

- Check documentation: `docs/SPECIFICATION.md`
- Search issues: https://github.com/yourusername/percept/issues
- API reference: `docs/API.md`

---

## Appendix

### Supported Object Classes

| Class | Subclasses |
|-------|------------|
| person | adult, child |
| vehicle | car, truck, motorcycle, bicycle |
| animal | dog, cat, bird |
| object | bag, bottle, phone |

### Keyboard Shortcuts Reference

| Key | Action |
|-----|--------|
| Space | Pause/Resume |
| D | Toggle depth view |
| F | Toggle fullscreen |
| S | Take snapshot |
| ? | Show help |

### Configuration Schema

See `docs/SPECIFICATION.md` Section 5 for complete configuration schema.

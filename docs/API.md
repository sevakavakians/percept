# PERCEPT API Reference

This document describes the REST API and WebSocket interfaces provided by PERCEPT's web UI.

## Base URL

```
http://localhost:8080
```

## Authentication

Currently, PERCEPT does not require authentication. Future versions may add API key support.

---

## Health Endpoints

### GET /health

Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600.5,
  "database_connected": true
}
```

---

## Dashboard API

### GET /api/dashboard

Get dashboard overview data.

**Response:**
```json
{
  "fps": 15.2,
  "cpu_usage": 45.5,
  "memory_usage": 62.3,
  "memory_used_mb": 1024.5,
  "memory_total_mb": 8192.0,
  "total_objects": 1250,
  "pending_review": 12,
  "cameras": [
    {
      "id": "cam-001",
      "name": "Front Camera",
      "connected": true,
      "fps": 30.0,
      "resolution": "640x480"
    }
  ],
  "alerts": [],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /api/metrics

Get detailed processing metrics.

**Response:**
```json
{
  "segmentation_latency": 35.2,
  "tracking_latency": 8.5,
  "classification_latency": 12.3,
  "total_latency": 58.0,
  "frames_processed": 15000,
  "objects_detected": 45000,
  "uptime_seconds": 3600.5
}
```

---

## Camera API

### GET /api/cameras

List all connected cameras.

**Response:**
```json
[
  {
    "id": "cam-001",
    "name": "Front Camera",
    "connected": true,
    "fps": 30.0,
    "resolution": "640x480",
    "type": "realsense"
  }
]
```

### GET /api/cameras/{camera_id}/frame

Get current frame as JPEG image.

**Parameters:**
- `camera_id` (path): Camera identifier

**Response:**
- Content-Type: `image/jpeg`
- Body: JPEG image data

### GET /api/cameras/{camera_id}/depth

Get current depth frame as colorized JPEG.

**Parameters:**
- `camera_id` (path): Camera identifier

**Response:**
- Content-Type: `image/jpeg`
- Body: Colorized depth image

---

## Pipeline API

### GET /api/pipeline/graph

Get pipeline DAG structure for visualization.

**Response:**
```json
{
  "name": "main",
  "nodes": [
    {
      "id": "capture",
      "type": "input",
      "label": "Camera Capture",
      "config": {}
    },
    {
      "id": "segment",
      "type": "process",
      "label": "FastSAM Segmentation",
      "config": {}
    }
  ],
  "edges": [
    {
      "source": "capture",
      "target": "segment",
      "label": ""
    }
  ]
}
```

### GET /api/pipeline/stages

List all pipeline stages with status.

**Response:**
```json
[
  {
    "id": "capture",
    "name": "Camera Capture",
    "enabled": true,
    "avg_latency_ms": 5.2
  },
  {
    "id": "segment",
    "name": "FastSAM Segmentation",
    "enabled": true,
    "avg_latency_ms": 35.0
  }
]
```

### GET /api/pipeline/stages/{stage_id}/output

Get output from a specific pipeline stage.

**Parameters:**
- `stage_id` (path): Stage identifier
- `frame_id` (query, optional): Specific frame ID

**Response:**
```json
{
  "stage_id": "segment",
  "frame_id": 1234,
  "timing_ms": 35.2,
  "output": {
    "masks": 5,
    "boxes": [...]
  }
}
```

---

## Objects API

### GET /api/objects

List detected objects with pagination.

**Parameters:**
- `page` (query, default: 1): Page number
- `page_size` (query, default: 20): Items per page
- `class_filter` (query, optional): Filter by class name
- `confidence_min` (query, optional): Minimum confidence
- `sort_by` (query, default: "last_seen"): Sort field

**Response:**
```json
{
  "items": [
    {
      "id": "obj-001",
      "primary_class": "person",
      "confidence": 0.92,
      "first_seen": "2024-01-15T10:00:00Z",
      "last_seen": "2024-01-15T10:30:00Z",
      "sighting_count": 15,
      "thumbnail_url": "/api/objects/obj-001/thumbnail"
    }
  ],
  "total": 1250,
  "page": 1,
  "page_size": 20,
  "pages": 63
}
```

### GET /api/objects/{object_id}

Get detailed object information.

**Parameters:**
- `object_id` (path): Object identifier

**Response:**
```json
{
  "id": "obj-001",
  "primary_class": "person",
  "confidence": 0.92,
  "first_seen": "2024-01-15T10:00:00Z",
  "last_seen": "2024-01-15T10:30:00Z",
  "sighting_count": 15,
  "attributes": {
    "gender": "male",
    "age_group": "adult",
    "clothing_upper": "blue_shirt"
  },
  "trajectory": [
    {"x": 100, "y": 200, "timestamp": "..."},
    {"x": 150, "y": 220, "timestamp": "..."}
  ],
  "embedding": null
}
```

### GET /api/objects/{object_id}/thumbnail

Get object thumbnail image.

**Response:**
- Content-Type: `image/jpeg`
- Body: Cropped object image

---

## Review API

### GET /api/review

Get items pending human review.

**Parameters:**
- `limit` (query, default: 20): Maximum items to return
- `reason` (query, optional): Filter by review reason

**Response:**
```json
[
  {
    "id": "review-001",
    "object_id": "obj-123",
    "primary_class": "person",
    "confidence": 0.45,
    "reason": "low_confidence",
    "created_at": "2024-01-15T10:25:00Z",
    "thumbnail_url": "/api/objects/obj-123/thumbnail"
  }
]
```

### POST /api/review/{review_id}

Submit review decision.

**Parameters:**
- `review_id` (path): Review item identifier

**Request Body:**
```json
{
  "action": "confirm",
  "class_override": null,
  "notes": "Clearly visible person"
}
```

**Actions:**
- `confirm`: Accept the classification
- `reject`: Mark as false positive
- `reclassify`: Change the class (requires `class_override`)

**Response:**
```json
{
  "success": true,
  "object_id": "obj-123",
  "action": "confirm"
}
```

---

## Configuration API

### GET /api/config

Get current system configuration.

**Response:**
```json
{
  "cameras": [...],
  "segmentation": {
    "model": "fastsam",
    "confidence_threshold": 0.5
  },
  "tracking": {
    "match_threshold": 0.7,
    "max_age": 30
  }
}
```

### PUT /api/config

Update system configuration.

**Request Body:**
```json
{
  "tracking": {
    "match_threshold": 0.8
  }
}
```

**Response:**
```json
{
  "success": true,
  "config": {...}
}
```

### POST /api/config/validate

Validate configuration without applying.

**Request Body:**
```json
{
  "config": {
    "tracking": {
      "match_threshold": 2.0
    }
  }
}
```

**Response:**
```json
{
  "valid": false,
  "errors": [
    {
      "path": "tracking.match_threshold",
      "message": "Value must be between 0 and 1"
    }
  ],
  "warnings": []
}
```

---

## WebSocket API

### WS /ws/events

Real-time event stream.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/events');
```

**Event Types:**

#### frame_processed
```json
{
  "event": "frame_processed",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "data": {
    "frame_id": 1234,
    "objects_detected": 5,
    "processing_time_ms": 58.2
  }
}
```

#### object_detected
```json
{
  "event": "object_detected",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "data": {
    "object_id": "obj-001",
    "class": "person",
    "confidence": 0.92,
    "bbox": [100, 200, 150, 300],
    "is_new": true
  }
}
```

#### review_needed
```json
{
  "event": "review_needed",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "data": {
    "review_id": "review-001",
    "object_id": "obj-123",
    "reason": "low_confidence"
  }
}
```

#### alert
```json
{
  "event": "alert",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "data": {
    "level": "warning",
    "message": "High CPU usage detected",
    "details": {"cpu_percent": 95.5}
  }
}
```

### WS /ws/stream

Binary frame stream for live video.

**Message Format:**
- Binary: Raw JPEG frame data
- Text: JSON control messages

**Control Messages:**
```json
{"action": "subscribe", "camera_id": "cam-001"}
{"action": "unsubscribe", "camera_id": "cam-001"}
{"action": "set_quality", "quality": 80}
```

---

## Error Responses

All endpoints may return error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid parameter value",
  "errors": [{"field": "page", "message": "Must be positive integer"}]
}
```

### 404 Not Found
```json
{
  "detail": "Object not found",
  "object_id": "obj-999"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Database not connected"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error",
  "trace_id": "abc123"
}
```

---

## Rate Limiting

Currently no rate limiting is applied. For production deployments, consider using a reverse proxy with rate limiting.

## CORS

CORS is enabled for all origins by default. Configure allowed origins in production via environment variables.

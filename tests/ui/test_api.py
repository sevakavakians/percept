"""Tests for PERCEPT UI REST API.

Tests verify API endpoints return correct data structures
and handle errors appropriately.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Skip if FastAPI not available
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def app():
    """Create test application."""
    from ui.app import app, app_state

    # Initialize without real database
    app_state.database = None
    app_state.config = None

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "timestamp" in data

    def test_health_includes_database_status(self, client):
        """Health endpoint includes database connection status."""
        response = client.get("/health")
        data = response.json()

        assert "database_connected" in data


# =============================================================================
# Dashboard API Tests
# =============================================================================

class TestDashboardAPI:
    """Test dashboard data endpoint."""

    def test_dashboard_returns_data(self, client):
        """Dashboard endpoint returns expected structure."""
        response = client.get("/api/dashboard")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "fps" in data
        assert "cpu_usage" in data
        assert "memory_usage" in data
        assert "total_objects" in data
        assert "pending_review" in data
        assert "cameras" in data
        assert "timestamp" in data

    def test_dashboard_system_metrics(self, client):
        """Dashboard includes system metrics."""
        response = client.get("/api/dashboard")
        data = response.json()

        assert 0 <= data["cpu_usage"] <= 100
        assert 0 <= data["memory_usage"] <= 100
        assert data["memory_used_mb"] >= 0
        assert data["memory_total_mb"] > 0

    def test_metrics_endpoint(self, client):
        """Metrics endpoint returns detailed data."""
        response = client.get("/api/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "segmentation_latency" in data
        assert "tracking_latency" in data
        assert "uptime_seconds" in data


# =============================================================================
# Camera API Tests
# =============================================================================

class TestCameraAPI:
    """Test camera endpoints."""

    def test_list_cameras(self, client):
        """List cameras endpoint returns array."""
        response = client.get("/api/cameras")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    def test_camera_frame(self, client):
        """Camera frame endpoint returns image."""
        response = client.get("/api/cameras/test-cam/frame")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 0

    def test_camera_depth(self, client):
        """Camera depth endpoint returns image."""
        response = client.get("/api/cameras/test-cam/depth")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"


# =============================================================================
# Pipeline API Tests
# =============================================================================

class TestPipelineAPI:
    """Test pipeline endpoints."""

    def test_get_pipeline_graph(self, client):
        """Pipeline graph endpoint returns DAG structure."""
        response = client.get("/api/pipeline/graph")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0
        assert len(data["edges"]) > 0

    def test_pipeline_nodes_have_required_fields(self, client):
        """Pipeline nodes have all required fields."""
        response = client.get("/api/pipeline/graph")
        data = response.json()

        for node in data["nodes"]:
            assert "id" in node
            assert "type" in node
            assert "label" in node

    def test_pipeline_edges_reference_valid_nodes(self, client):
        """Pipeline edges reference existing nodes."""
        response = client.get("/api/pipeline/graph")
        data = response.json()

        node_ids = {n["id"] for n in data["nodes"]}

        for edge in data["edges"]:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids

    def test_list_pipeline_stages(self, client):
        """List stages endpoint returns array."""
        response = client.get("/api/pipeline/stages")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_stage_output(self, client):
        """Stage output endpoint returns data."""
        response = client.get("/api/pipeline/stages/segment/output?frame_id=0")

        assert response.status_code == 200
        data = response.json()

        assert "stage_id" in data
        assert "frame_id" in data
        assert "timing_ms" in data


# =============================================================================
# Objects API Tests
# =============================================================================

class TestObjectsAPI:
    """Test objects endpoints."""

    def test_list_objects(self, client):
        """List objects returns paginated response."""
        response = client.get("/api/objects")

        assert response.status_code == 200
        data = response.json()

        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "pages" in data

    def test_list_objects_pagination(self, client):
        """Objects list supports pagination."""
        response = client.get("/api/objects?page=1&page_size=10")

        assert response.status_code == 200
        data = response.json()

        assert data["page"] == 1
        assert data["page_size"] == 10

    def test_get_object_not_found(self, client):
        """Getting nonexistent object returns 404."""
        response = client.get("/api/objects/nonexistent-id")

        # Without database, returns 503
        assert response.status_code in [404, 503]


# =============================================================================
# Configuration API Tests
# =============================================================================

class TestConfigAPI:
    """Test configuration endpoints."""

    def test_get_config(self, client):
        """Get config returns current configuration."""
        response = client.get("/api/config")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)

    def test_validate_config(self, client):
        """Validate config endpoint checks configuration."""
        response = client.post(
            "/api/config/validate",
            json={"config": {"cameras": []}}
        )

        assert response.status_code == 200
        data = response.json()

        assert "valid" in data
        assert "errors" in data
        assert "warnings" in data

    def test_validate_config_catches_errors(self, client):
        """Validation catches invalid thresholds."""
        response = client.post(
            "/api/config/validate",
            json={"config": {"tracking": {"match_threshold": 2.0}}}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is False
        assert len(data["errors"]) > 0


# =============================================================================
# Review API Tests
# =============================================================================

class TestReviewAPI:
    """Test review queue endpoints."""

    def test_get_review_queue(self, client):
        """Get review queue returns list."""
        response = client.get("/api/review")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    def test_review_queue_limit(self, client):
        """Review queue respects limit parameter."""
        response = client.get("/api/review?limit=5")

        assert response.status_code == 200
        data = response.json()

        assert len(data) <= 5


# =============================================================================
# Template Tests
# =============================================================================

class TestTemplates:
    """Test HTML template rendering."""

    def test_dashboard_page(self, client):
        """Dashboard page renders."""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "PERCEPT" in response.text

    def test_pipeline_page(self, client):
        """Pipeline page renders."""
        response = client.get("/pipeline")

        assert response.status_code == 200
        assert "Pipeline" in response.text

    def test_config_page(self, client):
        """Config page renders."""
        response = client.get("/config")

        assert response.status_code == 200
        assert "Config" in response.text

    def test_review_page(self, client):
        """Review page renders."""
        response = client.get("/review")

        assert response.status_code == 200
        assert "Review" in response.text

    def test_objects_page(self, client):
        """Objects page renders."""
        response = client.get("/objects")

        assert response.status_code == 200
        assert "Object" in response.text

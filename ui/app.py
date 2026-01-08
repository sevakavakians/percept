"""Main FastAPI application for PERCEPT UI.

Provides REST API endpoints, WebSocket streaming, and serves
the frontend dashboard for monitoring and configuration.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from percept.core.config import PerceptConfig

# Import stage visualization components
from ui.components.stage_frames import stage_buffer
from ui.components.stage_visualizers import (
    visualize_capture,
    visualize_segmentation,
    visualize_depth,
    visualize_depth_edges,
    visualize_depth_objects,
    visualize_depth_clusters,
)

# Import segmentation
from percept.segmentation import FastSAMSegmenter, FastSAMConfig, is_hailo_available
from percept.core.adapter import PipelineData

# Import routers
from ui.api.routes import router as api_router
from ui.api.websocket import router as ws_router

# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Global application state."""

    def __init__(self):
        self.config: Optional[PerceptConfig] = None
        self.database = None
        self.pipeline = None
        self.segmenter: Optional[FastSAMSegmenter] = None
        self.cameras: Dict[str, Any] = {}  # camera_id -> RealSenseCamera
        self.start_time: datetime = datetime.now()
        self.frame_count: int = 0
        self.connected_clients: set = set()
        self._lock = asyncio.Lock()

    async def initialize(self, config_path: Optional[str] = None):
        """Initialize application state."""
        async with self._lock:
            # Load configuration
            if config_path and Path(config_path).exists():
                self.config = PerceptConfig.load(config_path)
            else:
                self.config = PerceptConfig()

            # Initialize database connection
            try:
                from percept.persistence.database import PerceptDatabase
                db_path = self.config.database.get_absolute_path()
                db_path.parent.mkdir(parents=True, exist_ok=True)
                self.database = PerceptDatabase(str(db_path))
                self.database.initialize()
            except Exception as e:
                print(f"Warning: Could not initialize database: {e}")
                self.database = None

            # Initialize cameras
            try:
                from percept.capture.realsense import RealSenseCamera, REALSENSE_AVAILABLE
                if REALSENSE_AVAILABLE and self.config.cameras:
                    for cam_cfg in self.config.get_enabled_cameras():
                        try:
                            camera = RealSenseCamera.from_config(cam_cfg)
                            camera.start()
                            self.cameras[cam_cfg.id] = camera
                            print(f"Camera '{cam_cfg.id}' started successfully")
                        except Exception as e:
                            print(f"Warning: Could not start camera '{cam_cfg.id}': {e}")
                else:
                    print("RealSense not available or no cameras configured")
            except Exception as e:
                print(f"Warning: Could not initialize cameras: {e}")

            # Initialize segmentation pipeline
            try:
                if is_hailo_available():
                    self.segmenter = FastSAMSegmenter(FastSAMConfig())
                    self.segmenter.initialize()
                    print("FastSAM segmenter initialized with Hailo-8")
                else:
                    print("Hailo not available - segmentation disabled")
            except Exception as e:
                print(f"Warning: Could not initialize segmenter: {e}")
                self.segmenter = None

    async def shutdown(self):
        """Cleanup on shutdown."""
        async with self._lock:
            # Stop cameras
            for camera_id, camera in self.cameras.items():
                try:
                    camera.stop()
                    print(f"Camera '{camera_id}' stopped")
                except Exception as e:
                    print(f"Warning: Error stopping camera '{camera_id}': {e}")
            self.cameras.clear()

            # Cleanup segmenter
            if self.segmenter:
                try:
                    self.segmenter.cleanup()
                    print("Segmenter cleaned up")
                except Exception as e:
                    print(f"Warning: Error cleaning up segmenter: {e}")
                self.segmenter = None

            if self.database:
                self.database.close()

    @property
    def uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()


# Global state instance
app_state = AppState()


# =============================================================================
# Pipeline Processing Loop
# =============================================================================

async def run_pipeline_loop(interval_ms: float = 66):
    """Background task: capture frames and update stage visualizations.

    This runs continuously, capturing frames from cameras and updating
    the stage buffer with visualized outputs for streaming.

    Args:
        interval_ms: Target interval between frames (~15 FPS default)
    """
    frame_id = 0

    while True:
        try:
            for camera_id, camera in app_state.cameras.items():
                try:
                    # Capture frame
                    frame_data = camera.capture(timeout_ms=100)

                    if frame_data is None or frame_data.color is None:
                        continue

                    frame_id += 1
                    app_state.frame_count = frame_id

                    # Update capture stage visualization
                    capture_vis = visualize_capture(
                        frame=frame_data.color,
                        camera_id=camera_id,
                        frame_id=frame_id,
                        fps=frame_id / max(1, app_state.uptime_seconds),
                    )
                    stage_buffer.update("capture", capture_vis)

                    # Run segmentation if available
                    if app_state.segmenter is not None:
                        try:
                            # Create pipeline data from frame
                            pipeline_data = PipelineData(
                                image=frame_data.color,
                                depth=frame_data.depth,
                            )

                            # Run segmentation
                            seg_result = app_state.segmenter.process(pipeline_data)
                            masks = seg_result.get("masks", [])

                            # Visualize segmentation
                            seg_vis = visualize_segmentation(
                                frame=frame_data.color,
                                masks=masks,
                                alpha=0.4,
                            )
                            stage_buffer.update("segment", seg_vis)

                        except Exception as e:
                            # Log segmentation errors but continue
                            pass

                    # Run depth pipeline (CPU-based, no Hailo needed)
                    if frame_data.depth is not None:
                        try:
                            from percept.segmentation.depth_seg import (
                                DepthEdgeDetector,
                                DepthConnectedComponents,
                                DepthSegmentationConfig,
                            )

                            # Stage 1: Colorized depth visualization
                            depth_vis = visualize_depth(frame_data.depth)
                            stage_buffer.update("depth", depth_vis)

                            # Stage 2: Depth edge detection
                            config = DepthSegmentationConfig()
                            detector = DepthEdgeDetector(config)
                            edges = detector.detect_edges(frame_data.depth)
                            edges_vis = visualize_depth_edges(frame_data.depth, edges)
                            stage_buffer.update("depth_edges", edges_vis)

                            # Stage 3: Connected components segmentation
                            cc = DepthConnectedComponents(config)
                            labels, num_labels = cc.segment(frame_data.depth, edges)
                            objects_vis = visualize_depth_objects(
                                frame_data.depth, labels, num_labels
                            )
                            stage_buffer.update("depth_objects", objects_vis)

                            # Stage 4: 3D clusters visualization (placeholder for now)
                            clusters_vis = visualize_depth_clusters(
                                frame_data.color, frame_data.depth, []
                            )
                            stage_buffer.update("depth_clusters", clusters_vis)

                        except Exception as e:
                            # Log depth processing errors but continue
                            pass

                    # TODO: Run tracking and update "tracking" stage
                    # TODO: Run ReID and update "reid" stage
                    # TODO: Run classification pipelines

                except Exception as e:
                    # Log but don't crash on individual frame errors
                    pass

        except Exception as e:
            print(f"Pipeline loop error: {e}")

        # Control frame rate
        await asyncio.sleep(interval_ms / 1000.0)


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    config_path = os.environ.get("PERCEPT_CONFIG", "config/percept_config.yaml")
    await app_state.initialize(config_path)
    print(f"PERCEPT UI started at {app_state.start_time}")

    # Start background tasks
    from ui.api.websocket import start_metrics_broadcast
    metrics_task = asyncio.create_task(start_metrics_broadcast())
    pipeline_task = asyncio.create_task(run_pipeline_loop())

    yield

    # Shutdown
    metrics_task.cancel()
    pipeline_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    try:
        await pipeline_task
    except asyncio.CancelledError:
        pass
    await app_state.shutdown()
    print("PERCEPT UI shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="PERCEPT UI",
    description="Pipeline Visualization and Control Interface for PERCEPT",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - configurable via environment variable
# Set PERCEPT_CORS_ORIGINS to comma-separated list of allowed origins
# Default allows localhost origins for development
cors_origins_env = os.environ.get("PERCEPT_CORS_ORIGINS", "")
if cors_origins_env:
    cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]
else:
    # Default: allow common localhost origins for development
    cors_origins = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir)) if templates_dir.exists() else None

# Include routers
app.include_router(api_router, prefix="/api", tags=["api"])
app.include_router(ws_router, prefix="/ws", tags=["websocket"])


# =============================================================================
# Template Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render main dashboard page."""
    if templates:
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "title": "PERCEPT Dashboard"}
        )
    return HTMLResponse(content=get_fallback_html("Dashboard"), status_code=200)


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_view(request: Request):
    """Render pipeline visualization page."""
    if templates:
        return templates.TemplateResponse(
            "pipeline_view.html",
            {"request": request, "title": "Pipeline View"}
        )
    return HTMLResponse(content=get_fallback_html("Pipeline View"), status_code=200)


@app.get("/config", response_class=HTMLResponse)
async def config_editor(request: Request):
    """Render configuration editor page."""
    if templates:
        return templates.TemplateResponse(
            "config_editor.html",
            {"request": request, "title": "Configuration Editor"}
        )
    return HTMLResponse(content=get_fallback_html("Config Editor"), status_code=200)


@app.get("/review", response_class=HTMLResponse)
async def review_queue(request: Request):
    """Render human review queue page."""
    if templates:
        return templates.TemplateResponse(
            "review_queue.html",
            {"request": request, "title": "Review Queue"}
        )
    return HTMLResponse(content=get_fallback_html("Review Queue"), status_code=200)


@app.get("/objects", response_class=HTMLResponse)
async def object_gallery(request: Request):
    """Render object gallery page."""
    if templates:
        return templates.TemplateResponse(
            "object_gallery.html",
            {"request": request, "title": "Object Gallery"}
        )
    return HTMLResponse(content=get_fallback_html("Object Gallery"), status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "uptime_seconds": app_state.uptime_seconds,
        "database_connected": app_state.database is not None,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# Fallback HTML (when templates not available)
# =============================================================================

def get_fallback_html(page_name: str) -> str:
    """Generate fallback HTML when templates not available."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PERCEPT - {page_name}</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #1a1a2e;
                color: #eee;
                min-height: 100vh;
            }}
            .header {{
                background: #16213e;
                padding: 1rem 2rem;
                border-bottom: 1px solid #0f3460;
            }}
            .header h1 {{
                font-size: 1.5rem;
                color: #e94560;
            }}
            .nav {{
                display: flex;
                gap: 1rem;
                margin-top: 0.5rem;
            }}
            .nav a {{
                color: #94a3b8;
                text-decoration: none;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                transition: all 0.2s;
            }}
            .nav a:hover, .nav a.active {{
                background: #0f3460;
                color: #fff;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem;
            }}
            .card {{
                background: #16213e;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                border: 1px solid #0f3460;
            }}
            .card h2 {{
                color: #e94560;
                margin-bottom: 1rem;
                font-size: 1.2rem;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
            }}
            .stat {{
                text-align: center;
                padding: 1rem;
            }}
            .stat-value {{
                font-size: 2rem;
                font-weight: bold;
                color: #e94560;
            }}
            .stat-label {{
                color: #94a3b8;
                font-size: 0.9rem;
            }}
            #metrics {{ font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>PERCEPT</h1>
            <nav class="nav">
                <a href="/" class="{'active' if page_name == 'Dashboard' else ''}">Dashboard</a>
                <a href="/pipeline" class="{'active' if page_name == 'Pipeline View' else ''}">Pipeline</a>
                <a href="/config" class="{'active' if page_name == 'Config Editor' else ''}">Config</a>
                <a href="/review" class="{'active' if page_name == 'Review Queue' else ''}">Review</a>
                <a href="/objects" class="{'active' if page_name == 'Object Gallery' else ''}">Objects</a>
            </nav>
        </div>
        <div class="container">
            <div class="card">
                <h2>{page_name}</h2>
                <div class="stats-grid">
                    <div class="stat">
                        <div class="stat-value" id="fps">--</div>
                        <div class="stat-label">FPS</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="objects">--</div>
                        <div class="stat-label">Objects</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="pending">--</div>
                        <div class="stat-label">Pending Review</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="uptime">--</div>
                        <div class="stat-label">Uptime</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2>Live Metrics</h2>
                <pre id="metrics">Loading...</pre>
            </div>
        </div>
        <script>
            async function updateDashboard() {{
                try {{
                    const resp = await fetch('/api/dashboard');
                    const data = await resp.json();
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('objects').textContent = data.total_objects;
                    document.getElementById('pending').textContent = data.pending_review;
                    const uptime = Math.floor(data.timestamp ?
                        (Date.now() - new Date(data.timestamp).getTime()) / 1000 : 0);
                    document.getElementById('uptime').textContent = formatUptime(uptime);
                    document.getElementById('metrics').textContent = JSON.stringify(data, null, 2);
                }} catch (e) {{
                    document.getElementById('metrics').textContent = 'Error loading metrics';
                }}
            }}

            function formatUptime(seconds) {{
                const h = Math.floor(seconds / 3600);
                const m = Math.floor((seconds % 3600) / 60);
                const s = seconds % 60;
                return `${{h}}h ${{m}}m`;
            }}

            updateDashboard();
            setInterval(updateDashboard, 2000);
        </script>
    </body>
    </html>
    """


# =============================================================================
# Dependency Injection
# =============================================================================

def get_app_state() -> AppState:
    """Get application state for dependency injection."""
    return app_state


def get_database():
    """Get database instance."""
    return app_state.database


def get_config():
    """Get configuration instance."""
    return app_state.config


# =============================================================================
# Main Entry Point
# =============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
):
    """Run the PERCEPT UI server."""
    import uvicorn
    uvicorn.run(
        "ui.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server(reload=True)

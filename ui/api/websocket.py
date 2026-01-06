"""WebSocket handlers for PERCEPT UI.

Provides real-time streaming of frames, events, and metrics
to connected clients.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ui.models import (
    EventType,
    FrameProcessedEvent,
    ObjectDetectedEvent,
    ObjectUpdatedEvent,
    ReviewNeededEvent,
    WebSocketEvent,
)

router = APIRouter()


# =============================================================================
# Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.stream_connections: Set[WebSocket] = set()
        self.event_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, connection_type: str = "events"):
        """Accept and register a new connection."""
        await websocket.accept()
        self.active_connections.add(websocket)

        if connection_type == "stream":
            self.stream_connections.add(websocket)
        else:
            self.event_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a connection."""
        self.active_connections.discard(websocket)
        self.stream_connections.discard(websocket)
        self.event_connections.discard(websocket)

    async def send_event(self, websocket: WebSocket, event: WebSocketEvent):
        """Send event to a specific connection."""
        try:
            await websocket.send_json(event.model_dump(mode='json'))
        except Exception:
            self.disconnect(websocket)

    async def broadcast_event(self, event: WebSocketEvent):
        """Broadcast event to all event connections."""
        disconnected = set()
        for connection in self.event_connections:
            try:
                await connection.send_json(event.model_dump(mode='json'))
            except Exception:
                disconnected.add(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def send_frame(self, websocket: WebSocket, frame_data: bytes):
        """Send frame data to a specific connection."""
        try:
            await websocket.send_bytes(frame_data)
        except Exception:
            self.disconnect(websocket)

    async def broadcast_frame(self, frame_data: bytes):
        """Broadcast frame to all stream connections."""
        disconnected = set()
        for connection in self.stream_connections:
            try:
                await connection.send_bytes(frame_data)
            except Exception:
                disconnected.add(connection)

        for conn in disconnected:
            self.disconnect(conn)

    @property
    def connection_count(self) -> int:
        """Get total connection count."""
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()


# =============================================================================
# Event Broadcaster
# =============================================================================

class EventBroadcaster:
    """Broadcasts pipeline events to connected clients."""

    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager

    async def frame_processed(
        self,
        frame_id: int,
        camera_id: str,
        objects_detected: int,
        processing_time_ms: float,
    ):
        """Broadcast frame processed event."""
        event = WebSocketEvent(
            event=EventType.FRAME_PROCESSED,
            data=FrameProcessedEvent(
                frame_id=frame_id,
                camera_id=camera_id,
                objects_detected=objects_detected,
                processing_time_ms=processing_time_ms,
            ).model_dump(),
        )
        await self.manager.broadcast_event(event)

    async def object_detected(
        self,
        object_id: str,
        primary_class: str,
        confidence: float,
        camera_id: str,
        position: tuple = None,
    ):
        """Broadcast object detected event."""
        event = WebSocketEvent(
            event=EventType.OBJECT_DETECTED,
            data=ObjectDetectedEvent(
                object_id=object_id,
                primary_class=primary_class,
                confidence=confidence,
                position=position,
                camera_id=camera_id,
            ).model_dump(),
        )
        await self.manager.broadcast_event(event)

    async def object_updated(
        self,
        object_id: str,
        camera_id: str,
        was_matched: bool,
        confidence: float,
    ):
        """Broadcast object updated event."""
        event = WebSocketEvent(
            event=EventType.OBJECT_UPDATED,
            data=ObjectUpdatedEvent(
                object_id=object_id,
                camera_id=camera_id,
                was_matched=was_matched,
                confidence=confidence,
            ).model_dump(),
        )
        await self.manager.broadcast_event(event)

    async def review_needed(
        self,
        object_id: str,
        reason: str,
        confidence: float,
        priority: str = "normal",
    ):
        """Broadcast review needed event."""
        event = WebSocketEvent(
            event=EventType.REVIEW_NEEDED,
            data=ReviewNeededEvent(
                object_id=object_id,
                reason=reason,
                confidence=confidence,
                priority=priority,
            ).model_dump(),
        )
        await self.manager.broadcast_event(event)

    async def alert(
        self,
        level: str,
        message: str,
        source: str = "system",
    ):
        """Broadcast alert event."""
        event = WebSocketEvent(
            event=EventType.ALERT,
            data={
                "level": level,
                "message": message,
                "source": source,
            },
        )
        await self.manager.broadcast_event(event)

    async def config_changed(self, changes: Dict[str, Any]):
        """Broadcast config changed event."""
        event = WebSocketEvent(
            event=EventType.CONFIG_CHANGED,
            data=changes,
        )
        await self.manager.broadcast_event(event)

    async def metrics_update(self, metrics: Dict[str, Any]):
        """Broadcast metrics update."""
        event = WebSocketEvent(
            event=EventType.METRICS_UPDATE,
            data=metrics,
        )
        await self.manager.broadcast_event(event)


# Global broadcaster
broadcaster = EventBroadcaster(manager)


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time frame streaming.

    Clients receive binary JPEG frames for live video display.
    """
    await manager.connect(websocket, "stream")

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Stream connection established",
            "timestamp": datetime.now().isoformat(),
        })

        # Keep connection alive and wait for disconnect
        while True:
            try:
                # Wait for any incoming messages (ping/pong)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text("keepalive")

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


@router.websocket("/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline events.

    Clients receive JSON events for object detection, updates,
    alerts, and metrics.
    """
    await manager.connect(websocket, "events")

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Event connection established",
            "timestamp": datetime.now().isoformat(),
            "events": [e.value for e in EventType],
        })

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Handle commands from client
                try:
                    message = json.loads(data)
                    await handle_client_message(websocket, message)
                except json.JSONDecodeError:
                    if data == "ping":
                        await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send keepalive heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "connections": manager.connection_count,
                })

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


async def handle_client_message(websocket: WebSocket, message: Dict[str, Any]):
    """Handle incoming message from client."""
    msg_type = message.get("type", "")

    if msg_type == "subscribe":
        # Client subscribing to specific events
        events = message.get("events", [])
        await websocket.send_json({
            "type": "subscribed",
            "events": events,
        })

    elif msg_type == "unsubscribe":
        events = message.get("events", [])
        await websocket.send_json({
            "type": "unsubscribed",
            "events": events,
        })

    elif msg_type == "request_metrics":
        # Send current metrics
        from ui.app import app_state
        import psutil

        await websocket.send_json({
            "type": "metrics",
            "data": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "uptime_seconds": app_state.uptime_seconds,
                "connections": manager.connection_count,
            },
            "timestamp": datetime.now().isoformat(),
        })

    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {msg_type}",
        })


# =============================================================================
# Background Tasks
# =============================================================================

async def start_metrics_broadcast(interval: float = 2.0):
    """Background task to broadcast metrics periodically."""
    import psutil
    from ui.app import app_state

    while True:
        await asyncio.sleep(interval)

        if manager.event_connections:
            metrics = {
                "fps": app_state.frame_count / max(app_state.uptime_seconds, 1),
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "connections": manager.connection_count,
                "uptime_seconds": app_state.uptime_seconds,
            }
            await broadcaster.metrics_update(metrics)


# =============================================================================
# Utility Functions
# =============================================================================

def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager."""
    return manager


def get_broadcaster() -> EventBroadcaster:
    """Get the global event broadcaster."""
    return broadcaster

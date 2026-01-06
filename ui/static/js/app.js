// PERCEPT UI JavaScript

// =============================================================================
// Configuration
// =============================================================================

const API_BASE = '/api';
const WS_BASE = `ws://${window.location.host}/ws`;
const UPDATE_INTERVAL = 2000; // ms

// =============================================================================
// State Management
// =============================================================================

const state = {
    dashboard: null,
    cameras: [],
    objects: [],
    reviewQueue: [],
    pipelineGraph: null,
    connected: false,
    wsEvents: null,
    wsStream: null,
};

// =============================================================================
// API Client
// =============================================================================

const api = {
    async get(endpoint) {
        const response = await fetch(`${API_BASE}${endpoint}`);
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        return response.json();
    },

    async post(endpoint, data) {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        return response.json();
    },

    async put(endpoint, data) {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        return response.json();
    },
};

// =============================================================================
// WebSocket Client
// =============================================================================

class WebSocketClient {
    constructor(url, handlers = {}) {
        this.url = url;
        this.handlers = handlers;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log(`WebSocket connected: ${this.url}`);
            this.reconnectAttempts = 0;
            if (this.handlers.onConnect) this.handlers.onConnect();
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (this.handlers.onMessage) this.handlers.onMessage(data);
            } catch (e) {
                // Binary or non-JSON message
                if (this.handlers.onBinary) this.handlers.onBinary(event.data);
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed');
            if (this.handlers.onDisconnect) this.handlers.onDisconnect();
            this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (this.handlers.onError) this.handlers.onError(error);
        };
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            console.log(`Reconnecting in ${delay}ms...`);
            setTimeout(() => this.connect(), delay);
        }
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(typeof data === 'string' ? data : JSON.stringify(data));
        }
    }

    close() {
        if (this.ws) {
            this.maxReconnectAttempts = 0;
            this.ws.close();
        }
    }
}

// =============================================================================
// Dashboard Module
// =============================================================================

const dashboard = {
    async load() {
        try {
            state.dashboard = await api.get('/dashboard');
            this.render();
        } catch (error) {
            console.error('Failed to load dashboard:', error);
        }
    },

    render() {
        const d = state.dashboard;
        if (!d) return;

        // Update stat cards
        this.updateStat('fps', d.fps.toFixed(1));
        this.updateStat('cpu', `${d.cpu_usage.toFixed(0)}%`);
        this.updateStat('memory', `${d.memory_usage.toFixed(0)}%`);
        this.updateStat('objects', d.total_objects);
        this.updateStat('pending', d.pending_review);
        this.updateStat('queue', d.queue_depth);

        // Update camera status
        this.renderCameras(d.cameras);

        // Update alerts
        this.renderAlerts(d.alerts);
    },

    updateStat(id, value) {
        const el = document.getElementById(`stat-${id}`);
        if (el) el.textContent = value;
    },

    renderCameras(cameras) {
        const container = document.getElementById('camera-grid');
        if (!container) return;

        container.innerHTML = cameras.map(cam => `
            <div class="camera-feed">
                <img src="/api/cameras/${cam.id}/frame" alt="${cam.name}">
                <div class="camera-overlay">
                    <span class="camera-name">
                        <span class="status-dot ${cam.connected ? 'active' : 'error'}"></span>
                        ${cam.name || cam.id}
                    </span>
                    <span class="camera-fps">${cam.fps.toFixed(1)} FPS</span>
                </div>
            </div>
        `).join('');
    },

    renderAlerts(alerts) {
        const container = document.getElementById('alerts');
        if (!container) return;

        if (!alerts || alerts.length === 0) {
            container.innerHTML = '<p class="text-muted">No active alerts</p>';
            return;
        }

        container.innerHTML = alerts.map(alert => `
            <div class="alert alert-${alert.level}">
                <strong>${alert.level.toUpperCase()}</strong>
                <span>${alert.message}</span>
            </div>
        `).join('');
    },

    startAutoRefresh() {
        this.load();
        setInterval(() => this.load(), UPDATE_INTERVAL);
    },
};

// =============================================================================
// Pipeline Module
// =============================================================================

const pipeline = {
    async load() {
        try {
            state.pipelineGraph = await api.get('/pipeline/graph');
            this.render();
        } catch (error) {
            console.error('Failed to load pipeline:', error);
        }
    },

    render() {
        const graph = state.pipelineGraph;
        if (!graph) return;

        const container = document.getElementById('pipeline-graph');
        if (!container) return;

        // Simple horizontal layout
        const html = graph.nodes.map((node, i) => {
            const statusClass = node.status === 'running' ? 'active' :
                               node.status === 'error' ? 'error' : '';

            return `
                <div class="pipeline-node ${statusClass}" data-node="${node.id}">
                    <div class="node-label">${node.label}</div>
                    <div class="node-timing">${node.timing_ms.toFixed(1)}ms</div>
                </div>
                ${i < graph.nodes.length - 1 ? '<div class="pipeline-edge"></div>' : ''}
            `;
        }).join('');

        container.innerHTML = html;

        // Add click handlers
        container.querySelectorAll('.pipeline-node').forEach(el => {
            el.addEventListener('click', () => {
                const nodeId = el.dataset.node;
                this.showNodeDetails(nodeId);
            });
        });
    },

    async showNodeDetails(nodeId) {
        try {
            const output = await api.get(`/pipeline/stages/${nodeId}/output`);
            console.log('Stage output:', output);
            // Could show in modal
        } catch (error) {
            console.error('Failed to load stage output:', error);
        }
    },
};

// =============================================================================
// Objects Module
// =============================================================================

const objects = {
    page: 1,
    pageSize: 20,

    async load(page = 1) {
        try {
            this.page = page;
            const data = await api.get(`/objects?page=${page}&page_size=${this.pageSize}`);
            state.objects = data.items;
            this.render(data);
        } catch (error) {
            console.error('Failed to load objects:', error);
        }
    },

    render(data) {
        const container = document.getElementById('object-grid');
        if (!container) return;

        if (!data.items || data.items.length === 0) {
            container.innerHTML = '<p class="text-muted">No objects found</p>';
            return;
        }

        container.innerHTML = data.items.map(obj => `
            <div class="object-card" data-id="${obj.id}">
                <div class="object-thumbnail">
                    ${obj.thumbnail_url ?
                        `<img src="${obj.thumbnail_url}" alt="${obj.primary_class}">` :
                        `<span class="text-muted">${obj.primary_class.charAt(0).toUpperCase()}</span>`
                    }
                </div>
                <div class="object-info">
                    <div class="object-class">${obj.primary_class}</div>
                    <div class="object-confidence">${(obj.confidence * 100).toFixed(0)}% confidence</div>
                </div>
            </div>
        `).join('');

        // Pagination
        this.renderPagination(data);
    },

    renderPagination(data) {
        const container = document.getElementById('pagination');
        if (!container) return;

        container.innerHTML = `
            <button class="btn btn-secondary btn-sm" ${data.page <= 1 ? 'disabled' : ''} onclick="objects.load(${data.page - 1})">Previous</button>
            <span class="text-muted">Page ${data.page} of ${data.pages}</span>
            <button class="btn btn-secondary btn-sm" ${data.page >= data.pages ? 'disabled' : ''} onclick="objects.load(${data.page + 1})">Next</button>
        `;
    },
};

// =============================================================================
// Review Module
// =============================================================================

const review = {
    async load() {
        try {
            state.reviewQueue = await api.get('/review?limit=20');
            this.render();
        } catch (error) {
            console.error('Failed to load review queue:', error);
        }
    },

    render() {
        const container = document.getElementById('review-grid');
        if (!container) return;

        if (!state.reviewQueue || state.reviewQueue.length === 0) {
            container.innerHTML = '<p class="text-muted">No items pending review</p>';
            return;
        }

        container.innerHTML = state.reviewQueue.map(item => `
            <div class="review-item" data-id="${item.object_id}">
                <div class="review-image">
                    ${item.crop_url ?
                        `<img src="${item.crop_url}" alt="Object">` :
                        '<div class="flex items-center justify-center h-full text-muted">No image</div>'
                    }
                </div>
                <div class="review-content">
                    <div class="review-suggestion">
                        <span class="status-dot warning"></span>
                        <strong>${item.primary_class}</strong>
                        <span class="text-muted">(${(item.confidence * 100).toFixed(0)}%)</span>
                    </div>
                    <div class="text-sm text-muted">Reason: ${item.reason}</div>
                    <div class="review-actions">
                        <button class="btn btn-success btn-sm" onclick="review.approve('${item.object_id}', '${item.primary_class}')">
                            Confirm
                        </button>
                        <button class="btn btn-secondary btn-sm" onclick="review.showClassSelect('${item.object_id}')">
                            Change
                        </button>
                        <button class="btn btn-secondary btn-sm" onclick="review.skip('${item.object_id}')">
                            Skip
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    },

    async approve(objectId, className) {
        try {
            await api.post(`/review/${objectId}`, { primary_class: className });
            this.load(); // Refresh
        } catch (error) {
            console.error('Failed to submit review:', error);
        }
    },

    async skip(objectId) {
        try {
            await api.post(`/review/${objectId}/skip`);
            this.load(); // Refresh
        } catch (error) {
            console.error('Failed to skip review:', error);
        }
    },

    showClassSelect(objectId) {
        const newClass = prompt('Enter correct class:');
        if (newClass) {
            this.approve(objectId, newClass);
        }
    },
};

// =============================================================================
// Config Module
// =============================================================================

const config = {
    async load() {
        try {
            const data = await api.get('/config');
            this.render(data);
        } catch (error) {
            console.error('Failed to load config:', error);
        }
    },

    render(data) {
        const editor = document.getElementById('config-editor');
        if (!editor) return;

        editor.value = JSON.stringify(data, null, 2);
    },

    async save() {
        const editor = document.getElementById('config-editor');
        if (!editor) return;

        try {
            const configData = JSON.parse(editor.value);
            const result = await api.put('/config', {
                config: configData,
                validate_only: false,
            });

            if (result.success) {
                alert('Configuration saved successfully');
            } else {
                alert(`Error: ${result.error}`);
            }
        } catch (error) {
            alert(`Invalid JSON: ${error.message}`);
        }
    },

    async validate() {
        const editor = document.getElementById('config-editor');
        if (!editor) return;

        try {
            const configData = JSON.parse(editor.value);
            const result = await api.post('/config/validate', {
                config: configData,
            });

            if (result.valid) {
                alert('Configuration is valid');
            } else {
                const errors = result.errors.map(e => `${e.path}: ${e.message}`).join('\n');
                alert(`Validation errors:\n${errors}`);
            }
        } catch (error) {
            alert(`Invalid JSON: ${error.message}`);
        }
    },
};

// =============================================================================
// WebSocket Event Handling
// =============================================================================

function initWebSockets() {
    // Event WebSocket
    state.wsEvents = new WebSocketClient(`${WS_BASE}/events`, {
        onConnect: () => {
            state.connected = true;
            updateConnectionStatus(true);
        },
        onDisconnect: () => {
            state.connected = false;
            updateConnectionStatus(false);
        },
        onMessage: (data) => {
            handleEvent(data);
        },
    });
    state.wsEvents.connect();
}

function handleEvent(event) {
    console.log('Event:', event);

    switch (event.event || event.type) {
        case 'frame_processed':
            // Update FPS display
            break;
        case 'object_detected':
            // Add to object list
            showNotification(`New ${event.data.primary_class} detected`);
            break;
        case 'review_needed':
            // Update review count
            showNotification('New item needs review', 'warning');
            break;
        case 'alert':
            showNotification(event.data.message, event.data.level);
            break;
        case 'metrics_update':
            // Update metrics display
            if (event.data.fps !== undefined) {
                dashboard.updateStat('fps', event.data.fps.toFixed(1));
            }
            break;
    }
}

function updateConnectionStatus(connected) {
    const el = document.getElementById('connection-status');
    if (el) {
        el.className = `status-dot ${connected ? 'active' : 'error'}`;
        el.title = connected ? 'Connected' : 'Disconnected';
    }
}

function showNotification(message, level = 'info') {
    // Simple notification (could use toast library)
    console.log(`[${level.toUpperCase()}] ${message}`);
}

// =============================================================================
// Utility Functions
// =============================================================================

function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
}

function formatBytes(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let unitIndex = 0;
    while (bytes >= 1024 && unitIndex < units.length - 1) {
        bytes /= 1024;
        unitIndex++;
    }
    return `${bytes.toFixed(1)} ${units[unitIndex]}`;
}

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Determine which page we're on
    const path = window.location.pathname;

    // Initialize WebSockets
    initWebSockets();

    // Load appropriate modules
    if (path === '/' || path === '/dashboard' || path.includes('dashboard')) {
        dashboard.startAutoRefresh();
    } else if (path.includes('pipeline')) {
        pipeline.load();
    } else if (path.includes('objects')) {
        objects.load();
    } else if (path.includes('review')) {
        review.load();
    } else if (path.includes('config')) {
        config.load();
    }
});

// Export for inline handlers
window.objects = objects;
window.review = review;
window.config = config;

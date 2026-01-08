"""Pipeline DAG visualization component.

Generates pipeline graph data for frontend visualization
and provides utilities for graph manipulation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ui.models import NodeType, PipelineEdge, PipelineGraph, PipelineNode


# =============================================================================
# Graph Builder
# =============================================================================

class PipelineGraphBuilder:
    """Builder for constructing pipeline graphs."""

    def __init__(self, name: str = "main"):
        self.name = name
        self._nodes: Dict[str, PipelineNode] = {}
        self._edges: List[PipelineEdge] = []

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        label: str,
        description: str = "",
        module_name: Optional[str] = None,
    ) -> "PipelineGraphBuilder":
        """Add a node to the graph."""
        self._nodes[node_id] = PipelineNode(
            id=node_id,
            type=node_type,
            label=label,
            description=description,
            module_name=module_name,
        )
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        label: str = "",
        data_type: str = "",
    ) -> "PipelineGraphBuilder":
        """Add an edge between nodes."""
        self._edges.append(PipelineEdge(
            source=source,
            target=target,
            label=label,
            data_type=data_type,
        ))
        return self

    def build(self) -> PipelineGraph:
        """Build the final graph."""
        return PipelineGraph(
            name=self.name,
            nodes=list(self._nodes.values()),
            edges=self._edges,
        )


# =============================================================================
# Default Pipeline Templates
# =============================================================================

def get_default_pipeline_graph() -> PipelineGraph:
    """Get the default PERCEPT pipeline graph."""
    return (
        PipelineGraphBuilder("main")
        .add_node("capture", NodeType.INPUT, "Camera Capture",
                  "RGB-D frame acquisition from RealSense", "RealSenseCamera")
        .add_node("segment", NodeType.PROCESS, "Segmentation",
                  "FastSAM + depth-based fusion", "SegmentationFusion")
        .add_node("tracking", NodeType.PROCESS, "Tracking",
                  "ByteTrack multi-object tracking", "ByteTrackWrapper")
        .add_node("reid", NodeType.PROCESS, "ReID",
                  "Re-identification with FAISS gallery", "ReIDMatcher")
        .add_node("router", NodeType.CONDITIONAL, "Router",
                  "Route to classification pipeline", "PipelineRouter")
        .add_node("person", NodeType.PROCESS, "Person Pipeline",
                  "Pose, clothing, face analysis", "PersonPipeline")
        .add_node("vehicle", NodeType.PROCESS, "Vehicle Pipeline",
                  "Color, type, plate detection", "VehiclePipeline")
        .add_node("generic", NodeType.PROCESS, "Generic Pipeline",
                  "ImageNet, color, shape analysis", "GenericPipeline")
        .add_node("persist", NodeType.OUTPUT, "Database",
                  "SQLite storage + embedding sync", "PerceptDatabase")
        # DEPTH PIPELINE BRANCH
        .add_node("depth", NodeType.PROCESS, "Depth View",
                  "Colorized depth map visualization", "DepthVisualizer")
        .add_node("depth_edges", NodeType.PROCESS, "Depth Edges",
                  "Depth discontinuity detection", "DepthEdgeDetector")
        .add_node("depth_objects", NodeType.PROCESS, "Depth Objects",
                  "Connected component segmentation", "DepthConnectedComponents")
        .add_node("depth_clusters", NodeType.OUTPUT, "3D Clusters",
                  "Point cloud clustering", "PointCloudSegmenter")
        # MAIN PIPELINE EDGES
        .add_edge("capture", "segment", data_type="FrameData")
        .add_edge("segment", "tracking", data_type="ObjectMask[]")
        .add_edge("tracking", "reid", data_type="Track[]")
        .add_edge("reid", "router", data_type="ObjectSchema[]")
        .add_edge("router", "person", label="person")
        .add_edge("router", "vehicle", label="vehicle")
        .add_edge("router", "generic", label="other")
        .add_edge("person", "persist")
        .add_edge("vehicle", "persist")
        .add_edge("generic", "persist")
        # DEPTH PIPELINE EDGES
        .add_edge("capture", "depth", data_type="FrameData")
        .add_edge("depth", "depth_edges", data_type="depth_colorized")
        .add_edge("depth_edges", "depth_objects", data_type="edge_mask")
        .add_edge("depth_objects", "depth_clusters", data_type="ObjectMask[]")
        .build()
    )


def get_segmentation_subgraph() -> PipelineGraph:
    """Get segmentation sub-pipeline graph."""
    return (
        PipelineGraphBuilder("segmentation")
        .add_node("input", NodeType.INPUT, "RGB-D Frame")
        .add_node("fastsam", NodeType.PROCESS, "FastSAM",
                  "AI segmentation on Hailo-8", "FastSAMSegmenter")
        .add_node("depth_seg", NodeType.PROCESS, "Depth Segmentation",
                  "Depth discontinuity detection", "DepthSegmenter")
        .add_node("pointcloud", NodeType.PROCESS, "Point Cloud",
                  "3D clustering", "PointCloudSegmenter")
        .add_node("fusion", NodeType.PROCESS, "Fusion",
                  "Multi-method mask fusion", "SegmentationFusion")
        .add_node("output", NodeType.OUTPUT, "Object Masks")
        .add_edge("input", "fastsam")
        .add_edge("input", "depth_seg")
        .add_edge("input", "pointcloud")
        .add_edge("fastsam", "fusion")
        .add_edge("depth_seg", "fusion")
        .add_edge("pointcloud", "fusion")
        .add_edge("fusion", "output")
        .build()
    )


# =============================================================================
# Graph Analysis
# =============================================================================

class PipelineAnalyzer:
    """Analyze pipeline graph structure."""

    def __init__(self, graph: PipelineGraph):
        self.graph = graph
        self._node_map = {n.id: n for n in graph.nodes}
        self._adjacency = self._build_adjacency()

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        adj = {n.id: [] for n in self.graph.nodes}
        for edge in self.graph.edges:
            adj[edge.source].append(edge.target)
        return adj

    def get_node(self, node_id: str) -> Optional[PipelineNode]:
        """Get node by ID."""
        return self._node_map.get(node_id)

    def get_successors(self, node_id: str) -> List[str]:
        """Get successor node IDs."""
        return self._adjacency.get(node_id, [])

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessor node IDs."""
        predecessors = []
        for edge in self.graph.edges:
            if edge.target == node_id:
                predecessors.append(edge.source)
        return predecessors

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order."""
        visited = set()
        order = []

        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            for succ in self._adjacency.get(node_id, []):
                dfs(succ)
            order.append(node_id)

        for node in self.graph.nodes:
            dfs(node.id)

        return list(reversed(order))

    def get_input_nodes(self) -> List[PipelineNode]:
        """Get all input nodes."""
        return [n for n in self.graph.nodes if n.type == NodeType.INPUT]

    def get_output_nodes(self) -> List[PipelineNode]:
        """Get all output nodes."""
        return [n for n in self.graph.nodes if n.type == NodeType.OUTPUT]

    def get_critical_path(self) -> List[str]:
        """Get the critical path (longest path through DAG)."""
        # Simple implementation - assumes equal node costs
        topo_order = self.topological_sort()
        distances = {n: 0 for n in topo_order}
        predecessors = {n: None for n in topo_order}

        for node in topo_order:
            for succ in self._adjacency.get(node, []):
                if distances[succ] < distances[node] + 1:
                    distances[succ] = distances[node] + 1
                    predecessors[succ] = node

        # Find the end node with max distance
        max_dist = max(distances.values())
        end_node = [n for n, d in distances.items() if d == max_dist][0]

        # Reconstruct path
        path = []
        current = end_node
        while current:
            path.append(current)
            current = predecessors[current]

        return list(reversed(path))


# =============================================================================
# Graph Layout
# =============================================================================

@dataclass
class NodePosition:
    """Position of a node in the layout."""
    x: float
    y: float
    width: float = 150
    height: float = 60


@dataclass
class GraphLayout:
    """Layout information for graph visualization."""
    positions: Dict[str, NodePosition]
    width: float
    height: float


class LayoutEngine:
    """Calculate node positions for graph visualization."""

    def __init__(self, graph: PipelineGraph):
        self.graph = graph
        self.analyzer = PipelineAnalyzer(graph)

    def calculate_layout(
        self,
        node_width: float = 150,
        node_height: float = 60,
        h_spacing: float = 50,
        v_spacing: float = 80,
    ) -> GraphLayout:
        """Calculate positions for all nodes using layered layout."""
        # Assign layers based on longest path from inputs
        layers = self._assign_layers()

        # Group nodes by layer
        layer_nodes: Dict[int, List[str]] = {}
        for node_id, layer in layers.items():
            if layer not in layer_nodes:
                layer_nodes[layer] = []
            layer_nodes[layer].append(node_id)

        # Calculate positions
        positions = {}
        max_layer = max(layers.values()) if layers else 0
        max_nodes_in_layer = max(len(nodes) for nodes in layer_nodes.values()) if layer_nodes else 1

        total_width = max_nodes_in_layer * (node_width + h_spacing) + h_spacing
        total_height = (max_layer + 1) * (node_height + v_spacing) + v_spacing

        for layer, nodes in layer_nodes.items():
            y = v_spacing + layer * (node_height + v_spacing)
            layer_width = len(nodes) * (node_width + h_spacing) - h_spacing
            start_x = (total_width - layer_width) / 2

            for i, node_id in enumerate(nodes):
                x = start_x + i * (node_width + h_spacing)
                positions[node_id] = NodePosition(
                    x=x,
                    y=y,
                    width=node_width,
                    height=node_height,
                )

        return GraphLayout(
            positions=positions,
            width=total_width,
            height=total_height,
        )

    def _assign_layers(self) -> Dict[str, int]:
        """Assign layer numbers to nodes."""
        layers = {}

        # Initialize input nodes at layer 0
        for node in self.analyzer.get_input_nodes():
            layers[node.id] = 0

        # BFS to assign layers
        topo_order = self.analyzer.topological_sort()
        for node_id in topo_order:
            if node_id not in layers:
                # Layer is max of predecessor layers + 1
                pred_layers = [
                    layers.get(p, -1)
                    for p in self.analyzer.get_predecessors(node_id)
                ]
                layers[node_id] = max(pred_layers) + 1 if pred_layers else 0

        return layers

    def to_json(self) -> Dict[str, Any]:
        """Export layout as JSON for frontend."""
        layout = self.calculate_layout()
        return {
            "name": self.graph.name,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type.value,
                    "label": node.label,
                    "description": node.description,
                    "module": node.module_name,
                    "x": layout.positions[node.id].x,
                    "y": layout.positions[node.id].y,
                    "width": layout.positions[node.id].width,
                    "height": layout.positions[node.id].height,
                }
                for node in self.graph.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "data_type": edge.data_type,
                }
                for edge in self.graph.edges
            ],
            "width": layout.width,
            "height": layout.height,
        }

"""UMR graph representation and manipulation.

This module provides the UMRGraph class for representing Uniform Meaning Representation
as a directed graph with typed nodes and edges for semantic analysis.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import networkx as nx
import logging

logger = logging.getLogger(__name__)


@dataclass
class UMRNode:
    """Represents a node in a UMR graph."""
    
    id: str
    label: str
    node_type: str  # e.g., 'concept', 'event', 'property', 'quantifier'
    attributes: Dict[str, Any] = field(default_factory=dict)
    surface_form: Optional[str] = None
    language: Optional[str] = None
    
    def __post_init__(self):
        """Validate node after initialization."""
        if not self.id:
            raise ValueError("Node ID cannot be empty")
        if not self.label:
            raise ValueError("Node label cannot be empty")


@dataclass
class UMREdge:
    """Represents an edge in a UMR graph."""
    
    source: str
    target: str
    relation: str  # e.g., 'ARG0', 'ARG1', 'mod', 'quant', 'time'
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate edge after initialization."""
        if not self.source:
            raise ValueError("Edge source cannot be empty")
        if not self.target:
            raise ValueError("Edge target cannot be empty")
        if not self.relation:
            raise ValueError("Edge relation cannot be empty")


class UMRGraph:
    """Represents a Uniform Meaning Representation as a directed graph."""
    
    def __init__(self, graph_id: str = "umr_graph"):
        """Initialize an empty UMR graph.
        
        Args:
            graph_id: Unique identifier for this graph
        """
        self.graph_id = graph_id
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, UMRNode] = {}
        self.edges: List[UMREdge] = []
        self.metadata: Dict[str, Any] = {
            "source_text": None,
            "language": None,
            "creation_time": None
        }
        
    def add_node(self, node: UMRNode) -> None:
        """Add a node to the UMR graph.
        
        Args:
            node: UMRNode to add
        """
        if node.id in self.nodes:
            logger.warning(f"Node {node.id} already exists, overwriting")
        
        self.nodes[node.id] = node
        self.graph.add_node(
            node.id,
            label=node.label,
            node_type=node.node_type,
            attributes=node.attributes,
            surface_form=node.surface_form,
            language=node.language
        )
        
    def add_edge(self, edge: UMREdge) -> None:
        """Add an edge to the UMR graph.
        
        Args:
            edge: UMREdge to add
        """
        if edge.source not in self.nodes:
            raise ValueError(f"Source node {edge.source} not found")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node {edge.target} not found")
            
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            attributes=edge.attributes
        )
        
    def get_node(self, node_id: str) -> Optional[UMRNode]:
        """Get a node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            UMRNode if found, None otherwise
        """
        return self.nodes.get(node_id)
        
    def get_nodes_by_type(self, node_type: str) -> List[UMRNode]:
        """Get all nodes of a specific type.
        
        Args:
            node_type: Type of nodes to retrieve
            
        Returns:
            List of UMRNodes of the specified type
        """
        return [node for node in self.nodes.values() if node.node_type == node_type]
        
    def get_edges_by_relation(self, relation: str) -> List[UMREdge]:
        """Get all edges with a specific relation.
        
        Args:
            relation: Relation type to filter by
            
        Returns:
            List of UMREdges with the specified relation
        """
        return [edge for edge in self.edges if edge.relation == relation]
        
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[Tuple[str, str]]:
        """Get neighboring nodes and their relations.
        
        Args:
            node_id: ID of the node
            direction: 'in', 'out', or 'both'
            
        Returns:
            List of (neighbor_id, relation) tuples
        """
        if direction == "in":
            return [(pred, self.graph[pred][node_id]["relation"]) 
                   for pred in self.graph.predecessors(node_id)]
        elif direction == "out":
            return [(succ, self.graph[node_id][succ]["relation"]) 
                   for succ in self.graph.successors(node_id)]
        else:  # both
            neighbors = []
            for pred in self.graph.predecessors(node_id):
                neighbors.append((pred, self.graph[pred][node_id]["relation"]))
            for succ in self.graph.successors(node_id):
                neighbors.append((succ, self.graph[node_id][succ]["relation"]))
            return neighbors
            
    def find_paths(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        """Find all paths between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length to consider
            
        Returns:
            List of paths as lists of node IDs
        """
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            return paths
        except nx.NetworkXNoPath:
            return []
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert UMR graph to dictionary representation.
        
        Returns:
            Dictionary representation of the graph
        """
        return {
            "graph_id": self.graph_id,
            "metadata": self.metadata,
            "nodes": {
                node_id: {
                    "label": node.label,
                    "node_type": node.node_type,
                    "attributes": node.attributes,
                    "surface_form": node.surface_form,
                    "language": node.language
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "attributes": edge.attributes
                }
                for edge in self.edges
            ]
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UMRGraph":
        """Create UMR graph from dictionary representation.
        
        Args:
            data: Dictionary representation of the graph
            
        Returns:
            UMRGraph instance
        """
        graph = cls(data.get("graph_id", "umr_graph"))
        graph.metadata = data.get("metadata", {})
        
        # Add nodes
        for node_id, node_data in data.get("nodes", {}).items():
            node = UMRNode(
                id=node_id,
                label=node_data["label"],
                node_type=node_data["node_type"],
                attributes=node_data.get("attributes", {}),
                surface_form=node_data.get("surface_form"),
                language=node_data.get("language")
            )
            graph.add_node(node)
            
        # Add edges
        for edge_data in data.get("edges", []):
            edge = UMREdge(
                source=edge_data["source"],
                target=edge_data["target"],
                relation=edge_data["relation"],
                attributes=edge_data.get("attributes", {})
            )
            graph.add_edge(edge)
            
        return graph
        
    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self.nodes)
        
    def __repr__(self) -> str:
        """String representation of the UMR graph."""
        return f"UMRGraph(id='{self.graph_id}', nodes={len(self.nodes)}, edges={len(self.edges)})"

"""
EIL Graph Data Structures

Core data structures for representing semantic meaning as executable graphs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import uuid


class EILNodeType(Enum):
    """Types of EIL nodes."""
    ENTITY = "entity"
    EVENT = "event"
    PROPERTY = "property"
    QUANTIFIER = "quantifier"
    MODAL = "modal"
    NEGATION = "negation"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    LOGICAL = "logical"


class EILRelationType(Enum):
    """Types of EIL relations."""
    # Core semantic relations
    AGENT = "agent"
    PATIENT = "patient"
    THEME = "theme"
    GOAL = "goal"
    INSTRUMENT = "instrument"
    LOCATION = "location"
    TIME = "time"
    
    # Logical relations
    CAUSE = "cause"
    ENABLE = "enable"
    PREVENT = "prevent"
    IF_THEN = "if_then"
    AND = "and"
    OR = "or"
    NOT = "not"
    
    # Temporal relations
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    SIMULTANEOUS = "simultaneous"
    
    # Spatial relations
    IN = "in"
    ON = "on"
    NEAR = "near"
    FAR = "far"
    ABOVE = "above"
    BELOW = "below"
    
    # Quantification
    ALL = "all"
    SOME = "some"
    NONE = "none"
    MOST = "most"
    FEW = "few"
    
    # Modality
    CAN = "can"
    MUST = "must"
    SHOULD = "should"
    MIGHT = "might"
    WILL = "will"


@dataclass
class EILNode:
    """A node in the EIL graph representing a semantic entity, event, or concept."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    node_type: EILNodeType = EILNodeType.ENTITY
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "unknown"  # NSM, UD, LLM, etc.
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class EILRelation:
    """A relation between two EIL nodes."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: EILRelationType = EILRelationType.AND
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    scope: Optional[str] = None  # For quantifier scope
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class EILGraph:
    """An EIL graph representing the semantic structure of a text."""
    
    nodes: Dict[str, EILNode] = field(default_factory=dict)
    relations: Dict[str, EILRelation] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: EILNode) -> str:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        return node.id
    
    def add_relation(self, relation: EILRelation) -> str:
        """Add a relation to the graph."""
        # Validate that both nodes exist
        if relation.source_id not in self.nodes:
            raise ValueError(f"Source node {relation.source_id} not found")
        if relation.target_id not in self.nodes:
            raise ValueError(f"Target node {relation.target_id} not found")
        
        self.relations[relation.id] = relation
        return relation.id
    
    def get_node(self, node_id: str) -> Optional[EILNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_relation(self, relation_id: str) -> Optional[EILRelation]:
        """Get a relation by ID."""
        return self.relations.get(relation_id)
    
    def get_relations_by_node(self, node_id: str) -> List[EILRelation]:
        """Get all relations involving a specific node."""
        return [
            rel for rel in self.relations.values()
            if rel.source_id == node_id or rel.target_id == node_id
        ]
    
    def get_connected_nodes(self, node_id: str) -> List[EILNode]:
        """Get all nodes connected to a specific node."""
        connected_ids = set()
        for rel in self.get_relations_by_node(node_id):
            if rel.source_id == node_id:
                connected_ids.add(rel.target_id)
            else:
                connected_ids.add(rel.source_id)
        
        return [self.nodes[node_id] for node_id in connected_ids if node_id in self.nodes]
    
    def validate(self) -> List[str]:
        """Validate the graph structure and return any errors."""
        errors = []
        
        # Check for orphaned relations
        for rel_id, relation in self.relations.items():
            if relation.source_id not in self.nodes:
                errors.append(f"Relation {rel_id} references non-existent source node {relation.source_id}")
            if relation.target_id not in self.nodes:
                errors.append(f"Relation {rel_id} references non-existent target node {relation.target_id}")
        
        # Check for self-loops (usually invalid)
        for rel_id, relation in self.relations.items():
            if relation.source_id == relation.target_id:
                errors.append(f"Self-loop detected in relation {rel_id}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": {node_id: {
                "id": node.id,
                "label": node.label,
                "node_type": node.node_type.value,
                "properties": node.properties,
                "confidence": node.confidence,
                "source": node.source
            } for node_id, node in self.nodes.items()},
            "relations": {rel_id: {
                "id": relation.id,
                "source_id": relation.source_id,
                "target_id": relation.target_id,
                "relation_type": relation.relation_type.value,
                "properties": relation.properties,
                "confidence": relation.confidence,
                "scope": relation.scope
            } for rel_id, relation in self.relations.items()},
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EILGraph':
        """Create graph from dictionary."""
        graph = cls()
        
        # Add nodes
        for node_id, node_data in data.get("nodes", {}).items():
            node = EILNode(
                id=node_data["id"],
                label=node_data["label"],
                node_type=EILNodeType(node_data["node_type"]),
                properties=node_data.get("properties", {}),
                confidence=node_data.get("confidence", 1.0),
                source=node_data.get("source", "unknown")
            )
            graph.add_node(node)
        
        # Add relations
        for rel_id, rel_data in data.get("relations", {}).items():
            relation = EILRelation(
                id=rel_data["id"],
                source_id=rel_data["source_id"],
                target_id=rel_data["target_id"],
                relation_type=EILRelationType(rel_data["relation_type"]),
                properties=rel_data.get("properties", {}),
                confidence=rel_data.get("confidence", 1.0),
                scope=rel_data.get("scope")
            )
            graph.add_relation(relation)
        
        graph.metadata = data.get("metadata", {})
        return graph


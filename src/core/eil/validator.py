"""
EIL Graph Validator

Validates EIL graphs for legality, scope attachment, molecule caps, and max depth.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any
from enum import Enum

from .graph import EILGraph, EILNode, EILRelation, EILNodeType, EILRelationType


class ValidationLevel(Enum):
    """Validation levels."""
    LEGALITY = "legality"
    SCOPE = "scope"
    MOLECULE = "molecule"
    DEPTH = "depth"


@dataclass
class ValidationError:
    """A validation error."""
    level: ValidationLevel
    message: str
    node_id: Optional[str] = None
    relation_id: Optional[str] = None
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationResult:
    """Result of EIL graph validation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    info: List[ValidationError]
    metrics: Dict[str, Any]


class EILValidator:
    """Validates EIL graphs for various constraints."""
    
    def __init__(self):
        """Initialize the validator."""
        # Configuration
        self.max_depth = 10
        self.max_molecule_size = 50
        self.max_nodes_per_graph = 100
        
        # Legal node type combinations
        self.legal_combinations = {
            EILNodeType.ENTITY: {EILNodeType.EVENT, EILNodeType.PROPERTY, EILNodeType.SPATIAL, EILNodeType.TEMPORAL},
            EILNodeType.EVENT: {EILNodeType.ENTITY, EILNodeType.PROPERTY, EILNodeType.TEMPORAL, EILNodeType.SPATIAL},
            EILNodeType.PROPERTY: {EILNodeType.ENTITY, EILNodeType.EVENT},
            EILNodeType.QUANTIFIER: {EILNodeType.ENTITY, EILNodeType.EVENT, EILNodeType.PROPERTY},
            EILNodeType.MODAL: {EILNodeType.EVENT, EILNodeType.PROPERTY},
            EILNodeType.NEGATION: {EILNodeType.ENTITY, EILNodeType.EVENT, EILNodeType.PROPERTY, EILNodeType.LOGICAL},
            EILNodeType.TEMPORAL: {EILNodeType.EVENT, EILNodeType.ENTITY},
            EILNodeType.SPATIAL: {EILNodeType.ENTITY, EILNodeType.EVENT},
            EILNodeType.CAUSAL: {EILNodeType.EVENT, EILNodeType.ENTITY},
            EILNodeType.LOGICAL: {EILNodeType.ENTITY, EILNodeType.EVENT, EILNodeType.PROPERTY, EILNodeType.LOGICAL}
        }
        
        # Scope-critical relations
        self.scope_critical_relations = {
            EILRelationType.NOT,
            EILRelationType.ALL,
            EILRelationType.SOME,
            EILRelationType.NONE,
            EILRelationType.MUST,
            EILRelationType.CAN,
            EILRelationType.SHOULD
        }
    
    def validate_graph(self, graph: EILGraph) -> ValidationResult:
        """Validate an EIL graph comprehensively."""
        errors = []
        warnings = []
        info = []
        
        # Basic structure validation
        structure_errors = self._validate_structure(graph)
        errors.extend(structure_errors)
        
        # Legality validation
        legality_errors = self._validate_legality(graph)
        errors.extend(legality_errors)
        
        # Scope validation
        scope_errors, scope_warnings = self._validate_scope(graph)
        errors.extend(scope_errors)
        warnings.extend(scope_warnings)
        
        # Molecule validation
        molecule_errors, molecule_warnings = self._validate_molecules(graph)
        errors.extend(molecule_errors)
        warnings.extend(molecule_warnings)
        
        # Depth validation
        depth_errors, depth_warnings = self._validate_depth(graph)
        errors.extend(depth_errors)
        warnings.extend(depth_warnings)
        
        # Calculate metrics
        metrics = self._calculate_metrics(graph)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info,
            metrics=metrics
        )
    
    def _validate_structure(self, graph: EILGraph) -> List[ValidationError]:
        """Validate basic graph structure."""
        errors = []
        
        # Check for orphaned relations
        for rel_id, relation in graph.relations.items():
            if relation.source_id not in graph.nodes:
                errors.append(ValidationError(
                    level=ValidationLevel.LEGALITY,
                    message=f"Relation {rel_id} references non-existent source node {relation.source_id}",
                    relation_id=rel_id,
                    severity="error"
                ))
            
            if relation.target_id not in graph.nodes:
                errors.append(ValidationError(
                    level=ValidationLevel.LEGALITY,
                    message=f"Relation {rel_id} references non-existent target node {relation.target_id}",
                    relation_id=rel_id,
                    severity="error"
                ))
        
        # Check for self-loops
        for rel_id, relation in graph.relations.items():
            if relation.source_id == relation.target_id:
                errors.append(ValidationError(
                    level=ValidationLevel.LEGALITY,
                    message=f"Self-loop detected in relation {rel_id}",
                    relation_id=rel_id,
                    severity="error"
                ))
        
        return errors
    
    def _validate_legality(self, graph: EILGraph) -> List[ValidationError]:
        """Validate legal node type combinations."""
        errors = []
        
        for rel_id, relation in graph.relations.items():
            source_node = graph.get_node(relation.source_id)
            target_node = graph.get_node(relation.target_id)
            
            if source_node and target_node:
                # Check if this combination is legal
                legal_targets = self.legal_combinations.get(source_node.node_type, set())
                if target_node.node_type not in legal_targets:
                    errors.append(ValidationError(
                        level=ValidationLevel.LEGALITY,
                        message=f"Illegal connection: {source_node.node_type.value} -> {target_node.node_type.value}",
                        relation_id=rel_id,
                        severity="error"
                    ))
        
        return errors
    
    def _validate_scope(self, graph: EILGraph) -> tuple[List[ValidationError], List[ValidationError]]:
        """Validate scope attachment for quantifiers, negation, and modals."""
        errors = []
        warnings = []
        
        # Find scope-critical relations
        scope_relations = [
            rel for rel in graph.relations.values()
            if rel.relation_type in self.scope_critical_relations
        ]
        
        for relation in scope_relations:
            # Check if scope is properly attached
            if not relation.scope:
                warnings.append(ValidationError(
                    level=ValidationLevel.SCOPE,
                    message=f"Scope-critical relation {relation.relation_type.value} has no scope annotation",
                    relation_id=relation.id,
                    severity="warning"
                ))
            
            # Check for scope conflicts
            conflicting_scopes = self._find_scope_conflicts(graph, relation)
            for conflict in conflicting_scopes:
                errors.append(ValidationError(
                    level=ValidationLevel.SCOPE,
                    message=f"Scope conflict detected: {conflict}",
                    relation_id=relation.id,
                    severity="error"
                ))
        
        return errors, warnings
    
    def _find_scope_conflicts(self, graph: EILGraph, relation: EILRelation) -> List[str]:
        """Find scope conflicts for a given relation."""
        conflicts = []
        
        # Check for overlapping scopes
        for other_rel in graph.relations.values():
            if other_rel.id != relation.id and other_rel.scope:
                if relation.scope and relation.scope == other_rel.scope:
                    if relation.relation_type in [EILRelationType.NOT, EILRelationType.ALL]:
                        conflicts.append(f"Overlapping scope with {other_rel.relation_type.value}")
        
        return conflicts
    
    def _validate_molecules(self, graph: EILGraph) -> tuple[List[ValidationError], List[ValidationError]]:
        """Validate molecule size constraints."""
        errors = []
        warnings = []
        
        # Find connected components (molecules)
        molecules = self._find_molecules(graph)
        
        for i, molecule in enumerate(molecules):
            if len(molecule) > self.max_molecule_size:
                errors.append(ValidationError(
                    level=ValidationLevel.MOLECULE,
                    message=f"Molecule {i} exceeds size limit: {len(molecule)} > {self.max_molecule_size}",
                    severity="error"
                ))
            elif len(molecule) > self.max_molecule_size * 0.8:
                warnings.append(ValidationError(
                    level=ValidationLevel.MOLECULE,
                    message=f"Molecule {i} approaching size limit: {len(molecule)}",
                    severity="warning"
                ))
        
        return errors, warnings
    
    def _find_molecules(self, graph: EILGraph) -> List[Set[str]]:
        """Find connected components (molecules) in the graph."""
        visited = set()
        molecules = []
        
        for node_id in graph.nodes:
            if node_id not in visited:
                molecule = self._dfs_component(graph, node_id, visited)
                molecules.append(molecule)
        
        return molecules
    
    def _dfs_component(self, graph: EILGraph, start_id: str, visited: Set[str]) -> Set[str]:
        """DFS to find connected component."""
        component = set()
        stack = [start_id]
        
        while stack:
            node_id = stack.pop()
            if node_id not in visited:
                visited.add(node_id)
                component.add(node_id)
                
                # Add connected nodes
                for relation in graph.get_relations_by_node(node_id):
                    if relation.source_id == node_id and relation.target_id not in visited:
                        stack.append(relation.target_id)
                    elif relation.target_id == node_id and relation.source_id not in visited:
                        stack.append(relation.source_id)
        
        return component
    
    def _validate_depth(self, graph: EILGraph) -> tuple[List[ValidationError], List[ValidationError]]:
        """Validate graph depth constraints."""
        errors = []
        warnings = []
        
        # Calculate depth for each node
        depths = self._calculate_depths(graph)
        
        for node_id, depth in depths.items():
            if depth > self.max_depth:
                errors.append(ValidationError(
                    level=ValidationLevel.DEPTH,
                    message=f"Node {node_id} exceeds depth limit: {depth} > {self.max_depth}",
                    node_id=node_id,
                    severity="error"
                ))
            elif depth > self.max_depth * 0.8:
                warnings.append(ValidationError(
                    level=ValidationLevel.DEPTH,
                    message=f"Node {node_id} approaching depth limit: {depth}",
                    node_id=node_id,
                    severity="warning"
                ))
        
        return errors, warnings
    
    def _calculate_depths(self, graph: EILGraph) -> Dict[str, int]:
        """Calculate depth for each node in the graph."""
        depths = {}
        
        # Find root nodes (nodes with no incoming relations)
        root_nodes = set(graph.nodes.keys())
        for relation in graph.relations.values():
            if relation.target_id in root_nodes:
                root_nodes.remove(relation.target_id)
        
        # Calculate depths using BFS
        for root_id in root_nodes:
            self._bfs_depth(graph, root_id, depths)
        
        # Handle disconnected nodes
        for node_id in graph.nodes:
            if node_id not in depths:
                depths[node_id] = 0
        
        return depths
    
    def _bfs_depth(self, graph: EILGraph, start_id: str, depths: Dict[str, int]):
        """BFS to calculate depths."""
        queue = [(start_id, 0)]
        visited = set()
        
        while queue:
            node_id, depth = queue.pop(0)
            if node_id not in visited:
                visited.add(node_id)
                depths[node_id] = max(depths.get(node_id, 0), depth)
                
                # Add children
                for relation in graph.get_relations_by_node(node_id):
                    if relation.source_id == node_id and relation.target_id not in visited:
                        queue.append((relation.target_id, depth + 1))
    
    def _calculate_metrics(self, graph: EILGraph) -> Dict[str, Any]:
        """Calculate graph metrics."""
        return {
            "total_nodes": len(graph.nodes),
            "total_relations": len(graph.relations),
            "avg_confidence": sum(node.confidence for node in graph.nodes.values()) / len(graph.nodes) if graph.nodes else 0,
            "scope_critical_count": len([r for r in graph.relations.values() if r.relation_type in self.scope_critical_relations]),
            "molecule_count": len(self._find_molecules(graph)),
            "max_depth": max(self._calculate_depths(graph).values()) if graph.nodes else 0
        }


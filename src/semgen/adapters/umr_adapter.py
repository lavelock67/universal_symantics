"""
UMR Adapter - Converts UMR graphs to semantic features for the SemanticGenerator.

This adapter extracts semantic features from UMR graphs and converts them to
the same intermediate format that the SemanticGenerator expects from UD/SRL.
No primes are emitted here - only features that feed into the generator.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import logging
import yaml
from pathlib import Path

from ...umr.graph import UMRGraph, UMRNode, UMREdge

logger = logging.getLogger(__name__)


@dataclass
class SemFeatures:
    """Semantic features extracted from UMR graph."""
    
    events: List[Dict[str, Any]] = field(default_factory=list)
    args: List[Dict[str, Any]] = field(default_factory=list)
    scopes: List[Dict[str, Any]] = field(default_factory=list)
    quantifiers: List[Dict[str, Any]] = field(default_factory=list)
    negations: List[Dict[str, Any]] = field(default_factory=list)
    modality: List[Dict[str, Any]] = field(default_factory=list)
    spatial_roles: List[Dict[str, Any]] = field(default_factory=list)
    times: List[Dict[str, Any]] = field(default_factory=list)
    
    def merge(self, other: 'SemFeatures') -> 'SemFeatures':
        """Merge with another SemFeatures object."""
        return SemFeatures(
            events=self.events + other.events,
            args=self.args + other.args,
            scopes=self.scopes + other.scopes,
            quantifiers=self.quantifiers + other.quantifiers,
            negations=self.negations + other.negations,
            modality=self.modality + other.modality,
            spatial_roles=self.spatial_roles + other.spatial_roles,
            times=self.times + other.times
        )


class UMRAdapter:
    """Adapter for converting UMR graphs to semantic features."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the UMR adapter.
        
        Args:
            config_path: Path to UMR to EIL mapping configuration
        """
        self.config_path = config_path or "assets/umr_to_eil.yaml"
        self.mappings = self._load_mappings()
        self.spatial_maps = self._load_spatial_maps()
        
    def _load_mappings(self) -> Dict[str, Any]:
        """Load UMR to EIL mappings from configuration file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"UMR mapping config not found: {self.config_path}")
                return self._get_default_mappings()
        except Exception as e:
            logger.warning(f"Failed to load UMR mappings: {e}")
            return self._get_default_mappings()
    
    def _get_default_mappings(self) -> Dict[str, Any]:
        """Get default UMR to EIL mappings."""
        return {
            "events": {
                "agentive": "DO",
                "non_agentive": "HAPPEN"
            },
            "arguments": {
                "ARG0": "AGENT",
                "ARG1": "PATIENT", 
                "ARG2": "GOAL",
                "ARG3": "BENEFICIARY",
                "ARG4": "INSTRUMENT"
            },
            "modality": {
                "must": "MUST",
                "can": "CAN", 
                "should": "SHOULD",
                "may": "MAY"
            },
            "quantifiers": {
                "all": "ALL",
                "some": "SOME",
                "one": "ONE",
                "many": "MANY",
                "few": "FEW",
                "half": "HALF"
            },
            "negation": {
                "polarity": "NOT"
            },
            "spatial": {
                "prep": "spatial_relation"  # Maps to spatial_maps.json
            }
        }
    
    def _load_spatial_maps(self) -> Dict[str, Dict[str, List[str]]]:
        """Load spatial relation mappings."""
        try:
            spatial_file = Path("assets/spatial_maps.json")
            if spatial_file.exists():
                import json
                with open(spatial_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Spatial maps not found: assets/spatial_maps.json")
                return {}
        except Exception as e:
            logger.warning(f"Failed to load spatial maps: {e}")
            return {}
    
    def umr_to_features(self, umr_graph: UMRGraph, language: str = "en") -> SemFeatures:
        """Convert UMR graph to semantic features.
        
        Args:
            umr_graph: UMR graph to convert
            language: Source language code
            
        Returns:
            SemFeatures object with extracted semantic features
        """
        features = SemFeatures()
        
        # Extract events
        features.events = self._extract_events(umr_graph)
        
        # Extract arguments
        features.args = self._extract_arguments(umr_graph)
        
        # Extract scopes and quantifiers
        features.scopes = self._extract_scopes(umr_graph)
        features.quantifiers = self._extract_quantifiers(umr_graph)
        
        # Extract negation
        features.negations = self._extract_negation(umr_graph)
        
        # Extract modality
        features.modality = self._extract_modality(umr_graph)
        
        # Extract spatial roles
        features.spatial_roles = self._extract_spatial_roles(umr_graph, language)
        
        # Extract temporal expressions
        features.times = self._extract_temporal(umr_graph)
        
        logger.debug(f"Extracted {len(features.events)} events, {len(features.args)} args from UMR")
        return features
    
    def _extract_events(self, umr_graph: UMRGraph) -> List[Dict[str, Any]]:
        """Extract event nodes from UMR graph."""
        events = []
        
        for node_id, node in umr_graph.nodes.items():
            if node.node_type == "event":
                # Determine if event is agentive
                is_agentive = self._is_agentive_event(node, umr_graph)
                event_type = "agentive" if is_agentive else "non_agentive"
                
                events.append({
                    "id": node_id,
                    "label": node.label,
                    "type": event_type,
                    "confidence": 0.9,  # UMR events are high confidence
                    "attributes": node.attributes
                })
        
        return events
    
    def _is_agentive_event(self, event_node: UMRNode, umr_graph: UMRGraph) -> bool:
        """Check if an event is agentive (has an agent argument)."""
        for edge in umr_graph.edges:
            if edge.source == event_node.id and edge.relation == "ARG0":
                # Check if ARG0 is animate/agentive
                arg_node = umr_graph.nodes.get(edge.target)
                if arg_node and arg_node.attributes.get("animate", False):
                    return True
        return False
    
    def _extract_arguments(self, umr_graph: UMRGraph) -> List[Dict[str, Any]]:
        """Extract argument roles from UMR graph."""
        args = []
        
        for edge in umr_graph.edges:
            if edge.relation.startswith("ARG"):
                # Map UMR argument to EIL role
                eil_role = self.mappings["arguments"].get(edge.relation, "PATIENT")
                
                source_node = umr_graph.nodes.get(edge.source)
                target_node = umr_graph.nodes.get(edge.target)
                
                if source_node and target_node:
                    args.append({
                        "role": eil_role,
                        "event": source_node.label,
                        "argument": target_node.label,
                        "confidence": 0.85,
                        "source": "umr"
                    })
        
        return args
    
    def _extract_scopes(self, umr_graph: UMRGraph) -> List[Dict[str, Any]]:
        """Extract scope information from UMR graph."""
        scopes = []
        
        for edge in umr_graph.edges:
            if edge.relation in ["scope", "mod"]:
                source_node = umr_graph.nodes.get(edge.source)
                target_node = umr_graph.nodes.get(edge.target)
                
                if source_node and target_node:
                    scopes.append({
                        "scope_type": edge.relation,
                        "scope_holder": source_node.label,
                        "scoped_element": target_node.label,
                        "confidence": 0.8
                    })
        
        return scopes
    
    def _extract_quantifiers(self, umr_graph: UMRGraph) -> List[Dict[str, Any]]:
        """Extract quantifier information from UMR graph."""
        quantifiers = []
        
        for node_id, node in umr_graph.nodes.items():
            if node.node_type == "quantifier":
                quant_type = node.attributes.get("quant_type", "some")
                eil_quant = self.mappings["quantifiers"].get(quant_type, "SOME")
                
                quantifiers.append({
                    "quantifier": eil_quant,
                    "scope": node.label,
                    "confidence": 0.9,
                    "attributes": node.attributes
                })
        
        # Also check for quantifier edges
        for edge in umr_graph.edges:
            if edge.relation == "quant":
                source_node = umr_graph.nodes.get(edge.source)
                target_node = umr_graph.nodes.get(edge.target)
                
                if source_node and target_node:
                    quantifiers.append({
                        "quantifier": "SOME",  # Default
                        "scope": target_node.label,
                        "quantified": source_node.label,
                        "confidence": 0.8
                    })
        
        return quantifiers
    
    def _extract_negation(self, umr_graph: UMRGraph) -> List[Dict[str, Any]]:
        """Extract negation information from UMR graph."""
        negations = []
        
        for node_id, node in umr_graph.nodes.items():
            if node.attributes.get("polarity") == "-":
                negations.append({
                    "negated": node.label,
                    "confidence": 0.95,
                    "scope": "local"
                })
        
        # Check for negation edges
        for edge in umr_graph.edges:
            if edge.relation == "polarity" and edge.attributes.get("value") == "-":
                source_node = umr_graph.nodes.get(edge.source)
                target_node = umr_graph.nodes.get(edge.target)
                
                if source_node and target_node:
                    negations.append({
                        "negated": target_node.label,
                        "confidence": 0.9,
                        "scope": "local"
                    })
        
        return negations
    
    def _extract_modality(self, umr_graph: UMRGraph) -> List[Dict[str, Any]]:
        """Extract modality information from UMR graph."""
        modality = []
        
        for node_id, node in umr_graph.nodes.items():
            if node.node_type == "modality":
                mod_type = node.attributes.get("mod_type", "can")
                eil_mod = self.mappings["modality"].get(mod_type, "CAN")
                
                modality.append({
                    "modality": eil_mod,
                    "scope": node.label,
                    "confidence": 0.9
                })
        
        return modality
    
    def _extract_spatial_roles(self, umr_graph: UMRGraph, language: str) -> List[Dict[str, Any]]:
        """Extract spatial role information from UMR graph."""
        spatial_roles = []
        
        for edge in umr_graph.edges:
            if edge.relation in ["location", "prep"]:
                source_node = umr_graph.nodes.get(edge.source)
                target_node = umr_graph.nodes.get(edge.target)
                
                if source_node and target_node:
                    # Get spatial relation from spatial maps
                    spatial_rel = self._get_spatial_relation(target_node.label, language)
                    
                    if spatial_rel:
                        spatial_roles.append({
                            "relation": spatial_rel,
                            "figure": source_node.label,
                            "ground": target_node.label,
                            "confidence": 0.85,
                            "source": "umr"
                        })
        
        return spatial_roles
    
    def _get_spatial_relation(self, prep_label: str, language: str) -> Optional[str]:
        """Get spatial relation from preposition label using spatial maps."""
        if language in self.spatial_maps:
            for relation, preps in self.spatial_maps[language].items():
                if prep_label.lower() in [p.lower() for p in preps]:
                    return relation
        return None
    
    def _extract_temporal(self, umr_graph: UMRGraph) -> List[Dict[str, Any]]:
        """Extract temporal information from UMR graph."""
        times = []
        
        for node_id, node in umr_graph.nodes.items():
            if node.node_type == "time":
                times.append({
                    "time_expression": node.label,
                    "confidence": 0.9,
                    "attributes": node.attributes
                })
        
        # Check for temporal edges
        for edge in umr_graph.edges:
            if edge.relation == "time":
                source_node = umr_graph.nodes.get(edge.source)
                target_node = umr_graph.nodes.get(edge.target)
                
                if source_node and target_node:
                    times.append({
                        "time_expression": target_node.label,
                        "event": source_node.label,
                        "confidence": 0.8
                    })
        
        return times

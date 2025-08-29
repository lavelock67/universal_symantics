"""
Prime Sparsifier - Remove unlicensed global primes that aren't grounded by EIL edges.

This module implements post-processing to clean up spurious primes that are emitted
as global fillers rather than being properly grounded in the semantic structure.
"""

from typing import Set, Dict, Any, List
from dataclasses import dataclass

# EILGraph is defined in generator.py, so we'll use a type hint
EILGraph = Any  # Type hint for EILGraph
from .timing import timed_method


@dataclass
class PrimeSparsifierConfig:
    """Configuration for prime sparsification rules."""
    
    # Primes that should only be emitted when grounded by specific EIL edges
    licensed_primes: Set[str] = None
    
    # Primes that should never be emitted as global fillers
    forbidden_global_primes: Set[str] = None
    
    # EIL edge types that license specific primes
    prime_licenses: Dict[str, Set[str]] = None
    
    def __post_init__(self):
        if self.licensed_primes is None:
            self.licensed_primes = {
                "TIME", "PLACE", "WAY", "SOMETHING", "SOMEONE", "THING"
            }
        
        if self.forbidden_global_primes is None:
            self.forbidden_global_primes = {
                "SOMETHING"  # These should be nodes, not global primes
            }
        
        if self.prime_licenses is None:
            self.prime_licenses = {
                "TIME": {"temporal", "time_modifier", "when"},
                "PLACE": {"spatial", "location", "where"},
                "WAY": {"manner", "how"},
                "SOMETHING": {"object", "theme", "patient"},
                "SOMEONE": {"agent", "actor", "experiencer"},
                "THING": {"entity", "object", "concept"}
            }


class PrimeSparsifier:
    """
    Remove unlicensed global primes that aren't grounded by EIL edges.
    
    This ensures that primes like TIME, PLACE, WAY, SOMETHING are only emitted
    when they're properly licensed by the semantic structure, not as global fillers.
    """
    
    def __init__(self, config: PrimeSparsifierConfig = None):
        self.config = config or PrimeSparsifierConfig()
    
    @timed_method("sparsify", "semantic")
    def sparsify(self, graph: EILGraph) -> EILGraph:
        """
        Remove unlicensed primes from the graph.
        
        Args:
            graph: The EIL graph to clean up
            
        Returns:
            The cleaned EIL graph with unlicensed primes removed
        """
        if not graph:
            return graph
        
        # Get all primes currently in the graph
        current_primes = set(graph.get_primes())
        
        # Find primes that are licensed by EIL edges
        licensed_primes = self._find_licensed_primes(graph)
        
        # Remove unlicensed primes (but be more conservative)
        unlicensed_primes = current_primes - licensed_primes
        
        # Remove forbidden global primes
        forbidden_primes = current_primes & self.config.forbidden_global_primes
        
        # Remove spurious global primes that aren't properly licensed
        spurious_primes = {'PLACE', 'TIME', 'WAY', 'SOMETHING'}  # Keep SOMEONE and THING as they can be licensed
        primes_to_remove = (current_primes & spurious_primes) | forbidden_primes
        
        if primes_to_remove:
            print(f"🧹 PrimeSparsifier: Removing unlicensed primes: {primes_to_remove}")
            graph.remove_primes(primes_to_remove)
        
        return graph
    
    def _find_licensed_primes(self, graph: EILGraph) -> Set[str]:
        """
        Find primes that are licensed by EIL edges in the graph.
        
        Args:
            graph: The EIL graph to analyze
            
        Returns:
            Set of primes that are properly licensed
        """
        licensed_primes = set()
        
        # Analyze EIL edges to find licensed primes
        for edge in graph.get_edges():
            edge_type = edge.get('type', '')
            edge_label = edge.get('label', '')
            
            # Check if this edge licenses any primes
            for prime, license_types in self.config.prime_licenses.items():
                if edge_type in license_types or edge_label in license_types:
                    licensed_primes.add(prime)
        
        # Check nodes for entity types that license primes
        for node in graph.get_nodes():
            node_type = node.get('type', '')
            node_features = node.get('features', [])
            node_prime = node.get('prime', '')
            
            # Spatial primes are licensed by spatial patterns
            if node_prime in ['NEAR', 'INSIDE', 'ABOVE']:
                licensed_primes.add(node_prime)
            
            # Entity types that license specific primes
            if node_type == 'temporal' or 'time' in node_features:
                licensed_primes.add('TIME')
            elif node_type == 'spatial' or 'location' in node_features:
                licensed_primes.add('PLACE')
            elif node_type == 'manner' or 'way' in node_features:
                licensed_primes.add('WAY')
            elif node_type == 'agent' or 'actor' in node_features:
                licensed_primes.add('SOMEONE')
            elif node_type == 'object' or 'entity' in node_features:
                licensed_primes.add('THING')
        
        # Check for AGENT roles in SRL that license SOMEONE
        for edge in graph.get_edges():
            if edge.get('type') == 'agent' or edge.get('label') == 'AGENT':
                licensed_primes.add('SOMEONE')
        
        return licensed_primes
    
    def _is_temporal_expression(self, graph: EILGraph) -> bool:
        """Check if the graph contains explicit temporal expressions."""
        # Look for UD temporal modifiers
        for node in graph.get_nodes():
            if node.get('ud_dep') in ['obl:tmod', 'nmod:tmod']:
                return True
            if node.get('ner_type') == 'TIME':
                return True
        return False
    
    def _is_spatial_expression(self, graph: EILGraph) -> bool:
        """Check if the graph contains explicit spatial expressions."""
        # Look for locative nouns or spatial NER
        for node in graph.get_nodes():
            if node.get('ud_dep') in ['obl:loc', 'nmod:loc']:
                return True
            if node.get('ner_type') == 'LOCATION':
                return True
            # Check for spatial nouns
            if node.get('pos') == 'NOUN' and node.get('lemma') in [
                'place', 'location', 'area', 'region', 'zone',
                'lugar', 'sitio', 'zona', 'área',
                'lieu', 'endroit', 'zone', 'région',
                'Ort', 'Stelle', 'Bereich', 'Zone'
            ]:
                return True
        return False
    
    def _is_manner_expression(self, graph: EILGraph) -> bool:
        """Check if the graph contains explicit manner expressions."""
        # Look for manner adverbs or "way" constructions
        for node in graph.get_nodes():
            if node.get('ud_dep') == 'advmod' and node.get('pos') == 'ADV':
                return True
            # Check for "way" constructions
            if node.get('lemma') in ['way', 'manera', 'façon', 'Weise']:
                return True
        return False


def sparsify_primes(graph: EILGraph, config: PrimeSparsifierConfig = None) -> EILGraph:
    """
    Convenience function to sparsify primes in a graph.
    
    Args:
        graph: The EIL graph to clean up
        config: Optional configuration for sparsification rules
        
    Returns:
        The cleaned EIL graph
    """
    sparsifier = PrimeSparsifier(config)
    return sparsifier.sparsify(graph)

"""UMR generator for converting graphs back to natural language text.

This module provides generation capabilities to convert UMR graphs back to
natural language text for round-trip evaluation and translation.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict, deque

from .graph import UMRGraph, UMRNode, UMREdge

logger = logging.getLogger(__name__)


class UMRGenerator:
    """Generator for converting UMR graphs to natural language text."""
    
    def __init__(self, language: str = "en"):
        """Initialize the UMR generator.
        
        Args:
            language: Target language code ('en', 'es', 'fr')
        """
        self.language = language
        
        # Language-specific templates and patterns
        self.templates = self._load_templates()
        
        # Surface form mappings for common concepts
        self.surface_forms = {
            'en': {
                'person': 'person',
                'thing': 'thing',
                'place': 'place',
                'time': 'time',
                'action': 'action',
                'property': 'property'
            },
            'es': {
                'person': 'persona',
                'thing': 'cosa',
                'place': 'lugar',
                'time': 'tiempo',
                'action': 'acción',
                'property': 'propiedad'
            },
            'fr': {
                'person': 'personne',
                'thing': 'chose',
                'place': 'lieu',
                'time': 'temps',
                'action': 'action',
                'property': 'propriété'
            }
        }
        
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load language-specific generation templates.
        
        Returns:
            Dictionary of templates by language
        """
        return {
            'en': {
                'event_arg0': "{event} {arg0}",
                'event_arg1': "{event} {arg1}",
                'event_arg0_arg1': "{event} {arg0} {arg1}",
                'property_mod': "{concept} {property}",
                'quant_mod': "{quant} {concept}",
                'det_mod': "{det} {concept}",
                'prep_mod': "{concept} {prep} {target}",
                'neg_mod': "not {content}",
                'coord': "{left} and {right}",
                'appos': "{concept}, {appos}"
            },
            'es': {
                'event_arg0': "{arg0} {event}",
                'event_arg1': "{event} {arg1}",
                'event_arg0_arg1': "{arg0} {event} {arg1}",
                'property_mod': "{concept} {property}",
                'quant_mod': "{quant} {concept}",
                'det_mod': "{det} {concept}",
                'prep_mod': "{concept} {prep} {target}",
                'neg_mod': "no {content}",
                'coord': "{left} y {right}",
                'appos': "{concept}, {appos}"
            },
            'fr': {
                'event_arg0': "{arg0} {event}",
                'event_arg1': "{event} {arg1}",
                'event_arg0_arg1': "{arg0} {event} {arg1}",
                'property_mod': "{concept} {property}",
                'quant_mod': "{quant} {concept}",
                'det_mod': "{det} {concept}",
                'prep_mod': "{concept} {prep} {target}",
                'neg_mod': "ne pas {content}",
                'coord': "{left} et {right}",
                'appos': "{concept}, {appos}"
            }
        }
        
    def generate_text(self, graph: UMRGraph) -> str:
        """Generate text from UMR graph.
        
        Args:
            graph: UMR graph to convert to text
            
        Returns:
            Generated text string
        """
        if not graph.nodes:
            return ""
            
        # Find root nodes (nodes with no incoming edges)
        root_nodes = self._find_root_nodes(graph)
        
        if not root_nodes:
            # If no clear roots, use nodes with most outgoing edges
            root_nodes = self._find_central_nodes(graph)
            
        # Generate text for each root
        sentences = []
        for root in root_nodes:
            sentence = self._generate_sentence(graph, root)
            if sentence:
                sentences.append(sentence)
                
        return " ".join(sentences)
        
    def _find_root_nodes(self, graph: UMRGraph) -> List[str]:
        """Find root nodes (nodes with no incoming edges).
        
        Args:
            graph: UMR graph
            
        Returns:
            List of root node IDs
        """
        root_nodes = []
        for node_id in graph.nodes:
            incoming = graph.get_neighbors(node_id, "in")
            if not incoming:
                root_nodes.append(node_id)
        return root_nodes
        
    def _find_central_nodes(self, graph: UMRGraph) -> List[str]:
        """Find central nodes (nodes with most outgoing edges).
        
        Args:
            graph: UMR graph
            
        Returns:
            List of central node IDs
        """
        node_scores = {}
        for node_id in graph.nodes:
            outgoing = graph.get_neighbors(node_id, "out")
            node_scores[node_id] = len(outgoing)
            
        # Sort by score and return top nodes
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        return [node_id for node_id, score in sorted_nodes[:3]]
        
    def _generate_sentence(self, graph: UMRGraph, root_id: str) -> str:
        """Generate a sentence starting from a root node.
        
        Args:
            graph: UMR graph
            root_id: Root node ID
            
        Returns:
            Generated sentence
        """
        root_node = graph.get_node(root_id)
        if not root_node:
            return ""
            
        # Build sentence structure based on node type
        if root_node.node_type == "event":
            return self._generate_event_sentence(graph, root_id)
        elif root_node.node_type == "concept":
            return self._generate_concept_sentence(graph, root_id)
        elif root_node.node_type == "property":
            return self._generate_property_sentence(graph, root_id)
        else:
            return self._generate_simple_sentence(graph, root_id)
            
    def _generate_event_sentence(self, graph: UMRGraph, event_id: str) -> str:
        """Generate sentence for event nodes.
        
        Args:
            graph: UMR graph
            event_id: Event node ID
            
        Returns:
            Generated sentence
        """
        event_node = graph.get_node(event_id)
        if not event_node:
            return ""
            
        # Find arguments
        neighbors = graph.get_neighbors(event_id, "out")
        args = {}
        
        for neighbor_id, relation in neighbors:
            if relation.startswith("ARG"):
                args[relation] = neighbor_id
                
        # Generate based on available arguments
        templates = self.templates.get(self.language, {})
        
        if "ARG0" in args and "ARG1" in args:
            arg0_text = self._generate_node_text(graph, args["ARG0"])
            arg1_text = self._generate_node_text(graph, args["ARG1"])
            event_text = self._get_surface_form(event_node)
            
            return templates.get("event_arg0_arg1", "{event} {arg0} {arg1}").format(
                event=event_text, arg0=arg0_text, arg1=arg1_text
            )
        elif "ARG0" in args:
            arg0_text = self._generate_node_text(graph, args["ARG0"])
            event_text = self._get_surface_form(event_node)
            
            return templates.get("event_arg0", "{event} {arg0}").format(
                event=event_text, arg0=arg0_text
            )
        elif "ARG1" in args:
            arg1_text = self._generate_node_text(graph, args["ARG1"])
            event_text = self._get_surface_form(event_node)
            
            return templates.get("event_arg1", "{event} {arg1}").format(
                event=event_text, arg1=arg1_text
            )
        else:
            return self._get_surface_form(event_node)
            
    def _generate_concept_sentence(self, graph: UMRGraph, concept_id: str) -> str:
        """Generate sentence for concept nodes.
        
        Args:
            graph: UMR graph
            concept_id: Concept node ID
            
        Returns:
            Generated sentence
        """
        concept_node = graph.get_node(concept_id)
        if not concept_node:
            return ""
            
        # Find modifiers
        neighbors = graph.get_neighbors(concept_id, "in")
        modifiers = []
        
        for neighbor_id, relation in neighbors:
            if relation in ["mod", "det", "quant"]:
                modifier_text = self._generate_node_text(graph, neighbor_id)
                modifiers.append((relation, modifier_text))
                
        # Generate concept with modifiers
        concept_text = self._get_surface_form(concept_node)
        
        # Add modifiers in order
        for relation, modifier_text in modifiers:
            if relation == "det":
                concept_text = f"{modifier_text} {concept_text}"
            elif relation == "quant":
                concept_text = f"{modifier_text} {concept_text}"
            elif relation == "mod":
                concept_text = f"{concept_text} {modifier_text}"
                
        return concept_text
        
    def _generate_property_sentence(self, graph: UMRGraph, property_id: str) -> str:
        """Generate sentence for property nodes.
        
        Args:
            graph: UMR graph
            property_id: Property node ID
            
        Returns:
            Generated sentence
        """
        property_node = graph.get_node(property_id)
        if not property_node:
            return ""
            
        # Find what this property modifies
        neighbors = graph.get_neighbors(property_id, "out")
        modified_concepts = []
        
        for neighbor_id, relation in neighbors:
            if relation == "mod":
                concept_text = self._generate_node_text(graph, neighbor_id)
                modified_concepts.append(concept_text)
                
        if modified_concepts:
            property_text = self._get_surface_form(property_node)
            templates = self.templates.get(self.language, {})
            
            return templates.get("property_mod", "{concept} {property}").format(
                concept=modified_concepts[0], property=property_text
            )
        else:
            return self._get_surface_form(property_node)
            
    def _generate_simple_sentence(self, graph: UMRGraph, node_id: str) -> str:
        """Generate simple sentence for any node type.
        
        Args:
            graph: UMR graph
            node_id: Node ID
            
        Returns:
            Generated sentence
        """
        node = graph.get_node(node_id)
        if not node:
            return ""
            
        return self._get_surface_form(node)
        
    def _generate_node_text(self, graph: UMRGraph, node_id: str) -> str:
        """Generate text for a specific node.
        
        Args:
            graph: UMR graph
            node_id: Node ID
            
        Returns:
            Generated text for the node
        """
        node = graph.get_node(node_id)
        if not node:
            return ""
            
        # Use surface form if available
        if node.surface_form:
            return node.surface_form
            
        # Otherwise use label or fallback
        return self._get_surface_form(node)
        
    def _get_surface_form(self, node: UMRNode) -> str:
        """Get surface form for a node.
        
        Args:
            node: UMR node
            
        Returns:
            Surface form string
        """
        # Use surface form if available
        if node.surface_form:
            return node.surface_form
            
        # Use label
        if node.label:
            return node.label
            
        # Fallback to type-based surface form
        surface_forms = self.surface_forms.get(self.language, {})
        return surface_forms.get(node.node_type, node.node_type)
        
    def generate_batch(self, graphs: List[UMRGraph]) -> List[str]:
        """Generate text for multiple UMR graphs.
        
        Args:
            graphs: List of UMR graphs
            
        Returns:
            List of generated text strings
        """
        texts = []
        for graph in graphs:
            try:
                text = self.generate_text(graph)
                texts.append(text)
            except Exception as e:
                logger.error(f"Error generating text for graph {graph.graph_id}: {e}")
                texts.append("")
                
        return texts

# TODO: Improve text generation quality with more sophisticated templates
# TODO: Add support for complex sentence structures and discourse markers
# TODO: Enhance surface form mappings with more comprehensive vocabulary
# TODO: Add support for morphological variations and agreement
# TODO: Implement better sentence ordering and coherence

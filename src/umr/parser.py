"""UMR parser for converting text to Uniform Meaning Representation graphs.

This module provides parsing capabilities to convert natural language text
into UMR graphs using dependency parsing and semantic analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
import spacy
import logging
from collections import defaultdict
import re

from .graph import UMRGraph, UMRNode, UMREdge

logger = logging.getLogger(__name__)


class UMRParser:
    """Parser for converting text to UMR graphs."""
    
    def __init__(self, language: str = "en"):
        """Initialize the UMR parser.
        
        Args:
            language: Language code ('en', 'es', 'fr')
        """
        self.language = language
        self.nlp = self._load_spacy_model(language)
        
        # UMR-specific patterns and mappings
        self.event_verbs = {
            'en': {'run', 'walk', 'eat', 'sleep', 'work', 'play', 'read', 'write'},
            'es': {'correr', 'caminar', 'comer', 'dormir', 'trabajar', 'jugar', 'leer', 'escribir'},
            'fr': {'courir', 'marcher', 'manger', 'dormir', 'travailler', 'jouer', 'lire', 'écrire'}
        }
        
        self.property_adjectives = {
            'en': {'big', 'small', 'red', 'blue', 'fast', 'slow', 'good', 'bad'},
            'es': {'grande', 'pequeño', 'rojo', 'azul', 'rápido', 'lento', 'bueno', 'malo'},
            'fr': {'grand', 'petit', 'rouge', 'bleu', 'rapide', 'lent', 'bon', 'mauvais'}
        }
        
        # UMR relation mappings from UD dependencies
        self.dep_to_umr = {
            'nsubj': 'ARG0',      # nominal subject
            'nsubjpass': 'ARG0',  # passive nominal subject
            'dobj': 'ARG1',       # direct object
            'iobj': 'ARG2',       # indirect object
            'prep': 'mod',        # prepositional modifier
            'amod': 'mod',        # adjectival modifier
            'advmod': 'mod',      # adverbial modifier
            'nummod': 'quant',    # numeric modifier
            'det': 'det',         # determiner
            'aux': 'aux',         # auxiliary verb
            'cop': 'cop',         # copula
            'mark': 'mark',       # marker
            'cc': 'coord',        # coordination
            'conj': 'coord',      # conjunction
            'appos': 'appos',     # appositional modifier
            'compound': 'compound', # compound
            'case': 'case',       # case marker
            'obl': 'obl',         # oblique nominal
            'advcl': 'advcl',     # adverbial clause modifier
            'acl': 'acl',         # adjectival clause modifier
            'relcl': 'relcl',     # relative clause modifier
        }
        
    def _load_spacy_model(self, language: str) -> Optional[spacy.language.Language]:
        """Load spaCy model for the specified language.
        
        Args:
            language: Language code
            
        Returns:
            spaCy language model or None if not available
        """
        models = {
            'en': 'en_core_web_sm',
            'es': 'es_core_news_sm', 
            'fr': 'fr_core_news_sm'
        }
        
        model = models.get(language)
        if not model:
            logger.warning(f"No spaCy model available for language: {language}")
            return None
            
        try:
            return spacy.load(model)
        except OSError:
            logger.warning(f"spaCy model {model} not installed for {language}")
            return None
            
    def parse_text(self, text: str) -> UMRGraph:
        """Parse text into a UMR graph.
        
        Args:
            text: Input text to parse
            
        Returns:
            UMRGraph representation of the text
        """
        if not self.nlp:
            logger.error(f"No spaCy model available for language: {self.language}")
            return UMRGraph()
            
        # Create UMR graph
        graph = UMRGraph(f"umr_{self.language}_{hash(text) % 10000}")
        graph.metadata.update({
            "source_text": text,
            "language": self.language
        })
        
        # Parse with spaCy
        doc = self.nlp(text)
        
        # Extract nodes and edges
        self._extract_nodes(doc, graph)
        self._extract_edges(doc, graph)
        
        return graph
        
    def _extract_nodes(self, doc: spacy.tokens.Doc, graph: UMRGraph) -> None:
        """Extract nodes from spaCy document.
        
        Args:
            doc: spaCy document
            graph: UMR graph to populate
        """
        for token in doc:
            # Determine node type
            node_type = self._get_node_type(token)
            
            # Create node
            node = UMRNode(
                id=f"n{token.i}",
                label=token.lemma_.lower(),
                node_type=node_type,
                surface_form=token.text,
                language=self.language,
                attributes={
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "is_sent_start": token.is_sent_start,
                    "is_sent_end": token.is_sent_end
                }
            )
            
            graph.add_node(node)
            
    def _get_node_type(self, token: spacy.tokens.Token) -> str:
        """Determine UMR node type from spaCy token.
        
        Args:
            token: spaCy token
            
        Returns:
            UMR node type
        """
        # Event nodes (verbs)
        if token.pos_ == "VERB":
            return "event"
            
        # Concept nodes (nouns, pronouns)
        elif token.pos_ in {"NOUN", "PROPN", "PRON"}:
            return "concept"
            
        # Property nodes (adjectives)
        elif token.pos_ == "ADJ":
            return "property"
            
        # Quantifier nodes (numbers, determiners)
        elif token.pos_ in {"NUM", "DET"}:
            return "quantifier"
            
        # Function nodes (prepositions, conjunctions, auxiliaries)
        elif token.pos_ in {"ADP", "CCONJ", "SCONJ", "AUX"}:
            return "function"
            
        # Default to concept
        else:
            return "concept"
            
    def _extract_edges(self, doc: spacy.tokens.Doc, graph: UMRGraph) -> None:
        """Extract edges from spaCy document.
        
        Args:
            doc: spaCy document
            graph: UMR graph to populate
        """
        for token in doc:
            # Get dependency relation
            if token.dep_ != "ROOT" and token.head != token:
                relation = self.dep_to_umr.get(token.dep_, token.dep_)
                
                # Create edge
                edge = UMREdge(
                    source=f"n{token.head.i}",
                    target=f"n{token.i}",
                    relation=relation,
                    attributes={
                        "dep": token.dep_,
                        "head_pos": token.head.pos_,
                        "child_pos": token.pos_
                    }
                )
                
                graph.add_edge(edge)
                
    def parse_batch(self, texts: List[str]) -> List[UMRGraph]:
        """Parse multiple texts into UMR graphs.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of UMRGraph representations
        """
        graphs = []
        for text in texts:
            try:
                graph = self.parse_text(text)
                graphs.append(graph)
            except Exception as e:
                logger.error(f"Error parsing text '{text[:50]}...': {e}")
                graphs.append(UMRGraph())
                
        return graphs
        
    def extract_primitive_patterns(self, graph: UMRGraph) -> Dict[str, List[Dict[str, Any]]]:
        """Extract primitive patterns from UMR graph.
        
        Args:
            graph: UMR graph to analyze
            
        Returns:
            Dictionary mapping primitive types to detected patterns
        """
        patterns = {
            "spatial": [],
            "temporal": [],
            "causal": [],
            "logical": [],
            "quantitative": [],
            "structural": []
        }
        
        # Extract spatial patterns (location, direction)
        spatial_nodes = graph.get_nodes_by_type("concept")
        for node in spatial_nodes:
            if any(spatial_word in node.label for spatial_word in 
                   ["location", "place", "direction", "position"]):
                patterns["spatial"].append({
                    "node_id": node.id,
                    "label": node.label,
                    "type": "location"
                })
                
        # Extract temporal patterns (time, duration)
        temporal_nodes = graph.get_nodes_by_type("concept")
        for node in temporal_nodes:
            if any(temp_word in node.label for temp_word in 
                   ["time", "duration", "moment", "period"]):
                patterns["temporal"].append({
                    "node_id": node.id,
                    "label": node.label,
                    "type": "time"
                })
                
        # Extract causal patterns (cause-effect relations)
        causal_edges = graph.get_edges_by_relation("ARG1")
        for edge in causal_edges:
            source_node = graph.get_node(edge.source)
            if source_node and source_node.node_type == "event":
                patterns["causal"].append({
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "type": "cause_effect"
                })
                
        # Extract logical patterns (negation, conjunction)
        logical_nodes = graph.get_nodes_by_type("function")
        for node in logical_nodes:
            if node.label in ["not", "and", "or", "if", "then"]:
                patterns["logical"].append({
                    "node_id": node.id,
                    "label": node.label,
                    "type": "logical_operator"
                })
                
        # Extract quantitative patterns (numbers, measurements)
        quant_nodes = graph.get_nodes_by_type("quantifier")
        for node in quant_nodes:
            if re.match(r'\d+', node.label):
                patterns["quantitative"].append({
                    "node_id": node.id,
                    "label": node.label,
                    "type": "number"
                })
                
        # Extract structural patterns (part-whole, possession)
        structural_edges = graph.get_edges_by_relation("mod")
        for edge in structural_edges:
            source_node = graph.get_node(edge.source)
            target_node = graph.get_node(edge.target)
            if (source_node and target_node and 
                source_node.node_type == "concept" and 
                target_node.node_type == "concept"):
                patterns["structural"].append({
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "type": "modification"
                })
                
        return patterns

# TODO: Add support for more languages (German, Italian, Portuguese)
# TODO: Enhance primitive pattern extraction with more sophisticated semantic analysis
# TODO: Add support for complex sentence structures and discourse relations
# TODO: Integrate with BabelNet for sense disambiguation
# TODO: Add support for temporal and aspectual information in UMR graphs

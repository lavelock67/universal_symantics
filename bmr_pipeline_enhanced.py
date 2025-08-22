#!/usr/bin/env python3
"""
Enhanced BMR (BabelNet Meaning Representation) Pipeline System.

This script implements comprehensive BMR parsing and generation capabilities:
1. BMR parsing from text using BabelNet synset linking and semantic analysis
2. BMR generation back to natural language with sense-aware text production
3. Cross-language BMR alignment and validation
4. Integration with existing NSM and UMR systems
5. Quality assessment and evaluation of BMR representations
6. Advanced semantic analysis with synset-based meaning representation
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.sense.linker import CachedBabelNetLinker
    from src.nsm.translate import NSMTranslator
    from src.nsm.explicator import NSMExplicator
    from src.table.schema import PeriodicTable
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    exit(1)


def convert_numpy_types(obj):
    """Convert numpy types and other non-serializable types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_numpy_types(obj.__dict__)
    else:
        return obj


class BMRNode:
    """Represents a node in a BMR graph."""
    
    def __init__(self, node_id: str, synset_id: str = None, lemma: str = None, 
                 pos: str = None, language: str = "en", surface_form: str = None):
        """Initialize a BMR node.
        
        Args:
            node_id: Unique identifier for the node
            synset_id: BabelNet synset ID
            lemma: Base form of the word
            pos: Part of speech
            language: Language code
            surface_form: Surface form of the word
        """
        self.node_id = node_id
        self.synset_id = synset_id
        self.lemma = lemma
        self.pos = pos
        self.language = language
        self.surface_form = surface_form or lemma
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            'node_id': self.node_id,
            'synset_id': self.synset_id,
            'lemma': self.lemma,
            'pos': self.pos,
            'language': self.language,
            'surface_form': self.surface_form,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BMRNode':
        """Create node from dictionary representation."""
        node = cls(
            node_id=data['node_id'],
            synset_id=data.get('synset_id'),
            lemma=data.get('lemma'),
            pos=data.get('pos'),
            language=data.get('language', 'en'),
            surface_form=data.get('surface_form')
        )
        node.metadata = data.get('metadata', {})
        return node


class BMREdge:
    """Represents an edge in a BMR graph."""
    
    def __init__(self, edge_id: str, source_id: str, target_id: str, 
                 relation_type: str, weight: float = 1.0):
        """Initialize a BMR edge.
        
        Args:
            edge_id: Unique identifier for the edge
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of semantic relation
            weight: Edge weight/confidence
        """
        self.edge_id = edge_id
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.weight = weight
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            'edge_id': self.edge_id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type,
            'weight': self.weight,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BMREdge':
        """Create edge from dictionary representation."""
        edge = cls(
            edge_id=data['edge_id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            relation_type=data['relation_type'],
            weight=data.get('weight', 1.0)
        )
        edge.metadata = data.get('metadata', {})
        return edge


class BMRGraph:
    """Represents a BabelNet Meaning Representation graph."""
    
    def __init__(self, graph_id: str = None):
        """Initialize a BMR graph.
        
        Args:
            graph_id: Unique identifier for the graph
        """
        self.graph_id = graph_id or f"bmr_{int(time.time())}"
        self.nodes: Dict[str, BMRNode] = {}
        self.edges: Dict[str, BMREdge] = {}
        self.metadata = {
            'source_text': '',
            'language': 'en',
            'creation_time': time.time()
        }
    
    def add_node(self, node: BMRNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: BMREdge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.edge_id] = edge
    
    def get_node(self, node_id: str) -> Optional[BMRNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edges_from(self, node_id: str) -> List[BMREdge]:
        """Get all edges from a node."""
        return [edge for edge in self.edges.values() if edge.source_id == node_id]
    
    def get_edges_to(self, node_id: str) -> List[BMREdge]:
        """Get all edges to a node."""
        return [edge for edge in self.edges.values() if edge.target_id == node_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            'graph_id': self.graph_id,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BMRGraph':
        """Create graph from dictionary representation."""
        graph = cls(data['graph_id'])
        graph.metadata = data.get('metadata', {})
        
        # Load nodes
        for node_id, node_data in data.get('nodes', {}).items():
            graph.add_node(BMRNode.from_dict(node_data))
        
        # Load edges
        for edge_id, edge_data in data.get('edges', {}).items():
            graph.add_edge(BMREdge.from_dict(edge_data))
        
        return graph


class EnhancedBMRParser:
    """Enhanced BMR parser for converting text to BMR graphs."""
    
    def __init__(self):
        """Initialize the enhanced BMR parser."""
        self.babelnet_linker = CachedBabelNetLinker()
        self.sbert_model = None
        self.nsm_translator = NSMTranslator()
        self.nsm_explicator = NSMExplicator()
        self.languages = ['en', 'es', 'fr']
        
        # Load periodic table
        try:
            with open("data/nsm_periodic_table.json", 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            self.periodic_table = PeriodicTable.from_dict(table_data)
        except Exception as e:
            logger.warning(f"Failed to load periodic table: {e}")
            self.periodic_table = PeriodicTable()
        
        # BMR parsing parameters
        self.parsing_params = {
            'min_synset_confidence': 0.3,
            'max_synsets_per_word': 5,
            'relation_confidence_threshold': 0.5,
            'semantic_similarity_threshold': 0.6
        }
        
        # Language-specific BMR mappings
        self.language_bmr_mappings = {
            'en': {
                'subject_relations': ['nsubj', 'nsubjpass'],
                'object_relations': ['dobj', 'iobj', 'pobj'],
                'modifier_relations': ['amod', 'advmod', 'nummod'],
                'preposition_relations': ['prep']
            },
            'es': {
                'subject_relations': ['nsubj', 'nsubjpass'],
                'object_relations': ['dobj', 'iobj', 'pobj'],
                'modifier_relations': ['amod', 'advmod', 'nummod'],
                'preposition_relations': ['prep']
            },
            'fr': {
                'subject_relations': ['nsubj', 'nsubjpass'],
                'object_relations': ['dobj', 'iobj', 'pobj'],
                'modifier_relations': ['amod', 'advmod', 'nummod'],
                'preposition_relations': ['prep']
            }
        }
        
        # BMR relation mappings
        self.bmr_relations = {
            'nsubj': 'AGENT',
            'dobj': 'PATIENT',
            'iobj': 'RECIPIENT',
            'amod': 'PROPERTY',
            'advmod': 'MANNER',
            'prep': 'LOCATION',
            'nummod': 'QUANTITY',
            'det': 'DETERMINER',
            'aux': 'AUXILIARY',
            'cop': 'COPULA'
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for BMR semantic analysis...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def parse_text_to_bmr(self, text: str, language: str = "en") -> BMRGraph:
        """Parse text into a BMR graph.
        
        Args:
            text: Input text to parse
            language: Language code
            
        Returns:
            BMRGraph representation of the text
        """
        logger.info(f"Parsing text to BMR: {text} ({language})")
        
        # Create BMR graph
        graph = BMRGraph()
        graph.metadata.update({
            'source_text': text,
            'language': language
        })
        
        # Extract key terms and link to BabelNet
        terms = self._extract_key_terms(text, language)
        synset_mappings = self._link_terms_to_babelnet(terms, language)
        
        # Create nodes for each term
        node_mappings = {}
        for term, synsets in synset_mappings.items():
            if synsets:
                # Create node with best synset
                best_synset = synsets[0]  # Assume first is best for now
                node = BMRNode(
                    node_id=f"node_{len(graph.nodes)}",
                    synset_id=best_synset,
                    lemma=term,
                    language=language,
                    surface_form=term
                )
                node.metadata['all_synsets'] = synsets
                graph.add_node(node)
                node_mappings[term] = node.node_id
        
        # Extract semantic relations
        relations = self._extract_semantic_relations(text, language, node_mappings)
        
        # Create edges for relations
        for relation in relations:
            edge = BMREdge(
                edge_id=f"edge_{len(graph.edges)}",
                source_id=relation['source'],
                target_id=relation['target'],
                relation_type=relation['type'],
                weight=relation.get('weight', 1.0)
            )
            edge.metadata['confidence'] = relation.get('confidence', 1.0)
            graph.add_edge(edge)
        
        # Add NSM primitive annotations
        self._add_nsm_annotations(graph, text, language)
        
        return graph
    
    def _extract_key_terms(self, text: str, language: str) -> List[str]:
        """Extract key terms from text for BabelNet linking."""
        # Simple tokenization - could be enhanced with proper NLP
        words = text.lower().split()
        
        # Filter out common stop words
        stop_words = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
            'es': {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'pero', 'en', 'a', 'de', 'con', 'por'},
            'fr': {'le', 'la', 'les', 'un', 'une', 'et', 'ou', 'mais', 'dans', 'à', 'de', 'avec', 'par'}
        }
        
        lang_stop_words = stop_words.get(language, stop_words['en'])
        key_terms = [word for word in words if word not in lang_stop_words and len(word) > 2]
        
        return list(set(key_terms))  # Remove duplicates
    
    def _link_terms_to_babelnet(self, terms: List[str], language: str) -> Dict[str, List[str]]:
        """Link terms to BabelNet synsets."""
        if not terms:
            return {}
        
        # Convert language code for BabelNet
        lang_code = {'en': 'EN', 'es': 'ES', 'fr': 'FR'}.get(language, language.upper())
        
        # Prepare terms for linking
        term_pairs = [(term, lang_code) for term in terms]
        
        # Link terms to BabelNet
        synset_mappings = self.babelnet_linker.link_terms(term_pairs)
        
        # Convert back to term -> synsets mapping
        result = {}
        for term, lang in term_pairs:
            synsets = synset_mappings.get((term, lang), [])
            if synsets:
                result[term] = synsets
        
        return result
    
    def _extract_semantic_relations(self, text: str, language: str, 
                                   node_mappings: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract semantic relations between terms."""
        relations = []
        
        # Simple pattern-based relation extraction
        # This could be enhanced with proper dependency parsing
        
        words = text.lower().split()
        lang_relations = self.language_bmr_mappings.get(language, self.language_bmr_mappings['en'])
        
        # Extract subject-verb-object patterns
        for i, word in enumerate(words):
            if word in node_mappings:
                # Look for potential relations
                if i > 0:  # Check for subject relations
                    prev_word = words[i-1]
                    if prev_word in node_mappings:
                        relations.append({
                            'source': node_mappings[prev_word],
                            'target': node_mappings[word],
                            'type': 'AGENT',
                            'confidence': 0.7
                        })
                
                if i < len(words) - 1:  # Check for object relations
                    next_word = words[i+1]
                    if next_word in node_mappings:
                        relations.append({
                            'source': node_mappings[word],
                            'target': node_mappings[next_word],
                            'type': 'PATIENT',
                            'confidence': 0.7
                        })
        
        return relations
    
    def _add_nsm_annotations(self, graph: BMRGraph, text: str, language: str) -> None:
        """Add NSM primitive annotations to BMR graph."""
        try:
            # Detect NSM primitives
            primitives = self.nsm_translator.detect_primitives_in_text(text, language)
            
            # Add primitive annotations to graph metadata
            graph.metadata['nsm_primitives'] = primitives
            
            # Add primitive nodes if not already present
            for i, primitive in enumerate(primitives):
                primitive_node_id = f"nsm_primitive_{i}"
                if primitive_node_id not in graph.nodes:
                    node = BMRNode(
                        node_id=primitive_node_id,
                        lemma=primitive,
                        pos='PRIMITIVE',
                        language=language,
                        surface_form=primitive
                    )
                    node.metadata['type'] = 'nsm_primitive'
                    graph.add_node(node)
        
        except Exception as e:
            logger.warning(f"Failed to add NSM annotations: {e}")


class EnhancedBMRGenerator:
    """Enhanced BMR generator for converting BMR graphs back to text."""
    
    def __init__(self):
        """Initialize the enhanced BMR generator."""
        self.babelnet_linker = CachedBabelNetLinker()
        self.sbert_model = None
        self.languages = ['en', 'es', 'fr']
        
        # BMR generation templates
        self.generation_templates = {
            'en': {
                'AGENT': '{agent} {action}',
                'PATIENT': '{action} {patient}',
                'PROPERTY': '{entity} is {property}',
                'LOCATION': '{entity} is in {location}',
                'MANNER': '{action} {manner}',
                'QUANTITY': '{quantity} {entity}'
            },
            'es': {
                'AGENT': '{agent} {action}',
                'PATIENT': '{action} {patient}',
                'PROPERTY': '{entity} es {property}',
                'LOCATION': '{entity} está en {location}',
                'MANNER': '{action} {manner}',
                'QUANTITY': '{quantity} {entity}'
            },
            'fr': {
                'AGENT': '{agent} {action}',
                'PATIENT': '{action} {patient}',
                'PROPERTY': '{entity} est {property}',
                'LOCATION': '{entity} est dans {location}',
                'MANNER': '{action} {manner}',
                'QUANTITY': '{quantity} {entity}'
            }
        }
        
        # Surface form mappings
        self.surface_forms = {
            'en': {
                'person': 'person',
                'thing': 'thing',
                'action': 'action',
                'property': 'property',
                'location': 'location'
            },
            'es': {
                'person': 'persona',
                'thing': 'cosa',
                'action': 'acción',
                'property': 'propiedad',
                'location': 'lugar'
            },
            'fr': {
                'person': 'personne',
                'thing': 'chose',
                'action': 'action',
                'property': 'propriété',
                'location': 'lieu'
            }
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for BMR generation...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def generate_text_from_bmr(self, graph: BMRGraph, target_language: str = "en") -> str:
        """Generate text from BMR graph.
        
        Args:
            graph: BMR graph to generate from
            target_language: Target language for generation
            
        Returns:
            Generated text
        """
        logger.info(f"Generating text from BMR graph: {graph.graph_id} ({target_language})")
        
        if not graph.nodes:
            return ""
        
        # Extract surface forms from nodes
        surface_forms = {}
        for node_id, node in graph.nodes.items():
            if node.surface_form:
                surface_forms[node_id] = node.surface_form
            elif node.lemma:
                surface_forms[node_id] = node.lemma
            else:
                # Use default surface form based on node type
                default_form = self._get_default_surface_form(node, target_language)
                surface_forms[node_id] = default_form
        
        # Generate text using templates
        generated_parts = []
        templates = self.generation_templates.get(target_language, self.generation_templates['en'])
        
        # Process edges to generate text parts
        for edge in graph.edges.values():
            source_form = surface_forms.get(edge.source_id, 'something')
            target_form = surface_forms.get(edge.target_id, 'something')
            
            if edge.relation_type in templates:
                template = templates[edge.relation_type]
                try:
                    part = template.format(
                        agent=source_form if edge.relation_type == 'AGENT' else target_form,
                        action=source_form if edge.relation_type in ['AGENT', 'PATIENT'] else target_form,
                        patient=target_form if edge.relation_type == 'PATIENT' else source_form,
                        entity=source_form,
                        property=target_form,
                        location=target_form,
                        manner=target_form,
                        quantity=source_form
                    )
                    generated_parts.append(part)
                except KeyError:
                    # Fallback to simple concatenation
                    generated_parts.append(f"{source_form} {target_form}")
            else:
                # Fallback for unknown relation types
                generated_parts.append(f"{source_form} {target_form}")
        
        # Combine parts
        if generated_parts:
            generated_text = " ".join(generated_parts)
        else:
            # Fallback: just use surface forms
            generated_text = " ".join(surface_forms.values())
        
        return generated_text
    
    def _get_default_surface_form(self, node: BMRNode, language: str) -> str:
        """Get default surface form for a node."""
        # Try to determine node type from metadata or synset
        node_type = 'thing'  # Default
        
        if node.metadata.get('type') == 'nsm_primitive':
            node_type = 'primitive'
        elif node.pos:
            if node.pos.startswith('V'):
                node_type = 'action'
            elif node.pos.startswith('N'):
                node_type = 'thing'
            elif node.pos.startswith('ADJ'):
                node_type = 'property'
        
        # Get default surface form
        default_forms = self.surface_forms.get(language, self.surface_forms['en'])
        return default_forms.get(node_type, default_forms['thing'])


class BMRQualityEvaluator:
    """Evaluates the quality of BMR representations."""
    
    def __init__(self):
        """Initialize the BMR quality evaluator."""
        self.sbert_model = None
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for BMR quality evaluation...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def evaluate_bmr_quality(self, original_text: str, generated_text: str, 
                           bmr_graph: BMRGraph) -> Dict[str, Any]:
        """Evaluate the quality of BMR representation and generation.
        
        Args:
            original_text: Original input text
            generated_text: Generated text from BMR
            bmr_graph: BMR graph representation
            
        Returns:
            Quality evaluation results
        """
        evaluation = {
            'semantic_similarity': 0.0,
            'graph_completeness': 0.0,
            'synset_coverage': 0.0,
            'relation_quality': 0.0,
            'overall_quality': 0.0
        }
        
        # Semantic similarity between original and generated text
        if self.sbert_model:
            try:
                embeddings = self.sbert_model.encode([original_text, generated_text])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                evaluation['semantic_similarity'] = float(similarity)
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
        
        # Graph completeness (nodes and edges)
        total_nodes = len(bmr_graph.nodes)
        total_edges = len(bmr_graph.edges)
        nodes_with_synsets = sum(1 for node in bmr_graph.nodes.values() if node.synset_id)
        
        if total_nodes > 0:
            evaluation['graph_completeness'] = min(1.0, (total_nodes + total_edges) / 10.0)
            evaluation['synset_coverage'] = nodes_with_synsets / total_nodes
        
        # Relation quality (based on edge weights)
        if total_edges > 0:
            avg_edge_weight = sum(edge.weight for edge in bmr_graph.edges.values()) / total_edges
            evaluation['relation_quality'] = avg_edge_weight
        
        # Overall quality (weighted combination)
        evaluation['overall_quality'] = (
            0.4 * evaluation['semantic_similarity'] +
            0.2 * evaluation['graph_completeness'] +
            0.2 * evaluation['synset_coverage'] +
            0.2 * evaluation['relation_quality']
        )
        
        return evaluation


def main():
    """Main function to run enhanced BMR pipeline."""
    logger.info("Starting enhanced BMR pipeline...")
    
    # Initialize components
    parser = EnhancedBMRParser()
    generator = EnhancedBMRGenerator()
    evaluator = BMRQualityEvaluator()
    
    # Test BMR pipeline examples
    test_examples = [
        {"text": "The book is on the table", "language": "en"},
        {"text": "Esta cosa está en este lugar", "language": "es"},
        {"text": "Cette chose fait partie de l'ensemble", "language": "fr"},
        {"text": "The red car is fast", "language": "en"}
    ]
    
    # Process test examples
    bmr_results = []
    for example in test_examples:
        text = example["text"]
        language = example["language"]
        
        print(f"\nBMR Pipeline: {text} ({language})")
        
        try:
            # Parse text to BMR
            bmr_graph = parser.parse_text_to_bmr(text, language)
            print(f"BMR Graph: {len(bmr_graph.nodes)} nodes, {len(bmr_graph.edges)} edges")
            
            # Generate text from BMR
            generated_text = generator.generate_text_from_bmr(bmr_graph, language)
            print(f"Generated: {generated_text}")
            
            # Evaluate quality
            quality = evaluator.evaluate_bmr_quality(text, generated_text, bmr_graph)
            print(f"Quality: {quality['overall_quality']:.3f}")
            
            bmr_results.append({
                'example': example,
                'bmr_graph': bmr_graph.to_dict(),
                'generated_text': generated_text,
                'quality_evaluation': quality
            })
            
        except Exception as e:
            logger.error(f"BMR pipeline failed for {text}: {e}")
            bmr_results.append({
                'example': example,
                'error': str(e)
            })
    
    # Save results
    output_path = "data/bmr_pipeline_enhanced_report.json"
    report = {
        "metadata": {
            "report_type": "enhanced_BMR_pipeline_report",
            "timestamp": "2025-08-22",
            "enhanced_features": [
                "babelnet_synset_linking",
                "semantic_relation_extraction",
                "cross_language_generation",
                "nsm_primitive_integration",
                "quality_evaluation",
                "graph_based_representation"
            ]
        },
        "bmr_results": bmr_results,
        "summary": {
            "total_examples": len(bmr_results),
            "successful_parses": sum(1 for r in bmr_results if 'bmr_graph' in r),
            "avg_quality": np.mean([r.get('quality_evaluation', {}).get('overall_quality', 0.0) 
                                   for r in bmr_results if 'quality_evaluation' in r])
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(report), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced BMR pipeline report saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ENHANCED BMR PIPELINE SUMMARY")
    print("="*80)
    print(f"Total Examples: {len(bmr_results)}")
    successful = sum(1 for r in bmr_results if 'bmr_graph' in r)
    print(f"Successful Parses: {successful}/{len(bmr_results)}")
    
    qualities = [r.get('quality_evaluation', {}).get('overall_quality', 0.0) 
                for r in bmr_results if 'quality_evaluation' in r]
    if qualities:
        print(f"Average Quality: {np.mean(qualities):.3f}")
        print(f"Quality Range: {min(qualities):.3f} - {max(qualities):.3f}")
    
    print("\nBMR Graph Statistics:")
    for result in bmr_results:
        if 'bmr_graph' in result:
            graph = result['bmr_graph']
            nodes = len(graph.get('nodes', {}))
            edges = len(graph.get('edges', {}))
            print(f"  {result['example']['text']}: {nodes} nodes, {edges} edges")
    
    print("="*80)


if __name__ == "__main__":
    main()

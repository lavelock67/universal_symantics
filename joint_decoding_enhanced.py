#!/usr/bin/env python3
"""
Enhanced Joint Decoding System.

This script implements a comprehensive joint decoding system to combine
NSM explications with graph-based representations for improved text generation
and semantic coherence.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import time
from collections import defaultdict, Counter
import re
import networkx as nx

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.nsm.translate import NSMTranslator
    from src.nsm.explicator import NSMExplicator
    from src.nsm.enhanced_explicator import EnhancedNSMExplicator
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
        # Convert tuple keys to strings
        converted_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                # Convert tuple key to string representation
                str_key = f"{key[0]}_{key[1]}" if len(key) == 2 else str(key)
                converted_dict[str_key] = convert_numpy_types(value)
            else:
                converted_dict[str(key)] = convert_numpy_types(value)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_numpy_types(obj.__dict__)
    else:
        return obj


class SemanticGraph:
    """Semantic graph representation for joint decoding."""
    
    def __init__(self):
        """Initialize the semantic graph."""
        self.graph = nx.DiGraph()
        self.node_types = {
            'entity': 'entity',
            'property': 'property',
            'action': 'action',
            'relation': 'relation',
            'concept': 'concept'
        }
        
        # Graph construction parameters
        self.graph_params = {
            'max_nodes': 50,
            'max_edges': 100,
            'similarity_threshold': 0.7,
            'weight_decay': 0.9
        }
    
    def add_node(self, node_id: str, node_type: str, attributes: Dict[str, Any] = None):
        """Add a node to the semantic graph."""
        if node_id not in self.graph:
            self.graph.add_node(node_id, type=node_type, attributes=attributes or {})
    
    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0):
        """Add an edge to the semantic graph."""
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, relation=relation, weight=weight)
    
    def build_from_text(self, text: str, primitives: List[str]) -> bool:
        """Build semantic graph from text and primitives."""
        try:
            # Clear existing graph
            self.graph.clear()
            
            # Add primitive nodes
            for i, primitive in enumerate(primitives):
                node_id = f"primitive_{i}"
                self.add_node(node_id, 'concept', {'primitive': primitive, 'text': text})
            
            # Add entity nodes from text
            words = text.split()
            for i, word in enumerate(words):
                if len(word) > 2:  # Skip short words
                    node_id = f"entity_{i}"
                    self.add_node(node_id, 'entity', {'word': word, 'position': i})
            
            # Add edges based on primitive relationships
            for i, primitive in enumerate(primitives):
                primitive_node = f"primitive_{i}"
                
                # Connect to relevant entities
                for j, word in enumerate(words):
                    if len(word) > 2:
                        entity_node = f"entity_{j}"
                        
                        # Simple relationship detection
                        if primitive.lower() in word.lower() or word.lower() in primitive.lower():
                            self.add_edge(primitive_node, entity_node, 'contains', 0.8)
                        elif j > 0 and j < len(words) - 1:
                            # Adjacent words
                            self.add_edge(entity_node, f"entity_{j-1}", 'precedes', 0.6)
                            self.add_edge(entity_node, f"entity_{j+1}", 'follows', 0.6)
            
            return True
        
        except Exception as e:
            logger.warning(f"Graph construction failed: {e}")
            return False
    
    def get_subgraph(self, node_ids: List[str]) -> nx.DiGraph:
        """Get subgraph containing specified nodes."""
        return self.graph.subgraph(node_ids).copy()
    
    def get_node_neighbors(self, node_id: str, max_depth: int = 2) -> List[str]:
        """Get neighbors of a node up to specified depth."""
        neighbors = set()
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_node, depth = queue.pop(0)
            
            if depth > max_depth or current_node in visited:
                continue
            
            visited.add(current_node)
            neighbors.add(current_node)
            
            if depth < max_depth:
                # Add neighbors
                for neighbor in self.graph.predecessors(current_node):
                    queue.append((neighbor, depth + 1))
                for neighbor in self.graph.successors(current_node):
                    queue.append((neighbor, depth + 1))
        
        return list(neighbors)
    
    def calculate_node_importance(self, node_id: str) -> float:
        """Calculate importance score for a node."""
        if node_id not in self.graph:
            return 0.0
        
        # Simple importance based on degree centrality
        in_degree = self.graph.in_degree(node_id)
        out_degree = self.graph.out_degree(node_id)
        
        return (in_degree + out_degree) / (len(self.graph.nodes()) + 1)


class NSMGraphDecoder:
    """NSM+Graph joint decoder for text generation."""
    
    def __init__(self):
        """Initialize the NSM+Graph joint decoder."""
        self.sbert_model = None
        self.nsm_translator = NSMTranslator()
        
        # Decoding parameters
        self.decoding_params = {
            'max_length': 100,
            'temperature': 0.8,
            'top_k': 10,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'nsm_weight': 0.6,
            'graph_weight': 0.4
        }
        
        # Generation templates
        self.generation_templates = {
            'entity_description': [
                "The {entity} is {property}",
                "{entity} has the characteristic of {property}",
                "One can observe that {entity} is {property}"
            ],
            'action_description': [
                "{entity} {action}",
                "The {entity} performs {action}",
                "One can see that {entity} {action}"
            ],
            'relationship_description': [
                "{entity1} is related to {entity2} through {relation}",
                "There is a {relation} between {entity1} and {entity2}",
                "{entity1} and {entity2} share a {relation}"
            ]
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for joint decoding...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def decode_joint(self, text: str, language: str = "en", 
                    conditioning: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform joint NSM+Graph decoding."""
        try:
            # Extract primitives
            primitives = self.nsm_translator.detect_primitives_in_text(text, language)
            
            # Build semantic graph
            semantic_graph = SemanticGraph()
            graph_success = semantic_graph.build_from_text(text, primitives)
            
            # Generate NSM explication
            nsm_explication = self._generate_nsm_explication(text, primitives, language)
            
            # Generate graph-based representation
            graph_representation = self._generate_graph_representation(
                semantic_graph, text, primitives, language
            )
            
            # Joint decoding
            joint_output = self._perform_joint_decoding(
                nsm_explication, graph_representation, text, language, conditioning
            )
            
            return {
                'text': text,
                'language': language,
                'primitives': primitives,
                'nsm_explication': nsm_explication,
                'graph_representation': graph_representation,
                'joint_output': joint_output,
                'graph_success': graph_success,
                'conditioning': conditioning or {}
            }
        
        except Exception as e:
            logger.warning(f"Joint decoding failed: {e}")
            return {
                'text': text,
                'language': language,
                'primitives': [],
                'nsm_explication': '',
                'graph_representation': '',
                'joint_output': f"error({text})",
                'graph_success': False,
                'conditioning': conditioning or {},
                'error': str(e)
            }
    
    def _generate_nsm_explication(self, text: str, primitives: List[str], language: str) -> str:
        """Generate NSM explication."""
        try:
            if not primitives:
                return f"unknown({text})"
            
            # Simple NSM explication
            if len(primitives) == 1:
                return f"{primitives[0]}({text})"
            else:
                return f"{' AND '.join(primitives)}({text})"
        
        except Exception as e:
            logger.warning(f"NSM explication generation failed: {e}")
            return f"error({text})"
    
    def _generate_graph_representation(self, semantic_graph: SemanticGraph, 
                                    text: str, primitives: List[str], 
                                    language: str) -> Dict[str, Any]:
        """Generate graph-based representation."""
        try:
            if not semantic_graph.graph.nodes():
                return {'nodes': [], 'edges': [], 'structure': 'empty'}
            
            # Extract graph structure
            nodes = []
            for node_id, node_data in semantic_graph.graph.nodes(data=True):
                nodes.append({
                    'id': node_id,
                    'type': node_data.get('type', 'unknown'),
                    'attributes': node_data.get('attributes', {}),
                    'importance': semantic_graph.calculate_node_importance(node_id)
                })
            
            edges = []
            for source, target, edge_data in semantic_graph.graph.edges(data=True):
                edges.append({
                    'source': source,
                    'target': target,
                    'relation': edge_data.get('relation', 'unknown'),
                    'weight': edge_data.get('weight', 1.0)
                })
            
            # Generate text representation
            graph_text = self._graph_to_text(semantic_graph, nodes, edges, text)
            
            return {
                'nodes': nodes,
                'edges': edges,
                'structure': 'connected' if edges else 'disconnected',
                'text_representation': graph_text,
                'node_count': len(nodes),
                'edge_count': len(edges)
            }
        
        except Exception as e:
            logger.warning(f"Graph representation generation failed: {e}")
            return {'nodes': [], 'edges': [], 'structure': 'error'}
    
    def _graph_to_text(self, semantic_graph: SemanticGraph, nodes: List[Dict], 
                      edges: List[Dict], original_text: str) -> str:
        """Convert graph structure to text representation."""
        try:
            if not nodes:
                return f"empty_graph({original_text})"
            
            # Find important nodes
            important_nodes = sorted(nodes, key=lambda x: x['importance'], reverse=True)[:3]
            
            # Generate descriptions
            descriptions = []
            for node in important_nodes:
                node_type = node['type']
                attributes = node['attributes']
                
                if node_type == 'entity' and 'word' in attributes:
                    descriptions.append(f"entity {attributes['word']}")
                elif node_type == 'concept' and 'primitive' in attributes:
                    descriptions.append(f"concept {attributes['primitive']}")
                elif node_type == 'property' and 'property' in attributes:
                    descriptions.append(f"property {attributes['property']}")
            
            # Add relationship descriptions
            for edge in edges[:5]:  # Limit to top 5 edges
                source_node = next((n for n in nodes if n['id'] == edge['source']), None)
                target_node = next((n for n in nodes if n['id'] == edge['target']), None)
                
                if source_node and target_node:
                    source_name = self._get_node_name(source_node)
                    target_name = self._get_node_name(target_node)
                    relation = edge['relation']
                    
                    descriptions.append(f"{source_name} {relation} {target_name}")
            
            return f"graph({'; '.join(descriptions)})"
        
        except Exception as e:
            logger.warning(f"Graph to text conversion failed: {e}")
            return f"graph_error({original_text})"
    
    def _get_node_name(self, node: Dict[str, Any]) -> str:
        """Get readable name for a node."""
        attributes = node.get('attributes', {})
        
        if 'word' in attributes:
            return attributes['word']
        elif 'primitive' in attributes:
            return attributes['primitive']
        elif 'property' in attributes:
            return attributes['property']
        else:
            return node['id']
    
    def _perform_joint_decoding(self, nsm_explication: str, graph_representation: Dict[str, Any],
                              text: str, language: str, conditioning: Dict[str, Any]) -> str:
        """Perform joint decoding combining NSM and graph information."""
        try:
            # Extract conditioning parameters
            style = conditioning.get('style', 'descriptive')
            focus = conditioning.get('focus', 'balanced')
            length = conditioning.get('length', 'medium')
            
            # Get graph text representation
            graph_text = graph_representation.get('text_representation', '')
            
            # Combine NSM and graph information
            if focus == 'nsm':
                base_text = nsm_explication
                graph_weight = 0.2
            elif focus == 'graph':
                base_text = graph_text
                graph_weight = 0.8
            else:  # balanced
                base_text = f"{nsm_explication} AND {graph_text}"
                graph_weight = 0.5
            
            # Apply style conditioning
            if style == 'formal':
                joint_output = self._apply_formal_style(base_text, text)
            elif style == 'casual':
                joint_output = self._apply_casual_style(base_text, text)
            elif style == 'technical':
                joint_output = self._apply_technical_style(base_text, text)
            else:  # descriptive
                joint_output = self._apply_descriptive_style(base_text, text)
            
            # Apply length conditioning
            joint_output = self._apply_length_conditioning(joint_output, length)
            
            return joint_output
        
        except Exception as e:
            logger.warning(f"Joint decoding failed: {e}")
            return f"joint_error({text})"
    
    def _apply_formal_style(self, base_text: str, original_text: str) -> str:
        """Apply formal style to joint output."""
        return f"Formally, {base_text} represents the semantic structure of '{original_text}'"
    
    def _apply_casual_style(self, base_text: str, original_text: str) -> str:
        """Apply casual style to joint output."""
        return f"So basically, {base_text} is what '{original_text}' means"
    
    def _apply_technical_style(self, base_text: str, original_text: str) -> str:
        """Apply technical style to joint output."""
        return f"Technical analysis: {base_text} | Input: '{original_text}'"
    
    def _apply_descriptive_style(self, base_text: str, original_text: str) -> str:
        """Apply descriptive style to joint output."""
        return f"The text '{original_text}' can be understood as {base_text}"
    
    def _apply_length_conditioning(self, text: str, length: str) -> str:
        """Apply length conditioning to output."""
        words = text.split()
        
        if length == 'short' and len(words) > 10:
            return ' '.join(words[:10]) + "..."
        elif length == 'long' and len(words) < 20:
            # Add more descriptive elements
            return text + " with additional semantic context and structural information"
        else:
            return text


class JointDecodingEvaluator:
    """Evaluator for joint decoding quality."""
    
    def __init__(self):
        """Initialize the joint decoding evaluator."""
        self.sbert_model = None
        
        # Evaluation parameters
        self.eval_params = {
            'coherence_weight': 0.4,
            'fluency_weight': 0.3,
            'semantic_weight': 0.3,
            'min_score': 0.0,
            'max_score': 1.0
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for joint decoding evaluation...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def evaluate_joint_output(self, original_text: str, joint_output: str, 
                            nsm_explication: str, graph_representation: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate quality of joint decoding output."""
        try:
            # Calculate coherence score
            coherence_score = self._calculate_coherence(joint_output, nsm_explication, graph_representation)
            
            # Calculate fluency score
            fluency_score = self._calculate_fluency(joint_output)
            
            # Calculate semantic similarity
            semantic_score = self._calculate_semantic_similarity(original_text, joint_output)
            
            # Calculate overall score
            overall_score = (
                coherence_score * self.eval_params['coherence_weight'] +
                fluency_score * self.eval_params['fluency_weight'] +
                semantic_score * self.eval_params['semantic_weight']
            )
            
            return {
                'coherence_score': coherence_score,
                'fluency_score': fluency_score,
                'semantic_score': semantic_score,
                'overall_score': overall_score,
                'quality_level': self._get_quality_level(overall_score)
            }
        
        except Exception as e:
            logger.warning(f"Joint output evaluation failed: {e}")
            return {
                'coherence_score': 0.5,
                'fluency_score': 0.5,
                'semantic_score': 0.5,
                'overall_score': 0.5,
                'quality_level': 'fair'
            }
    
    def _calculate_coherence(self, joint_output: str, nsm_explication: str, 
                           graph_representation: Dict[str, Any]) -> float:
        """Calculate coherence score of joint output."""
        try:
            # Check if joint output incorporates both NSM and graph information
            nsm_present = any(word in joint_output.lower() for word in nsm_explication.lower().split())
            graph_present = graph_representation.get('structure', '') != 'empty'
            
            # Base coherence score
            base_score = 0.5
            
            if nsm_present:
                base_score += 0.25
            
            if graph_present:
                base_score += 0.25
            
            # Check for logical flow
            if 'AND' in joint_output or 'represents' in joint_output or 'means' in joint_output:
                base_score += 0.1
            
            return min(base_score, 1.0)
        
        except Exception as e:
            logger.warning(f"Coherence calculation failed: {e}")
            return 0.5
    
    def _calculate_fluency(self, joint_output: str) -> float:
        """Calculate fluency score of joint output."""
        if not joint_output:
            return 0.0
        
        # Simple fluency metrics
        words = joint_output.split()
        
        # Length penalty
        length_score = 1.0
        if len(words) < 5:
            length_score = 0.6
        elif len(words) > 30:
            length_score = 0.8
        
        # Grammar score
        grammar_score = 1.0
        if joint_output.startswith('error(') or joint_output.startswith('unknown('):
            grammar_score = 0.3
        elif joint_output.count('AND') > 3:
            grammar_score = 0.7
        
        return (length_score + grammar_score) / 2
    
    def _calculate_semantic_similarity(self, original_text: str, joint_output: str) -> float:
        """Calculate semantic similarity between original text and joint output."""
        if not self.sbert_model:
            return 0.5
        
        try:
            # Calculate embeddings
            embeddings = self.sbert_model.encode([original_text, joint_output])
            
            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return max(0.0, float(similarity))
        
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.5
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level based on score."""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'


class EnhancedJointDecodingSystem:
    """Enhanced joint decoding system with comprehensive analysis."""
    
    def __init__(self):
        """Initialize the enhanced joint decoding system."""
        self.decoder = NSMGraphDecoder()
        self.evaluator = JointDecodingEvaluator()
        
        # System parameters
        self.system_params = {
            'min_quality_threshold': 0.5,
            'max_generation_attempts': 3,
            'conditioning_options': ['descriptive', 'formal', 'casual', 'technical'],
            'focus_options': ['balanced', 'nsm', 'graph'],
            'length_options': ['short', 'medium', 'long']
        }
    
    def run_joint_decoding_analysis(self, test_texts: List[str], 
                                  languages: List[str] = ["en", "es", "fr"]) -> Dict[str, Any]:
        """Run comprehensive joint decoding analysis."""
        logger.info(f"Running joint decoding analysis for {len(test_texts)} texts")
        
        analysis_results = {
            'test_configuration': {
                'num_test_texts': len(test_texts),
                'languages': languages,
                'timestamp': time.time()
            },
            'decoding_results': [],
            'evaluation_results': [],
            'conditioning_analysis': {},
            'recommendations': []
        }
        
        # Test different conditioning options
        for style in self.system_params['conditioning_options']:
            for focus in self.system_params['focus_options']:
                for length in self.system_params['length_options']:
                    conditioning = {
                        'style': style,
                        'focus': focus,
                        'length': length
                    }
                    
                    for language in languages:
                        for text in test_texts:
                            # Perform joint decoding
                            decoding_result = self.decoder.decode_joint(text, language, conditioning)
                            analysis_results['decoding_results'].append(decoding_result)
                            
                            # Evaluate output
                            evaluation_result = self.evaluator.evaluate_joint_output(
                                text, 
                                decoding_result['joint_output'],
                                decoding_result['nsm_explication'],
                                decoding_result['graph_representation']
                            )
                            analysis_results['evaluation_results'].append(evaluation_result)
        
        # Analyze results
        analysis_results['conditioning_analysis'] = self._analyze_conditioning_results(
            analysis_results['decoding_results'],
            analysis_results['evaluation_results']
        )
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_decoding_recommendations(
            analysis_results['conditioning_analysis']
        )
        
        return analysis_results
    
    def _analyze_conditioning_results(self, decoding_results: List[Dict[str, Any]], 
                                   evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results across different conditioning options."""
        analysis = {
            'total_decodings': len(decoding_results),
            'successful_decodings': 0,
            'avg_overall_score': 0.0,
            'best_conditioning': {},
            'style_performance': defaultdict(list),
            'focus_performance': defaultdict(list),
            'length_performance': defaultdict(list),
            'quality_distribution': defaultdict(int)
        }
        
        overall_scores = []
        
        for i, decoding_result in enumerate(decoding_results):
            evaluation_result = evaluation_results[i] if i < len(evaluation_results) else {}
            
            if decoding_result.get('joint_output', '').startswith('error('):
                continue
            
            analysis['successful_decodings'] += 1
            
            # Collect scores by conditioning
            conditioning = decoding_result.get('conditioning', {})
            style = conditioning.get('style', 'unknown')
            focus = conditioning.get('focus', 'unknown')
            length = conditioning.get('length', 'unknown')
            
            overall_score = evaluation_result.get('overall_score', 0.0)
            overall_scores.append(overall_score)
            
            analysis['style_performance'][style].append(overall_score)
            analysis['focus_performance'][focus].append(overall_score)
            analysis['length_performance'][length].append(overall_score)
            
            # Quality distribution
            quality_level = evaluation_result.get('quality_level', 'fair')
            analysis['quality_distribution'][quality_level] += 1
        
        # Calculate averages
        if overall_scores:
            analysis['avg_overall_score'] = np.mean(overall_scores)
        
        # Find best conditioning
        for style in analysis['style_performance']:
            if analysis['style_performance'][style]:
                avg_score = np.mean(analysis['style_performance'][style])
                if 'style' not in analysis['best_conditioning'] or avg_score > analysis['best_conditioning']['style']['score']:
                    analysis['best_conditioning']['style'] = {'option': style, 'score': avg_score}
        
        for focus in analysis['focus_performance']:
            if analysis['focus_performance'][focus]:
                avg_score = np.mean(analysis['focus_performance'][focus])
                if 'focus' not in analysis['best_conditioning'] or avg_score > analysis['best_conditioning']['focus']['score']:
                    analysis['best_conditioning']['focus'] = {'option': focus, 'score': avg_score}
        
        for length in analysis['length_performance']:
            if analysis['length_performance'][length]:
                avg_score = np.mean(analysis['length_performance'][length])
                if 'length' not in analysis['best_conditioning'] or avg_score > analysis['best_conditioning']['length']['score']:
                    analysis['best_conditioning']['length'] = {'option': length, 'score': avg_score}
        
        return analysis
    
    def _generate_decoding_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for joint decoding."""
        recommendations = []
        
        # Success rate recommendations
        success_rate = analysis['successful_decodings'] / analysis['total_decodings']
        if success_rate < 0.8:
            recommendations.append(f"Low success rate ({success_rate:.1%}) - improve decoding robustness")
        
        # Quality recommendations
        if analysis['avg_overall_score'] < 0.6:
            recommendations.append("Low average quality - improve joint decoding algorithms")
        
        # Conditioning recommendations
        best_style = analysis['best_conditioning'].get('style', {}).get('option', 'unknown')
        best_focus = analysis['best_conditioning'].get('focus', {}).get('option', 'unknown')
        best_length = analysis['best_conditioning'].get('length', {}).get('option', 'unknown')
        
        recommendations.append(f"Best style: {best_style} - consider prioritizing this style")
        recommendations.append(f"Best focus: {best_focus} - consider prioritizing this focus")
        recommendations.append(f"Best length: {best_length} - consider prioritizing this length")
        
        # Quality distribution recommendations
        quality_dist = analysis['quality_distribution']
        if quality_dist.get('excellent', 0) < quality_dist.get('poor', 0):
            recommendations.append("More poor quality outputs than excellent - improve generation quality")
        
        return recommendations


def main():
    """Main function to run enhanced joint decoding."""
    logger.info("Starting enhanced joint decoding...")
    
    # Initialize joint decoding system
    joint_system = EnhancedJointDecodingSystem()
    
    # Test texts
    test_texts = [
        "The red car is parked near the building",
        "The cat is on the mat",
        "This is similar to that",
        "The book contains important information",
        "The weather is cold today"
    ]
    
    # Run joint decoding analysis
    analysis_results = joint_system.run_joint_decoding_analysis(test_texts, ["en", "es", "fr"])
    
    # Print results
    print("\n" + "="*80)
    print("ENHANCED JOINT DECODING RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Texts: {analysis_results['test_configuration']['num_test_texts']}")
    print(f"  Languages: {analysis_results['test_configuration']['languages']}")
    
    print(f"\nJoint Decoding Analysis:")
    analysis = analysis_results['conditioning_analysis']
    print(f"  Total Decodings: {analysis['total_decodings']}")
    print(f"  Successful Decodings: {analysis['successful_decodings']}")
    print(f"  Success Rate: {analysis['successful_decodings']/analysis['total_decodings']:.1%}")
    print(f"  Average Overall Score: {analysis['avg_overall_score']:.3f}")
    
    print(f"\nBest Conditioning Options:")
    for conditioning_type, info in analysis['best_conditioning'].items():
        print(f"  {conditioning_type}: {info['option']} (score: {info['score']:.3f})")
    
    print(f"\nQuality Distribution:")
    for quality, count in analysis['quality_distribution'].items():
        print(f"  {quality}: {count}")
    
    print(f"\nExample Joint Decodings:")
    for i, decoding_result in enumerate(analysis_results['decoding_results'][:3]):
        text = decoding_result['text']
        joint_output = decoding_result['joint_output']
        conditioning = decoding_result['conditioning']
        
        print(f"  {i+1}. Text: {text}")
        print(f"     Joint Output: {joint_output}")
        print(f"     Conditioning: {conditioning}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_path = "data/joint_decoding_enhanced_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(analysis_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced joint decoding report saved to: {output_path}")
    
    print("="*80)
    print("Enhanced joint decoding completed!")
    print("="*80)


if __name__ == "__main__":
    main()

"""UMR evaluator for metrics and comparison.

This module provides evaluation capabilities for UMR graphs including
triple-overlap scores (heuristic Smatch proxy), graph similarity, and
round-trip evaluation.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from collections import defaultdict
import networkx as nx
from difflib import SequenceMatcher

from .graph import UMRGraph, UMRNode, UMREdge

logger = logging.getLogger(__name__)


class UMREvaluator:
    """Evaluator for UMR graphs and generation quality."""
    
    def __init__(self):
        """Initialize the UMR evaluator."""
        pass
        
    def compute_triple_overlap_score(self, gold_graph: UMRGraph, pred_graph: UMRGraph) -> Dict[str, float]:
        """Compute heuristic triple-overlap score between gold and predicted graphs.
        
        This is not full Smatch; it compares (label, relation, label) triples and node-type triples.
        """
        gold_triples = self._extract_triples(gold_graph)
        pred_triples = self._extract_triples(pred_graph)
        
        correct = len(gold_triples.intersection(pred_triples))
        precision = correct / len(pred_triples) if pred_triples else 0.0
        recall = correct / len(gold_triples) if gold_triples else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gold_triples": len(gold_triples),
            "pred_triples": len(pred_triples),
            "correct_triples": correct
        }
        
    def compute_smatch_score(self, gold_graph: UMRGraph, pred_graph: UMRGraph) -> Dict[str, float]:
        """Backward-compatible alias to triple-overlap score.
        
        Note: This is a heuristic proxy, not a full Smatch implementation.
        """
        logger.warning("compute_smatch_score uses a heuristic triple-overlap proxy, not true Smatch")
        return self.compute_triple_overlap_score(gold_graph, pred_graph)
        
    def _extract_triples(self, graph: UMRGraph) -> Set[Tuple[str, str, str]]:
        """Extract triples from UMR graph.
        
        Args:
            graph: UMR graph
            
        Returns:
            Set of (source, relation, target) triples
        """
        triples = set()
        
        # Add edge triples
        for edge in graph.edges:
            source_node = graph.get_node(edge.source)
            target_node = graph.get_node(edge.target)
            
            if source_node and target_node:
                source_label = source_node.label
                target_label = target_node.label
                triples.add((source_label, edge.relation, target_label))
                
        # Add node triples (node type information)
        for node in graph.nodes.values():
            triples.add((node.label, "type", node.node_type))
            
        return triples
        
    def compute_graph_similarity(self, graph1: UMRGraph, graph2: UMRGraph) -> Dict[str, float]:
        """Compute similarity between two UMR graphs.
        
        Args:
            graph1: First UMR graph
            graph2: Second UMR graph
            
        Returns:
            Dictionary with similarity metrics
        """
        # Node similarity
        nodes1 = set(node.label for node in graph1.nodes.values())
        nodes2 = set(node.label for node in graph2.nodes.values())
        
        node_intersection = len(nodes1.intersection(nodes2))
        node_union = len(nodes1.union(nodes2))
        node_similarity = node_intersection / node_union if node_union > 0 else 0.0
        
        # Edge similarity
        edges1 = set((edge.source, edge.relation, edge.target) for edge in graph1.edges)
        edges2 = set((edge.source, edge.relation, edge.target) for edge in graph2.edges)
        
        edge_intersection = len(edges1.intersection(edges2))
        edge_union = len(edges1.union(edges2))
        edge_similarity = edge_intersection / edge_union if edge_union > 0 else 0.0
        
        # Structure similarity (using graph edit distance approximation)
        structure_similarity = self._compute_structure_similarity(graph1, graph2)
        
        return {
            "node_similarity": node_similarity,
            "edge_similarity": edge_similarity,
            "structure_similarity": structure_similarity,
            "overall_similarity": (node_similarity + edge_similarity + structure_similarity) / 3
        }
        
    def _compute_structure_similarity(self, graph1: UMRGraph, graph2: UMRGraph) -> float:
        """Compute structural similarity between graphs.
        
        Args:
            graph1: First UMR graph
            graph2: Second UMR graph
            
        Returns:
            Structural similarity score
        """
        # Compare graph properties
        props1 = self._get_graph_properties(graph1)
        props2 = self._get_graph_properties(graph2)
        
        # Compute similarity for each property
        similarities = []
        
        # Node count similarity
        node_count_sim = min(props1["node_count"], props2["node_count"]) / max(props1["node_count"], props2["node_count"])
        similarities.append(node_count_sim)
        
        # Edge count similarity
        edge_count_sim = min(props1["edge_count"], props2["edge_count"]) / max(props1["edge_count"], props2["edge_count"])
        similarities.append(edge_count_sim)
        
        # Average degree similarity
        avg_degree_sim = min(props1["avg_degree"], props2["avg_degree"]) / max(props1["avg_degree"], props2["avg_degree"])
        similarities.append(avg_degree_sim)
        
        # Node type distribution similarity
        type_sim = self._compute_distribution_similarity(props1["node_types"], props2["node_types"])
        similarities.append(type_sim)
        
        # Relation distribution similarity
        rel_sim = self._compute_distribution_similarity(props1["relations"], props2["relations"])
        similarities.append(rel_sim)
        
        return sum(similarities) / len(similarities)
        
    def _get_graph_properties(self, graph: UMRGraph) -> Dict[str, Any]:
        """Get structural properties of a graph.
        
        Args:
            graph: UMR graph
            
        Returns:
            Dictionary of graph properties
        """
        # Node and edge counts
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)
        
        # Average degree
        if node_count > 0:
            avg_degree = (2 * edge_count) / node_count
        else:
            avg_degree = 0.0
            
        # Node type distribution
        node_types = defaultdict(int)
        for node in graph.nodes.values():
            node_types[node.node_type] += 1
            
        # Relation distribution
        relations = defaultdict(int)
        for edge in graph.edges:
            relations[edge.relation] += 1
            
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "avg_degree": avg_degree,
            "node_types": dict(node_types),
            "relations": dict(relations)
        }
        
    def _compute_distribution_similarity(self, dist1: Dict[str, int], dist2: Dict[str, int]) -> float:
        """Compute similarity between two distributions.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Distribution similarity score
        """
        all_keys = set(dist1.keys()).union(set(dist2.keys()))
        
        if not all_keys:
            return 1.0
            
        # Normalize distributions
        total1 = sum(dist1.values()) or 1
        total2 = sum(dist2.values()) or 1
        
        similarities = []
        for key in all_keys:
            val1 = dist1.get(key, 0) / total1
            val2 = dist2.get(key, 0) / total2
            similarity = min(val1, val2) / max(val1, val2) if max(val1, val2) > 0 else 1.0
            similarities.append(similarity)
            
        return sum(similarities) / len(similarities)
        
    def evaluate_round_trip(self, original_text: str, generated_text: str) -> Dict[str, float]:
        """Evaluate round-trip text generation quality.
        
        Args:
            original_text: Original input text
            generated_text: Generated text from UMR graph
            
        Returns:
            Dictionary with round-trip evaluation metrics
        """
        # Text similarity using sequence matcher
        text_similarity = SequenceMatcher(None, original_text.lower(), generated_text.lower()).ratio()
        
        # Length similarity
        len_sim = min(len(original_text), len(generated_text)) / max(len(original_text), len(generated_text))
        
        # Word overlap
        words_orig = set(original_text.lower().split())
        words_gen = set(generated_text.lower().split())
        
        if words_orig and words_gen:
            word_overlap = len(words_orig.intersection(words_gen)) / len(words_orig.union(words_gen))
        else:
            word_overlap = 0.0
            
        # Content preservation (simple heuristic)
        content_preservation = (text_similarity + word_overlap) / 2
        
        return {
            "text_similarity": text_similarity,
            "length_similarity": len_sim,
            "word_overlap": word_overlap,
            "content_preservation": content_preservation
        }
        
    def evaluate_batch(self, gold_graphs: List[UMRGraph], pred_graphs: List[UMRGraph]) -> Dict[str, float]:
        """Evaluate a batch of UMR graphs."""
        if len(gold_graphs) != len(pred_graphs):
            raise ValueError("Gold and predicted graph lists must have same length")
            
        triple_scores = []
        similarity_scores = []
        
        for gold, pred in zip(gold_graphs, pred_graphs):
            # Triple-overlap score
            triple = self.compute_triple_overlap_score(gold, pred)
            triple_scores.append(triple)
            
            # Graph similarity
            similarity = self.compute_graph_similarity(gold, pred)
            similarity_scores.append(similarity)
            
        # Aggregate scores
        avg_triple = {
            "precision": sum(s["precision"] for s in triple_scores) / len(triple_scores),
            "recall": sum(s["recall"] for s in triple_scores) / len(triple_scores),
            "f1": sum(s["f1"] for s in triple_scores) / len(triple_scores)
        }
        
        avg_similarity = {
            "node_similarity": sum(s["node_similarity"] for s in similarity_scores) / len(similarity_scores),
            "edge_similarity": sum(s["edge_similarity"] for s in similarity_scores) / len(similarity_scores),
            "structure_similarity": sum(s["structure_similarity"] for s in similarity_scores) / len(similarity_scores),
            "overall_similarity": sum(s["overall_similarity"] for s in similarity_scores) / len(similarity_scores)
        }
        
        # Keep backward-compatible 'smatch' alias
        return {
            "triple_overlap": avg_triple,
            "smatch": avg_triple,
            "similarity": avg_similarity,
            "num_graphs": len(gold_graphs)
        }
        
    def extract_primitive_metrics(self, graph: UMRGraph) -> Dict[str, Any]:
        """Extract metrics related to primitive detection from UMR graph.
        
        Args:
            graph: UMR graph to analyze
            
        Returns:
            Dictionary with primitive-related metrics
        """
        # Count nodes by type
        node_type_counts = defaultdict(int)
        for node in graph.nodes.values():
            node_type_counts[node.node_type] += 1
            
        # Count relations by type
        relation_counts = defaultdict(int)
        for edge in graph.edges:
            relation_counts[edge.relation] += 1
            
        # Identify primitive patterns
        primitive_patterns = {
            "spatial": 0,
            "temporal": 0,
            "causal": 0,
            "logical": 0,
            "quantitative": 0,
            "structural": 0
        }
        
        # Spatial patterns (location, direction)
        for node in graph.nodes.values():
            if any(spatial_word in node.label for spatial_word in 
                   ["location", "place", "direction", "position", "in", "at", "on"]):
                primitive_patterns["spatial"] += 1
                
        # Temporal patterns (time, duration)
        for node in graph.nodes.values():
            if any(temp_word in node.label for temp_word in 
                   ["time", "duration", "moment", "period", "before", "after", "during"]):
                primitive_patterns["temporal"] += 1
                
        # Causal patterns (cause-effect relations)
        for edge in graph.edges:
            if edge.relation in ["ARG1", "cause", "result"]:
                primitive_patterns["causal"] += 1
                
        # Logical patterns (negation, conjunction)
        for node in graph.nodes.values():
            if node.label in ["not", "and", "or", "if", "then"]:
                primitive_patterns["logical"] += 1
                
        # Quantitative patterns (numbers, measurements)
        for node in graph.nodes.values():
            if node.node_type == "quantifier" or any(char.isdigit() for char in node.label):
                primitive_patterns["quantitative"] += 1
                
        # Structural patterns (part-whole, possession)
        for edge in graph.edges:
            if edge.relation in ["mod", "part", "has"]:
                primitive_patterns["structural"] += 1
                
        return {
            "node_type_distribution": dict(node_type_counts),
            "relation_distribution": dict(relation_counts),
            "primitive_patterns": primitive_patterns,
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "graph_density": len(graph.edges) / (len(graph.nodes) * (len(graph.nodes) - 1)) if len(graph.nodes) > 1 else 0
        }

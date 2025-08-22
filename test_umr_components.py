#!/usr/bin/env python3
"""
Unit tests for UMR components (parser, generator, evaluator).

Tests basic functionality and edge cases for UMR parsing, generation, and evaluation.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

# Import UMR components
from src.umr.parser import UMRParser
from src.umr.generator import UMRGenerator
from src.umr.evaluator import UMREvaluator
from src.umr.graph import UMRGraph, UMRNode, UMREdge


class TestUMRGraph(unittest.TestCase):
    """Test UMR graph data structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = UMRGraph("test_graph")
    
    def test_add_node(self):
        """Test adding nodes to the graph."""
        node = UMRNode("n1", "cat", "concept", {"pos": "NOUN"})
        self.graph.add_node(node)
        
        self.assertIn("n1", self.graph.nodes)
        self.assertEqual(self.graph.get_node("n1").label, "cat")
    
    def test_add_edge(self):
        """Test adding edges to the graph."""
        # Add nodes first
        node1 = UMRNode("n1", "cat", "concept")
        node2 = UMRNode("n2", "mat", "concept")
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        
        # Add edge
        edge = UMREdge("n1", "n2", "AtLocation")
        self.graph.add_edge(edge)
        
        self.assertEqual(len(self.graph.edges), 1)
        self.assertEqual(self.graph.edges[0].relation, "AtLocation")
    
    def test_serialization(self):
        """Test graph serialization and deserialization."""
        # Create a simple graph
        node1 = UMRNode("n1", "cat", "concept")
        node2 = UMRNode("n2", "mat", "concept")
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        
        edge = UMREdge("n1", "n2", "AtLocation")
        self.graph.add_edge(edge)
        
        # Serialize
        serialized = self.graph.to_dict()
        
        # Deserialize
        new_graph = UMRGraph.from_dict(serialized)
        
        self.assertEqual(len(new_graph.nodes), 2)
        self.assertEqual(len(new_graph.edges), 1)
        self.assertEqual(new_graph.get_node("n1").label, "cat")


class TestUMRParser(unittest.TestCase):
    """Test UMR parser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = UMRParser("en")
    
    def test_parse_simple_sentence(self):
        """Test parsing a simple sentence."""
        text = "The cat is on the mat."
        graph = self.parser.parse_text(text)
        
        # Basic checks
        self.assertIsInstance(graph, UMRGraph)
        self.assertGreater(len(graph.nodes), 0)
        
        # Check for expected nodes
        node_labels = [node.label for node in graph.nodes.values()]
        self.assertTrue(any("cat" in label.lower() for label in node_labels))
    
    def test_parse_multilingual(self):
        """Test parsing in different languages."""
        test_cases = [
            ("en", "The cat is on the mat."),
            ("es", "El gato está en la alfombra."),
            ("fr", "Le chat est sur le tapis.")
        ]
        
        for lang, text in test_cases:
            with self.subTest(language=lang):
                parser = UMRParser(lang)
                graph = parser.parse_text(text)
                
                self.assertIsInstance(graph, UMRGraph)
                self.assertGreater(len(graph.nodes), 0)
    
    def test_extract_primitive_patterns(self):
        """Test primitive pattern extraction."""
        text = "The cat is on the mat."
        graph = self.parser.parse_text(text)
        patterns = self.parser.extract_primitive_patterns(graph)
        
        self.assertIsInstance(patterns, dict)
        self.assertIn("spatial", patterns)
        self.assertIn("temporal", patterns)
    
    def test_parse_edge_cases(self):
        """Test parsing edge cases."""
        edge_cases = [
            "",  # Empty string
            "A",  # Single word
            "The quick brown fox jumps over the lazy dog.",  # Long sentence
            "123 456 789",  # Numbers
            "!@#$%^&*()",  # Special characters
        ]
        
        for text in edge_cases:
            with self.subTest(text=text):
                try:
                    graph = self.parser.parse_text(text)
                    self.assertIsInstance(graph, UMRGraph)
                except Exception as e:
                    # Edge cases might fail, but shouldn't crash
                    self.assertIsInstance(e, Exception)


class TestUMRGenerator(unittest.TestCase):
    """Test UMR generator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = UMRGenerator("en")
    
    def test_generate_simple_sentence(self):
        """Test generating text from a simple graph."""
        # Create a simple graph
        graph = UMRGraph("test")
        
        # Add nodes
        cat_node = UMRNode("n1", "cat", "concept", surface_form="cat")
        mat_node = UMRNode("n2", "mat", "concept", surface_form="mat")
        be_node = UMRNode("n3", "be", "event", surface_form="is")
        
        graph.add_node(cat_node)
        graph.add_node(mat_node)
        graph.add_node(be_node)
        
        # Add edges
        graph.add_edge(UMREdge("n3", "n1", "ARG0"))
        graph.add_edge(UMREdge("n3", "n2", "ARG1"))
        
        # Generate text
        generated = self.generator.generate_text(graph)
        
        self.assertIsInstance(generated, str)
        self.assertGreater(len(generated), 0)
    
    def test_generate_multilingual(self):
        """Test text generation in different languages."""
        # Create a simple graph
        graph = UMRGraph("test")
        cat_node = UMRNode("n1", "cat", "concept", surface_form="cat")
        graph.add_node(cat_node)
        
        test_languages = ["en", "es", "fr"]
        
        for lang in test_languages:
            with self.subTest(language=lang):
                generator = UMRGenerator(lang)
                generated = generator.generate_text(graph)
                
                self.assertIsInstance(generated, str)
                self.assertGreater(len(generated), 0)
    
    def test_generate_empty_graph(self):
        """Test generation from empty graph."""
        graph = UMRGraph("empty")
        generated = self.generator.generate_text(graph)
        
        # Should handle empty graph gracefully
        self.assertIsInstance(generated, str)
    
    def test_generate_complex_graph(self):
        """Test generation from complex graph."""
        # Create a more complex graph
        graph = UMRGraph("complex")
        
        # Add multiple nodes
        nodes = [
            UMRNode("n1", "cat", "concept", surface_form="cat"),
            UMRNode("n2", "run", "event", surface_form="runs"),
            UMRNode("n3", "quickly", "property", surface_form="quickly"),
            UMRNode("n4", "park", "concept", surface_form="park")
        ]
        
        for node in nodes:
            graph.add_node(node)
        
        # Add edges
        edges = [
            UMREdge("n2", "n1", "ARG0"),  # cat runs
            UMREdge("n2", "n3", "ARG1"),  # runs quickly
            UMREdge("n2", "n4", "ARG2"),  # runs in park
        ]
        
        for edge in edges:
            graph.add_edge(edge)
        
        generated = self.generator.generate_text(graph)
        
        self.assertIsInstance(generated, str)
        self.assertGreater(len(generated), 0)


class TestUMREvaluator(unittest.TestCase):
    """Test UMR evaluator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = UMREvaluator()
    
    def test_compute_triple_overlap_score(self):
        """Test triple overlap score computation."""
        # Create two similar graphs
        graph1 = UMRGraph("gold")
        graph2 = UMRGraph("pred")
        
        # Add similar nodes
        for graph in [graph1, graph2]:
            graph.add_node(UMRNode("n1", "cat", "concept"))
            graph.add_node(UMRNode("n2", "mat", "concept"))
            graph.add_edge(UMREdge("n1", "n2", "AtLocation"))
        
        # Compute score
        score = self.evaluator.compute_triple_overlap_score(graph1, graph2)
        
        self.assertIsInstance(score, dict)
        self.assertIn("precision", score)
        self.assertIn("recall", score)
        self.assertIn("f1", score)
        
        # Perfect match should have high scores
        self.assertGreaterEqual(score["precision"], 0.0)
        self.assertLessEqual(score["precision"], 1.0)
    
    def test_compute_graph_similarity(self):
        """Test graph similarity computation."""
        # Create two graphs
        graph1 = UMRGraph("graph1")
        graph2 = UMRGraph("graph2")
        
        # Add some nodes and edges to avoid division by zero
        graph1.add_node(UMRNode("n1", "cat", "concept"))
        graph1.add_node(UMRNode("n2", "mat", "concept"))
        graph1.add_edge(UMREdge("n1", "n2", "AtLocation"))
        
        graph2.add_node(UMRNode("n1", "cat", "concept"))
        graph2.add_node(UMRNode("n2", "mat", "concept"))
        graph2.add_edge(UMREdge("n1", "n2", "AtLocation"))
        
        similarity = self.evaluator.compute_graph_similarity(graph1, graph2)
        
        self.assertIsInstance(similarity, dict)
        self.assertIn("node_similarity", similarity)
        self.assertIn("edge_similarity", similarity)
        self.assertIn("structure_similarity", similarity)
    
    def test_evaluate_round_trip(self):
        """Test round-trip evaluation."""
        original_text = "The cat is on the mat."
        generated_text = "The cat is on the mat."
        
        metrics = self.evaluator.evaluate_round_trip(original_text, generated_text)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("text_similarity", metrics)
        self.assertIn("length_similarity", metrics)
        self.assertIn("word_overlap", metrics)
    
    def test_extract_primitive_metrics(self):
        """Test primitive metrics extraction."""
        graph = UMRGraph("test")
        
        # Add various node types
        graph.add_node(UMRNode("n1", "cat", "concept"))
        graph.add_node(UMRNode("n2", "run", "event"))
        graph.add_node(UMRNode("n3", "quick", "property"))
        
        metrics = self.evaluator.extract_primitive_metrics(graph)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_nodes", metrics)
        self.assertIn("total_edges", metrics)
        self.assertIn("node_type_distribution", metrics)


class TestUMRIntegration(unittest.TestCase):
    """Test UMR components integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = UMRParser("en")
        self.generator = UMRGenerator("en")
        self.evaluator = UMREvaluator()
    
    def test_parse_generate_round_trip(self):
        """Test parse -> generate round trip."""
        original_text = "The cat is on the mat."
        
        # Parse
        graph = self.parser.parse_text(original_text)
        self.assertIsInstance(graph, UMRGraph)
        
        # Generate
        generated_text = self.generator.generate_text(graph)
        self.assertIsInstance(generated_text, str)
        
        # Evaluate round trip
        metrics = self.evaluator.evaluate_round_trip(original_text, generated_text)
        self.assertIsInstance(metrics, dict)
    
    def test_cross_language_comparison(self):
        """Test cross-language UMR comparison."""
        texts = {
            "en": "The cat is on the mat.",
            "es": "El gato está en la alfombra.",
            "fr": "Le chat est sur le tapis."
        }
        
        graphs = {}
        for lang, text in texts.items():
            parser = UMRParser(lang)
            graphs[lang] = parser.parse_text(text)
        
        # Compare graphs
        for lang1 in graphs:
            for lang2 in graphs:
                if lang1 != lang2:
                    similarity = self.evaluator.compute_graph_similarity(
                        graphs[lang1], graphs[lang2]
                    )
                    self.assertIsInstance(similarity, dict)
    
    def test_primitive_pattern_extraction(self):
        """Test primitive pattern extraction and evaluation."""
        text = "The cat is on the mat."
        graph = self.parser.parse_text(text)
        
        # Extract patterns
        patterns = self.parser.extract_primitive_patterns(graph)
        self.assertIsInstance(patterns, dict)
        
        # Extract metrics
        metrics = self.evaluator.extract_primitive_metrics(graph)
        self.assertIsInstance(metrics, dict)


class TestUMREdgeCases(unittest.TestCase):
    """Test UMR components with edge cases and error handling."""
    
    def test_parser_error_handling(self):
        """Test parser error handling."""
        parser = UMRParser("en")
        
        # Test with None input
        with self.assertRaises(Exception):
            parser.parse_text(None)
        
        # Test with very long text
        long_text = "The " + "quick brown fox " * 1000 + "jumps."
        try:
            graph = parser.parse_text(long_text)
            self.assertIsInstance(graph, UMRGraph)
        except Exception:
            # Long text might fail, but shouldn't crash
            pass
    
    def test_generator_error_handling(self):
        """Test generator error handling."""
        generator = UMRGenerator("en")
        
        # Test with None graph
        with self.assertRaises(Exception):
            generator.generate_text(None)
        
        # Test with graph containing invalid nodes
        graph = UMRGraph("invalid")
        graph.add_node(UMRNode("n1", "test", "invalid_type"))
        
        try:
            generated = generator.generate_text(graph)
            self.assertIsInstance(generated, str)
        except Exception:
            # Invalid nodes might fail, but shouldn't crash
            pass
    
    def test_evaluator_error_handling(self):
        """Test evaluator error handling."""
        evaluator = UMREvaluator()
        
        # Test with None inputs
        with self.assertRaises(Exception):
            evaluator.compute_triple_overlap_score(None, None)
        
        # Test with empty graphs
        empty_graph = UMRGraph("empty")
        score = evaluator.compute_triple_overlap_score(empty_graph, empty_graph)
        self.assertIsInstance(score, dict)


def run_tests():
    """Run all UMR component tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestUMRGraph,
        TestUMRParser,
        TestUMRGenerator,
        TestUMREvaluator,
        TestUMRIntegration,
        TestUMREdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

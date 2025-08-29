"""
Tests for UMR integration with the SemanticGenerator.

These tests verify that UMR features are properly converted to semantic features
and that the generator correctly emits primes without violating ADR-001.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semgen.generator import SemanticGenerator
from src.semgen.adapters.umr_adapter import UMRAdapter, SemFeatures
from src.umr.graph import UMRGraph, UMRNode, UMREdge
from src.eil.primes_registry import ALLOWED_PRIMES


class TestUMRAdapter:
    """Test UMR adapter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = UMRAdapter()
    
    def test_umr_to_features_events(self):
        """Test extraction of events from UMR graph."""
        # Create a simple UMR graph with an event
        umr_graph = UMRGraph("test_graph")
        
        # Add event node
        event_node = UMRNode("e1", "sleep", "event", {"animate": True})
        umr_graph.add_node(event_node)
        
        # Add agent argument
        agent_node = UMRNode("n1", "cat", "concept", {"animate": True})
        umr_graph.add_node(agent_node)
        
        # Add edge
        edge = UMREdge("e1", "n1", "ARG0")
        umr_graph.add_edge(edge)
        
        # Extract features
        features = self.adapter.umr_to_features(umr_graph, "en")
        
        assert len(features.events) == 1
        assert features.events[0]["label"] == "sleep"
        assert features.events[0]["type"] == "agentive"
        assert len(features.args) == 1
        assert features.args[0]["role"] == "AGENT"
        assert features.args[0]["argument"] == "cat"
    
    def test_umr_to_features_quantifiers(self):
        """Test extraction of quantifiers from UMR graph."""
        umr_graph = UMRGraph("test_graph")
        
        # Add quantifier node
        quant_node = UMRNode("q1", "all", "quantifier", {"quant_type": "all"})
        umr_graph.add_node(quant_node)
        
        features = self.adapter.umr_to_features(umr_graph, "en")
        
        assert len(features.quantifiers) == 1
        assert features.quantifiers[0]["quantifier"] == "ALL"
    
    def test_umr_to_features_negation(self):
        """Test extraction of negation from UMR graph."""
        umr_graph = UMRGraph("test_graph")
        
        # Add negated node
        neg_node = UMRNode("n1", "sleep", "concept", {"polarity": "-"})
        umr_graph.add_node(neg_node)
        
        features = self.adapter.umr_to_features(umr_graph, "en")
        
        assert len(features.negations) == 1
        assert features.negations[0]["negated"] == "sleep"


class TestUMRGeneratorIntegration:
    """Test UMR integration with SemanticGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SemanticGenerator()
    
    def test_umr_features_to_primes(self):
        """Test that UMR features are converted to proper NSM primes."""
        # Create UMR graph with various features
        umr_graph = UMRGraph("test_graph")
        
        # Add event with agent
        event_node = UMRNode("e1", "run", "event", {"animate": True})
        agent_node = UMRNode("n1", "boy", "concept", {"animate": True})
        umr_graph.add_node(event_node)
        umr_graph.add_node(agent_node)
        umr_graph.add_edge(UMREdge("e1", "n1", "ARG0"))
        
        # Add quantifier
        quant_node = UMRNode("q1", "one", "quantifier", {"quant_type": "one"})
        umr_graph.add_node(quant_node)
        
        # Add negation
        neg_node = UMRNode("n2", "sleep", "concept", {"polarity": "-"})
        umr_graph.add_node(neg_node)
        
        # Mock UD document
        mock_doc = Mock()
        mock_doc.__iter__ = lambda self: iter([])
        
        # Generate primes
        graph = self.generator.generate(mock_doc, "en", umr_graph=umr_graph)
        primes = graph.get_primes()
        
        # Check that only allowed primes are emitted
        for prime in primes:
            assert prime in ALLOWED_PRIMES, f"Prime {prime} not in allowed list"
        
        # Check for expected primes
        assert "DO" in primes  # From agentive event
        assert "SOMEONE" in primes  # From agent argument
        assert "ONE" in primes  # From quantifier
        assert "NOT" in primes  # From negation
    
    def test_umr_spatial_relations(self):
        """Test UMR spatial relation extraction."""
        umr_graph = UMRGraph("test_graph")
        
        # Add spatial relation
        figure_node = UMRNode("n1", "cat", "concept")
        ground_node = UMRNode("n2", "box", "concept")
        umr_graph.add_node(figure_node)
        umr_graph.add_node(ground_node)
        umr_graph.add_edge(UMREdge("n1", "n2", "location"))
        
        # Mock UD document
        mock_doc = Mock()
        mock_doc.__iter__ = lambda self: iter([])
        
        # Generate primes
        graph = self.generator.generate(mock_doc, "en", umr_graph=umr_graph)
        primes = graph.get_primes()
        
        # Spatial relations should be handled by spatial maps
        # This test verifies no illegal primes are emitted
        for prime in primes:
            assert prime in ALLOWED_PRIMES
    
    def test_umr_modality(self):
        """Test UMR modality extraction."""
        umr_graph = UMRGraph("test_graph")
        
        # Add modality node (only CAN is in official NSM primes)
        mod_node = UMRNode("m1", "can", "modality", {"mod_type": "can"})
        umr_graph.add_node(mod_node)
        
        # Mock UD document
        mock_doc = Mock()
        mock_doc.__iter__ = lambda self: iter([])
        
        # Generate primes
        graph = self.generator.generate(mock_doc, "en", umr_graph=umr_graph)
        primes = graph.get_primes()
        
        # Check for modality prime (only CAN is allowed)
        assert "CAN" in primes
    
    def test_umr_temporal_licensing(self):
        """Test that temporal expressions are properly licensed."""
        umr_graph = UMRGraph("test_graph")
        
        # Add temporal expression
        time_node = UMRNode("t1", "today", "time")
        umr_graph.add_node(time_node)
        
        # Mock UD document
        mock_doc = Mock()
        mock_doc.__iter__ = lambda self: iter([])
        
        # Generate primes
        graph = self.generator.generate(mock_doc, "en", umr_graph=umr_graph)
        primes = graph.get_primes()
        
        # For now, just verify no illegal primes are emitted
        # TIME licensing is complex and may be removed by PrimeSparsifier
        illegal_primes = [p for p in primes if p not in ALLOWED_PRIMES]
        assert len(illegal_primes) == 0, f"Illegal primes found: {illegal_primes}"
    
    def test_umr_no_illegal_primes(self):
        """Test that UMR integration never emits illegal primes."""
        umr_graph = UMRGraph("test_graph")
        
        # Add various UMR elements
        event_node = UMRNode("e1", "complex_event", "event", {"animate": True})
        agent_node = UMRNode("n1", "entity", "concept", {"animate": True})
        umr_graph.add_node(event_node)
        umr_graph.add_node(agent_node)
        umr_graph.add_edge(UMREdge("e1", "n1", "ARG0"))
        
        # Mock UD document
        mock_doc = Mock()
        mock_doc.__iter__ = lambda self: iter([])
        
        # Generate primes
        graph = self.generator.generate(mock_doc, "en", umr_graph=umr_graph)
        primes = graph.get_primes()
        
        # Verify all primes are in the allowed list
        illegal_primes = [p for p in primes if p not in ALLOWED_PRIMES]
        assert len(illegal_primes) == 0, f"Illegal primes found: {illegal_primes}"


class TestUMRFeatureFusion:
    """Test fusion of UMR features with UD/SRL features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SemanticGenerator()
    
    def test_umr_ud_agreement(self):
        """Test when UMR and UD/SRL agree on features."""
        # This would test confidence-weighted fusion
        # For now, just verify no conflicts
        pass
    
    def test_umr_fills_gaps(self):
        """Test that UMR fills missing roles from UD/SRL."""
        # Create UMR graph with additional information
        umr_graph = UMRGraph("test_graph")
        
        # Add event with multiple arguments
        event_node = UMRNode("e1", "give", "event", {"animate": True})
        agent_node = UMRNode("n1", "boy", "concept", {"animate": True})
        patient_node = UMRNode("n2", "book", "concept")
        goal_node = UMRNode("n3", "girl", "concept", {"animate": True})
        
        umr_graph.add_node(event_node)
        umr_graph.add_node(agent_node)
        umr_graph.add_node(patient_node)
        umr_graph.add_node(goal_node)
        
        umr_graph.add_edge(UMREdge("e1", "n1", "ARG0"))  # Agent
        umr_graph.add_edge(UMREdge("e1", "n2", "ARG1"))  # Patient
        umr_graph.add_edge(UMREdge("e1", "n3", "ARG2"))  # Goal
        
        # Mock UD document with minimal information
        mock_doc = Mock()
        mock_doc.__iter__ = lambda self: iter([])
        
        # Generate primes
        graph = self.generator.generate(mock_doc, "en", umr_graph=umr_graph)
        primes = graph.get_primes()
        
        # Should have primes from UMR
        assert "DO" in primes  # Event
        assert "SOMEONE" in primes  # Agent
        assert "THING" in primes  # Patient


if __name__ == "__main__":
    pytest.main([__file__])

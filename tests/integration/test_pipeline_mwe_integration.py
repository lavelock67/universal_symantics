"""
Mandatory integration test for MWE pipeline integration.

This test ensures that MWE spans survive the full pipeline and are visible
to the semantic generator. It fails loudly if MWE normalization is bypassed.
"""

import pytest
import os
from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language


@pytest.fixture
def en_service():
    """Create English detection service."""
    return NSMDetectionService()


class TestPipelineMWEIntegration:
    """Test that MWE normalization survives the full pipeline."""
    
    def test_mwe_survives_pipeline(self, en_service):
        """Test that MWE spans are visible to the generator."""
        # Enable debug mode
        os.environ["USYM_DEBUG"] = "1"
        
        try:
            # Test with spatial MWE that should be normalized
            text = "He lives in front of the station."
            result = en_service.detect_primes(text, Language.ENGLISH)
            
            # Check pipeline metadata
            assert result.metadata["pipeline_path"] in ["semantic", "semantic+umr"]
            assert result.metadata["manual_detector_count"] == 0
            assert "semantic_trace" in result.metadata
            
            # Check semantic trace order
            trace = result.metadata["semantic_trace"]
            assert trace[0] == "mwe", f"MWE must run first, got: {trace}"
            assert "generator" in trace, f"Generator must run, got: {trace}"
            
            # Check that generator saw MWE spans
            if "debug" in result.metadata:
                debug = result.metadata["debug"]
                if "mwe" in debug:
                    mwe_debug = debug["mwe"]
                    assert "spans" in mwe_debug
                    spans_debug = mwe_debug["spans"]
                    assert "count" in spans_debug
                    assert spans_debug["count"] >= 1, f"Expected at least 1 MWE span, got: {spans_debug['count']}"
            
            # Check that spatial primes were detected
            spatial_primes = [p.text for p in result.primes if p.text in ["NEAR", "ABOVE", "INSIDE", "IN_FRONT"]]
            assert len(spatial_primes) >= 1, f"Expected spatial primes, got: {[p.text for p in result.primes]}"
            
        finally:
            # Clean up
            if "USYM_DEBUG" in os.environ:
                del os.environ["USYM_DEBUG"]
    
    def test_figurative_guards_work(self, en_service):
        """Test that figurative expressions don't trigger spatial primes."""
        # Enable debug mode
        os.environ["USYM_DEBUG"] = "1"
        
        try:
            # Test figurative expressions that should NOT trigger spatial primes
            figurative_cases = [
                "The deadline is near the future.",  # temporal, not spatial
                "The team is inside the organization.",  # social, not spatial
                "The topic is over the subject.",  # topical, not spatial
            ]
            
            for text in figurative_cases:
                result = en_service.detect_primes(text, Language.ENGLISH)
                
                # Check that no spatial primes were incorrectly detected
                spatial_primes = [p.text for p in result.primes if p.text in ["NEAR", "ABOVE", "INSIDE", "IN_FRONT"]]
                assert len(spatial_primes) == 0, f"Figurative expression '{text}' incorrectly triggered spatial primes: {spatial_primes}"
                
        finally:
            # Clean up
            if "USYM_DEBUG" in os.environ:
                del os.environ["USYM_DEBUG"]
    
    def test_pipeline_order_invariant(self, en_service):
        """Test that pipeline order is consistent."""
        # Enable debug mode
        os.environ["USYM_DEBUG"] = "1"
        
        try:
            text = "The book is inside the box."
            result = en_service.detect_primes(text, Language.ENGLISH)
            
            # Check semantic trace structure
            trace = result.metadata["semantic_trace"]
            
            # MWE must be first
            assert trace[0] == "mwe", f"MWE must be first in trace: {trace}"
            
            # UD must come after MWE
            assert "ud" in trace, f"UD must be in trace: {trace}"
            ud_index = trace.index("ud")
            assert ud_index > 0, f"UD must come after MWE: {trace}"
            
            # Generator must be last
            assert "generator" in trace, f"Generator must be in trace: {trace}"
            generator_index = trace.index("generator")
            assert generator_index == len(trace) - 1, f"Generator must be last: {trace}"
            
        finally:
            # Clean up
            if "USYM_DEBUG" in os.environ:
                del os.environ["USYM_DEBUG"]
    
    def test_counter_increments(self, en_service):
        """Test that generator invocation counter increments."""
        text = "The lamp is above the table."
        result = en_service.detect_primes(text, Language.ENGLISH)
        
        # Check that counter is present and positive
        assert "generator_invocations_total" in result.metadata
        assert result.metadata["generator_invocations_total"] > 0
    
    def test_no_manual_detectors(self, en_service):
        """Test that no manual detectors are used."""
        text = "The car is near the building."
        result = en_service.detect_primes(text, Language.ENGLISH)
        
        # Ensure manual detector count is zero
        assert result.metadata["manual_detector_count"] == 0
        
        # Ensure pipeline path is semantic
        assert result.metadata["pipeline_path"] in ["semantic", "semantic+umr"]


if __name__ == "__main__":
    pytest.main([__file__])

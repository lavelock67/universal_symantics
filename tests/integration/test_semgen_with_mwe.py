"""
Integration tests for MWE normalizer with semantic generator.

Tests pipeline order, ADR-001 compliance, and canary test integration.
"""

import pytest
import spacy
from spacy.language import Language

from src.nlp.components.mwe_normalizer import MWENormalizer
from src.semgen.generator import SemanticGenerator
from src.eil.primes_registry import ALLOWED_PRIMES


@pytest.fixture
def nlp_en():
    """Load English SpaCy model with MWE normalizer."""
    # Import to register the factory
    from src.nlp.components.mwe_normalizer import MWENormalizer, create_mwe_normalizer
    
    nlp = spacy.load("en_core_web_sm")
    # Add MWE normalizer after tagger but before parser
    nlp.add_pipe("mwe_normalizer", name="mwe_normalizer", after="tagger")
    return nlp


@pytest.fixture
def semantic_generator():
    """Create semantic generator."""
    return SemanticGenerator()


class TestMWESemanticGeneratorIntegration:
    """Test integration between MWE normalizer and semantic generator."""
    
    def test_pipeline_order_mwe_before_ud(self, nlp_en):
        """Test that MWE normalizer runs before UD parsing."""
        # Check pipeline order
        pipe_names = [pipe_name for pipe_name, _ in nlp_en.pipeline]
        
        # MWE normalizer should be first
        assert "mwe_normalizer" in pipe_names
        mwe_index = pipe_names.index("mwe_normalizer")
        
        # UD components should come after MWE
        ud_components = ["tagger", "parser", "attribute_ruler"]
        for ud_comp in ud_components:
            if ud_comp in pipe_names:
                ud_index = pipe_names.index(ud_comp)
                assert ud_index > mwe_index, f"UD component '{ud_comp}' should come after MWE normalizer"
    
    def test_mwe_spans_available_to_generator(self, nlp_en, semantic_generator):
        """Test that MWE spans are available to the semantic generator."""
        text = "The book is inside the box."
        doc = nlp_en(text)
        
        # Check that MWE spans are set
        assert doc._.mwe_spans_by_token is not None
        
        # Check that spans contain expected metadata
        found_inside_span = False
        for token_indices, spans in doc._.mwe_spans_by_token.items():
            for span in spans:
                if "inside" in span.text.lower():
                    found_inside_span = True
                    assert span._.mwe_meta is not None
                    assert span._.mwe_meta["prime_hint"] == "INSIDE"
                    break
            if found_inside_span:
                break
        
        assert found_inside_span, "Should find 'inside' MWE span"
        
        # Test that generator can access MWE spans
        # Note: This tests the interface, not the actual generation
        mwe_spans_by_token = doc._.mwe_spans_by_token
        assert isinstance(mwe_spans_by_token, dict)
        
        # Verify token indexing
        for token in doc:
            if token.i in mwe_spans_by_token:
                spans = mwe_spans_by_token[token.i]
                assert isinstance(spans, list)
                for span in spans:
                    assert hasattr(span, '_')
                    assert hasattr(span._, 'mwe_meta')
    
    def test_adr_001_compliance_no_mwe_prime_emission(self, nlp_en, semantic_generator):
        """Test that MWE normalizer never emits primes directly (ADR-001 compliance)."""
        text = "The book is inside the box."
        doc = nlp_en(text)
        
        # MWE normalizer should only provide hints, not emit primes
        for token_indices, spans in doc._.mwe_spans_by_token.items():
            for span in spans:
                meta = span._.mwe_meta
                # MWE provides hints, not direct primes
                assert "prime_hint" in meta
                assert meta["prime_hint"] in ["INSIDE", "NEAR", "ABOVE", "IN_FRONT"]
                # The hint is advisory only - actual prime emission happens in generator
                assert meta["source"] == "spatial_maps.json"
    
    def test_figurative_guards_work_in_pipeline(self, nlp_en):
        """Test that figurative guards prevent false spatial hits in full pipeline."""
        # Test figurative expressions that should be filtered out
        figurative_cases = [
            "The team is inside the organization.",  # social, not spatial
            "The topic is over the subject.",        # topical, not spatial
            "The deadline is near the future.",      # temporal, not spatial
        ]
        
        for text in figurative_cases:
            doc = nlp_en(text)
            
            # Check that figurative expressions are filtered out
            if doc._.mwe_spans_by_token:
                for token_indices, spans in doc._.mwe_spans_by_token.items():
                    for span in spans:
                        # Should not contain spatial MWEs for figurative cases
                        span_text = span.text.lower()
                        spatial_words = ["inside", "over", "near"]
                        for spatial_word in spatial_words:
                            if spatial_word in span_text:
                                # This should be filtered by guards
                                assert False, f"Figurative expression '{text}' should not contain spatial MWE '{spatial_word}'"
    
    def test_canary_test_compatibility(self, nlp_en, semantic_generator):
        """Test that MWE normalizer is compatible with canary test expectations."""
        # Test cases from canary tests
        canary_cases = [
            ("The book is inside the box.", "INSIDE"),
            ("He lives near the station.", "NEAR"),
            ("The lamp is above the table.", "ABOVE"),
        ]
        
        for text, expected_prime in canary_cases:
            doc = nlp_en(text)
            
            # Check that MWE spans are detected
            assert doc._.mwe_spans_by_token is not None
            
            # Check that the expected prime hint is present
            found_expected_hint = False
            for token_indices, spans in doc._.mwe_spans_by_token.items():
                for span in spans:
                    meta = span._.mwe_meta
                    if meta and meta["prime_hint"] == expected_prime:
                        found_expected_hint = True
                        break
                if found_expected_hint:
                    break
            
            assert found_expected_hint, f"Expected prime hint '{expected_prime}' not found for '{text}'"
    
    def test_metrics_export(self, nlp_en):
        """Test that MWE normalizer metrics are properly exported."""
        # Get the MWE normalizer component
        mwe_component = nlp_en.get_pipe("mwe_normalizer")
        
        # Reset metrics
        mwe_component.metrics = {
            'mwe_detected_total': 0,
            'mwe_guard_filtered_total': 0,
            'mwe_retokenize_applied_total': 0
        }
        
        # Process test cases
        test_cases = [
            "The book is inside the box.",      # Should be detected
            "The team is inside the org.",      # Should be filtered
        ]
        
        for text in test_cases:
            doc = nlp_en(text)
        
        # Check metrics
        metrics = mwe_component.get_metrics()
        assert isinstance(metrics, dict)
        assert 'mwe_detected_total' in metrics
        assert 'mwe_guard_filtered_total' in metrics
        assert 'mwe_retokenize_applied_total' in metrics
        
        # Should have detected at least one MWE
        assert metrics['mwe_detected_total'] > 0
    
    def test_retokenization_option(self, nlp_en):
        """Test that retokenization option works correctly."""
        # Create normalizer with retokenization enabled
        nlp_with_merge = spacy.load("en_core_web_sm")
        mwe_with_merge = MWENormalizer(nlp_with_merge, merge_adp=True)
        nlp_with_merge.add_pipe("mwe_normalizer", name="mwe_normalizer", after="tagger")
        
        text = "The book is inside the box."
        doc = nlp_with_merge(text)
        
        # Check that retokenization metrics are tracked
        mwe_component = nlp_with_merge.get_pipe("mwe_normalizer")
        metrics = mwe_component.get_metrics()
        assert 'mwe_retokenize_applied_total' in metrics
    
    def test_multilingual_support(self):
        """Test that MWE normalizer supports multiple languages."""
        # Test with different language models
        languages = ["en_core_web_sm", "es_core_news_sm", "fr_core_news_sm", "de_core_news_sm"]
        
        for lang_model in languages:
            try:
                nlp = spacy.load(lang_model)
                mwe_normalizer = MWENormalizer(nlp)
                
                # Check that language-specific maps are loaded
                assert mwe_normalizer.lang in ["en", "es", "fr", "de"]
                assert mwe_normalizer.maps is not None
                
                # Check that patterns are built
                assert hasattr(mwe_normalizer, 'matcher')
                
            except OSError:
                # Skip if model not available
                pytest.skip(f"SpaCy model {lang_model} not available")


class TestPipelineContract:
    """Test that the pipeline contract is maintained."""
    
    def test_mwe_hints_only_no_direct_primes(self, nlp_en):
        """Test that MWE normalizer only provides hints, never emits primes directly."""
        text = "The book is inside the box."
        doc = nlp_en(text)
        
        # MWE normalizer should only set spans and metadata
        assert hasattr(doc._, 'mwe_spans_by_token')
        
        # No direct prime emission from MWE layer
        # Primes should only come from SemanticGenerator
        for token_indices, spans in doc._.mwe_spans_by_token.items():
            for span in spans:
                meta = span._.mwe_meta
                # Only hints, not actual primes
                assert "prime_hint" in meta
                assert meta["prime_hint"] in ["INSIDE", "NEAR", "ABOVE", "IN_FRONT"]
                # Source should be spatial maps, not direct emission
                assert meta["source"] == "spatial_maps.json"
    
    def test_semantic_generator_remains_single_emitter(self, nlp_en, semantic_generator):
        """Test that SemanticGenerator remains the single prime emitter (ADR-001)."""
        text = "The book is inside the box."
        doc = nlp_en(text)
        
        # MWE normalizer should not emit any primes
        # Only provide hints for the generator
        
        # The actual prime emission should happen in SemanticGenerator
        # This test verifies the separation of concerns
        mwe_spans_by_token = doc._.mwe_spans_by_token
        
        # Check that MWE provides structured hints
        if mwe_spans_by_token:
            for token_indices, spans in mwe_spans_by_token.items():
                for span in spans:
                    # MWE provides metadata, not primes
                    assert span._.mwe_meta is not None
                    assert "prime_hint" in span._.mwe_meta
                    # The hint is advisory only
                    assert span._.mwe_meta["source"] == "spatial_maps.json"


if __name__ == "__main__":
    pytest.main([__file__])

"""
Unit tests for MWE normalization - English spatial expressions.

Tests positive cases, negative figurative cases, and chained ADP detection.
"""

import pytest
import spacy
from spacy.tokens import Doc

from src.nlp.components.mwe_normalizer import MWENormalizer, load_spatial_maps


@pytest.fixture
def nlp_en():
    """Load English SpaCy model."""
    return spacy.load("en_core_web_sm")


@pytest.fixture
def mwe_normalizer(nlp_en):
    """Create MWE normalizer for English."""
    return MWENormalizer(nlp_en)


class TestMWENormalizerEnglish:
    """Test MWE normalization for English spatial expressions."""
    
    def test_positive_spatial_cases(self, mwe_normalizer, nlp_en):
        """Test positive spatial MWE detection."""
        test_cases = [
            # NEAR expressions
            ("The house is near the station.", ["near"]),
            ("The store is close to the mall.", ["close", "to"]),
            ("The park is next to the school.", ["next", "to"]),
            ("The car is beside the building.", ["beside"]),
            ("The tree is by the river.", ["by"]),
            
            # INSIDE expressions
            ("The book is inside the box.", ["inside"]),
            ("The keys are inside of the drawer.", ["inside", "of"]),
            ("The cat is in the house.", ["in"]),
            ("The money is within the safe.", ["within"]),
            ("The documents are within the folder.", ["within"]),
            
            # ABOVE expressions
            ("The lamp is above the table.", ["above"]),
            ("The bird is over the tree.", ["over"]),
            ("The picture is on top of the wall.", ["on", "top", "of"]),
            ("The plane is up above the clouds.", ["up", "above"]),
            ("The mountain is higher than the hill.", ["higher", "than"]),
            
            # IN_FRONT expressions
            ("The car is in front of the building.", ["in", "front", "of"]),
        ]
        
        for text, expected_pattern in test_cases:
            doc = nlp_en(text)
            processed_doc = mwe_normalizer(doc)
            
            # Check that MWE spans are detected
            assert processed_doc._.mwe_spans_by_token is not None
            
            # Find the expected pattern in the document
            found_pattern = False
            for token_indices, spans in processed_doc._.mwe_spans_by_token.items():
                for span in spans:
                    span_text = span.text.lower()
                    if all(word in span_text for word in expected_pattern):
                        found_pattern = True
                        # Check metadata
                        assert span._.mwe_meta is not None
                        assert "prime_hint" in span._.mwe_meta
                        assert "canonical" in span._.mwe_meta
                        break
                if found_pattern:
                    break
            
            assert found_pattern, f"Expected pattern {expected_pattern} not found in '{text}'"
    
    def test_negative_figurative_cases(self, mwe_normalizer, nlp_en):
        """Test that figurative expressions are correctly filtered out."""
        test_cases = [
            # NEAR figurative
            ("The deadline is near the future.", "near future"),
            ("The topic is close to the subject.", "close to subject"),
            ("The time is next to the deadline.", "next to deadline"),
            
            # INSIDE figurative
            ("The team is inside the organization.", "inside organization"),
            ("The family is in the community.", "in community"),
            ("The group is within the company.", "within company"),
            
            # ABOVE figurative
            ("The discussion is over the topic.", "over topic"),
            ("The debate is above the subject.", "above subject"),
            ("The conversation is on top of the theme.", "on top of theme"),
            
            # IN_FRONT figurative
            ("The plan is in front of the future.", "in front of future"),
        ]
        
        for text, expected_filtered in test_cases:
            doc = nlp_en(text)
            processed_doc = mwe_normalizer(doc)
            
            # Check that figurative expressions are filtered out
            if processed_doc._.mwe_spans_by_token:
                for token_indices, spans in processed_doc._.mwe_spans_by_token.items():
                    for span in spans:
                        span_text = span.text.lower()
                        # Should not contain the figurative expression
                        assert not all(word in span_text for word in expected_filtered.split()), \
                            f"Figurative expression '{expected_filtered}' should be filtered out from '{text}'"
    
    def test_chained_adp_detection(self, mwe_normalizer, nlp_en):
        """Test that chained ADPs are detected as complete spans."""
        test_cases = [
            ("The car is in front of the old station.", "in front of"),
            ("The bird is on top of the tall tree.", "on top of"),
            ("The book is inside of the wooden box.", "inside of"),
            ("The cat is close to the big house.", "close to"),
        ]
        
        for text, expected_span in test_cases:
            doc = nlp_en(text)
            processed_doc = mwe_normalizer(doc)
            
            # Check that the complete span is detected
            found_complete_span = False
            for token_indices, spans in processed_doc._.mwe_spans_by_token.items():
                for span in spans:
                    if span.text.lower() == expected_span:
                        found_complete_span = True
                        # Verify span indices are correct
                        assert span.start < span.end
                        assert len(span) >= 2  # Should be at least 2 tokens
                        break
                if found_complete_span:
                    break
            
            assert found_complete_span, f"Expected complete span '{expected_span}' not found in '{text}'"
    
    def test_longest_match_preference(self, mwe_normalizer, nlp_en):
        """Test that longest matches are preferred over shorter ones."""
        # This sentence contains both "in" and "in front of"
        text = "The car is in front of the building."
        doc = nlp_en(text)
        processed_doc = mwe_normalizer(doc)
        
        # Should prefer "in front of" over just "in"
        found_in_front_of = False
        found_just_in = False
        
        for token_indices, spans in processed_doc._.mwe_spans_by_token.items():
            for span in spans:
                if span.text.lower() == "in front of":
                    found_in_front_of = True
                elif span.text.lower() == "in":
                    found_just_in = True
        
        assert found_in_front_of, "Should detect 'in front of'"
        # Should not detect just "in" when "in front of" is present
        assert not found_just_in, "Should not detect just 'in' when 'in front of' is present"
    
    def test_metrics_tracking(self, mwe_normalizer, nlp_en):
        """Test that metrics are properly tracked."""
        # Reset metrics
        mwe_normalizer.metrics = {
            'mwe_detected_total': 0,
            'mwe_guard_filtered_total': 0,
            'mwe_retokenize_applied_total': 0
        }
        
        # Process positive case
        doc = nlp_en("The house is near the station.")
        processed_doc = mwe_normalizer(doc)
        
        # Check metrics
        metrics = mwe_normalizer.get_metrics()
        assert metrics['mwe_detected_total'] > 0
        assert metrics['mwe_guard_filtered_total'] >= 0
        assert metrics['mwe_retokenize_applied_total'] >= 0
    
    def test_span_metadata(self, mwe_normalizer, nlp_en):
        """Test that span metadata is correctly set."""
        doc = nlp_en("The book is inside the box.")
        processed_doc = mwe_normalizer(doc)
        
        for token_indices, spans in processed_doc._.mwe_spans_by_token.items():
            for span in spans:
                if "inside" in span.text.lower():
                    meta = span._.mwe_meta
                    assert meta is not None
                    assert meta["prime_hint"] == "INSIDE"
                    assert meta["canonical"] == "inside"
                    assert meta["lang"] == "en"
                    assert meta["source"] == "spatial_maps.json"
                    assert "entry_id" in meta
                    assert "ud_constraints" in meta
                    assert "guards" in meta
                    return
        
        assert False, "No 'inside' span found"
    
    def test_token_indexing(self, mwe_normalizer, nlp_en):
        """Test that spans are correctly indexed by token."""
        doc = nlp_en("The book is inside the box.")
        processed_doc = mwe_normalizer(doc)
        
        assert processed_doc._.mwe_spans_by_token is not None
        
        # Check that each token in a span is indexed
        for token_indices, spans in processed_doc._.mwe_spans_by_token.items():
            assert isinstance(token_indices, int)
            assert isinstance(spans, list)
            for span in spans:
                assert isinstance(span, spacy.tokens.Span)
                # Check that the token index is within the span
                for token in span:
                    if token.i in processed_doc._.mwe_spans_by_token:
                        assert span in processed_doc._.mwe_spans_by_token[token.i]


class TestSpatialMapsLoading:
    """Test spatial maps loading functionality."""
    
    def test_load_spatial_maps(self):
        """Test that spatial maps can be loaded."""
        maps = load_spatial_maps()
        assert isinstance(maps, dict)
        assert "en" in maps
        assert "es" in maps
        assert "fr" in maps
        assert "de" in maps
        
        # Check structure
        for lang in ["en", "es", "fr", "de"]:
            assert "entries" in maps[lang]
            assert isinstance(maps[lang]["entries"], list)
            
            # Check at least one entry has required fields
            if maps[lang]["entries"]:
                entry = maps[lang]["entries"][0]
                assert "id" in entry
                assert "category" in entry
                assert "prime_hint" in entry
                assert "canonical_lemma" in entry
                assert "surface_variants" in entry
                assert "guards" in entry


if __name__ == "__main__":
    pytest.main([__file__])

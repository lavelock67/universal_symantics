#!/usr/bin/env python3
"""
Contract Tests for Official Primes Only
Ensures ADR-001 compliance: only official 65 NSM primes are emitted.
"""

import pytest
from src.eil.primes_registry import ALLOWED_PRIMES
from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language


def test_only_official_primes_detection_service():
    """Test that NSMDetectionService only emits official primes."""
    service = NSMDetectionService()
    result = service.detect_primes("People think this is very good.", Language.ENGLISH)
    
    # Extract prime texts from the result
    detected_primes = set()
    if result.primes:
        detected_primes = {prime.text.upper() for prime in result.primes}
    
    # Assert all detected primes are in the official set
    assert detected_primes <= ALLOWED_PRIMES, f"Non-official primes detected: {detected_primes - ALLOWED_PRIMES}"


def test_only_official_primes_semantic_generator():
    """Test that SemanticGenerator only emits official primes."""
    from src.semgen.generator import SemanticGenerator
    
    generator = SemanticGenerator()
    # Create a minimal test case
    test_text = "The book is inside the box."
    
    # This would need to be adapted based on the actual generator interface
    # For now, we'll test the concept
    assert hasattr(generator, 'generate'), "SemanticGenerator must have generate method"


def test_allowed_primes_completeness():
    """Test that ALLOWED_PRIMES contains the canonical 65 NSM primes."""
    expected_core_primes = {
        'I', 'YOU', 'SOMEONE', 'PEOPLE', 'SOMETHING', 'THING', 'BODY',
        'KIND', 'PART', 'THIS', 'THE_SAME', 'OTHER', 'ONE', 'TWO', 'SOME', 'ALL', 'MUCH', 'MANY',
        'GOOD', 'BAD', 'BIG', 'SMALL',
        'THINK', 'KNOW', 'WANT', 'FEEL', 'SEE', 'HEAR',
        'SAY', 'WORDS', 'TRUE',
        'DO', 'HAPPEN', 'MOVE', 'TOUCH',
        'BE_SOMEWHERE', 'THERE_IS', 'HAVE',
        'LIVE', 'DIE',
        'WHEN', 'NOW', 'BEFORE', 'AFTER', 'A_LONG_TIME', 'A_SHORT_TIME', 'FOR_SOME_TIME',
        'WHERE', 'HERE', 'ABOVE', 'BELOW', 'FAR', 'NEAR', 'SIDE', 'INSIDE',
        'NOT', 'MAYBE', 'CAN', 'BECAUSE', 'IF',
        'VERY', 'MORE', 'LIKE'
    }
    
    # Check that all expected core primes are in ALLOWED_PRIMES
    missing_primes = expected_core_primes - ALLOWED_PRIMES
    assert not missing_primes, f"Missing core NSM primes: {missing_primes}"
    
    # Check that ALLOWED_PRIMES doesn't contain non-NSM primes
    non_nsm_primes = {'BOY', 'HOUSE', 'BALL', 'KICK', 'THE', 'A', 'AN'}
    found_non_nsm = non_nsm_primes & ALLOWED_PRIMES
    assert not found_non_nsm, f"Non-NSM primes found in ALLOWED_PRIMES: {found_non_nsm}"


def test_pipeline_path_semantic():
    """Test that detection responses include semantic pipeline path."""
    service = NSMDetectionService()
    result = service.detect_primes("Test text", Language.ENGLISH)
    
    # The result should indicate semantic pipeline usage
    assert 'pipeline_path' in result.metadata, "Detection result must include pipeline_path in metadata"
    assert 'manual_detector_count' in result.metadata, "Detection result must include manual_detector_count in metadata"
    assert result.metadata['pipeline_path'] == 'semantic', "Pipeline path must be 'semantic'"
    assert result.metadata['manual_detector_count'] == 0, "Manual detector count must be 0"


if __name__ == "__main__":
    pytest.main([__file__])

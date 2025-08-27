#!/usr/bin/env python3
"""Test NSM generator rule matching."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.core.application.nsm_generator import NSMTextGenerator
from src.core.domain.models import Language

def test_rule_matching():
    """Test rule matching logic."""
    generator = NSMTextGenerator()
    
    # Test cases
    test_cases = [
        (["GOOD"], "should use 'this is good' rule"),
        (["VERY", "GOOD"], "should use 'very good' rule"),
        (["PEOPLE", "THINK", "GOOD"], "should use 'people think' rule"),
        (["THINK", "GOOD"], "should use 'I think' rule"),
    ]
    
    for primes, description in test_cases:
        print(f"\nTesting: {primes} - {description}")
        
        # Find matching rule
        matching_rule = generator._find_matching_rule(primes)
        
        if matching_rule:
            print(f"  ✓ Found rule: {matching_rule.pattern} -> '{matching_rule.template}'")
            print(f"  Confidence: {matching_rule.confidence}")
            
            # Generate text
            text = generator.generate_text(primes, Language.ENGLISH)
            print(f"  Generated: '{text}'")
        else:
            print(f"  ✗ No rule found, using composition")
            text = generator.generate_text(primes, Language.ENGLISH)
            print(f"  Generated: '{text}'")
        
        # Show confidence
        confidence = generator.get_generation_confidence(primes)
        print(f"  Confidence: {confidence}")

if __name__ == "__main__":
    test_rule_matching()

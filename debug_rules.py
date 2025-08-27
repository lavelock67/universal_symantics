#!/usr/bin/env python3
"""Debug NSM generator rule matching."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.core.application.nsm_generator import NSMTextGenerator
from src.core.domain.models import Language

def debug_rule_matching():
    """Debug the rule matching logic."""
    generator = NSMTextGenerator()
    
    test_cases = [
        ["VERY", "GOOD"],
        ["GOOD"],
        ["PEOPLE", "THINK", "GOOD"],
        ["NOT", "GOOD"],
    ]
    
    for primes in test_cases:
        print(f"\nTesting primes: {primes}")
        
        # Check each rule
        for i, rule in enumerate(generator.grammar_rules):
            score = generator._calculate_rule_match_score(rule.pattern, primes)
            if score > 0:
                print(f"  Rule {i}: {rule.pattern} -> score {score:.2f}")
        
        # Find best rule
        best_rule = generator._find_matching_rule(primes)
        if best_rule:
            print(f"  ✓ Best rule: {best_rule.pattern} -> '{best_rule.template}'")
        else:
            print(f"  ✗ No matching rule found")
        
        # Generate text
        text = generator.generate_text(primes, Language.ENGLISH)
        print(f"  Generated: '{text}'")

if __name__ == "__main__":
    debug_rule_matching()

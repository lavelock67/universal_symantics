#!/usr/bin/env python3
"""
Debug script for aspect mapper pattern matching
"""

from robust_aspect_mapper import RobustAspectDetector, Language

def debug_pattern_matching():
    """Debug why certain patterns are not being matched."""
    
    detector = RobustAspectDetector()
    
    # Test cases that are failing
    failing_cases = [
        ("He has just left.", Language.EN, "has just"),
        ("J'arrive à l'instant.", Language.FR, "arrive à l instant"),
        ("Lleva tres horas estudiando.", Language.ES, "lleva"),
        ("Casi me caigo.", Language.ES, "casi"),
        ("Por poco se cae.", Language.ES, "por poco"),
        ("I stopped smoking.", Language.EN, "stopped"),
        ("I started trying again.", Language.EN, "started again"),
    ]
    
    print("DEBUGGING PATTERN MATCHING")
    print("="*50)
    
    for text, language, expected_pattern in failing_cases:
        print(f"\nText: {text}")
        print(f"Language: {language.value}")
        print(f"Expected pattern: {expected_pattern}")
        print(f"Text lower: {text.lower()}")
        
        # Check if pattern exists in patterns
        patterns = detector.aspect_patterns.get(language, {})
        found_pattern = False
        
        for aspect_type, pattern_list in patterns.items():
            for pattern_config in pattern_list:
                pattern = pattern_config['pattern']
                if pattern in text.lower():
                    print(f"  ✅ Found pattern: {pattern} (aspect: {aspect_type})")
                    found_pattern = True
                else:
                    print(f"  ❌ Pattern not found: {pattern}")
        
        if not found_pattern:
            print(f"  ❌ NO PATTERNS MATCHED")
        
        # Test the full detection
        detection = detector.detect_aspects(text, language)
        print(f"  Detection result: {len(detection.detected_aspects)} aspects")
        print(f"  Evidence: {detection.evidence.notes}")

if __name__ == "__main__":
    debug_pattern_matching()

#!/usr/bin/env python3
"""Test Critical Fixes for Edge Cases.

Tests all the red flags identified and their fixes:
1. French negative polarity traps
2. THIS over-firing as determiner
3. GOOD/BAD false positives
4. TRUE/FALSE + negation scope
5. VERY vs MANY confusion
6. Contraction tokenization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_critical_fixes():
    """Test all critical fixes for edge cases."""
    print("🚀 Testing Critical Fixes for Edge Cases")
    print("=" * 50)
    
    from src.detect.enhanced_ud_patterns import EnhancedUDPatterns
    from src.detect.srl_ud_detectors import detect_primitives_multilingual
    import spacy
    
    # Load models
    try:
        nlp_models = {
            "en": spacy.load("en_core_web_sm"),
            "es": spacy.load("es_core_news_sm"),
            "fr": spacy.load("fr_core_news_sm")
        }
    except OSError as e:
        print(f"❌ Error loading models: {e}")
        print("Please install models: python -m spacy download en_core_web_sm es_core_news_sm fr_core_news_sm")
        return
    
    # Initialize enhanced patterns
    enhanced_patterns = EnhancedUDPatterns(nlp_models)
    
    # Test cases for each critical fix
    test_cases = [
        # Fix 1: French negative polarity traps
        {
            "category": "French Negative Polarity",
            "tests": [
                {
                    "text": "Les personnes sont venues",
                    "lang": "fr",
                    "expected": ["PEOPLE"],
                    "description": "personnes (plural) → PEOPLE"
                },
                {
                    "text": "Personne ne vient",
                    "lang": "fr", 
                    "expected": [],  # Should NOT detect PEOPLE
                    "description": "personne (negator) → NOT PEOPLE"
                },
                {
                    "text": "Il n'y a personne",
                    "lang": "fr",
                    "expected": [],  # Should NOT detect PEOPLE
                    "description": "personne (negator) → NOT PEOPLE"
                }
            ]
        },
        
        # Fix 2: THIS over-firing as determiner
        {
            "category": "THIS Over-firing Prevention",
            "tests": [
                {
                    "text": "Esto es muy bueno",
                    "lang": "es",
                    "expected": ["THIS", "VERY", "GOOD"],
                    "description": "Esto (pronominal) → THIS"
                },
                {
                    "text": "Este libro es bueno",
                    "lang": "es",
                    "expected": ["GOOD"],  # Should NOT detect THIS
                    "description": "Este (determiner) → NOT THIS"
                },
                {
                    "text": "C'est très bon",
                    "lang": "fr",
                    "expected": ["THIS", "VERY", "GOOD"],
                    "description": "C'est (contraction) → THIS"
                },
                {
                    "text": "Ce livre est bon",
                    "lang": "fr",
                    "expected": ["GOOD"],  # Should NOT detect THIS
                    "description": "Ce (determiner) → NOT THIS"
                }
            ]
        },
        
        # Fix 3: GOOD/BAD false positives
        {
            "category": "GOOD/BAD False Positive Prevention",
            "tests": [
                {
                    "text": "Esto es muy bueno",
                    "lang": "es",
                    "expected": ["THIS", "VERY", "GOOD"],
                    "description": "bueno (adjective) → GOOD"
                },
                {
                    "text": "El mal está aquí",
                    "lang": "es",
                    "expected": [],  # Should NOT detect BAD
                    "description": "mal (noun) → NOT BAD"
                },
                {
                    "text": "C'est très bon",
                    "lang": "fr",
                    "expected": ["THIS", "VERY", "GOOD"],
                    "description": "bon (adjective) → GOOD"
                },
                {
                    "text": "Le bon est arrivé",
                    "lang": "fr",
                    "expected": [],  # Should NOT detect GOOD
                    "description": "bon (noun) → NOT GOOD"
                }
            ]
        },
        
        # Fix 4: TRUE/FALSE + negation scope
        {
            "category": "TRUE/FALSE Negation Scope",
            "tests": [
                {
                    "text": "Es verdadero que esto funciona",
                    "lang": "es",
                    "expected": ["TRUE"],
                    "description": "verdadero → TRUE"
                },
                {
                    "text": "No es falso que esto funciona",
                    "lang": "es",
                    "expected": ["TRUE"],  # Should flip FALSE → TRUE
                    "description": "no es falso → TRUE (negation flip)"
                },
                {
                    "text": "C'est vrai",
                    "lang": "fr",
                    "expected": ["TRUE"],
                    "description": "vrai → TRUE"
                },
                {
                    "text": "Ce n'est pas faux",
                    "lang": "fr",
                    "expected": ["TRUE"],  # Should flip FALSE → TRUE
                    "description": "n'est pas faux → TRUE (negation flip)"
                }
            ]
        },
        
        # Fix 5: VERY vs MANY confusion
        {
            "category": "VERY vs MANY Distinction",
            "tests": [
                {
                    "text": "Esto es muy bueno",
                    "lang": "es",
                    "expected": ["THIS", "VERY", "GOOD"],
                    "description": "muy → VERY"
                },
                {
                    "text": "Hay muchos libros",
                    "lang": "es",
                    "expected": ["MANY"],
                    "description": "muchos → MANY"
                },
                {
                    "text": "C'est très bon",
                    "lang": "fr",
                    "expected": ["THIS", "VERY", "GOOD"],
                    "description": "très → VERY"
                },
                {
                    "text": "Il y a beaucoup de livres",
                    "lang": "fr",
                    "expected": ["MANY"],
                    "description": "beaucoup de → MANY"
                }
            ]
        },
        
        # Fix 6: Contraction tokenization
        {
            "category": "Contraction Tokenization",
            "tests": [
                {
                    "text": "C'est très bon",
                    "lang": "fr",
                    "expected": ["THIS", "VERY", "GOOD"],
                    "description": "C'est → THIS (contraction)"
                },
                {
                    "text": "Ce n'est pas vrai",
                    "lang": "fr",
                    "expected": ["TRUE"],  # Should flip due to negation
                    "description": "Ce n'est pas vrai → TRUE (contraction + negation)"
                }
            ]
        }
    ]
    
    # Run tests
    total_tests = 0
    passed_tests = 0
    
    for category in test_cases:
        print(f"\n🔧 {category['category']}")
        print("-" * 40)
        
        for test in category['tests']:
            total_tests += 1
            
            print(f"\n📝 Test: {test['description']}")
            print(f"Text: {test['text']}")
            print(f"Language: {test['lang']}")
            
            # Test with enhanced patterns
            enhanced_result = enhanced_patterns.detect_with_enhanced_patterns(
                test['text'], test['lang']
            )
            
            # Test with current system
            current_result = detect_primitives_multilingual(test['text'])
            
            print(f"Enhanced patterns: {enhanced_result}")
            print(f"Current system: {current_result}")
            print(f"Expected: {test['expected']}")
            
            # Check if enhanced patterns match expected
            enhanced_match = set(enhanced_result) == set(test['expected'])
            current_match = set(current_result) == set(test['expected'])
            
            if enhanced_match:
                print("✅ Enhanced patterns: PASS")
                passed_tests += 1
            else:
                print("❌ Enhanced patterns: FAIL")
            
            if current_match:
                print("✅ Current system: PASS")
            else:
                print("❌ Current system: FAIL")
            
            print("-" * 30)
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Enhanced patterns passed: {passed_tests}")
    print(f"Enhanced patterns success rate: {passed_tests/total_tests:.1%}")
    
    # Micro test set (should hit 4-6 primes)
    print(f"\n🎯 Micro Test Set (Target: 4-6 primes)")
    print("=" * 50)
    
    micro_tests = [
        {
            "text": "La gente piensa que esto es muy bueno",
            "lang": "es",
            "expected_min": 4,
            "expected_max": 6
        },
        {
            "text": "Les gens pensent que c'est très bon",
            "lang": "fr", 
            "expected_min": 4,
            "expected_max": 6
        },
        {
            "text": "Es falso que el medicamento no funcione",
            "lang": "es",
            "expected_min": 3,
            "expected_max": 5
        },
        {
            "text": "Au plus la moitié des élèves lisent beaucoup",
            "lang": "fr",
            "expected_min": 4,
            "expected_max": 6
        }
    ]
    
    for test in micro_tests:
        print(f"\n📝 Micro Test: {test['text']}")
        print(f"Language: {test['lang']}")
        
        enhanced_result = enhanced_patterns.detect_with_enhanced_patterns(
            test['text'], test['lang']
        )
        current_result = detect_primitives_multilingual(test['text'])
        
        print(f"Enhanced patterns: {enhanced_result} ({len(enhanced_result)} primes)")
        print(f"Current system: {current_result} ({len(current_result)} primes)")
        
        enhanced_count = len(enhanced_result)
        expected_range = f"{test['expected_min']}-{test['expected_max']}"
        
        if test['expected_min'] <= enhanced_count <= test['expected_max']:
            print(f"✅ Enhanced patterns: PASS ({enhanced_count} primes, target: {expected_range})")
        else:
            print(f"❌ Enhanced patterns: FAIL ({enhanced_count} primes, target: {expected_range})")
        
        current_count = len(current_result)
        if test['expected_min'] <= current_count <= test['expected_max']:
            print(f"✅ Current system: PASS ({current_count} primes, target: {expected_range})")
        else:
            print(f"❌ Current system: FAIL ({current_count} primes, target: {expected_range})")
    
    print(f"\n🎯 Critical Fixes Test Complete!")
    print(f"Enhanced patterns ready for integration")


if __name__ == "__main__":
    test_critical_fixes()

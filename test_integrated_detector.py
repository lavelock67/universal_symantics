#!/usr/bin/env python3
"""Test Integrated Detector with Critical Fixes."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_integrated_detector():
    """Test the integrated detector with all critical fixes."""
    print("🚀 Testing Integrated Detector with Critical Fixes")
    print("=" * 50)
    
    from src.detect.integrated_detector import IntegratedDetector
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
    
    # Initialize integrated detector
    integrated_detector = IntegratedDetector(nlp_models)
    
    # Test cases for micro test set (target: 4-6 primes)
    micro_tests = [
        {
            "text": "La gente piensa que esto es muy bueno",
            "lang": "es",
            "expected_min": 4,
            "expected_max": 6,
            "description": "Spanish: People think this is very good"
        },
        {
            "text": "Les gens pensent que c'est très bon",
            "lang": "fr",
            "expected_min": 4,
            "expected_max": 6,
            "description": "French: People think this is very good"
        },
        {
            "text": "Es falso que el medicamento no funcione",
            "lang": "es",
            "expected_min": 3,
            "expected_max": 5,
            "description": "Spanish: It is false that the medicine doesn't work"
        },
        {
            "text": "Au plus la moitié des élèves lisent beaucoup",
            "lang": "fr",
            "expected_min": 4,
            "expected_max": 6,
            "description": "French: At most half of the students read a lot"
        }
    ]
    
    print(f"\n🎯 Micro Test Set (Target: 4-6 primes)")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for test in micro_tests:
        total_tests += 1
        
        print(f"\n📝 Test: {test['description']}")
        print(f"Text: {test['text']}")
        print(f"Language: {test['lang']}")
        
        # Test integrated detector
        integrated_result = integrated_detector.detect_primes_integrated(test['text'], test['lang'])
        
        # Test current system
        current_result = detect_primitives_multilingual(test['text'])
        
        # Get detailed statistics
        stats = integrated_detector.get_detection_statistics(test['text'], test['lang'])
        
        print(f"Integrated detector: {integrated_result} ({len(integrated_result)} primes)")
        print(f"Current system: {current_result} ({len(current_result)} primes)")
        print(f"Expected range: {test['expected_min']}-{test['expected_max']} primes")
        
        # Show component breakdown
        print(f"  UD patterns: {stats['ud_primes']}")
        print(f"  MWE detection: {stats['mwe_primes']}")
        print(f"  Lexical patterns: {stats['lexical_primes']}")
        print(f"  Processing time: {stats['processing_time']:.3f}s")
        
        # Check if integrated detector meets target
        integrated_count = len(integrated_result)
        expected_range = f"{test['expected_min']}-{test['expected_max']}"
        
        if test['expected_min'] <= integrated_count <= test['expected_max']:
            print(f"✅ Integrated detector: PASS ({integrated_count} primes, target: {expected_range})")
            passed_tests += 1
        else:
            print(f"❌ Integrated detector: FAIL ({integrated_count} primes, target: {expected_range})")
        
        # Check current system
        current_count = len(current_result)
        if test['expected_min'] <= current_count <= test['expected_max']:
            print(f"✅ Current system: PASS ({current_count} primes, target: {expected_range})")
        else:
            print(f"❌ Current system: FAIL ({current_count} primes, target: {expected_range})")
        
        print("-" * 50)
    
    # Summary
    print(f"\n📊 Micro Test Summary")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Integrated detector passed: {passed_tests}")
    print(f"Integrated detector success rate: {passed_tests/total_tests:.1%}")
    
    # Test critical fixes specifically
    print(f"\n🔧 Critical Fixes Validation")
    print("=" * 50)
    
    critical_tests = [
        # French negative polarity
        {
            "text": "Personne ne vient",
            "lang": "fr",
            "expected": [],  # Should NOT detect PEOPLE
            "description": "French negative polarity: personne (negator)"
        },
        {
            "text": "Les personnes sont venues",
            "lang": "fr",
            "expected": ["PEOPLE"],  # Should detect PEOPLE
            "description": "French negative polarity: personnes (plural)"
        },
        
        # THIS over-firing prevention
        {
            "text": "Esto es muy bueno",
            "lang": "es",
            "expected": ["THIS", "VERY", "GOOD"],  # Should detect THIS
            "description": "Spanish THIS: esto (pronominal)"
        },
        {
            "text": "Este libro es bueno",
            "lang": "es",
            "expected": ["GOOD"],  # Should NOT detect THIS
            "description": "Spanish THIS: este (determiner)"
        },
        
        # TRUE/FALSE negation scope
        {
            "text": "No es falso que esto funciona",
            "lang": "es",
            "expected": ["TRUE"],  # Should flip FALSE → TRUE
            "description": "Spanish negation flip: no es falso → TRUE"
        },
        {
            "text": "Ce n'est pas faux",
            "lang": "fr",
            "expected": ["TRUE"],  # Should flip FALSE → TRUE
            "description": "French negation flip: n'est pas faux → TRUE"
        }
    ]
    
    critical_passed = 0
    critical_total = 0
    
    for test in critical_tests:
        critical_total += 1
        
        print(f"\n📝 Critical Test: {test['description']}")
        print(f"Text: {test['text']}")
        
        integrated_result = integrated_detector.detect_primes_integrated(test['text'], test['lang'])
        current_result = detect_primitives_multilingual(test['text'])
        
        print(f"Integrated: {integrated_result}")
        print(f"Current: {current_result}")
        print(f"Expected: {test['expected']}")
        
        # Check if integrated detector matches expected
        integrated_match = set(integrated_result) == set(test['expected'])
        current_match = set(current_result) == set(test['expected'])
        
        if integrated_match:
            print("✅ Integrated detector: PASS")
            critical_passed += 1
        else:
            print("❌ Integrated detector: FAIL")
        
        if current_match:
            print("✅ Current system: PASS")
        else:
            print("❌ Current system: FAIL")
    
    # Critical fixes summary
    print(f"\n📊 Critical Fixes Summary")
    print("=" * 50)
    print(f"Critical tests: {critical_total}")
    print(f"Integrated detector passed: {critical_passed}")
    print(f"Critical fixes success rate: {critical_passed/critical_total:.1%}")
    
    # Overall summary
    print(f"\n🎯 Overall Summary")
    print("=" * 50)
    print(f"Micro test success rate: {passed_tests/total_tests:.1%}")
    print(f"Critical fixes success rate: {critical_passed/critical_total:.1%}")
    
    if passed_tests/total_tests >= 0.75 and critical_passed/critical_total >= 0.8:
        print("🎉 Integrated detector ready for production!")
    else:
        print("⚠️  Integrated detector needs refinement")
    
    print(f"\n🚀 Integrated Detector Test Complete!")


if __name__ == "__main__":
    test_integrated_detector()

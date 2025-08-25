#!/usr/bin/env python3
"""
Comprehensive Robustness Test for Aspect Mapper
Tests for theater code and real functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robust_aspect_mapper import RobustAspectDetector, Language, AspectType


def test_aspect_mapper_robustness():
    """Test the aspect mapper with adversarial and edge cases."""
    
    detector = RobustAspectDetector()
    
    # === ADVERSARIAL TEST CASES ===
    adversarial_cases = [
        # Test overly simple patterns that should NOT trigger
        {
            'text': "I just want to go home.",  # "just" as adverb, not aspect
            'language': Language.EN,
            'expected_aspects': 0,
            'description': "Just as adverb, not aspect marker"
        },
        {
            'text': "This is almost perfect.",  # "almost" as degree, not aspect
            'language': Language.EN,
            'expected_aspects': 0,
            'description': "Almost as degree modifier, not aspect"
        },
        {
            'text': "Stop the car!",  # "stop" as imperative, not aspect
            'language': Language.EN,
            'expected_aspects': 0,
            'description': "Stop as imperative, not aspect"
        },
        {
            'text': "I have been to Paris.",  # "have been" as perfect, not progressive
            'language': Language.EN,
            'expected_aspects': 0,
            'description': "Have been as perfect, not progressive aspect"
        },
        {
            'text': "El libro está en la mesa.",  # "está" as copula, not aspect
            'language': Language.ES,
            'expected_aspects': 0,
            'description': "Está as copula, not aspect marker"
        },
        {
            'text': "Le livre est sur la table.",  # "est" as copula, not aspect
            'language': Language.FR,
            'expected_aspects': 0,
            'description': "Est as copula, not aspect marker"
        },
        
        # Test edge cases that should trigger
        {
            'text': "I have just finished the work.",
            'language': Language.EN,
            'expected_aspects': 1,
            'description': "Have just + past participle"
        },
        {
            'text': "Acaba de salir de la casa.",
            'language': Language.ES,
            'expected_aspects': 1,
            'description': "Acaba de + infinitive"
        },
        {
            'text': "Je viens juste d'arriver.",
            'language': Language.FR,
            'expected_aspects': 1,
            'description': "Viens juste de + infinitive"
        },
        
        # Test complex cases
        {
            'text': "I have been working on this project for three months.",
            'language': Language.EN,
            'expected_aspects': 1,
            'description': "Complex progressive with duration"
        },
        {
            'text': "Lleva tres años estudiando medicina.",
            'language': Language.ES,
            'expected_aspects': 1,
            'description': "Complex ongoing with duration"
        },
        {
            'text': "Il est en train de préparer le dîner.",
            'language': Language.FR,
            'expected_aspects': 1,
            'description': "Complex progressive"
        }
    ]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ROBUSTNESS TEST")
    print("="*80)
    
    correct = 0
    total = len(adversarial_cases)
    
    for i, test_case in enumerate(adversarial_cases):
        print(f"\nTest {i+1}: {test_case['description']}")
        print(f"Text: {test_case['text']} ({test_case['language'].value})")
        print("-" * 60)
        
        detection = detector.detect_aspects(test_case['text'], test_case['language'])
        detected_count = len(detection.detected_aspects)
        expected_count = test_case['expected_aspects']
        
        print(f"Expected aspects: {expected_count}")
        print(f"Detected aspects: {detected_count}")
        
        for aspect in detection.detected_aspects:
            print(f"  - {aspect['aspect_type']}: {aspect['pattern']} (confidence: {aspect['confidence']:.3f})")
        
        if detected_count == expected_count:
            correct += 1
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
            
            # Analyze why it failed
            if detected_count > expected_count:
                print("  → FALSE POSITIVE: Detected aspects when none expected")
            else:
                print("  → FALSE NEGATIVE: Missed expected aspects")
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n" + "="*80)
    print("ROBUSTNESS TEST RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {correct}/{total} ({accuracy:.1%})")
    
    if accuracy >= 0.8:
        print("✅ ROBUSTNESS TEST PASSED")
    else:
        print("❌ ROBUSTNESS TEST FAILED")
        print("   → System may be overfitted to simple test cases")
    
    return accuracy


def test_pattern_specificity():
    """Test if patterns are too broad or too narrow."""
    
    detector = RobustAspectDetector()
    
    # Test cases that should NOT match but might due to overly broad patterns
    false_positive_tests = [
        ("I just want to go home.", Language.EN),
        ("This is almost perfect.", Language.EN),
        ("Stop the car!", Language.EN),
        ("I have been to Paris.", Language.EN),
        ("El libro está en la mesa.", Language.ES),
        ("Le livre est sur la table.", Language.FR),
    ]
    
    false_positives = 0
    total_fp_tests = len(false_positive_tests)
    
    print(f"\n" + "="*80)
    print("PATTERN SPECIFICITY TEST")
    print("="*80)
    
    for text, language in false_positive_tests:
        detection = detector.detect_aspects(text, language)
        if len(detection.detected_aspects) > 0:
            false_positives += 1
            print(f"❌ FALSE POSITIVE: '{text}' detected {len(detection.detected_aspects)} aspects")
    
    fp_rate = false_positives / total_fp_tests if total_fp_tests > 0 else 0.0
    print(f"\nFalse Positive Rate: {false_positives}/{total_fp_tests} ({fp_rate:.1%})")
    
    if fp_rate <= 0.1:  # Allow 10% false positives
        print("✅ PATTERN SPECIFICITY ACCEPTABLE")
    else:
        print("❌ PATTERN SPECIFICITY TOO BROAD")
        print("   → Patterns are matching unintended contexts")
    
    return fp_rate


def main():
    """Run comprehensive robustness tests."""
    print("Starting comprehensive aspect mapper robustness tests...")
    
    # Test 1: Adversarial cases
    robustness_accuracy = test_aspect_mapper_robustness()
    
    # Test 2: Pattern specificity
    false_positive_rate = test_pattern_specificity()
    
    # Overall assessment
    print(f"\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    
    if robustness_accuracy >= 0.8 and false_positive_rate <= 0.1:
        print("✅ SYSTEM IS ROBUST - No theater code detected")
        print("   → Patterns are specific enough")
        print("   → Handles adversarial cases well")
    else:
        print("❌ SYSTEM MAY HAVE THEATER CODE")
        if robustness_accuracy < 0.8:
            print("   → Poor performance on adversarial cases")
        if false_positive_rate > 0.1:
            print("   → Patterns are too broad (false positives)")
        print("   → Consider improving pattern specificity and test coverage")


if __name__ == "__main__":
    main()

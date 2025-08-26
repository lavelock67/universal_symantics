#!/usr/bin/env python3
"""Test all Priority 1-4 improvements."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_all_improvements():
    """Test all Priority 1-4 improvements."""
    print("🚀 Testing All Priority 1-4 Improvements")
    print("=" * 60)
    
    from src.detect.srl_ud_detectors import detect_primitives_multilingual
    
    # Test cases
    test_cases = [
        # Priority 1: MWE Detection
        ("At most half of the students read a lot of books", "EN", ["NOT", "MORE", "HALF", "MANY", "PEOPLE", "READ"]),
        
        # Priority 2: Spanish Mental Predicates
        ("La gente piensa que esto es muy bueno", "ES", ["PEOPLE", "THINK", "THIS", "VERY", "GOOD"]),
        
        # Priority 2: French Mental Predicates
        ("Les gens pensent que c'est très bon", "FR", ["PEOPLE", "THINK", "THIS", "VERY", "GOOD"]),
        
        # Priority 3: Scope-Aware Quantifiers
        ("Au plus la moitié des élèves lisent", "FR", ["NOT", "MORE", "HALF", "PEOPLE", "READ"]),
        
        # Priority 3: Safety-Critical Negation
        ("Es falso que el medicamento no funcione", "ES", ["FALSE", "NOT", "DO"]),
        
        # Priority 4: Pragmatics - Politeness
        ("Please send me the report now", "EN", ["PLEASE", "DO", "NOW"]),
        
        # Priority 4: Pragmatics - Modality
        ("You must complete this task", "EN", ["YOU", "MUST", "DO", "THIS"]),
        
        # Priority 4: Pragmatics - Greetings
        ("Hello, thank you for your help", "EN", ["HELLO", "THANK", "YOU"]),
    ]
    
    total_tests = len(test_cases)
    successful_tests = 0
    
    for i, (text, lang, expected) in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {lang} - {text}")
        print(f"Expected: {expected}")
        
        try:
            detected = detect_primitives_multilingual(text)
            print(f"Detected: {detected}")
            
            # Calculate precision and recall
            detected_set = set(detected)
            expected_set = set(expected)
            
            if detected_set:
                precision = len(detected_set & expected_set) / len(detected_set)
                recall = len(detected_set & expected_set) / len(expected_set)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0
            
            print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
            
            if f1 > 0.3:  # At least 30% F1 score
                print("✅ PASS")
                successful_tests += 1
            else:
                print("❌ FAIL")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n🎯 SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests >= 6:
        print("🎉 EXCELLENT: Most improvements working!")
    elif successful_tests >= 4:
        print("👍 GOOD: Several improvements working!")
    else:
        print("⚠️  NEEDS WORK: Many improvements need attention")

if __name__ == "__main__":
    test_all_improvements()

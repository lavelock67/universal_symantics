#!/usr/bin/env python3
"""
Test Missing Primes Fixed

This script tests if the missing primes are now being detected correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.application.services import NSMDetectionService
from core.domain.models import Language

def test_missing_primes():
    """Test each missing prime individually."""
    
    # Missing primes from our analysis
    missing_primes = [
        "ABOVE", "A_LONG_TIME", "A_SHORT_TIME", "BE_SOMEONE", "BE_SOMEWHERE", 
        "FOR_SOME_TIME", "HAPPEN", "INSIDE", "NEAR", "ONE", "THERE_IS", 
        "THE_SAME", "WHERE", "WORDS"
    ]
    
    # Test sentences for each missing prime
    test_sentences = {
        "ABOVE": ["Look above the line", "The bird flew above the tree", "Above all else"],
        "A_LONG_TIME": ["I waited a long time", "It took a long time", "For a long time"],
        "A_SHORT_TIME": ["I stayed for a short time", "It was a short time", "In a short time"],
        "BE_SOMEONE": ["I want to be someone", "You can be someone", "To be someone"],
        "BE_SOMEWHERE": ["I want to be somewhere", "You can be somewhere", "To be somewhere"],
        "FOR_SOME_TIME": ["I waited for some time", "It lasted for some time", "For some time"],
        "HAPPEN": ["What will happen", "This can happen", "It might happen"],
        "INSIDE": ["Look inside the box", "Go inside the house", "Inside the room"],
        "NEAR": ["Stay near me", "It's near the door", "Near the window"],
        "ONE": ["I have one thing", "One person came", "Only one"],
        "THERE_IS": ["There is someone here", "There are many things", "There is a way"],
        "THE_SAME": ["It's the same thing", "We think the same", "The same person"],
        "WHERE": ["Where are you going", "I know where it is", "Where is it"],
        "WORDS": ["These are words", "I know these words", "The words are clear"]
    }
    
    detection_service = NSMDetectionService()
    
    print("🔍 TESTING MISSING PRIMES (FIXED)")
    print("=" * 60)
    
    detected_count = 0
    total_tests = 0
    
    for prime in missing_primes:
        print(f"\n🎯 Testing prime: {prime}")
        print("-" * 40)
        
        sentences = test_sentences.get(prime, [f"Test {prime}"])
        prime_detected = False
        
        for i, sentence in enumerate(sentences):
            total_tests += 1
            print(f"  Test {i+1}: '{sentence}'")
            
            try:
                result = detection_service.detect_primes(sentence, Language.ENGLISH)
                detected_primes = [p.text.upper() for p in result.primes]
                
                if prime in detected_primes:
                    print(f"    ✅ SUCCESS: {prime} detected!")
                    prime_detected = True
                    detected_count += 1
                else:
                    print(f"    ❌ FAILED: {prime} not detected")
                    print(f"    📝 Detected: {detected_primes}")
                        
            except Exception as e:
                print(f"    💥 ERROR: {e}")
        
        if not prime_detected:
            print(f"    🔍 Prime {prime} not detected in any test sentence")
    
    print(f"\n📊 SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Successful detections: {detected_count}")
    print(f"Success rate: {detected_count/total_tests*100:.1f}%")

if __name__ == "__main__":
    test_missing_primes()


#!/usr/bin/env python3
"""
Test specific missing primes and add patterns
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_missing_primes():
    """Test specific missing primes"""
    try:
        print("🔍 Testing specific missing primes...")
        
        from src.core.application.services import NSMDetectionService
        from src.core.domain.models import Language
        
        # Create service
        service = NSMDetectionService()
        print("✅ Service created")
        
        # Test specific missing primes
        test_cases = [
            ("good", "GOOD"),
            ("bad", "BAD"),
            ("big", "BIG"),
            ("small", "SMALL"),
            ("true", "TRUE"),
            ("false", "FALSE"),
            ("some", "SOME"),
            ("someone", "SOMEONE"),
            ("something", "SOMETHING"),
            ("one", "ONE"),
            ("two", "TWO"),
            ("more", "MORE"),
            ("like", "LIKE"),
            ("say", "SAY"),
            ("want", "WANT"),
            ("feel", "FEEL"),
            ("do", "DO"),
            ("happen", "HAPPEN"),
            ("move", "MOVE"),
            ("here", "HERE"),
            ("there", "THERE"),
            ("where", "WHERE"),
            ("above", "ABOVE"),
            ("below", "BELOW"),
            ("near", "NEAR"),
            ("far", "FAR"),
            ("inside", "INSIDE"),
            ("because", "BECAUSE"),
            ("maybe", "MAYBE"),
            ("a long time", "A_LONG_TIME"),
            ("a short time", "A_SHORT_TIME"),
            ("for some time", "FOR_SOME_TIME"),
            ("there is", "THERE_IS"),
            ("the same", "THE_SAME"),
        ]
        
        detected_primes = set()
        missing_primes = set()
        
        for text, expected_prime in test_cases:
            try:
                result = service.detect_primes(text, Language.ENGLISH)
                found_primes = [p.text for p in result.primes]
                if expected_prime in found_primes:
                    detected_primes.add(expected_prime)
                    print(f"✅ '{text}' -> {expected_prime}")
                else:
                    missing_primes.add(expected_prime)
                    print(f"❌ '{text}' -> Expected {expected_prime}, found {found_primes}")
            except Exception as e:
                missing_primes.add(expected_prime)
                print(f"💥 '{text}' -> Error: {e}")
        
        print(f"\n📊 Results:")
        print(f"  ✅ Detected: {len(detected_primes)} primes")
        print(f"  ❌ Missing: {len(missing_primes)} primes")
        print(f"  🎯 Coverage: {len(detected_primes)}/{len(test_cases)} ({len(detected_primes)/len(test_cases)*100:.1f}%)")
        
        if missing_primes:
            print(f"\n❌ Missing primes: {sorted(missing_primes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_missing_primes()
    if success:
        print("\n🎉 Test completed!")
    else:
        print("\n💥 Test failed!")


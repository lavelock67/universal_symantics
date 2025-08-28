#!/usr/bin/env python3
"""
Test MWE Detection

This script tests MWE detection specifically for THERE_IS and WHERE.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.application.services import NSMDetectionService
from core.domain.models import Language

def test_mwe_detection():
    """Test MWE detection specifically."""
    
    detection_service = NSMDetectionService()
    
    print("🔍 TESTING MWE DETECTION")
    print("=" * 60)
    
    # Test sentences for THERE_IS
    there_is_sentences = [
        "There is someone here",
        "There are many things",
        "There is a way"
    ]
    
    print("\n🎯 Testing THERE_IS MWE detection")
    print("-" * 40)
    
    for i, sentence in enumerate(there_is_sentences):
        print(f"  Test {i+1}: '{sentence}'")
        
        try:
            # Test MWE detection specifically
            mwes = detection_service.detect_mwes(sentence, Language.ENGLISH)
            print(f"    📝 MWEs detected: {[mwe.text for mwe in mwes]}")
            
            # Test full detection
            result = detection_service.detect_primes(sentence, Language.ENGLISH)
            detected_primes = [p.text.upper() for p in result.primes]
            print(f"    📝 Primes detected: {detected_primes}")
            
            if "THERE_IS" in detected_primes:
                print(f"    ✅ SUCCESS: THERE_IS detected!")
            else:
                print(f"    ❌ FAILED: THERE_IS not detected")
                
        except Exception as e:
            print(f"    💥 ERROR: {e}")
    
    # Test sentences for WHERE
    where_sentences = [
        "Where are you going",
        "I know where it is",
        "Where is it"
    ]
    
    print("\n🎯 Testing WHERE detection")
    print("-" * 40)
    
    for i, sentence in enumerate(where_sentences):
        print(f"  Test {i+1}: '{sentence}'")
        
        try:
            result = detection_service.detect_primes(sentence, Language.ENGLISH)
            detected_primes = [p.text.upper() for p in result.primes]
            print(f"    📝 Primes detected: {detected_primes}")
            
            if "WHERE" in detected_primes:
                print(f"    ✅ SUCCESS: WHERE detected!")
            else:
                print(f"    ❌ FAILED: WHERE not detected")
                
        except Exception as e:
            print(f"    💥 ERROR: {e}")

if __name__ == "__main__":
    test_mwe_detection()


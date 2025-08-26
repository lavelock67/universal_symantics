#!/usr/bin/env python3
"""Test MWE integration."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mwe_integration():
    """Test MWE integration in detection."""
    print("Testing MWE integration...")
    
    try:
        from src.detect.srl_ud_detectors import detect_primitives_multilingual
        from src.detect.mwe_tagger import MWETagger
        
        # Test text
        text = "At most half of the students read a lot of books"
        print(f"Input: {text}")
        
        # Test MWE detection directly
        mwe_tagger = MWETagger()
        mwes = mwe_tagger.detect_mwes(text)
        mwe_primes = mwe_tagger.get_primes_from_mwes(mwes)
        print(f"MWE detection: {mwe_primes}")
        
        # Test integrated detection
        result = detect_primitives_multilingual(text)
        print(f"Integrated result: {result}")
        
        # Check if MWE primes are in result
        missing = [p for p in mwe_primes if p not in result]
        if missing:
            print(f"Missing MWE primes: {missing}")
        else:
            print("All MWE primes found in result!")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_mwe_integration()

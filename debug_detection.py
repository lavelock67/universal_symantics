#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.core.application.services import NSMDetectionService
from src.shared.config.settings import get_settings
from src.core.domain.models import Language

def test_detection():
    """Test the detection service directly."""
    service = NSMDetectionService()
    
    # Test English
    print("Testing English detection:")
    text = "The people think this is very good"
    result = service.detect_primes(text, Language.ENGLISH)
    
    print(f"Input: {text}")
    print(f"Detected primes: {[prime.text for prime in result.primes]}")
    print("Expected: ['PEOPLE', 'THINK', 'THIS', 'VERY', 'GOOD']")
    
    # Test Spanish
    print("\nTesting Spanish detection:")
    text = "La gente piensa que esto es muy bueno"
    result = service.detect_primes(text, Language.SPANISH)
    
    print(f"Input: {text}")
    print(f"Detected primes: {[prime.text for prime in result.primes]}")
    print("Expected: ['PEOPLE', 'THINK', 'THIS', 'VERY', 'GOOD']")
    
    # Test MWE detection
    print("\nTesting MWE detection:")
    text = "At least half of the students read a lot of books"
    mwes = service.detect_mwes(text, Language.ENGLISH)
    
    print(f"Input: {text}")
    print(f"Detected MWEs: {[mwe.text for mwe in mwes]}")
    print("Expected: ['at least', 'half of', 'a lot of']")

if __name__ == "__main__":
    test_detection()

#!/usr/bin/env python3
"""Debug MWE detection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language

def debug_mwe_detection():
    """Debug MWE detection."""
    print("Testing MWE detection...")
    
    # Create detection service
    service = NSMDetectionService()
    
    # Test text
    text = "at least five people"
    language = Language.ENGLISH
    
    # Get MWE patterns
    patterns = service._get_mwe_patterns(language)
    print(f"MWE patterns: {patterns}")
    
    # Test MWE detection
    mwes = service.detect_mwes(text, language)
    print(f"Detected MWEs: {mwes}")
    
    for mwe in mwes:
        print(f"MWE: {mwe.text}, Type: {mwe.type}, Primes: {mwe.primes}")

if __name__ == "__main__":
    debug_mwe_detection()

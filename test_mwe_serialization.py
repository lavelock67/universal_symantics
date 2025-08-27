#!/usr/bin/env python3
"""Test MWE serialization."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language, MWE, MWEType
import json

def test_mwe_serialization():
    """Test MWE serialization."""
    print("Testing MWE serialization...")
    
    # Create a test MWE
    mwe = MWE(
        text="at least",
        type=MWEType.QUANTIFIER,
        primes=["MORE"],
        confidence=0.8,
        start=0,
        end=8,
        language=Language.ENGLISH
    )
    
    print(f"Original MWE: {mwe}")
    print(f"Primes: {mwe.primes}")
    
    # Test JSON serialization
    try:
        # Convert to dict
        mwe_dict = {
            "text": mwe.text,
            "type": mwe.type.value,
            "primes": mwe.primes,
            "confidence": mwe.confidence,
            "start": mwe.start,
            "end": mwe.end,
            "language": mwe.language.value,
            "frequency": mwe.frequency
        }
        
        print(f"MWE dict: {mwe_dict}")
        
        # Test JSON serialization
        json_str = json.dumps(mwe_dict)
        print(f"JSON: {json_str}")
        
        # Test deserialization
        mwe_dict_back = json.loads(json_str)
        print(f"Deserialized: {mwe_dict_back}")
        print(f"Primes after deserialization: {mwe_dict_back['primes']}")
        
    except Exception as e:
        print(f"Serialization error: {e}")

if __name__ == "__main__":
    test_mwe_serialization()

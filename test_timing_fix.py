#!/usr/bin/env python3
"""Test script to verify the timing fix."""

import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language

def test_timing_fix():
    """Test that the timing fix works correctly."""
    print("Testing timing fix...")
    
    # Create detection service
    service = NSMDetectionService()
    
    # Test detection
    text = "People think this is good"
    language = Language.ENGLISH
    
    start_time = time.time()
    result = service.detect_primes(text, language)
    end_time = time.time()
    
    actual_elapsed = end_time - start_time
    
    print(f"Actual elapsed time: {actual_elapsed:.6f} seconds")
    print(f"Result processing time: {result.processing_time:.6f} seconds")
    print(f"Difference: {abs(actual_elapsed - result.processing_time):.6f} seconds")
    
    # Check if the processing time is reasonable (should be < 1 second)
    if result.processing_time > 1.0:
        print("❌ ERROR: Processing time is too high - likely still using Unix timestamp")
        return False
    elif abs(actual_elapsed - result.processing_time) > 0.1:
        print("❌ ERROR: Processing time doesn't match actual elapsed time")
        return False
    else:
        print("✅ SUCCESS: Timing fix is working correctly")
        return True

if __name__ == "__main__":
    success = test_timing_fix()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Debug Detection Service

Test the detection service directly to identify issues.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_detection_service():
    """Test the detection service directly."""
    print("Testing detection service directly...")
    
    try:
        from src.core.application.services import create_detection_service
        from src.core.domain.models import Language
        
        # Create detection service
        detection_service = create_detection_service()
        print("  ✅ Detection service created")
        
        # Test detection
        text = "I think this is good"
        language = Language.ENGLISH
        
        print(f"  Testing detection for: '{text}'")
        result = detection_service.detect_primes(text, language)
        
        print(f"  ✅ Detection successful")
        print(f"  Primes found: {len(result.primes)}")
        for prime in result.primes:
            print(f"    - {prime.text} ({prime.type.value}) confidence: {prime.confidence}")
        
        print(f"  MWEs found: {len(result.mwes)}")
        for mwe in result.mwes:
            print(f"    - {mwe.text} ({mwe.type.value}) confidence: {mwe.confidence}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Detection service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prime_type_enum():
    """Test PrimeType enum."""
    print("\nTesting PrimeType enum...")
    
    try:
        from src.core.domain.models import PrimeType
        
        # Test all enum values
        enum_values = list(PrimeType)
        print(f"  ✅ PrimeType enum has {len(enum_values)} values:")
        for value in enum_values:
            print(f"    - {value.value}")
        
        # Test SPEECH specifically
        speech_type = PrimeType.SPEECH
        print(f"  ✅ SPEECH enum value: {speech_type.value}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PrimeType enum test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_services_import():
    """Test services import."""
    print("\nTesting services import...")
    
    try:
        from src.core.application.services import NSMDetectionService
        print("  ✅ NSMDetectionService import successful")
        
        # Test _get_prime_type method
        service = NSMDetectionService()
        print("  ✅ NSMDetectionService created")
        
        # Test the mapping
        prime_type = service._get_prime_type("WORDS")
        print(f"  ✅ WORDS maps to: {prime_type.value}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Services import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("=" * 60)
    print("DETECTION SERVICE DEBUG")
    print("=" * 60)
    
    # Test PrimeType enum
    if not test_prime_type_enum():
        print("\n❌ PrimeType enum test failed")
        return
    
    # Test services import
    if not test_services_import():
        print("\n❌ Services import test failed")
        return
    
    # Test detection service
    if not test_detection_service():
        print("\n❌ Detection service test failed")
        return
    
    print("\n" + "=" * 60)
    print("✅ ALL DEBUG TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()

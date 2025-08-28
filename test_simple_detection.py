#!/usr/bin/env python3
"""
Simple Detection Test

Test basic detection to identify the issue.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple_detection():
    """Test simple detection."""
    print("Testing Simple Detection...")
    
    try:
        from src.core.application.services import create_detection_service
        from src.core.domain.models import Language
        
        # Create detection service
        detection_service = create_detection_service()
        print("  ✅ Detection service created")
        
        # Test simple text
        test_texts = [
            "I think this is good",
            "I want to do something",
            "This is very big",
            "I feel bad",
            "I can see you",
            "I hear something",
            "I say words",
            "I know this",
            "I live here",
            "I die now"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n  Test {i}: '{text}'")
            try:
                result = detection_service.detect_primes(text, Language.ENGLISH)
                detected_primes = [p.text for p in result.primes]
                print(f"    ✅ Success: {detected_primes}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("=" * 50)
    print("SIMPLE DETECTION TEST")
    print("=" * 50)
    
    test_simple_detection()
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()

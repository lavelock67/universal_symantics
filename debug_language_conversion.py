#!/usr/bin/env python3
"""
Debug language conversion from string to enum
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_language_conversion():
    """Test language conversion"""
    try:
        print("🔍 Testing language conversion...")
        
        from src.core.domain.models import Language, PrimeDetectionRequest
        
        # Test direct enum creation
        print("📝 Testing direct enum creation...")
        lang1 = Language.ENGLISH
        print(f"✅ Language.ENGLISH: {lang1} (type: {type(lang1)})")
        
        # Test string to enum conversion
        print("📝 Testing string to enum conversion...")
        lang2 = Language("en")
        print(f"✅ Language('en'): {lang2} (type: {type(lang2)})")
        
        # Test Pydantic model creation
        print("📝 Testing Pydantic model creation...")
        request = PrimeDetectionRequest(text="test", language="en")
        print(f"✅ PrimeDetectionRequest: language={request.language} (type: {type(request.language)})")
        
        # Test with enum directly
        print("📝 Testing with enum directly...")
        request2 = PrimeDetectionRequest(text="test", language=Language.ENGLISH)
        print(f"✅ PrimeDetectionRequest with enum: language={request2.language} (type: {type(request2.language)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_language_conversion()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")

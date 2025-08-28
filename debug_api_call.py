#!/usr/bin/env python3
"""
Debug the exact API call that's failing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_api_call():
    """Test the exact API call that's failing"""
    try:
        print("🔍 Testing exact API call...")
        
        from src.core.application.services import NSMDetectionService
        from src.core.domain.models import Language
        
        # Create service
        service = NSMDetectionService()
        print("✅ Service created")
        
        # Test with string (this should fail)
        print("📝 Testing with string 'en'...")
        try:
            result = service.detect_primes("I think this is good", "en")
            print("❌ This should have failed but didn't!")
        except AttributeError as e:
            print(f"✅ Correctly failed with: {e}")
        except Exception as e:
            print(f"❌ Failed with wrong error: {e}")
        
        # Test with enum (this should work)
        print("📝 Testing with Language.ENGLISH...")
        try:
            result = service.detect_primes("I think this is good", Language.ENGLISH)
            print(f"✅ Worked! Found {len(result.primes)} primes")
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_call()
    if success:
        print("\n🎉 Test completed!")
    else:
        print("\n💥 Test failed!")

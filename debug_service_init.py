#!/usr/bin/env python3
"""
Debug detection service initialization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_service_init():
    """Test detection service initialization"""
    try:
        print("🔍 Testing detection service initialization...")
        
        # Test import
        print("📝 Testing import...")
        from src.core.application.services import create_detection_service, NSMDetectionService
        print("✅ Import successful")
        
        # Test direct instantiation
        print("📝 Testing direct instantiation...")
        service = NSMDetectionService()
        print("✅ Direct instantiation successful")
        
        # Test factory function
        print("📝 Testing factory function...")
        service2 = create_detection_service()
        print("✅ Factory function successful")
        
        # Test basic detection
        print("📝 Testing basic detection...")
        result = service.detect_primes("I think this is good", "en")
        print(f"✅ Detection successful: {len(result.primes)} primes detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_service_init()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")

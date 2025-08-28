#!/usr/bin/env python3
"""
Simulate the exact API call to see what's happening
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_api_simulation():
    """Simulate the exact API call"""
    try:
        print("🔍 Simulating API call...")
        
        from api.clean_nsm_api import app
        from fastapi.testclient import TestClient
        from src.core.application.services import create_detection_service
        
        # Create test client
        client = TestClient(app)
        
        # Test the exact request that's failing
        test_data = {
            "text": "I think this is very good",
            "language": "en"
        }
        
        print("📝 Testing with JSON data:", test_data)
        
        # Test the endpoint
        response = client.post("/detect", json=test_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Let's also test what happens if we create the request object manually
        print("\n📝 Testing manual request object creation...")
        from src.core.domain.models import PrimeDetectionRequest
        
        try:
            request = PrimeDetectionRequest(**test_data)
            print(f"✅ Request created: language={request.language} (type: {type(request.language)})")
            
            # Test the service call
            service = create_detection_service()
            result = service.detect_primes(request.text, request.language)
            print(f"✅ Service call successful: {len(result.primes)} primes")
            
        except Exception as e:
            print(f"❌ Manual test failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_simulation()
    if success:
        print("\n🎉 Test completed!")
    else:
        print("\n💥 Test failed!")

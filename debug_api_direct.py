#!/usr/bin/env python3
"""
Debug API directly to isolate HTTP 500 errors
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api.clean_nsm_api import app
from fastapi.testclient import TestClient

def test_api_directly():
    """Test the API endpoints directly without HTTP"""
    client = TestClient(app)
    
    print("🔍 Testing API directly...")
    
    # Test the detect endpoint
    test_data = {
        "text": "I think this is very good",
        "language": "en"
    }
    
    try:
        print("📝 Testing /detect endpoint...")
        response = client.post("/detect", json=test_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error testing /detect: {e}")
        import traceback
        traceback.print_exc()
    
    # Test the health endpoint
    try:
        print("\n📝 Testing /health endpoint...")
        response = client.get("/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error testing /health: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_directly()

#!/usr/bin/env python3
"""Debug script to test API timing issue."""

import requests
import json
import time

def test_api_timing():
    """Test the API timing issue."""
    url = "http://localhost:8001/detect"
    data = {
        "text": "People think this is good",
        "language": "en"
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("API Response:")
            print(json.dumps(result, indent=2))
            
            # Check both processing times
            if 'result' in result and 'processing_time' in result['result']:
                result_time = result['result']['processing_time']
                print(f"\nResult processing time: {result_time}")
                
                if result_time > 1000:  # More than 1000 seconds is definitely wrong
                    print("❌ BUG: Result processing time is still using Unix timestamp")
                else:
                    print("✅ Result processing time looks correct")
            
            if 'processing_time' in result:
                api_time = result['processing_time']
                print(f"API processing time: {api_time}")
                
                if api_time > 1000:
                    print("❌ BUG: API processing time is also wrong")
                else:
                    print("✅ API processing time looks correct")
        else:
            print(f"API request failed with status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api_timing()

#!/usr/bin/env python3
"""
Test EIL System

Test script to verify the EIL system is working correctly.
"""

import requests
import json
import time

# API base URL
API_BASE = "http://localhost:8000"

def test_eil_processing():
    """Test EIL processing endpoint."""
    print("Testing EIL Processing...")
    
    test_cases = [
        {
            "text": "I think this is good",
            "language": "en",
            "expected_primes": ["I", "THINK", "THIS", "GOOD"]
        },
        {
            "text": "All children are not playing",
            "language": "en", 
            "expected_primes": ["ALL", "NOT", "DO"]
        },
        {
            "text": "The cat is on the mat",
            "language": "en",
            "expected_primes": ["THING", "ON", "THING"]
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['text']}")
        
        try:
            response = requests.post(
                f"{API_BASE}/eil/process",
                json={
                    "text": test_case["text"],
                    "language": test_case["language"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    data = result["result"]
                    
                    print(f"  ✅ Success")
                    print(f"  Router Decision: {data['router_decision']}")
                    print(f"  Router Confidence: {data['router_confidence']:.3f}")
                    print(f"  Router Reasoning: {data['router_reasoning']}")
                    print(f"  Validation Valid: {data['validation_result']['is_valid']}")
                    print(f"  Validation Errors: {data['validation_result']['error_count']}")
                    print(f"  Validation Warnings: {data['validation_result']['warning_count']}")
                    print(f"  Processing Times: {data['processing_times']}")
                    
                    # Check source graph
                    source_graph = data['source_graph']
                    node_count = len(source_graph.get('nodes', {}))
                    relation_count = len(source_graph.get('relations', {}))
                    print(f"  Source Graph: {node_count} nodes, {relation_count} relations")
                    
                else:
                    print(f"  ❌ API Error: {result.get('error')}")
            else:
                print(f"  ❌ HTTP Error: {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Exception: {str(e)}")

def test_eil_translation():
    """Test EIL translation endpoint."""
    print("\n\nTesting EIL Translation...")
    
    test_cases = [
        {
            "text": "I think this is good",
            "source_language": "en",
            "target_language": "es"
        },
        {
            "text": "The cat is on the mat", 
            "source_language": "en",
            "target_language": "fr"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTranslation Test {i+1}: {test_case['text']} ({test_case['source_language']} → {test_case['target_language']})")
        
        try:
            response = requests.post(
                f"{API_BASE}/eil/translate",
                json=test_case,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    data = result["result"]
                    
                    print(f"  ✅ Success")
                    print(f"  Source: {data['source_text']}")
                    print(f"  Target: {data['target_text']}")
                    print(f"  Translation Confidence: {data['translation_confidence']:.3f}")
                    print(f"  Router Decision: {data['router_decision']}")
                    print(f"  Router Confidence: {data['router_confidence']:.3f}")
                    print(f"  Processing Times: {data['processing_times']}")
                    
                else:
                    print(f"  ❌ API Error: {result.get('error')}")
            else:
                print(f"  ❌ HTTP Error: {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Exception: {str(e)}")

def test_eil_stats():
    """Test EIL stats endpoint."""
    print("\n\nTesting EIL Stats...")
    
    try:
        response = requests.get(f"{API_BASE}/eil/stats", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                stats = result["result"]
                print(f"  ✅ Success")
                print(f"  Config: {json.dumps(stats.get('config', {}), indent=2)}")
                print(f"  Validator Config: {json.dumps(stats.get('validator_config', {}), indent=2)}")
                print(f"  Router Thresholds: {json.dumps(stats.get('router_thresholds', {}), indent=2)}")
            else:
                print(f"  ❌ API Error: {result.get('error')}")
        else:
            print(f"  ❌ HTTP Error: {response.status_code}")
            print(f"  Response: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Exception: {str(e)}")

def test_health():
    """Test API health."""
    print("Testing API Health...")
    
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "healthy":
                print("  ✅ API is healthy")
                return True
            else:
                print(f"  ❌ API unhealthy: {result}")
                return False
        else:
            print(f"  ❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Health check exception: {str(e)}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("EIL SYSTEM TEST")
    print("=" * 60)
    
    # Test health first
    if not test_health():
        print("\n❌ API is not healthy. Please start the server first.")
        return
    
    # Run tests
    test_eil_processing()
    test_eil_translation() 
    test_eil_stats()
    
    print("\n" + "=" * 60)
    print("EIL SYSTEM TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()


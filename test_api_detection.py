#!/usr/bin/env python3

import requests
import json

def test_api_detection():
    """Test the API detection directly."""
    api_url = "http://localhost:8001"
    
    # Test 1: English detection
    print("Testing English detection:")
    response = requests.post(
        f"{api_url}/detect",
        json={
            "text": "The people think this is very good",
            "language": "en",
            "methods": ["spacy", "structured", "multilingual", "mwe"],
            "include_deepnsm": True,
            "include_mdl": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        primes = [prime["text"] for prime in result["result"]["primes"]]
        print(f"Detected primes: {primes}")
        print(f"Expected: ['people', 'think', 'this', 'very', 'good']")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    # Test 2: MWE detection
    print("\nTesting MWE detection:")
    response = requests.post(
        f"{api_url}/mwe",
        json={
            "text": "At least half of the students read a lot of books",
            "language": "en"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        mwes = [mwe["text"] for mwe in result["mwes"]]
        print(f"Detected MWEs: {mwes}")
        print(f"Expected: ['at least', 'half of', 'a lot of']")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    # Test 3: Generation
    print("\nTesting Generation:")
    response = requests.post(
        f"{api_url}/generate",
        json={
            "primes": ["THINK", "GOOD"],
            "target_language": "en"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        generated_text = result["result"]["generated_text"]
        print(f"Generated text: {generated_text}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_api_detection()

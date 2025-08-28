#!/usr/bin/env python3
"""
Test Universal Translator

This script demonstrates the complete universal translation pipeline:
Source Text → Prime Detection → Prime Generation → Target Text
"""

import requests
import json
import time

def test_translation():
    """Test the universal translator."""
    print("🌍 UNIVERSAL TRANSLATOR TEST")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "source_text": "I think this is very good",
            "source_language": "en",
            "target_language": "es",
            "description": "Simple statement with evaluators"
        },
        {
            "source_text": "You know that some people want to do many things",
            "source_language": "en", 
            "target_language": "fr",
            "description": "Complex sentence with multiple primes"
        },
        {
            "source_text": "When the moment comes, you will see all kinds of things",
            "source_language": "en",
            "target_language": "es", 
            "description": "Temporal and spatial concepts"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Source ({test_case['source_language']}): '{test_case['source_text']}'")
        
        try:
            # Test translation
            response = requests.post(
                "http://localhost:8000/translate",
                headers={"Content-Type": "application/json"},
                json={
                    "text": test_case["source_text"],
                    "source_language": test_case["source_language"],
                    "target_language": test_case["target_language"],
                    "strategy": "lexical"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                translation = result["translation"]
                
                print(f"Target ({test_case['target_language']}): '{translation['target_text']}'")
                print(f"Confidence: {translation['confidence']:.2f}")
                print(f"Processing time: {translation['processing_time']:.3f}s")
                print(f"Detected primes: {translation['detected_primes']}")
                print(f"Prime count: {translation['metadata']['prime_count']}")
                
            else:
                print(f"❌ Translation failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test supported languages
    print(f"\n🌐 SUPPORTED LANGUAGES")
    print("=" * 30)
    
    try:
        response = requests.get("http://localhost:8000/translate/languages")
        if response.status_code == 200:
            languages = response.json()["languages"]
            print(f"Detection: {languages['detection']}")
            print(f"Generation: {languages['generation']}")
            print(f"Full pipeline: {languages['full_pipeline']}")
        else:
            print(f"❌ Failed to get languages: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test language coverage
    print(f"\n📊 LANGUAGE COVERAGE")
    print("=" * 30)
    
    for lang in ["en", "es", "fr"]:
        try:
            response = requests.get(f"http://localhost:8000/translate/coverage/{lang}")
            if response.status_code == 200:
                coverage = response.json()["coverage"]
                print(f"{lang.upper()}:")
                print(f"  Detection: {coverage['detection']['supported']}")
                print(f"  Generation: {coverage['generation']['coverage_percentage']:.1f}%")
            else:
                print(f"❌ Failed to get coverage for {lang}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting coverage for {lang}: {e}")

def test_prime_generation():
    """Test prime-to-text generation directly."""
    print(f"\n🔧 PRIME GENERATION TEST")
    print("=" * 40)
    
    # Test cases with specific primes
    test_primes = [
        ["I", "THINK", "THIS", "BE", "VERY", "GOOD"],
        ["YOU", "KNOW", "SOME", "PEOPLE", "WANT", "DO", "MANY", "THING"],
        ["WHEN", "MOMENT", "COME", "YOU", "WILL", "SEE", "ALL", "KIND", "THING"]
    ]
    
    for i, primes in enumerate(test_primes, 1):
        print(f"\n📝 Test {i}: {primes}")
        
        try:
            # Create a mock detection result with these primes
            mock_primes = [{"text": prime, "type": "test", "language": "en", "confidence": 0.8} for prime in primes]
            
            # Test generation to different languages
            for target_lang in ["en", "es", "fr"]:
                response = requests.post(
                    "http://localhost:8000/translate",
                    headers={"Content-Type": "application/json"},
                    json={
                        "text": " ".join(primes),  # Use primes as source text
                        "source_language": "en",
                        "target_language": target_lang,
                        "strategy": "lexical"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    translation = result["translation"]
                    print(f"  {target_lang.upper()}: '{translation['target_text']}'")
                else:
                    print(f"  {target_lang.upper()}: ❌ Failed")
                    
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Universal Translator Test...")
    print("Make sure the API server is running on http://localhost:8000")
    print()
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    # Run tests
    test_translation()
    test_prime_generation()
    
    print(f"\n🎉 Universal Translator Test Complete!")
    print("This demonstrates the complete pipeline:")
    print("Source Text → Prime Detection → Prime Generation → Target Text")

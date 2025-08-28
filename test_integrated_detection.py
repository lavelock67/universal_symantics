#!/usr/bin/env python3
"""
Test script to verify integrated UD and MWE detection.
"""

import requests
import json
import time

def test_detection(text, language="en"):
    """Test prime detection with the integrated system."""
    url = "http://localhost:8001/detect"
    
    payload = {
        "text": text,
        "language": language
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                primes = [prime["text"] for prime in result["result"]["primes"]]
                confidence = result["result"]["confidence"]
                processing_time = result["result"]["processing_time"]
                
                print(f"✅ Input: '{text}'")
                print(f"   Detected primes: {primes}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Processing time: {processing_time:.3f}s")
                print()
                return primes, confidence
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                return [], 0.0
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return [], 0.0
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return [], 0.0

def test_mwe_detection(text, language="en"):
    """Test MWE detection."""
    url = "http://localhost:8001/mwe"
    
    payload = {
        "text": text,
        "language": language
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                mwes = [mwe["text"] for mwe in result["mwes"]]
                primes = result["primes"]
                coverage = result["coverage"]
                
                print(f"✅ MWE Input: '{text}'")
                print(f"   Detected MWEs: {mwes}")
                print(f"   MWE Primes: {primes}")
                print(f"   Coverage: {coverage}")
                print()
                return mwes, primes
            else:
                print(f"❌ MWE Error: {result.get('error', 'Unknown error')}")
                return [], []
        else:
            print(f"❌ MWE HTTP Error: {response.status_code}")
            return [], []
    except Exception as e:
        print(f"❌ MWE Request failed: {str(e)}")
        return [], []

def main():
    """Run comprehensive tests."""
    print("🧪 TESTING INTEGRATED DETECTION SYSTEM")
    print("=" * 50)
    
    # Wait for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(5)
    
    # Test cases for UD detection
    ud_test_cases = [
        "I think this is very good",
        "The people cannot do this",
        "At least half of the students read many books",
        "This thing is bigger than that",
        "When the time comes, we will go there",
        "Because it is cold, I want warm water",
        "If you can see this, then you know it is true",
        "The world is very big and the sky is above us",
        "Some people think that all things are good",
        "Before today, I did not know this place"
    ]
    
    print("🔍 TESTING UD DETECTION (Dependency Parsing)")
    print("-" * 40)
    
    total_primes = 0
    total_confidence = 0.0
    test_count = 0
    
    for text in ud_test_cases:
        primes, confidence = test_detection(text)
        if primes:
            total_primes += len(primes)
            total_confidence += confidence
            test_count += 1
    
    if test_count > 0:
        avg_primes = total_primes / test_count
        avg_confidence = total_confidence / test_count
        print(f"📊 UD Detection Summary:")
        print(f"   Average primes per text: {avg_primes:.1f}")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Successful tests: {test_count}/{len(ud_test_cases)}")
    
    print("\n🔍 TESTING MWE DETECTION (Multi-Word Expressions)")
    print("-" * 40)
    
    mwe_test_cases = [
        "I do not want this",
        "You cannot see that",
        "At least half of the people",
        "A lot of books are here",
        "No more than three things",
        "Some of the students",
        "All of the water",
        "None of the people",
        "Very good and very bad",
        "Not true and not right"
    ]
    
    total_mwes = 0
    total_mwe_primes = 0
    mwe_test_count = 0
    
    for text in mwe_test_cases:
        mwes, primes = test_mwe_detection(text)
        if mwes:
            total_mwes += len(mwes)
            total_mwe_primes += len(primes)
            mwe_test_count += 1
    
    if mwe_test_count > 0:
        avg_mwes = total_mwes / mwe_test_count
        avg_mwe_primes = total_mwe_primes / mwe_test_count
        print(f"📊 MWE Detection Summary:")
        print(f"   Average MWEs per text: {avg_mwes:.1f}")
        print(f"   Average MWE primes per text: {avg_mwe_primes:.1f}")
        print(f"   Successful tests: {mwe_test_count}/{len(mwe_test_cases)}")
    
    print("\n🎯 TESTING CROSS-LINGUAL DETECTION")
    print("-" * 40)
    
    cross_lingual_tests = [
        ("I think this is good", "en"),
        ("Pienso que esto es bueno", "es"),
        ("Je pense que c'est bon", "fr")
    ]
    
    for text, lang in cross_lingual_tests:
        print(f"🌐 Testing {lang.upper()}: '{text}'")
        primes, confidence = test_detection(text, lang)
        if primes:
            print(f"   Expected: ['I', 'THINK', 'THIS', 'GOOD']")
            print(f"   Found: {primes}")
            print(f"   Confidence: {confidence:.3f}")
        print()
    
    print("✅ INTEGRATION TEST COMPLETE")

if __name__ == "__main__":
    main()

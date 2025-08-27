#!/usr/bin/env python3
"""
Test script to verify demo functionality and identify issues.
"""

import requests
import json
import time

def test_api_endpoints():
    """Test all API endpoints to identify issues."""
    base_url = "http://localhost:8001"
    
    print("🔍 Testing API Endpoints...")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Health endpoint working")
        else:
            print(f"   ❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health endpoint error: {e}")
    
    # Test 2: Prime detection
    print("\n2. Testing prime detection...")
    try:
        response = requests.post(
            f"{base_url}/detect",
            json={
                "text": "People think this is very good",
                "language": "en",
                "methods": ["spacy", "structured", "multilingual", "mwe"],
                "include_deepnsm": True,
                "include_mdl": True
            },
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            primes = []
            if "result" in result and "primes" in result["result"]:
                primes = [p["text"] for p in result["result"]["primes"]]
            elif "detected_primes" in result:
                primes = result["detected_primes"]
            print(f"   ✅ Prime detection working: {primes}")
        else:
            print(f"   ❌ Prime detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Prime detection error: {e}")
    
    # Test 3: MWE detection
    print("\n3. Testing MWE detection...")
    try:
        response = requests.post(
            f"{base_url}/mwe",
            json={
                "text": "At least half of the students read a lot of books",
                "language": "en"
            },
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            mwes = []
            if "result" in result and "mwes" in result["result"]:
                mwes = [m["text"] for m in result["result"]["mwes"]]
            elif "detected_mwes" in result:
                mwes = result["detected_mwes"]
            elif "mwes" in result:
                mwes = [m["text"] if isinstance(m, dict) else m for m in result["mwes"]]
            print(f"   ✅ MWE detection working: {mwes}")
        else:
            print(f"   ❌ MWE detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ MWE detection error: {e}")
    
    # Test 4: Text generation
    print("\n4. Testing text generation...")
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "primes": ["PEOPLE", "THINK", "GOOD"],
                "target_language": "en"
            },
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            generated_text = ""
            if "result" in result:
                generated_text = result["result"]["generated_text"]
            elif "generated_text" in result:
                generated_text = result["generated_text"]
            print(f"   ✅ Text generation working: '{generated_text}'")
        else:
            print(f"   ❌ Text generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Text generation error: {e}")
    
    # Test 5: Demo UI
    print("\n5. Testing demo UI...")
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Demo UI working")
        else:
            print(f"   ❌ Demo UI failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Demo UI error: {e}")
    
    # Test 6: Showcase demo
    print("\n6. Testing showcase demo...")
    try:
        response = requests.get("http://localhost:8080/showcase", timeout=5)
        if response.status_code == 200:
            print("   ✅ Showcase demo working")
        else:
            print(f"   ❌ Showcase demo failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Showcase demo error: {e}")

def test_showcase_demo():
    """Test the showcase demo functionality."""
    print("\n🎯 Testing Showcase Demo...")
    print("=" * 50)
    
    try:
        # Run the showcase demo
        import subprocess
        result = subprocess.run(
            ["python", "demo/showcase_demo.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Showcase demo completed successfully")
            # Extract key information from output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "Detected Primes:" in line or "Detected MWEs:" in line or "Generated Explication:" in line:
                    print(f"   {line.strip()}")
        else:
            print("❌ Showcase demo failed")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Showcase demo error: {e}")

def main():
    """Run all tests."""
    print("🚀 NSM Research Platform - Demo Test Suite")
    print("=" * 60)
    
    test_api_endpoints()
    test_showcase_demo()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print("   • API endpoints should be working")
    print("   • Demo UI should be accessible")
    print("   • Showcase demo should run successfully")
    print("\n🌐 Access URLs:")
    print("   • Demo UI: http://localhost:8080")
    print("   • Showcase: http://localhost:8080/showcase")
    print("   • Enhanced: http://localhost:8080/enhanced")
    print("   • Research: http://localhost:8080/research")

if __name__ == "__main__":
    main()

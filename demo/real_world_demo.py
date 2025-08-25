#!/usr/bin/env python3
"""
Real-World Demo for NSM Universal Translator

This script demonstrates the real-world capabilities of our
universal translator stack with actual examples.
"""

import requests
import json
import time
from typing import Dict, Any

def test_endpoint(endpoint: str, data: Dict[str, Any], description: str):
    """Test an API endpoint and display results."""
    print(f"\n{'='*60}")
    print(f"🌍 {description}")
    print(f"{'='*60}")
    
    print(f"📤 Input:")
    print(json.dumps(data, indent=2))
    
    try:
        start_time = time.time()
        response = requests.post(
            f"http://localhost:8001{endpoint}",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        processing_time = time.time() - start_time
        
        print(f"\n⏱️  Processing time: {processing_time:.3f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success (HTTP {response.status_code})")
            print(f"📥 Output:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"❌ Failed (HTTP {response.status_code})")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def main():
    """Run real-world demos."""
    print("🚀 NSM Universal Translator - Real-World Demo")
    print("Complete Universal Translator + Reasoning Stack")
    
    # Test 1: Enhanced Detection with MWE
    print("\n" + "="*80)
    print("🔍 TEST 1: Enhanced Detection with Multi-Word Expressions")
    print("="*80)
    
    test_cases = [
        {
            "text": "At most half of the students read a lot of books",
            "language": "en",
            "description": "English quantifiers with MWE detection"
        },
        {
            "text": "La gente piensa que esto es muy bueno",
            "language": "es",
            "description": "Spanish mental predicates"
        },
        {
            "text": "Les gens pensent que c'est très bon",
            "language": "fr",
            "description": "French evaluators"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📝 {test_case['description']}")
        print(f"Input: \"{test_case['text']}\"")
        
        success = test_endpoint(
            "/detect",
            {
                "text": test_case["text"],
                "language": test_case["language"],
                "methods": ["spacy", "structured", "multilingual", "mwe"],
                "include_deepnsm": True,
                "include_mdl": True,
                "include_temporal": True
            },
            f"Enhanced Detection - {test_case['language'].upper()}"
        )
        
        if success:
            print(f"✅ {test_case['description']} - SUCCESS")
        else:
            print(f"❌ {test_case['description']} - FAILED")
    
    # Test 2: DeepNSM Explication Generation
    print("\n" + "="*80)
    print("🧠 TEST 2: DeepNSM Explication Generation")
    print("="*80)
    
    explication_texts = [
        "I think you know the truth about this situation",
        "Most people want to feel good about their decisions",
        "All students read many books in the library"
    ]
    
    for text in explication_texts:
        print(f"\n📝 Explication for: \"{text}\"")
        
        success = test_endpoint(
            "/deepnsm",
            {
                "text": text,
                "language": "en",
                "include_validation": True
            },
            f"DeepNSM Explication - \"{text[:30]}...\""
        )
        
        if success:
            print(f"✅ DeepNSM explication - SUCCESS")
        else:
            print(f"❌ DeepNSM explication - FAILED")
    
    # Test 3: MDL Compression Validation
    print("\n" + "="*80)
    print("📊 TEST 3: MDL Compression Validation")
    print("="*80)
    
    mdl_texts = [
        "Simple sentence with basic structure",
        "Complex sentence with multiple clauses and quantifiers",
        "Very complex sentence with nested structures and multiple semantic elements"
    ]
    
    for text in mdl_texts:
        print(f"\n📝 MDL validation for: \"{text}\"")
        
        success = test_endpoint(
            "/mdl",
            {
                "text": text,
                "include_analysis": True
            },
            f"MDL Validation - \"{text[:30]}...\""
        )
        
        if success:
            print(f"✅ MDL validation - SUCCESS")
        else:
            print(f"❌ MDL validation - FAILED")
    
    # Test 4: Temporal Reasoning
    print("\n" + "="*80)
    print("⏰ TEST 4: Temporal Reasoning with ESN")
    print("="*80)
    
    temporal_texts = [
        "Yesterday I went to the store",
        "Today I am working on the project",
        "Tomorrow I will finish the task"
    ]
    
    for text in temporal_texts:
        print(f"\n📝 Temporal reasoning for: \"{text}\"")
        
        success = test_endpoint(
            "/temporal",
            {
                "text": text,
                "include_state": True
            },
            f"Temporal Reasoning - \"{text[:30]}...\""
        )
        
        if success:
            print(f"✅ Temporal reasoning - SUCCESS")
        else:
            print(f"❌ Temporal reasoning - FAILED")
    
    # Test 5: System Health and Statistics
    print("\n" + "="*80)
    print("🏥 TEST 5: System Health and Statistics")
    print("="*80)
    
    try:
        # Health check
        response = requests.get("http://localhost:8001/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ System Health:")
            print(json.dumps(health, indent=2))
        else:
            print(f"❌ Health check failed: {response.status_code}")
        
        # Statistics
        response = requests.get("http://localhost:8001/stats")
        if response.status_code == 200:
            stats = response.json()
            print("\n✅ System Statistics:")
            print(json.dumps(stats, indent=2))
        else:
            print(f"❌ Stats failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ System check failed: {str(e)}")
    
    print("\n" + "="*80)
    print("🎯 REAL-WORLD DEMO COMPLETE")
    print("="*80)
    print("This demonstrates our complete universal translator stack with:")
    print("✅ Enhanced detection with MWE support")
    print("✅ DeepNSM explication generation")
    print("✅ MDL compression validation")
    print("✅ Temporal reasoning with ESN")
    print("✅ Cross-language support (EN/ES/FR)")
    print("✅ Production monitoring and health checks")
    print("✅ Real-world performance and reliability")

if __name__ == "__main__":
    main()

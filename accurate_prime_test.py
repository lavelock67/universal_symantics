#!/usr/bin/env python3
"""
Accurate test of our actual prime detection capabilities.
"""

import requests
import json
from collections import defaultdict

# Corrected list - TOUCH appears in both Actions and Space, so we have 66 total
ALL_PRIMES = {
    # Phase 1: Substantives
    "I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING", "BODY",
    # Phase 2: Relational substantives
    "KIND", "PART",
    # Phase 3: Determiners and quantifiers
    "THIS", "THE_SAME", "OTHER", "ONE", "TWO", "SOME", "ALL", "MUCH", "MANY",
    # Phase 4: Evaluators and descriptors
    "GOOD", "BAD", "BIG", "SMALL",
    # Phase 5: Mental predicates
    "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR",
    # Phase 6: Speech
    "SAY", "WORDS", "TRUE", "FALSE",
    # Phase 7: Actions and events
    "DO", "HAPPEN", "MOVE", "TOUCH",
    # Phase 8: Location, existence, possession, specification
    "BE_SOMEWHERE", "THERE_IS", "HAVE", "BE_SOMEONE",
    # Phase 9: Life and death
    "LIVE", "DIE",
    # Phase 10: Time
    "WHEN", "NOW", "BEFORE", "AFTER", "A_LONG_TIME", "A_SHORT_TIME", "FOR_SOME_TIME", "MOMENT",
    # Phase 11: Space
    "WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR", "SIDE", "INSIDE", "TOUCH",
    # Logical concepts
    "NOT", "MAYBE", "CAN", "BECAUSE", "IF",
    # Intensifier and augmentor
    "VERY", "MORE",
    # Similarity
    "LIKE"
}

def test_actual_detection():
    """Test what we're actually detecting based on the logs."""
    print("🎯 ACCURATE PRIME DETECTION TEST")
    print("=" * 60)
    
    # Based on the logs, let's test what we're actually detecting
    test_sentences = [
        # From the logs - this detected 19 primes including ABILITY
        "I think this is very good because people know that all things are true when they can see and hear what happens in the world",
        
        # From the logs - this detected many spatial and temporal primes
        "Where are you going? Look above and below the line. The side of the mountain is steep. It took a long time to finish the work. The meeting lasted for some time. It was a short time before he arrived.",
        
        # From the logs - this detected existence and location primes
        "There is someone in the room. The person is somewhere in the building. Be someone who helps others.",
        
        # From the logs - this detected quantifiers and substantives
        "Many people have much money. This kind of thing is different from that kind. Part of the problem is that we don't know.",
        
        # From the logs - this detected similarity and comparison
        "This thing is like that thing. The same person came again.",
        
        # From the logs - this detected additional UD primes
        "You have the ability to do this. There is an obligation to help others. The work is finished now.",
        
        # Additional comprehensive tests
        "I want to do something big or small. I feel I should not be here. Maybe you can go far away.",
        "Someone said some things are true and some are false. I know this for some time now.",
        "My body can touch things in this place. I can see what is near and inside.",
        "If you live and then die, you have been somewhere after now.",
        "Two people think the same thing. One wants more than the other.",
        "When the moment comes, you will see all kinds of things in this world.",
        "I can say very big and very small things.",
        "You think some things are good and some things are bad. I know this.",
        "Before now and after this moment, people will hear and see.",
    ]
    
    all_detected_primes = set()
    detection_stats = defaultdict(int)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Test {i}: '{sentence[:80]}{'...' if len(sentence) > 80 else ''}'")
        
        try:
            response = requests.post("http://localhost:8001/detect", 
                                   headers={"Content-Type": "application/json"},
                                   json={"text": sentence, "language": "en"})
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    primes = [p["text"] for p in result["result"]["primes"]]
                    all_detected_primes.update(primes)
                    detection_stats["total_detections"] += len(primes)
                    
                    print(f"   ✅ Detected: {primes}")
                    print(f"   📊 Count: {len(primes)} primes")
                    print(f"   🎯 Confidence: {(result['result']['confidence'] * 100):.1f}%")
                    
                    # Check for standard primes
                    standard_primes = [p for p in primes if p in ALL_PRIMES]
                    if standard_primes:
                        print(f"   🎯 Standard primes: {standard_primes}")
                    
                    # Check for additional primes
                    additional_primes = [p for p in primes if p not in ALL_PRIMES]
                    if additional_primes:
                        print(f"   🔍 Additional primes: {additional_primes}")
                        
                else:
                    print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    # Analysis
    print(f"\n📊 ACCURATE ANALYSIS")
    print("=" * 60)
    
    # Standard prime analysis
    standard_detected = all_detected_primes & ALL_PRIMES
    standard_missing = ALL_PRIMES - all_detected_primes
    
    print(f"Standard NSM Primes:")
    print(f"   • Total: {len(ALL_PRIMES)} (corrected from 65 to 66)")
    print(f"   • Detected: {len(standard_detected)}")
    print(f"   • Missing: {len(standard_missing)}")
    print(f"   • Coverage: {(len(standard_detected) / len(ALL_PRIMES) * 100):.1f}%")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"   • Total unique primes detected: {len(all_detected_primes)}")
    print(f"   • Total prime detections: {detection_stats['total_detections']}")
    print(f"   • Average per sentence: {detection_stats['total_detections'] / len(test_sentences):.1f}")
    
    # Detailed breakdown
    print(f"\n✅ DETECTED PRIMES ({len(all_detected_primes)}):")
    for prime in sorted(all_detected_primes):
        category = "STANDARD" if prime in ALL_PRIMES else "ADDITIONAL"
        print(f"   • {prime} ({category})")
    
    if standard_missing:
        print(f"\n❌ MISSING STANDARD PRIMES ({len(standard_missing)}):")
        for prime in sorted(standard_missing):
            print(f"   • {prime}")
    
    print(f"\n🎉 FINAL ASSESSMENT:")
    if len(standard_detected) >= 50:
        print("   🏆 EXCELLENT: High coverage of standard NSM primes!")
    elif len(standard_detected) >= 40:
        print("   ✅ GOOD: Solid coverage of standard NSM primes!")
    else:
        print("   ⚠️ NEEDS WORK: Coverage could be improved.")
    
    print(f"   🔍 Additional semantic concepts detected: {len(all_detected_primes - ALL_PRIMES)}")
    print(f"   🎯 Total semantic coverage: {len(all_detected_primes)} unique concepts")

if __name__ == "__main__":
    test_actual_detection()


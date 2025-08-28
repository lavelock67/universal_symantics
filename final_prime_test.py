#!/usr/bin/env python3
"""
Final comprehensive test of the complete prime detection system.
"""

import requests
import json
from collections import defaultdict

# All 65 standard NSM primes plus additional UD primes
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

# Additional UD primes we've discovered
ADDITIONAL_UD_PRIMES = {
    "ABILITY", "OBLIGATION", "THERE_IS", "THE_SAME", "AGAIN", "FINISH"
}

def test_comprehensive_detection():
    """Test comprehensive prime detection with multiple approaches."""
    print("🎯 FINAL COMPREHENSIVE PRIME DETECTION TEST")
    print("=" * 60)
    
    # Test sentences designed to catch all primes
    test_sentences = [
        # Core sentence with many primes
        "I think this is very good because people know that all things are true when they can see and hear what happens in the world",
        
        # Spatial and temporal primes
        "Where are you going? Look above and below the line. The side of the mountain is steep. It took a long time to finish the work. The meeting lasted for some time. It was a short time before he arrived.",
        
        # Existence and location
        "There is someone in the room. The person is somewhere in the building. Be someone who helps others.",
        
        # Quantifiers and substantives
        "Many people have much money. This kind of thing is different from that kind. Part of the problem is that we don't know.",
        
        # Similarity and comparison
        "This thing is like that thing. The same person came again.",
        
        # Additional UD primes
        "You have the ability to do this. There is an obligation to help others. The work is finished now.",
    ]
    
    all_detected_primes = set()
    detection_stats = defaultdict(int)
    mwe_detections = []
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Test {i}: '{sentence[:80]}{'...' if len(sentence) > 80 else ''}'")
        
        try:
            # Test main detection
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
                    
                    # Check for additional UD primes
                    ud_primes = [p for p in primes if p in ADDITIONAL_UD_PRIMES]
                    if ud_primes:
                        print(f"   🔍 Additional UD primes: {ud_primes}")
                    
                    # Check for MWE primes
                    if result["result"].get("mwes"):
                        mwe_primes = []
                        for mwe in result["result"]["mwes"]:
                            mwe_primes.extend(mwe.get("primes", []))
                        if mwe_primes:
                            print(f"   🔗 MWE primes: {mwe_primes}")
                            mwe_detections.extend(mwe_primes)
                else:
                    print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    # Analysis
    print(f"\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # Standard prime analysis
    standard_detected = all_detected_primes & ALL_PRIMES
    standard_missing = ALL_PRIMES - all_detected_primes
    
    print(f"Standard NSM Primes:")
    print(f"   • Total: {len(ALL_PRIMES)}")
    print(f"   • Detected: {len(standard_detected)}")
    print(f"   • Missing: {len(standard_missing)}")
    print(f"   • Coverage: {(len(standard_detected) / len(ALL_PRIMES) * 100):.1f}%")
    
    # Additional UD prime analysis
    ud_detected = all_detected_primes & ADDITIONAL_UD_PRIMES
    print(f"\nAdditional UD Primes:")
    print(f"   • Total: {len(ADDITIONAL_UD_PRIMES)}")
    print(f"   • Detected: {len(ud_detected)}")
    print(f"   • Coverage: {(len(ud_detected) / len(ADDITIONAL_UD_PRIMES) * 100):.1f}%")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"   • Total unique primes detected: {len(all_detected_primes)}")
    print(f"   • Total prime detections: {detection_stats['total_detections']}")
    print(f"   • Average per sentence: {detection_stats['total_detections'] / len(test_sentences):.1f}")
    print(f"   • MWE primes detected: {len(set(mwe_detections))}")
    
    # Detailed breakdown
    print(f"\n✅ DETECTED PRIMES ({len(all_detected_primes)}):")
    for prime in sorted(all_detected_primes):
        category = "STANDARD" if prime in ALL_PRIMES else "ADDITIONAL"
        print(f"   • {prime} ({category})")
    
    if standard_missing:
        print(f"\n❌ MISSING STANDARD PRIMES ({len(standard_missing)}):")
        for prime in sorted(standard_missing):
            print(f"   • {prime}")
    
    # Coverage by category
    print(f"\n🎯 COVERAGE BY CATEGORY:")
    categories = {
        "Substantives": {"I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING", "BODY", "KIND", "PART"},
        "Quantifiers": {"THIS", "THE_SAME", "OTHER", "ONE", "TWO", "SOME", "ALL", "MUCH", "MANY"},
        "Evaluators": {"GOOD", "BAD", "BIG", "SMALL"},
        "Mental Predicates": {"THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR"},
        "Speech": {"SAY", "WORDS", "TRUE", "FALSE"},
        "Actions": {"DO", "HAPPEN", "MOVE", "TOUCH"},
        "Time": {"WHEN", "NOW", "BEFORE", "AFTER", "A_LONG_TIME", "A_SHORT_TIME", "FOR_SOME_TIME", "MOMENT"},
        "Space": {"WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR", "SIDE", "INSIDE"},
        "Logical": {"NOT", "MAYBE", "CAN", "BECAUSE", "IF"},
        "Intensifiers": {"VERY", "MORE", "LIKE"}
    }
    
    for category, primes in categories.items():
        detected = len(primes & all_detected_primes)
        total = len(primes)
        percentage = (detected / total * 100) if total > 0 else 0
        print(f"   • {category}: {detected}/{total} ({percentage:.1f}%)")
    
    print(f"\n🎉 FINAL ASSESSMENT:")
    if len(standard_detected) >= 60:
        print("   🏆 EXCELLENT: High coverage of standard NSM primes!")
    elif len(standard_detected) >= 50:
        print("   ✅ GOOD: Solid coverage of standard NSM primes!")
    else:
        print("   ⚠️ NEEDS WORK: Coverage could be improved.")
    
    print(f"   🔍 Additional semantic concepts detected: {len(ud_detected)}")
    print(f"   🎯 Total semantic coverage: {len(all_detected_primes)} unique concepts")

if __name__ == "__main__":
    test_comprehensive_detection()


#!/usr/bin/env python3
"""
Final test using the correct 65 NSM primes - each prime in exactly one category.
"""

import requests
import json
from collections import defaultdict

# The correct 65 NSM primes, each in exactly one category
CORRECT_NSM_PRIMES = {
    # Phase 1: Substantives (7 primes)
    "I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING", "BODY",
    
    # Phase 2: Relational substantives (2 primes)
    "KIND", "PART",
    
    # Phase 3: Determiners and quantifiers (9 primes)
    "THIS", "THE_SAME", "OTHER", "ONE", "TWO", "SOME", "ALL", "MUCH", "MANY",
    
    # Phase 4: Evaluators and descriptors (4 primes)
    "GOOD", "BAD", "BIG", "SMALL",
    
    # Phase 5: Mental predicates (6 primes)
    "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR",
    
    # Phase 6: Speech (4 primes)
    "SAY", "WORDS", "TRUE", "FALSE",
    
    # Phase 7: Actions and events (4 primes)
    "DO", "HAPPEN", "MOVE", "TOUCH",
    
    # Phase 8: Location, existence, possession, specification (4 primes)
    "BE_SOMEWHERE", "THERE_IS", "HAVE", "BE_SOMEONE",
    
    # Phase 9: Life and death (2 primes)
    "LIVE", "DIE",
    
    # Phase 10: Time (8 primes)
    "WHEN", "NOW", "BEFORE", "AFTER", "A_LONG_TIME", "A_SHORT_TIME", "FOR_SOME_TIME", "MOMENT",
    
    # Phase 11: Space (7 primes)
    "WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR", "INSIDE",
    
    # Logical concepts (5 primes)
    "NOT", "MAYBE", "CAN", "BECAUSE", "IF",
    
    # Intensifier and augmentor (3 primes)
    "VERY", "MORE", "LIKE"
}

def test_corrected_detection():
    """Test detection using the correct 65 NSM primes."""
    print("🎯 CORRECTED NSM PRIME DETECTION TEST")
    print("=" * 60)
    print(f"Testing against exactly {len(CORRECT_NSM_PRIMES)} NSM primes")
    print("=" * 60)
    
    # Comprehensive test sentences designed to catch all primes
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
        
        # Specific tests for missing primes
        "Look near the house and find one word.",
        "One person said many words near the tree.",
    ]
    
    all_detected_primes = set()
    detection_stats = defaultdict(int)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Test {i}: '{sentence[:80]}{'...' if len(sentence) > 80 else ''}'")
        
        try:
            response = requests.post("http://localhost:8000/detect", 
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
                    standard_primes = [p for p in primes if p in CORRECT_NSM_PRIMES]
                    if standard_primes:
                        print(f"   🎯 Standard primes: {standard_primes}")
                    
                    # Check for additional primes
                    additional_primes = [p for p in primes if p not in CORRECT_NSM_PRIMES]
                    if additional_primes:
                        print(f"   🔍 Additional primes: {additional_primes}")
                        
                else:
                    print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    # Analysis
    print(f"\n📊 CORRECTED ANALYSIS")
    print("=" * 60)
    
    # Standard prime analysis
    standard_detected = all_detected_primes & CORRECT_NSM_PRIMES
    standard_missing = CORRECT_NSM_PRIMES - all_detected_primes
    
    print(f"Standard NSM Primes:")
    print(f"   • Total: {len(CORRECT_NSM_PRIMES)} (corrected)")
    print(f"   • Detected: {len(standard_detected)}")
    print(f"   • Missing: {len(standard_missing)}")
    print(f"   • Coverage: {(len(standard_detected) / len(CORRECT_NSM_PRIMES) * 100):.1f}%")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"   • Total unique primes detected: {len(all_detected_primes)}")
    print(f"   • Total prime detections: {detection_stats['total_detections']}")
    print(f"   • Average per sentence: {detection_stats['total_detections'] / len(test_sentences):.1f}")
    
    # Detailed breakdown
    print(f"\n✅ DETECTED STANDARD PRIMES ({len(standard_detected)}):")
    for prime in sorted(standard_detected):
        print(f"   • {prime}")
    
    if standard_missing:
        print(f"\n❌ MISSING STANDARD PRIMES ({len(standard_missing)}):")
        for prime in sorted(standard_missing):
            print(f"   • {prime}")
    
    # Coverage by category
    print(f"\n🎯 COVERAGE BY CATEGORY:")
    categories = {
        "Substantives": {"I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING", "BODY"},
        "Relational Substantives": {"KIND", "PART"},
        "Determiners and Quantifiers": {"THIS", "THE_SAME", "OTHER", "ONE", "TWO", "SOME", "ALL", "MUCH", "MANY"},
        "Evaluators and Descriptors": {"GOOD", "BAD", "BIG", "SMALL"},
        "Mental Predicates": {"THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR"},
        "Speech": {"SAY", "WORDS", "TRUE", "FALSE"},
        "Actions and Events": {"DO", "HAPPEN", "MOVE", "TOUCH"},
        "Location, Existence, Possession": {"BE_SOMEWHERE", "THERE_IS", "HAVE", "BE_SOMEONE"},
        "Life and Death": {"LIVE", "DIE"},
        "Time": {"WHEN", "NOW", "BEFORE", "AFTER", "A_LONG_TIME", "A_SHORT_TIME", "FOR_SOME_TIME", "MOMENT"},
        "Space": {"WHERE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR", "INSIDE"},
        "Logical Concepts": {"NOT", "MAYBE", "CAN", "BECAUSE", "IF"},
        "Intensifiers": {"VERY", "MORE", "LIKE"}
    }
    
    for category, primes in categories.items():
        detected = len(primes & all_detected_primes)
        total = len(primes)
        percentage = (detected / total * 100) if total > 0 else 0
        print(f"   • {category}: {detected}/{total} ({percentage:.1f}%)")
    
    print(f"\n🎉 FINAL ASSESSMENT:")
    if len(standard_detected) >= 50:
        print("   🏆 EXCELLENT: High coverage of standard NSM primes!")
    elif len(standard_detected) >= 40:
        print("   ✅ GOOD: Solid coverage of standard NSM primes!")
    else:
        print("   ⚠️ NEEDS WORK: Coverage could be improved.")
    
    print(f"   🔍 Additional semantic concepts detected: {len(all_detected_primes - CORRECT_NSM_PRIMES)}")
    print(f"   🎯 Total semantic coverage: {len(all_detected_primes)} unique concepts")

if __name__ == "__main__":
    test_corrected_detection()


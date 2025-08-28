#!/usr/bin/env python3
"""
Test script to check detection of all 65 NSM primes.
"""

import requests
import json

# All 65 NSM primes from the UD system
ALL_NSM_PRIMES = {
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

def test_prime_detection():
    """Test detection of all 65 NSM primes."""
    print("🧪 TESTING ALL 65 NSM PRIMES")
    print("=" * 60)
    
    # Test sentences designed to include many primes
    test_sentences = [
        "I think this is very good because people know that all things are true when they can see and hear what happens in the world",
        "You want to do something big and small, but I feel that maybe we should not move far from here",
        "Someone said words that are true and false, but I know that everything happens for some time",
        "The body can touch things above and below, inside and outside, near and far from this place",
        "If you live and die, then you have been somewhere before and after now",
        "Two people think the same thing, but one person wants more than the other",
        "When the moment comes, you will see that all kinds of things happen in this world",
        "I can say that very big and very small things exist everywhere in the universe",
        "Maybe you think that some things are good and some things are bad, but I know the truth",
        "Before now and after this moment, people will hear and see what happens in life"
    ]
    
    detected_primes = set()
    total_detections = 0
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Test {i}: '{sentence}'")
        
        try:
            response = requests.post("http://localhost:8001/detect", 
                                   headers={"Content-Type": "application/json"},
                                   json={"text": sentence, "language": "en"})
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    primes = [p["text"] for p in result["result"]["primes"]]
                    detected_primes.update(primes)
                    total_detections += len(primes)
                    
                    print(f"   ✅ Detected: {primes}")
                    print(f"   📊 Count: {len(primes)} primes")
                    print(f"   🎯 Confidence: {(result['result']['confidence'] * 100):.1f}%")
                else:
                    print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    # Analysis
    print(f"\n📊 DETECTION ANALYSIS")
    print("=" * 60)
    print(f"Total unique primes detected: {len(detected_primes)}")
    print(f"Total prime detections: {total_detections}")
    print(f"Coverage of 65 NSM primes: {(len(detected_primes) / 65 * 100):.1f}%")
    
    print(f"\n✅ DETECTED PRIMES ({len(detected_primes)}):")
    for prime in sorted(detected_primes):
        print(f"   • {prime}")
    
    print(f"\n❌ MISSING PRIMES ({65 - len(detected_primes)}):")
    missing = ALL_NSM_PRIMES - detected_primes
    for prime in sorted(missing):
        print(f"   • {prime}")
    
    print(f"\n🎯 COVERAGE BREAKDOWN:")
    print(f"   • Substantives: {len([p for p in detected_primes if p in ['I', 'YOU', 'SOMEONE', 'PEOPLE', 'SOMETHING', 'THING', 'BODY', 'KIND', 'PART']])}/9")
    print(f"   • Quantifiers: {len([p for p in detected_primes if p in ['THIS', 'THE_SAME', 'OTHER', 'ONE', 'TWO', 'SOME', 'ALL', 'MUCH', 'MANY']])}/9")
    print(f"   • Evaluators: {len([p for p in detected_primes if p in ['GOOD', 'BAD', 'BIG', 'SMALL']])}/4")
    print(f"   • Mental Predicates: {len([p for p in detected_primes if p in ['THINK', 'KNOW', 'WANT', 'FEEL', 'SEE', 'HEAR']])}/6")
    print(f"   • Speech: {len([p for p in detected_primes if p in ['SAY', 'WORDS', 'TRUE', 'FALSE']])}/4")
    print(f"   • Actions: {len([p for p in detected_primes if p in ['DO', 'HAPPEN', 'MOVE', 'TOUCH']])}/4")
    print(f"   • Time: {len([p for p in detected_primes if p in ['WHEN', 'NOW', 'BEFORE', 'AFTER', 'A_LONG_TIME', 'A_SHORT_TIME', 'FOR_SOME_TIME', 'MOMENT']])}/8")
    print(f"   • Space: {len([p for p in detected_primes if p in ['WHERE', 'HERE', 'ABOVE', 'BELOW', 'FAR', 'NEAR', 'SIDE', 'INSIDE']])}/8")
    print(f"   • Logical: {len([p for p in detected_primes if p in ['NOT', 'MAYBE', 'CAN', 'BECAUSE', 'IF']])}/5")

if __name__ == "__main__":
    test_prime_detection()


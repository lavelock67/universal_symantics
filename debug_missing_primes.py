#!/usr/bin/env python3
"""
Debug Missing Primes

Test specific missing primes and add patterns for them.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_missing_primes():
    """Test specific missing primes."""
    print("Testing Missing Primes...")
    
    # Missing primes that need patterns
    missing_primes = {
        # Evaluators and descriptors (0/4)
        "GOOD": ["good", "great", "excellent", "wonderful"],
        "BAD": ["bad", "terrible", "awful", "horrible"],
        "BIG": ["big", "large", "huge", "enormous"],
        "SMALL": ["small", "tiny", "little", "mini"],
        
        # Mental predicates (3/6)
        "THINK": ["think", "thought", "thinking"],
        "WANT": ["want", "wish", "desire"],
        "FEEL": ["feel", "feeling", "felt"],
        
        # Space (0/7)
        "ABOVE": ["above", "over", "up"],
        "BELOW": ["below", "under", "down"],
        "FAR": ["far", "distant", "away"],
        "NEAR": ["near", "close", "nearby"],
        "INSIDE": ["inside", "in", "within"],
        "HERE": ["here", "this place"],
        "WHERE": ["where", "location"],
        
        # Speech (0/4)
        "SAY": ["say", "said", "saying"],
        "TRUE": ["true", "truth", "correct"],
        "FALSE": ["false", "wrong", "incorrect"],
        "WORDS": ["words", "word"],
        
        # Actions and events (1/4)
        "DO": ["do", "does", "did", "doing"],
        "HAPPEN": ["happen", "happens", "happened"],
        "MOVE": ["move", "moves", "moved", "moving"],
        
        # Determiners and quantifiers (4/9)
        "SOME": ["some", "several", "few"],
        "ONE": ["one", "single"],
        "OTHER": ["other", "another"],
        "THE_SAME": ["same", "identical"],
        
        # Location, existence, possession (2/4)
        "THERE_IS": ["there is", "there are", "exists"],
        "BE_SOMEONE": ["be someone", "be a person"],
        
        # Time (5/8)
        "A_LONG_TIME": ["long time", "a long time", "forever"],
        "A_SHORT_TIME": ["short time", "a short time", "briefly"],
        "FOR_SOME_TIME": ["for some time", "for a while"],
        
        # Logical concepts (3/5)
        "BECAUSE": ["because", "since", "as"],
        "MAYBE": ["maybe", "perhaps", "possibly"],
        
        # Intensifiers (0/3)
        "VERY": ["very", "really", "extremely"],
        "MORE": ["more", "additional", "extra"],
        "LIKE": ["like", "similar", "alike"],
        
        # Substantives (5/7)
        "SOMEONE": ["someone", "somebody", "person"],
        "SOMETHING": ["something", "thing", "object"],
    }
    
    try:
        from src.core.application.services import create_detection_service
        from src.core.domain.models import Language
        
        # Create detection service
        detection_service = create_detection_service()
        print("  ✅ Detection service created")
        
        # Test each missing prime
        detected_count = 0
        total_count = len(missing_primes)
        
        for prime, test_words in missing_primes.items():
            print(f"\n  Testing {prime}:")
            
            # Test each word for this prime
            found = False
            for word in test_words:
                try:
                    result = detection_service.detect_primes(f"I {word} this", Language.ENGLISH)
                    detected_primes = [p.text for p in result.primes]
                    
                    if prime in detected_primes:
                        print(f"    ✅ Found '{prime}' via '{word}'")
                        found = True
                        detected_count += 1
                        break
                    else:
                        print(f"    ❌ '{word}' did not detect '{prime}' (found: {detected_primes})")
                        
                except Exception as e:
                    print(f"    ❌ Error testing '{word}': {e}")
            
            if not found:
                print(f"    ❌ No pattern found for '{prime}'")
        
        print(f"\n📊 Results:")
        print(f"  Detected: {detected_count}/{total_count} missing primes")
        print(f"  Coverage: {detected_count/total_count*100:.1f}%")
        
        return detected_count, total_count
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

def add_missing_patterns():
    """Add missing patterns to the detection service."""
    print("\nAdding Missing Patterns...")
    
    try:
        from src.core.application.services import NSMDetectionService
        from src.core.domain.models import Language
        
        # Create service to access patterns
        service = NSMDetectionService()
        
        # Get current patterns
        current_patterns = service._get_lexical_patterns(Language.ENGLISH)
        print(f"  Current patterns: {len(current_patterns)}")
        
        # Missing patterns to add
        missing_patterns = {
            # Evaluators and descriptors
            "GOOD": {"lemma": "good"},
            "BAD": {"lemma": "bad"},
            "BIG": {"lemma": "big"},
            "SMALL": {"lemma": "small"},
            
            # Mental predicates
            "THINK": {"lemma": "think"},
            "WANT": {"lemma": "want"},
            "FEEL": {"lemma": "feel"},
            
            # Space
            "ABOVE": {"lemma": "above"},
            "BELOW": {"lemma": "below"},
            "FAR": {"lemma": "far"},
            "NEAR": {"lemma": "near"},
            "INSIDE": {"lemma": "inside"},
            "HERE": {"lemma": "here"},
            "WHERE": {"lemma": "where"},
            
            # Speech
            "SAY": {"lemma": "say"},
            "TRUE": {"lemma": "true"},
            "FALSE": {"lemma": "false"},
            "WORDS": {"lemma": "word"},
            
            # Actions and events
            "DO": {"lemma": "do"},
            "HAPPEN": {"lemma": "happen"},
            "MOVE": {"lemma": "move"},
            
            # Determiners and quantifiers
            "SOME": {"lemma": "some"},
            "ONE": {"lemma": "one"},
            "OTHER": {"lemma": "other"},
            "THE_SAME": {"lemma": "same"},
            
            # Location, existence, possession
            "THERE_IS": {"lemma": "there"},
            "BE_SOMEONE": {"lemma": "be"},
            
            # Time
            "A_LONG_TIME": {"lemma": "long"},
            "A_SHORT_TIME": {"lemma": "short"},
            "FOR_SOME_TIME": {"lemma": "for"},
            
            # Logical concepts
            "BECAUSE": {"lemma": "because"},
            "MAYBE": {"lemma": "maybe"},
            
            # Intensifiers
            "VERY": {"lemma": "very"},
            "MORE": {"lemma": "more"},
            "LIKE": {"lemma": "like"},
            
            # Substantives
            "SOMEONE": {"lemma": "someone"},
            "SOMETHING": {"lemma": "something"},
        }
        
        # Add patterns
        for prime, pattern in missing_patterns.items():
            if prime not in current_patterns:
                current_patterns[prime] = pattern
                print(f"  ✅ Added pattern for {prime}")
            else:
                print(f"  ⚠️ Pattern already exists for {prime}")
        
        print(f"  Total patterns after addition: {len(current_patterns)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to add patterns: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("DEBUG MISSING PRIMES")
    print("=" * 60)
    
    # Test current missing primes
    detected, total = test_missing_primes()
    
    # Add missing patterns
    if add_missing_patterns():
        print("\n" + "=" * 60)
        print("✅ PATTERNS ADDED - RESTART SERVER TO APPLY CHANGES")
        print("=" * 60)
    
    print(f"\n🎯 SUMMARY:")
    print(f"  Missing primes identified: {total}")
    print(f"  Already detected: {detected}")
    print(f"  Need patterns: {total - detected}")

if __name__ == "__main__":
    main()

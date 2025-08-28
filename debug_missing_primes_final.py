#!/usr/bin/env python3
"""
Debug the specific missing primes
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_missing_primes():
    """Test the specific missing primes"""
    try:
        print("🔍 Testing specific missing primes...")
        
        from src.core.application.services import NSMDetectionService
        from src.core.domain.models import Language
        import spacy
        
        # Create service
        service = NSMDetectionService()
        print("✅ Service created")
        
        # Test the missing primes
        missing_primes = ["above", "inside", "near", "one", "word"]
        
        for prime in missing_primes:
            print(f"\n📝 Testing '{prime}'...")
            
            # Test SpaCy tokenization
            spacy_model = service.spacy_models.get(Language.ENGLISH.value)
            doc = spacy_model(prime)
            token = doc[0]
            print(f"  Token: '{token.text}' -> lemma: '{token.lemma_}' -> pos: '{token.pos_}'")
            
            # Test lexical patterns
            patterns = service._get_lexical_patterns(Language.ENGLISH)
            prime_upper = prime.upper()
            # Handle special case for "word" -> "WORDS"
            if prime_upper == "WORD":
                prime_upper = "WORDS"
            pattern = patterns.get(prime_upper)
            print(f"  Pattern for {prime_upper}: {pattern}")
            
            if pattern:
                matches = service._matches_pattern(token, pattern)
                print(f"  Pattern matches: {matches}")
            
            # Test full detection
            result = service.detect_primes(prime, Language.ENGLISH)
            detected = [p.text for p in result.primes]
            print(f"  Detection result: {detected}")
            
            # Handle special case for "word" -> "WORDS"
            expected_prime = prime_upper
            if prime_upper == "WORD":
                expected_prime = "WORDS"
            
            if expected_prime in detected:
                print(f"  ✅ {expected_prime} detected!")
            else:
                print(f"  ❌ {expected_prime} NOT detected!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_missing_primes()
    if success:
        print("\n🎉 Test completed!")
    else:
        print("\n💥 Test failed!")

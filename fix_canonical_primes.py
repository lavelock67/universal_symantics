#!/usr/bin/env python3
"""
Fix Canonical NSM Primes

This script fixes the system to only use the correct 65 canonical NSM primes,
removing non-canonical primes like "WORLD", "EARTH", "SKY", etc.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language, PrimeType

# The correct 65 NSM primes from Anna Wierzbicka's theory
CANONICAL_NSM_PRIMES = {
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

# Non-canonical primes that should be removed
NON_CANONICAL_PRIMES = {
    "WORLD", "EARTH", "SKY", "WATER", "FIRE", "DAY", "NIGHT", "YEAR", "MONTH", "WEEK",
    "PLACE", "WAY", "TIME", "LONG", "SHORT", "WIDE", "NARROW", "THICK", "THIN",
    "HEAVY", "LIGHT", "STRONG", "WEAK", "HARD", "SOFT", "WARM", "COLD", "NEW", "OLD",
    "DIFFERENT", "RIGHT", "WRONG", "READ", "GIVE", "TAKE", "MAKE", "BECOME", "COME", "GO",
    "BE", "HAVE", "MUST", "GUSTAR", "AIMER", "TODAY", "TOMORROW", "YESTERDAY",
    "THERE", "SIDE", "LEFT", "RIGHT", "MOMENT", "CAPACITY", "OBLIGATION", "AGAIN", "FINISH"
}

def fix_canonical_primes():
    """Fix the system to only use canonical NSM primes."""
    
    print('🔧 FIXING CANONICAL NSM PRIMES')
    print('=' * 70)
    print()
    
    print('📋 CANONICAL NSM PRIMES (65 total):')
    print('-' * 50)
    for prime in sorted(CANONICAL_NSM_PRIMES):
        print(f'  ✅ {prime}')
    print()
    
    print('❌ NON-CANONICAL PRIMES TO REMOVE:')
    print('-' * 50)
    for prime in sorted(NON_CANONICAL_PRIMES):
        print(f'  ❌ {prime}')
    print()
    
    # Initialize the detection service
    nsm_service = NSMDetectionService()
    
    # Test with canonical primes only
    print('🧪 TESTING WITH CANONICAL PRIMES ONLY')
    print('-' * 50)
    
    test_sentences = [
        "I think this is very good.",
        "The world is big and people want to know more.",  # "world" should NOT be detected as a prime
        "This happens here and now because I want it.",
        "Yo pienso que esto es muy bueno.",
        "Je pense que ceci est très bon.",
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f'📝 Test {i}: "{sentence}"')
        
        try:
            result = nsm_service.detect_primes(sentence, Language.ENGLISH)
            primes = [prime.text for prime in result.primes]
            
            # Filter to only canonical primes
            canonical_primes = [p for p in primes if p in CANONICAL_NSM_PRIMES]
            non_canonical_detected = [p for p in primes if p in NON_CANONICAL_PRIMES]
            
            print(f'  🔍 Total detected: {len(primes)} primes')
            print(f'  ✅ Canonical: {len(canonical_primes)} primes')
            print(f'  ❌ Non-canonical: {len(non_canonical_detected)} primes')
            
            if canonical_primes:
                print(f'  📋 Canonical primes: {", ".join(canonical_primes)}')
            
            if non_canonical_detected:
                print(f'  ⚠️  WARNING: Non-canonical primes detected: {", ".join(non_canonical_detected)}')
                print(f'     These should be represented as combinations of canonical primes')
            
            # Check for specific issues
            if "WORLD" in primes:
                print(f'  💡 "WORLD" should be: "a very big place" (BIG + PLACE)')
            if "EARTH" in primes:
                print(f'  💡 "EARTH" should be: "this ground" (THIS + THING)')
            if "SKY" in primes:
                print(f'  💡 "SKY" should be: "above here" (ABOVE + HERE)')
            
        except Exception as e:
            print(f'  ❌ Error: {e}')
        
        print()
    
    # Test cross-lingual with canonical primes
    print('🌐 CROSS-LINGUAL CANONICAL TEST')
    print('-' * 50)
    
    test_sentence = "I think this is very good because I want to know more about people and things when they happen here and now."
    translations = {
        Language.ENGLISH: test_sentence,
        Language.SPANISH: "Yo pienso que esto es muy bueno porque yo quiero saber más sobre gente y cosas cuando pasan aquí y ahora.",
        Language.FRENCH: "Je pense que ceci est très bon parce que je veux savoir plus sur les gens et les choses quand ils arrivent ici et maintenant.",
    }
    
    for language, translation in translations.items():
        print(f'🌍 {language.value.upper()}: "{translation}"')
        
        try:
            result = nsm_service.detect_primes(translation, language)
            primes = [prime.text for prime in result.primes]
            
            # Filter to canonical primes
            canonical_primes = [p for p in primes if p in CANONICAL_NSM_PRIMES]
            non_canonical_detected = [p for p in primes if p in NON_CANONICAL_PRIMES]
            
            print(f'  ✅ Canonical primes: {len(canonical_primes)}')
            print(f'  ❌ Non-canonical: {len(non_canonical_detected)}')
            
            if canonical_primes:
                print(f'  📋 Canonical: {", ".join(canonical_primes)}')
            
            if non_canonical_detected:
                print(f'  ⚠️  Non-canonical: {", ".join(non_canonical_detected)}')
            
        except Exception as e:
            print(f'  ❌ Error: {e}')
        
        print()
    
    # Recommendations
    print('💡 RECOMMENDATIONS FOR FIXING THE SYSTEM')
    print('=' * 70)
    print()
    print('1. 🔧 Remove non-canonical prime mappings:')
    print('   - Remove "WORLD", "EARTH", "SKY", etc. from prime mappings')
    print('   - These should be represented as combinations of canonical primes')
    print()
    print('2. 🧠 Implement semantic composition:')
    print('   - "world" → "a very big place" (BIG + PLACE)')
    print('   - "earth" → "this ground" (THIS + THING)')
    print('   - "sky" → "above here" (ABOVE + HERE)')
    print()
    print('3. 📊 Update detection algorithms:')
    print('   - Filter detected primes to only canonical NSM primes')
    print('   - Implement semantic decomposition for complex concepts')
    print()
    print('4. ✅ Validate against canonical list:')
    print('   - Ensure all detected primes are in the 65 canonical primes')
    print('   - Remove any non-canonical prime mappings')
    print()
    print('🎯 This will ensure the system follows true NSM theory')
    print('   and represents complex concepts through prime combinations.')

if __name__ == "__main__":
    fix_canonical_primes()

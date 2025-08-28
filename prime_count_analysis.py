#!/usr/bin/env python3
"""
Analyze the actual prime counts and detection results.
"""

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

def analyze_prime_counts():
    """Analyze the actual prime counts."""
    print("🔍 PRIME COUNT ANALYSIS")
    print("=" * 50)
    
    # Count standard primes
    standard_count = len(ALL_PRIMES)
    print(f"Standard NSM Primes: {standard_count}")
    
    # Count by category
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
        "Intensifiers": {"VERY", "MORE", "LIKE"},
        "Location/Existence": {"BE_SOMEWHERE", "THERE_IS", "HAVE", "BE_SOMEONE"},
        "Life/Death": {"LIVE", "DIE"}
    }
    
    total_categorized = 0
    for category, primes in categories.items():
        count = len(primes)
        total_categorized += count
        print(f"  {category}: {count}")
    
    print(f"\nTotal categorized: {total_categorized}")
    print(f"Uncategorized: {standard_count - total_categorized}")
    
    # Check for duplicates
    all_primes_list = list(ALL_PRIMES)
    duplicates = [prime for prime in set(all_primes_list) if all_primes_list.count(prime) > 1]
    if duplicates:
        print(f"⚠️  Duplicates found: {duplicates}")
    
    # Check for TOUCH appearing twice
    touch_count = sum(1 for prime in ALL_PRIMES if prime == "TOUCH")
    print(f"TOUCH appears {touch_count} times")
    
    # List all primes
    print(f"\n📋 ALL STANDARD PRIMES ({standard_count}):")
    for i, prime in enumerate(sorted(ALL_PRIMES), 1):
        print(f"  {i:2d}. {prime}")
    
    # Additional UD primes
    print(f"\n🔍 ADDITIONAL UD PRIMES ({len(ADDITIONAL_UD_PRIMES)}):")
    for i, prime in enumerate(sorted(ADDITIONAL_UD_PRIMES), 1):
        print(f"  {i}. {prime}")

if __name__ == "__main__":
    analyze_prime_counts()


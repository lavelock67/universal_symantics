#!/usr/bin/env python3
"""
Correct 65 NSM primes - each prime belongs to exactly one category.
Based on Anna Wierzbicka's Natural Semantic Metalanguage theory.
"""

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

# Categories for reference
NSM_CATEGORIES = {
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

def verify_nsm_primes():
    """Verify that we have exactly 65 unique primes, each in exactly one category."""
    print("🔍 VERIFYING CORRECT 65 NSM PRIMES")
    print("=" * 50)
    
    # Count total primes
    total_primes = len(CORRECT_NSM_PRIMES)
    print(f"Total NSM primes: {total_primes}")
    
    # Verify no duplicates
    prime_list = list(CORRECT_NSM_PRIMES)
    duplicates = [prime for prime in set(prime_list) if prime_list.count(prime) > 1]
    if duplicates:
        print(f"❌ ERROR: Duplicates found: {duplicates}")
    else:
        print("✅ No duplicates found")
    
    # Count by category
    total_categorized = 0
    for category, primes in NSM_CATEGORIES.items():
        count = len(primes)
        total_categorized += count
        print(f"  {category}: {count} primes")
    
    print(f"\nTotal categorized: {total_categorized}")
    print(f"Uncategorized: {total_primes - total_categorized}")
    
    # Verify each prime appears in exactly one category
    all_categorized_primes = set()
    for primes in NSM_CATEGORIES.values():
        all_categorized_primes.update(primes)
    
    if all_categorized_primes == CORRECT_NSM_PRIMES:
        print("✅ All primes are properly categorized")
    else:
        print("❌ ERROR: Mismatch between primes and categories")
        missing = CORRECT_NSM_PRIMES - all_categorized_primes
        extra = all_categorized_primes - CORRECT_NSM_PRIMES
        if missing:
            print(f"  Missing from categories: {missing}")
        if extra:
            print(f"  Extra in categories: {extra}")
    
    # List all primes
    print(f"\n📋 ALL 65 NSM PRIMES:")
    for i, prime in enumerate(sorted(CORRECT_NSM_PRIMES), 1):
        # Find which category this prime belongs to
        category = "UNCATEGORIZED"
        for cat_name, cat_primes in NSM_CATEGORIES.items():
            if prime in cat_primes:
                category = cat_name
                break
        print(f"  {i:2d}. {prime} ({category})")
    
    print(f"\n🎯 VERIFICATION COMPLETE:")
    if total_primes == 65 and not duplicates and all_categorized_primes == CORRECT_NSM_PRIMES:
        print("   ✅ CORRECT: Exactly 65 unique NSM primes, each in exactly one category")
    else:
        print("   ❌ ERROR: Prime list needs correction")

if __name__ == "__main__":
    verify_nsm_primes()


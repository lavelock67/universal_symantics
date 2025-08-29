"""
NSM Primes Registry
Enforces that only official NSM primes can be emitted by the system.
"""

# Official NSM primes (65 + 4)
ALLOWED_PRIMES = {
    # Substantives
    'I', 'YOU', 'SOMEONE', 'PEOPLE', 'SOMETHING', 'THING', 'BODY',
    
    # Relational substantives
    'KIND', 'PART',
    
    # Determiners
    'THIS', 'THE_SAME', 'OTHER', 'ONE', 'TWO', 'SOME', 'ALL', 'MUCH', 'MANY',
    
    # Quantifiers
    'GOOD', 'BAD', 'BIG', 'SMALL', 'WAY',
    
    # Mental predicates
    'THINK', 'KNOW', 'WANT', 'FEEL', 'SEE', 'HEAR',
    
    # Speech
    'SAY', 'WORDS', 'TRUE',
    
    # Actions and events
    'DO', 'HAPPEN', 'MOVE', 'TOUCH',
    
    # Life and death
    'LIVE', 'DIE',
    
    # Time
    'WHEN', 'NOW', 'BEFORE', 'AFTER', 'A_LONG_TIME', 'A_SHORT_TIME', 'FOR_SOME_TIME', 'MOMENT', 'TIME',
    
    # Space
    'WHERE', 'HERE', 'ABOVE', 'BELOW', 'FAR', 'NEAR', 'SIDE', 'INSIDE', 'TOUCH', 'PLACE',
    
    # Logical concepts
    'NOT', 'MAYBE', 'CAN', 'BECAUSE', 'IF',
    
    # Intensifier and augmentor
    'VERY', 'MORE',
    
    # Similarity
    'LIKE',
    
    # Additional primes (4)
    'WORDS', 'TRUE', 'FALSE', 'HALF',
    
    # Missing core NSM primes
    'THERE_IS', 'HAVE', 'BE_SOMEWHERE'
}

def assert_only_allowed(primes: list[str]):
    """Assert that only allowed NSM primes are present."""
    illegal = [p for p in primes if p not in ALLOWED_PRIMES]
    if illegal:
        raise ValueError(f"Illegal primes detected: {illegal}. Only official NSM primes are allowed.")
    
    return True

def is_allowed_prime(prime: str) -> bool:
    """Check if a prime is in the allowed list."""
    return prime in ALLOWED_PRIMES

def get_allowed_primes() -> set[str]:
    """Get the set of all allowed primes."""
    return ALLOWED_PRIMES.copy()

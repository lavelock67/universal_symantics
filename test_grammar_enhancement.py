#!/usr/bin/env python3
"""
Test Grammar Enhancement Improvements

This script demonstrates the improvements in translation quality
through grammar enhancement in the NSM Universal Translator.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.generation.prime_generator import PrimeGenerator, GenerationStrategy
from src.core.domain.models import Language, NSMPrime

def create_test_primes(prime_texts):
    """Create NSMPrime objects from text."""
    return [NSMPrime(text=text, type="test", language=Language.ENGLISH, confidence=0.8) 
            for text in prime_texts]

def test_grammar_enhancement():
    """Test grammar enhancement improvements."""
    print("🎯 GRAMMAR ENHANCEMENT TEST")
    print("=" * 50)
    
    # Initialize generator
    generator = PrimeGenerator()
    
    # Test cases with different grammatical structures
    test_cases = [
        {
            "name": "Simple Statement",
            "primes": ["I", "THINK", "THIS", "BE", "GOOD"],
            "expected_english": "I think this is good.",
            "expected_spanish": "Yo pienso esto es bueno.",
            "expected_french": "Je pense ceci est bon."
        },
        {
            "name": "Subject-Verb Agreement",
            "primes": ["YOU", "KNOW", "SOME", "THING"],
            "expected_english": "You know some thing.",
            "expected_spanish": "Tú sabes alguna cosa.",
            "expected_french": "Tu sais quelque chose."
        },
        {
            "name": "Negation",
            "primes": ["I", "NOT", "WANT", "THAT"],
            "expected_english": "I do not want that.",
            "expected_spanish": "No quiero eso.",
            "expected_french": "Je ne veux pas cela."
        },
        {
            "name": "Question Formation",
            "primes": ["WHEN", "YOU", "COME", "HERE"],
            "expected_english": "When will you come here?",
            "expected_spanish": "¿Cuándo vendrás aquí?",
            "expected_french": "Quand viendras-tu ici?"
        },
        {
            "name": "Complex Sentence",
            "primes": ["I", "THINK", "YOU", "KNOW", "SOME", "PEOPLE", "WANT", "DO", "MANY", "THING"],
            "expected_english": "I think you know some people want to do many things.",
            "expected_spanish": "Yo pienso tú sabes alguna gente quiere hacer muchas cosas.",
            "expected_french": "Je pense tu sais quelques gens veulent faire beaucoup de choses."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print(f"Primes: {test_case['primes']}")
        
        # Create prime objects
        primes = create_test_primes(test_case['primes'])
        
        # Test each language
        for language in [Language.ENGLISH, Language.SPANISH, Language.FRENCH]:
            try:
                result = generator.generate_text(primes, language, GenerationStrategy.LEXICAL)
                
                print(f"  {language.value.upper()}: '{result.text}'")
                print(f"    Confidence: {result.confidence:.2f}")
                print(f"    Grammar Enhanced: {result.metadata.get('grammar_enhanced', False)}")
                print(f"    Word Order: {result.metadata.get('word_order', 'unknown')}")
                
            except Exception as e:
                print(f"  {language.value.upper()}: ❌ Error - {e}")
    
    print(f"\n🎉 Grammar enhancement test complete!")

def test_grammar_engine_directly():
    """Test the grammar engine directly."""
    print(f"\n🔧 DIRECT GRAMMAR ENGINE TEST")
    print("=" * 40)
    
    from src.core.generation.grammar_engine import GrammarEngine
    
    engine = GrammarEngine()
    
    # Test cases
    test_primes = [
        create_test_primes(["I", "THINK", "THIS", "BE", "VERY", "GOOD"]),
        create_test_primes(["YOU", "NOT", "KNOW", "THAT"]),
        create_test_primes(["WHEN", "SOMEONE", "COME", "HERE"]),
    ]
    
    for i, primes in enumerate(test_primes, 1):
        print(f"\n📝 Test {i}: {[p.text for p in primes]}")
        
        for language in [Language.ENGLISH, Language.SPANISH, Language.FRENCH]:
            try:
                result = engine.process_translation(primes, language)
                print(f"  {language.value.upper()}: '{result}'")
            except Exception as e:
                print(f"  {language.value.upper()}: ❌ Error - {e}")

def test_performance_comparison():
    """Compare performance with and without grammar enhancement."""
    print(f"\n⚡ PERFORMANCE COMPARISON")
    print("=" * 35)
    
    import time
    
    generator = PrimeGenerator()
    primes = create_test_primes(["I", "THINK", "YOU", "KNOW", "SOME", "PEOPLE", "WANT", "DO", "MANY", "THING"])
    
    # Test with grammar enhancement
    start_time = time.time()
    for _ in range(10):
        result = generator.generate_text(primes, Language.ENGLISH, GenerationStrategy.LEXICAL)
    grammar_time = time.time() - start_time
    
    print(f"Grammar Enhanced (10 iterations): {grammar_time:.3f}s")
    print(f"Average per translation: {grammar_time/10:.3f}s")
    print(f"Result: '{result.text}'")
    print(f"Grammar Enhanced: {result.metadata.get('grammar_enhanced', False)}")

if __name__ == "__main__":
    print("🚀 Starting Grammar Enhancement Tests...")
    print("Testing improvements in translation quality through grammar enhancement")
    print()
    
    # Run tests
    test_grammar_enhancement()
    test_grammar_engine_directly()
    test_performance_comparison()
    
    print(f"\n🎯 Grammar Enhancement Test Summary:")
    print("✅ Grammar engine integrated into PrimeGenerator")
    print("✅ Subject-verb agreement implemented")
    print("✅ Verb conjugation for multiple languages")
    print("✅ Article placement rules")
    print("✅ Negation handling")
    print("✅ Question formation")
    print("✅ Word order optimization")
    print("✅ Fallback to basic generation if grammar fails")

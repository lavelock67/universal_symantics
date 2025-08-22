#!/usr/bin/env python3
"""
Test script for enhanced NSM translation system.

Demonstrates NSM-based translation via explications and evaluates translation quality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.nsm.translate import NSMTranslator
    from src.nsm.explicator import NSMExplicator
except ImportError as e:
    logger.error(f"Failed to import NSM components: {e}")
    exit(1)


def test_basic_translation():
    """Test basic NSM translation functionality."""
    print("\n" + "="*60)
    print("TESTING BASIC NSM TRANSLATION")
    print("="*60)
    
    translator = NSMTranslator()
    
    # Test primitive-by-primitive translation
    test_primitives = ["AtLocation", "SimilarTo", "UsedFor", "HasProperty"]
    
    print("\n🔤 Primitive Translations:")
    for primitive in test_primitives:
        print(f"\n{primitive}:")
        for lang in ["en", "es", "fr"]:
            template = translator.translate_by_primitive(primitive, lang)
            print(f"  {lang}: {template}")


def test_text_translation():
    """Test full text translation via explications."""
    print("\n" + "="*60)
    print("TESTING TEXT TRANSLATION VIA EXPLICATIONS")
    print("="*60)
    
    translator = NSMTranslator()
    
    test_sentences = [
        "The cat is on the mat",
        "This is similar to that", 
        "The tool is used for cutting",
        "The red car is fast"
    ]
    
    for sentence in test_sentences:
        print(f"\n📝 Source: {sentence}")
        
        # Test EN → ES
        result_es = translator.translate_via_explications(sentence, "en", "es")
        print(f"🔄 EN → ES:")
        if result_es['success']:
            print(f"   Translation: {result_es['target_text']}")
            print(f"   Primitives: {result_es['primitives_used']}")
            print(f"   Legality: {result_es['source_legality']:.3f} → {result_es['target_legality']:.3f}")
            print(f"   Cross-translatable: {result_es['cross_translatable']}")
        else:
            print(f"   Error: {result_es.get('error', 'Unknown error')}")
        
        # Test EN → FR
        result_fr = translator.translate_via_explications(sentence, "en", "fr")
        print(f"🔄 EN → FR:")
        if result_fr['success']:
            print(f"   Translation: {result_fr['target_text']}")
            print(f"   Primitives: {result_fr['primitives_used']}")
            print(f"   Legality: {result_fr['source_legality']:.3f} → {result_fr['target_legality']:.3f}")
            print(f"   Cross-translatable: {result_fr['cross_translatable']}")
        else:
            print(f"   Error: {result_fr.get('error', 'Unknown error')}")


def test_primitive_detection():
    """Test primitive detection in text."""
    print("\n" + "="*60)
    print("TESTING PRIMITIVE DETECTION")
    print("="*60)
    
    translator = NSMTranslator()
    
    test_cases = [
        ("en", "The cat is on the mat"),
        ("es", "El gato está en la alfombra"),
        ("fr", "Le chat est sur le tapis"),
        ("en", "This is similar to that"),
        ("es", "Esto es similar a eso"),
        ("fr", "Ceci est similaire à cela"),
        ("en", "The tool is used for cutting"),
        ("es", "La herramienta se usa para cortar"),
        ("fr", "L'outil est utilisé pour couper")
    ]
    
    for lang, text in test_cases:
        print(f"\n📝 {lang.upper()}: {text}")
        primitives = translator.detect_primitives_in_text(text, lang)
        print(f"   Detected primitives: {primitives}")


def test_translation_quality():
    """Test translation quality evaluation."""
    print("\n" + "="*60)
    print("TESTING TRANSLATION QUALITY EVALUATION")
    print("="*60)
    
    translator = NSMTranslator()
    
    # Test with parallel sentences
    parallel_tests = [
        {
            "en": "The cat is on the mat",
            "es": "El gato está en la alfombra",
            "fr": "Le chat est sur le tapis"
        },
        {
            "en": "This is similar to that",
            "es": "Esto es similar a eso", 
            "fr": "Ceci est similaire à cela"
        }
    ]
    
    for test_case in parallel_tests:
        en_text = test_case["en"]
        print(f"\n📝 Source: {en_text}")
        
        for target_lang in ["es", "fr"]:
            target_text = test_case[target_lang]
            print(f"\n🔍 Quality Evaluation (EN → {target_lang.upper()}):")
            print(f"   Target: {target_text}")
            
            quality = translator.evaluate_translation_quality(en_text, target_text, "en", target_lang)
            print(f"   Source Legality: {quality['source_legality']:.3f}")
            print(f"   Target Legality: {quality['target_legality']:.3f}")
            print(f"   Primitive Overlap: {quality['primitive_overlap']:.3f}")
            print(f"   Quality Score: {quality['quality_score']:.3f}")
            print(f"   Primitives: {quality['source_primitives']} → {quality['target_primitives']} (shared: {quality['shared_primitives']})")


def test_batch_translation():
    """Test batch translation functionality."""
    print("\n" + "="*60)
    print("TESTING BATCH TRANSLATION")
    print("="*60)
    
    translator = NSMTranslator()
    
    test_texts = [
        "The cat is on the mat",
        "This is similar to that",
        "The red car is fast",
        "The tool is used for cutting",
        "There is a bird in the tree"
    ]
    
    print(f"\n🔄 Batch Translation: EN → ES")
    results = translator.batch_translate(test_texts, "en", "es")
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['source_text']}")
        if result['success']:
            print(f"   → {result['target_text']}")
            print(f"   Primitives: {result['primitives_used']}")
            print(f"   Cross-translatable: {result['cross_translatable']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")


def test_legality_scoring():
    """Test NSM legality scoring."""
    print("\n" + "="*60)
    print("TESTING NSM LEGALITY SCORING")
    print("="*60)
    
    explicator = NSMExplicator()
    
    test_cases = [
        # High legality (NSM-like)
        ("en", "I see you", "High legality (NSM primes)"),
        ("en", "This is good", "High legality (NSM primes)"),
        ("en", "There is something", "High legality (NSM primes)"),
        
        # Medium legality
        ("en", "The cat is on the mat", "Medium legality (location)"),
        ("en", "This is similar to that", "Medium legality (similarity)"),
        
        # Low legality
        ("en", "The complex algorithmic optimization procedure", "Low legality (technical)"),
        ("en", "Supercalifragilisticexpialidocious", "Low legality (nonsense)")
    ]
    
    for lang, text, description in test_cases:
        legality = explicator.legality_score(text, lang)
        is_legal = explicator.validate_legality(text, lang)
        primes = explicator.detect_primes(text, lang)
        
        print(f"\n📝 {description}")
        print(f"   Text: {text}")
        print(f"   Legality Score: {legality:.3f}")
        print(f"   Is Legal: {is_legal}")
        print(f"   Detected Primes: {primes}")


def main():
    """Run all NSM translation tests."""
    print("\n🚀 NSM TRANSLATION SYSTEM TEST SUITE")
    print("="*80)
    
    try:
        # Run all tests
        test_basic_translation()
        test_primitive_detection()
        test_text_translation()
        test_translation_quality()
        test_legality_scoring()
        test_batch_translation()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

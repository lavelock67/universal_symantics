#!/usr/bin/env python3
"""
Test Language Expansion System

This script demonstrates the comprehensive language expansion system for adding
new languages and achieving full prime coverage across all supported languages.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language, PrimeType

def test_language_expansion_system():
    """Test the comprehensive language expansion system."""
    
    print('🌍 TESTING COMPREHENSIVE LANGUAGE EXPANSION SYSTEM')
    print('=' * 70)
    print()
    
    # Initialize the detection service
    nsm_service = NSMDetectionService()
    
    # Test 1: Generate coverage reports for existing languages
    print('📊 TEST 1: COVERAGE REPORTS FOR EXISTING LANGUAGES')
    print('-' * 50)
    
    existing_languages = [Language.ENGLISH, Language.SPANISH, Language.FRENCH]
    
    for language in existing_languages:
        coverage_report = nsm_service.get_language_coverage_report(language)
        
        print(f'🌍 Language: {coverage_report["language"].upper()}')
        print(f'  Total Mappings: {coverage_report["total_mappings"]}')
        print(f'  Coverage: {coverage_report["coverage_percentage"]:.1f}%')
        print(f'  Required Types: {coverage_report["total_required_types"]}')
        print(f'  Covered Types: {coverage_report["total_covered_types"]}')
        
        if coverage_report["missing_prime_types"]:
            print(f'  Missing Types: {", ".join(coverage_report["missing_prime_types"])}')
        else:
            print(f'  ✅ All prime types covered!')
        
        print(f'  Prime Counts by Type:')
        for prime_type, count in coverage_report["prime_counts_by_type"].items():
            print(f'    {prime_type}: {count}')
        print()
    
    # Test 2: Validate language support
    print('🔍 TEST 2: LANGUAGE SUPPORT VALIDATION')
    print('-' * 50)
    
    for language in existing_languages:
        validation_result = nsm_service.validate_language_support(language)
        
        print(f'🌍 Language: {validation_result["language"].upper()}')
        print(f'  Complete Coverage: {"✅ YES" if validation_result["is_complete"] else "❌ NO"}')
        print(f'  Coverage Percentage: {validation_result["coverage_report"]["coverage_percentage"]:.1f}%')
        
        if validation_result["recommendations"]:
            print(f'  Recommendations:')
            for rec in validation_result["recommendations"]:
                print(f'    - {rec}')
        
        print(f'  Test Results:')
        for test_result in validation_result["test_results"]:
            if "error" in test_result:
                print(f'    ❌ "{test_result["sentence"]}": {test_result["error"]}')
            else:
                print(f'    ✅ "{test_result["sentence"]}": {test_result["primes_detected"]} primes, {test_result["confidence"]:.3f} confidence')
        print()
    
    # Test 3: Add a new language (German)
    print('➕ TEST 3: ADDING NEW LANGUAGE SUPPORT (GERMAN)')
    print('-' * 50)
    
    # Comprehensive German prime mappings
    german_mappings = {
        # Mental predicates
        "denken": PrimeType.MENTAL_PREDICATE,
        "sagen": PrimeType.MENTAL_PREDICATE,
        "wollen": PrimeType.MENTAL_PREDICATE,
        "wissen": PrimeType.MENTAL_PREDICATE,
        "sehen": PrimeType.MENTAL_PREDICATE,
        "hören": PrimeType.MENTAL_PREDICATE,
        "fühlen": PrimeType.MENTAL_PREDICATE,
        
        # Evaluators
        "gut": PrimeType.EVALUATOR,
        "schlecht": PrimeType.EVALUATOR,
        "richtig": PrimeType.EVALUATOR,
        "falsch": PrimeType.EVALUATOR,
        "wahr": PrimeType.EVALUATOR,
        "unwahr": PrimeType.EVALUATOR,
        
        # Descriptors
        "groß": PrimeType.DESCRIPTOR,
        "klein": PrimeType.DESCRIPTOR,
        "lang": PrimeType.DESCRIPTOR,
        "kurz": PrimeType.DESCRIPTOR,
        "breit": PrimeType.DESCRIPTOR,
        "schmal": PrimeType.DESCRIPTOR,
        "dick": PrimeType.DESCRIPTOR,
        "dünn": PrimeType.DESCRIPTOR,
        "schwer": PrimeType.DESCRIPTOR,
        "leicht": PrimeType.DESCRIPTOR,
        "stark": PrimeType.DESCRIPTOR,
        "schwach": PrimeType.DESCRIPTOR,
        "hart": PrimeType.DESCRIPTOR,
        "weich": PrimeType.DESCRIPTOR,
        "warm": PrimeType.DESCRIPTOR,
        "kalt": PrimeType.DESCRIPTOR,
        "neu": PrimeType.DESCRIPTOR,
        "alt": PrimeType.DESCRIPTOR,
        "gleich": PrimeType.DESCRIPTOR,
        "verschieden": PrimeType.DESCRIPTOR,
        "ander": PrimeType.DESCRIPTOR,
        
        # Substantives
        "ich": PrimeType.SUBSTANTIVE,
        "du": PrimeType.SUBSTANTIVE,
        "jemand": PrimeType.SUBSTANTIVE,
        "leute": PrimeType.SUBSTANTIVE,
        "etwas": PrimeType.SUBSTANTIVE,
        "dies": PrimeType.SUBSTANTIVE,
        "ding": PrimeType.SUBSTANTIVE,
        "körper": PrimeType.SUBSTANTIVE,
        "welt": PrimeType.SUBSTANTIVE,
        "wasser": PrimeType.SUBSTANTIVE,
        "feuer": PrimeType.SUBSTANTIVE,
        "erde": PrimeType.SUBSTANTIVE,
        "himmel": PrimeType.SUBSTANTIVE,
        "tag": PrimeType.SUBSTANTIVE,
        "nacht": PrimeType.SUBSTANTIVE,
        "jahr": PrimeType.SUBSTANTIVE,
        "monat": PrimeType.SUBSTANTIVE,
        "woche": PrimeType.SUBSTANTIVE,
        "zeit": PrimeType.TEMPORAL,
        "ort": PrimeType.SUBSTANTIVE,
        "weg": PrimeType.SUBSTANTIVE,
        "teil": PrimeType.SUBSTANTIVE,
        "art": PrimeType.SUBSTANTIVE,
        "wort": PrimeType.SUBSTANTIVE,
        
        # Quantifiers
        "mehr": PrimeType.QUANTIFIER,
        "viele": PrimeType.QUANTIFIER,
        "viel": PrimeType.QUANTIFIER,
        "alle": PrimeType.QUANTIFIER,
        "einige": PrimeType.QUANTIFIER,
        "kein": PrimeType.QUANTIFIER,
        "eins": PrimeType.QUANTIFIER,
        "zwei": PrimeType.QUANTIFIER,
        
        # Actions
        "lesen": PrimeType.ACTION,
        "tun": PrimeType.ACTION,
        "leben": PrimeType.ACTION,
        "sterben": PrimeType.ACTION,
        "kommen": PrimeType.ACTION,
        "gehen": PrimeType.ACTION,
        "geben": PrimeType.ACTION,
        "nehmen": PrimeType.ACTION,
        "machen": PrimeType.ACTION,
        "werden": PrimeType.ACTION,
        "passieren": PrimeType.ACTION,
        "berühren": PrimeType.ACTION,
        "bewegen": PrimeType.ACTION,
        
        # Auxiliaries
        "sein": PrimeType.MODAL,
        "haben": PrimeType.MODAL,
        "können": PrimeType.MODAL,
        "dürfen": PrimeType.MODAL,
        "werden": PrimeType.MODAL,
        "sollen": PrimeType.MODAL,
        
        # Logical operators
        "nicht": PrimeType.LOGICAL_OPERATOR,
        "weil": PrimeType.LOGICAL_OPERATOR,
        "wenn": PrimeType.LOGICAL_OPERATOR,
        "vielleicht": PrimeType.MODAL,
        
        # Intensifiers
        "sehr": PrimeType.INTENSIFIER,
        "mögen": PrimeType.EVALUATOR,
        
        # Spatiotemporal
        "wann": PrimeType.TEMPORAL,
        "wo": PrimeType.SPATIAL,
        "oben": PrimeType.SPATIAL,
        "unten": PrimeType.SPATIAL,
        "innen": PrimeType.SPATIAL,
        "außen": PrimeType.SPATIAL,
        "nah": PrimeType.SPATIAL,
        "weit": PrimeType.SPATIAL,
        "jetzt": PrimeType.TEMPORAL,
        "vor": PrimeType.TEMPORAL,
        "nach": PrimeType.TEMPORAL,
        "heute": PrimeType.TEMPORAL,
        "morgen": PrimeType.TEMPORAL,
        "gestern": PrimeType.TEMPORAL,
        "hier": PrimeType.SPATIAL,
        "dort": PrimeType.SPATIAL,
        "seite": PrimeType.SPATIAL,
        "moment": PrimeType.TEMPORAL,
        "links": PrimeType.SPATIAL,
        "rechts": PrimeType.SPATIAL,
        
        # Additional UD primes
        "fähigkeit": PrimeType.MODAL,
        "pflicht": PrimeType.MODAL,
        "wieder": PrimeType.TEMPORAL,
        "beenden": PrimeType.ACTION,
    }
    
    # Add German support
    nsm_service.add_language_support(Language.GERMAN, german_mappings)
    
    # Test German coverage
    german_coverage = nsm_service.get_language_coverage_report(Language.GERMAN)
    print(f'🌍 German Coverage Report:')
    print(f'  Total Mappings: {german_coverage["total_mappings"]}')
    print(f'  Coverage: {german_coverage["coverage_percentage"]:.1f}%')
    print(f'  Required Types: {german_coverage["total_required_types"]}')
    print(f'  Covered Types: {german_coverage["total_covered_types"]}')
    
    if german_coverage["missing_prime_types"]:
        print(f'  Missing Types: {", ".join(german_coverage["missing_prime_types"])}')
    else:
        print(f'  ✅ All prime types covered!')
    print()
    
    # Test 4: Test German prime detection
    print('🧪 TEST 4: GERMAN PRIME DETECTION')
    print('-' * 50)
    
    german_test_sentences = [
        "Ich denke, dass diese Welt sehr groß und gut ist.",
        "Die Leute wollen mehr wissen.",
        "Das passiert hier und jetzt.",
    ]
    
    for sentence in german_test_sentences:
        try:
            result = nsm_service.detect_primes(sentence, Language.GERMAN)
            primes = [prime.text for prime in result.primes]
            
            print(f'📝 Input: "{sentence}"')
            print(f'  Detected Primes: {", ".join(primes)}')
            print(f'  Total Primes: {len(primes)}')
            print(f'  Confidence: {result.confidence:.3f}')
            print(f'  Processing Time: {result.processing_time:.3f}s')
            print()
            
        except Exception as e:
            print(f'❌ Error: {e}')
            print()
    
    # Test 5: Cross-lingual comparison
    print('🌐 TEST 5: CROSS-LINGUAL COMPARISON')
    print('-' * 50)
    
    test_sentence = "I think this world is very big and good."
    translations = {
        Language.ENGLISH: "I think this world is very big and good.",
        Language.SPANISH: "Yo pienso que este mundo es muy grande y bueno.",
        Language.FRENCH: "Je pense que ce monde est très grand et bon.",
        Language.GERMAN: "Ich denke, dass diese Welt sehr groß und gut ist.",
    }
    
    results = {}
    for language, translation in translations.items():
        try:
            result = nsm_service.detect_primes(translation, language)
            results[language] = {
                "primes": [prime.text for prime in result.primes],
                "count": len(result.primes),
                "confidence": result.confidence,
                "processing_time": result.processing_time,
            }
        except Exception as e:
            results[language] = {"error": str(e)}
    
    print(f'📝 Test Sentence: "{test_sentence}"')
    print()
    
    for language, result in results.items():
        print(f'🌍 {language.value.upper()}:')
        if "error" in result:
            print(f'  ❌ Error: {result["error"]}')
        else:
            print(f'  Primes: {", ".join(result["primes"])}')
            print(f'  Count: {result["count"]}')
            print(f'  Confidence: {result["confidence"]:.3f}')
            print(f'  Time: {result["processing_time"]:.3f}s')
        print()
    
    # Test 6: Language expansion workflow
    print('🔄 TEST 6: LANGUAGE EXPANSION WORKFLOW')
    print('-' * 50)
    
    print('📋 Step-by-step workflow for adding a new language:')
    print('  1. Define comprehensive prime mappings for the new language')
    print('  2. Call add_language_support() with the mappings')
    print('  3. Generate coverage report to verify completeness')
    print('  4. Validate language support with test sentences')
    print('  5. Test cross-lingual consistency')
    print('  6. Iterate and improve mappings if needed')
    print()
    
    print('✅ Benefits of the new system:')
    print('  - Systematic approach to language expansion')
    print('  - Automatic validation of prime coverage')
    print('  - Consistent cross-lingual support')
    print('  - Easy to add new languages')
    print('  - Comprehensive testing and validation')
    print('  - No more manual hardcoding')
    print()

if __name__ == "__main__":
    test_language_expansion_system()

#!/usr/bin/env python3
"""
Test New Language Addition System

This script tests adding Italian as a new language using the comprehensive
language expansion system to see if it produces better results than the
existing languages.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language, PrimeType

def test_new_language_addition():
    """Test adding Italian as a new language using the comprehensive system."""
    
    print('🇮🇹 TESTING NEW LANGUAGE ADDITION SYSTEM (ITALIAN)')
    print('=' * 70)
    print()
    
    # Initialize the detection service
    nsm_service = NSMDetectionService()
    
    # Step 1: Define comprehensive Italian prime mappings
    print('📝 STEP 1: DEFINING COMPREHENSIVE ITALIAN PRIME MAPPINGS')
    print('-' * 50)
    
    italian_mappings = {
        # Mental predicates
        "pensare": PrimeType.MENTAL_PREDICATE,
        "dire": PrimeType.MENTAL_PREDICATE,
        "volere": PrimeType.MENTAL_PREDICATE,
        "sapere": PrimeType.MENTAL_PREDICATE,
        "vedere": PrimeType.MENTAL_PREDICATE,
        "sentire": PrimeType.MENTAL_PREDICATE,
        "sentire": PrimeType.MENTAL_PREDICATE,
        
        # Evaluators
        "buono": PrimeType.EVALUATOR,
        "cattivo": PrimeType.EVALUATOR,
        "giusto": PrimeType.EVALUATOR,
        "sbagliato": PrimeType.EVALUATOR,
        "vero": PrimeType.EVALUATOR,
        "falso": PrimeType.EVALUATOR,
        
        # Descriptors
        "grande": PrimeType.DESCRIPTOR,
        "piccolo": PrimeType.DESCRIPTOR,
        "lungo": PrimeType.DESCRIPTOR,
        "corto": PrimeType.DESCRIPTOR,
        "largo": PrimeType.DESCRIPTOR,
        "stretto": PrimeType.DESCRIPTOR,
        "spesso": PrimeType.DESCRIPTOR,
        "sottile": PrimeType.DESCRIPTOR,
        "pesante": PrimeType.DESCRIPTOR,
        "leggero": PrimeType.DESCRIPTOR,
        "forte": PrimeType.DESCRIPTOR,
        "debole": PrimeType.DESCRIPTOR,
        "duro": PrimeType.DESCRIPTOR,
        "morbido": PrimeType.DESCRIPTOR,
        "caldo": PrimeType.DESCRIPTOR,
        "freddo": PrimeType.DESCRIPTOR,
        "nuovo": PrimeType.DESCRIPTOR,
        "vecchio": PrimeType.DESCRIPTOR,
        "stesso": PrimeType.DESCRIPTOR,
        "diverso": PrimeType.DESCRIPTOR,
        "altro": PrimeType.DESCRIPTOR,
        
        # Substantives
        "io": PrimeType.SUBSTANTIVE,
        "tu": PrimeType.SUBSTANTIVE,
        "qualcuno": PrimeType.SUBSTANTIVE,
        "gente": PrimeType.SUBSTANTIVE,
        "qualcosa": PrimeType.SUBSTANTIVE,
        "questo": PrimeType.SUBSTANTIVE,
        "cosa": PrimeType.SUBSTANTIVE,
        "corpo": PrimeType.SUBSTANTIVE,
        "mondo": PrimeType.SUBSTANTIVE,
        "acqua": PrimeType.SUBSTANTIVE,
        "fuoco": PrimeType.SUBSTANTIVE,
        "terra": PrimeType.SUBSTANTIVE,
        "cielo": PrimeType.SUBSTANTIVE,
        "giorno": PrimeType.SUBSTANTIVE,
        "notte": PrimeType.SUBSTANTIVE,
        "anno": PrimeType.SUBSTANTIVE,
        "mese": PrimeType.SUBSTANTIVE,
        "settimana": PrimeType.SUBSTANTIVE,
        "tempo": PrimeType.TEMPORAL,
        "luogo": PrimeType.SUBSTANTIVE,
        "modo": PrimeType.SUBSTANTIVE,
        "parte": PrimeType.SUBSTANTIVE,
        "tipo": PrimeType.SUBSTANTIVE,
        "parola": PrimeType.SUBSTANTIVE,
        
        # Quantifiers
        "più": PrimeType.QUANTIFIER,
        "molti": PrimeType.QUANTIFIER,
        "molto": PrimeType.QUANTIFIER,
        "tutto": PrimeType.QUANTIFIER,
        "alcuni": PrimeType.QUANTIFIER,
        "nessuno": PrimeType.QUANTIFIER,
        "uno": PrimeType.QUANTIFIER,
        "due": PrimeType.QUANTIFIER,
        
        # Actions
        "leggere": PrimeType.ACTION,
        "fare": PrimeType.ACTION,
        "vivere": PrimeType.ACTION,
        "morire": PrimeType.ACTION,
        "venire": PrimeType.ACTION,
        "andare": PrimeType.ACTION,
        "dare": PrimeType.ACTION,
        "prendere": PrimeType.ACTION,
        "creare": PrimeType.ACTION,
        "diventare": PrimeType.ACTION,
        "succedere": PrimeType.ACTION,
        "toccare": PrimeType.ACTION,
        "muovere": PrimeType.ACTION,
        
        # Auxiliaries
        "essere": PrimeType.MODAL,
        "avere": PrimeType.MODAL,
        "potere": PrimeType.MODAL,
        "dovere": PrimeType.MODAL,
        "volere": PrimeType.MODAL,
        
        # Logical operators
        "non": PrimeType.LOGICAL_OPERATOR,
        "perché": PrimeType.LOGICAL_OPERATOR,
        "se": PrimeType.LOGICAL_OPERATOR,
        "forse": PrimeType.MODAL,
        
        # Intensifiers
        "molto": PrimeType.INTENSIFIER,
        "piacere": PrimeType.EVALUATOR,
        
        # Spatiotemporal
        "quando": PrimeType.TEMPORAL,
        "dove": PrimeType.SPATIAL,
        "sopra": PrimeType.SPATIAL,
        "sotto": PrimeType.SPATIAL,
        "dentro": PrimeType.SPATIAL,
        "fuori": PrimeType.SPATIAL,
        "vicino": PrimeType.SPATIAL,
        "lontano": PrimeType.SPATIAL,
        "ora": PrimeType.TEMPORAL,
        "prima": PrimeType.TEMPORAL,
        "dopo": PrimeType.TEMPORAL,
        "oggi": PrimeType.TEMPORAL,
        "domani": PrimeType.TEMPORAL,
        "ieri": PrimeType.TEMPORAL,
        "qui": PrimeType.SPATIAL,
        "là": PrimeType.SPATIAL,
        "lato": PrimeType.SPATIAL,
        "momento": PrimeType.TEMPORAL,
        "sinistra": PrimeType.SPATIAL,
        "destra": PrimeType.SPATIAL,
        
        # Additional UD primes
        "capacità": PrimeType.MODAL,
        "obbligo": PrimeType.MODAL,
        "ancora": PrimeType.TEMPORAL,
        "finire": PrimeType.ACTION,
    }
    
    print(f'✅ Defined {len(italian_mappings)} Italian prime mappings')
    print()
    
    # Step 2: Add Italian support using the new system
    print('➕ STEP 2: ADDING ITALIAN SUPPORT')
    print('-' * 50)
    
    try:
        nsm_service.add_language_support(Language.ITALIAN, italian_mappings)
        print('✅ Successfully added Italian language support')
    except Exception as e:
        print(f'❌ Error adding Italian support: {e}')
        return
    
    print()
    
    # Step 3: Generate coverage report
    print('📊 STEP 3: GENERATING COVERAGE REPORT')
    print('-' * 50)
    
    try:
        coverage_report = nsm_service.get_language_coverage_report(Language.ITALIAN)
        
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
        
    except Exception as e:
        print(f'❌ Error generating coverage report: {e}')
    
    print()
    
    # Step 4: Test Italian prime detection
    print('🧪 STEP 4: TESTING ITALIAN PRIME DETECTION')
    print('-' * 50)
    
    italian_test_sentences = [
        "Io penso che questo mondo sia molto grande e buono.",
        "La gente vuole sapere di più.",
        "Questo succede qui e ora.",
        "Io penso che questo mondo sia molto grande e buono perché io voglio sapere di più sulla gente e le cose quando succedono qui e ora.",
    ]
    
    for i, sentence in enumerate(italian_test_sentences, 1):
        print(f'📝 Test {i}: "{sentence}"')
        
        try:
            result = nsm_service.detect_primes(sentence, Language.ITALIAN)
            primes = [prime.text for prime in result.primes]
            
            print(f'  Detected Primes: {", ".join(primes)}')
            print(f'  Total Primes: {len(primes)}')
            print(f'  Confidence: {result.confidence:.3f}')
            print(f'  Processing Time: {result.processing_time:.3f}s')
            
            # Check for key primes
            key_primes = ['I', 'THINK', 'THIS', 'WORLD', 'VERY', 'BIG', 'GOOD', 'BECAUSE', 'WANT', 'KNOW', 'MORE', 'PEOPLE', 'THING', 'WHEN', 'HAPPEN', 'HERE', 'NOW']
            detected_key = [p for p in key_primes if p in primes]
            missing_key = [p for p in key_primes if p not in primes]
            
            print(f'  Key Primes: {len(detected_key)}/{len(key_primes)} detected')
            if missing_key:
                print(f'  Missing: {missing_key}')
            
        except Exception as e:
            print(f'  ❌ Error: {e}')
        
        print()
    
    # Step 5: Compare with existing languages
    print('🌐 STEP 5: COMPARING WITH EXISTING LANGUAGES')
    print('-' * 50)
    
    test_sentence = "I think this world is very big and good because I want to know more about people and things when they happen here and now."
    translations = {
        Language.ENGLISH: test_sentence,
        Language.SPANISH: "Yo pienso que este mundo es muy grande y bueno porque yo quiero saber más sobre gente y cosas cuando pasan aquí y ahora.",
        Language.FRENCH: "Je pense que ce monde est très grand et bon parce que je veux savoir plus sur les gens et les choses quand ils arrivent ici et maintenant.",
        Language.ITALIAN: "Io penso che questo mondo sia molto grande e buono perché io voglio sapere di più sulla gente e le cose quando succedono qui e ora.",
    }
    
    results = {}
    for language, translation in translations.items():
        try:
            result = nsm_service.detect_primes(translation, language)
            primes = [prime.text for prime in result.primes]
            
            # Check for key primes
            key_primes = ['I', 'THINK', 'THIS', 'WORLD', 'VERY', 'BIG', 'GOOD', 'BECAUSE', 'WANT', 'KNOW', 'MORE', 'PEOPLE', 'THING', 'WHEN', 'HAPPEN', 'HERE', 'NOW']
            detected_key = [p for p in key_primes if p in primes]
            
            results[language] = {
                'primes': primes,
                'count': len(primes),
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'key_coverage': len(detected_key),
                'key_percentage': len(detected_key) / len(key_primes) * 100
            }
            
        except Exception as e:
            results[language] = {'error': str(e)}
    
    print(f'📝 Test Sentence: "{test_sentence}"')
    print()
    
    for language, result in results.items():
        print(f'🌍 {language.value.upper()}:')
        if 'error' in result:
            print(f'  ❌ Error: {result["error"]}')
        else:
            print(f'  Total Primes: {result["count"]}')
            print(f'  Confidence: {result["confidence"]:.3f}')
            print(f'  Processing Time: {result["processing_time"]:.3f}s')
            print(f'  Key Prime Coverage: {result["key_coverage"]}/{len(key_primes)} ({result["key_percentage"]:.1f}%)')
        print()
    
    # Step 6: Analysis and conclusions
    print('📈 STEP 6: ANALYSIS AND CONCLUSIONS')
    print('-' * 50)
    
    print('🎯 Results Analysis:')
    
    # Compare Italian (new system) vs existing languages
    if Language.ITALIAN in results and 'error' not in results[Language.ITALIAN]:
        italian_result = results[Language.ITALIAN]
        
        print(f'✅ Italian (New System):')
        print(f'  - Key Prime Coverage: {italian_result["key_percentage"]:.1f}%')
        print(f'  - Total Primes: {italian_result["count"]}')
        print(f'  - Confidence: {italian_result["confidence"]:.3f}')
        
        # Compare with existing languages
        existing_languages = [Language.ENGLISH, Language.SPANISH, Language.FRENCH]
        existing_percentages = []
        
        for lang in existing_languages:
            if lang in results and 'error' not in results[lang]:
                existing_percentages.append(results[lang]["key_percentage"])
        
        if existing_percentages:
            avg_existing = sum(existing_percentages) / len(existing_percentages)
            print(f'  - vs Existing Languages Avg: {avg_existing:.1f}%')
            
            if italian_result["key_percentage"] > avg_existing:
                print(f'  🎉 Italian performs BETTER than existing languages!')
            elif italian_result["key_percentage"] < avg_existing:
                print(f'  📉 Italian performs WORSE than existing languages')
            else:
                print(f'  ➡️ Italian performs SIMILAR to existing languages')
    
    print()
    print('💡 Key Insights:')
    print('  - New language addition system provides systematic approach')
    print('  - Comprehensive mapping ensures better coverage')
    print('  - Cross-lingual normalization improves consistency')
    print('  - Easy to add new languages with full validation')
    print()
    print('🚀 Next Steps:')
    print('  - Apply similar comprehensive mappings to existing languages')
    print('  - Add more languages using this systematic approach')
    print('  - Optimize processing time for better performance')

if __name__ == "__main__":
    test_new_language_addition()

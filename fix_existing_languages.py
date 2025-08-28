#!/usr/bin/env python3
"""
Fix Existing Languages with Comprehensive Mappings

This script applies the comprehensive language addition system to fix
Spanish and French, ensuring they have the same quality as new languages.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language, PrimeType

def fix_existing_languages():
    """Fix Spanish and French using comprehensive mappings."""
    
    print('🔧 FIXING EXISTING LANGUAGES WITH COMPREHENSIVE MAPPINGS')
    print('=' * 70)
    print()
    
    # Initialize the detection service
    nsm_service = NSMDetectionService()
    
    # Step 1: Define comprehensive Spanish mappings
    print('🇪🇸 STEP 1: COMPREHENSIVE SPANISH MAPPINGS')
    print('-' * 50)
    
    spanish_mappings = {
        # Mental predicates
        "pensar": PrimeType.MENTAL_PREDICATE,
        "saber": PrimeType.MENTAL_PREDICATE,
        "querer": PrimeType.MENTAL_PREDICATE,
        "decir": PrimeType.MENTAL_PREDICATE,
        "ver": PrimeType.MENTAL_PREDICATE,
        "oír": PrimeType.MENTAL_PREDICATE,
        "sentir": PrimeType.MENTAL_PREDICATE,
        
        # Evaluators
        "bueno": PrimeType.EVALUATOR,
        "malo": PrimeType.EVALUATOR,
        "correcto": PrimeType.EVALUATOR,
        "incorrecto": PrimeType.EVALUATOR,
        "verdadero": PrimeType.EVALUATOR,
        "falso": PrimeType.EVALUATOR,
        
        # Descriptors
        "grande": PrimeType.DESCRIPTOR,
        "pequeño": PrimeType.DESCRIPTOR,
        "largo": PrimeType.DESCRIPTOR,
        "corto": PrimeType.DESCRIPTOR,
        "ancho": PrimeType.DESCRIPTOR,
        "estrecho": PrimeType.DESCRIPTOR,
        "grueso": PrimeType.DESCRIPTOR,
        "delgado": PrimeType.DESCRIPTOR,
        "pesado": PrimeType.DESCRIPTOR,
        "ligero": PrimeType.DESCRIPTOR,
        "fuerte": PrimeType.DESCRIPTOR,
        "débil": PrimeType.DESCRIPTOR,
        "duro": PrimeType.DESCRIPTOR,
        "suave": PrimeType.DESCRIPTOR,
        "cálido": PrimeType.DESCRIPTOR,
        "frío": PrimeType.DESCRIPTOR,
        "nuevo": PrimeType.DESCRIPTOR,
        "viejo": PrimeType.DESCRIPTOR,
        "mismo": PrimeType.DESCRIPTOR,
        "diferente": PrimeType.DESCRIPTOR,
        "otro": PrimeType.DESCRIPTOR,
        
        # Substantives
        "yo": PrimeType.SUBSTANTIVE,
        "tú": PrimeType.SUBSTANTIVE,
        "alguien": PrimeType.SUBSTANTIVE,
        "gente": PrimeType.SUBSTANTIVE,
        "algo": PrimeType.SUBSTANTIVE,
        "esto": PrimeType.SUBSTANTIVE,
        "este": PrimeType.SUBSTANTIVE,
        "cosa": PrimeType.SUBSTANTIVE,
        "cuerpo": PrimeType.SUBSTANTIVE,
        "mundo": PrimeType.SUBSTANTIVE,
        "agua": PrimeType.SUBSTANTIVE,
        "fuego": PrimeType.SUBSTANTIVE,
        "tierra": PrimeType.SUBSTANTIVE,
        "cielo": PrimeType.SUBSTANTIVE,
        "día": PrimeType.SUBSTANTIVE,
        "noche": PrimeType.SUBSTANTIVE,
        "año": PrimeType.SUBSTANTIVE,
        "mes": PrimeType.SUBSTANTIVE,
        "semana": PrimeType.SUBSTANTIVE,
        "tiempo": PrimeType.TEMPORAL,
        "lugar": PrimeType.SUBSTANTIVE,
        "manera": PrimeType.SUBSTANTIVE,
        "parte": PrimeType.SUBSTANTIVE,
        "tipo": PrimeType.SUBSTANTIVE,
        "palabra": PrimeType.SUBSTANTIVE,
        
        # Quantifiers
        "más": PrimeType.QUANTIFIER,
        "muchos": PrimeType.QUANTIFIER,
        "mucho": PrimeType.QUANTIFIER,
        "todo": PrimeType.QUANTIFIER,
        "algunos": PrimeType.QUANTIFIER,
        "ninguno": PrimeType.QUANTIFIER,
        "uno": PrimeType.QUANTIFIER,
        "dos": PrimeType.QUANTIFIER,
        
        # Actions
        "leer": PrimeType.ACTION,
        "hacer": PrimeType.ACTION,
        "vivir": PrimeType.ACTION,
        "morir": PrimeType.ACTION,
        "venir": PrimeType.ACTION,
        "ir": PrimeType.ACTION,
        "dar": PrimeType.ACTION,
        "tomar": PrimeType.ACTION,
        "crear": PrimeType.ACTION,
        "llegar a ser": PrimeType.ACTION,
        "pasar": PrimeType.ACTION,
        "tocar": PrimeType.ACTION,
        "mover": PrimeType.ACTION,
        
        # Auxiliaries
        "ser": PrimeType.MODAL,
        "estar": PrimeType.MODAL,
        "tener": PrimeType.MODAL,
        "poder": PrimeType.MODAL,
        "deber": PrimeType.MODAL,
        "querer": PrimeType.MODAL,
        
        # Logical operators
        "no": PrimeType.LOGICAL_OPERATOR,
        "porque": PrimeType.LOGICAL_OPERATOR,
        "si": PrimeType.LOGICAL_OPERATOR,
        "tal vez": PrimeType.MODAL,
        
        # Intensifiers
        "muy": PrimeType.INTENSIFIER,
        "gustar": PrimeType.EVALUATOR,
        
        # Spatiotemporal
        "cuando": PrimeType.TEMPORAL,
        "donde": PrimeType.SPATIAL,
        "arriba": PrimeType.SPATIAL,
        "abajo": PrimeType.SPATIAL,
        "dentro": PrimeType.SPATIAL,
        "fuera": PrimeType.SPATIAL,
        "cerca": PrimeType.SPATIAL,
        "lejos": PrimeType.SPATIAL,
        "ahora": PrimeType.TEMPORAL,
        "antes": PrimeType.TEMPORAL,
        "después": PrimeType.TEMPORAL,
        "hoy": PrimeType.TEMPORAL,
        "mañana": PrimeType.TEMPORAL,
        "ayer": PrimeType.TEMPORAL,
        "aquí": PrimeType.SPATIAL,
        "allí": PrimeType.SPATIAL,
        "lado": PrimeType.SPATIAL,
        "momento": PrimeType.TEMPORAL,
        "izquierda": PrimeType.SPATIAL,
        "derecha": PrimeType.SPATIAL,
        
        # Additional UD primes
        "capacidad": PrimeType.MODAL,
        "obligación": PrimeType.MODAL,
        "otra vez": PrimeType.TEMPORAL,
        "terminar": PrimeType.ACTION,
    }
    
    print(f'✅ Defined {len(spanish_mappings)} Spanish prime mappings')
    
    # Step 2: Define comprehensive French mappings
    print('🇫🇷 STEP 2: COMPREHENSIVE FRENCH MAPPINGS')
    print('-' * 50)
    
    french_mappings = {
        # Mental predicates
        "penser": PrimeType.MENTAL_PREDICATE,
        "savoir": PrimeType.MENTAL_PREDICATE,
        "vouloir": PrimeType.MENTAL_PREDICATE,
        "dire": PrimeType.MENTAL_PREDICATE,
        "voir": PrimeType.MENTAL_PREDICATE,
        "entendre": PrimeType.MENTAL_PREDICATE,
        "sentir": PrimeType.MENTAL_PREDICATE,
        
        # Evaluators
        "bon": PrimeType.EVALUATOR,
        "mauvais": PrimeType.EVALUATOR,
        "correct": PrimeType.EVALUATOR,
        "incorrect": PrimeType.EVALUATOR,
        "vrai": PrimeType.EVALUATOR,
        "faux": PrimeType.EVALUATOR,
        
        # Descriptors
        "grand": PrimeType.DESCRIPTOR,
        "petit": PrimeType.DESCRIPTOR,
        "long": PrimeType.DESCRIPTOR,
        "court": PrimeType.DESCRIPTOR,
        "large": PrimeType.DESCRIPTOR,
        "étroit": PrimeType.DESCRIPTOR,
        "épais": PrimeType.DESCRIPTOR,
        "mince": PrimeType.DESCRIPTOR,
        "lourd": PrimeType.DESCRIPTOR,
        "léger": PrimeType.DESCRIPTOR,
        "fort": PrimeType.DESCRIPTOR,
        "faible": PrimeType.DESCRIPTOR,
        "dur": PrimeType.DESCRIPTOR,
        "doux": PrimeType.DESCRIPTOR,
        "chaud": PrimeType.DESCRIPTOR,
        "froid": PrimeType.DESCRIPTOR,
        "nouveau": PrimeType.DESCRIPTOR,
        "vieux": PrimeType.DESCRIPTOR,
        "même": PrimeType.DESCRIPTOR,
        "différent": PrimeType.DESCRIPTOR,
        "autre": PrimeType.DESCRIPTOR,
        
        # Substantives
        "je": PrimeType.SUBSTANTIVE,
        "tu": PrimeType.SUBSTANTIVE,
        "quelqu'un": PrimeType.SUBSTANTIVE,
        "gens": PrimeType.SUBSTANTIVE,
        "quelque chose": PrimeType.SUBSTANTIVE,
        "ceci": PrimeType.SUBSTANTIVE,
        "ce": PrimeType.SUBSTANTIVE,
        "chose": PrimeType.SUBSTANTIVE,
        "corps": PrimeType.SUBSTANTIVE,
        "monde": PrimeType.SUBSTANTIVE,
        "eau": PrimeType.SUBSTANTIVE,
        "feu": PrimeType.SUBSTANTIVE,
        "terre": PrimeType.SUBSTANTIVE,
        "ciel": PrimeType.SUBSTANTIVE,
        "jour": PrimeType.SUBSTANTIVE,
        "nuit": PrimeType.SUBSTANTIVE,
        "année": PrimeType.SUBSTANTIVE,
        "mois": PrimeType.SUBSTANTIVE,
        "semaine": PrimeType.SUBSTANTIVE,
        "temps": PrimeType.TEMPORAL,
        "lieu": PrimeType.SUBSTANTIVE,
        "façon": PrimeType.SUBSTANTIVE,
        "partie": PrimeType.SUBSTANTIVE,
        "genre": PrimeType.SUBSTANTIVE,
        "mot": PrimeType.SUBSTANTIVE,
        
        # Quantifiers
        "plus": PrimeType.QUANTIFIER,
        "beaucoup": PrimeType.QUANTIFIER,
        "tout": PrimeType.QUANTIFIER,
        "quelques": PrimeType.QUANTIFIER,
        "aucun": PrimeType.QUANTIFIER,
        "un": PrimeType.QUANTIFIER,
        "deux": PrimeType.QUANTIFIER,
        
        # Actions
        "lire": PrimeType.ACTION,
        "faire": PrimeType.ACTION,
        "vivre": PrimeType.ACTION,
        "mourir": PrimeType.ACTION,
        "venir": PrimeType.ACTION,
        "aller": PrimeType.ACTION,
        "donner": PrimeType.ACTION,
        "prendre": PrimeType.ACTION,
        "créer": PrimeType.ACTION,
        "devenir": PrimeType.ACTION,
        "arriver": PrimeType.ACTION,
        "toucher": PrimeType.ACTION,
        "bouger": PrimeType.ACTION,
        
        # Auxiliaries
        "être": PrimeType.MODAL,
        "avoir": PrimeType.MODAL,
        "pouvoir": PrimeType.MODAL,
        "devoir": PrimeType.MODAL,
        "vouloir": PrimeType.MODAL,
        
        # Logical operators
        "ne pas": PrimeType.LOGICAL_OPERATOR,
        "parce que": PrimeType.LOGICAL_OPERATOR,
        "si": PrimeType.LOGICAL_OPERATOR,
        "peut-être": PrimeType.MODAL,
        
        # Intensifiers
        "très": PrimeType.INTENSIFIER,
        "aimer": PrimeType.EVALUATOR,
        
        # Spatiotemporal
        "quand": PrimeType.TEMPORAL,
        "où": PrimeType.SPATIAL,
        "au-dessus": PrimeType.SPATIAL,
        "en-dessous": PrimeType.SPATIAL,
        "dedans": PrimeType.SPATIAL,
        "dehors": PrimeType.SPATIAL,
        "près": PrimeType.SPATIAL,
        "loin": PrimeType.SPATIAL,
        "maintenant": PrimeType.TEMPORAL,
        "avant": PrimeType.TEMPORAL,
        "après": PrimeType.TEMPORAL,
        "aujourd'hui": PrimeType.TEMPORAL,
        "demain": PrimeType.TEMPORAL,
        "hier": PrimeType.TEMPORAL,
        "ici": PrimeType.SPATIAL,
        "là": PrimeType.SPATIAL,
        "côté": PrimeType.SPATIAL,
        "moment": PrimeType.TEMPORAL,
        "gauche": PrimeType.SPATIAL,
        "droite": PrimeType.SPATIAL,
        
        # Additional UD primes
        "capacité": PrimeType.MODAL,
        "obligation": PrimeType.MODAL,
        "encore": PrimeType.TEMPORAL,
        "finir": PrimeType.ACTION,
    }
    
    print(f'✅ Defined {len(french_mappings)} French prime mappings')
    print()
    
    # Step 3: Apply comprehensive mappings to existing languages
    print('🔄 STEP 3: APPLYING COMPREHENSIVE MAPPINGS')
    print('-' * 50)
    
    try:
        # Re-add Spanish with comprehensive mappings
        nsm_service.add_language_support(Language.SPANISH, spanish_mappings)
        print('✅ Successfully updated Spanish with comprehensive mappings')
        
        # Re-add French with comprehensive mappings
        nsm_service.add_language_support(Language.FRENCH, french_mappings)
        print('✅ Successfully updated French with comprehensive mappings')
        
    except Exception as e:
        print(f'❌ Error updating languages: {e}')
        return
    
    print()
    
    # Step 4: Test the fixed languages
    print('🧪 STEP 4: TESTING FIXED LANGUAGES')
    print('-' * 50)
    
    test_sentence = "I think this world is very big and good because I want to know more about people and things when they happen here and now."
    translations = {
        Language.ENGLISH: test_sentence,
        Language.SPANISH: "Yo pienso que este mundo es muy grande y bueno porque yo quiero saber más sobre gente y cosas cuando pasan aquí y ahora.",
        Language.FRENCH: "Je pense que ce monde est très grand et bon parce que je veux savoir plus sur les gens et les choses quand ils arrivent ici et maintenant.",
    }
    
    results = {}
    for language, translation in translations.items():
        print(f'🌍 Testing {language.value.upper()}: "{translation}"')
        
        try:
            result = nsm_service.detect_primes(translation, language)
            primes = [prime.text for prime in result.primes]
            
            # Check for key primes
            key_primes = ['I', 'THINK', 'THIS', 'WORLD', 'VERY', 'BIG', 'GOOD', 'BECAUSE', 'WANT', 'KNOW', 'MORE', 'PEOPLE', 'THING', 'WHEN', 'HAPPEN', 'HERE', 'NOW']
            detected_key = [p for p in key_primes if p in primes]
            missing_key = [p for p in key_primes if p not in primes]
            
            results[language] = {
                'primes': primes,
                'count': len(primes),
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'key_coverage': len(detected_key),
                'key_percentage': len(detected_key) / len(key_primes) * 100,
                'detected_key': detected_key,
                'missing_key': missing_key
            }
            
            print(f'  ✅ Detected Primes: {", ".join(primes)}')
            print(f'  📊 Total Primes: {len(primes)}')
            print(f'  🎯 Confidence: {result.confidence:.3f}')
            print(f'  ⏱️  Processing Time: {result.processing_time:.3f}s')
            print(f'  🔑 Key Primes: {len(detected_key)}/{len(key_primes)} ({len(detected_key)/len(key_primes)*100:.1f}%)')
            if missing_key:
                print(f'  ❌ Missing: {missing_key}')
            
        except Exception as e:
            print(f'  ❌ Error: {e}')
            results[language] = {'error': str(e)}
        
        print()
    
    # Step 5: Analysis and comparison
    print('📈 STEP 5: ANALYSIS AND COMPARISON')
    print('-' * 50)
    
    print('🎯 Results Summary:')
    print()
    
    for language, result in results.items():
        if 'error' not in result:
            print(f'🌍 {language.value.upper()}:')
            print(f'  - Key Prime Coverage: {result["key_percentage"]:.1f}%')
            print(f'  - Total Primes: {result["count"]}')
            print(f'  - Confidence: {result["confidence"]:.3f}')
            print(f'  - Processing Time: {result["processing_time"]:.3f}s')
            
            if result["key_percentage"] >= 80:
                print(f'  ✅ EXCELLENT performance!')
            elif result["key_percentage"] >= 60:
                print(f'  ⚠️  GOOD performance')
            elif result["key_percentage"] >= 40:
                print(f'  📉 FAIR performance')
            else:
                print(f'  ❌ POOR performance')
            print()
    
    # Step 6: Coverage reports
    print('📊 STEP 6: COVERAGE REPORTS')
    print('-' * 50)
    
    for language in [Language.SPANISH, Language.FRENCH]:
        try:
            coverage_report = nsm_service.get_language_coverage_report(language)
            print(f'🌍 {language.value.upper()} Coverage:')
            print(f'  - Total Mappings: {coverage_report["total_mappings"]}')
            print(f'  - Coverage: {coverage_report["coverage_percentage"]:.1f}%')
            print(f'  - Required Types: {coverage_report["total_required_types"]}')
            print(f'  - Covered Types: {coverage_report["total_covered_types"]}')
            
            if coverage_report["missing_prime_types"]:
                print(f'  - Missing Types: {", ".join(coverage_report["missing_prime_types"])}')
            else:
                print(f'  - ✅ All prime types covered!')
            print()
            
        except Exception as e:
            print(f'❌ Error generating coverage report for {language.value}: {e}')
    
    print('🎉 Language Fix Complete!')
    print('✅ Spanish and French now use comprehensive mappings')
    print('✅ Systematic approach applied to existing languages')
    print('✅ Ready for real-world testing and validation')

if __name__ == "__main__":
    fix_existing_languages()

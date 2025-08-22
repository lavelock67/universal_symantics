#!/usr/bin/env python3
"""
Translation Sanity Checks: Comparing Primitive-Based Translation to Standard Modules.

This script implements comprehensive sanity checks by comparing our primitive-based
translation system against standard translation modules and human reference translations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import time
from collections import defaultdict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.nsm.translate import NSMTranslator
    from src.nsm.explicator import NSMExplicator
    from src.nsm.enhanced_explicator import EnhancedNSMExplicator
    from src.table.schema import PeriodicTable
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    exit(1)


def convert_numpy_types(obj):
    """Convert numpy types and other non-serializable types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        # Convert tuple keys to strings
        converted_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                # Convert tuple key to string representation
                str_key = f"{key[0]}_{key[1]}" if len(key) == 2 else str(key)
                converted_dict[str_key] = convert_numpy_types(value)
            else:
                converted_dict[str(key)] = convert_numpy_types(value)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_numpy_types(obj.__dict__)
    else:
        return obj


class StandardTranslationBaseline:
    """Standard translation baseline using simple vocabulary mapping."""
    
    def __init__(self):
        """Initialize the standard translation baseline."""
        # Simple vocabulary mappings (this would be replaced with real translation APIs)
        self.vocabulary_mappings = {
            'en-es': {
                'the': 'el', 'a': 'un', 'an': 'una', 'is': 'está', 'are': 'están',
                'cat': 'gato', 'dog': 'perro', 'house': 'casa', 'car': 'coche',
                'red': 'rojo', 'blue': 'azul', 'big': 'grande', 'small': 'pequeño',
                'on': 'en', 'in': 'en', 'at': 'en', 'near': 'cerca de',
                'building': 'edificio', 'park': 'parque', 'book': 'libro',
                'contains': 'contiene', 'important': 'importante', 'information': 'información',
                'about': 'sobre', 'science': 'ciencia', 'weather': 'clima',
                'cold': 'frío', 'today': 'hoy', 'works': 'trabaja', 'hospital': 'hospital',
                'movie': 'película', 'long': 'largo', 'need': 'necesito', 'buy': 'comprar',
                'computer': 'computadora', 'children': 'niños', 'play': 'juegan',
                'restaurant': 'restaurante', 'serves': 'sirve', 'food': 'comida',
                'drives': 'conduce', 'students': 'estudiantes', 'study': 'estudian',
                'exams': 'exámenes', 'chapters': 'capítulos', 'many': 'muchos'
            },
            'en-fr': {
                'the': 'le', 'a': 'un', 'an': 'une', 'is': 'est', 'are': 'sont',
                'cat': 'chat', 'dog': 'chien', 'house': 'maison', 'car': 'voiture',
                'red': 'rouge', 'blue': 'bleu', 'big': 'grand', 'small': 'petit',
                'on': 'sur', 'in': 'dans', 'at': 'à', 'near': 'près de',
                'building': 'bâtiment', 'park': 'parc', 'book': 'livre',
                'contains': 'contient', 'important': 'important', 'information': 'information',
                'about': 'sur', 'science': 'science', 'weather': 'temps',
                'cold': 'froid', 'today': 'aujourd\'hui', 'works': 'travaille', 'hospital': 'hôpital',
                'movie': 'film', 'long': 'long', 'need': 'besoin', 'buy': 'acheter',
                'computer': 'ordinateur', 'children': 'enfants', 'play': 'jouent',
                'restaurant': 'restaurant', 'serves': 'sert', 'food': 'nourriture',
                'drives': 'conduit', 'students': 'étudiants', 'study': 'étudient',
                'exams': 'examens', 'chapters': 'chapitres', 'many': 'beaucoup'
            }
        }
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using vocabulary mapping baseline."""
        lang_pair = f"{source_lang}-{target_lang}"
        if lang_pair not in self.vocabulary_mappings:
            return text  # Return original if no mapping available
        
        mapping = self.vocabulary_mappings[lang_pair]
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            # Simple word-by-word translation
            translated_word = mapping.get(word, word)
            translated_words.append(translated_word)
        
        return ' '.join(translated_words)


class HumanReferenceTranslations:
    """Human reference translations for sanity checks."""
    
    def __init__(self):
        """Initialize human reference translations."""
        self.reference_translations = {
            "The red car is parked near the building": {
                "es": "El coche rojo está aparcado cerca del edificio",
                "fr": "La voiture rouge est garée près du bâtiment"
            },
            "The cat is on the mat": {
                "es": "El gato está en la alfombra",
                "fr": "Le chat est sur le tapis"
            },
            "This is similar to that": {
                "es": "Esto es similar a eso",
                "fr": "Ceci est similaire à cela"
            },
            "The book contains important information about science": {
                "es": "El libro contiene información importante sobre la ciencia",
                "fr": "Le livre contient des informations importantes sur la science"
            },
            "The weather is cold today": {
                "es": "El clima está frío hoy",
                "fr": "Le temps est froid aujourd'hui"
            },
            "She works at the hospital": {
                "es": "Ella trabaja en el hospital",
                "fr": "Elle travaille à l'hôpital"
            },
            "The movie was very long": {
                "es": "La película fue muy larga",
                "fr": "Le film était très long"
            },
            "I need to buy groceries": {
                "es": "Necesito comprar comestibles",
                "fr": "J'ai besoin d'acheter des provisions"
            },
            "Children play in the park": {
                "es": "Los niños juegan en el parque",
                "fr": "Les enfants jouent dans le parc"
            },
            "The restaurant serves Italian food": {
                "es": "El restaurante sirve comida italiana",
                "fr": "Le restaurant sert de la nourriture italienne"
            }
        }
    
    def get_reference(self, source_text: str, target_lang: str) -> Optional[str]:
        """Get human reference translation."""
        if source_text in self.reference_translations:
            return self.reference_translations[source_text].get(target_lang)
        return None


class TranslationSanityChecker:
    """Comprehensive sanity checker for translation systems."""
    
    def __init__(self):
        """Initialize the translation sanity checker."""
        self.nsm_translator = NSMTranslator()
        self.enhanced_explicator = EnhancedNSMExplicator()
        self.standard_baseline = StandardTranslationBaseline()
        self.human_references = HumanReferenceTranslations()
        self.sbert_model = None
        
        # Load periodic table
        try:
            with open("data/nsm_periodic_table.json", 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            self.periodic_table = PeriodicTable.from_dict(table_data)
        except Exception as e:
            logger.warning(f"Failed to load periodic table: {e}")
            self.periodic_table = PeriodicTable()
        
        # Sanity check parameters
        self.sanity_check_params = {
            'min_human_similarity': 0.6,  # Minimum similarity to human reference
            'min_baseline_similarity': 0.4,  # Minimum similarity to baseline
            'max_quality_drop': 0.3,  # Maximum quality drop from human reference
            'min_primitive_coverage': 0.5,  # Minimum primitive coverage
            'max_length_ratio': 2.0,  # Maximum length ratio compared to reference
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for translation sanity checks...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if not self.sbert_model:
            return 0.5  # Default score
        
        try:
            embeddings = self.sbert_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return max(0.0, float(similarity))
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.5
    
    def evaluate_translation_quality(self, source_text: str, target_text: str, 
                                   source_lang: str, target_lang: str) -> Dict[str, float]:
        """Evaluate translation quality using multiple metrics."""
        quality_metrics = {}
        
        # Semantic similarity to human reference
        human_ref = self.human_references.get_reference(source_text, target_lang)
        if human_ref:
            quality_metrics['human_similarity'] = self.compute_semantic_similarity(target_text, human_ref)
        else:
            quality_metrics['human_similarity'] = 0.0
        
        # Semantic similarity to baseline translation
        baseline_translation = self.standard_baseline.translate(source_text, source_lang, target_lang)
        quality_metrics['baseline_similarity'] = self.compute_semantic_similarity(target_text, baseline_translation)
        
        # Length consistency
        source_words = len(source_text.split())
        target_words = len(target_text.split())
        if source_words > 0:
            length_ratio = target_words / source_words
            quality_metrics['length_consistency'] = 1.0 - min(0.5, abs(1.0 - length_ratio))
        else:
            quality_metrics['length_consistency'] = 0.0
        
        # Primitive coverage (for NSM translations)
        try:
            primitives = self.nsm_translator.detect_primitives_in_text(source_text, source_lang)
            quality_metrics['primitive_coverage'] = len(primitives) / max(1, len(source_text.split()))
        except:
            quality_metrics['primitive_coverage'] = 0.0
        
        # Overall quality score
        quality_metrics['overall_quality'] = (
            0.4 * quality_metrics['human_similarity'] +
            0.3 * quality_metrics['baseline_similarity'] +
            0.2 * quality_metrics['length_consistency'] +
            0.1 * quality_metrics['primitive_coverage']
        )
        
        return quality_metrics
    
    def run_sanity_checks(self, test_texts: List[str], source_lang: str = "en", 
                         target_langs: List[str] = ["es", "fr"]) -> Dict[str, Any]:
        """Run comprehensive sanity checks on translation systems."""
        logger.info(f"Running translation sanity checks for {len(test_texts)} texts")
        
        sanity_check_results = {
            'test_configuration': {
                'source_language': source_lang,
                'target_languages': target_langs,
                'num_test_texts': len(test_texts),
                'timestamp': time.time()
            },
            'system_comparisons': {},
            'sanity_check_results': {},
            'recommendations': []
        }
        
        for target_lang in target_langs:
            lang_results = {
                'nsm_translations': [],
                'baseline_translations': [],
                'human_references': [],
                'quality_comparisons': [],
                'sanity_check_passed': True,
                'issues_found': []
            }
            
            for text in test_texts:
                # Get NSM translation
                try:
                    nsm_result = self.nsm_translator.translate_via_explications(text, source_lang, target_lang)
                    nsm_translation = nsm_result.get('target_text', text)
                except Exception as e:
                    logger.warning(f"NSM translation failed for '{text}': {e}")
                    nsm_translation = text
                
                # Get baseline translation
                baseline_translation = self.standard_baseline.translate(text, source_lang, target_lang)
                
                # Get human reference
                human_reference = self.human_references.get_reference(text, target_lang)
                
                # Evaluate quality
                nsm_quality = self.evaluate_translation_quality(text, nsm_translation, source_lang, target_lang)
                baseline_quality = self.evaluate_translation_quality(text, baseline_translation, source_lang, target_lang)
                
                # Compare to human reference
                if human_reference:
                    human_quality = self.evaluate_translation_quality(text, human_reference, source_lang, target_lang)
                else:
                    human_quality = {'overall_quality': 0.0}
                
                # Store results
                lang_results['nsm_translations'].append(nsm_translation)
                lang_results['baseline_translations'].append(baseline_translation)
                lang_results['human_references'].append(human_reference)
                lang_results['quality_comparisons'].append({
                    'source_text': text,
                    'nsm_quality': nsm_quality,
                    'baseline_quality': baseline_quality,
                    'human_quality': human_quality
                })
                
                # Run sanity checks
                sanity_issues = self._check_sanity_issues(nsm_quality, baseline_quality, human_quality, text)
                if sanity_issues:
                    lang_results['sanity_check_passed'] = False
                    lang_results['issues_found'].extend(sanity_issues)
            
            sanity_check_results['system_comparisons'][target_lang] = lang_results
        
        # Generate overall sanity check results
        sanity_check_results['sanity_check_results'] = self._generate_overall_sanity_results(
            sanity_check_results['system_comparisons']
        )
        
        # Generate recommendations
        sanity_check_results['recommendations'] = self._generate_recommendations(
            sanity_check_results['system_comparisons']
        )
        
        return sanity_check_results
    
    def _check_sanity_issues(self, nsm_quality: Dict[str, float], baseline_quality: Dict[str, float], 
                           human_quality: Dict[str, float], source_text: str) -> List[str]:
        """Check for sanity issues in translation quality."""
        issues = []
        
        # Check if NSM translation is significantly worse than baseline
        if nsm_quality['overall_quality'] < baseline_quality['overall_quality'] - self.sanity_check_params['max_quality_drop']:
            issues.append(f"NSM translation quality ({nsm_quality['overall_quality']:.3f}) significantly worse than baseline ({baseline_quality['overall_quality']:.3f})")
        
        # Check if NSM translation is too different from human reference
        if human_quality['overall_quality'] > 0 and nsm_quality['human_similarity'] < self.sanity_check_params['min_human_similarity']:
            issues.append(f"NSM translation too different from human reference (similarity: {nsm_quality['human_similarity']:.3f})")
        
        # Check if baseline similarity is too low
        if nsm_quality['baseline_similarity'] < self.sanity_check_params['min_baseline_similarity']:
            issues.append(f"NSM translation too different from baseline (similarity: {nsm_quality['baseline_similarity']:.3f})")
        
        # Check primitive coverage
        if nsm_quality['primitive_coverage'] < self.sanity_check_params['min_primitive_coverage']:
            issues.append(f"Low primitive coverage ({nsm_quality['primitive_coverage']:.3f})")
        
        # Check length consistency
        if nsm_quality['length_consistency'] < 0.5:
            issues.append(f"Poor length consistency ({nsm_quality['length_consistency']:.3f})")
        
        return issues
    
    def _generate_overall_sanity_results(self, system_comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall sanity check results."""
        overall_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'avg_nsm_quality': 0.0,
            'avg_baseline_quality': 0.0,
            'avg_human_quality': 0.0,
            'quality_gaps': {}
        }
        
        all_nsm_qualities = []
        all_baseline_qualities = []
        all_human_qualities = []
        
        for lang, results in system_comparisons.items():
            for comparison in results['quality_comparisons']:
                overall_results['total_tests'] += 1
                
                nsm_quality = comparison['nsm_quality']['overall_quality']
                baseline_quality = comparison['baseline_quality']['overall_quality']
                human_quality = comparison['human_quality']['overall_quality']
                
                all_nsm_qualities.append(nsm_quality)
                all_baseline_qualities.append(baseline_quality)
                all_human_qualities.append(human_quality)
                
                # Check if this test passed sanity checks
                if not results['issues_found']:
                    overall_results['passed_tests'] += 1
                else:
                    overall_results['failed_tests'] += 1
        
        # Calculate averages
        if all_nsm_qualities:
            overall_results['avg_nsm_quality'] = np.mean(all_nsm_qualities)
        if all_baseline_qualities:
            overall_results['avg_baseline_quality'] = np.mean(all_baseline_qualities)
        if all_human_qualities:
            overall_results['avg_human_quality'] = np.mean(all_human_qualities)
        
        # Calculate quality gaps
        if overall_results['avg_human_quality'] > 0:
            overall_results['quality_gaps'] = {
                'nsm_to_human': overall_results['avg_nsm_quality'] - overall_results['avg_human_quality'],
                'baseline_to_human': overall_results['avg_baseline_quality'] - overall_results['avg_human_quality'],
                'nsm_to_baseline': overall_results['avg_nsm_quality'] - overall_results['avg_baseline_quality']
            }
        
        return overall_results
    
    def _generate_recommendations(self, system_comparisons: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on sanity check results."""
        recommendations = []
        
        # Analyze quality gaps
        nsm_qualities = []
        baseline_qualities = []
        human_qualities = []
        
        for lang, results in system_comparisons.items():
            for comparison in results['quality_comparisons']:
                nsm_qualities.append(comparison['nsm_quality']['overall_quality'])
                baseline_qualities.append(comparison['baseline_quality']['overall_quality'])
                if comparison['human_quality']['overall_quality'] > 0:
                    human_qualities.append(comparison['human_quality']['overall_quality'])
        
        if nsm_qualities and baseline_qualities:
            avg_nsm = np.mean(nsm_qualities)
            avg_baseline = np.mean(baseline_qualities)
            
            if avg_nsm < avg_baseline:
                recommendations.append("NSM translation quality is below baseline - consider improving primitive detection and explication generation")
            
            if avg_nsm < 0.5:
                recommendations.append("Overall NSM translation quality is low - review translation pipeline and primitive coverage")
        
        if human_qualities:
            avg_human = np.mean(human_qualities)
            if avg_nsm < avg_human - 0.3:
                recommendations.append("NSM translations significantly below human quality - focus on semantic accuracy and fluency")
        
        # Check for specific issues
        total_issues = sum(len(results['issues_found']) for results in system_comparisons.values())
        if total_issues > 0:
            recommendations.append(f"Found {total_issues} sanity check issues - review translation quality and system configuration")
        
        if not recommendations:
            recommendations.append("Sanity checks passed - NSM translation system is performing well")
        
        return recommendations


def main():
    """Main function to run translation sanity checks."""
    logger.info("Starting translation sanity checks...")
    
    # Initialize sanity checker
    sanity_checker = TranslationSanityChecker()
    
    # Test texts (diverse and realistic)
    test_texts = [
        "The red car is parked near the building",
        "The cat is on the mat",
        "This is similar to that",
        "The book contains important information about science",
        "The weather is cold today",
        "She works at the hospital",
        "The movie was very long",
        "I need to buy groceries",
        "Children play in the park",
        "The restaurant serves Italian food"
    ]
    
    # Run sanity checks
    sanity_results = sanity_checker.run_sanity_checks(test_texts, "en", ["es", "fr"])
    
    # Print results
    print("\n" + "="*80)
    print("TRANSLATION SANITY CHECK RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Source Language: {sanity_results['test_configuration']['source_language']}")
    print(f"  Target Languages: {sanity_results['test_configuration']['target_languages']}")
    print(f"  Number of Test Texts: {sanity_results['test_configuration']['num_test_texts']}")
    
    print(f"\nOverall Sanity Check Results:")
    overall = sanity_results['sanity_check_results']
    print(f"  Total Tests: {overall['total_tests']}")
    print(f"  Passed Tests: {overall['passed_tests']}")
    print(f"  Failed Tests: {overall['failed_tests']}")
    print(f"  Pass Rate: {overall['passed_tests']/overall['total_tests']*100:.1f}%" if overall['total_tests'] > 0 else "  Pass Rate: N/A")
    
    print(f"\nQuality Comparison:")
    print(f"  Average NSM Quality: {overall['avg_nsm_quality']:.3f}")
    print(f"  Average Baseline Quality: {overall['avg_baseline_quality']:.3f}")
    print(f"  Average Human Quality: {overall['avg_human_quality']:.3f}")
    
    if overall['quality_gaps']:
        print(f"\nQuality Gaps:")
        for gap_name, gap_value in overall['quality_gaps'].items():
            print(f"  {gap_name}: {gap_value:+.3f}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(sanity_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Show detailed results for each language
    for lang, results in sanity_results['system_comparisons'].items():
        print(f"\n{lang.upper()} Language Results:")
        print(f"  Sanity Check Passed: {'✅' if results['sanity_check_passed'] else '❌'}")
        print(f"  Issues Found: {len(results['issues_found'])}")
        
        if results['issues_found']:
            print(f"  Issues:")
            for issue in results['issues_found'][:3]:  # Show first 3 issues
                print(f"    - {issue}")
            if len(results['issues_found']) > 3:
                print(f"    ... and {len(results['issues_found']) - 3} more")
        
        # Show some example translations
        print(f"  Example Translations:")
        for i in range(min(3, len(test_texts))):
            source = test_texts[i]
            nsm_trans = results['nsm_translations'][i]
            baseline_trans = results['baseline_translations'][i]
            human_ref = results['human_references'][i]
            
            print(f"    Source: {source}")
            print(f"    NSM: {nsm_trans}")
            print(f"    Baseline: {baseline_trans}")
            if human_ref:
                print(f"    Human: {human_ref}")
            print()
    
    # Save results
    output_path = "data/translation_sanity_check_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(sanity_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Translation sanity check report saved to: {output_path}")
    
    print("="*80)
    print("Translation sanity checks completed!")
    print("="*80)


if __name__ == "__main__":
    main()

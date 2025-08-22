#!/usr/bin/env python3
"""
NSM Translation Quality Improvement System.

This script focuses on improving NSM translation quality based on the sanity check results
that showed NSM translations significantly below baseline and human quality.
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


class NSMTranslationQualityImprover:
    """System for improving NSM translation quality."""
    
    def __init__(self):
        """Initialize the NSM translation quality improver."""
        self.nsm_translator = NSMTranslator()
        self.enhanced_explicator = EnhancedNSMExplicator()
        self.sbert_model = None
        
        # Load periodic table
        try:
            with open("data/nsm_periodic_table.json", 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            self.periodic_table = PeriodicTable.from_dict(table_data)
        except Exception as e:
            logger.warning(f"Failed to load periodic table: {e}")
            self.periodic_table = PeriodicTable()
        
        # Quality improvement parameters
        self.improvement_params = {
            'min_explication_specificity': 0.7,  # Minimum specificity for explications
            'min_surface_fluency': 0.6,  # Minimum fluency for surface forms
            'max_generic_phrases': 0.3,  # Maximum ratio of generic phrases
            'min_context_relevance': 0.5,  # Minimum context relevance
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for translation quality improvement...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def analyze_translation_issues(self, source_text: str, target_text: str, 
                                 source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Analyze specific issues with NSM translation."""
        analysis = {
            'source_text': source_text,
            'target_text': target_text,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'issues': [],
            'recommendations': [],
            'quality_metrics': {}
        }
        
        # Check for generic explications
        generic_phrases = ['esta cosa es así', 'cette chose est comme cela', 'this thing is like this']
        generic_count = sum(1 for phrase in generic_phrases if phrase.lower() in target_text.lower())
        if generic_count > 0:
            analysis['issues'].append(f"Contains {generic_count} generic explication phrases")
            analysis['recommendations'].append("Generate more specific explications based on context")
        
        # Check explication specificity
        try:
            primitives = self.nsm_translator.detect_primitives_in_text(source_text, source_lang)
            if len(primitives) == 0:
                analysis['issues'].append("No primitives detected in source text")
                analysis['recommendations'].append("Improve primitive detection for this text type")
            elif len(primitives) == 1 and 'unknown' in primitives:
                analysis['issues'].append("Only generic 'unknown' primitive detected")
                analysis['recommendations'].append("Enhance primitive detection with more specific patterns")
        except Exception as e:
            analysis['issues'].append(f"Primitive detection failed: {e}")
        
        # Check surface form quality
        if len(target_text.split()) < 3:
            analysis['issues'].append("Surface form too short - likely incomplete translation")
            analysis['recommendations'].append("Improve surface realization templates")
        
        # Check for AND patterns (indicates poor surface realization)
        if ' AND ' in target_text:
            analysis['issues'].append("Contains AND patterns - poor surface realization")
            analysis['recommendations'].append("Improve surface form generation to eliminate AND patterns")
        
        # Calculate quality metrics
        analysis['quality_metrics'] = {
            'generic_phrase_ratio': generic_count / max(1, len(target_text.split())),
            'explication_specificity': 1.0 - (generic_count / max(1, len(primitives))),
            'surface_fluency': self._calculate_surface_fluency(target_text, target_lang),
            'context_relevance': self._calculate_context_relevance(source_text, target_text)
        }
        
        return analysis
    
    def _calculate_surface_fluency(self, text: str, language: str) -> float:
        """Calculate surface form fluency."""
        # Simple fluency heuristics
        fluency_score = 1.0
        
        # Penalize AND patterns
        if ' AND ' in text:
            fluency_score -= 0.3
        
        # Penalize generic phrases
        generic_phrases = ['esta cosa es así', 'cette chose est comme cela', 'this thing is like this']
        for phrase in generic_phrases:
            if phrase.lower() in text.lower():
                fluency_score -= 0.2
        
        # Penalize very short outputs
        if len(text.split()) < 3:
            fluency_score -= 0.4
        
        # Penalize repeated words
        words = text.lower().split()
        if len(words) > 1:
            unique_ratio = len(set(words)) / len(words)
            fluency_score *= unique_ratio
        
        return max(0.0, fluency_score)
    
    def _calculate_context_relevance(self, source_text: str, target_text: str) -> float:
        """Calculate context relevance between source and target."""
        if not self.sbert_model:
            return 0.5
        
        try:
            embeddings = self.sbert_model.encode([source_text, target_text])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return max(0.0, float(similarity))
        except Exception as e:
            logger.warning(f"Context relevance calculation failed: {e}")
            return 0.5
    
    def generate_improved_explications(self, text: str, language: str) -> List[str]:
        """Generate improved, more specific explications."""
        improved_explications = []
        
        try:
            # Get primitives
            primitives = self.nsm_translator.detect_primitives_in_text(text, language)
            
            # Generate context-specific explications
            for primitive in primitives:
                if primitive == 'unknown':
                    # Try to infer primitive from context
                    inferred_primitive = self._infer_primitive_from_context(text, language)
                    if inferred_primitive:
                        primitive = inferred_primitive
                
                # Generate specific explication
                explication = self._generate_specific_explication(text, primitive, language)
                if explication:
                    improved_explications.append(explication)
            
            # If no specific explications, generate fallback
            if not improved_explications:
                fallback = self._generate_contextual_fallback(text, language)
                if fallback:
                    improved_explications.append(fallback)
        
        except Exception as e:
            logger.warning(f"Failed to generate improved explications: {e}")
        
        return improved_explications
    
    def _infer_primitive_from_context(self, text: str, language: str) -> Optional[str]:
        """Infer primitive from text context."""
        text_lower = text.lower()
        
        # Location patterns
        location_words = ['near', 'at', 'in', 'on', 'cerca de', 'en', 'sur', 'dans']
        if any(word in text_lower for word in location_words):
            return 'AtLocation'
        
        # Similarity patterns
        similarity_words = ['similar', 'like', 'parecido', 'semblable', 'como', 'comme']
        if any(word in text_lower for word in similarity_words):
            return 'SimilarTo'
        
        # Property patterns
        property_words = ['red', 'blue', 'big', 'small', 'rojo', 'azul', 'grande', 'petit']
        if any(word in text_lower for word in property_words):
            return 'HasProperty'
        
        # Action patterns
        action_words = ['works', 'plays', 'trabaja', 'juega', 'travaille', 'joue']
        if any(word in text_lower for word in action_words):
            return 'UsedFor'
        
        return None
    
    def _generate_specific_explication(self, text: str, primitive: str, language: str) -> Optional[str]:
        """Generate specific explication based on primitive and context."""
        try:
            # Get language-specific templates
            templates = self._get_language_templates(language)
            
            if primitive in templates:
                template = templates[primitive]
                
                # Extract context elements
                context_elements = self._extract_context_elements(text, primitive, language)
                
                # Fill template with context
                explication = template.format(**context_elements)
                return explication
            
        except Exception as e:
            logger.warning(f"Failed to generate specific explication: {e}")
        
        return None
    
    def _get_language_templates(self, language: str) -> Dict[str, str]:
        """Get language-specific explication templates."""
        templates = {
            'en': {
                'AtLocation': "{entity} is at {location}",
                'SimilarTo': "{entity} is similar to {target}",
                'HasProperty': "{entity} has property {property}",
                'UsedFor': "{entity} is used for {purpose}",
                'Contains': "{container} contains {content}"
            },
            'es': {
                'AtLocation': "{entity} está en {location}",
                'SimilarTo': "{entity} es similar a {target}",
                'HasProperty': "{entity} tiene la propiedad {property}",
                'UsedFor': "{entity} se usa para {purpose}",
                'Contains': "{container} contiene {content}"
            },
            'fr': {
                'AtLocation': "{entity} est à {location}",
                'SimilarTo': "{entity} est similaire à {target}",
                'HasProperty': "{entity} a la propriété {property}",
                'UsedFor': "{entity} est utilisé pour {purpose}",
                'Contains': "{container} contient {content}"
            }
        }
        
        return templates.get(language, templates['en'])
    
    def _extract_context_elements(self, text: str, primitive: str, language: str) -> Dict[str, str]:
        """Extract context elements for template filling."""
        elements = {
            'entity': 'thing',
            'location': 'place',
            'target': 'other thing',
            'property': 'property',
            'purpose': 'purpose',
            'container': 'container',
            'content': 'content'
        }
        
        # Simple extraction based on primitive type
        words = text.split()
        
        if primitive == 'AtLocation':
            # Extract entity and location
            if len(words) >= 4:
                elements['entity'] = ' '.join(words[:2])  # First two words as entity
                elements['location'] = ' '.join(words[-2:])  # Last two words as location
        
        elif primitive == 'HasProperty':
            # Extract entity and property
            if len(words) >= 3:
                elements['entity'] = words[1]  # Second word as entity
                elements['property'] = words[2]  # Third word as property
        
        elif primitive == 'SimilarTo':
            # Extract entity and target
            if len(words) >= 4:
                elements['entity'] = words[1]  # Second word as entity
                elements['target'] = words[-1]  # Last word as target
        
        return elements
    
    def _generate_contextual_fallback(self, text: str, language: str) -> Optional[str]:
        """Generate contextual fallback explication."""
        # Extract key words and create simple explication
        words = text.split()
        if len(words) >= 2:
            key_word = words[1]  # Second word is often the main entity
            
            if language == 'es':
                return f"{key_word} está en un lugar"
            elif language == 'fr':
                return f"{key_word} est dans un lieu"
            else:
                return f"{key_word} is in a place"
        
        return None
    
    def improve_surface_realization(self, explications: List[str], language: str) -> str:
        """Improve surface realization from explications."""
        if not explications:
            return ""
        
        # Simple surface realization improvement
        if len(explications) == 1:
            # Single explication - clean it up
            explication = explications[0]
            
            # Remove AND patterns
            if ' AND ' in explication:
                parts = explication.split(' AND ')
                # Take the most specific part
                explication = max(parts, key=len)
            
            # Clean up generic phrases
            generic_replacements = {
                'esta cosa es así': 'está aquí',
                'cette chose est comme cela': 'est là',
                'this thing is like this': 'is here'
            }
            
            for generic, replacement in generic_replacements.items():
                explication = explication.replace(generic, replacement)
            
            return explication
        
        else:
            # Multiple explications - combine intelligently
            # Remove duplicates and generic phrases
            cleaned_explications = []
            for explication in explications:
                if ' AND ' not in explication and not any(generic in explication.lower() 
                    for generic in ['esta cosa es así', 'cette chose est comme cela', 'this thing is like this']):
                    cleaned_explications.append(explication)
            
            if cleaned_explications:
                return cleaned_explications[0]  # Take the first good one
            else:
                return explications[0]  # Fallback to first
    
    def run_quality_improvement(self, test_texts: List[str], source_lang: str = "en", 
                              target_langs: List[str] = ["es", "fr"]) -> Dict[str, Any]:
        """Run comprehensive quality improvement analysis."""
        logger.info(f"Running NSM translation quality improvement for {len(test_texts)} texts")
        
        improvement_results = {
            'test_configuration': {
                'source_language': source_lang,
                'target_languages': target_langs,
                'num_test_texts': len(test_texts),
                'timestamp': time.time()
            },
            'improvement_analysis': {},
            'recommendations': [],
            'quality_metrics': {}
        }
        
        all_analyses = []
        
        for target_lang in target_langs:
            lang_analyses = []
            
            for text in test_texts:
                # Get current NSM translation
                try:
                    current_result = self.nsm_translator.translate_via_explications(text, source_lang, target_lang)
                    current_translation = current_result.get('target_text', text)
                except Exception as e:
                    logger.warning(f"Current NSM translation failed for '{text}': {e}")
                    current_translation = text
                
                # Analyze issues
                analysis = self.analyze_translation_issues(text, current_translation, source_lang, target_lang)
                
                # Generate improved explications
                improved_explications = self.generate_improved_explications(text, target_lang)
                
                # Improve surface realization
                improved_translation = self.improve_surface_realization(improved_explications, target_lang)
                
                # Analyze improved version
                improved_analysis = self.analyze_translation_issues(text, improved_translation, source_lang, target_lang)
                
                # Compare results
                comparison = {
                    'source_text': text,
                    'current_translation': current_translation,
                    'improved_translation': improved_translation,
                    'current_analysis': analysis,
                    'improved_analysis': improved_analysis,
                    'improvement': {
                        'fluency_improvement': improved_analysis['quality_metrics']['surface_fluency'] - 
                                             analysis['quality_metrics']['surface_fluency'],
                        'specificity_improvement': improved_analysis['quality_metrics']['explication_specificity'] - 
                                                 analysis['quality_metrics']['explication_specificity'],
                        'context_improvement': improved_analysis['quality_metrics']['context_relevance'] - 
                                             analysis['quality_metrics']['context_relevance']
                    }
                }
                
                lang_analyses.append(comparison)
                all_analyses.append(comparison)
            
            improvement_results['improvement_analysis'][target_lang] = lang_analyses
        
        # Generate overall metrics
        improvement_results['quality_metrics'] = self._generate_overall_metrics(all_analyses)
        
        # Generate recommendations
        improvement_results['recommendations'] = self._generate_improvement_recommendations(all_analyses)
        
        return improvement_results
    
    def _generate_overall_metrics(self, all_analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Generate overall quality improvement metrics."""
        metrics = {
            'avg_fluency_improvement': 0.0,
            'avg_specificity_improvement': 0.0,
            'avg_context_improvement': 0.0,
            'total_improvements': 0,
            'significant_improvements': 0
        }
        
        fluency_improvements = []
        specificity_improvements = []
        context_improvements = []
        
        for analysis in all_analyses:
            improvement = analysis['improvement']
            
            fluency_improvements.append(improvement['fluency_improvement'])
            specificity_improvements.append(improvement['specificity_improvement'])
            context_improvements.append(improvement['context_improvement'])
            
            # Count significant improvements
            total_improvement = (improvement['fluency_improvement'] + 
                               improvement['specificity_improvement'] + 
                               improvement['context_improvement'])
            
            if total_improvement > 0.1:  # 10% improvement threshold
                metrics['significant_improvements'] += 1
        
        if fluency_improvements:
            metrics['avg_fluency_improvement'] = np.mean(fluency_improvements)
        if specificity_improvements:
            metrics['avg_specificity_improvement'] = np.mean(specificity_improvements)
        if context_improvements:
            metrics['avg_context_improvement'] = np.mean(context_improvements)
        
        metrics['total_improvements'] = len(all_analyses)
        
        return metrics
    
    def _generate_improvement_recommendations(self, all_analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on improvement analysis."""
        recommendations = []
        
        # Analyze common issues
        fluency_issues = 0
        specificity_issues = 0
        context_issues = 0
        
        for analysis in all_analyses:
            current_analysis = analysis['current_analysis']
            
            if current_analysis['quality_metrics']['surface_fluency'] < 0.5:
                fluency_issues += 1
            if current_analysis['quality_metrics']['explication_specificity'] < 0.5:
                specificity_issues += 1
            if current_analysis['quality_metrics']['context_relevance'] < 0.5:
                context_issues += 1
        
        total_analyses = len(all_analyses)
        
        if fluency_issues > total_analyses * 0.5:
            recommendations.append("High fluency issues detected - focus on improving surface realization templates")
        
        if specificity_issues > total_analyses * 0.5:
            recommendations.append("High specificity issues detected - enhance explication generation with better context extraction")
        
        if context_issues > total_analyses * 0.5:
            recommendations.append("High context relevance issues detected - improve semantic preservation in translation pipeline")
        
        # Check for improvements
        avg_improvement = (np.mean([a['improvement']['fluency_improvement'] for a in all_analyses]) +
                         np.mean([a['improvement']['specificity_improvement'] for a in all_analyses]) +
                         np.mean([a['improvement']['context_improvement'] for a in all_analyses])) / 3
        
        if avg_improvement > 0.1:
            recommendations.append(f"Significant improvement achieved ({avg_improvement:.1%}) - continue with current approach")
        else:
            recommendations.append("Limited improvement achieved - consider more fundamental changes to translation pipeline")
        
        return recommendations


def main():
    """Main function to run NSM translation quality improvement."""
    logger.info("Starting NSM translation quality improvement...")
    
    # Initialize improver
    improver = NSMTranslationQualityImprover()
    
    # Test texts (same as sanity checks for comparison)
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
    
    # Run quality improvement
    improvement_results = improver.run_quality_improvement(test_texts, "en", ["es", "fr"])
    
    # Print results
    print("\n" + "="*80)
    print("NSM TRANSLATION QUALITY IMPROVEMENT RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Source Language: {improvement_results['test_configuration']['source_language']}")
    print(f"  Target Languages: {improvement_results['test_configuration']['target_languages']}")
    print(f"  Number of Test Texts: {improvement_results['test_configuration']['num_test_texts']}")
    
    print(f"\nQuality Improvement Metrics:")
    metrics = improvement_results['quality_metrics']
    print(f"  Average Fluency Improvement: {metrics['avg_fluency_improvement']:+.3f}")
    print(f"  Average Specificity Improvement: {metrics['avg_specificity_improvement']:+.3f}")
    print(f"  Average Context Improvement: {metrics['avg_context_improvement']:+.3f}")
    print(f"  Significant Improvements: {metrics['significant_improvements']}/{metrics['total_improvements']}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(improvement_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Show detailed results for each language
    for lang, analyses in improvement_results['improvement_analysis'].items():
        print(f"\n{lang.upper()} Language Improvements:")
        
        # Show some example improvements
        for i, analysis in enumerate(analyses[:3]):  # Show first 3
            print(f"\n  Example {i+1}:")
            print(f"    Source: {analysis['source_text']}")
            print(f"    Current: {analysis['current_translation']}")
            print(f"    Improved: {analysis['improved_translation']}")
            
            improvement = analysis['improvement']
            print(f"    Fluency: {improvement['fluency_improvement']:+.3f}")
            print(f"    Specificity: {improvement['specificity_improvement']:+.3f}")
            print(f"    Context: {improvement['context_improvement']:+.3f}")
    
    # Save results
    output_path = "data/nsm_translation_quality_improvement_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(improvement_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"NSM translation quality improvement report saved to: {output_path}")
    
    print("="*80)
    print("NSM translation quality improvement completed!")
    print("="*80)


if __name__ == "__main__":
    main()

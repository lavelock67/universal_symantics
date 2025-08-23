#!/usr/bin/env python3
"""
Primitive Detection Enhancement System.

This script implements a genuine primitive detection enhancement system to honestly
improve the low detection rate (34.7%) by analyzing failures and implementing real improvements.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import time
from collections import defaultdict, Counter
import re

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
        converted_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
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


class PrimitiveDetectionAnalyzer:
    """Analyzer for primitive detection failures and patterns."""
    
    def __init__(self):
        """Initialize the primitive detection analyzer."""
        self.nsm_translator = NSMTranslator()
        self.sbert_model = None
        
        # Analysis configuration
        self.analysis_config = {
            'min_confidence_threshold': 0.3,
            'enable_semantic_analysis': True,
            'enable_pattern_analysis': True,
            'enable_failure_categorization': True
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic analysis."""
        try:
            logger.info("Loading SBERT model for primitive detection analysis...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def analyze_detection_failures(self, test_texts: List[str], languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Analyze primitive detection failures to understand root causes."""
        logger.info("Analyzing primitive detection failures...")
        
        analysis_results = {
            'total_texts': len(test_texts) * len(languages),
            'successful_detections': 0,
            'failed_detections': 0,
            'failure_categories': defaultdict(int),
            'failure_patterns': defaultdict(list),
            'semantic_analysis': {},
            'recommendations': []
        }
        
        for language in languages:
            for text in test_texts:
                try:
                    # Attempt primitive detection
                    primitives = self.nsm_translator.detect_primitives_in_text(text, language)
                    
                    if primitives:
                        analysis_results['successful_detections'] += 1
                    else:
                        analysis_results['failed_detections'] += 1
                        
                        # Analyze failure
                        failure_analysis = self._analyze_single_failure(text, language)
                        analysis_results['failure_categories'][failure_analysis['category']] += 1
                        analysis_results['failure_patterns'][failure_analysis['pattern']].append({
                            'text': text,
                            'language': language,
                            'analysis': failure_analysis
                        })
                
                except Exception as e:
                    analysis_results['failed_detections'] += 1
                    analysis_results['failure_categories']['exception'] += 1
                    logger.warning(f"Detection failed for '{text}' in {language}: {e}")
        
        # Perform semantic analysis
        if self.analysis_config['enable_semantic_analysis']:
            analysis_results['semantic_analysis'] = self._perform_semantic_analysis(test_texts, languages)
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_failure_recommendations(analysis_results)
        
        return analysis_results
    
    def _analyze_single_failure(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze a single detection failure."""
        analysis = {
            'category': 'unknown',
            'pattern': 'unknown',
            'potential_primitives': [],
            'confidence': 0.0,
            'suggestions': []
        }
        
        try:
            # Analyze text characteristics
            words = text.lower().split()
            word_count = len(words)
            
            # Check for common patterns that might indicate primitive presence
            if any(word in ['not', 'no', 'never', 'none'] for word in words):
                analysis['category'] = 'negation_missing'
                analysis['pattern'] = 'negation_detection'
                analysis['potential_primitives'] = ['NOT', 'NEG']
                analysis['suggestions'].append('Add negation primitive detection')
            
            elif any(word in ['can', 'could', 'might', 'must', 'should', 'will'] for word in words):
                analysis['category'] = 'modality_missing'
                analysis['pattern'] = 'modality_detection'
                analysis['potential_primitives'] = ['CAN', 'MUST', 'SHOULD', 'MIGHT', 'WILL']
                analysis['suggestions'].append('Add modality primitive detection')
            
            elif any(word in ['all', 'some', 'many', 'few', 'most'] for word in words):
                analysis['category'] = 'quantifier_missing'
                analysis['pattern'] = 'quantifier_detection'
                analysis['potential_primitives'] = ['ALL', 'SOME', 'MANY', 'FEW', 'MOST']
                analysis['suggestions'].append('Add quantifier primitive detection')
            
            elif any(word in ['like', 'hate', 'enjoy', 'prefer', 'feel'] for word in words):
                analysis['category'] = 'experiencer_missing'
                analysis['pattern'] = 'experiencer_detection'
                analysis['potential_primitives'] = ['LIKE', 'HATE', 'ENJOY', 'PREFER', 'FEEL']
                analysis['suggestions'].append('Add experiencer primitive detection')
            
            elif any(word in ['almost', 'just', 'still', 'already', 'yet'] for word in words):
                analysis['category'] = 'aspectual_missing'
                analysis['pattern'] = 'aspectual_detection'
                analysis['potential_primitives'] = ['ALMOST', 'JUST', 'STILL', 'ALREADY', 'YET']
                analysis['suggestions'].append('Add aspectual primitive detection')
            
            elif any(word in ['cause', 'because', 'result', 'lead'] for word in words):
                analysis['category'] = 'causation_missing'
                analysis['pattern'] = 'causation_detection'
                analysis['potential_primitives'] = ['CAUSE', 'BECAUSE', 'RESULT']
                analysis['suggestions'].append('Add causation primitive detection')
            
            elif word_count < 3:
                analysis['category'] = 'too_short'
                analysis['pattern'] = 'length_constraint'
                analysis['suggestions'].append('Consider minimum text length requirements')
            
            elif word_count > 20:
                analysis['category'] = 'too_complex'
                analysis['pattern'] = 'complexity_constraint'
                analysis['suggestions'].append('Consider text complexity limits')
            
            else:
                analysis['category'] = 'general_failure'
                analysis['pattern'] = 'general_detection'
                analysis['suggestions'].append('Review general primitive detection logic')
            
            # Calculate confidence based on pattern match
            if analysis['potential_primitives']:
                analysis['confidence'] = 0.7
            else:
                analysis['confidence'] = 0.3
        
        except Exception as e:
            analysis['category'] = 'analysis_error'
            analysis['pattern'] = 'error'
            analysis['suggestions'].append(f'Fix analysis error: {str(e)}')
        
        return analysis
    
    def _perform_semantic_analysis(self, test_texts: List[str], languages: List[str]) -> Dict[str, Any]:
        """Perform semantic analysis of detection patterns."""
        if not self.sbert_model:
            return {'error': 'SBERT model not available'}
        
        try:
            semantic_analysis = {
                'text_embeddings': [],
                'semantic_clusters': {},
                'similarity_patterns': []
            }
            
            # Generate embeddings for all texts
            embeddings = self.sbert_model.encode(test_texts)
            semantic_analysis['text_embeddings'] = embeddings.tolist()
            
            # Analyze semantic similarity patterns
            for i, text1 in enumerate(test_texts):
                for j, text2 in enumerate(test_texts[i+1:], i+1):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    
                    if similarity > 0.8:  # High similarity threshold
                        semantic_analysis['similarity_patterns'].append({
                            'text1': text1,
                            'text2': text2,
                            'similarity': float(similarity)
                        })
            
            return semantic_analysis
        
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_failure_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate honest recommendations based on failure analysis."""
        recommendations = []
        
        # Calculate failure rate
        total = analysis_results['total_texts']
        failure_rate = analysis_results['failed_detections'] / total if total > 0 else 0
        
        recommendations.append(f"Current failure rate: {failure_rate:.1%} - significant improvement needed")
        
        # Analyze failure categories
        failure_categories = analysis_results['failure_categories']
        
        if failure_categories.get('negation_missing', 0) > 0:
            recommendations.append(f"Negation detection missing in {failure_categories['negation_missing']} cases - implement negation primitive detection")
        
        if failure_categories.get('modality_missing', 0) > 0:
            recommendations.append(f"Modality detection missing in {failure_categories['modality_missing']} cases - implement modality primitive detection")
        
        if failure_categories.get('quantifier_missing', 0) > 0:
            recommendations.append(f"Quantifier detection missing in {failure_categories['quantifier_missing']} cases - implement quantifier primitive detection")
        
        if failure_categories.get('experiencer_missing', 0) > 0:
            recommendations.append(f"Experiencer detection missing in {failure_categories['experiencer_missing']} cases - implement experiencer primitive detection")
        
        if failure_categories.get('aspectual_missing', 0) > 0:
            recommendations.append(f"Aspectual detection missing in {failure_categories['aspectual_missing']} cases - implement aspectual primitive detection")
        
        if failure_categories.get('causation_missing', 0) > 0:
            recommendations.append(f"Causation detection missing in {failure_categories['causation_missing']} cases - implement causation primitive detection")
        
        if failure_categories.get('too_short', 0) > 0:
            recommendations.append(f"Text too short in {failure_categories['too_short']} cases - review minimum length requirements")
        
        if failure_categories.get('too_complex', 0) > 0:
            recommendations.append(f"Text too complex in {failure_categories['too_complex']} cases - review complexity handling")
        
        if failure_categories.get('exception', 0) > 0:
            recommendations.append(f"Exceptions in {failure_categories['exception']} cases - improve error handling")
        
        # Overall recommendations
        if failure_rate > 0.5:
            recommendations.append("High failure rate indicates fundamental issues with primitive detection approach")
        
        if len(failure_categories) > 5:
            recommendations.append("Multiple failure categories suggest need for comprehensive primitive detection overhaul")
        
        return recommendations


class PrimitiveDetectionEnhancer:
    """Enhancer for improving primitive detection capabilities."""
    
    def __init__(self):
        """Initialize the primitive detection enhancer."""
        self.analyzer = PrimitiveDetectionAnalyzer()
        self.nsm_translator = NSMTranslator()
        
        # Enhancement configuration
        self.enhancement_config = {
            'enable_negation_detection': True,
            'enable_modality_detection': True,
            'enable_quantifier_detection': True,
            'enable_experiencer_detection': True,
            'enable_aspectual_detection': True,
            'enable_causation_detection': True,
            'confidence_threshold': 0.3
        }
        
        # Enhanced detection patterns
        self.enhanced_patterns = self._load_enhanced_patterns()
    
    def _load_enhanced_patterns(self) -> Dict[str, List[str]]:
        """Load enhanced detection patterns."""
        return {
            'negation': [
                r'\b(not|no|never|none|nobody|nothing|nowhere|neither|nor)\b',
                r'\b(doesn\'t|don\'t|didn\'t|won\'t|can\'t|couldn\'t|shouldn\'t|wouldn\'t)\b',
                r'\b(isn\'t|aren\'t|wasn\'t|weren\'t|hasn\'t|haven\'t|hadn\'t)\b'
            ],
            'modality': [
                r'\b(can|could|might|may|must|should|will|would|shall)\b',
                r'\b(able to|capable of|likely to|supposed to|going to)\b'
            ],
            'quantifier': [
                r'\b(all|some|many|few|most|several|various|numerous|countless)\b',
                r'\b(each|every|any|either|neither|both|none)\b'
            ],
            'experiencer': [
                r'\b(like|love|hate|enjoy|prefer|feel|think|believe|know)\b',
                r'\b(want|need|desire|hope|wish|fear|worry|care)\b'
            ],
            'aspectual': [
                r'\b(almost|just|still|already|yet|finally|eventually|gradually)\b',
                r'\b(start|begin|continue|stop|finish|complete|end)\b'
            ],
            'causation': [
                r'\b(cause|caused|because|since|as|due to|result in|lead to)\b',
                r'\b(make|force|compel|require|necessitate|trigger)\b'
            ]
        }
    
    def enhance_primitive_detection(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Enhance primitive detection for a given text."""
        try:
            # Get original detection
            original_primitives = self.nsm_translator.detect_primitives_in_text(text, language)
            
            # Apply enhanced detection
            enhanced_primitives = self._apply_enhanced_detection(text, language)
            
            # Combine results
            combined_primitives = list(set(original_primitives + enhanced_primitives))
            
            return {
                'original_primitives': original_primitives,
                'enhanced_primitives': enhanced_primitives,
                'combined_primitives': combined_primitives,
                'improvement': len(enhanced_primitives) > 0,
                'detection_confidence': self._calculate_detection_confidence(text, combined_primitives)
            }
        
        except Exception as e:
            logger.warning(f"Enhanced detection failed for '{text}': {e}")
            return {
                'original_primitives': [],
                'enhanced_primitives': [],
                'combined_primitives': [],
                'improvement': False,
                'error': str(e)
            }
    
    def _apply_enhanced_detection(self, text: str, language: str) -> List[str]:
        """Apply enhanced detection patterns."""
        enhanced_primitives = []
        text_lower = text.lower()
        
        # Check each pattern category
        for category, patterns in self.enhanced_patterns.items():
            if self._should_apply_category(category):
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        primitive = self._map_pattern_to_primitive(category, pattern)
                        if primitive:
                            enhanced_primitives.append(primitive)
        
        return enhanced_primitives
    
    def _should_apply_category(self, category: str) -> bool:
        """Check if a category should be applied based on configuration."""
        config_key = f'enable_{category}_detection'
        return self.enhancement_config.get(config_key, True)
    
    def _map_pattern_to_primitive(self, category: str, pattern: str) -> Optional[str]:
        """Map a detected pattern to a primitive."""
        mapping = {
            'negation': {
                r'\b(not|no|never|none)\b': 'NOT',
                r'\b(doesn\'t|don\'t|didn\'t)\b': 'NOT',
                r'\b(isn\'t|aren\'t|wasn\'t)\b': 'NOT'
            },
            'modality': {
                r'\b(can|could)\b': 'CAN',
                r'\b(must|should)\b': 'MUST',
                r'\b(might|may)\b': 'MIGHT',
                r'\b(will|would)\b': 'WILL'
            },
            'quantifier': {
                r'\b(all)\b': 'ALL',
                r'\b(some|several)\b': 'SOME',
                r'\b(many|numerous)\b': 'MANY',
                r'\b(few)\b': 'FEW',
                r'\b(most)\b': 'MOST'
            },
            'experiencer': {
                r'\b(like|love)\b': 'LIKE',
                r'\b(hate)\b': 'HATE',
                r'\b(enjoy)\b': 'ENJOY',
                r'\b(prefer)\b': 'PREFER',
                r'\b(feel)\b': 'FEEL'
            },
            'aspectual': {
                r'\b(almost)\b': 'ALMOST',
                r'\b(just)\b': 'JUST',
                r'\b(still)\b': 'STILL',
                r'\b(already)\b': 'ALREADY',
                r'\b(yet)\b': 'YET'
            },
            'causation': {
                r'\b(cause|caused)\b': 'CAUSE',
                r'\b(because|since)\b': 'BECAUSE',
                r'\b(result in|lead to)\b': 'RESULT'
            }
        }
        
        category_mapping = mapping.get(category, {})
        return category_mapping.get(pattern)
    
    def _calculate_detection_confidence(self, text: str, primitives: List[str]) -> float:
        """Calculate confidence in primitive detection."""
        if not primitives:
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Boost confidence based on primitive count
        if len(primitives) >= 3:
            confidence += 0.2
        elif len(primitives) >= 2:
            confidence += 0.1
        
        # Boost confidence based on text length appropriateness
        word_count = len(text.split())
        if 3 <= word_count <= 15:
            confidence += 0.1
        
        # Reduce confidence for very short or very long texts
        if word_count < 2:
            confidence -= 0.2
        elif word_count > 25:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))


class PrimitiveDetectionEnhancementSystem:
    """Comprehensive primitive detection enhancement system."""
    
    def __init__(self):
        """Initialize the enhancement system."""
        self.analyzer = PrimitiveDetectionAnalyzer()
        self.enhancer = PrimitiveDetectionEnhancer()
        
        # System configuration
        self.system_config = {
            'enable_analysis': True,
            'enable_enhancement': True,
            'enable_comparison': True,
            'test_texts_per_language': 50
        }
    
    def run_enhancement_analysis(self, test_texts: List[str], languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Run comprehensive primitive detection enhancement analysis."""
        logger.info(f"Running primitive detection enhancement analysis for {len(test_texts)} texts")
        
        analysis_results = {
            'test_configuration': {
                'num_test_texts': len(test_texts),
                'languages': languages,
                'timestamp': time.time()
            },
            'failure_analysis': {},
            'enhancement_results': [],
            'comparison_analysis': {},
            'recommendations': []
        }
        
        # Analyze failures
        if self.system_config['enable_analysis']:
            logger.info("Analyzing detection failures...")
            analysis_results['failure_analysis'] = self.analyzer.analyze_detection_failures(test_texts, languages)
        
        # Apply enhancements
        if self.system_config['enable_enhancement']:
            logger.info("Applying detection enhancements...")
            for language in languages:
                for text in test_texts:
                    enhancement_result = self.enhancer.enhance_primitive_detection(text, language)
                    enhancement_result['text'] = text
                    enhancement_result['language'] = language
                    analysis_results['enhancement_results'].append(enhancement_result)
        
        # Compare results
        if self.system_config['enable_comparison']:
            logger.info("Comparing original vs enhanced detection...")
            analysis_results['comparison_analysis'] = self._compare_detection_results(
                analysis_results['enhancement_results']
            )
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_enhancement_recommendations(
            analysis_results
        )
        
        return analysis_results
    
    def _compare_detection_results(self, enhancement_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare original vs enhanced detection results."""
        comparison = {
            'total_texts': len(enhancement_results),
            'original_detections': 0,
            'enhanced_detections': 0,
            'improvements': 0,
            'detection_rates': {
                'original': 0.0,
                'enhanced': 0.0,
                'improvement': 0.0
            },
            'confidence_analysis': {
                'original_avg': 0.0,
                'enhanced_avg': 0.0,
                'confidence_improvement': 0.0
            }
        }
        
        original_confidences = []
        enhanced_confidences = []
        
        for result in enhancement_results:
            # Count detections
            if result.get('original_primitives'):
                comparison['original_detections'] += 1
            
            if result.get('combined_primitives'):
                comparison['enhanced_detections'] += 1
            
            if result.get('improvement', False):
                comparison['improvements'] += 1
            
            # Track confidence
            original_confidence = len(result.get('original_primitives', [])) / 10.0  # Simple confidence
            enhanced_confidence = result.get('detection_confidence', 0.0)
            
            original_confidences.append(original_confidence)
            enhanced_confidences.append(enhanced_confidence)
        
        # Calculate rates
        total = comparison['total_texts']
        if total > 0:
            comparison['detection_rates']['original'] = comparison['original_detections'] / total
            comparison['detection_rates']['enhanced'] = comparison['enhanced_detections'] / total
            comparison['detection_rates']['improvement'] = comparison['improvements'] / total
        
        # Calculate confidence averages
        if original_confidences:
            comparison['confidence_analysis']['original_avg'] = np.mean(original_confidences)
        if enhanced_confidences:
            comparison['confidence_analysis']['enhanced_avg'] = np.mean(enhanced_confidences)
        
        comparison['confidence_analysis']['confidence_improvement'] = (
            comparison['confidence_analysis']['enhanced_avg'] - 
            comparison['confidence_analysis']['original_avg']
        )
        
        return comparison
    
    def _generate_enhancement_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate honest recommendations for enhancement."""
        recommendations = []
        
        # Get comparison data
        comparison = analysis_results.get('comparison_analysis', {})
        failure_analysis = analysis_results.get('failure_analysis', {})
        
        # Detection rate recommendations
        original_rate = comparison.get('detection_rates', {}).get('original', 0.0)
        enhanced_rate = comparison.get('detection_rates', {}).get('enhanced', 0.0)
        improvement_rate = comparison.get('detection_rates', {}).get('improvement', 0.0)
        
        recommendations.append(f"Original detection rate: {original_rate:.1%}")
        recommendations.append(f"Enhanced detection rate: {enhanced_rate:.1%}")
        recommendations.append(f"Improvement rate: {improvement_rate:.1%}")
        
        if enhanced_rate < 0.8:
            recommendations.append("Enhanced detection rate still below 80% - significant work needed")
        
        if improvement_rate < 0.1:
            recommendations.append("Low improvement rate suggests enhancement approach needs refinement")
        
        # Confidence recommendations
        confidence_improvement = comparison.get('confidence_analysis', {}).get('confidence_improvement', 0.0)
        if confidence_improvement < 0.1:
            recommendations.append("Minimal confidence improvement - review enhancement quality")
        
        # Failure analysis recommendations
        failure_recommendations = failure_analysis.get('recommendations', [])
        recommendations.extend(failure_recommendations)
        
        # Overall assessment
        if enhanced_rate < 0.5:
            recommendations.append("CRITICAL: Detection rate below 50% - fundamental approach review required")
        elif enhanced_rate < 0.7:
            recommendations.append("IMPORTANT: Detection rate below 70% - substantial improvements needed")
        else:
            recommendations.append("Detection rate above 70% - good progress, focus on remaining edge cases")
        
        return recommendations


def main():
    """Main function to run primitive detection enhancement analysis."""
    logger.info("Starting primitive detection enhancement analysis...")
    
    # Initialize enhancement system
    enhancement_system = PrimitiveDetectionEnhancementSystem()
    
    # Test texts (focusing on texts that likely failed detection)
    test_texts = [
        # Negation examples
        "The cat is not on the mat",
        "I do not like this weather",
        "She does not work here",
        "The book does not contain that information",
        "Children do not play here",
        
        # Modality examples
        "The cat might be on the mat",
        "The weather could be cold",
        "She should work here",
        "The book must contain this",
        "Children can play here",
        
        # Quantifier examples
        "All cats are on mats",
        "Some weather is cold",
        "Many people work here",
        "Few books contain this",
        "Most children play here",
        
        # Experiencer examples
        "I like this weather",
        "She enjoys her work",
        "They love this book",
        "Children hate homework",
        "We prefer this option",
        
        # Aspectual examples
        "I almost finished the work",
        "She just arrived",
        "They still work here",
        "We already ate",
        "He has not yet arrived",
        
        # Causation examples
        "The rain caused the flood",
        "Eating caused the sickness",
        "The fire caused the damage",
        "The accident caused the injury",
        "The storm caused the power outage",
        
        # Complex examples
        "Could you please help me?",
        "Would you mind closing the door?",
        "I'm sorry for the delay",
        "Thank you for your help",
        "Please take a seat",
        
        # Short examples
        "Yes",
        "No",
        "Maybe",
        "Stop",
        "Go",
        
        # Long examples
        "The complex interaction between multiple factors in the system caused a cascade of unexpected events that ultimately led to the complete failure of the entire infrastructure",
        "Despite the numerous attempts to resolve the issue through various methodologies and approaches, the problem persisted and continued to manifest in different forms across multiple components",
        "The comprehensive analysis of the data revealed significant patterns and correlations that were previously unknown and provided valuable insights into the underlying mechanisms",
        
        # Mixed examples
        "I can't believe you didn't tell me about this",
        "She might have been able to help if we had asked earlier",
        "All of the students should have completed their assignments by now",
        "Most people would probably enjoy this movie",
        "The weather could potentially cause problems for the event"
    ]
    
    # Run enhancement analysis
    analysis_results = enhancement_system.run_enhancement_analysis(test_texts, ["en"])
    
    # Print results
    print("\n" + "="*80)
    print("PRIMITIVE DETECTION ENHANCEMENT ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Texts: {analysis_results['test_configuration']['num_test_texts']}")
    print(f"  Languages: {analysis_results['test_configuration']['languages']}")
    
    print(f"\nFailure Analysis:")
    failure_analysis = analysis_results['failure_analysis']
    print(f"  Total Texts: {failure_analysis['total_texts']}")
    print(f"  Successful Detections: {failure_analysis['successful_detections']}")
    print(f"  Failed Detections: {failure_analysis['failed_detections']}")
    print(f"  Failure Rate: {failure_analysis['failed_detections']/failure_analysis['total_texts']:.1%}")
    
    print(f"\nFailure Categories:")
    for category, count in sorted(failure_analysis['failure_categories'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")
    
    print(f"\nComparison Analysis:")
    comparison = analysis_results['comparison_analysis']
    print(f"  Original Detection Rate: {comparison['detection_rates']['original']:.1%}")
    print(f"  Enhanced Detection Rate: {comparison['detection_rates']['enhanced']:.1%}")
    print(f"  Improvement Rate: {comparison['detection_rates']['improvement']:.1%}")
    print(f"  Confidence Improvement: {comparison['confidence_analysis']['confidence_improvement']:.3f}")
    
    print(f"\nExample Enhancement Results:")
    for i, result in enumerate(analysis_results['enhancement_results'][:5]):
        text = result['text']
        original = result.get('original_primitives', [])
        enhanced = result.get('enhanced_primitives', [])
        combined = result.get('combined_primitives', [])
        improvement = result.get('improvement', False)
        
        print(f"  {i+1}. Text: {text}")
        print(f"     Original: {original}")
        print(f"     Enhanced: {enhanced}")
        print(f"     Combined: {combined}")
        print(f"     Improvement: {improvement}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_path = "data/primitive_detection_enhancement_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(analysis_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Primitive detection enhancement report saved to: {output_path}")
    
    print("="*80)
    print("Primitive detection enhancement analysis completed!")
    print("="*80)


if __name__ == "__main__":
    main()

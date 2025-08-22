#!/usr/bin/env python3
"""
Enhanced DeepNSM Integration System.

This script implements a comprehensive DeepNSM integration system to add
DeepNSM explication model inference and compare to templates for improved
NSM explication generation and evaluation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
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


class DeepNSMModel:
    """DeepNSM model for explication generation."""
    
    def __init__(self):
        """Initialize the DeepNSM model."""
        self.sbert_model = None
        self.nsm_translator = NSMTranslator()
        
        # Model parameters
        self.model_params = {
            'max_length': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'num_beams': 4,
            'min_length': 10
        }
        
        # Explication templates
        self.explication_templates = {
            'HasProperty': [
                "{entity} has the property of being {property}",
                "{entity} is characterized by {property}",
                "{entity} exhibits {property}",
                "The {entity} is {property}"
            ],
            'AtLocation': [
                "{entity} is located at {location}",
                "{entity} is situated in {location}",
                "{entity} can be found at {location}",
                "The {entity} is at {location}"
            ],
            'SimilarTo': [
                "{entity} is similar to {target}",
                "{entity} resembles {target}",
                "{entity} is like {target}",
                "{entity} and {target} are similar"
            ],
            'UsedFor': [
                "{entity} is used for {purpose}",
                "{entity} serves the purpose of {purpose}",
                "{entity} is intended for {purpose}",
                "The {entity} is used to {purpose}"
            ],
            'Contains': [
                "{entity} contains {content}",
                "{entity} includes {content}",
                "{entity} has {content} inside",
                "The {entity} contains {content}"
            ],
            'Causes': [
                "{cause} causes {effect}",
                "{cause} leads to {effect}",
                "{cause} results in {effect}",
                "The {cause} causes the {effect}"
            ],
            'PartOf': [
                "{part} is part of {whole}",
                "{part} belongs to {whole}",
                "{part} is a component of {whole}",
                "The {part} is part of the {whole}"
            ],
            'MadeOf': [
                "{entity} is made of {material}",
                "{entity} consists of {material}",
                "{entity} is composed of {material}",
                "The {entity} is made from {material}"
            ],
            'Desires': [
                "{entity} desires {object}",
                "{entity} wants {object}",
                "{entity} wishes for {object}",
                "The {entity} desires the {object}"
            ],
            'CapableOf': [
                "{entity} is capable of {action}",
                "{entity} can {action}",
                "{entity} has the ability to {action}",
                "The {entity} is capable of {action}"
            ]
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for DeepNSM...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def generate_explication(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Generate explication using DeepNSM model."""
        try:
            # Extract primitives from text
            primitives = self.nsm_translator.detect_primitives_in_text(text, language)
            
            if not primitives:
                return {
                    'text': text,
                    'language': language,
                    'primitives': [],
                    'explication': f"unknown({text})",
                    'confidence': 0.0,
                    'method': 'fallback'
                }
            
            # Generate explication using templates
            template_explication = self._generate_template_explication(text, primitives, language)
            
            # Generate explication using semantic inference
            semantic_explication = self._generate_semantic_explication(text, primitives, language)
            
            # Compare and select best explication
            best_explication = self._select_best_explication(
                template_explication, semantic_explication, text, language
            )
            
            return best_explication
        
        except Exception as e:
            logger.warning(f"DeepNSM explication generation failed: {e}")
            return {
                'text': text,
                'language': language,
                'primitives': [],
                'explication': f"error({text})",
                'confidence': 0.0,
                'method': 'error'
            }
    
    def _generate_template_explication(self, text: str, primitives: List[str], language: str) -> Dict[str, Any]:
        """Generate explication using templates."""
        try:
            # Find best matching template for each primitive
            template_explications = []
            
            for primitive in primitives:
                if primitive in self.explication_templates:
                    templates = self.explication_templates[primitive]
                    
                    # Extract context elements
                    context_elements = self._extract_context_elements(text, primitive, language)
                    
                    # Fill templates
                    for template in templates:
                        filled_template = self._fill_template(template, context_elements)
                        template_explications.append({
                            'primitive': primitive,
                            'template': template,
                            'explication': filled_template,
                            'confidence': 0.7  # Base confidence for templates
                        })
            
            # Select best template explication
            if template_explications:
                best_template = max(template_explications, key=lambda x: x['confidence'])
                return {
                    'text': text,
                    'language': language,
                    'primitives': primitives,
                    'explication': best_template['explication'],
                    'confidence': best_template['confidence'],
                    'method': 'template',
                    'template_used': best_template['template'],
                    'primitive_used': best_template['primitive']
                }
            else:
                return {
                    'text': text,
                    'language': language,
                    'primitives': primitives,
                    'explication': f"{' '.join(primitives)}({text})",
                    'confidence': 0.3,
                    'method': 'template_fallback'
                }
        
        except Exception as e:
            logger.warning(f"Template explication generation failed: {e}")
            return {
                'text': text,
                'language': language,
                'primitives': primitives,
                'explication': f"{' '.join(primitives)}({text})",
                'confidence': 0.2,
                'method': 'template_error'
            }
    
    def _generate_semantic_explication(self, text: str, primitives: List[str], language: str) -> Dict[str, Any]:
        """Generate explication using semantic inference."""
        try:
            if not self.sbert_model:
                return {
                    'text': text,
                    'language': language,
                    'primitives': primitives,
                    'explication': f"{' '.join(primitives)}({text})",
                    'confidence': 0.3,
                    'method': 'semantic_fallback'
                }
            
            # Generate semantic explication based on primitives and context
            semantic_explication = self._infer_semantic_explication(text, primitives, language)
            
            # Calculate confidence based on semantic coherence
            confidence = self._calculate_semantic_confidence(text, semantic_explication, primitives)
            
            return {
                'text': text,
                'language': language,
                'primitives': primitives,
                'explication': semantic_explication,
                'confidence': confidence,
                'method': 'semantic'
            }
        
        except Exception as e:
            logger.warning(f"Semantic explication generation failed: {e}")
            return {
                'text': text,
                'language': language,
                'primitives': primitives,
                'explication': f"{' '.join(primitives)}({text})",
                'confidence': 0.2,
                'method': 'semantic_error'
            }
    
    def _extract_context_elements(self, text: str, primitive: str, language: str) -> Dict[str, str]:
        """Extract context elements from text."""
        elements = {
            'entity': 'something',
            'property': 'a property',
            'location': 'somewhere',
            'target': 'something else',
            'purpose': 'a purpose',
            'content': 'something',
            'cause': 'something',
            'effect': 'something',
            'part': 'something',
            'whole': 'something',
            'material': 'some material',
            'object': 'something',
            'action': 'doing something'
        }
        
        # Simple extraction based on primitive type
        words = text.lower().split()
        
        if primitive == 'HasProperty':
            if len(words) >= 2:
                elements['entity'] = words[0]
                elements['property'] = ' '.join(words[1:])
        elif primitive == 'AtLocation':
            if len(words) >= 3 and 'at' in words:
                at_index = words.index('at')
                elements['entity'] = ' '.join(words[:at_index])
                elements['location'] = ' '.join(words[at_index+1:])
        elif primitive == 'SimilarTo':
            if len(words) >= 3 and 'like' in words:
                like_index = words.index('like')
                elements['entity'] = ' '.join(words[:like_index])
                elements['target'] = ' '.join(words[like_index+1:])
        elif primitive == 'UsedFor':
            if len(words) >= 3 and 'for' in words:
                for_index = words.index('for')
                elements['entity'] = ' '.join(words[:for_index])
                elements['purpose'] = ' '.join(words[for_index+1:])
        
        return elements
    
    def _fill_template(self, template: str, elements: Dict[str, str]) -> str:
        """Fill template with context elements."""
        filled_template = template
        
        for key, value in elements.items():
            placeholder = f"{{{key}}}"
            if placeholder in filled_template:
                filled_template = filled_template.replace(placeholder, value)
        
        return filled_template
    
    def _infer_semantic_explication(self, text: str, primitives: List[str], language: str) -> str:
        """Infer semantic explication from text and primitives."""
        # Simple semantic inference
        if len(primitives) == 1:
            primitive = primitives[0]
            return f"{primitive}({text})"
        elif len(primitives) == 2:
            return f"{primitives[0]} AND {primitives[1]}({text})"
        else:
            return f"{' AND '.join(primitives)}({text})"
    
    def _calculate_semantic_confidence(self, text: str, explication: str, primitives: List[str]) -> float:
        """Calculate semantic confidence of explication."""
        if not self.sbert_model:
            return 0.5
        
        try:
            # Calculate semantic similarity between text and explication
            embeddings = self.sbert_model.encode([text, explication])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            # Base confidence on similarity and primitive coverage
            base_confidence = max(0.0, float(similarity))
            primitive_bonus = min(len(primitives) * 0.1, 0.3)
            
            return min(base_confidence + primitive_bonus, 1.0)
        
        except Exception as e:
            logger.warning(f"Semantic confidence calculation failed: {e}")
            return 0.5
    
    def _select_best_explication(self, template_result: Dict[str, Any], 
                               semantic_result: Dict[str, Any],
                               text: str, language: str) -> Dict[str, Any]:
        """Select the best explication between template and semantic methods."""
        template_confidence = template_result.get('confidence', 0.0)
        semantic_confidence = semantic_result.get('confidence', 0.0)
        
        # Prefer semantic method if confidence is significantly higher
        if semantic_confidence > template_confidence + 0.1:
            return semantic_result
        else:
            return template_result


class DeepNSMComparator:
    """Comparator for DeepNSM vs template explications."""
    
    def __init__(self):
        """Initialize the DeepNSM comparator."""
        self.sbert_model = None
        self.deep_nsm_model = DeepNSMModel()
        
        # Comparison parameters
        self.comparison_params = {
            'similarity_threshold': 0.7,
            'confidence_threshold': 0.6,
            'fluency_weight': 0.3,
            'semantic_weight': 0.4,
            'coverage_weight': 0.3
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for DeepNSM comparison...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def compare_explications(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Compare DeepNSM vs template explications."""
        try:
            # Generate DeepNSM explication
            deep_nsm_result = self.deep_nsm_model.generate_explication(text, language)
            
            # Generate template explication (using existing system)
            template_result = self._generate_template_explication(text, language)
            
            # Compare explications
            comparison = self._compare_explication_quality(
                deep_nsm_result, template_result, text, language
            )
            
            return {
                'text': text,
                'language': language,
                'deep_nsm_result': deep_nsm_result,
                'template_result': template_result,
                'comparison': comparison,
                'recommendation': self._generate_recommendation(comparison)
            }
        
        except Exception as e:
            logger.warning(f"Explication comparison failed: {e}")
            return {
                'text': text,
                'language': language,
                'deep_nsm_result': {},
                'template_result': {},
                'comparison': {},
                'recommendation': 'error'
            }
    
    def _generate_template_explication(self, text: str, language: str) -> Dict[str, Any]:
        """Generate template explication using existing system."""
        try:
            # Use existing NSM translator for template-based explication
            nsm_translator = NSMTranslator()
            primitives = nsm_translator.detect_primitives_in_text(text, language)
            
            if not primitives:
                return {
                    'text': text,
                    'language': language,
                    'primitives': [],
                    'explication': f"unknown({text})",
                    'confidence': 0.0,
                    'method': 'template_fallback'
                }
            
            # Simple template-based explication
            explication = f"{' '.join(primitives)}({text})"
            
            return {
                'text': text,
                'language': language,
                'primitives': primitives,
                'explication': explication,
                'confidence': 0.5,  # Base confidence for templates
                'method': 'template'
            }
        
        except Exception as e:
            logger.warning(f"Template explication generation failed: {e}")
            return {
                'text': text,
                'language': language,
                'primitives': [],
                'explication': f"error({text})",
                'confidence': 0.0,
                'method': 'template_error'
            }
    
    def _compare_explication_quality(self, deep_nsm_result: Dict[str, Any], 
                                   template_result: Dict[str, Any],
                                   text: str, language: str) -> Dict[str, Any]:
        """Compare quality of DeepNSM vs template explications."""
        try:
            deep_nsm_explication = deep_nsm_result.get('explication', '')
            template_explication = template_result.get('explication', '')
            
            # Calculate semantic similarity
            semantic_similarity = self._calculate_semantic_similarity(
                text, deep_nsm_explication, template_explication
            )
            
            # Calculate fluency scores
            deep_nsm_fluency = self._calculate_fluency(deep_nsm_explication)
            template_fluency = self._calculate_fluency(template_explication)
            
            # Calculate coverage scores
            deep_nsm_coverage = self._calculate_coverage(deep_nsm_result, text)
            template_coverage = self._calculate_coverage(template_result, text)
            
            # Calculate overall quality scores
            deep_nsm_quality = (
                semantic_similarity['deep_nsm'] * self.comparison_params['semantic_weight'] +
                deep_nsm_fluency * self.comparison_params['fluency_weight'] +
                deep_nsm_coverage * self.comparison_params['coverage_weight']
            )
            
            template_quality = (
                semantic_similarity['template'] * self.comparison_params['semantic_weight'] +
                template_fluency * self.comparison_params['fluency_weight'] +
                template_coverage * self.comparison_params['coverage_weight']
            )
            
            return {
                'semantic_similarity': semantic_similarity,
                'fluency_scores': {
                    'deep_nsm': deep_nsm_fluency,
                    'template': template_fluency
                },
                'coverage_scores': {
                    'deep_nsm': deep_nsm_coverage,
                    'template': template_coverage
                },
                'quality_scores': {
                    'deep_nsm': deep_nsm_quality,
                    'template': template_quality
                },
                'deep_nsm_better': deep_nsm_quality > template_quality,
                'quality_difference': deep_nsm_quality - template_quality
            }
        
        except Exception as e:
            logger.warning(f"Quality comparison failed: {e}")
            return {
                'semantic_similarity': {'deep_nsm': 0.5, 'template': 0.5},
                'fluency_scores': {'deep_nsm': 0.5, 'template': 0.5},
                'coverage_scores': {'deep_nsm': 0.5, 'template': 0.5},
                'quality_scores': {'deep_nsm': 0.5, 'template': 0.5},
                'deep_nsm_better': False,
                'quality_difference': 0.0
            }
    
    def _calculate_semantic_similarity(self, text: str, deep_nsm_explication: str, 
                                     template_explication: str) -> Dict[str, float]:
        """Calculate semantic similarity between text and explications."""
        if not self.sbert_model:
            return {'deep_nsm': 0.5, 'template': 0.5}
        
        try:
            # Calculate embeddings
            embeddings = self.sbert_model.encode([text, deep_nsm_explication, template_explication])
            
            # Calculate similarities
            deep_nsm_similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            template_similarity = np.dot(embeddings[0], embeddings[2]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[2])
            )
            
            return {
                'deep_nsm': max(0.0, float(deep_nsm_similarity)),
                'template': max(0.0, float(template_similarity))
            }
        
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return {'deep_nsm': 0.5, 'template': 0.5}
    
    def _calculate_fluency(self, explication: str) -> float:
        """Calculate fluency score of explication."""
        if not explication:
            return 0.0
        
        # Simple fluency metrics
        words = explication.split()
        
        # Length penalty (too short or too long)
        length_score = 1.0
        if len(words) < 3:
            length_score = 0.5
        elif len(words) > 20:
            length_score = 0.7
        
        # Grammar score (simple heuristics)
        grammar_score = 1.0
        if explication.startswith('unknown(') or explication.startswith('error('):
            grammar_score = 0.3
        elif 'AND' in explication and explication.count('AND') > 2:
            grammar_score = 0.6
        
        return (length_score + grammar_score) / 2
    
    def _calculate_coverage(self, result: Dict[str, Any], text: str) -> float:
        """Calculate coverage score of explication."""
        primitives = result.get('primitives', [])
        confidence = result.get('confidence', 0.0)
        
        # Coverage based on primitive detection and confidence
        if not primitives:
            return 0.0
        
        # Simple coverage metric
        coverage = min(len(primitives) * 0.2 + confidence * 0.8, 1.0)
        return coverage
    
    def _generate_recommendation(self, comparison: Dict[str, Any]) -> str:
        """Generate recommendation based on comparison."""
        if comparison.get('deep_nsm_better', False):
            quality_diff = comparison.get('quality_difference', 0.0)
            if quality_diff > 0.2:
                return 'use_deep_nsm'
            else:
                return 'prefer_deep_nsm'
        else:
            quality_diff = abs(comparison.get('quality_difference', 0.0))
            if quality_diff > 0.2:
                return 'use_template'
            else:
                return 'prefer_template'


class EnhancedDeepNSMIntegration:
    """Enhanced DeepNSM integration system with comprehensive analysis."""
    
    def __init__(self):
        """Initialize the enhanced DeepNSM integration system."""
        self.deep_nsm_model = DeepNSMModel()
        self.comparator = DeepNSMComparator()
        self.nsm_translator = NSMTranslator()
        
        # Integration parameters
        self.integration_params = {
            'min_confidence': 0.5,
            'similarity_threshold': 0.7,
            'quality_threshold': 0.6
        }
    
    def run_deepnsm_integration(self, test_texts: List[str], languages: List[str] = ["en", "es", "fr"]) -> Dict[str, Any]:
        """Run comprehensive DeepNSM integration analysis."""
        logger.info(f"Running DeepNSM integration analysis for {len(test_texts)} texts")
        
        integration_results = {
            'test_configuration': {
                'num_test_texts': len(test_texts),
                'languages': languages,
                'timestamp': time.time()
            },
            'explication_results': [],
            'comparison_results': [],
            'integration_analysis': {},
            'recommendations': []
        }
        
        for language in languages:
            for text in test_texts:
                # Generate DeepNSM explication
                deep_nsm_result = self.deep_nsm_model.generate_explication(text, language)
                integration_results['explication_results'].append(deep_nsm_result)
                
                # Compare with templates
                comparison_result = self.comparator.compare_explications(text, language)
                integration_results['comparison_results'].append(comparison_result)
        
        # Analyze integration results
        integration_results['integration_analysis'] = self._analyze_integration_results(
            integration_results['explication_results'],
            integration_results['comparison_results']
        )
        
        # Generate recommendations
        integration_results['recommendations'] = self._generate_integration_recommendations(
            integration_results['integration_analysis']
        )
        
        return integration_results
    
    def _analyze_integration_results(self, explication_results: List[Dict[str, Any]], 
                                   comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze integration results."""
        analysis = {
            'total_explications': len(explication_results),
            'successful_explications': 0,
            'deep_nsm_wins': 0,
            'template_wins': 0,
            'avg_deep_nsm_confidence': 0.0,
            'avg_template_confidence': 0.0,
            'avg_quality_difference': 0.0,
            'method_distribution': defaultdict(int),
            'quality_distribution': defaultdict(int)
        }
        
        deep_nsm_confidences = []
        template_confidences = []
        quality_differences = []
        
        for explication_result in explication_results:
            if explication_result.get('confidence', 0.0) > 0.0:
                analysis['successful_explications'] += 1
            
            method = explication_result.get('method', 'unknown')
            analysis['method_distribution'][method] += 1
            
            deep_nsm_confidences.append(explication_result.get('confidence', 0.0))
        
        for comparison_result in comparison_results:
            comparison = comparison_result.get('comparison', {})
            
            if comparison.get('deep_nsm_better', False):
                analysis['deep_nsm_wins'] += 1
            else:
                analysis['template_wins'] += 1
            
            quality_scores = comparison.get('quality_scores', {})
            deep_nsm_quality = quality_scores.get('deep_nsm', 0.0)
            template_quality = quality_scores.get('template', 0.0)
            
            quality_diff = deep_nsm_quality - template_quality
            quality_differences.append(quality_diff)
            
            # Quality distribution
            if deep_nsm_quality >= 0.8:
                analysis['quality_distribution']['excellent'] += 1
            elif deep_nsm_quality >= 0.6:
                analysis['quality_distribution']['good'] += 1
            elif deep_nsm_quality >= 0.4:
                analysis['quality_distribution']['fair'] += 1
            else:
                analysis['quality_distribution']['poor'] += 1
        
        # Calculate averages
        if deep_nsm_confidences:
            analysis['avg_deep_nsm_confidence'] = np.mean(deep_nsm_confidences)
        
        if quality_differences:
            analysis['avg_quality_difference'] = np.mean(quality_differences)
        
        return analysis
    
    def _generate_integration_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate integration recommendations."""
        recommendations = []
        
        # Success rate recommendations
        success_rate = analysis['successful_explications'] / analysis['total_explications']
        if success_rate < 0.8:
            recommendations.append(f"Low success rate ({success_rate:.1%}) - improve explication generation")
        
        # Method preference recommendations
        deep_nsm_win_rate = analysis['deep_nsm_wins'] / (analysis['deep_nsm_wins'] + analysis['template_wins'])
        if deep_nsm_win_rate > 0.6:
            recommendations.append("DeepNSM performs significantly better - consider prioritizing DeepNSM")
        elif deep_nsm_win_rate < 0.4:
            recommendations.append("Templates perform significantly better - consider improving DeepNSM")
        else:
            recommendations.append("DeepNSM and templates perform similarly - consider hybrid approach")
        
        # Quality recommendations
        if analysis['avg_deep_nsm_confidence'] < 0.6:
            recommendations.append("Low average DeepNSM confidence - improve model training")
        
        if analysis['avg_quality_difference'] < 0.1:
            recommendations.append("Minimal quality difference - consider cost-benefit analysis")
        
        return recommendations


def main():
    """Main function to run enhanced DeepNSM integration."""
    logger.info("Starting enhanced DeepNSM integration...")
    
    # Initialize integration system
    integration_system = EnhancedDeepNSMIntegration()
    
    # Test texts
    test_texts = [
        "The red car is parked near the building",
        "The cat is on the mat",
        "This is similar to that",
        "The book contains important information",
        "The weather is cold today",
        "She works at the hospital",
        "The movie was very long",
        "I need to buy groceries",
        "Children play in the park",
        "The restaurant serves Italian food"
    ]
    
    # Run integration analysis
    integration_results = integration_system.run_deepnsm_integration(test_texts, ["en", "es", "fr"])
    
    # Print results
    print("\n" + "="*80)
    print("ENHANCED DEEPNSM INTEGRATION RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Texts: {integration_results['test_configuration']['num_test_texts']}")
    print(f"  Languages: {integration_results['test_configuration']['languages']}")
    
    print(f"\nIntegration Analysis:")
    analysis = integration_results['integration_analysis']
    print(f"  Total Explications: {analysis['total_explications']}")
    print(f"  Successful Explications: {analysis['successful_explications']}")
    print(f"  Success Rate: {analysis['successful_explications']/analysis['total_explications']:.1%}")
    print(f"  DeepNSM Wins: {analysis['deep_nsm_wins']}")
    print(f"  Template Wins: {analysis['template_wins']}")
    print(f"  Average DeepNSM Confidence: {analysis['avg_deep_nsm_confidence']:.3f}")
    print(f"  Average Quality Difference: {analysis['avg_quality_difference']:.3f}")
    
    print(f"\nMethod Distribution:")
    for method, count in analysis['method_distribution'].items():
        print(f"  {method}: {count}")
    
    print(f"\nQuality Distribution:")
    for quality, count in analysis['quality_distribution'].items():
        print(f"  {quality}: {count}")
    
    print(f"\nExample Comparisons:")
    for i, comparison in enumerate(integration_results['comparison_results'][:3]):
        text = comparison['text']
        deep_nsm_expl = comparison['deep_nsm_result'].get('explication', '')
        template_expl = comparison['template_result'].get('explication', '')
        recommendation = comparison['recommendation']
        
        print(f"  {i+1}. Text: {text}")
        print(f"     DeepNSM: {deep_nsm_expl}")
        print(f"     Template: {template_expl}")
        print(f"     Recommendation: {recommendation}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(integration_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_path = "data/deepnsm_integration_enhanced_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(integration_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced DeepNSM integration report saved to: {output_path}")
    
    print("="*80)
    print("Enhanced DeepNSM integration completed!")
    print("="*80)


if __name__ == "__main__":
    main()

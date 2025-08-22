#!/usr/bin/env python3
"""
Enhanced Advanced UD Patterns System.

This script implements a comprehensive advanced UD patterns system to add
dependency-based detection for improved primitive detection and semantic analysis.
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
import spacy
from spacy.tokens import Doc, Token
from spacy.symbols import nsubj, dobj, prep, pobj, amod, advmod, conj, cc, det, aux

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


class UDDependencyPattern:
    """Universal Dependencies dependency pattern for primitive detection."""
    
    def __init__(self, name: str, pattern: List[Dict[str, Any]], primitive_type: str, confidence: float = 1.0):
        """Initialize a UD dependency pattern."""
        self.name = name
        self.pattern = pattern
        self.primitive_type = primitive_type
        self.confidence = confidence
        
        # Pattern validation
        self._validate_pattern()
    
    def _validate_pattern(self):
        """Validate the dependency pattern structure."""
        if not self.pattern:
            raise ValueError("Pattern cannot be empty")
        
        for step in self.pattern:
            if not isinstance(step, dict):
                raise ValueError("Each pattern step must be a dictionary")
            
            required_keys = ['dep', 'pos']
            for key in required_keys:
                if key not in step:
                    raise ValueError(f"Pattern step missing required key: {key}")
    
    def match(self, doc: Doc) -> List[Dict[str, Any]]:
        """Match the pattern against a spaCy document."""
        matches = []
        
        try:
            for token in doc:
                if self._matches_token(token, self.pattern[0]):
                    match_result = self._extend_match(token, self.pattern, doc)
                    if match_result:
                        matches.append(match_result)
        
        except Exception as e:
            logger.warning(f"Pattern matching failed for {self.name}: {e}")
        
        return matches
    
    def _matches_token(self, token: Token, pattern_step: Dict[str, Any]) -> bool:
        """Check if a token matches a pattern step."""
        # Check dependency relation
        if token.dep_ != pattern_step['dep']:
            return False
        
        # Check part of speech
        if token.pos_ != pattern_step['pos']:
            return False
        
        # Check additional constraints
        if 'text' in pattern_step and token.text.lower() != pattern_step['text'].lower():
            return False
        
        if 'lemma' in pattern_step and token.lemma_.lower() != pattern_step['lemma'].lower():
            return False
        
        return True
    
    def _extend_match(self, start_token: Token, pattern: List[Dict[str, Any]], doc: Doc) -> Optional[Dict[str, Any]]:
        """Extend a match from a starting token."""
        try:
            matched_tokens = [start_token]
            current_token = start_token
            
            for i, pattern_step in enumerate(pattern[1:], 1):
                # Find next token based on dependency relation
                next_token = self._find_next_token(current_token, pattern_step, doc)
                
                if next_token is None:
                    return None
                
                matched_tokens.append(next_token)
                current_token = next_token
            
            return {
                'pattern_name': self.name,
                'primitive_type': self.primitive_type,
                'confidence': self.confidence,
                'tokens': [token.text for token in matched_tokens],
                'lemmas': [token.lemma_ for token in matched_tokens],
                'deps': [token.dep_ for token in matched_tokens],
                'pos': [token.pos_ for token in matched_tokens]
            }
        
        except Exception as e:
            logger.warning(f"Match extension failed: {e}")
            return None
    
    def _find_next_token(self, current_token: Token, pattern_step: Dict[str, Any], doc: Doc) -> Optional[Token]:
        """Find the next token based on dependency relations."""
        # Check children
        for child in current_token.children:
            if self._matches_token(child, pattern_step):
                return child
        
        # Check head
        if current_token.head != current_token and self._matches_token(current_token.head, pattern_step):
            return current_token.head
        
        # Check siblings
        for sibling in current_token.head.children:
            if sibling != current_token and self._matches_token(sibling, pattern_step):
                return sibling
        
        return None


class UDPatternRegistry:
    """Registry for UD dependency patterns."""
    
    def __init__(self):
        """Initialize the UD pattern registry."""
        self.patterns: Dict[str, UDDependencyPattern] = {}
        self.primitive_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Load base patterns
        self._load_base_patterns()
    
    def _load_base_patterns(self):
        """Load base UD dependency patterns."""
        base_patterns = [
            # Subject-Verb-Object patterns
            {
                'name': 'nsubj_verb_dobj',
                'pattern': [
                    {'dep': 'nsubj', 'pos': 'NOUN'},
                    {'dep': 'ROOT', 'pos': 'VERB'},
                    {'dep': 'dobj', 'pos': 'NOUN'}
                ],
                'primitive_type': 'action',
                'confidence': 0.9
            },
            {
                'name': 'nsubj_verb_prep_pobj',
                'pattern': [
                    {'dep': 'nsubj', 'pos': 'NOUN'},
                    {'dep': 'ROOT', 'pos': 'VERB'},
                    {'dep': 'prep', 'pos': 'ADP'},
                    {'dep': 'pobj', 'pos': 'NOUN'}
                ],
                'primitive_type': 'location',
                'confidence': 0.8
            },
            
            # Property patterns
            {
                'name': 'nsubj_cop_amod',
                'pattern': [
                    {'dep': 'nsubj', 'pos': 'NOUN'},
                    {'dep': 'ROOT', 'pos': 'AUX'},
                    {'dep': 'acomp', 'pos': 'ADJ'}
                ],
                'primitive_type': 'property',
                'confidence': 0.9
            },
            {
                'name': 'nsubj_cop_det_amod',
                'pattern': [
                    {'dep': 'nsubj', 'pos': 'NOUN'},
                    {'dep': 'ROOT', 'pos': 'AUX'},
                    {'dep': 'det', 'pos': 'DET'},
                    {'dep': 'attr', 'pos': 'ADJ'}
                ],
                'primitive_type': 'property',
                'confidence': 0.8
            },
            
            # Comparison patterns
            {
                'name': 'nsubj_cop_prep_pobj',
                'pattern': [
                    {'dep': 'nsubj', 'pos': 'PRON'},
                    {'dep': 'ROOT', 'pos': 'AUX'},
                    {'dep': 'prep', 'pos': 'ADP'},
                    {'dep': 'pobj', 'pos': 'PRON'}
                ],
                'primitive_type': 'comparison',
                'confidence': 0.7
            },
            
            # Possession patterns
            {
                'name': 'nsubj_verb_det_noun',
                'pattern': [
                    {'dep': 'nsubj', 'pos': 'NOUN'},
                    {'dep': 'ROOT', 'pos': 'VERB'},
                    {'dep': 'dobj', 'pos': 'NOUN'}
                ],
                'primitive_type': 'possession',
                'confidence': 0.8
            },
            
            # Temporal patterns
            {
                'name': 'nsubj_verb_advmod',
                'pattern': [
                    {'dep': 'nsubj', 'pos': 'NOUN'},
                    {'dep': 'ROOT', 'pos': 'VERB'},
                    {'dep': 'advmod', 'pos': 'ADV'}
                ],
                'primitive_type': 'temporal',
                'confidence': 0.7
            },
            
            # Causality patterns
            {
                'name': 'nsubj_verb_cconj_verb',
                'pattern': [
                    {'dep': 'nsubj', 'pos': 'NOUN'},
                    {'dep': 'ROOT', 'pos': 'VERB'},
                    {'dep': 'cc', 'pos': 'CCONJ'},
                    {'dep': 'conj', 'pos': 'VERB'}
                ],
                'primitive_type': 'causality',
                'confidence': 0.6
            }
        ]
        
        for pattern_data in base_patterns:
            pattern = UDDependencyPattern(**pattern_data)
            self.add_pattern(pattern)
    
    def add_pattern(self, pattern: UDDependencyPattern):
        """Add a pattern to the registry."""
        self.patterns[pattern.name] = pattern
        self.primitive_patterns[pattern.primitive_type].append(pattern.name)
    
    def get_patterns_by_type(self, primitive_type: str) -> List[UDDependencyPattern]:
        """Get patterns by primitive type."""
        pattern_names = self.primitive_patterns.get(primitive_type, [])
        return [self.patterns[name] for name in pattern_names if name in self.patterns]
    
    def get_all_patterns(self) -> List[UDDependencyPattern]:
        """Get all patterns."""
        return list(self.patterns.values())


class UDDependencyDetector:
    """Universal Dependencies dependency detector for primitive detection."""
    
    def __init__(self):
        """Initialize the UD dependency detector."""
        self.nlp = None
        self.pattern_registry = UDPatternRegistry()
        self.sbert_model = None
        
        # Detection parameters
        self.detection_params = {
            'min_confidence': 0.5,
            'max_patterns_per_text': 10,
            'enable_semantic_validation': True,
            'semantic_threshold': 0.7
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load spaCy and SBERT models."""
        try:
            logger.info("Loading spaCy model for UD dependency detection...")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            self.nlp = None
        
        try:
            logger.info("Loading SBERT model for semantic validation...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def detect_primitives(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Detect primitives using UD dependency patterns."""
        try:
            if not self.nlp:
                return {
                    'text': text,
                    'language': language,
                    'primitives': [],
                    'patterns_matched': [],
                    'confidence': 0.0,
                    'error': 'spaCy model not available'
                }
            
            # Parse text with spaCy
            doc = self.nlp(text)
            
            # Apply all patterns
            all_matches = []
            for pattern in self.pattern_registry.get_all_patterns():
                matches = pattern.match(doc)
                all_matches.extend(matches)
            
            # Filter and rank matches
            filtered_matches = self._filter_matches(all_matches)
            ranked_matches = self._rank_matches(filtered_matches, text)
            
            # Extract primitives
            primitives = self._extract_primitives(ranked_matches)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(ranked_matches)
            
            return {
                'text': text,
                'language': language,
                'primitives': primitives,
                'patterns_matched': ranked_matches,
                'confidence': confidence,
                'doc_info': {
                    'num_tokens': len(doc),
                    'num_sentences': len(list(doc.sents)),
                    'pos_distribution': self._get_pos_distribution(doc)
                }
            }
        
        except Exception as e:
            logger.warning(f"UD primitive detection failed: {e}")
            return {
                'text': text,
                'language': language,
                'primitives': [],
                'patterns_matched': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _filter_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter matches based on confidence threshold."""
        return [match for match in matches if match['confidence'] >= self.detection_params['min_confidence']]
    
    def _rank_matches(self, matches: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Rank matches by confidence and semantic relevance."""
        try:
            # Sort by confidence first
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Apply semantic validation if enabled
            if self.detection_params['enable_semantic_validation'] and self.sbert_model:
                for match in matches:
                    semantic_score = self._calculate_semantic_relevance(match, text)
                    match['semantic_score'] = semantic_score
                    match['final_score'] = (match['confidence'] + semantic_score) / 2
                
                # Re-sort by final score
                matches.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Limit number of matches
            return matches[:self.detection_params['max_patterns_per_text']]
        
        except Exception as e:
            logger.warning(f"Match ranking failed: {e}")
            return matches
    
    def _calculate_semantic_relevance(self, match: Dict[str, Any], text: str) -> float:
        """Calculate semantic relevance of a match to the original text."""
        try:
            if not self.sbert_model:
                return 0.5
            
            # Create representation of the match
            match_text = ' '.join(match['tokens'])
            
            # Calculate semantic similarity
            embeddings = self.sbert_model.encode([text, match_text])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return max(0.0, float(similarity))
        
        except Exception as e:
            logger.warning(f"Semantic relevance calculation failed: {e}")
            return 0.5
    
    def _extract_primitives(self, matches: List[Dict[str, Any]]) -> List[str]:
        """Extract primitive types from matches."""
        primitives = []
        
        for match in matches:
            primitive_type = match['primitive_type']
            if primitive_type not in primitives:
                primitives.append(primitive_type)
        
        return primitives
    
    def _calculate_confidence(self, matches: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence from matches."""
        if not matches:
            return 0.0
        
        # Use the highest confidence match
        max_confidence = max(match.get('final_score', match['confidence']) for match in matches)
        
        # Boost confidence based on number of matches
        num_matches = len(matches)
        if num_matches > 1:
            max_confidence = min(1.0, max_confidence + 0.1)
        
        return max_confidence
    
    def _get_pos_distribution(self, doc: Doc) -> Dict[str, int]:
        """Get part-of-speech distribution."""
        pos_counts = Counter(token.pos_ for token in doc)
        return dict(pos_counts)


class UDPatternAnalyzer:
    """Analyzer for UD pattern performance and optimization."""
    
    def __init__(self):
        """Initialize the UD pattern analyzer."""
        self.detector = UDDependencyDetector()
        self.nsm_translator = NSMTranslator()
        
        # Analysis parameters
        self.analysis_params = {
            'min_test_size': 10,
            'confidence_threshold': 0.6,
            'semantic_threshold': 0.7
        }
    
    def analyze_pattern_performance(self, test_texts: List[str], 
                                  languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Analyze performance of UD patterns."""
        logger.info(f"Analyzing UD pattern performance for {len(test_texts)} texts")
        
        analysis_results = {
            'test_configuration': {
                'num_test_texts': len(test_texts),
                'languages': languages,
                'timestamp': time.time()
            },
            'detection_results': [],
            'pattern_performance': {},
            'primitive_distribution': defaultdict(int),
            'confidence_analysis': {},
            'recommendations': []
        }
        
        # Run detection on test texts
        for language in languages:
            for text in test_texts:
                detection_result = self.detector.detect_primitives(text, language)
                analysis_results['detection_results'].append(detection_result)
                
                # Compare with baseline NSM detection
                baseline_primitives = self.nsm_translator.detect_primitives_in_text(text, language)
                detection_result['baseline_primitives'] = baseline_primitives
        
        # Analyze pattern performance
        analysis_results['pattern_performance'] = self._analyze_pattern_performance(
            analysis_results['detection_results']
        )
        
        # Analyze primitive distribution
        analysis_results['primitive_distribution'] = self._analyze_primitive_distribution(
            analysis_results['detection_results']
        )
        
        # Analyze confidence
        analysis_results['confidence_analysis'] = self._analyze_confidence(
            analysis_results['detection_results']
        )
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_recommendations(
            analysis_results
        )
        
        return analysis_results
    
    def _analyze_pattern_performance(self, detection_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance of individual patterns."""
        pattern_stats = defaultdict(lambda: {
            'matches': 0,
            'total_confidence': 0.0,
            'successful_detections': 0,
            'baseline_agreement': 0
        })
        
        for result in detection_results:
            for match in result.get('patterns_matched', []):
                pattern_name = match['pattern_name']
                pattern_stats[pattern_name]['matches'] += 1
                pattern_stats[pattern_name]['total_confidence'] += match.get('final_score', match['confidence'])
                
                # Check if detection was successful
                if result.get('confidence', 0.0) >= self.analysis_params['confidence_threshold']:
                    pattern_stats[pattern_name]['successful_detections'] += 1
                
                # Check agreement with baseline
                ud_primitives = set(result.get('primitives', []))
                baseline_primitives = set(result.get('baseline_primitives', []))
                if ud_primitives & baseline_primitives:  # Intersection
                    pattern_stats[pattern_name]['baseline_agreement'] += 1
        
        # Calculate averages
        for pattern_name, stats in pattern_stats.items():
            if stats['matches'] > 0:
                stats['avg_confidence'] = stats['total_confidence'] / stats['matches']
                stats['success_rate'] = stats['successful_detections'] / stats['matches']
                stats['agreement_rate'] = stats['baseline_agreement'] / stats['matches']
            else:
                stats['avg_confidence'] = 0.0
                stats['success_rate'] = 0.0
                stats['agreement_rate'] = 0.0
        
        return dict(pattern_stats)
    
    def _analyze_primitive_distribution(self, detection_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of detected primitives."""
        primitive_counts = defaultdict(int)
        
        for result in detection_results:
            for primitive in result.get('primitives', []):
                primitive_counts[primitive] += 1
        
        return dict(primitive_counts)
    
    def _analyze_confidence(self, detection_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze confidence distribution."""
        confidences = [result.get('confidence', 0.0) for result in detection_results]
        
        if not confidences:
            return {
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'confidence_distribution': {}
            }
        
        confidence_distribution = {
            'high': len([c for c in confidences if c >= 0.8]),
            'medium': len([c for c in confidences if 0.6 <= c < 0.8]),
            'low': len([c for c in confidences if c < 0.6])
        }
        
        return {
            'avg_confidence': np.mean(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'confidence_distribution': confidence_distribution
        }
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Pattern performance recommendations
        pattern_performance = analysis_results['pattern_performance']
        for pattern_name, stats in pattern_performance.items():
            if stats['success_rate'] < 0.5:
                recommendations.append(f"Pattern '{pattern_name}' has low success rate ({stats['success_rate']:.1%}) - consider improving")
            
            if stats['agreement_rate'] < 0.3:
                recommendations.append(f"Pattern '{pattern_name}' has low baseline agreement ({stats['agreement_rate']:.1%}) - consider refinement")
        
        # Confidence recommendations
        confidence_analysis = analysis_results['confidence_analysis']
        if confidence_analysis['avg_confidence'] < 0.6:
            recommendations.append("Low average confidence - consider adjusting pattern thresholds")
        
        # Primitive distribution recommendations
        primitive_dist = analysis_results['primitive_distribution']
        if len(primitive_dist) < 3:
            recommendations.append("Limited primitive diversity - consider adding more pattern types")
        
        return recommendations


class EnhancedUDPatternSystem:
    """Enhanced UD pattern system with comprehensive analysis."""
    
    def __init__(self):
        """Initialize the enhanced UD pattern system."""
        self.detector = UDDependencyDetector()
        self.analyzer = UDPatternAnalyzer()
        
        # System parameters
        self.system_params = {
            'enable_pattern_optimization': True,
            'enable_semantic_validation': True,
            'min_detection_confidence': 0.5,
            'max_patterns_per_text': 10
        }
    
    def run_ud_pattern_analysis(self, test_texts: List[str], 
                              languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Run comprehensive UD pattern analysis."""
        logger.info(f"Running UD pattern analysis for {len(test_texts)} texts")
        
        # Run pattern analysis
        analysis_results = self.analyzer.analyze_pattern_performance(test_texts, languages)
        
        # Add system summary
        analysis_results['system_summary'] = self._generate_system_summary(analysis_results)
        
        return analysis_results
    
    def _generate_system_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system summary."""
        detection_results = analysis_results['detection_results']
        
        total_detections = len(detection_results)
        successful_detections = len([r for r in detection_results if r.get('confidence', 0.0) >= 0.5])
        
        avg_confidence = analysis_results['confidence_analysis']['avg_confidence']
        
        pattern_performance = analysis_results['pattern_performance']
        total_patterns = len(pattern_performance)
        high_performing_patterns = len([p for p in pattern_performance.values() if p['success_rate'] >= 0.7])
        
        return {
            'total_detections': total_detections,
            'successful_detections': successful_detections,
            'success_rate': successful_detections / total_detections if total_detections > 0 else 0.0,
            'avg_confidence': avg_confidence,
            'total_patterns': total_patterns,
            'high_performing_patterns': high_performing_patterns,
            'pattern_success_rate': high_performing_patterns / total_patterns if total_patterns > 0 else 0.0
        }


def main():
    """Main function to run enhanced UD pattern analysis."""
    logger.info("Starting enhanced UD pattern analysis...")
    
    # Initialize UD pattern system
    ud_system = EnhancedUDPatternSystem()
    
    # Test texts with various dependency structures
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
        "The restaurant serves Italian food",
        "The dog chased the cat",
        "The teacher explained the lesson",
        "The student studied for the exam",
        "The company hired new employees",
        "The government passed the law"
    ]
    
    # Run UD pattern analysis
    analysis_results = ud_system.run_ud_pattern_analysis(test_texts, ["en"])
    
    # Print results
    print("\n" + "="*80)
    print("ENHANCED UD PATTERN ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Texts: {analysis_results['test_configuration']['num_test_texts']}")
    print(f"  Languages: {analysis_results['test_configuration']['languages']}")
    
    print(f"\nSystem Summary:")
    summary = analysis_results['system_summary']
    print(f"  Total Detections: {summary['total_detections']}")
    print(f"  Successful Detections: {summary['successful_detections']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Average Confidence: {summary['avg_confidence']:.3f}")
    print(f"  Total Patterns: {summary['total_patterns']}")
    print(f"  High-Performing Patterns: {summary['high_performing_patterns']}")
    print(f"  Pattern Success Rate: {summary['pattern_success_rate']:.1%}")
    
    print(f"\nPattern Performance:")
    pattern_performance = analysis_results['pattern_performance']
    for pattern_name, stats in sorted(pattern_performance.items(), key=lambda x: x[1]['success_rate'], reverse=True):
        print(f"  {pattern_name}:")
        print(f"    Matches: {stats['matches']}")
        print(f"    Success Rate: {stats['success_rate']:.1%}")
        print(f"    Agreement Rate: {stats['agreement_rate']:.1%}")
        print(f"    Avg Confidence: {stats['avg_confidence']:.3f}")
    
    print(f"\nPrimitive Distribution:")
    primitive_dist = analysis_results['primitive_distribution']
    for primitive, count in sorted(primitive_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {primitive}: {count}")
    
    print(f"\nConfidence Analysis:")
    confidence_analysis = analysis_results['confidence_analysis']
    print(f"  Average Confidence: {confidence_analysis['avg_confidence']:.3f}")
    print(f"  Min Confidence: {confidence_analysis['min_confidence']:.3f}")
    print(f"  Max Confidence: {confidence_analysis['max_confidence']:.3f}")
    print(f"  Distribution:")
    for level, count in confidence_analysis['confidence_distribution'].items():
        print(f"    {level}: {count}")
    
    print(f"\nExample Detections:")
    for i, result in enumerate(analysis_results['detection_results'][:3]):
        text = result['text']
        primitives = result.get('primitives', [])
        confidence = result.get('confidence', 0.0)
        baseline_primitives = result.get('baseline_primitives', [])
        
        print(f"  {i+1}. Text: {text}")
        print(f"     UD Primitives: {primitives}")
        print(f"     Baseline Primitives: {baseline_primitives}")
        print(f"     Confidence: {confidence:.3f}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_path = "data/advanced_ud_patterns_enhanced_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(analysis_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced UD pattern analysis report saved to: {output_path}")
    
    print("="*80)
    print("Enhanced UD pattern analysis completed!")
    print("="*80)


if __name__ == "__main__":
    main()

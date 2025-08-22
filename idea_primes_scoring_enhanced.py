#!/usr/bin/env python3
"""
Enhanced Idea Primes Scoring System.

This script implements a comprehensive idea primes scoring system to score
idea-primes with ΔMDL (Minimum Description Length), cross-lingual transfer,
and stability metrics for improved primitive evaluation and selection.
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
import math
import hashlib

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


class MDLScorer:
    """Minimum Description Length scorer for idea-primes."""
    
    def __init__(self):
        """Initialize the MDL scorer."""
        self.base_encoding_length = 8  # bits per character
        self.compression_threshold = 0.7
        
    def calculate_mdl_score(self, prime_value: str, usage_contexts: List[str]) -> Dict[str, float]:
        """Calculate MDL score for an idea-prime."""
        try:
            # Calculate description length without the prime
            original_length = self._calculate_original_length(usage_contexts)
            
            # Calculate description length with the prime
            compressed_length = self._calculate_compressed_length(prime_value, usage_contexts)
            
            # Calculate ΔMDL (change in description length)
            delta_mdl = original_length - compressed_length
            
            # Calculate compression ratio
            compression_ratio = compressed_length / original_length if original_length > 0 else 1.0
            
            # Calculate efficiency score
            efficiency_score = max(0.0, delta_mdl / original_length) if original_length > 0 else 0.0
            
            return {
                'original_length': original_length,
                'compressed_length': compressed_length,
                'delta_mdl': delta_mdl,
                'compression_ratio': compression_ratio,
                'efficiency_score': efficiency_score,
                'is_compressive': compression_ratio < self.compression_threshold
            }
        
        except Exception as e:
            logger.warning(f"MDL calculation failed: {e}")
            return {
                'original_length': 0.0,
                'compressed_length': 0.0,
                'delta_mdl': 0.0,
                'compression_ratio': 1.0,
                'efficiency_score': 0.0,
                'is_compressive': False
            }
    
    def _calculate_original_length(self, contexts: List[str]) -> float:
        """Calculate original description length."""
        total_length = 0.0
        
        for context in contexts:
            # Simple character-based encoding
            total_length += len(context) * self.base_encoding_length
        
        return total_length
    
    def _calculate_compressed_length(self, prime_value: str, contexts: List[str]) -> float:
        """Calculate compressed description length using the prime."""
        # Prime definition cost
        prime_cost = len(prime_value) * self.base_encoding_length
        
        # Usage cost (references to the prime)
        usage_cost = 0.0
        
        for context in contexts:
            # If context contains the prime, use shorter reference
            if prime_value.lower() in context.lower():
                # Use prime reference instead of full context
                reference_cost = len(prime_value) * self.base_encoding_length
                usage_cost += min(reference_cost, len(context) * self.base_encoding_length)
            else:
                # Keep original context
                usage_cost += len(context) * self.base_encoding_length
        
        return prime_cost + usage_cost


class CrossLingualTransferScorer:
    """Cross-lingual transfer scorer for idea-primes."""
    
    def __init__(self):
        """Initialize the cross-lingual transfer scorer."""
        self.sbert_model = None
        self.languages = ['en', 'es', 'fr']
        
        # Transfer parameters
        self.transfer_params = {
            'similarity_threshold': 0.7,
            'consistency_threshold': 0.6,
            'min_transfer_ratio': 0.5
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for cross-lingual transfer scoring...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def calculate_transfer_score(self, prime_value: str, language_contexts: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate cross-lingual transfer score for an idea-prime."""
        try:
            transfer_scores = {}
            consistency_scores = {}
            
            # Calculate transfer scores for each language pair
            for lang1 in self.languages:
                for lang2 in self.languages:
                    if lang1 >= lang2:
                        continue
                    
                    contexts1 = language_contexts.get(lang1, [])
                    contexts2 = language_contexts.get(lang2, [])
                    
                    if not contexts1 or not contexts2:
                        continue
                    
                    # Calculate semantic similarity between language contexts
                    similarity = self._calculate_language_similarity(contexts1, contexts2)
                    
                    # Calculate consistency
                    consistency = self._calculate_consistency(contexts1, contexts2)
                    
                    pair_key = f"{lang1}_{lang2}"
                    transfer_scores[pair_key] = similarity
                    consistency_scores[pair_key] = consistency
            
            # Calculate overall transfer score
            if transfer_scores:
                overall_transfer = np.mean(list(transfer_scores.values()))
                overall_consistency = np.mean(list(consistency_scores.values()))
                transfer_ratio = len(transfer_scores) / (len(self.languages) * (len(self.languages) - 1) / 2)
            else:
                overall_transfer = 0.0
                overall_consistency = 0.0
                transfer_ratio = 0.0
            
            return {
                'overall_transfer': overall_transfer,
                'overall_consistency': overall_consistency,
                'transfer_ratio': transfer_ratio,
                'pair_scores': transfer_scores,
                'consistency_scores': consistency_scores,
                'is_transferable': overall_transfer >= self.transfer_params['similarity_threshold'],
                'is_consistent': overall_consistency >= self.transfer_params['consistency_threshold']
            }
        
        except Exception as e:
            logger.warning(f"Cross-lingual transfer calculation failed: {e}")
            return {
                'overall_transfer': 0.0,
                'overall_consistency': 0.0,
                'transfer_ratio': 0.0,
                'pair_scores': {},
                'consistency_scores': {},
                'is_transferable': False,
                'is_consistent': False
            }
    
    def _calculate_language_similarity(self, contexts1: List[str], contexts2: List[str]) -> float:
        """Calculate semantic similarity between language contexts."""
        if not self.sbert_model or not contexts1 or not contexts2:
            return 0.5
        
        try:
            # Sample contexts for efficiency
            sample_size = min(5, len(contexts1), len(contexts2))
            sample1 = contexts1[:sample_size]
            sample2 = contexts2[:sample_size]
            
            # Calculate embeddings
            embeddings1 = self.sbert_model.encode(sample1)
            embeddings2 = self.sbert_model.encode(sample2)
            
            # Calculate pairwise similarities
            similarities = []
            for emb1 in embeddings1:
                for emb2 in embeddings2:
                    similarity = np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2)
                    )
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
        
        except Exception as e:
            logger.warning(f"Language similarity calculation failed: {e}")
            return 0.5
    
    def _calculate_consistency(self, contexts1: List[str], contexts2: List[str]) -> float:
        """Calculate consistency between language contexts."""
        if not contexts1 or not contexts2:
            return 0.0
        
        # Simple consistency based on context length and structure
        avg_length1 = np.mean([len(ctx) for ctx in contexts1])
        avg_length2 = np.mean([len(ctx) for ctx in contexts2])
        
        # Length consistency
        length_ratio = min(avg_length1, avg_length2) / max(avg_length1, avg_length2)
        
        # Structure consistency (simple word count)
        avg_words1 = np.mean([len(ctx.split()) for ctx in contexts1])
        avg_words2 = np.mean([len(ctx.split()) for ctx in contexts2])
        
        word_ratio = min(avg_words1, avg_words2) / max(avg_words1, avg_words2)
        
        return (length_ratio + word_ratio) / 2


class StabilityScorer:
    """Stability scorer for idea-primes."""
    
    def __init__(self):
        """Initialize the stability scorer."""
        self.stability_params = {
            'min_occurrences': 3,
            'consistency_threshold': 0.7,
            'variance_threshold': 0.3
        }
    
    def calculate_stability_score(self, prime_value: str, usage_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate stability score for an idea-prime."""
        try:
            if len(usage_history) < self.stability_params['min_occurrences']:
                return {
                    'stability_score': 0.0,
                    'consistency': 0.0,
                    'variance': 1.0,
                    'frequency': len(usage_history),
                    'is_stable': False,
                    'trend': 'insufficient_data'
                }
            
            # Extract usage patterns
            usage_patterns = self._extract_usage_patterns(usage_history)
            
            # Calculate consistency
            consistency = self._calculate_consistency(usage_patterns)
            
            # Calculate variance
            variance = self._calculate_variance(usage_patterns)
            
            # Calculate trend
            trend = self._calculate_trend(usage_history)
            
            # Calculate overall stability score
            stability_score = (
                consistency * 0.4 +
                (1.0 - variance) * 0.3 +
                self._calculate_frequency_score(len(usage_history)) * 0.3
            )
            
            return {
                'stability_score': stability_score,
                'consistency': consistency,
                'variance': variance,
                'frequency': len(usage_history),
                'is_stable': stability_score >= 0.6,
                'trend': trend,
                'usage_patterns': usage_patterns
            }
        
        except Exception as e:
            logger.warning(f"Stability calculation failed: {e}")
            return {
                'stability_score': 0.0,
                'consistency': 0.0,
                'variance': 1.0,
                'frequency': 0,
                'is_stable': False,
                'trend': 'error'
            }
    
    def _extract_usage_patterns(self, usage_history: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Extract usage patterns from history."""
        patterns = {
            'contexts': [],
            'languages': [],
            'timestamps': [],
            'confidence_scores': []
        }
        
        for usage in usage_history:
            patterns['contexts'].append(usage.get('context', ''))
            patterns['languages'].append(usage.get('language', 'unknown'))
            patterns['timestamps'].append(usage.get('timestamp', 0))
            patterns['confidence_scores'].append(usage.get('confidence', 0.0))
        
        return patterns
    
    def _calculate_consistency(self, patterns: Dict[str, List[Any]]) -> float:
        """Calculate consistency of usage patterns."""
        if not patterns['contexts']:
            return 0.0
        
        # Language consistency
        language_counts = Counter(patterns['languages'])
        language_consistency = max(language_counts.values()) / len(patterns['languages'])
        
        # Confidence consistency
        confidence_scores = patterns['confidence_scores']
        if confidence_scores:
            confidence_variance = np.var(confidence_scores)
            confidence_consistency = max(0.0, 1.0 - confidence_variance)
        else:
            confidence_consistency = 0.0
        
        return (language_consistency + confidence_consistency) / 2
    
    def _calculate_variance(self, patterns: Dict[str, List[Any]]) -> float:
        """Calculate variance of usage patterns."""
        confidence_scores = patterns['confidence_scores']
        
        if len(confidence_scores) < 2:
            return 0.0
        
        return np.var(confidence_scores)
    
    def _calculate_trend(self, usage_history: List[Dict[str, Any]]) -> str:
        """Calculate usage trend."""
        if len(usage_history) < 3:
            return 'insufficient_data'
        
        # Sort by timestamp
        sorted_history = sorted(usage_history, key=lambda x: x.get('timestamp', 0))
        
        # Calculate confidence trend
        confidences = [usage.get('confidence', 0.0) for usage in sorted_history]
        
        if len(confidences) >= 3:
            # Simple trend calculation
            first_half = np.mean(confidences[:len(confidences)//2])
            second_half = np.mean(confidences[len(confidences)//2:])
            
            if second_half > first_half * 1.1:
                return 'improving'
            elif second_half < first_half * 0.9:
                return 'declining'
            else:
                return 'stable'
        
        return 'insufficient_data'
    
    def _calculate_frequency_score(self, frequency: int) -> float:
        """Calculate frequency-based score."""
        if frequency >= 10:
            return 1.0
        elif frequency >= 5:
            return 0.8
        elif frequency >= 3:
            return 0.6
        else:
            return 0.3


class EnhancedIdeaPrimesScorer:
    """Enhanced idea primes scoring system with comprehensive metrics."""
    
    def __init__(self):
        """Initialize the enhanced idea primes scorer."""
        self.mdl_scorer = MDLScorer()
        self.transfer_scorer = CrossLingualTransferScorer()
        self.stability_scorer = StabilityScorer()
        self.nsm_translator = NSMTranslator()
        
        # Scoring parameters
        self.scoring_params = {
            'mdl_weight': 0.3,
            'transfer_weight': 0.3,
            'stability_weight': 0.2,
            'nsm_compatibility_weight': 0.2,
            'min_overall_score': 0.5
        }
    
    def score_idea_prime(self, prime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score an idea-prime with comprehensive metrics."""
        logger.info(f"Scoring idea-prime: {prime_data.get('value', 'unknown')}")
        
        prime_value = prime_data.get('value', '')
        attribute = prime_data.get('attribute', '')
        category = prime_data.get('category', 'general')
        
        # Generate usage contexts for scoring
        usage_contexts = self._generate_usage_contexts(prime_value, attribute)
        language_contexts = self._generate_language_contexts(prime_value, attribute)
        usage_history = self._generate_usage_history(prime_data)
        
        # Calculate individual scores
        mdl_score = self.mdl_scorer.calculate_mdl_score(prime_value, usage_contexts)
        transfer_score = self.transfer_scorer.calculate_transfer_score(prime_value, language_contexts)
        stability_score = self.stability_scorer.calculate_stability_score(prime_value, usage_history)
        nsm_compatibility = self._calculate_nsm_compatibility(prime_value, attribute)
        
        # Calculate weighted overall score
        overall_score = (
            mdl_score['efficiency_score'] * self.scoring_params['mdl_weight'] +
            transfer_score['overall_transfer'] * self.scoring_params['transfer_weight'] +
            stability_score['stability_score'] * self.scoring_params['stability_weight'] +
            nsm_compatibility * self.scoring_params['nsm_compatibility_weight']
        )
        
        # Determine quality level
        quality_level = self._determine_quality_level(overall_score)
        
        return {
            'prime_id': prime_data.get('prime_id', ''),
            'prime_value': prime_value,
            'attribute': attribute,
            'category': category,
            'mdl_score': mdl_score,
            'transfer_score': transfer_score,
            'stability_score': stability_score,
            'nsm_compatibility': nsm_compatibility,
            'overall_score': overall_score,
            'quality_level': quality_level,
            'is_high_quality': overall_score >= self.scoring_params['min_overall_score'],
            'recommendations': self._generate_scoring_recommendations(
                mdl_score, transfer_score, stability_score, nsm_compatibility
            )
        }
    
    def _generate_usage_contexts(self, prime_value: str, attribute: str) -> List[str]:
        """Generate usage contexts for MDL scoring."""
        contexts = [
            f"The {attribute} is {prime_value}",
            f"This has {attribute} {prime_value}",
            f"The {prime_value} {attribute}",
            f"Something with {attribute} {prime_value}",
            f"An object with {attribute} {prime_value}"
        ]
        return contexts
    
    def _generate_language_contexts(self, prime_value: str, attribute: str) -> Dict[str, List[str]]:
        """Generate language-specific contexts for transfer scoring."""
        contexts = {
            'en': [
                f"The {attribute} is {prime_value}",
                f"This has {attribute} {prime_value}"
            ],
            'es': [
                f"El {attribute} es {prime_value}",
                f"Esto tiene {attribute} {prime_value}"
            ],
            'fr': [
                f"Le {attribute} est {prime_value}",
                f"Cela a {attribute} {prime_value}"
            ]
        }
        return contexts
    
    def _generate_usage_history(self, prime_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate usage history for stability scoring."""
        # Simulate usage history
        history = []
        base_time = time.time()
        
        for i in range(5):  # Simulate 5 usages
            usage = {
                'context': f"Usage {i+1} of {prime_data.get('value', '')}",
                'language': 'en',
                'timestamp': base_time + i * 3600,  # 1 hour intervals
                'confidence': prime_data.get('confidence', 0.5) + np.random.normal(0, 0.1)
            }
            history.append(usage)
        
        return history
    
    def _calculate_nsm_compatibility(self, prime_value: str, attribute: str) -> float:
        """Calculate NSM compatibility score."""
        try:
            # Check if prime matches NSM patterns
            nsm_patterns = [
                'HasProperty', 'AtLocation', 'SimilarTo', 'UsedFor', 'Contains',
                'Causes', 'PartOf', 'MadeOf', 'Desires', 'CapableOf'
            ]
            
            # Simple pattern matching
            prime_lower = prime_value.lower()
            for pattern in nsm_patterns:
                pattern_lower = pattern.lower()
                if pattern_lower in prime_lower or prime_lower in pattern_lower:
                    return 0.8
            
            # Check attribute compatibility
            compatible_attributes = [
                'agent', 'patient', 'location', 'time', 'manner', 'cause',
                'effect', 'property', 'goal', 'source', 'path'
            ]
            
            if attribute.lower() in compatible_attributes:
                return 0.6
            
            return 0.3
        
        except Exception as e:
            logger.warning(f"NSM compatibility calculation failed: {e}")
            return 0.3
    
    def _determine_quality_level(self, overall_score: float) -> str:
        """Determine quality level based on overall score."""
        if overall_score >= 0.8:
            return 'excellent'
        elif overall_score >= 0.6:
            return 'good'
        elif overall_score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_scoring_recommendations(self, mdl_score: Dict[str, Any], 
                                       transfer_score: Dict[str, Any],
                                       stability_score: Dict[str, Any],
                                       nsm_compatibility: float) -> List[str]:
        """Generate recommendations based on scoring results."""
        recommendations = []
        
        # MDL recommendations
        if not mdl_score['is_compressive']:
            recommendations.append("Low compression ratio - consider if prime provides sufficient value")
        
        if mdl_score['efficiency_score'] < 0.3:
            recommendations.append("Low MDL efficiency - prime may not be worth the encoding cost")
        
        # Transfer recommendations
        if not transfer_score['is_transferable']:
            recommendations.append("Low cross-lingual transfer - may not generalize well across languages")
        
        if not transfer_score['is_consistent']:
            recommendations.append("Low cross-lingual consistency - usage varies significantly across languages")
        
        # Stability recommendations
        if not stability_score['is_stable']:
            recommendations.append("Low stability - usage patterns are inconsistent")
        
        if stability_score['trend'] == 'declining':
            recommendations.append("Declining usage trend - consider if prime is still relevant")
        
        # NSM compatibility recommendations
        if nsm_compatibility < 0.5:
            recommendations.append("Low NSM compatibility - may not align well with semantic primitives")
        
        return recommendations
    
    def run_comprehensive_scoring(self, idea_primes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive scoring on all idea-primes."""
        logger.info(f"Running comprehensive scoring on {len(idea_primes)} idea-primes")
        
        scoring_results = {
            'test_configuration': {
                'num_primes': len(idea_primes),
                'timestamp': time.time()
            },
            'scored_primes': [],
            'scoring_statistics': {},
            'quality_distribution': {},
            'recommendations': []
        }
        
        # Score each prime
        for prime_data in idea_primes:
            scored_prime = self.score_idea_prime(prime_data)
            scoring_results['scored_primes'].append(scored_prime)
        
        # Calculate statistics
        scoring_results['scoring_statistics'] = self._calculate_scoring_statistics(
            scoring_results['scored_primes']
        )
        
        # Analyze quality distribution
        scoring_results['quality_distribution'] = self._analyze_quality_distribution(
            scoring_results['scored_primes']
        )
        
        # Generate overall recommendations
        scoring_results['recommendations'] = self._generate_overall_recommendations(
            scoring_results['scoring_statistics']
        )
        
        return scoring_results
    
    def _calculate_scoring_statistics(self, scored_primes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate scoring statistics."""
        if not scored_primes:
            return {}
        
        overall_scores = [prime['overall_score'] for prime in scored_primes]
        mdl_scores = [prime['mdl_score']['efficiency_score'] for prime in scored_primes]
        transfer_scores = [prime['transfer_score']['overall_transfer'] for prime in scored_primes]
        stability_scores = [prime['stability_score']['stability_score'] for prime in scored_primes]
        nsm_scores = [prime['nsm_compatibility'] for prime in scored_primes]
        
        return {
            'num_primes': len(scored_primes),
            'avg_overall_score': np.mean(overall_scores),
            'avg_mdl_score': np.mean(mdl_scores),
            'avg_transfer_score': np.mean(transfer_scores),
            'avg_stability_score': np.mean(stability_scores),
            'avg_nsm_score': np.mean(nsm_scores),
            'high_quality_count': sum(1 for prime in scored_primes if prime['is_high_quality']),
            'quality_ratio': sum(1 for prime in scored_primes if prime['is_high_quality']) / len(scored_primes)
        }
    
    def _analyze_quality_distribution(self, scored_primes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze quality level distribution."""
        quality_counts = Counter(prime['quality_level'] for prime in scored_primes)
        return dict(quality_counts)
    
    def _generate_overall_recommendations(self, statistics: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations."""
        recommendations = []
        
        if statistics['avg_overall_score'] < 0.6:
            recommendations.append("Low average overall score - consider improving prime selection criteria")
        
        if statistics['avg_mdl_score'] < 0.5:
            recommendations.append("Low average MDL efficiency - primes may not provide sufficient compression")
        
        if statistics['avg_transfer_score'] < 0.6:
            recommendations.append("Low average cross-lingual transfer - consider language-specific optimization")
        
        if statistics['avg_stability_score'] < 0.5:
            recommendations.append("Low average stability - usage patterns are inconsistent")
        
        if statistics['quality_ratio'] < 0.5:
            recommendations.append("Low quality ratio - consider stricter selection criteria")
        
        return recommendations


def main():
    """Main function to run enhanced idea primes scoring."""
    logger.info("Starting enhanced idea primes scoring...")
    
    # Initialize scorer
    scorer = EnhancedIdeaPrimesScorer()
    
    # Sample idea-primes for testing
    sample_primes = [
        {
            'prime_id': 'prime_001',
            'value': 'agent',
            'attribute': 'semantic_role',
            'category': 'semantic',
            'confidence': 0.8
        },
        {
            'prime_id': 'prime_002',
            'value': 'location',
            'attribute': 'spatial',
            'category': 'semantic',
            'confidence': 0.7
        },
        {
            'prime_id': 'prime_003',
            'value': 'past',
            'attribute': 'tense',
            'category': 'temporal',
            'confidence': 0.9
        },
        {
            'prime_id': 'prime_004',
            'value': 'positive',
            'attribute': 'polarity',
            'category': 'polarity',
            'confidence': 0.6
        },
        {
            'prime_id': 'prime_005',
            'value': 'cause',
            'attribute': 'causality',
            'category': 'causality',
            'confidence': 0.8
        }
    ]
    
    # Run comprehensive scoring
    scoring_results = scorer.run_comprehensive_scoring(sample_primes)
    
    # Print results
    print("\n" + "="*80)
    print("ENHANCED IDEA PRIMES SCORING RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Primes: {scoring_results['test_configuration']['num_primes']}")
    
    print(f"\nScoring Statistics:")
    stats = scoring_results['scoring_statistics']
    print(f"  Average Overall Score: {stats['avg_overall_score']:.3f}")
    print(f"  Average MDL Score: {stats['avg_mdl_score']:.3f}")
    print(f"  Average Transfer Score: {stats['avg_transfer_score']:.3f}")
    print(f"  Average Stability Score: {stats['avg_stability_score']:.3f}")
    print(f"  Average NSM Score: {stats['avg_nsm_score']:.3f}")
    print(f"  High Quality Count: {stats['high_quality_count']}/{stats['num_primes']}")
    print(f"  Quality Ratio: {stats['quality_ratio']:.1%}")
    
    print(f"\nQuality Distribution:")
    for quality, count in scoring_results['quality_distribution'].items():
        print(f"  {quality}: {count}")
    
    print(f"\nTop 3 Scored Primes:")
    sorted_primes = sorted(scoring_results['scored_primes'], 
                          key=lambda x: x['overall_score'], reverse=True)
    for i, prime in enumerate(sorted_primes[:3], 1):
        print(f"  {i}. {prime['prime_value']} ({prime['attribute']})")
        print(f"     Overall Score: {prime['overall_score']:.3f} ({prime['quality_level']})")
        print(f"     MDL: {prime['mdl_score']['efficiency_score']:.3f}, Transfer: {prime['transfer_score']['overall_transfer']:.3f}")
        print(f"     Stability: {prime['stability_score']['stability_score']:.3f}, NSM: {prime['nsm_compatibility']:.3f}")
    
    print(f"\nOverall Recommendations:")
    for i, rec in enumerate(scoring_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_path = "data/idea_primes_scoring_enhanced_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(scoring_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced idea primes scoring report saved to: {output_path}")
    
    print("="*80)
    print("Enhanced idea primes scoring completed!")
    print("="*80)


if __name__ == "__main__":
    main()

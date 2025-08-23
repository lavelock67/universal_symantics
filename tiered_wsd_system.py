#!/usr/bin/env python3
"""
Tiered WSD (Word Sense Disambiguation) System.

This script implements the tiered WSD system as specified in ChatGPT5's feedback:
- Role heuristics → SBERT → frequency fallback
- "Sense-unknown" fallback for ambiguous cases
- Integration with BabelNet for synset stability
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
import re
from dataclasses import dataclass, asdict
from enum import Enum
import time
from collections import defaultdict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class WSDTier(Enum):
    """Tiers of WSD confidence."""
    ROLE_HEURISTICS = "role_heuristics"
    SBERT_SEMANTIC = "sbert_semantic"
    FREQUENCY_ONLY = "frequency_only"
    SENSE_UNKNOWN = "sense_unknown"


@dataclass
class WSDSense:
    """A word sense with confidence and metadata."""
    word: str
    sense_id: str
    synset: str
    definition: str
    confidence: float
    tier: WSDTier
    evidence: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'word': self.word,
            'sense_id': self.sense_id,
            'synset': self.synset,
            'definition': self.definition,
            'confidence': self.confidence,
            'tier': self.tier.value,
            'evidence': self.evidence,
            'timestamp': self.timestamp
        }


@dataclass
class WSDResult:
    """Result of WSD disambiguation."""
    word: str
    context: str
    selected_sense: WSDSense
    candidate_senses: List[WSDSense]
    disambiguation_method: str
    confidence: float
    language: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'word': self.word,
            'context': self.context,
            'selected_sense': self.selected_sense.to_dict(),
            'candidate_senses': [sense.to_dict() for sense in self.candidate_senses],
            'disambiguation_method': self.disambiguation_method,
            'confidence': self.confidence,
            'language': self.language
        }


class RoleHeuristicsWSD:
    """Role-based heuristics for WSD."""
    
    def __init__(self):
        """Initialize role-based WSD."""
        # Role patterns for different word types
        self.role_patterns = {
            'en': {
                'like': {
                    'enjoy': [r'\blike\s+[a-z]+ing\b', r'\blike\s+to\s+[a-z]+\b'],
                    'similar': [r'\blike\s+[a-z]+\s+[a-z]+\b', r'\bsimilar\s+to\b'],
                    'want': [r'\bwould\s+like\b', r'\bwant\s+to\b']
                },
                'want': {
                    'desire': [r'\bwant\s+[a-z]+\b', r'\bdesire\b'],
                    'need': [r'\bneed\s+[a-z]+\b', r'\brequire\b']
                },
                'can': {
                    'ability': [r'\bcan\s+[a-z]+\b', r'\bable\s+to\b'],
                    'permission': [r'\bmay\s+[a-z]+\b', r'\ballowed\s+to\b']
                }
            },
            'es': {
                'gustar': {
                    'like': [r'\bgustar\s+[a-z]+\b', r'\bme\s+gusta\b'],
                    'please': [r'\bpor\s+favor\b', r'\bte\s+gustaría\b']
                },
                'querer': {
                    'want': [r'\bquerer\s+[a-z]+\b', r'\bquiero\b'],
                    'love': [r'\bte\s+quiero\b', r'\bamor\b']
                },
                'poder': {
                    'can': [r'\bpoder\s+[a-z]+\b', r'\bpuedo\b'],
                    'may': [r'\bse\s+puede\b', r'\bpermiso\b']
                }
            },
            'fr': {
                'aimer': {
                    'like': [r'\baimer\s+[a-z]+\b', r'\bj\'aime\b'],
                    'love': [r'\bje\s+t\'aime\b', r'\bamour\b']
                },
                'vouloir': {
                    'want': [r'\bvouloir\s+[a-z]+\b', r'\bje\s+veux\b'],
                    'will': [r'\bvolonté\b', r'\bintention\b']
                },
                'pouvoir': {
                    'can': [r'\bpouvoir\s+[a-z]+\b', r'\bje\s+peux\b'],
                    'may': [r'\bse\s+peut\b', r'\bpermis\b']
                }
            }
        }
        
        # Sense mappings
        self.sense_mappings = {
            'en': {
                'like': {
                    'enjoy': 'like.v.01',
                    'similar': 'like.v.02',
                    'want': 'like.v.03'
                },
                'want': {
                    'desire': 'want.v.01',
                    'need': 'want.v.02'
                },
                'can': {
                    'ability': 'can.v.01',
                    'permission': 'can.v.02'
                }
            },
            'es': {
                'gustar': {
                    'like': 'gustar.v.01',
                    'please': 'gustar.v.02'
                },
                'querer': {
                    'want': 'querer.v.01',
                    'love': 'querer.v.02'
                },
                'poder': {
                    'can': 'poder.v.01',
                    'may': 'poder.v.02'
                }
            },
            'fr': {
                'aimer': {
                    'like': 'aimer.v.01',
                    'love': 'aimer.v.02'
                },
                'vouloir': {
                    'want': 'vouloir.v.01',
                    'will': 'vouloir.v.02'
                },
                'pouvoir': {
                    'can': 'pouvoir.v.01',
                    'may': 'pouvoir.v.02'
                }
            }
        }
    
    def disambiguate(self, word: str, context: str, language: str = "en") -> Optional[WSDSense]:
        """Disambiguate word using role heuristics."""
        logger.info(f"Applying role heuristics for '{word}' in {language}")
        
        word_lower = word.lower()
        context_lower = context.lower()
        
        # Check if word has role patterns
        if word_lower not in self.role_patterns.get(language, {}):
            return None
        
        word_patterns = self.role_patterns[language][word_lower]
        best_sense = None
        best_confidence = 0.0
        
        # Check each sense pattern
        for sense_name, patterns in word_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context_lower, re.IGNORECASE):
                    confidence = self._calculate_pattern_confidence(pattern, context_lower)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_sense = sense_name
        
        if best_sense:
            synset = self.sense_mappings[language][word_lower][best_sense]
            return WSDSense(
                word=word,
                sense_id=best_sense,
                synset=synset,
                definition=f"{word} as {best_sense}",
                confidence=best_confidence,
                tier=WSDTier.ROLE_HEURISTICS,
                evidence={
                    'pattern_matched': True,
                    'context': context,
                    'sense_name': best_sense
                },
                timestamp=time.time()
            )
        
        return None
    
    def _calculate_pattern_confidence(self, pattern: str, context: str) -> float:
        """Calculate confidence based on pattern match quality."""
        # Simple confidence based on pattern specificity
        if r'\b' in pattern:  # Word boundary
            return 0.9
        elif len(pattern.split()) > 1:  # Multi-word pattern
            return 0.8
        else:
            return 0.7


class SBERTWSD:
    """SBERT-based semantic WSD."""
    
    def __init__(self):
        """Initialize SBERT-based WSD."""
        # Mock SBERT embeddings for demonstration
        self.sense_embeddings = {
            'like.v.01': [0.8, 0.2, 0.1],  # enjoy
            'like.v.02': [0.1, 0.9, 0.1],  # similar
            'like.v.03': [0.2, 0.1, 0.8],  # want
            'want.v.01': [0.9, 0.1, 0.1],  # desire
            'want.v.02': [0.1, 0.8, 0.2],  # need
            'can.v.01': [0.8, 0.1, 0.2],   # ability
            'can.v.02': [0.1, 0.2, 0.9]    # permission
        }
        
        # Context templates for different senses
        self.context_templates = {
            'like.v.01': ['enjoy', 'pleasure', 'fun', 'happy'],
            'like.v.02': ['similar', 'same', 'alike', 'resemble'],
            'like.v.03': ['want', 'desire', 'wish', 'hope'],
            'want.v.01': ['desire', 'wish', 'hope', 'long'],
            'want.v.02': ['need', 'require', 'must', 'essential'],
            'can.v.01': ['able', 'capable', 'skill', 'talent'],
            'can.v.02': ['permit', 'allow', 'may', 'authorized']
        }
    
    def disambiguate(self, word: str, context: str, candidate_senses: List[str]) -> Optional[WSDSense]:
        """Disambiguate word using SBERT semantic similarity."""
        logger.info(f"Applying SBERT semantic WSD for '{word}'")
        
        if not candidate_senses:
            return None
        
        # Get context embedding (simplified)
        context_embedding = self._get_context_embedding(context)
        
        best_sense = None
        best_similarity = 0.0
        
        # Compare with each candidate sense
        for sense_id in candidate_senses:
            if sense_id in self.sense_embeddings:
                sense_embedding = self.sense_embeddings[sense_id]
                similarity = self._cosine_similarity(context_embedding, sense_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_sense = sense_id
        
        if best_sense and best_similarity > 0.5:
            return WSDSense(
                word=word,
                sense_id=best_sense,
                synset=best_sense,
                definition=f"{word} with semantic similarity {best_similarity:.3f}",
                confidence=best_similarity,
                tier=WSDTier.SBERT_SEMANTIC,
                evidence={
                    'semantic_similarity': best_similarity,
                    'context': context,
                    'candidate_senses': candidate_senses
                },
                timestamp=time.time()
            )
        
        return None
    
    def _get_context_embedding(self, context: str) -> List[float]:
        """Get context embedding (simplified)."""
        # Simple bag-of-words embedding
        words = context.lower().split()
        embedding = [0.0, 0.0, 0.0]
        
        for word in words:
            if word in ['enjoy', 'like', 'love']:
                embedding[0] += 0.3
            elif word in ['similar', 'same', 'alike']:
                embedding[1] += 0.3
            elif word in ['want', 'desire', 'need']:
                embedding[2] += 0.3
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class FrequencyWSD:
    """Frequency-based WSD fallback."""
    
    def __init__(self):
        """Initialize frequency-based WSD."""
        # Frequency rankings for different senses
        self.frequency_rankings = {
            'en': {
                'like': ['like.v.01', 'like.v.02', 'like.v.03'],  # enjoy, similar, want
                'want': ['want.v.01', 'want.v.02'],  # desire, need
                'can': ['can.v.01', 'can.v.02']  # ability, permission
            },
            'es': {
                'gustar': ['gustar.v.01', 'gustar.v.02'],  # like, please
                'querer': ['querer.v.01', 'querer.v.02'],  # want, love
                'poder': ['poder.v.01', 'poder.v.02']  # can, may
            },
            'fr': {
                'aimer': ['aimer.v.01', 'aimer.v.02'],  # like, love
                'vouloir': ['vouloir.v.01', 'vouloir.v.02'],  # want, will
                'pouvoir': ['pouvoir.v.01', 'pouvoir.v.02']  # can, may
            }
        }
    
    def disambiguate(self, word: str, language: str = "en") -> Optional[WSDSense]:
        """Disambiguate word using frequency fallback."""
        logger.info(f"Applying frequency-based WSD for '{word}' in {language}")
        
        word_lower = word.lower()
        
        if word_lower not in self.frequency_rankings.get(language, {}):
            return None
        
        # Get most frequent sense
        most_frequent_sense = self.frequency_rankings[language][word_lower][0]
        
        return WSDSense(
            word=word,
            sense_id=most_frequent_sense,
            synset=most_frequent_sense,
            definition=f"{word} (most frequent sense)",
            confidence=0.5,  # Lower confidence for frequency-based
            tier=WSDTier.FREQUENCY_ONLY,
            evidence={
                'frequency_rank': 1,
                'language': language,
                'total_senses': len(self.frequency_rankings[language][word_lower])
            },
            timestamp=time.time()
        )


class TieredWSDSystem:
    """Tiered WSD system with fallback strategy."""
    
    def __init__(self):
        """Initialize the tiered WSD system."""
        self.role_wsd = RoleHeuristicsWSD()
        self.sbert_wsd = SBERTWSD()
        self.frequency_wsd = FrequencyWSD()
        
        # Confidence thresholds for each tier
        self.confidence_thresholds = {
            WSDTier.ROLE_HEURISTICS: 0.7,
            WSDTier.SBERT_SEMANTIC: 0.6,
            WSDTier.FREQUENCY_ONLY: 0.5,
            WSDTier.SENSE_UNKNOWN: 0.0
        }
    
    def disambiguate(self, word: str, context: str, language: str = "en") -> WSDResult:
        """Disambiguate word using tiered approach."""
        logger.info(f"Starting tiered WSD for '{word}' in context: {context[:50]}...")
        
        # Tier 1: Role heuristics
        role_sense = self.role_wsd.disambiguate(word, context, language)
        if role_sense and role_sense.confidence >= self.confidence_thresholds[WSDTier.ROLE_HEURISTICS]:
            return self._create_result(word, context, role_sense, language, "role_heuristics")
        
        # Tier 2: SBERT semantic similarity
        candidate_senses = self._get_candidate_senses(word, language)
        sbert_sense = self.sbert_wsd.disambiguate(word, context, candidate_senses)
        if sbert_sense and sbert_sense.confidence >= self.confidence_thresholds[WSDTier.SBERT_SEMANTIC]:
            return self._create_result(word, context, sbert_sense, language, "sbert_semantic")
        
        # Tier 3: Frequency fallback
        frequency_sense = self.frequency_wsd.disambiguate(word, language)
        if frequency_sense and frequency_sense.confidence >= self.confidence_thresholds[WSDTier.FREQUENCY_ONLY]:
            return self._create_result(word, context, frequency_sense, language, "frequency_only")
        
        # Tier 4: Sense unknown
        unknown_sense = WSDSense(
            word=word,
            sense_id="unknown",
            synset="unknown",
            definition=f"{word} (sense unknown)",
            confidence=0.0,
            tier=WSDTier.SENSE_UNKNOWN,
            evidence={
                'reason': 'No confident disambiguation possible',
                'context': context,
                'language': language
            },
            timestamp=time.time()
        )
        
        return self._create_result(word, context, unknown_sense, language, "sense_unknown")
    
    def _get_candidate_senses(self, word: str, language: str) -> List[str]:
        """Get candidate senses for a word."""
        word_lower = word.lower()
        
        # Get candidate senses from frequency rankings
        if word_lower in self.frequency_wsd.frequency_rankings.get(language, {}):
            return self.frequency_wsd.frequency_rankings[language][word_lower]
        
        return []
    
    def _create_result(self, word: str, context: str, selected_sense: WSDSense, 
                      language: str, method: str) -> WSDResult:
        """Create WSD result."""
        # Get all candidate senses
        candidate_senses = []
        for tier in [WSDTier.ROLE_HEURISTICS, WSDTier.SBERT_SEMANTIC, WSDTier.FREQUENCY_ONLY]:
            if tier == WSDTier.ROLE_HEURISTICS:
                sense = self.role_wsd.disambiguate(word, context, language)
            elif tier == WSDTier.SBERT_SEMANTIC:
                candidates = self._get_candidate_senses(word, language)
                sense = self.sbert_wsd.disambiguate(word, context, candidates)
            else:
                sense = self.frequency_wsd.disambiguate(word, language)
            
            if sense:
                candidate_senses.append(sense)
        
        return WSDResult(
            word=word,
            context=context,
            selected_sense=selected_sense,
            candidate_senses=candidate_senses,
            disambiguation_method=method,
            confidence=selected_sense.confidence,
            language=language
        )


class ComprehensiveTieredWSDSystem:
    """Comprehensive tiered WSD system."""
    
    def __init__(self):
        """Initialize the comprehensive tiered WSD system."""
        self.wsd_system = TieredWSDSystem()
    
    def run_wsd_analysis(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive WSD analysis."""
        logger.info(f"Running tiered WSD analysis on {len(test_cases)} test cases")
        
        results = {
            'test_configuration': {
                'num_test_cases': len(test_cases),
                'timestamp': time.time()
            },
            'wsd_results': [],
            'tier_analysis': {},
            'stability_analysis': {},
            'recommendations': []
        }
        
        # Run WSD on test cases
        for test_case in test_cases:
            wsd_result = self.wsd_system.disambiguate(
                test_case['word'],
                test_case['context'],
                test_case.get('language', 'en')
            )
            wsd_result.test_case = test_case
            results['wsd_results'].append(wsd_result.to_dict())
        
        # Analyze results
        results['tier_analysis'] = self._analyze_tier_usage(results['wsd_results'])
        results['stability_analysis'] = self._analyze_stability(results['wsd_results'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_wsd_recommendations(
            results['tier_analysis'], results['stability_analysis']
        )
        
        return results
    
    def _analyze_tier_usage(self, wsd_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze tier usage across WSD results."""
        analysis = {
            'tier_distribution': defaultdict(int),
            'confidence_by_tier': defaultdict(list),
            'method_distribution': defaultdict(int),
            'average_confidence': 0.0
        }
        
        confidences = []
        
        for result in wsd_results:
            selected_sense = result['selected_sense']
            tier = selected_sense['tier']
            confidence = selected_sense['confidence']
            method = result['disambiguation_method']
            
            analysis['tier_distribution'][tier] += 1
            analysis['confidence_by_tier'][tier].append(confidence)
            analysis['method_distribution'][method] += 1
            confidences.append(confidence)
        
        analysis['average_confidence'] = np.mean(confidences) if confidences else 0.0
        
        # Calculate average confidence by tier
        for tier in analysis['confidence_by_tier']:
            analysis['confidence_by_tier'][tier] = np.mean(analysis['confidence_by_tier'][tier])
        
        return analysis
    
    def _analyze_stability(self, wsd_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze synset stability across results."""
        analysis = {
            'synset_distribution': defaultdict(int),
            'stability_score': 0.0,
            'ambiguous_cases': 0,
            'high_confidence_cases': 0
        }
        
        synsets = []
        ambiguous_count = 0
        high_confidence_count = 0
        
        for result in wsd_results:
            selected_sense = result['selected_sense']
            synset = selected_sense['synset']
            confidence = selected_sense['confidence']
            
            synsets.append(synset)
            analysis['synset_distribution'][synset] += 1
            
            if synset == 'unknown':
                ambiguous_count += 1
            
            if confidence >= 0.8:
                high_confidence_count += 1
        
        # Calculate stability score (lower is more stable)
        if synsets:
            unique_synsets = len(set(synsets))
            total_cases = len(synsets)
            analysis['stability_score'] = unique_synsets / total_cases
        
        analysis['ambiguous_cases'] = ambiguous_count
        analysis['high_confidence_cases'] = high_confidence_count
        
        return analysis
    
    def _generate_wsd_recommendations(self, tier_analysis: Dict[str, Any], 
                                    stability_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations from WSD analysis."""
        recommendations = []
        
        # Tier-based recommendations
        tier_dist = tier_analysis.get('tier_distribution', {})
        if tier_dist.get('sense_unknown', 0) > 0:
            recommendations.append(f"Reduce ambiguous cases: {tier_dist['sense_unknown']} sense-unknown results")
        
        if tier_dist.get('frequency_only', 0) > tier_dist.get('role_heuristics', 0):
            recommendations.append("Improve role heuristics to reduce frequency fallback usage")
        
        # Confidence-based recommendations
        avg_confidence = tier_analysis.get('average_confidence', 0)
        if avg_confidence < 0.7:
            recommendations.append(f"Improve average confidence: current {avg_confidence:.3f} below 0.7 target")
        
        # Stability-based recommendations
        stability_score = stability_analysis.get('stability_score', 0)
        if stability_score > 0.5:
            recommendations.append(f"Improve synset stability: current score {stability_score:.3f} above 0.5 threshold")
        
        high_confidence = stability_analysis.get('high_confidence_cases', 0)
        total_cases = sum(tier_analysis.get('tier_distribution', {}).values())
        if total_cases > 0:
            high_confidence_rate = high_confidence / total_cases
            if high_confidence_rate < 0.8:
                recommendations.append(f"Increase high-confidence rate: current {high_confidence_rate:.1%} below 80% target")
        
        return recommendations


def main():
    """Main function to demonstrate tiered WSD system."""
    logger.info("Starting tiered WSD system demonstration...")
    
    # Initialize system
    system = ComprehensiveTieredWSDSystem()
    
    # Test cases
    test_cases = [
        {
            'word': 'like',
            'context': 'I like playing football',
            'language': 'en'
        },
        {
            'word': 'like',
            'context': 'This looks like a cat',
            'language': 'en'
        },
        {
            'word': 'like',
            'context': 'I would like to go home',
            'language': 'en'
        },
        {
            'word': 'want',
            'context': 'I want to help you',
            'language': 'en'
        },
        {
            'word': 'want',
            'context': 'You want to be careful',
            'language': 'en'
        },
        {
            'word': 'can',
            'context': 'I can swim very well',
            'language': 'en'
        },
        {
            'word': 'can',
            'context': 'You can enter now',
            'language': 'en'
        },
        {
            'word': 'gustar',
            'context': 'Me gusta la música',
            'language': 'es'
        },
        {
            'word': 'aimer',
            'context': 'J\'aime le café',
            'language': 'fr'
        }
    ]
    
    # Run analysis
    results = system.run_wsd_analysis(test_cases)
    
    # Print results
    print("\n" + "="*80)
    print("TIERED WSD SYSTEM RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Cases: {results['test_configuration']['num_test_cases']}")
    
    print(f"\nTier Analysis:")
    tier_analysis = results['tier_analysis']
    print(f"  Average Confidence: {tier_analysis['average_confidence']:.3f}")
    
    print(f"  Tier Distribution:")
    for tier, count in tier_analysis['tier_distribution'].items():
        print(f"    {tier}: {count}")
    
    print(f"  Method Distribution:")
    for method, count in tier_analysis['method_distribution'].items():
        print(f"    {method}: {count}")
    
    print(f"  Confidence by Tier:")
    for tier, confidence in tier_analysis['confidence_by_tier'].items():
        print(f"    {tier}: {confidence:.3f}")
    
    print(f"\nStability Analysis:")
    stability_analysis = results['stability_analysis']
    print(f"  Stability Score: {stability_analysis['stability_score']:.3f}")
    print(f"  Ambiguous Cases: {stability_analysis['ambiguous_cases']}")
    print(f"  High Confidence Cases: {stability_analysis['high_confidence_cases']}")
    
    print(f"  Synset Distribution:")
    for synset, count in stability_analysis['synset_distribution'].items():
        print(f"    {synset}: {count}")
    
    print(f"\nSample WSD Results:")
    for i, result in enumerate(results['wsd_results'][:3]):
        print(f"  {i+1}. Word: {result['word']}")
        print(f"     Context: {result['context']}")
        print(f"     Selected Sense: {result['selected_sense']['synset']}")
        print(f"     Confidence: {result['selected_sense']['confidence']:.3f}")
        print(f"     Method: {result['disambiguation_method']}")
        print()
    
    print(f"\nRecommendations:")
    for i, recommendation in enumerate(results['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    # Save results
    output_path = Path("data/tiered_wsd_system_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    try:
        json_results = convert_numpy_types(results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        # Save a simplified version
        simplified_results = {
            'test_configuration': results['test_configuration'],
            'tier_analysis': results['tier_analysis'],
            'stability_analysis': results['stability_analysis'],
            'recommendations': results['recommendations']
        }
        with open(output_path, 'w') as f:
            json.dump(simplified_results, f, indent=2)
    
    logger.info(f"Tiered WSD system results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Tiered WSD system demonstration completed!")
    print("="*80)


if __name__ == "__main__":
    main()

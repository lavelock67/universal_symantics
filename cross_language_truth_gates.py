#!/usr/bin/env python3
"""
Cross-Language Truth Gates System.

This script implements the cross-language truth gates as specified in ChatGPT5's feedback:
- Prime-set Jaccard gate on EN↔FR↔ES parallel items (after normalizing auxiliaries/idioms)
- Synset stability gate (tiered WSD → BabelNet) so surface rewrites can't drift sense
- Confusion matrix for frequently swapped primes
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


class GateType(Enum):
    """Types of cross-language truth gates."""
    PRIME_SET_JACCARD = "prime_set_jaccard"
    SYNSET_STABILITY = "synset_stability"
    CONFUSION_MATRIX = "confusion_matrix"


@dataclass
class GateResult:
    """Result of a cross-language truth gate evaluation."""
    gate_type: GateType
    test_name: str
    languages: List[str]
    score: float
    threshold: float
    passed: bool
    details: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'gate_type': self.gate_type.value,
            'test_name': self.test_name,
            'languages': self.languages,
            'score': self.score,
            'threshold': self.threshold,
            'passed': self.passed,
            'details': self.details,
            'timestamp': self.timestamp
        }


@dataclass
class ParallelItem:
    """A parallel text item across multiple languages."""
    item_id: str
    texts: Dict[str, str]  # language -> text
    explications: Dict[str, str]  # language -> NSM explication
    detected_primes: Dict[str, List[str]]  # language -> list of primes
    synsets: Dict[str, List[str]]  # language -> list of BabelNet synsets
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'item_id': self.item_id,
            'texts': self.texts,
            'explications': self.explications,
            'detected_primes': self.detected_primes,
            'synsets': self.synsets
        }


class PrimeSetJaccardGate:
    """Prime-set Jaccard gate on parallel items."""
    
    def __init__(self, threshold: float = 0.85):
        """Initialize the Jaccard gate."""
        self.threshold = threshold
        
        # Auxiliary/idiom normalization patterns
        self.auxiliary_patterns = {
            'en': [r'\bdo\s+not\b', r'\bdoes\s+not\b', r'\bdid\s+not\b'],
            'es': [r'\bno\s+[a-z]+\b', r'\bno\s+[a-z]+n\b'],
            'fr': [r'\bne\s+[a-z]+\s+pas\b', r'\bn\'[a-z]+\s+pas\b']
        }
        
        # Frequently swapped primes to watch
        self.swapped_prime_pairs = [
            ('WANT', 'LIKE'), ('WANT', 'NEED'),
            ('NOT', 'NO'), ('BEFORE', 'AFTER'),
            ('CAN', 'MUST'), ('SOME', 'ALL')
        ]
    
    def evaluate_parallel_item(self, item: ParallelItem) -> GateResult:
        """Evaluate a parallel item using prime-set Jaccard similarity."""
        logger.info(f"Evaluating prime-set Jaccard for item {item.item_id}")
        
        # Normalize primes across languages
        normalized_primes = {}
        for lang, primes in item.detected_primes.items():
            normalized_primes[lang] = self._normalize_primes(primes, lang)
        
        # Calculate Jaccard similarities between all language pairs
        jaccard_scores = {}
        language_pairs = []
        
        langs = list(normalized_primes.keys())
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                lang1, lang2 = langs[i], langs[j]
                jaccard = self._calculate_jaccard(
                    normalized_primes[lang1], 
                    normalized_primes[lang2]
                )
                jaccard_scores[f"{lang1}-{lang2}"] = jaccard
                language_pairs.append((lang1, lang2))
        
        # Calculate overall score (average of all pairs)
        overall_score = np.mean(list(jaccard_scores.values())) if jaccard_scores else 0.0
        
        # Check for swapped primes
        swapped_prime_issues = self._detect_swapped_primes(normalized_primes)
        
        # Determine if gate passes
        passed = overall_score >= self.threshold and len(swapped_prime_issues) == 0
        
        details = {
            'jaccard_scores': jaccard_scores,
            'normalized_primes': normalized_primes,
            'swapped_prime_issues': swapped_prime_issues,
            'language_pairs': language_pairs
        }
        
        return GateResult(
            gate_type=GateType.PRIME_SET_JACCARD,
            test_name=f"prime_set_jaccard_{item.item_id}",
            languages=langs,
            score=overall_score,
            threshold=self.threshold,
            passed=passed,
            details=details,
            timestamp=time.time()
        )
    
    def _normalize_primes(self, primes: List[str], language: str) -> Set[str]:
        """Normalize primes by removing auxiliaries and idioms."""
        normalized = set()
        
        for prime in primes:
            # Remove auxiliary patterns
            normalized_prime = prime
            patterns = self.auxiliary_patterns.get(language, [])
            for pattern in patterns:
                normalized_prime = re.sub(pattern, '', normalized_prime, flags=re.IGNORECASE)
            
            # Clean up extra spaces
            normalized_prime = re.sub(r'\s+', ' ', normalized_prime).strip()
            
            if normalized_prime:
                normalized.add(normalized_prime)
        
        return normalized
    
    def _calculate_jaccard(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0  # Both empty sets are considered identical
        if not set1 or not set2:
            return 0.0  # One empty set means no similarity
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union)
    
    def _detect_swapped_primes(self, normalized_primes: Dict[str, Set[str]]) -> List[Dict[str, Any]]:
        """Detect swapped prime pairs across languages."""
        issues = []
        
        langs = list(normalized_primes.keys())
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                lang1, lang2 = langs[i], langs[j]
                primes1 = normalized_primes[lang1]
                primes2 = normalized_primes[lang2]
                
                for prime1, prime2 in self.swapped_prime_pairs:
                    if prime1 in primes1 and prime2 in primes2:
                        issues.append({
                            'languages': [lang1, lang2],
                            'swapped_primes': [prime1, prime2],
                            'description': f"Prime {prime1} in {lang1} vs {prime2} in {lang2}"
                        })
                    elif prime2 in primes1 and prime1 in primes2:
                        issues.append({
                            'languages': [lang1, lang2],
                            'swapped_primes': [prime2, prime1],
                            'description': f"Prime {prime2} in {lang1} vs {prime1} in {lang2}"
                        })
        
        return issues


class SynsetStabilityGate:
    """Synset stability gate using tiered WSD → BabelNet."""
    
    def __init__(self, improvement_threshold: float = 10.0):
        """Initialize the synset stability gate."""
        self.improvement_threshold = improvement_threshold
        
        # Tiered WSD confidence levels
        self.wsd_tiers = {
            'high': 0.9,    # Role heuristics + SBERT + frequency
            'medium': 0.7,  # SBERT + frequency
            'low': 0.5,     # Frequency only
            'unknown': 0.0  # Sense-unknown fallback
        }
        
        # Common sense drift patterns
        self.sense_drift_patterns = {
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
    
    def evaluate_parallel_item(self, item: ParallelItem) -> GateResult:
        """Evaluate synset stability across languages."""
        logger.info(f"Evaluating synset stability for item {item.item_id}")
        
        # Analyze synset stability
        stability_analysis = self._analyze_synset_stability(item)
        
        # Calculate stability scores
        stability_scores = {}
        drift_issues = []
        
        for lang_pair in stability_analysis['language_pairs']:
            lang1, lang2 = lang_pair
            pair_key = f"{lang1}-{lang2}"
            
            # Calculate top-1 agreement
            top1_agreement = stability_analysis['top1_agreement'].get(pair_key, 0.0)
            stability_scores[pair_key] = top1_agreement
            
            # Check for sense drift
            drift = stability_analysis['sense_drift'].get(pair_key, [])
            if drift:
                drift_issues.extend(drift)
        
        # Calculate overall stability score
        overall_score = np.mean(list(stability_scores.values())) if stability_scores else 0.0
        
        # Determine if gate passes (improvement over baseline)
        passed = overall_score >= self.improvement_threshold and len(drift_issues) == 0
        
        details = {
            'stability_scores': stability_scores,
            'stability_analysis': stability_analysis,
            'drift_issues': drift_issues,
            'language_pairs': stability_analysis['language_pairs']
        }
        
        return GateResult(
            gate_type=GateType.SYNSET_STABILITY,
            test_name=f"synset_stability_{item.item_id}",
            languages=list(item.synsets.keys()),
            score=overall_score,
            threshold=self.improvement_threshold,
            passed=passed,
            details=details,
            timestamp=time.time()
        )
    
    def _analyze_synset_stability(self, item: ParallelItem) -> Dict[str, Any]:
        """Analyze synset stability across languages."""
        analysis = {
            'language_pairs': [],
            'top1_agreement': {},
            'sense_drift': {},
            'confidence_distribution': defaultdict(int)
        }
        
        langs = list(item.synsets.keys())
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                lang1, lang2 = langs[i], langs[j]
                pair_key = f"{lang1}-{lang2}"
                analysis['language_pairs'].append((lang1, lang2))
                
                # Get synsets for this pair
                synsets1 = item.synsets.get(lang1, [])
                synsets2 = item.synsets.get(lang2, [])
                
                # Calculate top-1 agreement
                top1_agreement = self._calculate_top1_agreement(synsets1, synsets2)
                analysis['top1_agreement'][pair_key] = top1_agreement
                
                # Detect sense drift
                drift = self._detect_sense_drift(synsets1, synsets2, lang1, lang2)
                analysis['sense_drift'][pair_key] = drift
                
                # Track confidence distribution
                for synset in synsets1 + synsets2:
                    confidence = self._get_synset_confidence(synset)
                    analysis['confidence_distribution'][confidence] += 1
        
        return analysis
    
    def _calculate_top1_agreement(self, synsets1: List[str], synsets2: List[str]) -> float:
        """Calculate top-1 synset agreement between two languages."""
        if not synsets1 or not synsets2:
            return 0.0
        
        # Get top synset from each language
        top1_1 = synsets1[0] if synsets1 else None
        top1_2 = synsets2[0] if synsets2 else None
        
        if top1_1 == top1_2:
            return 1.0
        else:
            return 0.0
    
    def _detect_sense_drift(self, synsets1: List[str], synsets2: List[str], 
                           lang1: str, lang2: str) -> List[Dict[str, Any]]:
        """Detect sense drift between languages."""
        drift_issues = []
        
        # Check for known drift patterns
        for word, expected_senses in self.sense_drift_patterns.get(lang1, {}).items():
            for synset in synsets1:
                if any(sense in synset for sense in expected_senses):
                    # Check if corresponding word in lang2 has different sense
                    for word2, expected_senses2 in self.sense_drift_patterns.get(lang2, {}).items():
                        for synset2 in synsets2:
                            if any(sense in synset2 for sense in expected_senses2):
                                if synset != synset2:
                                    drift_issues.append({
                                        'languages': [lang1, lang2],
                                        'synsets': [synset, synset2],
                                        'description': f"Sense drift detected: {synset} vs {synset2}"
                                    })
        
        return drift_issues
    
    def _get_synset_confidence(self, synset: str) -> str:
        """Get confidence tier for a synset."""
        # Simple heuristic based on synset format
        if 'v.01' in synset or 'n.01' in synset:
            return 'high'
        elif 'v.02' in synset or 'n.02' in synset:
            return 'medium'
        elif 'v.03' in synset or 'n.03' in synset:
            return 'low'
        else:
            return 'unknown'


class ConfusionMatrixGate:
    """Confusion matrix for frequently swapped primes."""
    
    def __init__(self):
        """Initialize the confusion matrix gate."""
        self.frequently_swapped_primes = {
            'WANT': ['LIKE', 'NEED'],
            'LIKE': ['WANT', 'NEED'],
            'NEED': ['WANT', 'LIKE'],
            'NOT': ['NO'],
            'NO': ['NOT'],
            'BEFORE': ['AFTER'],
            'AFTER': ['BEFORE'],
            'CAN': ['MUST'],
            'MUST': ['CAN'],
            'SOME': ['ALL'],
            'ALL': ['SOME']
        }
    
    def evaluate_parallel_item(self, item: ParallelItem) -> GateResult:
        """Evaluate confusion matrix for frequently swapped primes."""
        logger.info(f"Evaluating confusion matrix for item {item.item_id}")
        
        # Build confusion matrix
        confusion_matrix = self._build_confusion_matrix(item.detected_primes)
        
        # Calculate confusion metrics
        confusion_metrics = self._calculate_confusion_metrics(confusion_matrix)
        
        # Determine if gate passes (low confusion)
        passed = confusion_metrics['overall_confusion_rate'] < 0.1  # Less than 10% confusion
        
        details = {
            'confusion_matrix': confusion_matrix,
            'confusion_metrics': confusion_metrics,
            'frequently_swapped_primes': self.frequently_swapped_primes
        }
        
        return GateResult(
            gate_type=GateType.CONFUSION_MATRIX,
            test_name=f"confusion_matrix_{item.item_id}",
            languages=list(item.detected_primes.keys()),
            score=1.0 - confusion_metrics['overall_confusion_rate'],  # Invert confusion rate
            threshold=0.9,  # 90% accuracy
            passed=passed,
            details=details,
            timestamp=time.time()
        )
    
    def _build_confusion_matrix(self, detected_primes: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        """Build confusion matrix for frequently swapped primes."""
        matrix = defaultdict(lambda: defaultdict(int))
        
        # Count occurrences of each prime across languages
        prime_counts = defaultdict(lambda: defaultdict(int))
        for lang, primes in detected_primes.items():
            for prime in primes:
                prime_counts[prime][lang] += 1
        
        # Build confusion matrix for frequently swapped pairs
        for prime1, swapped_primes in self.frequently_swapped_primes.items():
            for prime2 in swapped_primes:
                # Count co-occurrences across languages
                for lang in detected_primes.keys():
                    if prime_counts[prime1][lang] > 0 and prime_counts[prime2][lang] > 0:
                        matrix[prime1][prime2] += 1
                        matrix[prime2][prime1] += 1
        
        return dict(matrix)
    
    def _calculate_confusion_metrics(self, confusion_matrix: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """Calculate confusion metrics from matrix."""
        total_confusions = 0
        total_occurrences = 0
        
        for prime1, confusions in confusion_matrix.items():
            for prime2, count in confusions.items():
                if prime1 != prime2:
                    total_confusions += count
                total_occurrences += count
        
        overall_confusion_rate = total_confusions / total_occurrences if total_occurrences > 0 else 0.0
        
        # Calculate per-prime confusion rates
        per_prime_confusion = {}
        for prime in confusion_matrix.keys():
            total_prime_confusions = sum(confusion_matrix[prime].values())
            self_confusions = confusion_matrix[prime].get(prime, 0)
            other_confusions = total_prime_confusions - self_confusions
            per_prime_confusion[prime] = other_confusions / total_prime_confusions if total_prime_confusions > 0 else 0.0
        
        return {
            'overall_confusion_rate': overall_confusion_rate,
            'per_prime_confusion': per_prime_confusion,
            'total_confusions': total_confusions,
            'total_occurrences': total_occurrences
        }


class CrossLanguageTruthGateSystem:
    """Comprehensive cross-language truth gate system."""
    
    def __init__(self):
        """Initialize the cross-language truth gate system."""
        self.jaccard_gate = PrimeSetJaccardGate(threshold=0.85)
        self.synset_gate = SynsetStabilityGate(improvement_threshold=10.0)
        self.confusion_gate = ConfusionMatrixGate()
    
    def evaluate_parallel_items(self, items: List[ParallelItem]) -> Dict[str, Any]:
        """Evaluate all parallel items using all gates."""
        logger.info(f"Evaluating {len(items)} parallel items with cross-language truth gates")
        
        results = {
            'evaluation_config': {
                'num_items': len(items),
                'jaccard_threshold': self.jaccard_gate.threshold,
                'synset_improvement_threshold': self.synset_gate.improvement_threshold,
                'timestamp': time.time()
            },
            'gate_results': {
                'jaccard': [],
                'synset_stability': [],
                'confusion_matrix': []
            },
            'summary': {},
            'recommendations': []
        }
        
        # Run all gates on all items
        for item in items:
            # Jaccard gate
            jaccard_result = self.jaccard_gate.evaluate_parallel_item(item)
            results['gate_results']['jaccard'].append(jaccard_result.to_dict())
            
            # Synset stability gate
            synset_result = self.synset_gate.evaluate_parallel_item(item)
            results['gate_results']['synset_stability'].append(synset_result.to_dict())
            
            # Confusion matrix gate
            confusion_result = self.confusion_gate.evaluate_parallel_item(item)
            results['gate_results']['confusion_matrix'].append(confusion_result.to_dict())
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary_statistics(results['gate_results'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['summary'])
        
        return results
    
    def _calculate_summary_statistics(self, gate_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate summary statistics for all gate results."""
        summary = {
            'jaccard_gate': {
                'passed': 0,
                'failed': 0,
                'average_score': 0.0,
                'score_distribution': defaultdict(int)
            },
            'synset_stability_gate': {
                'passed': 0,
                'failed': 0,
                'average_score': 0.0,
                'score_distribution': defaultdict(int)
            },
            'confusion_matrix_gate': {
                'passed': 0,
                'failed': 0,
                'average_score': 0.0,
                'score_distribution': defaultdict(int)
            }
        }
        
        # Analyze each gate type
        for gate_type, results in gate_results.items():
            if not results:
                continue
                
            passed = sum(1 for r in results if r['passed'])
            failed = len(results) - passed
            scores = [r['score'] for r in results]
            avg_score = np.mean(scores) if scores else 0.0
            
            summary[f"{gate_type}_gate"]['passed'] = passed
            summary[f"{gate_type}_gate"]['failed'] = failed
            summary[f"{gate_type}_gate"]['average_score'] = avg_score
            
            # Score distribution
            for score in scores:
                if score >= 0.9:
                    summary[f"{gate_type}_gate"]['score_distribution']['excellent'] += 1
                elif score >= 0.8:
                    summary[f"{gate_type}_gate"]['score_distribution']['good'] += 1
                elif score >= 0.7:
                    summary[f"{gate_type}_gate"]['score_distribution']['fair'] += 1
                else:
                    summary[f"{gate_type}_gate"]['score_distribution']['poor'] += 1
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations from summary statistics."""
        recommendations = []
        
        # Jaccard gate recommendations
        jaccard_summary = summary.get('jaccard_gate', {})
        if jaccard_summary.get('failed', 0) > 0:
            recommendations.append(f"Improve prime-set alignment: {jaccard_summary['failed']} items failed Jaccard gate")
        
        if jaccard_summary.get('average_score', 0) < 0.85:
            recommendations.append(f"Boost Jaccard scores: current average {jaccard_summary['average_score']:.3f} below 0.85 threshold")
        
        # Synset stability recommendations
        synset_summary = summary.get('synset_stability_gate', {})
        if synset_summary.get('failed', 0) > 0:
            recommendations.append(f"Fix synset drift: {synset_summary['failed']} items failed stability gate")
        
        if synset_summary.get('average_score', 0) < 10.0:
            recommendations.append(f"Improve synset stability: current average {synset_summary['average_score']:.1f} below 10.0 threshold")
        
        # Confusion matrix recommendations
        confusion_summary = summary.get('confusion_matrix_gate', {})
        if confusion_summary.get('failed', 0) > 0:
            recommendations.append(f"Reduce prime confusion: {confusion_summary['failed']} items failed confusion gate")
        
        # Overall recommendations
        total_passed = (jaccard_summary.get('passed', 0) + 
                       synset_summary.get('passed', 0) + 
                       confusion_summary.get('passed', 0))
        total_tests = (jaccard_summary.get('passed', 0) + jaccard_summary.get('failed', 0) +
                      synset_summary.get('passed', 0) + synset_summary.get('failed', 0) +
                      confusion_summary.get('passed', 0) + confusion_summary.get('failed', 0))
        
        if total_tests > 0:
            pass_rate = total_passed / total_tests
            if pass_rate < 0.9:
                recommendations.append(f"Overall pass rate {pass_rate:.1%} below 90% target")
        
        return recommendations


def main():
    """Main function to demonstrate cross-language truth gates."""
    logger.info("Starting cross-language truth gates demonstration...")
    
    # Initialize system
    system = CrossLanguageTruthGateSystem()
    
    # Sample parallel items
    parallel_items = [
        ParallelItem(
            item_id="test_001",
            texts={
                'en': "I want to help you",
                'es': "Quiero ayudarte",
                'fr': "Je veux t'aider"
            },
            explications={
                'en': "WANT(i, HELP(i, you))",
                'es': "WANT(i, HELP(i, you))",
                'fr': "WANT(i, HELP(i, you))"
            },
            detected_primes={
                'en': ['WANT', 'HELP'],
                'es': ['WANT', 'HELP'],
                'fr': ['WANT', 'HELP']
            },
            synsets={
                'en': ['want.v.01', 'help.v.01'],
                'es': ['querer.v.01', 'ayudar.v.01'],
                'fr': ['vouloir.v.01', 'aider.v.01']
            }
        ),
        ParallelItem(
            item_id="test_002",
            texts={
                'en': "I do not like this weather",
                'es': "No me gusta este tiempo",
                'fr': "Je n'aime pas ce temps"
            },
            explications={
                'en': "NOT LIKE(i, weather(this))",
                'es': "NOT LIKE(i, weather(this))",
                'fr': "NOT LIKE(i, weather(this))"
            },
            detected_primes={
                'en': ['NOT', 'LIKE'],
                'es': ['NOT', 'LIKE'],
                'fr': ['NOT', 'LIKE']
            },
            synsets={
                'en': ['not.r.01', 'like.v.01'],
                'es': ['no.r.01', 'gustar.v.01'],
                'fr': ['ne.r.01', 'aimer.v.01']
            }
        ),
        ParallelItem(
            item_id="test_003",
            texts={
                'en': "All children can play here",
                'es': "Todos los niños pueden jugar aquí",
                'fr': "Tous les enfants peuvent jouer ici"
            },
            explications={
                'en': "ALL(child, CAN(play(child))) HERE",
                'es': "ALL(child, CAN(play(child))) HERE",
                'fr': "ALL(child, CAN(play(child))) HERE"
            },
            detected_primes={
                'en': ['ALL', 'CAN', 'HERE'],
                'es': ['ALL', 'CAN', 'HERE'],
                'fr': ['ALL', 'CAN', 'HERE']
            },
            synsets={
                'en': ['all.a.01', 'can.v.01', 'here.r.01'],
                'es': ['todo.a.01', 'poder.v.01', 'aquí.r.01'],
                'fr': ['tout.a.01', 'pouvoir.v.01', 'ici.r.01']
            }
        )
    ]
    
    # Run evaluation
    results = system.evaluate_parallel_items(parallel_items)
    
    # Print results
    print("\n" + "="*80)
    print("CROSS-LANGUAGE TRUTH GATES RESULTS")
    print("="*80)
    
    print(f"Evaluation Configuration:")
    print(f"  Number of Items: {results['evaluation_config']['num_items']}")
    print(f"  Jaccard Threshold: {results['evaluation_config']['jaccard_threshold']}")
    print(f"  Synset Improvement Threshold: {results['evaluation_config']['synset_improvement_threshold']}")
    
    print(f"\nSummary Statistics:")
    summary = results['summary']
    
    for gate_name, gate_summary in summary.items():
        print(f"  {gate_name.replace('_', ' ').title()}:")
        print(f"    Passed: {gate_summary['passed']}")
        print(f"    Failed: {gate_summary['failed']}")
        print(f"    Average Score: {gate_summary['average_score']:.3f}")
        print(f"    Score Distribution: {dict(gate_summary['score_distribution'])}")
    
    print(f"\nSample Gate Results:")
    for i, item in enumerate(parallel_items[:2]):
        print(f"  Item {i+1}: {item.item_id}")
        jaccard_result = results['gate_results']['jaccard'][i]
        synset_result = results['gate_results']['synset_stability'][i]
        confusion_result = results['gate_results']['confusion_matrix'][i]
        
        print(f"    Jaccard: {jaccard_result['score']:.3f} ({'PASS' if jaccard_result['passed'] else 'FAIL'})")
        print(f"    Synset Stability: {synset_result['score']:.3f} ({'PASS' if synset_result['passed'] else 'FAIL'})")
        print(f"    Confusion Matrix: {confusion_result['score']:.3f} ({'PASS' if confusion_result['passed'] else 'FAIL'})")
        print()
    
    print(f"\nRecommendations:")
    for i, recommendation in enumerate(results['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    # Save results
    output_path = Path("data/cross_language_truth_gates_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert results to JSON-serializable format
    try:
        json_results = convert_numpy_types(results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        # Save a simplified version
        simplified_results = {
            'evaluation_config': results['evaluation_config'],
            'summary': results['summary'],
            'recommendations': results['recommendations']
        }
        with open(output_path, 'w') as f:
            json.dump(simplified_results, f, indent=2)
    
    logger.info(f"Cross-language truth gates results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Cross-language truth gates demonstration completed!")
    print("="*80)


if __name__ == "__main__":
    main()

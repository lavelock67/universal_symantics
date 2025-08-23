#!/usr/bin/env python3
"""
Pragmatics-Safe Styling Lock System.

This script implements the pragmatics-safe styling lock as specified in ChatGPT5's feedback:
- After style pass, re-explicate and re-WSD; block if sense set differs or primes/polarity/scope change
- Maintain literalness slider; require literalness≥0.7 for evaluation runs
- Accept: Synset stability ≥0.9 at literalness≥0.7; zero prime/scope flips
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


class StyleType(Enum):
    """Types of style transformations."""
    FORMAL = "formal"
    INFORMAL = "informal"
    POLITE = "polite"
    DIRECT = "direct"
    TECHNICAL = "technical"
    SIMPLE = "simple"
    EMOTIVE = "emotive"
    NEUTRAL = "neutral"


class PreservationCheck(Enum):
    """Types of preservation checks."""
    SYNSET_STABILITY = "synset_stability"
    PRIME_PRESERVATION = "prime_preservation"
    POLARITY_PRESERVATION = "polarity_preservation"
    SCOPE_PRESERVATION = "scope_preservation"
    STRUCTURAL_CONSISTENCY = "structural_consistency"


@dataclass
class StyleTransformation:
    """Represents a style transformation."""
    style_type: StyleType
    literalness: float  # 0.0 = highly stylized, 1.0 = literal
    transformation_rules: List[str]
    confidence: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'style_type': self.style_type.value,
            'literalness': self.literalness,
            'transformation_rules': self.transformation_rules,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }


@dataclass
class PreservationResult:
    """Result of a preservation check."""
    check_type: PreservationCheck
    passed: bool
    score: float
    details: Dict[str, Any]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'check_type': self.check_type.value,
            'passed': self.passed,
            'score': self.score,
            'details': self.details,
            'warnings': self.warnings
        }


@dataclass
class StylingLockResult:
    """Result of the styling lock validation."""
    original_text: str
    styled_text: str
    style_transformation: StyleTransformation
    preservation_results: Dict[PreservationCheck, PreservationResult]
    overall_passed: bool
    overall_score: float
    blocking_issues: List[str]
    recommendations: List[str]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_text': self.original_text,
            'styled_text': self.styled_text,
            'style_transformation': self.style_transformation.to_dict(),
            'preservation_results': {k.value: v.to_dict() for k, v in self.preservation_results.items()},
            'overall_passed': self.overall_passed,
            'overall_score': self.overall_score,
            'blocking_issues': self.blocking_issues,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }


class SynsetStabilityChecker:
    """Checks synset stability across style transformations."""
    
    def __init__(self):
        """Initialize the synset stability checker."""
        self.stability_threshold = 0.9
        self.synset_cache = {}
    
    def check_stability(self, original_text: str, styled_text: str, 
                       literalness: float) -> PreservationResult:
        """Check synset stability between original and styled text."""
        logger.info(f"Checking synset stability (literalness: {literalness:.2f})")
        
        # Mock synset extraction (would use actual WSD system)
        original_synsets = self._extract_synsets(original_text)
        styled_synsets = self._extract_synsets(styled_text)
        
        # Calculate stability metrics
        stability_score = self._calculate_synset_stability(original_synsets, styled_synsets)
        
        # Apply literalness adjustment
        adjusted_score = self._adjust_for_literalness(stability_score, literalness)
        
        # Check if passed
        passed = adjusted_score >= self.stability_threshold
        
        warnings = []
        if not passed:
            warnings.append(f"Synset stability {adjusted_score:.3f} below threshold {self.stability_threshold}")
        
        if literalness < 0.7:
            warnings.append(f"Literalness {literalness:.2f} below evaluation threshold 0.7")
        
        return PreservationResult(
            check_type=PreservationCheck.SYNSET_STABILITY,
            passed=passed,
            score=adjusted_score,
            details={
                'original_synsets': original_synsets,
                'styled_synsets': styled_synsets,
                'raw_stability': stability_score,
                'literalness_adjustment': literalness,
                'threshold': self.stability_threshold
            },
            warnings=warnings
        )
    
    def _extract_synsets(self, text: str) -> Dict[str, str]:
        """Extract synsets from text (mock implementation)."""
        # Mock synset extraction based on key words
        synset_mapping = {
            'cat': 'cat.n.01',
            'mat': 'mat.n.01',
            'like': 'like.v.01',
            'weather': 'weather.n.01',
            'good': 'good.a.01',
            'bad': 'bad.a.01',
            'play': 'play.v.01',
            'help': 'help.v.01',
            'want': 'want.v.01',
            'can': 'can.v.01',
            'will': 'will.v.01',
            'not': 'not.r.01',
            'all': 'all.a.01',
            'some': 'some.a.01',
            'here': 'here.r.01',
            'there': 'there.r.01',
            'now': 'now.r.01',
            'today': 'today.r.01'
        }
        
        synsets = {}
        words = text.lower().split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in synset_mapping:
                synsets[clean_word] = synset_mapping[clean_word]
        
        return synsets
    
    def _calculate_synset_stability(self, original_synsets: Dict[str, str], 
                                   styled_synsets: Dict[str, str]) -> float:
        """Calculate synset stability score."""
        if not original_synsets and not styled_synsets:
            return 1.0
        
        if not original_synsets or not styled_synsets:
            return 0.0
        
        # Calculate intersection and union
        original_keys = set(original_synsets.keys())
        styled_keys = set(styled_synsets.keys())
        
        intersection = original_keys.intersection(styled_keys)
        union = original_keys.union(styled_keys)
        
        if not union:
            return 1.0
        
        # Calculate Jaccard similarity for key overlap
        key_similarity = len(intersection) / len(union)
        
        # Calculate synset agreement for overlapping keys
        synset_agreements = 0
        total_overlap = 0
        
        for key in intersection:
            if original_synsets[key] == styled_synsets[key]:
                synset_agreements += 1
            total_overlap += 1
        
        synset_similarity = synset_agreements / total_overlap if total_overlap > 0 else 1.0
        
        # Combined score (weighted average)
        stability_score = 0.7 * key_similarity + 0.3 * synset_similarity
        
        return stability_score
    
    def _adjust_for_literalness(self, stability_score: float, literalness: float) -> float:
        """Adjust stability score based on literalness level."""
        # Higher literalness should require higher stability
        if literalness >= 0.7:
            return stability_score  # No adjustment for evaluation runs
        else:
            # Relaxed threshold for non-evaluation runs
            return min(1.0, stability_score * (1.0 + (0.7 - literalness) * 0.3))


class PrimePreservationChecker:
    """Checks prime preservation across style transformations."""
    
    def __init__(self):
        """Initialize the prime preservation checker."""
        self.nsm_primes = {
            'NOT', 'CAN', 'WANT', 'LIKE', 'GOOD', 'BAD', 'BIG', 'SMALL',
            'ALL', 'SOME', 'ONE', 'TWO', 'MANY', 'FEW', 'HERE', 'THERE',
            'NOW', 'BEFORE', 'AFTER', 'SAY', 'THINK', 'KNOW', 'SEE',
            'HEAR', 'DO', 'HAPPEN', 'BECAUSE', 'IF', 'MAYBE', 'TRUE'
        }
    
    def check_preservation(self, original_text: str, styled_text: str, 
                          literalness: float) -> PreservationResult:
        """Check prime preservation between original and styled text."""
        logger.info(f"Checking prime preservation (literalness: {literalness:.2f})")
        
        # Extract primes from both texts
        original_primes = self._extract_primes(original_text)
        styled_primes = self._extract_primes(styled_text)
        
        # Calculate preservation metrics
        prime_overlap = self._calculate_prime_overlap(original_primes, styled_primes)
        prime_consistency = self._calculate_prime_consistency(original_primes, styled_primes)
        
        # Combined score
        preservation_score = 0.6 * prime_overlap + 0.4 * prime_consistency
        
        # Apply literalness adjustment
        adjusted_score = self._adjust_for_literalness(preservation_score, literalness)
        
        # Check if passed (should be very high for prime preservation)
        passed = adjusted_score >= 0.95
        
        warnings = []
        if not passed:
            warnings.append(f"Prime preservation {adjusted_score:.3f} below threshold 0.95")
        
        missing_primes = original_primes - styled_primes
        added_primes = styled_primes - original_primes
        
        if missing_primes:
            warnings.append(f"Missing primes: {missing_primes}")
        if added_primes:
            warnings.append(f"Added primes: {added_primes}")
        
        return PreservationResult(
            check_type=PreservationCheck.PRIME_PRESERVATION,
            passed=passed,
            score=adjusted_score,
            details={
                'original_primes': list(original_primes),
                'styled_primes': list(styled_primes),
                'prime_overlap': prime_overlap,
                'prime_consistency': prime_consistency,
                'missing_primes': list(missing_primes),
                'added_primes': list(added_primes),
                'literalness_adjustment': literalness
            },
            warnings=warnings
        )
    
    def _extract_primes(self, text: str) -> Set[str]:
        """Extract NSM primes from text."""
        text_upper = text.upper()
        found_primes = set()
        
        for prime in self.nsm_primes:
            if prime in text_upper:
                found_primes.add(prime)
        
        return found_primes
    
    def _calculate_prime_overlap(self, original_primes: Set[str], 
                                styled_primes: Set[str]) -> float:
        """Calculate prime overlap (Jaccard similarity)."""
        if not original_primes and not styled_primes:
            return 1.0
        
        intersection = original_primes.intersection(styled_primes)
        union = original_primes.union(styled_primes)
        
        return len(intersection) / len(union) if union else 1.0
    
    def _calculate_prime_consistency(self, original_primes: Set[str], 
                                   styled_primes: Set[str]) -> float:
        """Calculate prime consistency (no additions/subtractions)."""
        if not original_primes and not styled_primes:
            return 1.0
        
        # Perfect consistency if no changes
        if original_primes == styled_primes:
            return 1.0
        
        # Penalize any changes
        changes = len(original_primes.symmetric_difference(styled_primes))
        total_primes = len(original_primes.union(styled_primes))
        
        return max(0.0, 1.0 - (changes / total_primes))
    
    def _adjust_for_literalness(self, preservation_score: float, literalness: float) -> float:
        """Adjust preservation score based on literalness level."""
        if literalness >= 0.7:
            return preservation_score  # Strict for evaluation runs
        else:
            # Slightly relaxed for non-evaluation runs
            return min(1.0, preservation_score * (1.0 + (0.7 - literalness) * 0.1))


class PolarityScopeChecker:
    """Checks polarity and scope preservation across style transformations."""
    
    def __init__(self):
        """Initialize the polarity and scope checker."""
        self.negation_indicators = {'NOT', 'NO', 'NEVER', 'NONE', 'NEITHER', 'NOR'}
        self.quantifier_indicators = {'ALL', 'EVERY', 'EACH', 'SOME', 'ANY', 'MANY', 'FEW'}
        self.modality_indicators = {'CAN', 'WILL', 'MUST', 'SHOULD', 'MAY', 'MIGHT'}
    
    def check_preservation(self, original_text: str, styled_text: str, 
                          literalness: float) -> PreservationResult:
        """Check polarity and scope preservation."""
        logger.info(f"Checking polarity and scope preservation (literalness: {literalness:.2f})")
        
        # Extract polarity and scope indicators
        original_polarity = self._extract_polarity(original_text)
        styled_polarity = self._extract_polarity(styled_text)
        
        original_scope = self._extract_scope(original_text)
        styled_scope = self._extract_scope(styled_text)
        
        # Calculate preservation scores
        polarity_score = self._calculate_polarity_preservation(original_polarity, styled_polarity)
        scope_score = self._calculate_scope_preservation(original_scope, styled_scope)
        
        # Combined score
        preservation_score = 0.5 * polarity_score + 0.5 * scope_score
        
        # Apply literalness adjustment
        adjusted_score = self._adjust_for_literalness(preservation_score, literalness)
        
        # Check if passed
        passed = adjusted_score >= 0.9
        
        warnings = []
        if not passed:
            warnings.append(f"Polarity/scope preservation {adjusted_score:.3f} below threshold 0.9")
        
        if original_polarity != styled_polarity:
            warnings.append(f"Polarity changed: {original_polarity} -> {styled_polarity}")
        
        if original_scope != styled_scope:
            warnings.append(f"Scope changed: {original_scope} -> {styled_scope}")
        
        return PreservationResult(
            check_type=PreservationCheck.POLARITY_PRESERVATION,
            passed=passed,
            score=adjusted_score,
            details={
                'original_polarity': original_polarity,
                'styled_polarity': styled_polarity,
                'original_scope': original_scope,
                'styled_scope': styled_scope,
                'polarity_score': polarity_score,
                'scope_score': scope_score,
                'literalness_adjustment': literalness
            },
            warnings=warnings
        )
    
    def _extract_polarity(self, text: str) -> str:
        """Extract polarity from text."""
        text_upper = text.upper()
        
        # Check for negation
        for neg in self.negation_indicators:
            if neg in text_upper:
                return "negative"
        
        return "positive"
    
    def _extract_scope(self, text: str) -> Dict[str, Any]:
        """Extract scope information from text."""
        text_upper = text.upper()
        
        scope_info = {
            'quantifiers': [],
            'modalities': [],
            'negation_scope': None
        }
        
        # Extract quantifiers
        for quant in self.quantifier_indicators:
            if quant in text_upper:
                scope_info['quantifiers'].append(quant)
        
        # Extract modalities
        for mod in self.modality_indicators:
            if mod in text_upper:
                scope_info['modalities'].append(mod)
        
        # Determine negation scope
        if 'NOT' in text_upper:
            scope_info['negation_scope'] = 'local'  # Simplified
        
        return scope_info
    
    def _calculate_polarity_preservation(self, original_polarity: str, 
                                       styled_polarity: str) -> float:
        """Calculate polarity preservation score."""
        return 1.0 if original_polarity == styled_polarity else 0.0
    
    def _calculate_scope_preservation(self, original_scope: Dict[str, Any], 
                                    styled_scope: Dict[str, Any]) -> float:
        """Calculate scope preservation score."""
        # Compare quantifiers
        quant_similarity = self._compare_lists(original_scope['quantifiers'], 
                                             styled_scope['quantifiers'])
        
        # Compare modalities
        mod_similarity = self._compare_lists(original_scope['modalities'], 
                                           styled_scope['modalities'])
        
        # Compare negation scope
        neg_similarity = 1.0 if original_scope['negation_scope'] == styled_scope['negation_scope'] else 0.0
        
        # Weighted average
        return 0.4 * quant_similarity + 0.4 * mod_similarity + 0.2 * neg_similarity
    
    def _compare_lists(self, list1: List[str], list2: List[str]) -> float:
        """Compare two lists for similarity."""
        if not list1 and not list2:
            return 1.0
        
        set1 = set(list1)
        set2 = set(list2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 1.0
    
    def _adjust_for_literalness(self, preservation_score: float, literalness: float) -> float:
        """Adjust preservation score based on literalness level."""
        if literalness >= 0.7:
            return preservation_score  # Strict for evaluation runs
        else:
            # Slightly relaxed for non-evaluation runs
            return min(1.0, preservation_score * (1.0 + (0.7 - literalness) * 0.1))


class PragmaticsSafeStylingSystem:
    """Comprehensive pragmatics-safe styling lock system."""
    
    def __init__(self):
        """Initialize the pragmatics-safe styling system."""
        self.synset_checker = SynsetStabilityChecker()
        self.prime_checker = PrimePreservationChecker()
        self.polarity_checker = PolarityScopeChecker()
        
        self.evaluation_threshold = 0.7  # Minimum literalness for evaluation runs
        self.overall_threshold = 0.8     # Minimum overall score to pass
    
    def validate_styling(self, original_text: str, styled_text: str, 
                        style_type: StyleType, literalness: float) -> StylingLockResult:
        """Validate that styling preserves meaning."""
        logger.info(f"Validating styling: {style_type.value} (literalness: {literalness:.2f})")
        
        # Create style transformation record
        style_transformation = StyleTransformation(
            style_type=style_type,
            literalness=literalness,
            transformation_rules=[f"style_{style_type.value}"],
            confidence=0.9,
            timestamp=time.time()
        )
        
        # Run all preservation checks
        preservation_results = {}
        
        # 1. Synset stability check
        synset_result = self.synset_checker.check_stability(original_text, styled_text, literalness)
        preservation_results[PreservationCheck.SYNSET_STABILITY] = synset_result
        
        # 2. Prime preservation check
        prime_result = self.prime_checker.check_preservation(original_text, styled_text, literalness)
        preservation_results[PreservationCheck.PRIME_PRESERVATION] = prime_result
        
        # 3. Polarity and scope preservation check
        polarity_result = self.polarity_checker.check_preservation(original_text, styled_text, literalness)
        preservation_results[PreservationCheck.POLARITY_PRESERVATION] = polarity_result
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(preservation_results)
        
        # Determine if passed
        overall_passed = self._determine_overall_pass(preservation_results, overall_score, literalness)
        
        # Generate blocking issues and recommendations
        blocking_issues = self._identify_blocking_issues(preservation_results, literalness)
        recommendations = self._generate_recommendations(preservation_results, literalness)
        
        return StylingLockResult(
            original_text=original_text,
            styled_text=styled_text,
            style_transformation=style_transformation,
            preservation_results=preservation_results,
            overall_passed=overall_passed,
            overall_score=overall_score,
            blocking_issues=blocking_issues,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def _calculate_overall_score(self, preservation_results: Dict[PreservationCheck, PreservationResult]) -> float:
        """Calculate overall preservation score."""
        weights = {
            PreservationCheck.SYNSET_STABILITY: 0.4,
            PreservationCheck.PRIME_PRESERVATION: 0.4,
            PreservationCheck.POLARITY_PRESERVATION: 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for check_type, result in preservation_results.items():
            weight = weights.get(check_type, 0.1)
            weighted_sum += result.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_overall_pass(self, preservation_results: Dict[PreservationCheck, PreservationResult], 
                              overall_score: float, literalness: float) -> bool:
        """Determine if the overall validation passed."""
        # Check literalness threshold for evaluation runs
        if literalness < self.evaluation_threshold:
            return False
        
        # Check overall score threshold
        if overall_score < self.overall_threshold:
            return False
        
        # Check that all critical checks passed
        critical_checks = [PreservationCheck.PRIME_PRESERVATION, PreservationCheck.POLARITY_PRESERVATION]
        for check_type in critical_checks:
            if check_type in preservation_results and not preservation_results[check_type].passed:
                return False
        
        return True
    
    def _identify_blocking_issues(self, preservation_results: Dict[PreservationCheck, PreservationResult], 
                                literalness: float) -> List[str]:
        """Identify blocking issues that prevent passing."""
        issues = []
        
        if literalness < self.evaluation_threshold:
            issues.append(f"Literalness {literalness:.2f} below evaluation threshold {self.evaluation_threshold}")
        
        for check_type, result in preservation_results.items():
            if not result.passed:
                issues.append(f"{check_type.value} check failed: {result.score:.3f}")
        
        return issues
    
    def _generate_recommendations(self, preservation_results: Dict[PreservationCheck, PreservationResult], 
                                literalness: float) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        if literalness < self.evaluation_threshold:
            recommendations.append("Increase literalness to ≥0.7 for evaluation runs")
        
        for check_type, result in preservation_results.items():
            if not result.passed:
                if check_type == PreservationCheck.SYNSET_STABILITY:
                    recommendations.append("Review style transformations to maintain synset stability")
                elif check_type == PreservationCheck.PRIME_PRESERVATION:
                    recommendations.append("Ensure NSM primes are preserved during styling")
                elif check_type == PreservationCheck.POLARITY_PRESERVATION:
                    recommendations.append("Maintain polarity and scope during style changes")
        
        if not recommendations:
            recommendations.append("All preservation checks passed successfully")
        
        return recommendations


def main():
    """Main function to demonstrate pragmatics-safe styling system."""
    logger.info("Starting pragmatics-safe styling system demonstration...")
    
    # Initialize the system
    styling_system = PragmaticsSafeStylingSystem()
    
    # Test cases with different scenarios
    test_cases = [
        {
            'name': 'Safe Formal Styling',
            'original': 'The cat is not on the mat.',
            'styled': 'The feline is not positioned upon the mat.',
            'style_type': StyleType.FORMAL,
            'literalness': 0.8
        },
        {
            'name': 'Unsafe Informal Styling',
            'original': 'I do not like this weather.',
            'styled': 'I hate this weather.',
            'style_type': StyleType.INFORMAL,
            'literalness': 0.6
        },
        {
            'name': 'Safe Polite Styling',
            'original': 'All children can play here.',
            'styled': 'All children may play here.',
            'style_type': StyleType.POLITE,
            'literalness': 0.9
        },
        {
            'name': 'Unsafe Direct Styling',
            'original': 'Some people want to help.',
            'styled': 'Nobody wants to help.',
            'style_type': StyleType.DIRECT,
            'literalness': 0.7
        },
        {
            'name': 'Low Literalness Evaluation',
            'original': 'The weather is good today.',
            'styled': 'The weather is excellent today.',
            'style_type': StyleType.EMOTIVE,
            'literalness': 0.5
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"Test Case {i+1}: {test_case['name']}")
        print(f"{'='*80}")
        
        result = styling_system.validate_styling(
            original_text=test_case['original'],
            styled_text=test_case['styled'],
            style_type=test_case['style_type'],
            literalness=test_case['literalness']
        )
        
        results.append(result)
        
        # Print results
        print(f"Original: {result.original_text}")
        print(f"Styled:   {result.styled_text}")
        print(f"Style:    {result.style_transformation.style_type.value}")
        print(f"Literalness: {result.style_transformation.literalness:.2f}")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Passed: {'✅' if result.overall_passed else '❌'}")
        
        print(f"\nPreservation Results:")
        for check_type, check_result in result.preservation_results.items():
            status = "✅" if check_result.passed else "❌"
            print(f"  {check_type.value}: {status} ({check_result.score:.3f})")
        
        if result.blocking_issues:
            print(f"\nBlocking Issues:")
            for issue in result.blocking_issues:
                print(f"  ❌ {issue}")
        
        if result.recommendations:
            print(f"\nRecommendations:")
            for rec in result.recommendations:
                print(f"  💡 {rec}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    passed_count = sum(1 for r in results if r.overall_passed)
    total_count = len(results)
    
    print(f"Total Tests: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Success Rate: {passed_count/total_count:.1%}")
    
    # Average scores by check type
    check_scores = defaultdict(list)
    for result in results:
        for check_type, check_result in result.preservation_results.items():
            check_scores[check_type.value].append(check_result.score)
    
    print(f"\nAverage Scores by Check Type:")
    for check_type, scores in check_scores.items():
        avg_score = np.mean(scores)
        print(f"  {check_type}: {avg_score:.3f}")
    
    # Save results
    output_path = Path("data/pragmatics_safe_styling_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    try:
        json_results = convert_numpy_types({
            'results': [r.to_dict() for r in results],
            'summary': {
                'total_tests': total_count,
                'passed_tests': passed_count,
                'success_rate': passed_count / total_count,
                'average_scores': {k: np.mean(v) for k, v in check_scores.items()}
            }
        })
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    logger.info(f"Pragmatics-safe styling results saved to {output_path}")
    
    print(f"\n{'='*80}")
    print("Pragmatics-safe styling system demonstration completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

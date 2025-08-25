#!/usr/bin/env python3
"""
Risk-Coverage Router for Selective Correctness

This module implements a risk-coverage router that wraps detectors and decoders
with selective correctness, as specified in the plan.
"""

import logging
import math
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RouterDecision(Enum):
    """Router decisions."""
    TRANSLATE = "translate"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"

@dataclass
class RiskCoverageConfig:
    """Configuration for risk-coverage routing."""
    legality_threshold: float = 0.9
    roundtrip_drift_threshold: float = 0.15
    mdl_delta_threshold: float = 0.0
    sense_confidence_threshold: float = 0.7
    coverage_buckets: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.coverage_buckets is None:
            self.coverage_buckets = [
                (0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)
            ]

@dataclass
class RouterResult:
    """Result of risk-coverage routing."""
    decision: RouterDecision
    risk_estimate: float
    coverage_bucket: str
    reasons: List[str]
    confidence: float
    metadata: Dict[str, Any]

class RiskCoverageRouter:
    """Risk-coverage router for selective correctness."""
    
    def __init__(self, config: Optional[RiskCoverageConfig] = None):
        """Initialize the risk-coverage router.
        
        Args:
            config: Router configuration
        """
        self.config = config or RiskCoverageConfig()
        self.risk_history: List[float] = []
        self.coverage_history: List[float] = []
        self.decision_history: List[RouterDecision] = []
    
    def route_detection(self, detection_result: Dict[str, Any]) -> RouterResult:
        """Route detection result with risk assessment.
        
        Args:
            detection_result: Result from NSM detection
            
        Returns:
            Router result with decision and metadata
        """
        reasons = []
        risk_factors = []
        
        # Check legality
        legality = detection_result.get('legality_score', 0.0)
        if legality < self.config.legality_threshold:
            reasons.append(f"low_legality({legality:.3f})")
            risk_factors.append(1.0 - legality)
        
        # Check sense confidence
        sense_confidence = detection_result.get('sense_confidence', 1.0)
        if sense_confidence < self.config.sense_confidence_threshold:
            reasons.append(f"low_sense_confidence({sense_confidence:.3f})")
            risk_factors.append(1.0 - sense_confidence)
        
        # Check coverage
        coverage = detection_result.get('coverage', 1.0)
        coverage_bucket = self._get_coverage_bucket(coverage)
        
        # Calculate risk estimate
        risk_estimate = self._calculate_risk_estimate(risk_factors, coverage)
        
        # Make decision
        decision = self._make_decision(risk_estimate, coverage, reasons)
        
        # Update history
        self._update_history(risk_estimate, coverage, decision)
        
        return RouterResult(
            decision=decision,
            risk_estimate=risk_estimate,
            coverage_bucket=coverage_bucket,
            reasons=reasons,
            confidence=1.0 - risk_estimate,
            metadata={
                "legality": legality,
                "sense_confidence": sense_confidence,
                "coverage": coverage,
                "risk_factors": risk_factors
            }
        )
    
    def route_generation(self, generation_result: Dict[str, Any], 
                        original_text: str) -> RouterResult:
        """Route generation result with round-trip validation.
        
        Args:
            generation_result: Result from NSM generation
            original_text: Original input text
            
        Returns:
            Router result with decision and metadata
        """
        reasons = []
        risk_factors = []
        
        # Check legality
        legality = generation_result.get('legality_score', 0.0)
        if legality < self.config.legality_threshold:
            reasons.append(f"low_legality({legality:.3f})")
            risk_factors.append(1.0 - legality)
        
        # Check round-trip drift
        roundtrip_drift = self._calculate_roundtrip_drift(generation_result, original_text)
        if roundtrip_drift > self.config.roundtrip_drift_threshold:
            reasons.append(f"high_roundtrip_drift({roundtrip_drift:.3f})")
            risk_factors.append(roundtrip_drift)
        
        # Check MDL delta
        mdl_delta = generation_result.get('mdl_delta', 0.0)
        if mdl_delta > self.config.mdl_delta_threshold:
            reasons.append(f"positive_mdl_delta({mdl_delta:.3f})")
            risk_factors.append(mdl_delta)
        
        # Check coverage
        coverage = generation_result.get('coverage', 1.0)
        coverage_bucket = self._get_coverage_bucket(coverage)
        
        # Calculate risk estimate
        risk_estimate = self._calculate_risk_estimate(risk_factors, coverage)
        
        # Make decision
        decision = self._make_decision(risk_estimate, coverage, reasons)
        
        # Update history
        self._update_history(risk_estimate, coverage, decision)
        
        return RouterResult(
            decision=decision,
            risk_estimate=risk_estimate,
            coverage_bucket=coverage_bucket,
            reasons=reasons,
            confidence=1.0 - risk_estimate,
            metadata={
                "legality": legality,
                "roundtrip_drift": roundtrip_drift,
                "mdl_delta": mdl_delta,
                "coverage": coverage,
                "risk_factors": risk_factors
            }
        )
    
    def _calculate_risk_estimate(self, risk_factors: List[float], coverage: float) -> float:
        """Calculate risk estimate from risk factors and coverage.
        
        Args:
            risk_factors: List of risk factors
            coverage: Coverage score
            
        Returns:
            Risk estimate (0.0 = no risk, 1.0 = high risk)
        """
        if not risk_factors:
            return 0.0
        
        # Combine risk factors (max of individual risks)
        combined_risk = max(risk_factors)
        
        # Adjust for coverage (lower coverage = higher risk)
        coverage_penalty = 1.0 - coverage
        adjusted_risk = combined_risk + (coverage_penalty * 0.3)
        
        return min(1.0, adjusted_risk)
    
    def _get_coverage_bucket(self, coverage: float) -> str:
        """Get coverage bucket for the given coverage score.
        
        Args:
            coverage: Coverage score
            
        Returns:
            Coverage bucket string
        """
        for low, high in self.config.coverage_buckets:
            if low <= coverage < high:
                return f"{low:.1f}-{high:.1f}"
        return "1.0"
    
    def _make_decision(self, risk_estimate: float, coverage: float, 
                      reasons: List[str]) -> RouterDecision:
        """Make routing decision based on risk and coverage.
        
        Args:
            risk_estimate: Risk estimate
            coverage: Coverage score
            reasons: List of risk reasons
            
        Returns:
            Router decision
        """
        # High risk -> abstain
        if risk_estimate > 0.7:
            return RouterDecision.ABSTAIN
        
        # Medium risk with low coverage -> clarify
        if risk_estimate > 0.3 and coverage < 0.5:
            return RouterDecision.CLARIFY
        
        # Low risk or high coverage -> translate
        return RouterDecision.TRANSLATE
    
    def _calculate_roundtrip_drift(self, generation_result: Dict[str, Any], 
                                 original_text: str) -> float:
        """Calculate round-trip drift between original and generated text.
        
        Args:
            generation_result: Generation result
            original_text: Original input text
            
        Returns:
            Round-trip drift score
        """
        # This is a simplified implementation
        # In practice, you'd re-parse the generated text and compare to original
        
        generated_text = generation_result.get('generated_text', '')
        original_primes = generation_result.get('original_primes', [])
        generated_primes = generation_result.get('generated_primes', [])
        
        # Calculate prime overlap
        if not original_primes or not generated_primes:
            return 0.5  # Default drift for missing primes
        
        overlap = len(set(original_primes) & set(generated_primes))
        total = len(set(original_primes) | set(generated_primes))
        
        if total == 0:
            return 1.0
        
        drift = 1.0 - (overlap / total)
        return drift
    
    def _update_history(self, risk_estimate: float, coverage: float, 
                       decision: RouterDecision):
        """Update router history.
        
        Args:
            risk_estimate: Risk estimate
            coverage: Coverage score
            decision: Router decision
        """
        self.risk_history.append(risk_estimate)
        self.coverage_history.append(coverage)
        self.decision_history.append(decision)
        
        # Keep only last 1000 entries
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
            self.coverage_history = self.coverage_history[-1000:]
            self.decision_history = self.decision_history[-1000:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics.
        
        Returns:
            Router statistics
        """
        if not self.risk_history:
            return {
                "total_decisions": 0,
                "decision_distribution": {},
                "avg_risk": 0.0,
                "avg_coverage": 0.0
            }
        
        # Decision distribution
        decision_counts = {}
        for decision in RouterDecision:
            decision_counts[decision.value] = self.decision_history.count(decision)
        
        return {
            "total_decisions": len(self.decision_history),
            "decision_distribution": decision_counts,
            "avg_risk": sum(self.risk_history) / len(self.risk_history),
            "avg_coverage": sum(self.coverage_history) / len(self.coverage_history),
            "risk_histogram": self._create_histogram(self.risk_history, 10),
            "coverage_histogram": self._create_histogram(self.coverage_history, 10)
        }
    
    def _create_histogram(self, values: List[float], bins: int) -> Dict[str, int]:
        """Create histogram from values.
        
        Args:
            values: List of values
            bins: Number of bins
            
        Returns:
            Histogram as dictionary
        """
        if not values:
            return {}
        
        min_val = min(values)
        max_val = max(values)
        bin_size = (max_val - min_val) / bins
        
        histogram = {}
        for i in range(bins):
            bin_start = min_val + (i * bin_size)
            bin_end = min_val + ((i + 1) * bin_size)
            bin_label = f"{bin_start:.2f}-{bin_end:.2f}"
            
            count = sum(1 for v in values if bin_start <= v < bin_end)
            histogram[bin_label] = count
        
        return histogram

class SelectiveCorrectnessWrapper:
    """Wrapper for detectors and decoders with selective correctness."""
    
    def __init__(self, router: RiskCoverageRouter):
        """Initialize the selective correctness wrapper.
        
        Args:
            router: Risk-coverage router
        """
        self.router = router
    
    def detect_with_router(self, text: str, detector_func) -> Dict[str, Any]:
        """Run detection with risk-coverage routing.
        
        Args:
            text: Input text
            detector_func: Detection function
            
        Returns:
            Detection result with routing decision
        """
        # Run detection
        detection_result = detector_func(text)
        
        # Route result
        router_result = self.router.route_detection(detection_result)
        
        # Add routing information
        detection_result['router_decision'] = router_result.decision.value
        detection_result['risk_estimate'] = router_result.risk_estimate
        detection_result['coverage_bucket'] = router_result.coverage_bucket
        detection_result['router_reasons'] = router_result.reasons
        detection_result['router_confidence'] = router_result.confidence
        
        return detection_result
    
    def generate_with_router(self, prompt: str, generator_func) -> Dict[str, Any]:
        """Run generation with risk-coverage routing.
        
        Args:
            prompt: Input prompt
            generator_func: Generation function
            
        Returns:
            Generation result with routing decision
        """
        # Run generation
        generation_result = generator_func(prompt)
        
        # Route result
        router_result = self.router.route_generation(generation_result, prompt)
        
        # Add routing information
        generation_result['router_decision'] = router_result.decision.value
        generation_result['risk_estimate'] = router_result.risk_estimate
        generation_result['coverage_bucket'] = router_result.coverage_bucket
        generation_result['router_reasons'] = router_result.reasons
        generation_result['router_confidence'] = router_result.confidence
        
        return generation_result

def create_risk_coverage_router(config: Optional[RiskCoverageConfig] = None) -> RiskCoverageRouter:
    """Create a risk-coverage router.
    
    Args:
        config: Router configuration (optional)
        
    Returns:
        Risk-coverage router
    """
    return RiskCoverageRouter(config)

def create_selective_correctness_wrapper(config: Optional[RiskCoverageConfig] = None) -> SelectiveCorrectnessWrapper:
    """Create a selective correctness wrapper.
    
    Args:
        config: Router configuration (optional)
        
    Returns:
        Selective correctness wrapper
    """
    router = create_risk_coverage_router(config)
    return SelectiveCorrectnessWrapper(router)

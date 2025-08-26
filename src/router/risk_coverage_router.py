#!/usr/bin/env python3
"""Risk-Coverage Router for Selective Correctness.

Makes intelligent translate/clarify/abstain decisions based on:
- Legality scores
- Round-trip drift
- MDL delta
- Sense confidence
- Coverage metrics
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class RouterDecision(Enum):
    """Router decision types."""
    TRANSLATE = "translate"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


@dataclass
class RouterResult:
    """Result from risk-coverage router."""
    decision: RouterDecision
    risk_estimate: float
    coverage_bucket: str
    confidence: float
    reasons: List[str]
    metadata: Dict[str, Any]


class RiskCoverageRouter:
    """Risk-coverage router for selective correctness."""
    
    def __init__(self):
        """Initialize the router with thresholds and statistics."""
        # Risk thresholds
        self.legality_threshold = 0.9
        self.drift_threshold = 0.15
        self.mdl_threshold = 0.0
        self.confidence_threshold = 0.7
        self.coverage_threshold = 0.6
        
        # Coverage buckets
        self.coverage_buckets = [
            (0.0, 0.2, "0.0-0.2"),
            (0.2, 0.4, "0.2-0.4"),
            (0.4, 0.6, "0.4-0.6"),
            (0.6, 0.8, "0.6-0.8"),
            (0.8, 1.0, "0.8-1.0")
        ]
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "decisions": Counter(),
            "risk_distribution": defaultdict(int),
            "coverage_distribution": defaultdict(int),
            "error_rates": defaultdict(float),
            "avg_processing_time": 0.0
        }
        
        # Performance tracking
        self.processing_times = []
        
        logger.info("Risk-coverage router initialized")
    
    def route_detection(self, text: str, detected_primes: List[str], 
                       legality_score: Optional[float] = None,
                       sense_confidence: Optional[float] = None) -> RouterResult:
        """Route detection requests.
        
        Args:
            text: Input text
            detected_primes: List of detected NSM primes
            legality_score: Grammar legality score (0-1)
            sense_confidence: Sense disambiguation confidence (0-1)
            
        Returns:
            Router decision with metadata
        """
        start_time = time.time()
        
        try:
            # Calculate coverage
            coverage = self._calculate_coverage(detected_primes, text)
            coverage_bucket = self._get_coverage_bucket(coverage)
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(
                text, detected_primes, legality_score, sense_confidence, coverage
            )
            
            # Make decision
            decision, confidence, reasons = self._make_decision(risk_factors)
            
            # Calculate overall risk estimate
            risk_estimate = self._calculate_risk_estimate(risk_factors)
            
            # Create result
            result = RouterResult(
                decision=decision,
                risk_estimate=risk_estimate,
                coverage_bucket=coverage_bucket,
                confidence=confidence,
                reasons=reasons,
                metadata={
                    "coverage": coverage,
                    "risk_factors": risk_factors,
                    "processing_time": time.time() - start_time
                }
            )
            
            # Update statistics
            self._update_stats(result, time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Router error: {e}")
            # Default to abstain on error
            return RouterResult(
                decision=RouterDecision.ABSTAIN,
                risk_estimate=1.0,
                coverage_bucket="0.0-0.2",
                confidence=0.0,
                reasons=[f"router_error: {str(e)}"],
                metadata={"error": str(e)}
            )
    
    def route_generation(self, generation_result: Dict[str, Any], 
                        original_text: str) -> RouterResult:
        """Route generation requests.
        
        Args:
            generation_result: Result from generation system
            original_text: Original input text
            
        Returns:
            Router decision with metadata
        """
        start_time = time.time()
        
        try:
            # Extract metrics from generation result
            legality = generation_result.get("legality", 0.0)
            drift = generation_result.get("drift", {}).get("graph_f1", 0.0)
            mdl_delta = generation_result.get("mdl_delta", 0.0)
            
            # Calculate coverage from generated primes
            generated_primes = generation_result.get("generated_primes", [])
            coverage = self._calculate_coverage(generated_primes, original_text)
            coverage_bucket = self._get_coverage_bucket(coverage)
            
            # Calculate risk factors
            risk_factors = {
                "legality": legality,
                "drift": 1.0 - drift,  # Convert to risk
                "mdl_delta": max(0, mdl_delta),  # Only positive deltas are risky
                "coverage": 1.0 - coverage,  # Convert to risk
                "confidence": 1.0 - generation_result.get("confidence", 0.0)
            }
            
            # Make decision
            decision, confidence, reasons = self._make_decision(risk_factors)
            
            # Calculate overall risk estimate
            risk_estimate = self._calculate_risk_estimate(risk_factors)
            
            # Create result
            result = RouterResult(
                decision=decision,
                risk_estimate=risk_estimate,
                coverage_bucket=coverage_bucket,
                confidence=confidence,
                reasons=reasons,
                metadata={
                    "coverage": coverage,
                    "risk_factors": risk_factors,
                    "processing_time": time.time() - start_time
                }
            )
            
            # Update statistics
            self._update_stats(result, time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Router error: {e}")
            return RouterResult(
                decision=RouterDecision.ABSTAIN,
                risk_estimate=1.0,
                coverage_bucket="0.0-0.2",
                confidence=0.0,
                reasons=[f"router_error: {str(e)}"],
                metadata={"error": str(e)}
            )
    
    def _calculate_coverage(self, primes: List[str], text: str) -> float:
        """Calculate coverage of detected primes relative to text complexity."""
        if not primes or not text:
            return 0.0
        
        # Simple coverage: ratio of detected primes to expected primes
        # Expected primes based on text length and complexity
        words = text.split()
        expected_primes = min(len(words) * 0.3, 10)  # Assume 30% of words could be primes, max 10
        
        return min(len(primes) / expected_primes, 1.0) if expected_primes > 0 else 0.0
    
    def _get_coverage_bucket(self, coverage: float) -> str:
        """Get coverage bucket for the given coverage value."""
        for min_cov, max_cov, bucket in self.coverage_buckets:
            if min_cov <= coverage < max_cov:
                return bucket
        return "0.8-1.0"  # Default to highest bucket
    
    def _calculate_risk_factors(self, text: str, primes: List[str], 
                               legality: Optional[float], 
                               sense_confidence: Optional[float],
                               coverage: float) -> Dict[str, float]:
        """Calculate individual risk factors."""
        risk_factors = {}
        
        # Legality risk (inverse of legality score)
        risk_factors["legality"] = 1.0 - (legality or 0.5)
        
        # Coverage risk (inverse of coverage)
        risk_factors["coverage"] = 1.0 - coverage
        
        # Sense confidence risk (inverse of confidence)
        risk_factors["sense_confidence"] = 1.0 - (sense_confidence or 0.5)
        
        # Complexity risk (based on text length and prime diversity)
        complexity_risk = min(len(text.split()) / 20.0, 1.0)  # Normalize by 20 words
        risk_factors["complexity"] = complexity_risk
        
        # Prime diversity risk (too few or too many primes)
        prime_count = len(primes)
        if prime_count == 0:
            diversity_risk = 1.0
        elif prime_count < 2:
            diversity_risk = 0.7
        elif prime_count > 8:
            diversity_risk = 0.5
        else:
            diversity_risk = 0.2
        risk_factors["diversity"] = diversity_risk
        
        return risk_factors
    
    def _make_decision(self, risk_factors: Dict[str, float]) -> Tuple[RouterDecision, float, List[str]]:
        """Make routing decision based on risk factors."""
        reasons = []
        
        # Calculate weighted risk score
        weights = {
            "legality": 0.3,
            "coverage": 0.25,
            "sense_confidence": 0.2,
            "complexity": 0.15,
            "diversity": 0.1
        }
        
        weighted_risk = sum(risk_factors.get(factor, 0.0) * weight 
                           for factor, weight in weights.items())
        
        # Determine decision based on risk thresholds
        if weighted_risk <= 0.2:
            decision = RouterDecision.TRANSLATE
            confidence = 0.9
            reasons.append("low_risk")
        elif weighted_risk <= 0.4:
            decision = RouterDecision.TRANSLATE
            confidence = 0.7
            reasons.append("medium_risk")
        elif weighted_risk <= 0.6:
            decision = RouterDecision.CLARIFY
            confidence = 0.5
            reasons.append("high_risk")
        else:
            decision = RouterDecision.ABSTAIN
            confidence = 0.3
            reasons.append("very_high_risk")
        
        # Add specific reasons
        for factor, risk in risk_factors.items():
            if risk > 0.7:
                reasons.append(f"high_{factor}_risk")
            elif risk > 0.5:
                reasons.append(f"medium_{factor}_risk")
        
        return decision, confidence, reasons
    
    def _calculate_risk_estimate(self, risk_factors: Dict[str, float]) -> float:
        """Calculate overall risk estimate."""
        weights = {
            "legality": 0.3,
            "coverage": 0.25,
            "sense_confidence": 0.2,
            "complexity": 0.15,
            "diversity": 0.1
        }
        
        return sum(risk_factors.get(factor, 0.0) * weight 
                  for factor, weight in weights.items())
    
    def _update_stats(self, result: RouterResult, processing_time: float):
        """Update router statistics."""
        self.stats["total_requests"] += 1
        self.stats["decisions"][result.decision.value] += 1
        self.stats["coverage_distribution"][result.coverage_bucket] += 1
        
        # Update risk distribution
        risk_bucket = f"{int(result.risk_estimate * 10) * 10}-{(int(result.risk_estimate * 10) + 1) * 10}"
        self.stats["risk_distribution"][risk_bucket] += 1
        
        # Update processing time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 1000:
            self.processing_times.pop(0)
        self.stats["avg_processing_time"] = sum(self.processing_times) / len(self.processing_times)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        total = self.stats["total_requests"]
        if total == 0:
            return self.stats
        
        # Calculate decision percentages
        decision_pcts = {}
        for decision, count in self.stats["decisions"].items():
            decision_pcts[decision] = count / total
        
        # Calculate coverage percentages
        coverage_pcts = {}
        for bucket, count in self.stats["coverage_distribution"].items():
            coverage_pcts[bucket] = count / total
        
        # Calculate risk percentages
        risk_pcts = {}
        for bucket, count in self.stats["risk_distribution"].items():
            risk_pcts[bucket] = count / total
        
        return {
            "total_requests": total,
            "decision_distribution": dict(self.stats["decisions"]),
            "decision_percentages": decision_pcts,
            "coverage_distribution": dict(self.stats["coverage_distribution"]),
            "coverage_percentages": coverage_pcts,
            "risk_distribution": dict(self.stats["risk_distribution"]),
            "risk_percentages": risk_pcts,
            "avg_processing_time": self.stats["avg_processing_time"],
            "error_rates": dict(self.stats["error_rates"])
        }
    
    def reset_statistics(self):
        """Reset router statistics."""
        self.stats = {
            "total_requests": 0,
            "decisions": Counter(),
            "risk_distribution": defaultdict(int),
            "coverage_distribution": defaultdict(int),
            "error_rates": defaultdict(float),
            "avg_processing_time": 0.0
        }
        self.processing_times = []
        logger.info("Router statistics reset")

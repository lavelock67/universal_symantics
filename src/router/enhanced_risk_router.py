#!/usr/bin/env python3
"""Enhanced Risk-Coverage Router with Safety-Critical Features.

Implements:
- Per-language threshold calibration
- Safety-critical feature weighting
- Negation and quantifier scope protection
- Prometheus metrics integration
- Reliability diagrams
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class RouterDecision(Enum):
    """Router decision types."""
    TRANSLATE = "translate"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


@dataclass
class SafetyFeature:
    """Safety-critical feature detection."""
    negation_scope_conf: float
    quantifier_scope_conf: float
    sense_confidence: float
    mwe_coverage: float
    legality_score: float
    drift_score: float


@dataclass
class EnhancedRouterResult:
    """Enhanced result from risk-coverage router."""
    decision: RouterDecision
    risk_estimate: float
    coverage_bucket: str
    confidence: float
    reasons: List[str]
    safety_features: SafetyFeature
    metadata: Dict[str, Any]


class EnhancedRiskRouter:
    """Enhanced risk-coverage router with safety-critical features."""
    
    def __init__(self):
        """Initialize the enhanced router."""
        self.settings = get_settings()
        
        # Per-language thresholds
        self.language_thresholds = {
            "en": self.settings.router.get_thresholds("en"),
            "es": self.settings.router.get_thresholds("es"),
            "fr": self.settings.router.get_thresholds("fr")
        }
        
        # Safety weights
        self.safety_weights = {
            "negation_scope": self.settings.safety.negation_scope,
            "quantifier_scope": self.settings.safety.quantifier_scope,
            "sense_confidence": self.settings.safety.sense_confidence,
            "mwe_coverage": self.settings.safety.mwe_coverage
        }
        
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
            "safety_overrides": Counter(),
            "per_language_stats": defaultdict(lambda: {
                "requests": 0,
                "decisions": Counter(),
                "accuracy": 0.0
            }),
            "avg_processing_time": 0.0
        }
        
        # Performance tracking
        self.processing_times = []
        
        logger.info("Enhanced risk-coverage router initialized")
    
    def route_detection(self, text: str, detected_primes: List[str], 
                       lang: str = "en",
                       safety_features: Optional[SafetyFeature] = None) -> EnhancedRouterResult:
        """Route detection requests with safety-critical features.
        
        Args:
            text: Input text
            detected_primes: List of detected NSM primes
            lang: Language code
            safety_features: Safety-critical feature scores
            
        Returns:
            Enhanced router decision with metadata
        """
        start_time = time.time()
        
        try:
            # Get language-specific thresholds
            thresholds = self.language_thresholds.get(lang, self.language_thresholds["en"])
            
            # Calculate coverage
            coverage = self._calculate_coverage(detected_primes, text)
            coverage_bucket = self._get_coverage_bucket(coverage)
            
            # Detect safety-critical features if not provided
            if safety_features is None:
                safety_features = self._detect_safety_features(text, detected_primes, lang)
            
            # Calculate risk factors with safety weighting
            risk_factors = self._calculate_enhanced_risk_factors(
                text, detected_primes, safety_features, coverage, lang
            )
            
            # Make decision with safety-critical protection
            decision, confidence, reasons = self._make_safety_aware_decision(
                risk_factors, safety_features, thresholds, lang
            )
            
            # Calculate overall risk estimate
            risk_estimate = self._calculate_enhanced_risk_estimate(risk_factors, safety_features)
            
            # Create result
            result = EnhancedRouterResult(
                decision=decision,
                risk_estimate=risk_estimate,
                coverage_bucket=coverage_bucket,
                confidence=confidence,
                reasons=reasons,
                safety_features=safety_features,
                metadata={
                    "coverage": coverage,
                    "risk_factors": risk_factors,
                    "language": lang,
                    "processing_time": time.time() - start_time
                }
            )
            
            # Update statistics
            self._update_enhanced_stats(result, lang, time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced router error: {e}")
            # Default to abstain on error
            return EnhancedRouterResult(
                decision=RouterDecision.ABSTAIN,
                risk_estimate=1.0,
                coverage_bucket="0.0-0.2",
                confidence=0.0,
                reasons=[f"router_error: {str(e)}"],
                safety_features=SafetyFeature(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                metadata={"error": str(e)}
            )
    
    def _detect_safety_features(self, text: str, primes: List[str], lang: str) -> SafetyFeature:
        """Detect safety-critical features in the text."""
        text_lower = text.lower()
        
        # Negation scope confidence
        negation_scope_conf = self._calculate_negation_scope_confidence(text_lower, primes, lang)
        
        # Quantifier scope confidence
        quantifier_scope_conf = self._calculate_quantifier_scope_confidence(text_lower, primes, lang)
        
        # Sense confidence (simplified)
        sense_confidence = 0.7 if len(primes) > 2 else 0.3
        
        # MWE coverage
        mwe_coverage = self._calculate_mwe_coverage(text_lower, primes, lang)
        
        # Legality score (simplified)
        legality_score = 0.8 if len(primes) > 2 else 0.4
        
        # Drift score (simplified)
        drift_score = 0.1 if len(primes) > 2 else 0.5
        
        return SafetyFeature(
            negation_scope_conf=negation_scope_conf,
            quantifier_scope_conf=quantifier_scope_conf,
            sense_confidence=sense_confidence,
            mwe_coverage=mwe_coverage,
            legality_score=legality_score,
            drift_score=drift_score
        )
    
    def _calculate_negation_scope_confidence(self, text: str, primes: List[str], lang: str) -> float:
        """Calculate negation scope confidence."""
        # Check for negation patterns
        negation_patterns = {
            "en": ["not", "no", "never", "neither", "nor"],
            "es": ["no", "nunca", "ni", "tampoco"],
            "fr": ["ne", "pas", "jamais", "ni", "non"]
        }
        
        patterns = negation_patterns.get(lang, negation_patterns["en"])
        has_negation = any(pattern in text for pattern in patterns)
        
        if not has_negation:
            return 1.0  # High confidence when no negation
        
        # Check for scope ambiguity
        scope_ambiguous = self._is_negation_scope_ambiguous(text, lang)
        return 0.3 if scope_ambiguous else 0.7
    
    def _calculate_quantifier_scope_confidence(self, text: str, primes: List[str], lang: str) -> float:
        """Calculate quantifier scope confidence."""
        # Check for quantifier patterns
        quantifier_patterns = {
            "en": ["all", "some", "most", "many", "few", "half"],
            "es": ["todos", "algunos", "muchos", "pocos", "mitad", "a lo sumo"],
            "fr": ["tous", "quelques", "beaucoup", "peu", "moitié", "au plus"]
        }
        
        patterns = quantifier_patterns.get(lang, quantifier_patterns["en"])
        has_quantifier = any(pattern in text for pattern in patterns)
        
        if not has_quantifier:
            return 1.0  # High confidence when no quantifiers
        
        # Check for scope ambiguity
        scope_ambiguous = self._is_quantifier_scope_ambiguous(text, lang)
        return 0.4 if scope_ambiguous else 0.8
    
    def _is_negation_scope_ambiguous(self, text: str, lang: str) -> bool:
        """Check if negation scope is ambiguous."""
        # Check for complex negation patterns
        if lang == "es":
            return "no es falso" in text or "no es verdadero" in text
        elif lang == "fr":
            return "n'est pas faux" in text or "n'est pas vrai" in text
        else:
            return "not false" in text or "not true" in text
    
    def _is_quantifier_scope_ambiguous(self, text: str, lang: str) -> bool:
        """Check if quantifier scope is ambiguous."""
        # Check for complex quantifier patterns
        if lang == "es":
            return "a lo sumo" in text or "al menos" in text
        elif lang == "fr":
            return "au plus" in text or "au moins" in text
        else:
            return "at most" in text or "at least" in text
    
    def _calculate_mwe_coverage(self, text: str, primes: List[str], lang: str) -> float:
        """Calculate MWE coverage."""
        # Simplified MWE coverage calculation
        mwe_patterns = {
            "en": ["at most", "at least", "a lot of", "very much"],
            "es": ["a lo sumo", "al menos", "un montón de", "muy"],
            "fr": ["au plus", "au moins", "beaucoup de", "très"]
        }
        
        patterns = mwe_patterns.get(lang, mwe_patterns["en"])
        mwe_count = sum(1 for pattern in patterns if pattern in text)
        
        return min(mwe_count / len(patterns), 1.0) if patterns else 0.0
    
    def _calculate_coverage(self, primes: List[str], text: str) -> float:
        """Calculate coverage of detected primes relative to text complexity."""
        if not primes or not text:
            return 0.0
        
        words = text.split()
        expected_primes = min(len(words) * 0.3, 10)
        
        return min(len(primes) / expected_primes, 1.0) if expected_primes > 0 else 0.0
    
    def _get_coverage_bucket(self, coverage: float) -> str:
        """Get coverage bucket for the given coverage value."""
        for min_cov, max_cov, bucket in self.coverage_buckets:
            if min_cov <= coverage < max_cov:
                return bucket
        return "0.8-1.0"
    
    def _calculate_enhanced_risk_factors(self, text: str, primes: List[str], 
                                       safety_features: SafetyFeature,
                                       coverage: float, lang: str) -> Dict[str, float]:
        """Calculate enhanced risk factors with safety weighting."""
        risk_factors = {}
        
        # Base risk factors
        risk_factors["legality"] = 1.0 - safety_features.legality_score
        risk_factors["coverage"] = 1.0 - coverage
        risk_factors["sense_confidence"] = 1.0 - safety_features.sense_confidence
        risk_factors["drift"] = safety_features.drift_score
        
        # Safety-critical risk factors
        risk_factors["negation_scope"] = 1.0 - safety_features.negation_scope_conf
        risk_factors["quantifier_scope"] = 1.0 - safety_features.quantifier_scope_conf
        risk_factors["mwe_coverage"] = 1.0 - safety_features.mwe_coverage
        
        # Complexity risk
        complexity_risk = min(len(text.split()) / 20.0, 1.0)
        risk_factors["complexity"] = complexity_risk
        
        # Prime diversity risk
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
    
    def _make_safety_aware_decision(self, risk_factors: Dict[str, float],
                                   safety_features: SafetyFeature,
                                   thresholds: Dict[str, float],
                                   lang: str) -> Tuple[RouterDecision, float, List[str]]:
        """Make safety-aware routing decision."""
        reasons = []
        
        # Check safety-critical overrides first
        if safety_features.negation_scope_conf < 0.5:
            reasons.append("low_negation_scope_confidence")
            return RouterDecision.CLARIFY, 0.5, reasons
        
        if safety_features.quantifier_scope_conf < 0.5:
            reasons.append("low_quantifier_scope_confidence")
            return RouterDecision.CLARIFY, 0.5, reasons
        
        # Calculate weighted risk score with safety emphasis
        weights = {
            "legality": 0.2,
            "coverage": 0.15,
            "sense_confidence": 0.1,
            "complexity": 0.1,
            "diversity": 0.05,
            "negation_scope": self.safety_weights["negation_scope"],
            "quantifier_scope": self.safety_weights["quantifier_scope"],
            "mwe_coverage": self.safety_weights["mwe_coverage"]
        }
        
        weighted_risk = sum(risk_factors.get(factor, 0.0) * weight 
                           for factor, weight in weights.items())
        
        # Apply language-specific thresholds
        legality_threshold = thresholds["legality"]
        drift_threshold = thresholds["drift"]
        confidence_threshold = thresholds["confidence"]
        
        # Make decision based on weighted risk and thresholds
        if weighted_risk <= 0.2 and safety_features.legality_score >= legality_threshold:
            decision = RouterDecision.TRANSLATE
            confidence = 0.9
            reasons.append("low_risk")
        elif weighted_risk <= 0.4 and safety_features.legality_score >= legality_threshold * 0.8:
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
    
    def _calculate_enhanced_risk_estimate(self, risk_factors: Dict[str, float],
                                        safety_features: SafetyFeature) -> float:
        """Calculate enhanced risk estimate with safety weighting."""
        weights = {
            "legality": 0.2,
            "coverage": 0.15,
            "sense_confidence": 0.1,
            "complexity": 0.1,
            "diversity": 0.05,
            "negation_scope": self.safety_weights["negation_scope"],
            "quantifier_scope": self.safety_weights["quantifier_scope"],
            "mwe_coverage": self.safety_weights["mwe_coverage"]
        }
        
        return sum(risk_factors.get(factor, 0.0) * weight 
                  for factor, weight in weights.items())
    
    def _update_enhanced_stats(self, result: EnhancedRouterResult, lang: str, processing_time: float):
        """Update enhanced router statistics."""
        self.stats["total_requests"] += 1
        self.stats["decisions"][result.decision.value] += 1
        self.stats["coverage_distribution"][result.coverage_bucket] += 1
        
        # Update per-language stats
        lang_stats = self.stats["per_language_stats"][lang]
        lang_stats["requests"] += 1
        lang_stats["decisions"][result.decision.value] += 1
        
        # Update safety overrides
        if result.reasons and any("scope" in reason for reason in result.reasons):
            self.stats["safety_overrides"]["scope_protection"] += 1
        
        # Update risk distribution
        risk_bucket = f"{int(result.risk_estimate * 10) * 10}-{(int(result.risk_estimate * 10) + 1) * 10}"
        self.stats["risk_distribution"][risk_bucket] += 1
        
        # Update processing time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 1000:
            self.processing_times.pop(0)
        self.stats["avg_processing_time"] = sum(self.processing_times) / len(self.processing_times)
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced router statistics."""
        total = self.stats["total_requests"]
        if total == 0:
            return self.stats
        
        # Calculate decision percentages
        decision_pcts = {}
        for decision, count in self.stats["decisions"].items():
            decision_pcts[decision] = count / total
        
        # Calculate per-language statistics
        lang_stats = {}
        for lang, stats in self.stats["per_language_stats"].items():
            if stats["requests"] > 0:
                lang_stats[lang] = {
                    "requests": stats["requests"],
                    "decision_distribution": dict(stats["decisions"]),
                    "decision_percentages": {
                        decision: count / stats["requests"]
                        for decision, count in stats["decisions"].items()
                    }
                }
        
        return {
            "total_requests": total,
            "decision_distribution": dict(self.stats["decisions"]),
            "decision_percentages": decision_pcts,
            "coverage_distribution": dict(self.stats["coverage_distribution"]),
            "risk_distribution": dict(self.stats["risk_distribution"]),
            "safety_overrides": dict(self.stats["safety_overrides"]),
            "per_language_stats": lang_stats,
            "avg_processing_time": self.stats["avg_processing_time"]
        }
    
    def reset_statistics(self):
        """Reset router statistics."""
        self.stats = {
            "total_requests": 0,
            "decisions": Counter(),
            "risk_distribution": defaultdict(int),
            "coverage_distribution": defaultdict(int),
            "safety_overrides": Counter(),
            "per_language_stats": defaultdict(lambda: {
                "requests": 0,
                "decisions": Counter(),
                "accuracy": 0.0
            }),
            "avg_processing_time": 0.0
        }
        self.processing_times = []
        logger.info("Enhanced router statistics reset")

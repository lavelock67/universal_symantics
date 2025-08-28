"""
EIL Router

Decision layer that routes based on legality, scope confidence, and graph-F1.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

from .graph import EILGraph
from .validator import ValidationResult


class RouterDecision(Enum):
    """Router decisions."""
    TRANSLATE = "translate"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"
    REGENERATE = "regenerate"


@dataclass
class RouterResult:
    """Result of router decision."""
    decision: RouterDecision
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


class EILRouter:
    """Routes EIL graphs based on validation and confidence metrics."""
    
    def __init__(self):
        """Initialize the router."""
        # Thresholds
        self.legality_threshold = 0.9
        self.scope_threshold = 0.8
        self.confidence_threshold = 0.7
        self.graph_f1_threshold = 0.8
        
        # Language-specific thresholds
        self.language_thresholds = {
            "en": {"legality": 0.9, "scope": 0.8, "confidence": 0.7},
            "es": {"legality": 0.85, "scope": 0.75, "confidence": 0.65},
            "fr": {"legality": 0.85, "scope": 0.75, "confidence": 0.65},
            "de": {"legality": 0.8, "scope": 0.7, "confidence": 0.6},
            "zh": {"legality": 0.75, "scope": 0.65, "confidence": 0.55},
            "ja": {"legality": 0.75, "scope": 0.65, "confidence": 0.55}
        }
        
        # Hard rules for specific cases
        self.hard_rules = {
            "negation_scope": True,  # Must have clear negation scope
            "quantifier_scope": True,  # Must have clear quantifier scope
            "modal_scope": True,  # Must have clear modal scope
            "max_errors": 3,  # Maximum validation errors before abstain
            "max_warnings": 10  # Maximum warnings before clarify
        }
    
    def route(self, 
              source_graph: EILGraph, 
              validation_result: ValidationResult,
              target_language: str = "en",
              round_trip_f1: Optional[float] = None) -> RouterResult:
        """Route based on validation results and optional round-trip F1."""
        
        # Get language-specific thresholds
        thresholds = self.language_thresholds.get(target_language, self.language_thresholds["en"])
        
        # Calculate decision metrics
        legality_score = self._calculate_legality_score(validation_result)
        scope_score = self._calculate_scope_score(validation_result)
        confidence_score = self._calculate_confidence_score(source_graph)
        
        # Apply hard rules
        hard_rule_violations = self._check_hard_rules(validation_result)
        
        # Make decision
        if hard_rule_violations:
            return RouterResult(
                decision=RouterDecision.ABSTAIN,
                confidence=0.0,
                reasoning=f"Hard rule violations: {', '.join(hard_rule_violations)}",
                metadata={
                    "legality_score": legality_score,
                    "scope_score": scope_score,
                    "confidence_score": confidence_score,
                    "hard_rule_violations": hard_rule_violations
                }
            )
        
        # Check for regeneration (low confidence)
        if confidence_score < thresholds["confidence"] * 0.5:
            return RouterResult(
                decision=RouterDecision.REGENERATE,
                confidence=confidence_score,
                reasoning=f"Low confidence score: {confidence_score:.3f} < {thresholds['confidence'] * 0.5:.3f}",
                metadata={
                    "legality_score": legality_score,
                    "scope_score": scope_score,
                    "confidence_score": confidence_score
                }
            )
        
        # Check for abstain (multiple threshold failures)
        failed_thresholds = []
        if legality_score < thresholds["legality"]:
            failed_thresholds.append(f"legality ({legality_score:.3f} < {thresholds['legality']:.3f})")
        if scope_score < thresholds["scope"]:
            failed_thresholds.append(f"scope ({scope_score:.3f} < {thresholds['scope']:.3f})")
        if confidence_score < thresholds["confidence"]:
            failed_thresholds.append(f"confidence ({confidence_score:.3f} < {thresholds['confidence']:.3f})")
        
        if len(failed_thresholds) >= 2:
            return RouterResult(
                decision=RouterDecision.ABSTAIN,
                confidence=min(legality_score, scope_score, confidence_score),
                reasoning=f"Multiple threshold failures: {', '.join(failed_thresholds)}",
                metadata={
                    "legality_score": legality_score,
                    "scope_score": scope_score,
                    "confidence_score": confidence_score,
                    "failed_thresholds": failed_thresholds
                }
            )
        
        # Check for clarify (warnings or single threshold failure)
        if (len(validation_result.warnings) > self.hard_rules["max_warnings"] or 
            len(failed_thresholds) == 1):
            return RouterResult(
                decision=RouterDecision.CLARIFY,
                confidence=min(legality_score, scope_score, confidence_score),
                reasoning=f"Clarification needed: {'warnings' if len(validation_result.warnings) > self.hard_rules['max_warnings'] else failed_thresholds[0]}",
                metadata={
                    "legality_score": legality_score,
                    "scope_score": scope_score,
                    "confidence_score": confidence_score,
                    "warning_count": len(validation_result.warnings),
                    "failed_thresholds": failed_thresholds
                }
            )
        
        # Check round-trip F1 if provided
        if round_trip_f1 is not None and round_trip_f1 < self.graph_f1_threshold:
            return RouterResult(
                decision=RouterDecision.CLARIFY,
                confidence=round_trip_f1,
                reasoning=f"Low round-trip F1: {round_trip_f1:.3f} < {self.graph_f1_threshold:.3f}",
                metadata={
                    "legality_score": legality_score,
                    "scope_score": scope_score,
                    "confidence_score": confidence_score,
                    "round_trip_f1": round_trip_f1
                }
            )
        
        # Default to translate
        return RouterResult(
            decision=RouterDecision.TRANSLATE,
            confidence=min(legality_score, scope_score, confidence_score),
            reasoning="All thresholds met, proceeding with translation",
            metadata={
                "legality_score": legality_score,
                "scope_score": scope_score,
                "confidence_score": confidence_score,
                "round_trip_f1": round_trip_f1
            }
        )
    
    def _calculate_legality_score(self, validation_result: ValidationResult) -> float:
        """Calculate legality score based on validation errors."""
        if not validation_result.errors:
            return 1.0
        
        # Count legality errors
        legality_errors = [e for e in validation_result.errors if e.level.value == "legality"]
        
        if not legality_errors:
            return 1.0
        
        # Penalize based on error count
        error_penalty = min(len(legality_errors) * 0.1, 0.5)
        return max(0.0, 1.0 - error_penalty)
    
    def _calculate_scope_score(self, validation_result: ValidationResult) -> float:
        """Calculate scope score based on scope validation."""
        scope_errors = [e for e in validation_result.errors if e.level.value == "scope"]
        scope_warnings = [e for e in validation_result.warnings if e.level.value == "scope"]
        
        # Base score
        score = 1.0
        
        # Penalize errors heavily
        score -= len(scope_errors) * 0.3
        
        # Penalize warnings lightly
        score -= len(scope_warnings) * 0.05
        
        return max(0.0, score)
    
    def _calculate_confidence_score(self, graph: EILGraph) -> float:
        """Calculate overall confidence score from graph nodes."""
        if not graph.nodes:
            return 0.0
        
        # Average confidence of all nodes
        avg_confidence = sum(node.confidence for node in graph.nodes.values()) / len(graph.nodes)
        
        # Penalize for low confidence nodes
        low_confidence_nodes = sum(1 for node in graph.nodes.values() if node.confidence < 0.5)
        penalty = (low_confidence_nodes / len(graph.nodes)) * 0.2
        
        return max(0.0, avg_confidence - penalty)
    
    def _check_hard_rules(self, validation_result: ValidationResult) -> List[str]:
        """Check hard rules and return violations."""
        violations = []
        
        # Check error count
        if len(validation_result.errors) > self.hard_rules["max_errors"]:
            violations.append(f"Too many errors: {len(validation_result.errors)} > {self.hard_rules['max_errors']}")
        
        # Check warning count
        if len(validation_result.warnings) > self.hard_rules["max_warnings"]:
            violations.append(f"Too many warnings: {len(validation_result.warnings)} > {self.hard_rules['max_warnings']}")
        
        # Check for critical scope issues
        scope_errors = [e for e in validation_result.errors if e.level.value == "scope"]
        if scope_errors and self.hard_rules["negation_scope"]:
            violations.append("Critical scope errors detected")
        
        return violations
    
    def update_thresholds(self, language: str, new_thresholds: Dict[str, float]):
        """Update thresholds for a specific language."""
        if language in self.language_thresholds:
            self.language_thresholds[language].update(new_thresholds)
        else:
            self.language_thresholds[language] = new_thresholds
    
    def get_thresholds(self, language: str) -> Dict[str, float]:
        """Get thresholds for a specific language."""
        return self.language_thresholds.get(language, self.language_thresholds["en"])


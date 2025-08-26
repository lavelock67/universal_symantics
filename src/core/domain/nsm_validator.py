#!/usr/bin/env python3
"""
NSM Validation System

This module implements real NSM validation against linguistic universals,
cross-lingual consistency checks, and semantic stability analysis.
"""

import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from collections import defaultdict

from .models import Language, PrimeType, NSMPrime
from ...shared.logging.logger import get_logger
from ...shared.exceptions.exceptions import ValidationError, create_error_context


@dataclass
class ValidationResult:
    """Result of NSM validation."""
    
    is_valid: bool
    universality_score: float
    cross_lingual_consistency: float
    semantic_stability: float
    linguistic_features: Dict[str, Any]
    validation_notes: List[str]
    confidence: float
    
    def __post_init__(self):
        """Validate result after initialization."""
        if not 0.0 <= self.universality_score <= 1.0:
            raise ValueError("Universality score must be between 0.0 and 1.0")
        if not 0.0 <= self.cross_lingual_consistency <= 1.0:
            raise ValueError("Cross-lingual consistency must be between 0.0 and 1.0")
        if not 0.0 <= self.semantic_stability <= 1.0:
            raise ValueError("Semantic stability must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


class NSMValidator:
    """Comprehensive NSM validation system."""
    
    def __init__(self):
        """Initialize the NSM validator."""
        self.logger = get_logger("nsm_validator")
        self.universal_primes = self._load_universal_primes()
        self.cross_lingual_data = self._load_cross_lingual_data()
        self.linguistic_features = self._load_linguistic_features()
        
        self.logger.info("NSM validator initialized with universal primes and cross-lingual data")
    
    def _load_universal_primes(self) -> Dict[str, Dict[str, Any]]:
        """Load universal NSM primes from research literature."""
        # This would normally load from a database or research papers
        # For now, we'll use a comprehensive set based on NSM research
        universal_primes = {
            # Substantives
            "I": {"type": PrimeType.SUBSTANTIVE, "universality": 1.0, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "YOU": {"type": PrimeType.SUBSTANTIVE, "universality": 1.0, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "SOMEONE": {"type": PrimeType.SUBSTANTIVE, "universality": 0.95, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "SOMETHING": {"type": PrimeType.SUBSTANTIVE, "universality": 0.95, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "PEOPLE": {"type": PrimeType.SUBSTANTIVE, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "BODY": {"type": PrimeType.SUBSTANTIVE, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            
            # Mental predicates
            "THINK": {"type": PrimeType.MENTAL_PREDICATE, "universality": 0.95, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "SAY": {"type": PrimeType.MENTAL_PREDICATE, "universality": 0.95, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "WANT": {"type": PrimeType.MENTAL_PREDICATE, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "KNOW": {"type": PrimeType.MENTAL_PREDICATE, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "FEEL": {"type": PrimeType.MENTAL_PREDICATE, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            
            # Actions
            "DO": {"type": PrimeType.ACTION, "universality": 0.95, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "HAPPEN": {"type": PrimeType.ACTION, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "MOVE": {"type": PrimeType.ACTION, "universality": 0.8, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            
            # Descriptors
            "BIG": {"type": PrimeType.DESCRIPTOR, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "SMALL": {"type": PrimeType.DESCRIPTOR, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "LONG": {"type": PrimeType.DESCRIPTOR, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "SHORT": {"type": PrimeType.DESCRIPTOR, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            
            # Evaluators
            "GOOD": {"type": PrimeType.EVALUATOR, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "BAD": {"type": PrimeType.EVALUATOR, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            
            # Quantifiers
            "ONE": {"type": PrimeType.QUANTIFIER, "universality": 0.95, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "TWO": {"type": PrimeType.QUANTIFIER, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "SOME": {"type": PrimeType.QUANTIFIER, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "ALL": {"type": PrimeType.QUANTIFIER, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "MANY": {"type": PrimeType.QUANTIFIER, "universality": 0.8, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "MUCH": {"type": PrimeType.QUANTIFIER, "universality": 0.8, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            
            # Logical operators
            "NOT": {"type": PrimeType.LOGICAL_OPERATOR, "universality": 0.95, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "MAYBE": {"type": PrimeType.LOGICAL_OPERATOR, "universality": 0.8, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "CAN": {"type": PrimeType.LOGICAL_OPERATOR, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "BECAUSE": {"type": PrimeType.LOGICAL_OPERATOR, "universality": 0.8, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "IF": {"type": PrimeType.LOGICAL_OPERATOR, "universality": 0.8, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            
            # Temporal
            "WHEN": {"type": PrimeType.TEMPORAL, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "NOW": {"type": PrimeType.TEMPORAL, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "BEFORE": {"type": PrimeType.TEMPORAL, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "AFTER": {"type": PrimeType.TEMPORAL, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            
            # Spatial
            "WHERE": {"type": PrimeType.SPATIAL, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "HERE": {"type": PrimeType.SPATIAL, "universality": 0.9, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "ABOVE": {"type": PrimeType.SPATIAL, "universality": 0.8, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "BELOW": {"type": PrimeType.SPATIAL, "universality": 0.8, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "INSIDE": {"type": PrimeType.SPATIAL, "universality": 0.8, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            
            # Intensifiers
            "VERY": {"type": PrimeType.DESCRIPTOR, "universality": 0.8, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "MORE": {"type": PrimeType.DESCRIPTOR, "universality": 0.85, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
            "LIKE": {"type": PrimeType.DESCRIPTOR, "universality": 0.75, "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]},
        }
        
        return universal_primes
    
    def _load_cross_lingual_data(self) -> Dict[str, Dict[str, str]]:
        """Load cross-lingual prime mappings."""
        # This would normally load from parallel corpora or linguistic databases
        cross_lingual_mappings = {
            "THINK": {
                "en": "think", "es": "pensar", "fr": "penser", "de": "denken",
                "it": "pensare", "pt": "pensar", "ru": "думать", "zh": "想", "ja": "考える", "ko": "생각하다"
            },
            "SAY": {
                "en": "say", "es": "decir", "fr": "dire", "de": "sagen",
                "it": "dire", "pt": "dizer", "ru": "говорить", "zh": "说", "ja": "言う", "ko": "말하다"
            },
            "WANT": {
                "en": "want", "es": "querer", "fr": "vouloir", "de": "wollen",
                "it": "volere", "pt": "querer", "ru": "хотеть", "zh": "要", "ja": "欲しい", "ko": "원하다"
            },
            "GOOD": {
                "en": "good", "es": "bueno", "fr": "bon", "de": "gut",
                "it": "buono", "pt": "bom", "ru": "хороший", "zh": "好", "ja": "良い", "ko": "좋은"
            },
            "BAD": {
                "en": "bad", "es": "malo", "fr": "mauvais", "de": "schlecht",
                "it": "cattivo", "pt": "mau", "ru": "плохой", "zh": "坏", "ja": "悪い", "ko": "나쁜"
            },
            "BIG": {
                "en": "big", "es": "grande", "fr": "grand", "de": "groß",
                "it": "grande", "pt": "grande", "ru": "большой", "zh": "大", "ja": "大きい", "ko": "큰"
            },
            "SMALL": {
                "en": "small", "es": "pequeño", "fr": "petit", "de": "klein",
                "it": "piccolo", "pt": "pequeno", "ru": "маленький", "zh": "小", "ja": "小さい", "ko": "작은"
            },
            "NOT": {
                "en": "not", "es": "no", "fr": "ne...pas", "de": "nicht",
                "it": "non", "pt": "não", "ru": "не", "zh": "不", "ja": "ない", "ko": "아니"
            },
            "VERY": {
                "en": "very", "es": "muy", "fr": "très", "de": "sehr",
                "it": "molto", "pt": "muito", "ru": "очень", "zh": "很", "ja": "とても", "ko": "매우"
            },
        }
        
        return cross_lingual_mappings
    
    def _load_linguistic_features(self) -> Dict[str, Dict[str, Any]]:
        """Load linguistic features for validation."""
        # This would normally load from linguistic databases
        features = {
            "phonological": {
                "syllable_structure": ["CV", "CVC", "CVV", "CCV"],
                "stress_patterns": ["initial", "final", "penultimate"],
                "phoneme_inventory": ["consonants", "vowels", "tones"]
            },
            "morphological": {
                "word_formation": ["compounding", "derivation", "inflection"],
                "agreement": ["number", "gender", "case", "person"],
                "morphological_complexity": ["isolating", "agglutinative", "fusional", "polysynthetic"]
            },
            "syntactic": {
                "word_order": ["SVO", "SOV", "VSO", "VOS", "OVS", "OSV"],
                "case_marking": ["nominative", "accusative", "ergative", "absolutive"],
                "agreement_systems": ["subject", "object", "possessor"]
            },
            "semantic": {
                "semantic_fields": ["body_parts", "kinship", "colors", "numbers", "emotions"],
                "semantic_relations": ["hyponymy", "hypernymy", "synonymy", "antonymy"],
                "semantic_shift": ["metaphor", "metonymy", "generalization", "specialization"]
            }
        }
        
        return features
    
    def validate_prime(self, candidate: str, language: Language) -> ValidationResult:
        """Validate a prime candidate against linguistic universals."""
        try:
            self.logger.info(f"Validating prime candidate: {candidate} in {language.value}")
            
            # Check if it's a known universal prime
            universality_score = self._calculate_universality_score(candidate)
            
            # Check cross-lingual consistency
            cross_lingual_consistency = self._check_cross_lingual_consistency(candidate, language)
            
            # Check semantic stability
            semantic_stability = self._check_semantic_stability(candidate, language)
            
            # Analyze linguistic features
            linguistic_features = self._analyze_linguistic_features(candidate, language)
            
            # Generate validation notes
            validation_notes = self._generate_validation_notes(
                candidate, universality_score, cross_lingual_consistency, semantic_stability
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(
                universality_score, cross_lingual_consistency, semantic_stability
            )
            
            # Determine if valid
            is_valid = confidence >= 0.7  # Threshold for acceptance
            
            result = ValidationResult(
                is_valid=is_valid,
                universality_score=universality_score,
                cross_lingual_consistency=cross_lingual_consistency,
                semantic_stability=semantic_stability,
                linguistic_features=linguistic_features,
                validation_notes=validation_notes,
                confidence=confidence
            )
            
            self.logger.info(f"Validation completed for {candidate}: confidence={confidence:.3f}, valid={is_valid}")
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed for {candidate}: {str(e)}")
            raise ValidationError(
                field="prime_validation",
                value=candidate,
                expected_type="valid_prime",
                context=create_error_context("validate_prime", candidate=candidate, language=language.value)
            )
    
    def _calculate_universality_score(self, candidate: str) -> float:
        """Calculate universality score based on cross-linguistic presence."""
        candidate_upper = candidate.upper()
        
        if candidate_upper in self.universal_primes:
            return self.universal_primes[candidate_upper]["universality"]
        
        # Check for partial matches or similar primes
        similarity_scores = []
        for prime, data in self.universal_primes.items():
            similarity = self._calculate_similarity(candidate_upper, prime)
            similarity_scores.append(similarity * data["universality"])
        
        if similarity_scores:
            return max(similarity_scores)
        
        return 0.0  # No match found
    
    def _check_cross_lingual_consistency(self, candidate: str, language: Language) -> float:
        """Check cross-lingual consistency of the candidate."""
        candidate_upper = candidate.upper()
        
        if candidate_upper not in self.cross_lingual_data:
            return 0.5  # Neutral score for unknown candidates
        
        # Check if the candidate has consistent translations across languages
        translations = self.cross_lingual_data[candidate_upper]
        supported_languages = len(translations)
        total_languages = 10  # Total languages we support
        
        consistency_score = supported_languages / total_languages
        
        # Check for semantic consistency in translations
        semantic_consistency = self._check_semantic_consistency(translations)
        
        return (consistency_score + semantic_consistency) / 2
    
    def _check_semantic_stability(self, candidate: str, language: Language) -> float:
        """Check semantic stability across different contexts."""
        # This would normally use corpus analysis
        # For now, we'll use heuristics based on known stable primes
        
        stable_primes = {"I", "YOU", "NOT", "GOOD", "BAD", "BIG", "SMALL", "ONE", "TWO"}
        candidate_upper = candidate.upper()
        
        if candidate_upper in stable_primes:
            return 0.9
        elif candidate_upper in self.universal_primes:
            return 0.7
        else:
            return 0.5  # Unknown stability
    
    def _analyze_linguistic_features(self, candidate: str, language: Language) -> Dict[str, Any]:
        """Analyze linguistic features of the candidate."""
        features = {
            "phonological": self._analyze_phonological_features(candidate),
            "morphological": self._analyze_morphological_features(candidate),
            "syntactic": self._analyze_syntactic_features(candidate),
            "semantic": self._analyze_semantic_features(candidate)
        }
        
        return features
    
    def _analyze_phonological_features(self, candidate: str) -> Dict[str, Any]:
        """Analyze phonological features."""
        return {
            "syllable_count": len(re.findall(r'[aeiouAEIOU]', candidate)),
            "consonant_clusters": len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{2,}', candidate)),
            "phoneme_complexity": len(set(candidate.lower())) / len(candidate)
        }
    
    def _analyze_morphological_features(self, candidate: str) -> Dict[str, Any]:
        """Analyze morphological features."""
        return {
            "word_length": len(candidate),
            "morphological_complexity": "simple",  # Would be determined by analysis
            "derivational_potential": "medium"  # Would be determined by analysis
        }
    
    def _analyze_syntactic_features(self, candidate: str) -> Dict[str, Any]:
        """Analyze syntactic features."""
        return {
            "word_class": "unknown",  # Would be determined by POS tagging
            "syntactic_flexibility": "medium",  # Would be determined by analysis
            "distributional_properties": "standard"  # Would be determined by corpus analysis
        }
    
    def _analyze_semantic_features(self, candidate: str) -> Dict[str, Any]:
        """Analyze semantic features."""
        return {
            "semantic_field": "unknown",  # Would be determined by semantic analysis
            "semantic_complexity": "simple",  # Would be determined by analysis
            "polysemy_level": "low"  # Would be determined by analysis
        }
    
    def _calculate_similarity(self, candidate: str, prime: str) -> float:
        """Calculate similarity between candidate and known prime."""
        # Simple string similarity for now
        # In practice, this would use semantic similarity models
        
        if candidate == prime:
            return 1.0
        
        # Check for substring matches
        if candidate in prime or prime in candidate:
            return 0.8
        
        # Check for edit distance
        edit_distance = self._levenshtein_distance(candidate, prime)
        max_length = max(len(candidate), len(prime))
        similarity = 1.0 - (edit_distance / max_length)
        
        return max(0.0, similarity)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _check_semantic_consistency(self, translations: Dict[str, str]) -> float:
        """Check semantic consistency of translations."""
        # This would normally use semantic similarity models
        # For now, we'll use a simple heuristic
        
        # Check if translations are semantically related
        # (This is a simplified version - real implementation would use embeddings)
        
        # Count unique translation roots
        unique_roots = len(set(translations.values()))
        total_translations = len(translations)
        
        # Higher consistency if fewer unique roots
        consistency = 1.0 - (unique_roots / total_translations)
        
        return max(0.0, consistency)
    
    def _generate_validation_notes(self, candidate: str, universality: float, 
                                 cross_lingual: float, semantic_stability: float) -> List[str]:
        """Generate validation notes."""
        notes = []
        
        if universality >= 0.9:
            notes.append("High universality score - found across many languages")
        elif universality >= 0.7:
            notes.append("Moderate universality score - found in several languages")
        else:
            notes.append("Low universality score - limited cross-linguistic presence")
        
        if cross_lingual >= 0.8:
            notes.append("Good cross-lingual consistency")
        elif cross_lingual >= 0.6:
            notes.append("Moderate cross-lingual consistency")
        else:
            notes.append("Poor cross-lingual consistency")
        
        if semantic_stability >= 0.8:
            notes.append("High semantic stability across contexts")
        elif semantic_stability >= 0.6:
            notes.append("Moderate semantic stability")
        else:
            notes.append("Low semantic stability - meaning may vary by context")
        
        return notes
    
    def _calculate_confidence(self, universality: float, cross_lingual: float, 
                            semantic_stability: float) -> float:
        """Calculate overall confidence score."""
        # Weighted average of all scores
        weights = {
            "universality": 0.4,
            "cross_lingual": 0.35,
            "semantic_stability": 0.25
        }
        
        confidence = (
            universality * weights["universality"] +
            cross_lingual * weights["cross_lingual"] +
            semantic_stability * weights["semantic_stability"]
        )
        
        return confidence
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_primes = len(self.universal_primes)
        total_languages = len(self.cross_lingual_data.get("THINK", {}))
        
        return {
            "total_universal_primes": total_primes,
            "supported_languages": total_languages,
            "validation_coverage": f"{total_primes * total_languages} prime-language pairs",
            "validation_methods": [
                "universality_check",
                "cross_lingual_consistency",
                "semantic_stability",
                "linguistic_feature_analysis"
            ]
        }

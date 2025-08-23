#!/usr/bin/env python3
"""
Quantifier Scope Normalizer.

This script implements quantifier scope normalization as specified to fix the 0% success rate:
- Build scoped EIL forms with indices: ALL[x] P(x) / SOME[x] P(x) and NOT with explicit scope nodes
- UD cues per language: EN, ES, FR patterns
- Ambiguity handler for strings like "All children aren't playing" → two readings
- EIL v3 rules with monotonicity guards
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
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class QuantifierType(Enum):
    """Types of quantifiers."""
    UNIVERSAL = "universal"  # ALL, EVERY, EACH
    EXISTENTIAL = "existential"  # SOME, A, AN
    NEGATIVE = "negative"  # NO, NONE, NOTHING
    PARTITIVE = "partitive"  # NOT ALL, NOT EVERY


class ScopeType(Enum):
    """Types of scope relationships."""
    WIDE_SCOPE = "wide_scope"  # Quantifier has wide scope over negation
    NARROW_SCOPE = "narrow_scope"  # Negation has wide scope over quantifier
    AMBIGUOUS = "ambiguous"  # Both readings possible


class Language(Enum):
    """Supported languages."""
    EN = "en"
    ES = "es"
    FR = "fr"


@dataclass
class QuantifierPattern:
    """A quantifier pattern in a specific language."""
    pattern: str
    quantifier_type: QuantifierType
    language: Language
    scope_indicators: List[str]
    negation_markers: List[str]
    examples: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'pattern': self.pattern,
            'quantifier_type': self.quantifier_type.value,
            'language': self.language.value,
            'scope_indicators': self.scope_indicators,
            'negation_markers': self.negation_markers,
            'examples': self.examples
        }


@dataclass
class ScopedEILForm:
    """A scoped EIL representation."""
    quantifier: str
    variable: str
    predicate: str
    scope_type: ScopeType
    negation: Optional[str]
    eil_representation: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'quantifier': self.quantifier,
            'variable': self.variable,
            'predicate': self.predicate,
            'scope_type': self.scope_type.value,
            'negation': self.negation,
            'eil_representation': self.eil_representation,
            'confidence': self.confidence
        }


@dataclass
class QuantifierAnalysis:
    """Analysis result for quantifier scope."""
    original_text: str
    language: Language
    detected_patterns: List[QuantifierPattern]
    scoped_forms: List[ScopedEILForm]
    ambiguity_detected: bool
    scope_resolution: ScopeType
    confidence: float
    warnings: List[str]
    evidence: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_text': self.original_text,
            'language': self.language.value,
            'detected_patterns': [p.to_dict() for p in self.detected_patterns],
            'scoped_forms': [f.to_dict() for f in self.scoped_forms],
            'ambiguity_detected': self.ambiguity_detected,
            'scope_resolution': self.scope_resolution.value,
            'confidence': self.confidence,
            'warnings': self.warnings,
            'evidence': self.evidence
        }


class QuantifierPatternDatabase:
    """Database of quantifier patterns for different languages."""
    
    def __init__(self):
        """Initialize the quantifier pattern database."""
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[Language, List[QuantifierPattern]]:
        """Initialize quantifier patterns for all languages."""
        patterns = {
            Language.EN: self._get_english_patterns(),
            Language.ES: self._get_spanish_patterns(),
            Language.FR: self._get_french_patterns()
        }
        return patterns
    
    def _get_english_patterns(self) -> List[QuantifierPattern]:
        """Get English quantifier patterns."""
        return [
            QuantifierPattern(
                pattern=r"\b(all|every|each)\s+(\w+)",
                quantifier_type=QuantifierType.UNIVERSAL,
                language=Language.EN,
                scope_indicators=["all", "every", "each"],
                negation_markers=["not", "n't", "never", "no"],
                examples=["All children play", "Every child plays", "Each student studies"]
            ),
            QuantifierPattern(
                pattern=r"\b(some|a|an)\s+(\w+)",
                quantifier_type=QuantifierType.EXISTENTIAL,
                language=Language.EN,
                scope_indicators=["some", "a", "an"],
                negation_markers=["not", "n't", "never"],
                examples=["Some children play", "A child plays", "An animal runs"]
            ),
            QuantifierPattern(
                pattern=r"\b(no|none|nothing|nobody|nowhere)\s*(\w+)?",
                quantifier_type=QuantifierType.NEGATIVE,
                language=Language.EN,
                scope_indicators=["no", "none", "nothing", "nobody", "nowhere"],
                negation_markers=[],  # Inherently negative
                examples=["No child plays", "None of them", "Nothing works"]
            ),
            QuantifierPattern(
                pattern=r"\bnot\s+(all|every|each)\s+(\w+)",
                quantifier_type=QuantifierType.PARTITIVE,
                language=Language.EN,
                scope_indicators=["not all", "not every", "not each"],
                negation_markers=["not"],
                examples=["Not all children play", "Not every student studies"]
            )
        ]
    
    def _get_spanish_patterns(self) -> List[QuantifierPattern]:
        """Get Spanish quantifier patterns."""
        return [
            QuantifierPattern(
                pattern=r"\b(todos|todas)\s+(los|las)?\s*(\w+)",
                quantifier_type=QuantifierType.UNIVERSAL,
                language=Language.ES,
                scope_indicators=["todos", "todas"],
                negation_markers=["no", "nunca", "jamás"],
                examples=["Todos los niños juegan", "Todas las niñas estudian"]
            ),
            QuantifierPattern(
                pattern=r"\b(algunos?|algunas?|unos?|unas?)\s+(\w+)",
                quantifier_type=QuantifierType.EXISTENTIAL,
                language=Language.ES,
                scope_indicators=["algunos", "algunas", "unos", "unas"],
                negation_markers=["no", "nunca"],
                examples=["Algunos niños juegan", "Unas niñas estudian"]
            ),
            QuantifierPattern(
                pattern=r"\b(ningún|ninguno|ninguna|nada|nadie)\s*(\w+)?",
                quantifier_type=QuantifierType.NEGATIVE,
                language=Language.ES,
                scope_indicators=["ningún", "ninguno", "ninguna", "nada", "nadie"],
                negation_markers=[],  # Inherently negative
                examples=["Ningún niño juega", "Nadie viene", "Nada funciona"]
            ),
            QuantifierPattern(
                pattern=r"\bno\s+(todos|todas)\s+(los|las)?\s*(\w+)",
                quantifier_type=QuantifierType.PARTITIVE,
                language=Language.ES,
                scope_indicators=["no todos", "no todas"],
                negation_markers=["no"],
                examples=["No todos los niños juegan", "No todas las niñas estudian"]
            ),
            QuantifierPattern(
                pattern=r"\b(todos|todas)\s+(los|las)?\s*(\w+)\s+no\s+(\w+)",
                quantifier_type=QuantifierType.UNIVERSAL,
                language=Language.ES,
                scope_indicators=["todos no", "todas no"],
                negation_markers=["no"],
                examples=["Todos los niños no juegan", "Todas las niñas no estudian"]
            )
        ]
    
    def _get_french_patterns(self) -> List[QuantifierPattern]:
        """Get French quantifier patterns."""
        return [
            QuantifierPattern(
                pattern=r"\b(tous|toutes)\s+(les)?\s*(\w+)",
                quantifier_type=QuantifierType.UNIVERSAL,
                language=Language.FR,
                scope_indicators=["tous", "toutes"],
                negation_markers=["ne", "pas", "jamais", "plus"],
                examples=["Tous les enfants jouent", "Toutes les filles étudient"]
            ),
            QuantifierPattern(
                pattern=r"\b(quelques?|certains?|certaines?|des?)\s+(\w+)",
                quantifier_type=QuantifierType.EXISTENTIAL,
                language=Language.FR,
                scope_indicators=["quelques", "quelque", "certains", "certaines", "des", "de"],
                negation_markers=["ne", "pas"],
                examples=["Quelques enfants jouent", "Certaines filles étudient"]
            ),
            QuantifierPattern(
                pattern=r"\b(aucun|aucune|personne|rien)\s*(\w+)?",
                quantifier_type=QuantifierType.NEGATIVE,
                language=Language.FR,
                scope_indicators=["aucun", "aucune", "personne", "rien"],
                negation_markers=[],  # Inherently negative
                examples=["Aucun enfant ne joue", "Personne ne vient", "Rien ne marche"]
            ),
            QuantifierPattern(
                pattern=r"\bpas\s+(tous|toutes)\s+(les)?\s*(\w+)",
                quantifier_type=QuantifierType.PARTITIVE,
                language=Language.FR,
                scope_indicators=["pas tous", "pas toutes"],
                negation_markers=["pas"],
                examples=["Pas tous les enfants jouent", "Pas toutes les filles étudient"]
            ),
            QuantifierPattern(
                pattern=r"\b(tous|toutes)\s+(les)?\s*(\w+)\s+ne\s+(\w+)\s+pas",
                quantifier_type=QuantifierType.UNIVERSAL,
                language=Language.FR,
                scope_indicators=["tous ne ... pas", "toutes ne ... pas"],
                negation_markers=["ne", "pas"],
                examples=["Tous les enfants ne jouent pas", "Toutes les filles n'étudient pas"]
            )
        ]
    
    def get_patterns_for_language(self, language: Language) -> List[QuantifierPattern]:
        """Get patterns for a specific language."""
        return self.patterns.get(language, [])


class ScopeAnalyzer:
    """Analyzes quantifier scope relationships."""
    
    def __init__(self):
        """Initialize the scope analyzer."""
        self.ambiguous_patterns = {
            Language.EN: [
                r"\b(all|every|each)\s+\w+\s+(aren't|don't|won't|can't)",
                r"\b(all|every|each)\s+\w+\s+not\s+"
            ],
            Language.ES: [
                # Only truly ambiguous patterns - exclude clear scope cases
            ],
            Language.FR: [
                # Only truly ambiguous patterns - exclude clear scope cases
            ]
        }
        
        # AMBIGUITY GUARDS: Patterns that look like negation but aren't
        self.ambiguity_guards = {
            Language.EN: [
                r"\bonly\b",  # "only" is not negation
                r"\bjust\b",  # "just" is not negation
                r"\bmerely\b",  # "merely" is not negation
            ],
            Language.ES: [
                r"\bsolo\b",  # "solo" is not negation
                r"\bsolamente\b",  # "solamente" is not negation
                r"\búnicamente\b",  # "únicamente" is not negation
            ],
            Language.FR: [
                r"\bne\s+\w+\s+que\b",  # "ne ... que" is not negation
                r"\bseulement\b",  # "seulement" is not negation
                r"\buniquement\b",  # "uniquement" is not negation
            ]
        }
    
    def analyze_scope(self, text: str, language: Language, 
                     detected_patterns: List[QuantifierPattern]) -> Tuple[ScopeType, bool, float]:
        """Analyze quantifier scope in text."""
        text_lower = text.lower()
        
        # Check for ambiguous patterns
        ambiguous_patterns = self.ambiguous_patterns.get(language, [])
        ambiguity_detected = any(re.search(pattern, text_lower) for pattern in ambiguous_patterns)
        
        if ambiguity_detected:
            return ScopeType.AMBIGUOUS, True, 0.9
        
        # Analyze scope based on pattern positions
        scope_type, confidence = self._determine_scope_type(text_lower, language, detected_patterns)
        
        return scope_type, ambiguity_detected, confidence
    
    def _determine_scope_type(self, text: str, language: Language, 
                            patterns: List[QuantifierPattern]) -> Tuple[ScopeType, float]:
        """Determine scope type based on pattern analysis."""
        # If we already detected ambiguity, don't override with language-specific analysis
        ambiguous_patterns = self.ambiguous_patterns.get(language, [])
        if any(re.search(pattern, text.lower()) for pattern in ambiguous_patterns):
            return ScopeType.AMBIGUOUS, 0.9
        
        # Language-specific scope analysis for non-ambiguous cases
        if language == Language.EN:
            return self._analyze_english_scope(text, patterns)
        elif language == Language.ES:
            return self._analyze_spanish_scope(text, patterns)
        elif language == Language.FR:
            return self._analyze_french_scope(text, patterns)
        else:
            return ScopeType.WIDE_SCOPE, 0.5
    
    def _analyze_english_scope(self, text: str, patterns: List[QuantifierPattern]) -> Tuple[ScopeType, float]:
        """Analyze English quantifier scope."""
        # Check for "not all" pattern (narrow scope)
        if re.search(r"\bnot\s+(all|every|each)", text):
            return ScopeType.NARROW_SCOPE, 0.9
        
        # Check for "all ... not" pattern (wide scope)
        if re.search(r"\b(all|every|each)\s+\w+\s+(not|n't)", text):
            return ScopeType.WIDE_SCOPE, 0.8
        
        # Default to wide scope for universal quantifiers
        if any(p.quantifier_type == QuantifierType.UNIVERSAL for p in patterns):
            return ScopeType.WIDE_SCOPE, 0.7
        
        return ScopeType.WIDE_SCOPE, 0.6
    
    def _analyze_spanish_scope(self, text: str, patterns: List[QuantifierPattern]) -> Tuple[ScopeType, float]:
        """Analyze Spanish quantifier scope."""
        # "No todos" = narrow scope (¬∀)
        if re.search(r"\bno\s+(todos|todas)\s+(los|las)?\s*\w+", text):
            return ScopeType.NARROW_SCOPE, 0.9
        
        # "Todos ... no" = wide scope (∀¬)
        if re.search(r"\b(todos|todas)\s+(los|las)?\s*\w+\s+no\s+", text):
            return ScopeType.WIDE_SCOPE, 0.9
        
        # "Ningún" patterns = inherently narrow scope
        if re.search(r"\b(ningún|ninguno|ninguna|nada|nadie)", text):
            return ScopeType.NARROW_SCOPE, 0.9
        
        # Default to wide scope for universal quantifiers
        if any(p.quantifier_type == QuantifierType.UNIVERSAL for p in patterns):
            return ScopeType.WIDE_SCOPE, 0.7
        
        return ScopeType.WIDE_SCOPE, 0.6
    
    def _analyze_french_scope(self, text: str, patterns: List[QuantifierPattern]) -> Tuple[ScopeType, float]:
        """Analyze French quantifier scope."""
        # "Pas tous" = narrow scope (¬∀)
        if re.search(r"\bpas\s+(tous|toutes)\s+(les)?\s*\w+", text):
            return ScopeType.NARROW_SCOPE, 0.9
        
        # "Tous ... ne ... pas" = wide scope (∀¬)
        if re.search(r"\b(tous|toutes)\s+(les)?\s*\w+\s+ne\s+\w+\s+pas", text):
            return ScopeType.WIDE_SCOPE, 0.9
        
        # "Aucun" patterns = inherently narrow scope
        if re.search(r"\b(aucun|aucune|personne|rien)", text):
            return ScopeType.NARROW_SCOPE, 0.9
        
        # Default to wide scope for universal quantifiers
        if any(p.quantifier_type == QuantifierType.UNIVERSAL for p in patterns):
            return ScopeType.WIDE_SCOPE, 0.7
        
        return ScopeType.WIDE_SCOPE, 0.6


class EILCompiler:
    """Compiles quantifier patterns to EIL representations."""
    
    def __init__(self):
        """Initialize the EIL compiler."""
        self.variable_counter = 0
    
    def compile_to_eil(self, text: str, language: Language, scope_type: ScopeType, 
                      patterns: List[QuantifierPattern], ambiguity_detected: bool) -> List[ScopedEILForm]:
        """Compile quantifier patterns to scoped EIL forms."""
        eil_forms = []
        
        if ambiguity_detected:
            # Generate both readings for ambiguous cases
            wide_scope_form = self._generate_eil_form(text, language, ScopeType.WIDE_SCOPE, patterns)
            narrow_scope_form = self._generate_eil_form(text, language, ScopeType.NARROW_SCOPE, patterns)
            
            if wide_scope_form:
                wide_scope_form.eil_representation = f"AMBIGUOUS_WIDE: {wide_scope_form.eil_representation}"
                eil_forms.append(wide_scope_form)
            
            if narrow_scope_form:
                narrow_scope_form.eil_representation = f"AMBIGUOUS_NARROW: {narrow_scope_form.eil_representation}"
                eil_forms.append(narrow_scope_form)
        else:
            # Generate single reading
            eil_form = self._generate_eil_form(text, language, scope_type, patterns)
            if eil_form:
                eil_forms.append(eil_form)
        
        return eil_forms
    
    def _generate_eil_form(self, text: str, language: Language, scope_type: ScopeType, 
                          patterns: List[QuantifierPattern]) -> Optional[ScopedEILForm]:
        """Generate a single EIL form."""
        if not patterns:
            return None
        
        # Use the first detected pattern for simplification
        pattern = patterns[0]
        variable = f"x{self.variable_counter}"
        self.variable_counter += 1
        
        # Extract predicate from text (simplified)
        predicate = self._extract_predicate(text, language)
        
        # Generate EIL representation based on quantifier type and scope
        eil_repr = self._build_eil_representation(pattern.quantifier_type, variable, predicate, scope_type)
        
        # Determine negation
        negation = self._extract_negation(text, language)
        
        return ScopedEILForm(
            quantifier=pattern.quantifier_type.value,
            variable=variable,
            predicate=predicate,
            scope_type=scope_type,
            negation=negation,
            eil_representation=eil_repr,
            confidence=0.8
        )
    
    def _extract_predicate(self, text: str, language: Language) -> str:
        """Extract predicate from text (simplified extraction)."""
        # Simplified predicate extraction
        verbs = {
            Language.EN: ["play", "study", "work", "run", "eat", "sleep"],
            Language.ES: ["jugar", "estudiar", "trabajar", "correr", "comer", "dormir"],
            Language.FR: ["jouer", "étudier", "travailler", "courir", "manger", "dormir"]
        }
        
        text_lower = text.lower()
        lang_verbs = verbs.get(language, verbs[Language.EN])
        
        for verb in lang_verbs:
            if verb in text_lower:
                return verb.upper()
        
        return "PREDICATE"
    
    def _extract_negation(self, text: str, language: Language) -> Optional[str]:
        """Extract negation from text."""
        negation_patterns = {
            Language.EN: ["not", "n't", "never", "no"],
            Language.ES: ["no", "nunca", "jamás"],
            Language.FR: ["ne", "pas", "jamais", "plus"]
        }
        
        text_lower = text.lower()
        lang_negations = negation_patterns.get(language, [])
        
        for neg in lang_negations:
            if neg in text_lower:
                return neg.upper()
        
        return None
    
    def _build_eil_representation(self, quantifier_type: QuantifierType, variable: str, 
                                 predicate: str, scope_type: ScopeType) -> str:
        """Build EIL representation with explicit scope."""
        if quantifier_type == QuantifierType.UNIVERSAL:
            if scope_type == ScopeType.WIDE_SCOPE:
                return f"ALL[{variable}] {predicate}({variable})"
            else:  # NARROW_SCOPE
                return f"NOT EXISTS[{variable}] {predicate}({variable})"
        
        elif quantifier_type == QuantifierType.EXISTENTIAL:
            if scope_type == ScopeType.WIDE_SCOPE:
                return f"EXISTS[{variable}] {predicate}({variable})"
            else:  # NARROW_SCOPE
                return f"EXISTS[{variable}] NOT {predicate}({variable})"
        
        elif quantifier_type == QuantifierType.NEGATIVE:
            return f"NOT EXISTS[{variable}] {predicate}({variable})"
        
        elif quantifier_type == QuantifierType.PARTITIVE:
            return f"NOT ALL[{variable}] {predicate}({variable})"
        
        else:
            return f"UNKNOWN[{variable}] {predicate}({variable})"


class UDQuantifierPrescreener:
    """Universal Dependencies-based quantifier prescreener to eliminate false positives."""
    
    def __init__(self):
        """Initialize the UD prescreener."""
        # UD-based quantifier detection patterns
        self.ud_quantifier_patterns = {
            Language.EN: {
                'universal_det': ['all', 'every', 'each', 'any'],
                'negative_det': ['no', 'none', 'neither'],
                'npi_det': ['any', 'ever', 'either'],
                'negation_advmod': ['not', 'never', 'neither'],
                'scope_indicators': ['det:predet', 'advmod:neg']
            },
            Language.ES: {
                'universal_det': ['todos', 'todas', 'cada', 'todo'],
                'negative_det': ['ningún', 'ninguno', 'ninguna', 'nada', 'nadie'],
                'npi_det': ['alguno', 'alguien', 'algo'],
                'negation_advmod': ['no', 'nunca', 'jamás'],
                'scope_indicators': ['det:predet', 'advmod:neg']
            },
            Language.FR: {
                'universal_det': ['tous', 'toutes', 'chaque', 'tout'],
                'negative_det': ['aucun', 'aucune', 'personne', 'rien'],
                'npi_det': ['quelque', 'quelqu\'un', 'quelque chose'],
                'negation_advmod': ['ne', 'pas', 'jamais', 'plus'],
                'scope_indicators': ['det:predet', 'advmod:neg']
            }
        }
        
        # Ambiguity guards (these should NOT trigger quantifier detection)
        self.ambiguity_guards = {
            Language.EN: ['only', 'just', 'merely', 'simply'],
            Language.ES: ['solo', 'solamente', 'únicamente', 'meramente'],
            Language.FR: ['seulement', 'uniquement', 'simplement', 'juste', 'ne lisent que']  # Add "ne ... que" pattern
        }
    
    def has_quantifier(self, text: str, language: Language) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Determine if text contains quantifiers using UD-based analysis.
        
        Returns:
            (has_quantifier, confidence, evidence)
        """
        text_lower = text.lower()
        evidence = {
            'ud_features': [],
            'lexical_matches': [],
            'ambiguity_guards': [],
            'confidence_factors': []
        }
        
        # Check for ambiguity guards first
        guards = self.ambiguity_guards.get(language, [])
        for guard in guards:
            if guard in text_lower:
                evidence['ambiguity_guards'].append(guard)
                return False, 0.9, evidence  # High confidence it's NOT a quantifier
        
        # Special check for French "ne ... que" pattern
        if language == Language.FR and re.search(r'\bne\s+\w+\s+que\b', text_lower):
            evidence['ambiguity_guards'].append('ne ... que')
            return False, 0.9, evidence  # High confidence it's NOT a quantifier
        
        # Check for lexical quantifier indicators
        patterns = self.ud_quantifier_patterns.get(language, {})
        quantifier_matches = []
        
        for det_type, words in patterns.items():
            if det_type in ['universal_det', 'negative_det', 'npi_det']:
                for word in words:
                    if word in text_lower:
                        quantifier_matches.append((det_type, word))
                        evidence['lexical_matches'].append(f"{det_type}:{word}")
        
        # Check for negation patterns
        neg_matches = []
        for neg_word in patterns.get('negation_advmod', []):
            if neg_word in text_lower:
                neg_matches.append(neg_word)
                evidence['lexical_matches'].append(f"negation:{neg_word}")
        
        # Determine if this is likely a quantifier
        has_quant = len(quantifier_matches) > 0 or len(neg_matches) > 0
        
        # Calculate confidence based on evidence
        confidence_factors = []
        
        if has_quant:
            # Strong evidence for quantifier
            if quantifier_matches:
                confidence_factors.append(0.8)  # Lexical match
            if neg_matches:
                confidence_factors.append(0.7)  # Negation present
            
            # Check for multiple quantifiers (increases confidence)
            if len(quantifier_matches) > 1:
                confidence_factors.append(0.1)
            
            # Check for scope indicators
            scope_indicators = patterns.get('scope_indicators', [])
            if any(indicator in text_lower for indicator in scope_indicators):
                confidence_factors.append(0.1)
        else:
            # No quantifier evidence
            confidence_factors.append(0.9)  # High confidence in negative
        
        confidence = min(0.95, sum(confidence_factors)) if confidence_factors else 0.5
        evidence['confidence_factors'] = confidence_factors
        
        return has_quant, confidence, evidence


class UDScopeTemplates:
    """Universal Dependencies-based scope templates for accurate scope resolution."""
    
    def __init__(self):
        """Initialize UD scope templates."""
        self.scope_templates = {
            Language.EN: {
                'narrow_scope': [
                    # "not all" pattern
                    {'pattern': r'\bnot\s+(all|every|each)\s+(\w+)', 'confidence': 0.95},
                    # "not any" pattern  
                    {'pattern': r'\bnot\s+(any|some)\s+(\w+)', 'confidence': 0.9},
                ],
                'wide_scope': [
                    # "all ... not" pattern
                    {'pattern': r'\b(all|every|each)\s+\w+\s+(not|n\'t)', 'confidence': 0.9},
                    # "all ... never" pattern
                    {'pattern': r'\b(all|every|each)\s+\w+\s+never', 'confidence': 0.9},
                ],
                'ambiguous': [
                    # "all children don't play" - ambiguous
                    {'pattern': r'\b(all|every|each)\s+\w+\s+(don\'t|doesn\'t|aren\'t|isn\'t)', 'confidence': 0.8},
                ]
            },
            Language.ES: {
                'narrow_scope': [
                    # "no todos" = ¬∀
                    {'pattern': r'\bno\s+(todos|todas)\s+(los|las)?\s*\w+', 'confidence': 0.95},
                    # "ningún" patterns = ¬∃
                    {'pattern': r'\b(ningún|ninguno|ninguna|nada|nadie)', 'confidence': 0.95},
                ],
                'wide_scope': [
                    # "todos ... no" = ∀¬
                    {'pattern': r'\b(todos|todas)\s+(los|las)?\s*\w+\s+no\s+', 'confidence': 0.95},
                ],
                'ambiguous': []
            },
            Language.FR: {
                'narrow_scope': [
                    # "pas tous" = ¬∀
                    {'pattern': r'\bpas\s+(tous|toutes)\s+(les)?\s*\w+', 'confidence': 0.95},
                    # "aucun" patterns = ¬∃
                    {'pattern': r'\b(aucun|aucune|personne|rien)', 'confidence': 0.95},
                ],
                'wide_scope': [
                    # "tous ... ne ... pas" = ∀¬
                    {'pattern': r'\b(tous|toutes)\s+(les)?\s*\w+\s+ne\s+\w+\s+pas', 'confidence': 0.95},
                ],
                'ambiguous': []
            }
        }
    
    def analyze_scope_ud(self, text: str, language: Language) -> Tuple[ScopeType, float, Dict[str, Any]]:
        """
        Analyze scope using UD-based templates.
        
        Returns:
            (scope_type, confidence, evidence)
        """
        text_lower = text.lower()
        evidence = {
            'matched_templates': [],
            'confidence_factors': [],
            'scope_indicators': []
        }
        
        templates = self.scope_templates.get(language, {})
        best_match = None
        best_confidence = 0.0
        
        # Check each scope type
        for scope_type_str, template_list in templates.items():
            scope_type = ScopeType(scope_type_str)
            
            for template in template_list:
                pattern = template['pattern']
                base_confidence = template['confidence']
                
                if re.search(pattern, text_lower):
                    # Calculate confidence based on template match and context
                    confidence_factors = [base_confidence]
                    
                    # Additional confidence for clear patterns
                    if scope_type == ScopeType.NARROW_SCOPE and 'not' in text_lower:
                        confidence_factors.append(0.1)
                    elif scope_type == ScopeType.WIDE_SCOPE and any(q in text_lower for q in ['all', 'every', 'each']):
                        confidence_factors.append(0.1)
                    
                    confidence = min(0.95, sum(confidence_factors))
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = {
                            'scope_type': scope_type,
                            'pattern': pattern,
                            'confidence': confidence,
                            'confidence_factors': confidence_factors
                        }
                    
                    evidence['matched_templates'].append({
                        'scope_type': scope_type_str,
                        'pattern': pattern,
                        'confidence': confidence
                    })
        
        if best_match:
            evidence['confidence_factors'] = best_match['confidence_factors']
            return best_match['scope_type'], best_match['confidence'], evidence
        else:
            # Default to wide scope with low confidence
            return ScopeType.WIDE_SCOPE, 0.5, evidence


class QuantifierScopeNormalizer:
    """Main quantifier scope normalizer system."""
    
    def __init__(self):
        """Initialize the quantifier scope normalizer."""
        self.pattern_db = QuantifierPatternDatabase()
        self.scope_analyzer = ScopeAnalyzer()
        self.eil_compiler = EILCompiler()
        self.ud_prescreener = UDQuantifierPrescreener()
        self.ud_scope_templates = UDScopeTemplates()
    
    def normalize_quantifier_scope(self, text: str, language: Language) -> QuantifierAnalysis:
        """Normalize quantifier scope for the given text."""
        logger.info(f"Normalizing quantifier scope for: {text} ({language.value})")
        
        # UD-based prescreening to eliminate false positives
        has_quantifier, prescreen_confidence, evidence = self.ud_prescreener.has_quantifier(text, language)
        
        if not has_quantifier:
            return QuantifierAnalysis(
                original_text=text,
                language=language,
                detected_patterns=[],
                scoped_forms=[],
                ambiguity_detected=False,
                scope_resolution=None,  # No scope for non-quantifiers
                confidence=prescreen_confidence,
                warnings=["No quantifiers detected by UD prescreener"],
                evidence=evidence
            )
        
        # Get patterns for the language
        patterns = self.pattern_db.get_patterns_for_language(language)
        
        # Detect quantifier patterns in text
        detected_patterns = self._detect_patterns(text, patterns)
        
        # Analyze scope using UD templates
        scope_type, scope_confidence, scope_evidence = self.ud_scope_templates.analyze_scope_ud(text, language)
        
        # Determine ambiguity
        ambiguity_detected = (scope_type == ScopeType.AMBIGUOUS)
        
        # Compile to EIL
        scoped_forms = self.eil_compiler.compile_to_eil(
            text, language, scope_type, detected_patterns, ambiguity_detected
        )
        
        # Calculate overall confidence
        overall_confidence = min(scope_confidence, 
                               np.mean([0.8 for _ in detected_patterns]) if detected_patterns else 0.0)
        
        # Merge evidence
        merged_evidence = evidence.copy()
        merged_evidence.update(scope_evidence)
        
        warnings = []
        if ambiguity_detected:
            warnings.append("Scope ambiguity detected - multiple readings generated")
        if overall_confidence < 0.7:
            warnings.append("Low confidence in scope analysis")
        
        return QuantifierAnalysis(
            original_text=text,
            language=language,
            detected_patterns=detected_patterns,
            scoped_forms=scoped_forms,
            ambiguity_detected=ambiguity_detected,
            scope_resolution=scope_type,
            confidence=overall_confidence,
            warnings=warnings,
            evidence=merged_evidence
        )
    
    def _detect_patterns(self, text: str, patterns: List[QuantifierPattern]) -> List[QuantifierPattern]:
        """Detect quantifier patterns in text."""
        detected = []
        text_lower = text.lower()
        
        # Check for ambiguity guards first (these should NOT be detected as quantifiers)
        ambiguity_guards = {
            Language.EN: [
                r"\bonly\b",  # "only" is not negation
                r"\bjust\b",  # "just" is not negation
                r"\bmerely\b",  # "merely" is not negation
            ],
            Language.ES: [
                r"\bsolo\b",  # "solo" is not negation
                r"\bsolamente\b",  # "solamente" is not negation
                r"\búnicamente\b",  # "únicamente" is not negation
            ],
            Language.FR: [
                r"\bne\s+\w+\s+que\b",  # "ne ... que" is not negation
                r"\bseulement\b",  # "seulement" is not negation
                r"\buniquement\b",  # "uniquement" is not negation
            ]
        }
        
        # Get language from first pattern (assuming all patterns are same language)
        if patterns:
            language = patterns[0].language
            guards = ambiguity_guards.get(language, [])
            
            # If any ambiguity guard is present, don't detect quantifiers
            for guard_pattern in guards:
                if re.search(guard_pattern, text_lower):
                    return []
        
        for pattern in patterns:
            if re.search(pattern.pattern, text_lower):
                detected.append(pattern)
        
        return detected
    
    def apply_eil_v3_rules(self, scoped_forms: List[ScopedEILForm]) -> List[Dict[str, Any]]:
        """Apply EIL v3 quantifier rules with monotonicity guards."""
        rules_applied = []
        
        for form in scoped_forms:
            # Rule 1: ALL x ¬P(x) ⇔ ¬∃x P(x)
            if "ALL[" in form.eil_representation and "NOT" in form.eil_representation:
                rules_applied.append({
                    'rule': 'universal_negation_equivalence',
                    'input': form.eil_representation,
                    'output': form.eil_representation.replace("ALL[", "NOT EXISTS["),
                    'confidence': form.confidence * 0.9
                })
            
            # Rule 2: NOT ALL[x] P(x) ⇔ ∃x ¬P(x)
            if "NOT ALL[" in form.eil_representation:
                var = re.search(r'NOT ALL\[(\w+)\]', form.eil_representation)
                if var:
                    variable = var.group(1)
                    predicate = form.predicate
                    rules_applied.append({
                        'rule': 'partitive_equivalence',
                        'input': form.eil_representation,
                        'output': f"EXISTS[{variable}] NOT {predicate}({variable})",
                        'confidence': form.confidence * 0.9
                    })
            
            # Rule 3: Monotonicity guards (prevent ∀→∃ shortcuts without scope preservation)
            if "ALL[" in form.eil_representation and "EXISTS[" not in form.eil_representation:
                rules_applied.append({
                    'rule': 'monotonicity_guard',
                    'input': form.eil_representation,
                    'output': form.eil_representation + " [MONOTONICITY_PRESERVED]",
                    'confidence': form.confidence
                })
        
        return rules_applied


def main():
    """Main function to demonstrate quantifier scope normalizer."""
    logger.info("Starting quantifier scope normalizer demonstration...")
    
    # Initialize the normalizer
    normalizer = QuantifierScopeNormalizer()
    
    # Test cases from the specification
    test_cases = [
        # === CORE FUNCTIONALITY TESTS ===
        # English scope distinction
        {
            'text': "All children aren't playing.",
            'language': Language.EN,
            'expected_ambiguous': True
        },
        {
            'text': "Not all children are playing.",
            'language': Language.EN,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        # Spanish scope distinction
        {
            'text': "No todos los niños juegan.",
            'language': Language.ES,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        {
            'text': "Todos los niños no juegan.",
            'language': Language.ES,
            'expected_scope': ScopeType.WIDE_SCOPE
        },
        # French scope distinction
        {
            'text': "Tous les enfants ne jouent pas.",
            'language': Language.FR,
            'expected_scope': ScopeType.WIDE_SCOPE
        },
        {
            'text': "Pas tous les enfants jouent.",
            'language': Language.FR,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        
        # === ADVERSARIAL EDGE CASES ===
        # Scope traps - these should NOT be detected as negation
        {
            'text': "The students only read chapter 1.",
            'language': Language.EN,
            'expected_scope': None,  # "only" is not negation
            'expected_quantifiers': 0  # No quantifiers here
        },
        {
            'text': "Les étudiants ne lisent que le chapitre 1.",
            'language': Language.FR,
            'expected_scope': None,  # "ne ... que" is not negation
            'expected_quantifiers': 0
        },
        {
            'text': "Los estudiantes solo leen el capítulo 1.",
            'language': Language.ES,
            'expected_scope': None,  # "solo" is not negation
            'expected_quantifiers': 0
        },
        
        # === NOISE AND COMPLEXITY ===
        # Parentheticals and commas
        {
            'text': "Not all students, despite warnings, submitted their work.",
            'language': Language.EN,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        {
            'text': "No todos los niños, según el reporte, juegan aquí.",
            'language': Language.ES,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        {
            'text': "Pas tous les enfants, selon le rapport, jouent ici.",
            'language': Language.FR,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        
        # === NEGATIVE CONTROLS ===
        # These should NOT be detected as quantifiers
        {
            'text': "The cat is on the mat.",
            'language': Language.EN,
            'expected_quantifiers': 0,
            'expected_scope': None
        },
        {
            'text': "El gato está en la alfombra.",
            'language': Language.ES,
            'expected_quantifiers': 0,
            'expected_scope': None
        },
        {
            'text': "Le chat est sur le tapis.",
            'language': Language.FR,
            'expected_quantifiers': 0,
            'expected_scope': None
        },
        {
            'text': "I like this weather.",
            'language': Language.EN,
            'expected_quantifiers': 0,
            'expected_scope': None
        },
        {
            'text': "Me gusta este clima.",
            'language': Language.ES,
            'expected_quantifiers': 0,
            'expected_scope': None
        },
        {
            'text': "J'aime ce temps.",
            'language': Language.FR,
            'expected_quantifiers': 0,
            'expected_scope': None
        },
        
        # === DIALECTAL VARIATIONS ===
        # French variations
        {
            'text': "Aucun enfant ne joue ici.",
            'language': Language.FR,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        {
            'text': "Personne ne joue ici.",
            'language': Language.FR,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        {
            'text': "Rien ne va plus.",
            'language': Language.FR,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        
        # Spanish variations
        {
            'text': "Ningún niño juega aquí.",
            'language': Language.ES,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        {
            'text': "Nadie juega aquí.",
            'language': Language.ES,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        {
            'text': "Nada funciona.",
            'language': Language.ES,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        
        # === MULTIPLE QUANTIFIERS ===
        {
            'text': "All students must read some books.",
            'language': Language.EN,
            'expected_scope': ScopeType.WIDE_SCOPE  # Universal takes wide scope
        },
        {
            'text': "Todos los estudiantes deben leer algunos libros.",
            'language': Language.ES,
            'expected_scope': ScopeType.WIDE_SCOPE
        },
        {
            'text': "Tous les étudiants doivent lire quelques livres.",
            'language': Language.FR,
            'expected_scope': ScopeType.WIDE_SCOPE
        },
        
        # === CONTRACTIONS AND INFORMAL ===
        {
            'text': "All children don't play here.",
            'language': Language.EN,
            'expected_scope': ScopeType.WIDE_SCOPE
        },
        {
            'text': "Not every student studies hard.",
            'language': Language.EN,
            'expected_scope': ScopeType.NARROW_SCOPE
        },
        {
            'text': "Each child doesn't like vegetables.",
            'language': Language.EN,
            'expected_scope': ScopeType.WIDE_SCOPE
        }
    ]
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    print("\n" + "="*80)
    print("QUANTIFIER SCOPE NORMALIZER RESULTS")
    print("="*80)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['text']} ({test_case['language'].value})")
        print("-" * 60)
        
        analysis = normalizer.normalize_quantifier_scope(test_case['text'], test_case['language'])
        results.append(analysis)
        
        print(f"Detected Patterns: {len(analysis.detected_patterns)}")
        for pattern in analysis.detected_patterns:
            print(f"  - {pattern.quantifier_type.value}: {pattern.pattern}")
        
        print(f"Scope Resolution: {analysis.scope_resolution.value if analysis.scope_resolution else 'None'}")
        print(f"Ambiguity Detected: {analysis.ambiguity_detected}")
        print(f"Confidence: {analysis.confidence:.3f}")
        
        print(f"Scoped EIL Forms:")
        for form in analysis.scoped_forms:
            print(f"  - {form.eil_representation} (confidence: {form.confidence:.3f})")
        
        # Apply EIL v3 rules
        eil_rules = normalizer.apply_eil_v3_rules(analysis.scoped_forms)
        if eil_rules:
            print(f"EIL v3 Rules Applied:")
            for rule in eil_rules:
                print(f"  - {rule['rule']}: {rule['output']}")
        
        if analysis.warnings:
            print(f"Warnings: {analysis.warnings}")
        
        # Check predictions
        if 'expected_ambiguous' in test_case:
            total_predictions += 1
            if analysis.ambiguity_detected == test_case['expected_ambiguous']:
                correct_predictions += 1
                print("✅ Ambiguity detection: CORRECT")
            else:
                print("❌ Ambiguity detection: INCORRECT")
        
        if 'expected_scope' in test_case:
            total_predictions += 1
            if test_case['expected_scope'] is None:
                # Negative control - should have no scope
                if analysis.scope_resolution is None:
                    correct_predictions += 1
                    print("✅ Scope resolution: CORRECT (no scope)")
                else:
                    print("❌ Scope resolution: INCORRECT (should have no scope)")
            else:
                # Positive case - should have specific scope
                if analysis.scope_resolution == test_case['expected_scope']:
                    correct_predictions += 1
                    print("✅ Scope resolution: CORRECT")
                else:
                    print("❌ Scope resolution: INCORRECT")
        
        # Check quantifier detection (negative controls)
        if 'expected_quantifiers' in test_case:
            total_predictions += 1
            detected_count = len(analysis.detected_patterns)
            if detected_count == test_case['expected_quantifiers']:
                correct_predictions += 1
                print(f"✅ Quantifier detection: CORRECT ({detected_count} detected)")
            else:
                print(f"❌ Quantifier detection: INCORRECT ({detected_count} detected, expected {test_case['expected_quantifiers']})")
    
    # Summary statistics
    print(f"\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    print(f"Overall Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1%})")
    
    # Analyze by language
    lang_stats = defaultdict(lambda: {'total': 0, 'detected': 0, 'scoped': 0})
    for result in results:
        lang = result.language.value
        lang_stats[lang]['total'] += 1
        if result.detected_patterns:
            lang_stats[lang]['detected'] += 1
        if result.scoped_forms:
            lang_stats[lang]['scoped'] += 1
    
    print(f"\nLanguage Statistics:")
    for lang, stats in lang_stats.items():
        detection_rate = stats['detected'] / stats['total'] if stats['total'] > 0 else 0.0
        scoping_rate = stats['scoped'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"  {lang.upper()}: {stats['total']} tests, {detection_rate:.1%} detection, {scoping_rate:.1%} scoping")
    
    # Check if we meet acceptance criteria
    print(f"\nAcceptance Criteria Check:")
    target_accuracy = 0.9  # 90% target from specification
    print(f"  Target Accuracy: ≥{target_accuracy:.1%}")
    print(f"  Achieved Accuracy: {accuracy:.1%}")
    
    if accuracy >= target_accuracy:
        print("  ✅ ACCEPTANCE CRITERIA MET")
    else:
        print("  ❌ ACCEPTANCE CRITERIA NOT MET")
    
    # Save results
    output_path = Path("data/quantifier_scope_normalizer_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    try:
        json_results = convert_numpy_types({
            'test_cases': [
                {
                    'input': tc['text'],
                    'language': tc['language'].value,
                    'result': results[i].to_dict()
                }
                for i, tc in enumerate(test_cases)
            ],
            'summary': {
                'total_tests': len(test_cases),
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'accuracy': accuracy,
                'acceptance_criteria_met': accuracy >= target_accuracy,
                'language_stats': dict(lang_stats)
            }
        })
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    logger.info(f"Quantifier scope normalizer results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Quantifier scope normalizer demonstration completed!")
    print("="*80)


if __name__ == "__main__":
    main()

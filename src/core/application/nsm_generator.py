#!/usr/bin/env python3
"""
NSM Text Generator

Real NSM-based text generation using semantic composition and grammar rules.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from ..domain.models import Language, PrimeType, NSMPrime


@dataclass
class NSMGrammarRule:
    """Represents an NSM grammar rule for text generation."""
    
    pattern: List[str]  # Prime sequence pattern
    template: str       # Generation template
    constraints: Dict[str, Any]  # Grammatical constraints
    confidence: float   # Rule confidence


class NSMTextGenerator:
    """Real NSM-based text generator using semantic composition."""
    
    def __init__(self):
        """Initialize the NSM text generator."""
        self.grammar_rules = self._initialize_grammar_rules()
        self.prime_mappings = self._initialize_prime_mappings()
        
    def _initialize_grammar_rules(self) -> List[NSMGrammarRule]:
        """Initialize NSM grammar rules for text generation."""
        return [
            # Mental predicates
            NSMGrammarRule(
                pattern=["PEOPLE", "THINK", "GOOD"],
                template="people think this is good",
                constraints={"subject": "PEOPLE", "object": "THIS"},
                confidence=0.9
            ),
            NSMGrammarRule(
                pattern=["THINK", "GOOD"],
                template="I think this is good",
                constraints={},
                confidence=0.8
            ),
            
            # Evaluative statements
            NSMGrammarRule(
                pattern=["PEOPLE", "GOOD"],
                template="{subject} are good",
                constraints={"subject": "PEOPLE"},
                confidence=0.8
            ),
            NSMGrammarRule(
                pattern=["THIS", "GOOD"],
                template="this is good",
                constraints={},
                confidence=0.9
            ),
            
            # Quantification
            NSMGrammarRule(
                pattern=["MANY", "PEOPLE"],
                template="many people",
                constraints={},
                confidence=0.9
            ),
            NSMGrammarRule(
                pattern=["MORE", "PEOPLE"],
                template="more people",
                constraints={},
                confidence=0.9
            ),
            
            # Negation
            NSMGrammarRule(
                pattern=["NOT", "GOOD"],
                template="this is not good",
                constraints={},
                confidence=0.9
            ),
            NSMGrammarRule(
                pattern=["NOT", "THINK"],
                template="I do not think",
                constraints={},
                confidence=0.8
            ),
            
            # Action predicates
            NSMGrammarRule(
                pattern=["PEOPLE", "DO", "THIS"],
                template="{subject} do{s} this",
                constraints={"subject": "PEOPLE"},
                confidence=0.8
            ),
            NSMGrammarRule(
                pattern=["READ", "MANY"],
                template="read many",
                constraints={},
                confidence=0.8
            ),
            
            # Intensification
            NSMGrammarRule(
                pattern=["VERY", "GOOD"],
                template="very good",
                constraints={},
                confidence=0.9
            ),
            NSMGrammarRule(
                pattern=["VERY", "BAD"],
                template="very bad",
                constraints={},
                confidence=0.9
            ),
            NSMGrammarRule(
                pattern=["GOOD"],
                template="this is good",
                constraints={},
                confidence=0.9
            ),
            NSMGrammarRule(
                pattern=["BAD"],
                template="this is bad",
                constraints={},
                confidence=0.9
            ),
        ]
    
    def _initialize_prime_mappings(self) -> Dict[str, Dict[str, str]]:
        """Initialize prime to surface form mappings for different languages."""
        return {
            "en": {
                "PEOPLE": "people",
                "THINK": "think",
                "GOOD": "good",
                "BAD": "bad",
                "THIS": "this",
                "VERY": "very",
                "NOT": "not",
                "MORE": "more",
                "MANY": "many",
                "READ": "read",
                "DO": "do",
                "TRUE": "true",
                "FALSE": "false",
                "SOME": "some",
                "ALL": "all",
                "HALF": "half",
            },
            "es": {
                "PEOPLE": "gente",
                "THINK": "piensan",
                "GOOD": "bueno",
                "BAD": "malo",
                "THIS": "esto",
                "VERY": "muy",
                "NOT": "no",
                "MORE": "más",
                "MANY": "muchos",
                "READ": "leer",
                "DO": "hacer",
                "TRUE": "verdadero",
                "FALSE": "falso",
                "SOME": "algunos",
                "ALL": "todos",
                "HALF": "mitad",
            },
            "fr": {
                "PEOPLE": "gens",
                "THINK": "pensent",
                "GOOD": "bon",
                "BAD": "mauvais",
                "THIS": "ceci",
                "VERY": "très",
                "NOT": "pas",
                "MORE": "plus",
                "MANY": "beaucoup",
                "READ": "lire",
                "DO": "faire",
                "TRUE": "vrai",
                "FALSE": "faux",
                "SOME": "quelques",
                "ALL": "tous",
                "HALF": "moitié",
            }
        }
    
    def generate_text(self, primes: List[str], target_language: Language) -> str:
        """Generate text from NSM primes using real grammar rules."""
        if not primes:
            return ""
        
        # Find matching grammar rule
        matching_rule = self._find_matching_rule(primes)
        
        if matching_rule:
            return self._apply_grammar_rule(matching_rule, primes, target_language)
        else:
            # Fallback: compose from individual primes
            return self._compose_from_primes(primes, target_language)
    
    def _find_matching_rule(self, primes: List[str]) -> Optional[NSMGrammarRule]:
        """Find the best matching grammar rule for the given primes."""
        best_rule = None
        best_score = 0
        
        for rule in self.grammar_rules:
            score = self._calculate_rule_match_score(rule.pattern, primes)
            if score > best_score:
                best_score = score
                best_rule = rule
        
        # Only use rule if it matches well (exact pattern match or high coverage)
        if best_score >= 0.8:  # Require 80% match for rule usage
            return best_rule
        return None
    
    def _calculate_rule_match_score(self, pattern: List[str], primes: List[str]) -> float:
        """Calculate how well a rule pattern matches the given primes."""
        if not pattern or not primes:
            return 0.0
        
        # Check for exact pattern match first
        if len(pattern) == len(primes):
            if all(p in primes for p in pattern):
                return 1.0
        
        # For partial matches, check pattern coverage
        pattern_matches = 0
        for pattern_prime in pattern:
            if pattern_prime in primes:
                pattern_matches += 1
        
        # For input coverage, check how many input primes are used
        input_matches = 0
        for prime in primes:
            if prime in pattern:
                input_matches += 1
        
        # Calculate scores
        pattern_score = pattern_matches / len(pattern) if pattern else 0.0
        input_score = input_matches / len(primes) if primes else 0.0
        
        # For high-quality matches, both scores should be high
        return min(pattern_score, input_score)
    
    def _apply_grammar_rule(self, rule: NSMGrammarRule, primes: List[str], 
                           target_language: Language) -> str:
        """Apply a grammar rule to generate text."""
        # Get surface forms for the target language
        lang_code = target_language.value
        surface_forms = self.prime_mappings.get(lang_code, self.prime_mappings["en"])
        
        # Apply the template
        text = rule.template
        
        # Replace placeholders with actual values
        if "{subject}" in text:
            subject = self._get_subject(primes, surface_forms)
            text = text.replace("{subject}", subject)
        
        if "{object}" in text:
            object_prime = self._get_object(primes, surface_forms)
            text = text.replace("{object}", object_prime)
        
        # Handle subject-verb agreement
        if "{s}" in text:
            subject = self._get_subject(primes, surface_forms)
            suffix = "s" if subject != "I" else ""
            text = text.replace("{s}", suffix)
        
        # Clean up any remaining template artifacts
        text = text.replace("  ", " ").strip()
        
        # Translate the template to target language
        return self._translate_template(text, target_language, surface_forms)
    
    def _get_subject(self, primes: List[str], surface_forms: Dict[str, str]) -> str:
        """Get the subject from primes."""
        for prime in primes:
            if prime == "PEOPLE":
                return surface_forms.get("PEOPLE", "people")
            elif prime == "I":
                return "I"
        return "this"
    
    def _get_object(self, primes: List[str], surface_forms: Dict[str, str]) -> str:
        """Get the object from primes."""
        for prime in primes:
            if prime in ["THIS", "GOOD", "BAD"]:
                return surface_forms.get(prime, prime.lower())
        return "this"
    
    def _compose_from_primes(self, primes: List[str], target_language: Language) -> str:
        """Compose text from individual primes when no rule matches."""
        lang_code = target_language.value
        surface_forms = self.prime_mappings.get(lang_code, self.prime_mappings["en"])
        
        # Convert primes to surface forms
        words = []
        for prime in primes:
            surface_form = surface_forms.get(prime, prime.lower())
            words.append(surface_form)
        
        # Basic composition rules
        if len(words) == 1:
            return words[0]
        elif len(words) == 2:
            # Handle common two-word patterns
            if words[0] in ["very", "muy", "très"]:
                return f"{words[0]} {words[1]}"
            elif words[1] in ["good", "bad", "bueno", "malo", "bon", "mauvais"]:
                return f"this is {words[0]} {words[1]}"
            elif words[0] in ["good", "bad", "bueno", "malo", "bon", "mauvais"]:
                return f"this is {words[0]}"
            else:
                return f"{words[0]} {words[1]}"
        else:
            # For longer sequences, use basic composition
            return " ".join(words)
    
    def _translate_template(self, template: str, target_language: Language, 
                           surface_forms: Dict[str, str]) -> str:
        """Translate a template to the target language."""
        if target_language == Language.ENGLISH:
            return template
        
        # Simple template translation rules
        translated = template
        
        # Common patterns for different languages
        if target_language == Language.SPANISH:
            translated = translated.replace("this is", "esto es")
            translated = translated.replace("I think", "yo pienso")
            translated = translated.replace("People think", "La gente piensa")
            translated = translated.replace("people", "gente")
            translated = translated.replace("think", "piensan")
            translated = translated.replace("good", "bueno")
            translated = translated.replace("bad", "malo")
            translated = translated.replace("very", "muy")
            translated = translated.replace("not", "no")
            translated = translated.replace("many", "muchos")
            translated = translated.replace("more", "más")
            translated = translated.replace("do", "hacer")
            translated = translated.replace("read", "leer")
            
        elif target_language == Language.FRENCH:
            translated = translated.replace("this is", "ceci est")
            translated = translated.replace("I think", "je pense")
            translated = translated.replace("People think", "Les gens pensent")
            translated = translated.replace("people", "gens")
            translated = translated.replace("think", "pensent")
            translated = translated.replace("good", "bon")
            translated = translated.replace("bad", "mauvais")
            translated = translated.replace("very", "très")
            translated = translated.replace("not", "pas")
            translated = translated.replace("many", "beaucoup")
            translated = translated.replace("more", "plus")
            translated = translated.replace("do", "faire")
            translated = translated.replace("read", "lire")
        
        return translated
    
    def get_generation_confidence(self, primes: List[str]) -> float:
        """Calculate confidence in the generation based on rule matching."""
        if not primes:
            return 0.0
        
        matching_rule = self._find_matching_rule(primes)
        if matching_rule:
            return matching_rule.confidence
        
        # Lower confidence for composition without rules
        return 0.5


# Global generator instance
_nsm_generator = None

def get_nsm_generator() -> NSMTextGenerator:
    """Get the global NSM text generator instance."""
    global _nsm_generator
    if _nsm_generator is None:
        _nsm_generator = NSMTextGenerator()
    return _nsm_generator

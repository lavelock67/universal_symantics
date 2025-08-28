#!/usr/bin/env python3
"""
Prime-to-Text Generation Service

This module converts detected NSM primes back to natural language text,
enabling the reverse direction of the universal translator pipeline.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from ..domain.models import Language, NSMPrime
from .grammar_engine import GrammarEngine
from .language_expansion import LanguageExpansion

logger = logging.getLogger(__name__)

class GenerationStrategy(Enum):
    """Strategies for prime-to-text generation."""
    LEXICAL = "lexical"  # Direct word mapping
    CONTEXTUAL = "contextual"  # Context-aware generation
    TEMPLATE = "template"  # Template-based generation

@dataclass
class GenerationResult:
    """Result of prime-to-text generation."""
    text: str
    confidence: float
    strategy: GenerationStrategy
    source_primes: List[NSMPrime]
    metadata: Dict

class PrimeGenerator:
    """Converts NSM primes back to natural language text."""
    
    def __init__(self):
        """Initialize the prime generator."""
        self.language_mappings = {}
        self.templates = {}
        self.grammar_engine = GrammarEngine()
        self.language_expansion = LanguageExpansion()
        self._load_mappings()
        self._load_templates()
        self._integrate_extended_languages()
    
    def _load_mappings(self):
        """Load language-specific prime-to-word mappings."""
        # English mappings
        self.language_mappings[Language.ENGLISH] = {
            # Substantives
            "I": "I",
            "YOU": "you",
            "SOMEONE": "someone",
            "PEOPLE": "people",
            "SOMETHING": "something",
            "THING": "thing",
            "BODY": "body",
            
            # Relational substantives
            "KIND": "kind",
            "PART": "part",
            
            # Determiners and quantifiers
            "THIS": "this",
            "THE_SAME": "the same",
            "OTHER": "other",
            "ONE": "one",
            "TWO": "two",
            "SOME": "some",
            "ALL": "all",
            "MUCH": "much",
            "MANY": "many",
            
            # Evaluators and descriptors
            "GOOD": "good",
            "BAD": "bad",
            "BIG": "big",
            "SMALL": "small",
            
            # Mental predicates
            "THINK": "think",
            "KNOW": "know",
            "WANT": "want",
            "FEEL": "feel",
            "SEE": "see",
            "HEAR": "hear",
            
            # Speech
            "SAY": "say",
            "WORDS": "words",
            "TRUE": "true",
            "FALSE": "false",
            
            # Actions and events
            "DO": "do",
            "HAPPEN": "happen",
            "MOVE": "move",
            "TOUCH": "touch",
            
            # Location, existence, possession
            "BE_SOMEWHERE": "somewhere",
            "THERE_IS": "there is",
            "HAVE": "have",
            "BE_SOMEONE": "someone",
            
            # Life and death
            "LIVE": "live",
            "DIE": "die",
            
            # Time
            "WHEN": "when",
            "NOW": "now",
            "BEFORE": "before",
            "AFTER": "after",
            "A_LONG_TIME": "a long time",
            "A_SHORT_TIME": "a short time",
            "FOR_SOME_TIME": "for some time",
            "MOMENT": "moment",
            
            # Space
            "WHERE": "where",
            "HERE": "here",
            "ABOVE": "above",
            "BELOW": "below",
            "FAR": "far",
            "NEAR": "near",
            "INSIDE": "inside",
            
            # Logical concepts
            "NOT": "not",
            "MAYBE": "maybe",
            "CAN": "can",
            "BECAUSE": "because",
            "IF": "if",
            
            # Intensifiers
            "VERY": "very",
            "MORE": "more",
            "LIKE": "like",
        }
        
        # Spanish mappings (comprehensive)
        self.language_mappings[Language.SPANISH] = {
            # Substantives
            "I": "yo", "YOU": "tú", "SOMEONE": "alguien", "PEOPLE": "gente",
            "SOMETHING": "algo", "THING": "cosa", "BODY": "cuerpo",
            
            # Relational substantives
            "KIND": "tipo", "PART": "parte",
            
            # Determiners and quantifiers
            "THIS": "esto", "THE_SAME": "lo mismo", "OTHER": "otro",
            "ONE": "uno", "TWO": "dos", "SOME": "algunos", "ALL": "todos",
            "MUCH": "mucho", "MANY": "muchos",
            
            # Evaluators and descriptors
            "GOOD": "bueno", "BAD": "malo", "BIG": "grande", "SMALL": "pequeño",
            
            # Mental predicates
            "THINK": "pensar", "KNOW": "saber", "WANT": "querer",
            "FEEL": "sentir", "SEE": "ver", "HEAR": "oír",
            
            # Speech
            "SAY": "decir", "WORDS": "palabras", "TRUE": "verdadero", "FALSE": "falso",
            
            # Actions and events
            "DO": "hacer", "HAPPEN": "pasar", "MOVE": "mover", "TOUCH": "tocar",
            
            # Location, existence, possession
            "BE_SOMEWHERE": "en algún lugar", "THERE_IS": "hay", "HAVE": "tener",
            "BE_SOMEONE": "ser alguien",
            
            # Life and death
            "LIVE": "vivir", "DIE": "morir",
            
            # Time
            "WHEN": "cuándo", "NOW": "ahora", "BEFORE": "antes", "AFTER": "después",
            "A_LONG_TIME": "mucho tiempo", "A_SHORT_TIME": "poco tiempo",
            "FOR_SOME_TIME": "por algún tiempo", "MOMENT": "momento",
            
            # Space
            "WHERE": "dónde", "HERE": "aquí", "ABOVE": "arriba", "BELOW": "abajo",
            "FAR": "lejos", "NEAR": "cerca", "INSIDE": "dentro",
            
            # Logical concepts
            "NOT": "no", "MAYBE": "tal vez", "CAN": "poder", "BECAUSE": "porque", "IF": "si",
            
            # Intensifiers
            "VERY": "muy", "MORE": "más", "LIKE": "como",
        }
        
        # French mappings (comprehensive)
        self.language_mappings[Language.FRENCH] = {
            # Substantives
            "I": "je", "YOU": "tu", "SOMEONE": "quelqu'un", "PEOPLE": "gens",
            "SOMETHING": "quelque chose", "THING": "chose", "BODY": "corps",
            
            # Relational substantives
            "KIND": "genre", "PART": "partie",
            
            # Determiners and quantifiers
            "THIS": "ceci", "THE_SAME": "le même", "OTHER": "autre",
            "ONE": "un", "TWO": "deux", "SOME": "quelques", "ALL": "tous",
            "MUCH": "beaucoup", "MANY": "beaucoup",
            
            # Evaluators and descriptors
            "GOOD": "bon", "BAD": "mauvais", "BIG": "grand", "SMALL": "petit",
            
            # Mental predicates
            "THINK": "penser", "KNOW": "savoir", "WANT": "vouloir",
            "FEEL": "sentir", "SEE": "voir", "HEAR": "entendre",
            
            # Speech
            "SAY": "dire", "WORDS": "mots", "TRUE": "vrai", "FALSE": "faux",
            
            # Actions and events
            "DO": "faire", "HAPPEN": "arriver", "MOVE": "bouger", "TOUCH": "toucher",
            
            # Location, existence, possession
            "BE_SOMEWHERE": "quelque part", "THERE_IS": "il y a", "HAVE": "avoir",
            "BE_SOMEONE": "être quelqu'un",
            
            # Life and death
            "LIVE": "vivre", "DIE": "mourir",
            
            # Time
            "WHEN": "quand", "NOW": "maintenant", "BEFORE": "avant", "AFTER": "après",
            "A_LONG_TIME": "longtemps", "A_SHORT_TIME": "peu de temps",
            "FOR_SOME_TIME": "pendant quelque temps", "MOMENT": "moment",
            
            # Space
            "WHERE": "où", "HERE": "ici", "ABOVE": "au-dessus", "BELOW": "en-dessous",
            "FAR": "loin", "NEAR": "près", "INSIDE": "dedans",
            
            # Logical concepts
            "NOT": "ne pas", "MAYBE": "peut-être", "CAN": "pouvoir", "BECAUSE": "parce que", "IF": "si",
            
            # Intensifiers
            "VERY": "très", "MORE": "plus", "LIKE": "comme",
        }
    
    def _integrate_extended_languages(self):
        """Integrate extended language mappings from language expansion."""
        for language in self.language_expansion.get_supported_languages():
            # Always use extended mappings to ensure consistency
            extended_mappings = self.language_expansion.get_mappings(language)
            self.language_mappings[language] = extended_mappings
            logger.info(f"Integrated extended mappings for {language.value}: {len(extended_mappings)} primes")
    
    def _load_templates(self):
        """Load generation templates for common patterns."""
        self.templates = {
            "statement": "{subject} {predicate}",
            "question": "{predicate} {subject}",
            "negation": "{subject} {not} {predicate}",
            "condition": "if {condition}, {consequence}",
            "cause": "{effect} because {cause}",
        }
    
    def generate_text(self, primes: List[NSMPrime], target_language: Language, 
                     strategy: GenerationStrategy = GenerationStrategy.LEXICAL) -> GenerationResult:
        """Generate natural language text from detected primes.
        
        Args:
            primes: List of detected NSM primes
            target_language: Target language for generation
            strategy: Generation strategy to use
            
        Returns:
            GenerationResult with generated text and metadata
        """
        logger.info(f"Generating text from {len(primes)} primes in {target_language.value}")
        
        if strategy == GenerationStrategy.LEXICAL:
            return self._generate_lexical(primes, target_language)
        elif strategy == GenerationStrategy.CONTEXTUAL:
            return self._generate_contextual(primes, target_language)
        elif strategy == GenerationStrategy.TEMPLATE:
            return self._generate_template(primes, target_language)
        else:
            raise ValueError(f"Unknown generation strategy: {strategy}")
    
    def _generate_lexical(self, primes: List[NSMPrime], target_language: Language) -> GenerationResult:
        """Generate text using direct lexical mapping with grammar enhancement."""
        if not primes:
            return GenerationResult(
                text="",
                confidence=0.0,
                strategy=GenerationStrategy.LEXICAL,
                source_primes=primes,
                metadata={"error": "No primes provided"}
            )
        
        # Get language mappings
        mappings = self.language_mappings.get(target_language, {})
        
        if not mappings:
            logger.warning(f"No mappings found for language: {target_language.value}")
            return GenerationResult(
                text="",
                confidence=0.0,
                strategy=GenerationStrategy.LEXICAL,
                source_primes=primes,
                metadata={"error": f"No mappings for {target_language.value}"}
            )
        
        # Apply grammar enhancement
        try:
            text = self.grammar_engine.process_translation(primes, target_language)
            confidence = 0.8  # Higher confidence with grammar enhancement
            grammar_enhanced = True
        except Exception as e:
            logger.warning(f"Grammar enhancement failed, falling back to basic: {e}")
            # Fallback to basic word joining
            words = []
            mapped_count = 0
            
            for prime in primes:
                word = mappings.get(prime.text, prime.text.lower())
                words.append(word)
                if word != prime.text.lower():
                    mapped_count += 1
            
            confidence = mapped_count / len(primes) if primes else 0.0
            text = " ".join(words)
            
            # Capitalize first letter
            if text:
                text = text[0].upper() + text[1:]
            
            grammar_enhanced = False
        
        return GenerationResult(
            text=text,
            confidence=confidence,
            strategy=GenerationStrategy.LEXICAL,
            source_primes=primes,
            metadata={
                "prime_count": len(primes),
                "grammar_enhanced": grammar_enhanced,
                "word_order": "enhanced" if grammar_enhanced else "basic"
            }
        )
    
    def _generate_contextual(self, primes: List[NSMPrime], target_language: Language) -> GenerationResult:
        """Generate text using context-aware generation."""
        # For now, fall back to lexical generation
        # TODO: Implement context-aware generation
        logger.info("Contextual generation not yet implemented, falling back to lexical")
        return self._generate_lexical(primes, target_language)
    
    def _generate_template(self, primes: List[NSMPrime], target_language: Language) -> GenerationResult:
        """Generate text using template-based generation."""
        # For now, fall back to lexical generation
        # TODO: Implement template-based generation
        logger.info("Template generation not yet implemented, falling back to lexical")
        return self._generate_lexical(primes, target_language)
    
    def get_supported_languages(self) -> List[Language]:
        """Get list of supported languages for generation."""
        return list(self.language_mappings.keys())
    
    def get_coverage(self, target_language: Language) -> Dict[str, int]:
        """Get coverage statistics for a language."""
        mappings = self.language_mappings.get(target_language, {})
        total_primes = len(self.language_mappings.get(Language.ENGLISH, {}))
        mapped_primes = len(mappings)
        
        return {
            "total_primes": total_primes,
            "mapped_primes": mapped_primes,
            "coverage_percentage": (mapped_primes / total_primes * 100) if total_primes > 0 else 0
        }

#!/usr/bin/env python3
"""
Grammar Engine for NSM Universal Translator

This module provides grammatical structure and rules to transform
prime-based translations into proper, grammatically correct sentences.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging

from ..domain.models import Language, NSMPrime
from .language_expansion import LanguageExpansion

logger = logging.getLogger(__name__)

class GrammarRule(Enum):
    """Types of grammar rules."""
    SUBJECT_VERB_AGREEMENT = "subject_verb_agreement"
    TENSE_CONJUGATION = "tense_conjugation"
    ARTICLE_PLACEMENT = "article_placement"
    ADJECTIVE_ORDER = "adjective_order"
    NEGATION_PLACEMENT = "negation_placement"
    QUESTION_FORMATION = "question_formation"
    CONJUNCTION_PLACEMENT = "conjunction_placement"

@dataclass
class GrammarStructure:
    """Represents a grammatical sentence structure."""
    subject: Optional[str] = None
    verb: Optional[str] = None
    object_direct: Optional[str] = None
    object_indirect: Optional[str] = None
    adjectives: List[str] = None
    adverbs: List[str] = None
    tense: str = "present"
    mood: str = "indicative"
    negation: bool = False
    question: bool = False

class GrammarEngine:
    """Grammar engine for improving translation quality."""
    
    def __init__(self):
        """Initialize the grammar engine."""
        self.grammar_rules = {}
        self.conjugation_tables = {}
        self.article_rules = {}
        self.language_expansion = LanguageExpansion()
        self._load_grammar_rules()
        self._load_conjugation_tables()
        self._load_article_rules()
        self._integrate_extended_grammar_rules()
    
    def _load_grammar_rules(self):
        """Load language-specific grammar rules."""
        # English grammar rules
        self.grammar_rules[Language.ENGLISH] = {
            "word_order": "SVO",  # Subject-Verb-Object
            "adjective_position": "before_noun",
            "adverb_position": "before_verb",
            "negation_word": "not",
            "question_inversion": True,
            "articles": ["a", "an", "the"],
            "auxiliary_verbs": ["be", "have", "do", "can", "will", "would", "should"]
        }
        
        # Spanish grammar rules
        self.grammar_rules[Language.SPANISH] = {
            "word_order": "SVO",
            "adjective_position": "after_noun",
            "adverb_position": "after_verb",
            "negation_word": "no",
            "question_inversion": False,
            "articles": ["el", "la", "los", "las", "un", "una", "unos", "unas"],
            "auxiliary_verbs": ["ser", "estar", "haber", "tener", "poder", "deber"]
        }
        
        # French grammar rules
        self.grammar_rules[Language.FRENCH] = {
            "word_order": "SVO",
            "adjective_position": "after_noun",
            "adverb_position": "after_verb",
            "negation_word": "ne pas",
            "question_inversion": True,
            "articles": ["le", "la", "les", "un", "une", "des"],
            "auxiliary_verbs": ["être", "avoir", "faire", "pouvoir", "devoir"]
        }
    
    def _integrate_extended_grammar_rules(self):
        """Integrate extended grammar rules from language expansion."""
        for language in self.language_expansion.get_supported_languages():
            # Always use extended grammar rules to ensure consistency
            extended_rules = self.language_expansion.get_grammar_rules(language)
            self.grammar_rules[language] = extended_rules
            logger.info(f"Integrated extended grammar rules for {language.value}")
    
    def _load_conjugation_tables(self):
        """Load verb conjugation tables."""
        # English conjugations (simplified)
        self.conjugation_tables[Language.ENGLISH] = {
            "present": {
                "I": "think",
                "you": "think", 
                "he": "thinks",
                "she": "thinks",
                "it": "thinks",
                "we": "think",
                "they": "think"
            },
            "past": {
                "I": "thought",
                "you": "thought",
                "he": "thought", 
                "she": "thought",
                "it": "thought",
                "we": "thought",
                "they": "thought"
            },
            "future": {
                "I": "will think",
                "you": "will think",
                "he": "will think",
                "she": "will think", 
                "it": "will think",
                "we": "will think",
                "they": "will think"
            }
        }
        
        # Spanish conjugations (basic)
        self.conjugation_tables[Language.SPANISH] = {
            "present": {
                "yo": "pienso",
                "tú": "piensas",
                "él": "piensa",
                "ella": "piensa",
                "nosotros": "pensamos",
                "ellos": "piensan"
            },
            "past": {
                "yo": "pensé",
                "tú": "pensaste", 
                "él": "pensó",
                "ella": "pensó",
                "nosotros": "pensamos",
                "ellos": "pensaron"
            },
            "future": {
                "yo": "pensaré",
                "tú": "pensarás",
                "él": "pensará",
                "ella": "pensará",
                "nosotros": "pensaremos", 
                "ellos": "pensarán"
            }
        }
        
        # French conjugations (basic)
        self.conjugation_tables[Language.FRENCH] = {
            "present": {
                "je": "pense",
                "tu": "penses",
                "il": "pense",
                "elle": "pense",
                "nous": "pensons",
                "ils": "pensent"
            },
            "past": {
                "je": "pensais",
                "tu": "pensais",
                "il": "pensait",
                "elle": "pensait",
                "nous": "pensions",
                "ils": "pensaient"
            },
            "future": {
                "je": "penserai",
                "tu": "penseras",
                "il": "pensera",
                "elle": "pensera",
                "nous": "penserons",
                "ils": "penseront"
            }
        }
    
    def _load_article_rules(self):
        """Load article placement rules."""
        self.article_rules = {
            Language.ENGLISH: {
                "definite": "the",
                "indefinite_singular": "a",
                "indefinite_plural": "some",
                "vowel_sound": "an"
            },
            Language.SPANISH: {
                "definite_masculine": "el",
                "definite_feminine": "la", 
                "definite_plural_m": "los",
                "definite_plural_f": "las",
                "indefinite_masculine": "un",
                "indefinite_feminine": "una",
                "indefinite_plural_m": "unos",
                "indefinite_plural_f": "unas"
            },
            Language.FRENCH: {
                "definite_masculine": "le",
                "definite_feminine": "la",
                "definite_plural": "les", 
                "indefinite_masculine": "un",
                "indefinite_feminine": "une",
                "indefinite_plural": "des"
            }
        }
    
    def analyze_structure(self, primes: List[NSMPrime]) -> GrammarStructure:
        """Analyze primes to determine grammatical structure."""
        structure = GrammarStructure()
        
        # Extract subject (I, YOU, SOMEONE, PEOPLE)
        subject_primes = ["I", "YOU", "SOMEONE", "PEOPLE"]
        for prime in primes:
            if prime.text in subject_primes:
                structure.subject = prime.text
                break
        
        # Extract verb (THINK, KNOW, WANT, DO, etc.)
        verb_primes = ["THINK", "KNOW", "WANT", "DO", "SAY", "SEE", "HEAR", "FEEL"]
        for prime in primes:
            if prime.text in verb_primes:
                structure.verb = prime.text
                break
        
        # Extract object (THING, SOMETHING, etc.)
        object_primes = ["THING", "SOMETHING", "THIS", "THAT"]
        for prime in primes:
            if prime.text in object_primes:
                structure.object_direct = prime.text
                break
        
        # Extract adjectives (GOOD, BAD, BIG, SMALL, etc.)
        adjective_primes = ["GOOD", "BAD", "BIG", "SMALL", "VERY", "MANY", "SOME", "ALL"]
        structure.adjectives = [p.text for p in primes if p.text in adjective_primes]
        
        # Extract adverbs (NOW, HERE, THERE, etc.)
        adverb_primes = ["NOW", "HERE", "THERE", "WHEN", "WHERE", "VERY"]
        structure.adverbs = [p.text for p in primes if p.text in adverb_primes]
        
        # Check for negation
        structure.negation = any(p.text == "NOT" for p in primes)
        
        # Check for questions
        question_primes = ["WHEN", "WHERE", "WHAT", "WHO", "WHY", "HOW"]
        structure.question = any(p.text in question_primes for p in primes)
        
        return structure
    
    def conjugate_verb(self, verb: str, subject: str, tense: str, language: Language) -> str:
        """Conjugate a verb based on subject and tense."""
        if language not in self.conjugation_tables:
            return verb
        
        conjugations = self.conjugation_tables[language]
        if tense not in conjugations:
            return verb
        
        # Map subject to conjugation key
        subject_mapping = {
            Language.ENGLISH: {
                "I": "I", "YOU": "you", "SOMEONE": "he", "PEOPLE": "they"
            },
            Language.SPANISH: {
                "I": "yo", "YOU": "tú", "SOMEONE": "él", "PEOPLE": "ellos"
            },
            Language.FRENCH: {
                "I": "je", "YOU": "tu", "SOMEONE": "il", "PEOPLE": "ils"
            }
        }
        
        conjugation_key = subject_mapping.get(language, {}).get(subject, subject)
        
        # Get the conjugated form, fallback to original verb if not found
        conjugated = conjugations[tense].get(conjugation_key, verb)
        
        # For now, return the base verb form if conjugation not found
        # This will be improved with more comprehensive conjugation tables
        return conjugated if conjugated != verb else verb.lower()
    
    def add_articles(self, words: List[str], language: Language) -> List[str]:
        """Add appropriate articles to nouns."""
        if language not in self.article_rules:
            return words
        
        rules = self.article_rules[language]
        result = []
        
        for i, word in enumerate(words):
            # Check if this is a noun that needs an article
            if word.lower() in ["thing", "cosa", "chose", "person", "persona", "personne"]:
                if language == Language.ENGLISH:
                    result.append("a")
                elif language == Language.SPANISH:
                    if word.lower() in ["cosa", "persona"]:
                        result.append("una" if word.lower().endswith("a") else "un")
                elif language == Language.FRENCH:
                    if word.lower() in ["chose", "personne"]:
                        result.append("une" if word.lower().endswith("e") else "un")
            
            result.append(word)
        
        return result
    
    def apply_word_order(self, structure: GrammarStructure, language: Language) -> List[str]:
        """Apply language-specific word order rules."""
        if language not in self.grammar_rules:
            return []
        
        rules = self.grammar_rules[language]
        words = []
        
        # Build sentence structure properly
        sentence_parts = []
        
        # Subject
        if structure.subject:
            subject_word = self._map_word_to_language(structure.subject, language)
            sentence_parts.append(subject_word)
        
        # Verb
        if structure.verb:
            conjugated_verb = self.conjugate_verb(
                structure.verb, 
                structure.subject or "I", 
                structure.tense, 
                language
            )
            verb_word = self._map_word_to_language(conjugated_verb, language)
            sentence_parts.append(verb_word)
        
        # Object
        if structure.object_direct:
            object_word = self._map_word_to_language(structure.object_direct, language)
            sentence_parts.append(object_word)
        
        # Adjectives (position depends on language)
        if structure.adjectives:
            adj_words = [self._map_word_to_language(adj, language) for adj in structure.adjectives]
            if rules["adjective_position"] == "before_noun" and structure.object_direct:
                # Insert adjectives before the object
                sentence_parts = sentence_parts[:-1] + adj_words + sentence_parts[-1:]
            elif rules["adjective_position"] == "after_noun":
                sentence_parts.extend(adj_words)
        
        # Adverbs
        if structure.adverbs:
            adv_words = [self._map_word_to_language(adv, language) for adv in structure.adverbs]
            if rules["adverb_position"] == "before_verb" and len(sentence_parts) > 1:
                sentence_parts.insert(1, adv_words[0] if adv_words else "")
            elif rules["adverb_position"] == "after_verb":
                sentence_parts.extend(adv_words)
        
        # Handle negation
        if structure.negation:
            neg_word = rules["negation_word"]
            if language == Language.ENGLISH:
                # Insert "not" after the verb
                if len(sentence_parts) > 1:
                    sentence_parts.insert(2, neg_word)
            elif language == Language.SPANISH:
                sentence_parts.insert(0, neg_word)
            elif language == Language.FRENCH:
                sentence_parts.insert(0, "ne")
                sentence_parts.append("pas")
        
        # Handle questions
        if structure.question and rules["question_inversion"]:
            # Basic question inversion
            if len(sentence_parts) > 1:
                sentence_parts[0], sentence_parts[1] = sentence_parts[1], sentence_parts[0]
        
        return sentence_parts
    
    def improve_translation(self, words: List[str], language: Language) -> str:
        """Apply grammar improvements to a list of words."""
        # Join words with proper spacing
        text = " ".join(words)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Add punctuation
        if not text.endswith((".", "!", "?")):
            text += "."
        
        # Fix common issues
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Fix punctuation spacing
        
        return text
    
    def _map_word_to_language(self, word: str, language: Language) -> str:
        """Map a word to the target language using basic mappings."""
        # Basic word mappings for common primes
        mappings = {
            Language.ENGLISH: {
                "I": "I", "YOU": "you", "SOMEONE": "someone", "PEOPLE": "people",
                "THING": "thing", "SOMETHING": "something", "THIS": "this", "THAT": "that",
                "GOOD": "good", "BAD": "bad", "BIG": "big", "SMALL": "small",
                "THINK": "think", "KNOW": "know", "WANT": "want", "SAY": "say",
                "DO": "do", "CAN": "can", "NOT": "not", "VERY": "very",
                "MANY": "many", "SOME": "some", "ALL": "all", "NOW": "now",
                "HERE": "here", "THERE": "there", "WHEN": "when", "WHERE": "where"
            },
            Language.SPANISH: {
                "I": "yo", "YOU": "tú", "SOMEONE": "alguien", "PEOPLE": "gente",
                "THING": "cosa", "SOMETHING": "algo", "THIS": "esto", "THAT": "eso",
                "GOOD": "bueno", "BAD": "malo", "BIG": "grande", "SMALL": "pequeño",
                "THINK": "pensar", "KNOW": "saber", "WANT": "querer", "SAY": "decir",
                "DO": "hacer", "CAN": "poder", "NOT": "no", "VERY": "muy",
                "MANY": "muchos", "SOME": "algunos", "ALL": "todos", "NOW": "ahora",
                "HERE": "aquí", "THERE": "allí", "WHEN": "cuándo", "WHERE": "dónde"
            },
            Language.FRENCH: {
                "I": "je", "YOU": "tu", "SOMEONE": "quelqu'un", "PEOPLE": "gens",
                "THING": "chose", "SOMETHING": "quelque chose", "THIS": "ceci", "THAT": "cela",
                "GOOD": "bon", "BAD": "mauvais", "BIG": "grand", "SMALL": "petit",
                "THINK": "penser", "KNOW": "savoir", "WANT": "vouloir", "SAY": "dire",
                "DO": "faire", "CAN": "pouvoir", "NOT": "ne pas", "VERY": "très",
                "MANY": "beaucoup", "SOME": "quelques", "ALL": "tous", "NOW": "maintenant",
                "HERE": "ici", "THERE": "là", "WHEN": "quand", "WHERE": "où"
            }
        }
        
        # Get the mapping for the target language
        language_mappings = mappings.get(language, {})
        
        # Return the mapped word or the original word in lowercase
        return language_mappings.get(word.upper(), word.lower())
    
    def process_translation(self, primes: List[NSMPrime], language: Language) -> str:
        """Process primes through the grammar engine to produce proper sentences."""
        try:
            # Analyze grammatical structure
            structure = self.analyze_structure(primes)
            
            # Apply word order and grammar rules
            words = self.apply_word_order(structure, language)
            
            # Improve the final translation
            result = self.improve_translation(words, language)
            
            logger.info(f"Grammar processing: {len(primes)} primes -> '{result}'")
            return result
            
        except Exception as e:
            logger.error(f"Grammar processing error: {e}")
            # Fallback to simple word joining
            return " ".join([p.text.lower() for p in primes]).capitalize() + "."

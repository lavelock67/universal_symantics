#!/usr/bin/env python3
"""
Language-Specific Exponent Lexicons for NSM Detection

This module provides allolexy tables and UD-morph maps for achieving
cross-lingual parity in NSM prime detection.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"

@dataclass
class Exponent:
    """Language-specific exponent for an NSM prime."""
    surface_form: str
    language: Language
    prime: str
    confidence: float
    ud_features: Dict[str, str]
    morphological_form: str
    register: str = "neutral"  # formal, informal, neutral

class ExponentLexicon:
    """Language-specific exponent lexicons for NSM primes."""
    
    def __init__(self):
        """Initialize exponent lexicons for all languages."""
        self.english_exponents = self._load_english_exponents()
        self.spanish_exponents = self._load_spanish_exponents()
        self.french_exponents = self._load_french_exponents()
        
        # Create reverse mappings for lookup
        self._create_reverse_mappings()
        
        # Create a unified exponents structure for easy access
        self.exponents = {
            "en": self.english_exponents,
            "es": self.spanish_exponents,
            "fr": self.french_exponents
        }
    
    def _load_english_exponents(self) -> Dict[str, List[Exponent]]:
        """Load English exponents for NSM primes."""
        return {
            # Substantives
            "PEOPLE": [
                Exponent("people", Language.ENGLISH, "PEOPLE", 0.95, {"Number": "Plur"}, "people", "neutral"),
                Exponent("person", Language.ENGLISH, "PEOPLE", 0.90, {"Number": "Sing"}, "person", "neutral"),
                Exponent("human", Language.ENGLISH, "PEOPLE", 0.85, {"Number": "Sing"}, "human", "formal"),
                Exponent("individual", Language.ENGLISH, "PEOPLE", 0.80, {"Number": "Sing"}, "individual", "formal"),
            ],
            "THING": [
                Exponent("thing", Language.ENGLISH, "THING", 0.95, {"Number": "Sing"}, "thing", "neutral"),
                Exponent("things", Language.ENGLISH, "THING", 0.90, {"Number": "Plur"}, "things", "neutral"),
                Exponent("object", Language.ENGLISH, "THING", 0.85, {"Number": "Sing"}, "object", "formal"),
                Exponent("item", Language.ENGLISH, "THING", 0.80, {"Number": "Sing"}, "item", "neutral"),
            ],
            "BODY": [
                Exponent("body", Language.ENGLISH, "BODY", 0.95, {"Number": "Sing"}, "body", "neutral"),
                Exponent("bodies", Language.ENGLISH, "BODY", 0.90, {"Number": "Plur"}, "bodies", "neutral"),
            ],
            
            # Mental Predicates
            "THINK": [
                Exponent("think", Language.ENGLISH, "THINK", 0.95, {"VerbForm": "Fin"}, "think", "neutral"),
                Exponent("thinks", Language.ENGLISH, "THINK", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "thinks", "neutral"),
                Exponent("thought", Language.ENGLISH, "THINK", 0.85, {"VerbForm": "Part", "Tense": "Past"}, "thought", "neutral"),
                Exponent("thinking", Language.ENGLISH, "THINK", 0.80, {"VerbForm": "Part", "Tense": "Pres"}, "thinking", "neutral"),
            ],
            "KNOW": [
                Exponent("know", Language.ENGLISH, "KNOW", 0.95, {"VerbForm": "Fin"}, "know", "neutral"),
                Exponent("knows", Language.ENGLISH, "KNOW", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "knows", "neutral"),
                Exponent("knew", Language.ENGLISH, "KNOW", 0.85, {"VerbForm": "Fin", "Tense": "Past"}, "knew", "neutral"),
                Exponent("known", Language.ENGLISH, "KNOW", 0.80, {"VerbForm": "Part", "Tense": "Past"}, "known", "neutral"),
            ],
            "WANT": [
                Exponent("want", Language.ENGLISH, "WANT", 0.95, {"VerbForm": "Fin"}, "want", "neutral"),
                Exponent("wants", Language.ENGLISH, "WANT", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "wants", "neutral"),
                Exponent("wanted", Language.ENGLISH, "WANT", 0.85, {"VerbForm": "Fin", "Tense": "Past"}, "wanted", "neutral"),
                Exponent("wanting", Language.ENGLISH, "WANT", 0.80, {"VerbForm": "Part", "Tense": "Pres"}, "wanting", "neutral"),
            ],
            "FEEL": [
                Exponent("feel", Language.ENGLISH, "FEEL", 0.95, {"VerbForm": "Fin"}, "feel", "neutral"),
                Exponent("feels", Language.ENGLISH, "FEEL", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "feels", "neutral"),
                Exponent("felt", Language.ENGLISH, "FEEL", 0.85, {"VerbForm": "Fin", "Tense": "Past"}, "felt", "neutral"),
                Exponent("feeling", Language.ENGLISH, "FEEL", 0.80, {"VerbForm": "Part", "Tense": "Pres"}, "feeling", "neutral"),
            ],
            
            # Quantifiers
            "ALL": [
                Exponent("all", Language.ENGLISH, "ALL", 0.95, {"PronType": "Tot"}, "all", "neutral"),
                Exponent("every", Language.ENGLISH, "ALL", 0.90, {"PronType": "Tot"}, "every", "neutral"),
                Exponent("each", Language.ENGLISH, "ALL", 0.85, {"PronType": "Tot"}, "each", "neutral"),
            ],
            "SOME": [
                Exponent("some", Language.ENGLISH, "SOME", 0.95, {"PronType": "Ind"}, "some", "neutral"),
                Exponent("several", Language.ENGLISH, "SOME", 0.85, {"PronType": "Ind"}, "several", "neutral"),
                Exponent("few", Language.ENGLISH, "SOME", 0.80, {"PronType": "Ind"}, "few", "neutral"),
            ],
            "MANY": [
                Exponent("many", Language.ENGLISH, "MANY", 0.95, {"PronType": "Ind"}, "many", "neutral"),
                Exponent("much", Language.ENGLISH, "MANY", 0.90, {"PronType": "Ind"}, "much", "neutral"),
                Exponent("numerous", Language.ENGLISH, "MANY", 0.85, {"PronType": "Ind"}, "numerous", "formal"),
            ],
            "MOST": [
                Exponent("most", Language.ENGLISH, "MOST", 0.95, {"PronType": "Ind"}, "most", "neutral"),
                Exponent("majority", Language.ENGLISH, "MOST", 0.90, {"PronType": "Ind"}, "majority", "formal"),
            ],
            
            # Evaluators
            "GOOD": [
                Exponent("good", Language.ENGLISH, "GOOD", 0.95, {"Degree": "Pos"}, "good", "neutral"),
                Exponent("better", Language.ENGLISH, "GOOD", 0.90, {"Degree": "Cmp"}, "better", "neutral"),
                Exponent("best", Language.ENGLISH, "GOOD", 0.85, {"Degree": "Sup"}, "best", "neutral"),
                Exponent("excellent", Language.ENGLISH, "GOOD", 0.80, {"Degree": "Pos"}, "excellent", "formal"),
            ],
            "BAD": [
                Exponent("bad", Language.ENGLISH, "BAD", 0.95, {"Degree": "Pos"}, "bad", "neutral"),
                Exponent("worse", Language.ENGLISH, "BAD", 0.90, {"Degree": "Cmp"}, "worse", "neutral"),
                Exponent("worst", Language.ENGLISH, "BAD", 0.85, {"Degree": "Sup"}, "worst", "neutral"),
                Exponent("terrible", Language.ENGLISH, "BAD", 0.80, {"Degree": "Pos"}, "terrible", "neutral"),
            ],
            
            # Intensifiers
            "VERY": [
                Exponent("very", Language.ENGLISH, "VERY", 0.95, {"Degree": "Pos"}, "very", "neutral"),
                Exponent("extremely", Language.ENGLISH, "VERY", 0.90, {"Degree": "Pos"}, "extremely", "formal"),
                Exponent("really", Language.ENGLISH, "VERY", 0.85, {"Degree": "Pos"}, "really", "informal"),
            ],
            "MORE": [
                Exponent("more", Language.ENGLISH, "MORE", 0.95, {"Degree": "Cmp"}, "more", "neutral"),
                Exponent("less", Language.ENGLISH, "MORE", 0.90, {"Degree": "Cmp"}, "less", "neutral"),
            ],
            
            # Logical Operators
            "NOT": [
                Exponent("not", Language.ENGLISH, "NOT", 0.95, {"Polarity": "Neg"}, "not", "neutral"),
                Exponent("no", Language.ENGLISH, "NOT", 0.90, {"Polarity": "Neg"}, "no", "neutral"),
                Exponent("never", Language.ENGLISH, "NOT", 0.85, {"Polarity": "Neg"}, "never", "neutral"),
            ],
            "TRUE": [
                Exponent("true", Language.ENGLISH, "TRUE", 0.95, {"Degree": "Pos"}, "true", "neutral"),
                Exponent("correct", Language.ENGLISH, "TRUE", 0.90, {"Degree": "Pos"}, "correct", "neutral"),
                Exponent("right", Language.ENGLISH, "TRUE", 0.85, {"Degree": "Pos"}, "right", "neutral"),
            ],
            "FALSE": [
                Exponent("false", Language.ENGLISH, "FALSE", 0.95, {"Degree": "Pos"}, "false", "neutral"),
                Exponent("wrong", Language.ENGLISH, "FALSE", 0.90, {"Degree": "Pos"}, "wrong", "neutral"),
                Exponent("incorrect", Language.ENGLISH, "FALSE", 0.85, {"Degree": "Pos"}, "incorrect", "neutral"),
            ],
        }
    
    def _load_spanish_exponents(self) -> Dict[str, List[Exponent]]:
        """Load Spanish exponents for NSM primes."""
        return {
            # Substantives
            "PEOPLE": [
                Exponent("gente", Language.SPANISH, "PEOPLE", 0.95, {"Number": "Sing"}, "gente", "neutral"),
                Exponent("personas", Language.SPANISH, "PEOPLE", 0.90, {"Number": "Plur"}, "personas", "neutral"),
                Exponent("persona", Language.SPANISH, "PEOPLE", 0.85, {"Number": "Sing"}, "persona", "neutral"),
                Exponent("individuo", Language.SPANISH, "PEOPLE", 0.80, {"Number": "Sing"}, "individuo", "formal"),
            ],
            "THING": [
                Exponent("cosa", Language.SPANISH, "THING", 0.95, {"Number": "Sing"}, "cosa", "neutral"),
                Exponent("cosas", Language.SPANISH, "THING", 0.90, {"Number": "Plur"}, "cosas", "neutral"),
                Exponent("objeto", Language.SPANISH, "THING", 0.85, {"Number": "Sing"}, "objeto", "formal"),
                Exponent("elemento", Language.SPANISH, "THING", 0.80, {"Number": "Sing"}, "elemento", "formal"),
            ],
            "BODY": [
                Exponent("cuerpo", Language.SPANISH, "BODY", 0.95, {"Number": "Sing"}, "cuerpo", "neutral"),
                Exponent("cuerpos", Language.SPANISH, "BODY", 0.90, {"Number": "Plur"}, "cuerpos", "neutral"),
            ],
            
            # Mental Predicates
            "THINK": [
                Exponent("pensar", Language.SPANISH, "THINK", 0.95, {"VerbForm": "Inf"}, "pensar", "neutral"),
                Exponent("piensa", Language.SPANISH, "THINK", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "piensa", "neutral"),
                Exponent("pensó", Language.SPANISH, "THINK", 0.85, {"VerbForm": "Fin", "Tense": "Past"}, "pensó", "neutral"),
                Exponent("pensando", Language.SPANISH, "THINK", 0.80, {"VerbForm": "Part", "Tense": "Pres"}, "pensando", "neutral"),
            ],
            "KNOW": [
                Exponent("saber", Language.SPANISH, "KNOW", 0.95, {"VerbForm": "Inf"}, "saber", "neutral"),
                Exponent("sabe", Language.SPANISH, "KNOW", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "sabe", "neutral"),
                Exponent("supo", Language.SPANISH, "KNOW", 0.85, {"VerbForm": "Fin", "Tense": "Past"}, "supo", "neutral"),
                Exponent("sabido", Language.SPANISH, "KNOW", 0.80, {"VerbForm": "Part", "Tense": "Past"}, "sabido", "neutral"),
            ],
            "WANT": [
                Exponent("querer", Language.SPANISH, "WANT", 0.95, {"VerbForm": "Inf"}, "querer", "neutral"),
                Exponent("quiere", Language.SPANISH, "WANT", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "quiere", "neutral"),
                Exponent("quiso", Language.SPANISH, "WANT", 0.85, {"VerbForm": "Fin", "Tense": "Past"}, "quiso", "neutral"),
                Exponent("querido", Language.SPANISH, "WANT", 0.80, {"VerbForm": "Part", "Tense": "Past"}, "querido", "neutral"),
            ],
            "FEEL": [
                Exponent("sentir", Language.SPANISH, "FEEL", 0.95, {"VerbForm": "Inf"}, "sentir", "neutral"),
                Exponent("siente", Language.SPANISH, "FEEL", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "siente", "neutral"),
                Exponent("sintió", Language.SPANISH, "FEEL", 0.85, {"VerbForm": "Fin", "Tense": "Past"}, "sintió", "neutral"),
                Exponent("sentido", Language.SPANISH, "FEEL", 0.80, {"VerbForm": "Part", "Tense": "Past"}, "sentido", "neutral"),
            ],
            
            # Quantifiers
            "ALL": [
                Exponent("todo", Language.SPANISH, "ALL", 0.95, {"PronType": "Tot"}, "todo", "neutral"),
                Exponent("todos", Language.SPANISH, "ALL", 0.90, {"PronType": "Tot", "Number": "Plur"}, "todos", "neutral"),
                Exponent("cada", Language.SPANISH, "ALL", 0.85, {"PronType": "Tot"}, "cada", "neutral"),
            ],
            "SOME": [
                Exponent("alguno", Language.SPANISH, "SOME", 0.95, {"PronType": "Ind"}, "alguno", "neutral"),
                Exponent("algunos", Language.SPANISH, "SOME", 0.90, {"PronType": "Ind", "Number": "Plur"}, "algunos", "neutral"),
                Exponent("pocos", Language.SPANISH, "SOME", 0.85, {"PronType": "Ind", "Number": "Plur"}, "pocos", "neutral"),
            ],
            "MANY": [
                Exponent("mucho", Language.SPANISH, "MANY", 0.95, {"PronType": "Ind"}, "mucho", "neutral"),
                Exponent("muchos", Language.SPANISH, "MANY", 0.90, {"PronType": "Ind", "Number": "Plur"}, "muchos", "neutral"),
                Exponent("numerosos", Language.SPANISH, "MANY", 0.85, {"PronType": "Ind", "Number": "Plur"}, "numerosos", "formal"),
            ],
            "MOST": [
                Exponent("mayoría", Language.SPANISH, "MOST", 0.95, {"PronType": "Ind"}, "mayoría", "neutral"),
                Exponent("la mayoría", Language.SPANISH, "MOST", 0.90, {"PronType": "Ind"}, "la mayoría", "neutral"),
            ],
            
            # Evaluators
            "GOOD": [
                Exponent("bueno", Language.SPANISH, "GOOD", 0.95, {"Degree": "Pos"}, "bueno", "neutral"),
                Exponent("mejor", Language.SPANISH, "GOOD", 0.90, {"Degree": "Cmp"}, "mejor", "neutral"),
                Exponent("excelente", Language.SPANISH, "GOOD", 0.85, {"Degree": "Pos"}, "excelente", "formal"),
            ],
            "BAD": [
                Exponent("malo", Language.SPANISH, "BAD", 0.95, {"Degree": "Pos"}, "malo", "neutral"),
                Exponent("peor", Language.SPANISH, "BAD", 0.90, {"Degree": "Cmp"}, "peor", "neutral"),
                Exponent("terrible", Language.SPANISH, "BAD", 0.85, {"Degree": "Pos"}, "terrible", "neutral"),
            ],
            
            # Intensifiers
            "VERY": [
                Exponent("muy", Language.SPANISH, "VERY", 0.95, {"Degree": "Pos"}, "muy", "neutral"),
                Exponent("extremadamente", Language.SPANISH, "VERY", 0.90, {"Degree": "Pos"}, "extremadamente", "formal"),
                Exponent("realmente", Language.SPANISH, "VERY", 0.85, {"Degree": "Pos"}, "realmente", "neutral"),
            ],
            "MORE": [
                Exponent("más", Language.SPANISH, "MORE", 0.95, {"Degree": "Cmp"}, "más", "neutral"),
                Exponent("menos", Language.SPANISH, "MORE", 0.90, {"Degree": "Cmp"}, "menos", "neutral"),
            ],
            
            # Logical Operators
            "NOT": [
                Exponent("no", Language.SPANISH, "NOT", 0.95, {"Polarity": "Neg"}, "no", "neutral"),
                Exponent("nunca", Language.SPANISH, "NOT", 0.90, {"Polarity": "Neg"}, "nunca", "neutral"),
                Exponent("jamás", Language.SPANISH, "NOT", 0.85, {"Polarity": "Neg"}, "jamás", "neutral"),
            ],
            "TRUE": [
                Exponent("verdadero", Language.SPANISH, "TRUE", 0.95, {"Degree": "Pos"}, "verdadero", "neutral"),
                Exponent("correcto", Language.SPANISH, "TRUE", 0.90, {"Degree": "Pos"}, "correcto", "neutral"),
                Exponent("cierto", Language.SPANISH, "TRUE", 0.85, {"Degree": "Pos"}, "cierto", "neutral"),
            ],
            "FALSE": [
                Exponent("falso", Language.SPANISH, "FALSE", 0.95, {"Degree": "Pos"}, "falso", "neutral"),
                Exponent("incorrecto", Language.SPANISH, "FALSE", 0.90, {"Degree": "Pos"}, "incorrecto", "neutral"),
                Exponent("erróneo", Language.SPANISH, "FALSE", 0.85, {"Degree": "Pos"}, "erróneo", "neutral"),
            ],
        }
    
    def _load_french_exponents(self) -> Dict[str, List[Exponent]]:
        """Load French exponents for NSM primes."""
        return {
            # Substantives
            "PEOPLE": [
                Exponent("gens", Language.FRENCH, "PEOPLE", 0.95, {"Number": "Plur"}, "gens", "neutral"),
                Exponent("personnes", Language.FRENCH, "PEOPLE", 0.90, {"Number": "Plur"}, "personnes", "neutral"),
                Exponent("personne", Language.FRENCH, "PEOPLE", 0.85, {"Number": "Sing"}, "personne", "neutral"),
                Exponent("individu", Language.FRENCH, "PEOPLE", 0.80, {"Number": "Sing"}, "individu", "formal"),
            ],
            "THING": [
                Exponent("chose", Language.FRENCH, "THING", 0.95, {"Number": "Sing"}, "chose", "neutral"),
                Exponent("choses", Language.FRENCH, "THING", 0.90, {"Number": "Plur"}, "choses", "neutral"),
                Exponent("objet", Language.FRENCH, "THING", 0.85, {"Number": "Sing"}, "objet", "formal"),
                Exponent("élément", Language.FRENCH, "THING", 0.80, {"Number": "Sing"}, "élément", "formal"),
            ],
            "BODY": [
                Exponent("corps", Language.FRENCH, "BODY", 0.95, {"Number": "Sing"}, "corps", "neutral"),
                Exponent("corps", Language.FRENCH, "BODY", 0.90, {"Number": "Plur"}, "corps", "neutral"),
            ],
            
            # Mental Predicates
            "THINK": [
                Exponent("penser", Language.FRENCH, "THINK", 0.95, {"VerbForm": "Inf"}, "penser", "neutral"),
                Exponent("pense", Language.FRENCH, "THINK", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "pense", "neutral"),
                Exponent("pensait", Language.FRENCH, "THINK", 0.85, {"VerbForm": "Fin", "Tense": "Imp"}, "pensait", "neutral"),
                Exponent("pensant", Language.FRENCH, "THINK", 0.80, {"VerbForm": "Part", "Tense": "Pres"}, "pensant", "neutral"),
            ],
            "KNOW": [
                Exponent("savoir", Language.FRENCH, "KNOW", 0.95, {"VerbForm": "Inf"}, "savoir", "neutral"),
                Exponent("sait", Language.FRENCH, "KNOW", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "sait", "neutral"),
                Exponent("savait", Language.FRENCH, "KNOW", 0.85, {"VerbForm": "Fin", "Tense": "Imp"}, "savait", "neutral"),
                Exponent("su", Language.FRENCH, "KNOW", 0.80, {"VerbForm": "Part", "Tense": "Past"}, "su", "neutral"),
            ],
            "WANT": [
                Exponent("vouloir", Language.FRENCH, "WANT", 0.95, {"VerbForm": "Inf"}, "vouloir", "neutral"),
                Exponent("veut", Language.FRENCH, "WANT", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "veut", "neutral"),
                Exponent("voulait", Language.FRENCH, "WANT", 0.85, {"VerbForm": "Fin", "Tense": "Imp"}, "voulait", "neutral"),
                Exponent("voulu", Language.FRENCH, "WANT", 0.80, {"VerbForm": "Part", "Tense": "Past"}, "voulu", "neutral"),
            ],
            "FEEL": [
                Exponent("sentir", Language.FRENCH, "FEEL", 0.95, {"VerbForm": "Inf"}, "sentir", "neutral"),
                Exponent("sent", Language.FRENCH, "FEEL", 0.90, {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}, "sent", "neutral"),
                Exponent("sentait", Language.FRENCH, "FEEL", 0.85, {"VerbForm": "Fin", "Tense": "Imp"}, "sentait", "neutral"),
                Exponent("senti", Language.FRENCH, "FEEL", 0.80, {"VerbForm": "Part", "Tense": "Past"}, "senti", "neutral"),
            ],
            
            # Quantifiers
            "ALL": [
                Exponent("tout", Language.FRENCH, "ALL", 0.95, {"PronType": "Tot"}, "tout", "neutral"),
                Exponent("tous", Language.FRENCH, "ALL", 0.90, {"PronType": "Tot", "Number": "Plur"}, "tous", "neutral"),
                Exponent("chaque", Language.FRENCH, "ALL", 0.85, {"PronType": "Tot"}, "chaque", "neutral"),
            ],
            "SOME": [
                Exponent("quelque", Language.FRENCH, "SOME", 0.95, {"PronType": "Ind"}, "quelque", "neutral"),
                Exponent("quelques", Language.FRENCH, "SOME", 0.90, {"PronType": "Ind", "Number": "Plur"}, "quelques", "neutral"),
                Exponent("peu", Language.FRENCH, "SOME", 0.85, {"PronType": "Ind"}, "peu", "neutral"),
            ],
            "MANY": [
                Exponent("beaucoup", Language.FRENCH, "MANY", 0.95, {"PronType": "Ind"}, "beaucoup", "neutral"),
                Exponent("nombreux", Language.FRENCH, "MANY", 0.90, {"PronType": "Ind", "Number": "Plur"}, "nombreux", "formal"),
                Exponent("plusieurs", Language.FRENCH, "MANY", 0.85, {"PronType": "Ind", "Number": "Plur"}, "plusieurs", "neutral"),
            ],
            "MOST": [
                Exponent("majorité", Language.FRENCH, "MOST", 0.95, {"PronType": "Ind"}, "majorité", "neutral"),
                Exponent("la majorité", Language.FRENCH, "MOST", 0.90, {"PronType": "Ind"}, "la majorité", "neutral"),
            ],
            
            # Evaluators
            "GOOD": [
                Exponent("bon", Language.FRENCH, "GOOD", 0.95, {"Degree": "Pos"}, "bon", "neutral"),
                Exponent("meilleur", Language.FRENCH, "GOOD", 0.90, {"Degree": "Cmp"}, "meilleur", "neutral"),
                Exponent("excellent", Language.FRENCH, "GOOD", 0.85, {"Degree": "Pos"}, "excellent", "formal"),
            ],
            "BAD": [
                Exponent("mauvais", Language.FRENCH, "BAD", 0.95, {"Degree": "Pos"}, "mauvais", "neutral"),
                Exponent("pire", Language.FRENCH, "BAD", 0.90, {"Degree": "Cmp"}, "pire", "neutral"),
                Exponent("terrible", Language.FRENCH, "BAD", 0.85, {"Degree": "Pos"}, "terrible", "neutral"),
            ],
            
            # Intensifiers
            "VERY": [
                Exponent("très", Language.FRENCH, "VERY", 0.95, {"Degree": "Pos"}, "très", "neutral"),
                Exponent("extrêmement", Language.FRENCH, "VERY", 0.90, {"Degree": "Pos"}, "extrêmement", "formal"),
                Exponent("vraiment", Language.FRENCH, "VERY", 0.85, {"Degree": "Pos"}, "vraiment", "neutral"),
            ],
            "MORE": [
                Exponent("plus", Language.FRENCH, "MORE", 0.95, {"Degree": "Cmp"}, "plus", "neutral"),
                Exponent("moins", Language.FRENCH, "MORE", 0.90, {"Degree": "Cmp"}, "moins", "neutral"),
            ],
            
            # Logical Operators
            "NOT": [
                Exponent("ne", Language.FRENCH, "NOT", 0.95, {"Polarity": "Neg"}, "ne", "neutral"),
                Exponent("pas", Language.FRENCH, "NOT", 0.90, {"Polarity": "Neg"}, "pas", "neutral"),
                Exponent("jamais", Language.FRENCH, "NOT", 0.85, {"Polarity": "Neg"}, "jamais", "neutral"),
            ],
            "TRUE": [
                Exponent("vrai", Language.FRENCH, "TRUE", 0.95, {"Degree": "Pos"}, "vrai", "neutral"),
                Exponent("correct", Language.FRENCH, "TRUE", 0.90, {"Degree": "Pos"}, "correct", "neutral"),
                Exponent("juste", Language.FRENCH, "TRUE", 0.85, {"Degree": "Pos"}, "juste", "neutral"),
            ],
            "FALSE": [
                Exponent("faux", Language.FRENCH, "FALSE", 0.95, {"Degree": "Pos"}, "faux", "neutral"),
                Exponent("incorrect", Language.FRENCH, "FALSE", 0.90, {"Degree": "Pos"}, "incorrect", "neutral"),
                Exponent("erroné", Language.FRENCH, "FALSE", 0.85, {"Degree": "Pos"}, "erroné", "formal"),
            ],
        }
    
    def _create_reverse_mappings(self):
        """Create reverse mappings for efficient lookup."""
        self.surface_to_prime = {}
        self.language_exponents = {
            Language.ENGLISH: {},
            Language.SPANISH: {},
            Language.FRENCH: {}
        }
        
        # Process English exponents
        for prime, exponents in self.english_exponents.items():
            for exp in exponents:
                self.surface_to_prime[exp.surface_form.lower()] = prime
                if prime not in self.language_exponents[Language.ENGLISH]:
                    self.language_exponents[Language.ENGLISH][prime] = []
                self.language_exponents[Language.ENGLISH][prime].append(exp)
        
        # Process Spanish exponents
        for prime, exponents in self.spanish_exponents.items():
            for exp in exponents:
                self.surface_to_prime[exp.surface_form.lower()] = prime
                if prime not in self.language_exponents[Language.SPANISH]:
                    self.language_exponents[Language.SPANISH][prime] = []
                self.language_exponents[Language.SPANISH][prime].append(exp)
        
        # Process French exponents
        for prime, exponents in self.french_exponents.items():
            for exp in exponents:
                self.surface_to_prime[exp.surface_form.lower()] = prime
                if prime not in self.language_exponents[Language.FRENCH]:
                    self.language_exponents[Language.FRENCH][prime] = []
                self.language_exponents[Language.FRENCH][prime].append(exp)
    
    def get_exponents_for_prime(self, prime: str, language: Language) -> List[Exponent]:
        """Get all exponents for a given prime in a specific language.
        
        Args:
            prime: NSM prime name
            language: Target language
            
        Returns:
            List of exponents for the prime
        """
        return self.language_exponents.get(language, {}).get(prime, [])
    
    def get_prime_for_surface(self, surface_form: str) -> Optional[str]:
        """Get the NSM prime for a given surface form.
        
        Args:
            surface_form: Surface form to look up
            
        Returns:
            NSM prime name or None if not found
        """
        return self.surface_to_prime.get(surface_form.lower())
    
    def get_best_exponent(self, prime: str, language: Language, 
                         ud_features: Optional[Dict[str, str]] = None,
                         register: str = "neutral") -> Optional[Exponent]:
        """Get the best exponent for a prime given UD features and register.
        
        Args:
            prime: NSM prime name
            language: Target language
            ud_features: UD morphological features
            register: Formality register (formal, informal, neutral)
            
        Returns:
            Best matching exponent or None
        """
        exponents = self.get_exponents_for_prime(prime, language)
        if not exponents:
            return None
        
        # Filter by register first
        register_exponents = [exp for exp in exponents if exp.register == register]
        if not register_exponents:
            register_exponents = [exp for exp in exponents if exp.register == "neutral"]
        
        if not ud_features:
            # Return highest confidence exponent
            return max(register_exponents, key=lambda x: x.confidence)
        
        # Score exponents based on UD feature match
        best_exponent = None
        best_score = -1
        
        for exp in register_exponents:
            score = 0
            for feature, value in ud_features.items():
                if feature in exp.ud_features and exp.ud_features[feature] == value:
                    score += 1
            
            # Add confidence bonus
            score += exp.confidence
            
            if score > best_score:
                best_score = score
                best_exponent = exp
        
        return best_exponent
    
    def get_coverage_stats(self, language: Language) -> Dict[str, int]:
        """Get coverage statistics for a language.
        
        Args:
            language: Target language
            
        Returns:
            Coverage statistics
        """
        exponents = self.language_exponents.get(language, {})
        return {
            "total_primes": len(exponents),
            "total_exponents": sum(len(exps) for exps in exponents.values()),
            "avg_exponents_per_prime": sum(len(exps) for exps in exponents.values()) / len(exponents) if exponents else 0
        }

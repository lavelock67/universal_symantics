#!/usr/bin/env python3
"""
Enhanced NSM Explicator with Improved Legality and Substitutability.

This module provides enhanced NSM explication generation with better
legality validation and improved templates for better substitutability.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from .enhanced_legality import EnhancedNSMLegalityValidator, NSMLegalityResult

class EnhancedNSMExplicator:
    """Enhanced NSM explicator with improved legality and substitutability."""
    
    def __init__(self, exponents_path: str = "data/nsm_exponents_en_es_fr.json"):
        """Initialize the enhanced explicator."""
        self.legality_validator = EnhancedNSMLegalityValidator(exponents_path)
        self.exponents = self.legality_validator.exponents
        self.enhanced_templates = self._build_enhanced_templates()
        
    def _build_enhanced_templates(self) -> Dict[str, Dict[str, str]]:
        """Build enhanced templates for better substitutability."""
        return {
            "AtLocation": {
                "en": "this thing is in this place",
                "es": "esta cosa está en este lugar", 
                "fr": "cette chose est dans ce lieu"
            },
            "HasProperty": {
                "en": "this thing is like this",
                "es": "esta cosa es así",
                "fr": "cette chose est comme cela"
            },
            "UsedFor": {
                "en": "someone can do something with this thing",
                "es": "alguien puede hacer algo con esta cosa",
                "fr": "quelqu'un peut faire quelque chose avec cette chose"
            },
            "SimilarTo": {
                "en": "this thing is like another thing",
                "es": "esta cosa es como otra cosa",
                "fr": "cette chose est comme une autre chose"
            },
            "DifferentFrom": {
                "en": "this thing is not like another thing",
                "es": "esta cosa no es como otra cosa",
                "fr": "cette chose n'est pas comme une autre chose"
            },
            "PartOf": {
                "en": "one part is part of another thing",
                "es": "una parte es parte de otra cosa",
                "fr": "une partie est partie d'une autre chose"
            },
            "Causes": {
                "en": "something happens because something else happens",
                "es": "algo pasa porque otra cosa pasa",
                "fr": "quelque chose arrive parce qu'une autre chose arrive"
            },
            "Not": {
                "en": "someone does not do something",
                "es": "alguien no hace algo",
                "fr": "quelqu'un ne fait pas quelque chose"
            },
            "Exist": {
                "en": "there is something",
                "es": "hay algo",
                "fr": "il y a quelque chose"
            },
            "Can": {
                "en": "someone can do something",
                "es": "alguien puede hacer algo",
                "fr": "quelqu'un peut faire quelque chose"
            },
            "Want": {
                "en": "someone wants something",
                "es": "alguien quiere algo",
                "fr": "quelqu'un veut quelque chose"
            },
            "Think": {
                "en": "someone thinks something",
                "es": "alguien piensa algo",
                "fr": "quelqu'un pense quelque chose"
            },
            "Feel": {
                "en": "someone feels something",
                "es": "alguien siente algo",
                "fr": "quelqu'un ressent quelque chose"
            },
            "See": {
                "en": "someone sees something",
                "es": "alguien ve algo",
                "fr": "quelqu'un voit quelque chose"
            },
            "Say": {
                "en": "someone says something",
                "es": "alguien dice algo",
                "fr": "quelqu'un dit quelque chose"
            },
            "Do": {
                "en": "someone does something",
                "es": "alguien hace algo",
                "fr": "quelqu'un fait quelque chose"
            },
            "Happen": {
                "en": "something happens",
                "es": "algo pasa",
                "fr": "quelque chose arrive"
            },
            "Good": {
                "en": "this thing is good",
                "es": "esta cosa es buena",
                "fr": "cette chose est bonne"
            },
            "Bad": {
                "en": "this thing is bad",
                "es": "esta cosa es mala",
                "fr": "cette chose est mauvaise"
            },
            "Big": {
                "en": "this thing is big",
                "es": "esta cosa es grande",
                "fr": "cette chose est grande"
            },
            "Small": {
                "en": "this thing is small",
                "es": "esta cosa es pequeña",
                "fr": "cette chose est petite"
            },
            "Same": {
                "en": "this thing is the same as another thing",
                "es": "esta cosa es la misma que otra cosa",
                "fr": "cette chose est la même qu'une autre chose"
            },
            "Other": {
                "en": "this thing is other than another thing",
                "es": "esta cosa es otra que otra cosa",
                "fr": "cette chose est autre qu'une autre chose"
            },
            "All": {
                "en": "all things are like this",
                "es": "todas las cosas son así",
                "fr": "toutes les choses sont comme cela"
            },
            "Some": {
                "en": "some things are like this",
                "es": "algunas cosas son así",
                "fr": "quelques choses sont comme cela"
            },
            "Many": {
                "en": "many things are like this",
                "es": "muchas cosas son así",
                "fr": "beaucoup de choses sont comme cela"
            },
            "Few": {
                "en": "few things are like this",
                "es": "pocas cosas son así",
                "fr": "peu de choses sont comme cela"
            },
            "Before": {
                "en": "something happens before something else",
                "es": "algo pasa antes que otra cosa",
                "fr": "quelque chose arrive avant autre chose"
            },
            "After": {
                "en": "something happens after something else",
                "es": "algo pasa después que otra cosa",
                "fr": "quelque chose arrive après autre chose"
            },
            "Now": {
                "en": "something happens now",
                "es": "algo pasa ahora",
                "fr": "quelque chose arrive maintenant"
            },
            "Here": {
                "en": "something is here",
                "es": "algo está aquí",
                "fr": "quelque chose est ici"
            },
            "Where": {
                "en": "something is in this place",
                "es": "algo está en este lugar",
                "fr": "quelque chose est dans ce lieu"
            },
            "Very": {
                "en": "this thing is very like this",
                "es": "esta cosa es muy así",
                "fr": "cette chose est très comme cela"
            },
            "More": {
                "en": "this thing is more like this than another thing",
                "es": "esta cosa es más así que otra cosa",
                "fr": "cette chose est plus comme cela qu'une autre chose"
            },
            "If": {
                "en": "if something happens then something else happens",
                "es": "si algo pasa entonces otra cosa pasa",
                "fr": "si quelque chose arrive alors autre chose arrive"
            },
            "Because": {
                "en": "something happens because something else happens",
                "es": "algo pasa porque otra cosa pasa",
                "fr": "quelque chose arrive parce qu'autre chose arrive"
            }
        }
    
    def template_for_primitive(self, primitive: str, lang: str = "en") -> str:
        """Generate enhanced template for a primitive."""
        # Try enhanced templates first
        if primitive in self.enhanced_templates:
            return self.enhanced_templates[primitive].get(lang, self.enhanced_templates[primitive]["en"])
        
        # Fallback to basic templates
        basic_templates = {
            "AtLocation": {
                "en": "something is in a place",
                "es": "algo está en un lugar",
                "fr": "quelque chose est dans un lieu"
            },
            "HasProperty": {
                "en": "this thing is like this",
                "es": "esta cosa es así",
                "fr": "cette chose est comme cela"
            },
            "UsedFor": {
                "en": "someone can do something with this thing",
                "es": "alguien puede hacer algo con esta cosa",
                "fr": "quelqu'un peut faire quelque chose avec cette chose"
            },
            "SimilarTo": {
                "en": "this thing is like another thing",
                "es": "esta cosa es como otra cosa",
                "fr": "cette chose est comme une autre chose"
            },
            "DifferentFrom": {
                "en": "this thing is not like another thing",
                "es": "esta cosa no es como otra cosa",
                "fr": "cette chose n'est pas comme une autre chose"
            },
            "PartOf": {
                "en": "one part is part of another thing",
                "es": "una parte es parte de otra cosa",
                "fr": "une partie est partie d'une autre chose"
            },
            "Causes": {
                "en": "something happens because something else happens",
                "es": "algo pasa porque otra cosa pasa",
                "fr": "quelque chose arrive parce qu'une autre chose arrive"
            },
            "Not": {
                "en": "someone does not do something",
                "es": "alguien no hace algo",
                "fr": "quelqu'un ne fait pas quelque chose"
            },
            "Exist": {
                "en": "there is something",
                "es": "hay algo",
                "fr": "il y a quelque chose"
            }
        }
        
        if primitive in basic_templates:
            return basic_templates[primitive].get(lang, basic_templates[primitive]["en"])
        
        # Default template
        return {
            "en": "something is like this",
            "es": "algo es así",
            "fr": "quelque chose est comme cela"
        }.get(lang, "something is like this")
    
    def validate_legality(self, text: str, lang: str) -> bool:
        """Validate NSM legality using enhanced validator."""
        return self.legality_validator.is_legal(text, lang)
    
    def legality_score(self, text: str, lang: str) -> float:
        """Get NSM legality score using enhanced validator."""
        return self.legality_validator.legality_score(text, lang)
    
    def detailed_legality_analysis(self, text: str, lang: str) -> NSMLegalityResult:
        """Get detailed legality analysis."""
        return self.legality_validator.validate_legality(text, lang)
    
    def detect_primes(self, text: str, lang: str) -> List[str]:
        """Detect NSM primes in text."""
        result = self.legality_validator.validate_legality(text, lang)
        return result.detected_primes
    
    def generate_explication(self, text: str, lang: str, primitive: Optional[str] = None) -> str:
        """Generate NSM explication for text."""
        if primitive:
            return self.template_for_primitive(primitive, lang)
        
        # Try to detect the most likely primitive
        detected_primes = self.detect_primes(text, lang)
        
        # Map common words to primitives
        word_to_primitive = {
            "en": {
                "location": "AtLocation",
                "place": "AtLocation",
                "in": "AtLocation",
                "on": "AtLocation",
                "at": "AtLocation",
                "property": "HasProperty",
                "like": "HasProperty",
                "similar": "SimilarTo",
                "different": "DifferentFrom",
                "part": "PartOf",
                "cause": "Causes",
                "because": "Causes",
                "not": "Not",
                "exist": "Exist",
                "there": "Exist",
                "can": "Can",
                "want": "Want",
                "think": "Think",
                "feel": "Feel",
                "see": "See",
                "say": "Say",
                "do": "Do",
                "happen": "Happen",
                "good": "Good",
                "bad": "Bad",
                "big": "Big",
                "small": "Small",
                "same": "Same",
                "other": "Other",
                "all": "All",
                "some": "Some",
                "many": "Many",
                "few": "Few",
                "before": "Before",
                "after": "After",
                "now": "Now",
                "here": "Here",
                "where": "Where",
                "very": "Very",
                "more": "More",
                "if": "If"
            },
            "es": {
                "lugar": "AtLocation",
                "en": "AtLocation",
                "sobre": "AtLocation",
                "a": "AtLocation",
                "propiedad": "HasProperty",
                "como": "HasProperty",
                "similar": "SimilarTo",
                "diferente": "DifferentFrom",
                "parte": "PartOf",
                "causa": "Causes",
                "porque": "Causes",
                "no": "Not",
                "existe": "Exist",
                "hay": "Exist",
                "puede": "Can",
                "quiere": "Want",
                "piensa": "Think",
                "siente": "Feel",
                "ve": "See",
                "dice": "Say",
                "hace": "Do",
                "pasa": "Happen",
                "bueno": "Good",
                "malo": "Bad",
                "grande": "Big",
                "pequeño": "Small",
                "mismo": "Same",
                "otro": "Other",
                "todos": "All",
                "algunos": "Some",
                "muchos": "Many",
                "pocos": "Few",
                "antes": "Before",
                "después": "After",
                "ahora": "Now",
                "aquí": "Here",
                "donde": "Where",
                "muy": "Very",
                "más": "More",
                "si": "If"
            },
            "fr": {
                "lieu": "AtLocation",
                "dans": "AtLocation",
                "sur": "AtLocation",
                "à": "AtLocation",
                "propriété": "HasProperty",
                "comme": "HasProperty",
                "semblable": "SimilarTo",
                "différent": "DifferentFrom",
                "partie": "PartOf",
                "cause": "Causes",
                "parce": "Causes",
                "ne": "Not",
                "pas": "Not",
                "existe": "Exist",
                "il": "Exist",
                "peut": "Can",
                "veut": "Want",
                "pense": "Think",
                "ressent": "Feel",
                "voit": "See",
                "dit": "Say",
                "fait": "Do",
                "arrive": "Happen",
                "bon": "Good",
                "mauvais": "Bad",
                "grand": "Big",
                "petit": "Small",
                "même": "Same",
                "autre": "Other",
                "tous": "All",
                "quelques": "Some",
                "beaucoup": "Many",
                "peu": "Few",
                "avant": "Before",
                "après": "After",
                "maintenant": "Now",
                "ici": "Here",
                "où": "Where",
                "très": "Very",
                "plus": "More",
                "si": "If"
            }
        }
        
        # Find the most likely primitive based on content
        words = text.lower().split()
        lang_mapping = word_to_primitive.get(lang, word_to_primitive["en"])
        
        for word in words:
            if word in lang_mapping:
                return self.template_for_primitive(lang_mapping[word], lang)
        
        # Default to HasProperty if no specific primitive detected
        return self.template_for_primitive("HasProperty", lang)
    
    def evaluate_substitutability(self, original: str, explication: str, lang: str) -> float:
        """Evaluate substitutability between original text and explication."""
        # Simple heuristic based on word overlap and legality
        original_words = set(original.lower().split())
        explication_words = set(explication.lower().split())
        
        # Calculate word overlap
        overlap = len(original_words.intersection(explication_words))
        total_unique = len(original_words.union(explication_words))
        
        if total_unique == 0:
            return 0.0
        
        overlap_score = overlap / total_unique
        
        # Get legality scores
        original_legality = self.legality_score(original, lang)
        explication_legality = self.legality_score(explication, lang)
        
        # Combine scores
        legality_score = (original_legality + explication_legality) / 2
        
        # Weighted combination
        substitutability = 0.4 * overlap_score + 0.6 * legality_score
        
        return substitutability
    
    def generate_improved_template(self, primitive: str, lang: str, context: Optional[str] = None) -> str:
        """Generate improved template with context awareness."""
        base_template = self.template_for_primitive(primitive, lang)
        
        if not context:
            return base_template
        
        # Try to improve template based on context
        context_words = context.lower().split()
        
        # Add context-specific improvements
        if "location" in context_words or "place" in context_words:
            if primitive == "AtLocation":
                return {
                    "en": "this thing is in this specific place",
                    "es": "esta cosa está en este lugar específico",
                    "fr": "cette chose est dans ce lieu spécifique"
                }.get(lang, base_template)
        
        if "property" in context_words or "characteristic" in context_words:
            if primitive == "HasProperty":
                return {
                    "en": "this thing has this specific property",
                    "es": "esta cosa tiene esta propiedad específica",
                    "fr": "cette chose a cette propriété spécifique"
                }.get(lang, base_template)
        
        return base_template


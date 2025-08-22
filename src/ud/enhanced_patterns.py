#!/usr/bin/env python3
"""
Enhanced UD Patterns for Advanced Primitive Detection.

This module provides comprehensive UD dependency patterns for detecting
primitives across EN/ES/FR languages with advanced pattern matching.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import re
from dataclasses import dataclass

@dataclass
class UDPattern:
    """Represents a UD dependency pattern for primitive detection."""
    name: str
    primitive: str
    patterns: List[Dict[str, Any]]
    languages: List[str]
    confidence: float
    description: str

class EnhancedUDPatterns:
    """Enhanced UD patterns for comprehensive primitive detection."""
    
    def __init__(self):
        """Initialize enhanced UD patterns."""
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, UDPattern]:
        """Initialize comprehensive UD patterns."""
        patterns = {}
        
        # Location patterns
        patterns["AtLocation"] = UDPattern(
            name="AtLocation",
            primitive="AtLocation",
            patterns=[
                # EN: "X is in/at/on Y"
                {"lang": "en", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "be"},
                    {"DEP": "case", "LOWER": {"IN": ["in", "at", "on", "inside", "outside"]}},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]},
                # ES: "X está en/sobre/a Y"
                {"lang": "es", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "estar"},
                    {"DEP": "case", "LOWER": {"IN": ["en", "sobre", "a", "dentro", "fuera"]}},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]},
                # FR: "X est à/dans/sur Y"
                {"lang": "fr", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "être"},
                    {"DEP": "case", "LOWER": {"IN": ["à", "dans", "sur", "dedans", "dehors"]}},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]}
            ],
            languages=["en", "es", "fr"],
            confidence=0.9,
            description="Entity located at/in/on another entity"
        )
        
        # Property patterns
        patterns["HasProperty"] = UDPattern(
            name="HasProperty",
            primitive="HasProperty",
            patterns=[
                # EN: "X is ADJ"
                {"lang": "en", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "be"},
                    {"DEP": "acomp", "POS": "ADJ"}
                ]},
                # ES: "X es ADJ"
                {"lang": "es", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "ser"},
                    {"DEP": "acomp", "POS": "ADJ"}
                ]},
                # FR: "X est ADJ"
                {"lang": "fr", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "être"},
                    {"DEP": "acomp", "POS": "ADJ"}
                ]}
            ],
            languages=["en", "es", "fr"],
            confidence=0.95,
            description="Entity has a specific property"
        )
        
        # Part-whole patterns
        patterns["PartOf"] = UDPattern(
            name="PartOf",
            primitive="PartOf",
            patterns=[
                # EN: "X is part of Y"
                {"lang": "en", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "be"},
                    {"DEP": "attr", "LOWER": "part"},
                    {"DEP": "case", "LOWER": "of"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]},
                # ES: "X es parte de Y"
                {"lang": "es", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "ser"},
                    {"DEP": "attr", "LOWER": "parte"},
                    {"DEP": "case", "LOWER": "de"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]},
                # FR: "X est partie de Y"
                {"lang": "fr", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "être"},
                    {"DEP": "attr", "LOWER": "partie"},
                    {"DEP": "case", "LOWER": "de"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]}
            ],
            languages=["en", "es", "fr"],
            confidence=0.9,
            description="Entity is part of another entity"
        )
        
        # Causation patterns
        patterns["Causes"] = UDPattern(
            name="Causes",
            primitive="Causes",
            patterns=[
                # EN: "X causes Y"
                {"lang": "en", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "ROOT", "LEMMA": "cause"},
                    {"DEP": "dobj", "POS": "NOUN"}
                ]},
                # ES: "X causa Y"
                {"lang": "es", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "ROOT", "LEMMA": "causar"},
                    {"DEP": "dobj", "POS": "NOUN"}
                ]},
                # FR: "X cause Y"
                {"lang": "fr", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "ROOT", "LEMMA": "causer"},
                    {"DEP": "dobj", "POS": "NOUN"}
                ]}
            ],
            languages=["en", "es", "fr"],
            confidence=0.85,
            description="Entity causes another entity or event"
        )
        
        # Purpose patterns
        patterns["UsedFor"] = UDPattern(
            name="UsedFor",
            primitive="UsedFor",
            patterns=[
                # EN: "X is used for Y"
                {"lang": "en", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "auxpass", "LEMMA": "be"},
                    {"DEP": "ROOT", "LEMMA": "use"},
                    {"DEP": "prep", "LOWER": "for"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]},
                # ES: "X se usa para Y"
                {"lang": "es", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "auxpass", "LEMMA": "usar"},
                    {"DEP": "prep", "LOWER": "para"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]},
                # FR: "X est utilisé pour Y"
                {"lang": "fr", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "auxpass", "LEMMA": "utiliser"},
                    {"DEP": "prep", "LOWER": "pour"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]}
            ],
            languages=["en", "es", "fr"],
            confidence=0.9,
            description="Entity is used for a specific purpose"
        )
        
        # Similarity patterns
        patterns["SimilarTo"] = UDPattern(
            name="SimilarTo",
            primitive="SimilarTo",
            patterns=[
                # EN: "X is similar to Y"
                {"lang": "en", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "be"},
                    {"DEP": "acomp", "LOWER": "similar"},
                    {"DEP": "prep", "LOWER": "to"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]},
                # ES: "X es similar a Y"
                {"lang": "es", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "ser"},
                    {"DEP": "acomp", "LOWER": "similar"},
                    {"DEP": "prep", "LOWER": "a"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]},
                # FR: "X est similaire à Y"
                {"lang": "fr", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "être"},
                    {"DEP": "acomp", "LOWER": "similaire"},
                    {"DEP": "prep", "LOWER": "à"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]}
            ],
            languages=["en", "es", "fr"],
            confidence=0.85,
            description="Entity is similar to another entity"
        )
        
        # Difference patterns
        patterns["DifferentFrom"] = UDPattern(
            name="DifferentFrom",
            primitive="DifferentFrom",
            patterns=[
                # EN: "X is different from Y"
                {"lang": "en", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "be"},
                    {"DEP": "acomp", "LOWER": "different"},
                    {"DEP": "prep", "LOWER": "from"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]},
                # ES: "X es diferente de Y"
                {"lang": "es", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "ser"},
                    {"DEP": "acomp", "LOWER": "diferente"},
                    {"DEP": "prep", "LOWER": "de"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]},
                # FR: "X est différent de Y"
                {"lang": "fr", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "être"},
                    {"DEP": "acomp", "LOWER": "différent"},
                    {"DEP": "prep", "LOWER": "de"},
                    {"DEP": "pobj", "POS": "NOUN"}
                ]}
            ],
            languages=["en", "es", "fr"],
            confidence=0.85,
            description="Entity is different from another entity"
        )
        
        # Existence patterns
        patterns["Exist"] = UDPattern(
            name="Exist",
            primitive="Exist",
            patterns=[
                # EN: "There is X"
                {"lang": "en", "pattern": [
                    {"DEP": "expl", "LOWER": "there"},
                    {"DEP": "ROOT", "LEMMA": "be"},
                    {"DEP": "nsubj", "POS": "NOUN"}
                ]},
                # ES: "Hay X"
                {"lang": "es", "pattern": [
                    {"DEP": "ROOT", "LOWER": "hay"},
                    {"DEP": "dobj", "POS": "NOUN"}
                ]},
                # FR: "Il y a X"
                {"lang": "fr", "pattern": [
                    {"DEP": "expl", "LOWER": "il"},
                    {"DEP": "ROOT", "LOWER": "y"},
                    {"DEP": "aux", "LOWER": "a"},
                    {"DEP": "dobj", "POS": "NOUN"}
                ]}
            ],
            languages=["en", "es", "fr"],
            confidence=0.9,
            description="Entity exists"
        )
        
        # Negation patterns
        patterns["Not"] = UDPattern(
            name="Not",
            primitive="Not",
            patterns=[
                # EN: "X is not Y"
                {"lang": "en", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "cop", "LEMMA": "be"},
                    {"DEP": "neg", "LOWER": "not"},
                    {"DEP": "acomp", "POS": "ADJ"}
                ]},
                # ES: "X no es Y"
                {"lang": "es", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "neg", "LOWER": "no"},
                    {"DEP": "cop", "LEMMA": "ser"},
                    {"DEP": "acomp", "POS": "ADJ"}
                ]},
                # FR: "X n'est pas Y"
                {"lang": "fr", "pattern": [
                    {"DEP": "nsubj", "POS": "NOUN"},
                    {"DEP": "neg", "LOWER": "ne"},
                    {"DEP": "cop", "LEMMA": "être"},
                    {"DEP": "neg", "LOWER": "pas"},
                    {"DEP": "acomp", "POS": "ADJ"}
                ]}
            ],
            languages=["en", "es", "fr"],
            confidence=0.95,
            description="Negation of a property or state"
        )
        
        return patterns
    
    def get_patterns_for_language(self, language: str) -> List[UDPattern]:
        """Get all patterns for a specific language."""
        return [pattern for pattern in self.patterns.values() 
                if language in pattern.languages]
    
    def get_patterns_for_primitive(self, primitive: str) -> List[UDPattern]:
        """Get all patterns for a specific primitive."""
        return [pattern for pattern in self.patterns.values() 
                if pattern.primitive == primitive]
    
    def get_all_patterns(self) -> Dict[str, UDPattern]:
        """Get all patterns."""
        return self.patterns
    
    def add_custom_pattern(self, pattern: UDPattern):
        """Add a custom pattern."""
        self.patterns[pattern.name] = pattern
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about the patterns."""
        total_patterns = len(self.patterns)
        languages = set()
        primitives = set()
        
        for pattern in self.patterns.values():
            languages.update(pattern.languages)
            primitives.add(pattern.primitive)
        
        return {
            "total_patterns": total_patterns,
            "languages": list(languages),
            "primitives": list(primitives),
            "patterns_per_language": {
                lang: len(self.get_patterns_for_language(lang))
                for lang in languages
            },
            "patterns_per_primitive": {
                prim: len(self.get_patterns_for_primitive(prim))
                for prim in primitives
            }
        }


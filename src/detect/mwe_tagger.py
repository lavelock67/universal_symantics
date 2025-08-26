#!/usr/bin/env python3
"""
Multi-Word Expression (MWE) Tagger for NSM Detection

This module provides MWE detection for quantifiers and intensifiers
that span multiple words, improving detection accuracy.
"""

from typing import List, Dict, Set, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum

class MWEType(Enum):
    """Types of multi-word expressions."""
    QUANTIFIER = "quantifier"
    INTENSIFIER = "intensifier"
    NEGATION = "negation"
    MODALITY = "modality"

@dataclass
class MWE:
    """Multi-word expression."""
    text: str
    type: MWEType
    primes: List[str]
    confidence: float
    start: int
    end: int

class MWETagger:
    """Multi-word expression tagger for NSM detection."""
    
    def __init__(self):
        """Initialize MWE tagger with language-specific lexicons."""
        self.lexicons = {}
        self._load_lexicons()
        self._compile_patterns()
    
    def _load_quantifier_mwes(self) -> Dict[str, Dict]:
        """Load quantifier multi-word expressions."""
        return {
            # English quantifiers
            "at most": {"primes": ["NOT", "MORE"], "confidence": 0.9},
            "no more than": {"primes": ["NOT", "MORE"], "confidence": 0.9},
            "hardly any": {"primes": ["NOT", "MANY"], "confidence": 0.8},
            "almost all": {"primes": ["MOST"], "confidence": 0.85},
            "nearly all": {"primes": ["MOST"], "confidence": 0.85},
            "the majority of": {"primes": ["MOST"], "confidence": 0.9},
            "the minority of": {"primes": ["FEW"], "confidence": 0.9},
            "a few": {"primes": ["SOME"], "confidence": 0.8},
            "quite a few": {"primes": ["MANY"], "confidence": 0.8},
            "a lot of": {"primes": ["MANY"], "confidence": 0.9},
            "lots of": {"primes": ["MANY"], "confidence": 0.9},
            "plenty of": {"primes": ["MANY"], "confidence": 0.8},
            "a great deal of": {"primes": ["MANY"], "confidence": 0.8},
            "a large number of": {"primes": ["MANY"], "confidence": 0.8},
            "a small number of": {"primes": ["FEW"], "confidence": 0.8},
            "less than": {"primes": ["NOT", "MORE"], "confidence": 0.9},
            "more than": {"primes": ["MORE"], "confidence": 0.9},
            "greater than": {"primes": ["MORE"], "confidence": 0.9},
            "fewer than": {"primes": ["NOT", "MORE"], "confidence": 0.9},
            
            # Spanish quantifiers
            "a lo sumo": {"primes": ["NOT", "MORE"], "confidence": 0.9},
            "no más de": {"primes": ["NOT", "MORE"], "confidence": 0.9},
            "casi ningún": {"primes": ["NOT", "MANY"], "confidence": 0.8},
            "casi todos": {"primes": ["MOST"], "confidence": 0.85},
            "la mayoría de": {"primes": ["MOST"], "confidence": 0.9},
            "la minoría de": {"primes": ["FEW"], "confidence": 0.9},
            "unos pocos": {"primes": ["SOME"], "confidence": 0.8},
            "bastantes": {"primes": ["MANY"], "confidence": 0.8},
            "muchos de": {"primes": ["MANY"], "confidence": 0.9},
            "pocos de": {"primes": ["FEW"], "confidence": 0.9},
            
            # French quantifiers
            "au plus": {"primes": ["NOT", "MORE"], "confidence": 0.9},
            "pas plus de": {"primes": ["NOT", "MORE"], "confidence": 0.9},
            "presque aucun": {"primes": ["NOT", "MANY"], "confidence": 0.8},
            "presque tous": {"primes": ["MOST"], "confidence": 0.85},
            "la plupart de": {"primes": ["MOST"], "confidence": 0.9},
            "la minorité de": {"primes": ["FEW"], "confidence": 0.9},
            "quelques": {"primes": ["SOME"], "confidence": 0.8},
            "beaucoup de": {"primes": ["MANY"], "confidence": 0.9},
            "peu de": {"primes": ["FEW"], "confidence": 0.9},
        }
    
    def _load_intensifier_mwes(self) -> Dict[str, Dict]:
        """Load intensifier multi-word expressions."""
        return {
            # English intensifiers
            "way more": {"primes": ["VERY", "MORE"], "confidence": 0.9},
            "far too": {"primes": ["VERY", "MORE"], "confidence": 0.9},
            "much more": {"primes": ["VERY", "MORE"], "confidence": 0.9},
            "a lot more": {"primes": ["VERY", "MORE"], "confidence": 0.9},
            "significantly more": {"primes": ["VERY", "MORE"], "confidence": 0.8},
            "considerably more": {"primes": ["VERY", "MORE"], "confidence": 0.8},
            "substantially more": {"primes": ["VERY", "MORE"], "confidence": 0.8},
            "extremely": {"primes": ["VERY"], "confidence": 0.9},
            "incredibly": {"primes": ["VERY"], "confidence": 0.9},
            "exceptionally": {"primes": ["VERY"], "confidence": 0.8},
            "remarkably": {"primes": ["VERY"], "confidence": 0.8},
            "particularly": {"primes": ["VERY"], "confidence": 0.8},
            "especially": {"primes": ["VERY"], "confidence": 0.8},
            "notably": {"primes": ["VERY"], "confidence": 0.8},
            
            # Spanish intensifiers
            "mucho más": {"primes": ["VERY", "MORE"], "confidence": 0.9},
            "bastante más": {"primes": ["VERY", "MORE"], "confidence": 0.9},
            "considerablemente más": {"primes": ["VERY", "MORE"], "confidence": 0.8},
            "extremadamente": {"primes": ["VERY"], "confidence": 0.9},
            "increíblemente": {"primes": ["VERY"], "confidence": 0.9},
            "excepcionalmente": {"primes": ["VERY"], "confidence": 0.8},
            "particularmente": {"primes": ["VERY"], "confidence": 0.8},
            "especialmente": {"primes": ["VERY"], "confidence": 0.8},
            
            # French intensifiers
            "beaucoup plus": {"primes": ["VERY", "MORE"], "confidence": 0.9},
            "considérablement plus": {"primes": ["VERY", "MORE"], "confidence": 0.8},
            "extrêmement": {"primes": ["VERY"], "confidence": 0.9},
            "incroyablement": {"primes": ["VERY"], "confidence": 0.9},
            "exceptionnellement": {"primes": ["VERY"], "confidence": 0.8},
            "particulièrement": {"primes": ["VERY"], "confidence": 0.8},
            "spécialement": {"primes": ["VERY"], "confidence": 0.8},
        }
    
    def _load_negation_mwes(self) -> Dict[str, Dict]:
        """Load negation multi-word expressions."""
        return {
            # English negations
            "not at all": {"primes": ["NOT"], "confidence": 0.9},
            "by no means": {"primes": ["NOT"], "confidence": 0.9},
            "in no way": {"primes": ["NOT"], "confidence": 0.9},
            "under no circumstances": {"primes": ["NOT"], "confidence": 0.9},
            "not even": {"primes": ["NOT"], "confidence": 0.8},
            "not once": {"primes": ["NOT"], "confidence": 0.8},
            "never ever": {"primes": ["NOT"], "confidence": 0.9},
            
            # Spanish negations
            "de ninguna manera": {"primes": ["NOT"], "confidence": 0.9},
            "bajo ninguna circunstancia": {"primes": ["NOT"], "confidence": 0.9},
            "ni siquiera": {"primes": ["NOT"], "confidence": 0.8},
            "nunca jamás": {"primes": ["NOT"], "confidence": 0.9},
            
            # French negations
            "en aucune façon": {"primes": ["NOT"], "confidence": 0.9},
            "sous aucune circonstance": {"primes": ["NOT"], "confidence": 0.9},
            "même pas": {"primes": ["NOT"], "confidence": 0.8},
            "jamais jamais": {"primes": ["NOT"], "confidence": 0.9},
        }
    
    def _load_modality_mwes(self) -> Dict[str, Dict]:
        """Load modality multi-word expressions."""
        return {
            # English modality
            "have to": {"primes": ["MUST"], "confidence": 0.9},
            "need to": {"primes": ["MUST"], "confidence": 0.9},
            "ought to": {"primes": ["MUST"], "confidence": 0.8},
            "supposed to": {"primes": ["MUST"], "confidence": 0.8},
            "allowed to": {"primes": ["CAN"], "confidence": 0.9},
            "permitted to": {"primes": ["CAN"], "confidence": 0.9},
            "able to": {"primes": ["CAN"], "confidence": 0.9},
            "capable of": {"primes": ["CAN"], "confidence": 0.8},
            
            # Spanish modality
            "tener que": {"primes": ["MUST"], "confidence": 0.9},
            "necesitar": {"primes": ["MUST"], "confidence": 0.9},
            "deber": {"primes": ["MUST"], "confidence": 0.8},
            "poder": {"primes": ["CAN"], "confidence": 0.9},
            "ser capaz de": {"primes": ["CAN"], "confidence": 0.8},
            
            # French modality
            "devoir": {"primes": ["MUST"], "confidence": 0.9},
            "avoir besoin de": {"primes": ["MUST"], "confidence": 0.9},
            "être censé": {"primes": ["MUST"], "confidence": 0.8},
            "pouvoir": {"primes": ["CAN"], "confidence": 0.9},
            "être capable de": {"primes": ["CAN"], "confidence": 0.8},
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.patterns = {}
        for lang, lexicon in self.lexicons.items():
            self.patterns[lang] = {}
            for mwe_text, mwe_info in lexicon.items():
                # Create a case-insensitive pattern
                pattern = re.compile(r'\b' + re.escape(mwe_text.lower()) + r'\b', re.IGNORECASE)
                self.patterns[lang][mwe_text] = pattern
    
    def detect_mwes(self, text: str) -> List[MWE]:
        """Detect multi-word expressions in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected MWEs
        """
        detected_mwes = []
        text_lower = text.lower()
        
        # Detect MWEs from all languages
        for lang, lexicon in self.lexicons.items():
            for mwe_text, mwe_info in lexicon.items():
                # Check if MWE is in text (case-insensitive)
                if mwe_text.lower() in text_lower:
                    # Find the actual position in the original text
                    start_pos = text_lower.find(mwe_text.lower())
                    end_pos = start_pos + len(mwe_text)
                    
                    detected_mwes.append(MWE(
                        text=mwe_text,
                        type=mwe_info["type"],
                        primes=mwe_info["primes"],
                        confidence=mwe_info.get("confidence", 0.8),
                        start=start_pos,
                        end=end_pos
                    ))
        
        # Sort by start position
        detected_mwes.sort(key=lambda x: x.start)
        
        return detected_mwes
    
    def get_primes_from_mwes(self, mwes: List[MWE]) -> List[str]:
        """Extract NSM primes from detected MWEs.
        
        Args:
            mwes: List of detected MWEs
            
        Returns:
            List of NSM primes
        """
        primes = []
        for mwe in mwes:
            primes.extend(mwe.primes)
        return list(set(primes))  # Remove duplicates
    
    def get_mwe_coverage(self, text: str) -> Dict[str, float]:
        """Calculate MWE coverage statistics.
        
        Args:
            text: Input text
            
        Returns:
            Coverage statistics by MWE type
        """
        mwes = self.detect_mwes(text)
        text_length = len(text.split())
        
        coverage = {
            "quantifier": 0.0,
            "intensifier": 0.0,
            "negation": 0.0,
            "modality": 0.0,
            "total": 0.0
        }
        
        if text_length == 0:
            return coverage
        
        for mwe in mwes:
            mwe_words = len(mwe.text.split())
            coverage[mwe.type.value] += mwe_words / text_length
            coverage["total"] += mwe_words / text_length
        
        return coverage

    def _load_lexicons(self):
        """Load language-specific MWE lexicons."""
        self.lexicons = {
            "en": {
                # Quantifiers
                "at most": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MORE"]},
                "at least": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "LESS"]},
                "no more than": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MORE"]},
                "no less than": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "LESS"]},
                "hardly any": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MANY"]},
                "a lot of": {"type": MWEType.QUANTIFIER, "primes": ["MANY"]},
                "lots of": {"type": MWEType.QUANTIFIER, "primes": ["MANY"]},
                "plenty of": {"type": MWEType.QUANTIFIER, "primes": ["MANY"]},
                "most of": {"type": MWEType.QUANTIFIER, "primes": ["MOST"]},
                "some of": {"type": MWEType.QUANTIFIER, "primes": ["SOME"]},
                "all of": {"type": MWEType.QUANTIFIER, "primes": ["ALL"]},
                "none of": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "SOME"]},
                
                # Intensifiers
                "very much": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "way more": {"type": MWEType.INTENSIFIER, "primes": ["VERY", "MORE"]},
                "far too": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "really very": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "extremely": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "incredibly": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                
                # Negations
                "not at all": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "by no means": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "in no way": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "under no circumstances": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                
                # Modalities
                "have to": {"type": MWEType.MODALITY, "primes": ["CAN"]},
                "need to": {"type": MWEType.MODALITY, "primes": ["WANT"]},
                "ought to": {"type": MWEType.MODALITY, "primes": ["CAN"]},
                "supposed to": {"type": MWEType.MODALITY, "primes": ["CAN"]}
            },
            "es": {
                # Spanish Quantifiers
                "a lo sumo": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MORE"]},
                "como máximo": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MORE"]},
                "al menos": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "LESS"]},
                "por lo menos": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "LESS"]},
                "no más de": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MORE"]},
                "no menos de": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "LESS"]},
                "apenas": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MANY"]},
                "casi": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "ALL"]},
                "muchos de": {"type": MWEType.QUANTIFIER, "primes": ["MANY"]},
                "pocos de": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MANY"]},
                "la mayoría de": {"type": MWEType.QUANTIFIER, "primes": ["MOST"]},
                "algunos de": {"type": MWEType.QUANTIFIER, "primes": ["SOME"]},
                "todos de": {"type": MWEType.QUANTIFIER, "primes": ["ALL"]},
                "ninguno de": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "SOME"]},
                "un montón de": {"type": MWEType.QUANTIFIER, "primes": ["MANY"]},
                "un montón": {"type": MWEType.QUANTIFIER, "primes": ["MANY"]},
                
                # Spanish Intensifiers
                "muy mucho": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "mucho más": {"type": MWEType.INTENSIFIER, "primes": ["VERY", "MORE"]},
                "demasiado": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "extremadamente": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "increíblemente": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "sumamente": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "extraordinariamente": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                
                # Spanish Negations
                "de ninguna manera": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "en absoluto": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "para nada": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "bajo ninguna circunstancia": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "ni siquiera": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                
                # Spanish Modalities
                "tener que": {"type": MWEType.MODALITY, "primes": ["CAN"]},
                "necesitar": {"type": MWEType.MODALITY, "primes": ["WANT"]},
                "deber": {"type": MWEType.MODALITY, "primes": ["CAN"]},
                "suponer": {"type": MWEType.MODALITY, "primes": ["THINK"]}
            },
            "fr": {
                # French Quantifiers
                "au plus": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MORE"]},
                "tout au plus": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MORE"]},
                "au moins": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "LESS"]},
                "du moins": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "LESS"]},
                "pas plus de": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MORE"]},
                "pas moins de": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "LESS"]},
                "à peine": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MANY"]},
                "presque": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "ALL"]},
                "beaucoup de": {"type": MWEType.QUANTIFIER, "primes": ["MANY"]},
                "peu de": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "MANY"]},
                "la plupart de": {"type": MWEType.QUANTIFIER, "primes": ["MOST"]},
                "quelques": {"type": MWEType.QUANTIFIER, "primes": ["SOME"]},
                "tous les": {"type": MWEType.QUANTIFIER, "primes": ["ALL"]},
                "aucun de": {"type": MWEType.QUANTIFIER, "primes": ["NOT", "SOME"]},
                "un tas de": {"type": MWEType.QUANTIFIER, "primes": ["MANY"]},
                "une tonne de": {"type": MWEType.QUANTIFIER, "primes": ["MANY"]},
                
                # French Intensifiers
                "très beaucoup": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "beaucoup plus": {"type": MWEType.INTENSIFIER, "primes": ["VERY", "MORE"]},
                "trop": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "extrêmement": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "incroyablement": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "exceptionnellement": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                "remarquablement": {"type": MWEType.INTENSIFIER, "primes": ["VERY"]},
                
                # French Negations
                "en aucune façon": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "absolument pas": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "pas du tout": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "sous aucune circonstance": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                "même pas": {"type": MWEType.NEGATION, "primes": ["NOT"]},
                
                # French Modalities
                "devoir": {"type": MWEType.MODALITY, "primes": ["CAN"]},
                "avoir besoin de": {"type": MWEType.MODALITY, "primes": ["WANT"]},
                "falloir": {"type": MWEType.MODALITY, "primes": ["CAN"]},
                "supposer": {"type": MWEType.MODALITY, "primes": ["THINK"]}
            }
        }

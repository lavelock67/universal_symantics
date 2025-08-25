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
        """Initialize the MWE tagger with lexicons."""
        self.quantifier_mwes = self._load_quantifier_mwes()
        self.intensifier_mwes = self._load_intensifier_mwes()
        self.negation_mwes = self._load_negation_mwes()
        self.modality_mwes = self._load_modality_mwes()
        
        # Compile regex patterns for efficiency
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
        # Create patterns for each MWE type
        self.quantifier_patterns = self._create_patterns(self.quantifier_mwes)
        self.intensifier_patterns = self._create_patterns(self.intensifier_mwes)
        self.negation_patterns = self._create_patterns(self.negation_mwes)
        self.modality_patterns = self._create_patterns(self.modality_mwes)
    
    def _create_patterns(self, mwe_dict: Dict[str, Dict]) -> List[Tuple[re.Pattern, str, Dict]]:
        """Create regex patterns for MWE matching."""
        patterns = []
        for mwe_text, mwe_info in mwe_dict.items():
            # Create case-insensitive pattern
            pattern = re.compile(r'\b' + re.escape(mwe_text) + r'\b', re.IGNORECASE)
            patterns.append((pattern, mwe_text, mwe_info))
        return patterns
    
    def detect_mwes(self, text: str) -> List[MWE]:
        """Detect multi-word expressions in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected MWEs
        """
        detected_mwes = []
        text_lower = text.lower()
        
        # Detect quantifiers
        for pattern, mwe_text, mwe_info in self.quantifier_patterns:
            for match in pattern.finditer(text):
                detected_mwes.append(MWE(
                    text=mwe_text,
                    type=MWEType.QUANTIFIER,
                    primes=mwe_info["primes"],
                    confidence=mwe_info["confidence"],
                    start=match.start(),
                    end=match.end()
                ))
        
        # Detect intensifiers
        for pattern, mwe_text, mwe_info in self.intensifier_patterns:
            for match in pattern.finditer(text):
                detected_mwes.append(MWE(
                    text=mwe_text,
                    type=MWEType.INTENSIFIER,
                    primes=mwe_info["primes"],
                    confidence=mwe_info["confidence"],
                    start=match.start(),
                    end=match.end()
                ))
        
        # Detect negations
        for pattern, mwe_text, mwe_info in self.negation_patterns:
            for match in pattern.finditer(text):
                detected_mwes.append(MWE(
                    text=mwe_text,
                    type=MWEType.NEGATION,
                    primes=mwe_info["primes"],
                    confidence=mwe_info["confidence"],
                    start=match.start(),
                    end=match.end()
                ))
        
        # Detect modality
        for pattern, mwe_text, mwe_info in self.modality_patterns:
            for match in pattern.finditer(text):
                detected_mwes.append(MWE(
                    text=mwe_text,
                    type=MWEType.MODALITY,
                    primes=mwe_info["primes"],
                    confidence=mwe_info["confidence"],
                    start=match.start(),
                    end=match.end()
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

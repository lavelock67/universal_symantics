#!/usr/bin/env python3
"""
Cultural Adaptation System

This module provides cultural adaptation for translations, ensuring
culturally appropriate and contextually aware translations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging

from src.core.domain.models import Language

logger = logging.getLogger(__name__)

class AdaptationType(Enum):
    """Types of cultural adaptation."""
    IDIOMATIC_EXPRESSION = "idiomatic_expression"
    POLITENESS_LEVEL = "politeness_level"
    CULTURAL_NORM = "cultural_norm"
    FORMALITY = "formality"

@dataclass
class AdaptationChange:
    """Represents a single adaptation change."""
    type: AdaptationType
    before: str
    after: str
    justification: str
    confidence: float

@dataclass
class AdaptationResult:
    """Result of cultural adaptation."""
    adapted_text: str
    changes: List[AdaptationChange]
    invariants_checked: bool
    invariants_violated: bool
    violation_details: List[str]

class CulturalInvariantChecker:
    """Checks for invariant violations in cultural adaptation."""
    
    def __init__(self):
        """Initialize the invariant checker."""
        # Patterns that must not be changed
        self.invariant_patterns = {
            # Logical operators
            "NOT": r"\b(not|no|nicht|ne|non|no)\b",
            "TRUE": r"\b(true|verdadero|vrai|wahr|vero)\b",
            "FALSE": r"\b(false|falso|faux|falsch|falso)\b",
            
            # Numbers and quantifiers
            "NUMBERS": r"\b\d+\b",
            "ALL": r"\b(all|todos|toutes|alle|tutti)\b",
            "SOME": r"\b(some|algunos|quelques|einige|alcuni)\b",
            "HALF": r"\b(half|mitad|moitié|hälfte|metà)\b",
            "MORE": r"\b(more|más|plus|mehr|più)\b",
            "LESS": r"\b(less|menos|moins|weniger|meno)\b",
            
            # Time expressions (dayparts)
            "MORNING": r"\b(morning|mañana|matin|morgen|mattina)\b",
            "AFTERNOON": r"\b(afternoon|tarde|après-midi|nachmittag|pomeriggio)\b",
            "EVENING": r"\b(evening|noche|soir|abend|sera)\b",
            "NIGHT": r"\b(night|noche|nuit|nacht|notte)\b",
            
            # Date/time patterns
            "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "TIME": r"\b\d{1,2}:\d{2}\s*(am|pm)?\b",
            
            # Named entities (basic patterns)
            "PERSON": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
            "PLACE": r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b",
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE) 
            for key, pattern in self.invariant_patterns.items()
        }
    
    def extract_invariants(self, text: str) -> Dict[str, List[str]]:
        """Extract invariant expressions from text."""
        invariants = {}
        
        for invariant_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                invariants[invariant_type] = list(set(matches))
        
        return invariants
    
    def check_invariant_violations(self, original_text: str, adapted_text: str) -> Tuple[bool, List[str]]:
        """Check if adaptation violates any invariants."""
        violations = []
        
        # Extract invariants from original text
        original_invariants = self.extract_invariants(original_text)
        
        # Check if any invariants are missing or changed in adapted text
        for invariant_type, original_values in original_invariants.items():
            pattern = self.compiled_patterns[invariant_type]
            adapted_matches = pattern.findall(adapted_text)
            
            # Check for missing invariants
            for original_value in original_values:
                if original_value.lower() not in [match.lower() for match in adapted_matches]:
                    violations.append(f"Missing invariant {invariant_type}: '{original_value}'")
            
            # Check for changed invariants (e.g., "morning" → "night")
            if invariant_type in ["MORNING", "AFTERNOON", "EVENING", "NIGHT"]:
                if self._has_time_violation(original_text, adapted_text):
                    violations.append(f"Time violation: daypart changed inappropriately")
        
        return len(violations) == 0, violations
    
    def _has_time_violation(self, original: str, adapted: str) -> bool:
        """Check for time/daypart violations."""
        time_mappings = {
            "morning": ["afternoon", "evening", "night"],
            "afternoon": ["morning", "night"],
            "evening": ["morning", "afternoon"],
            "night": ["morning", "afternoon", "evening"]
        }
        
        for daypart, incompatible in time_mappings.items():
            if re.search(rf"\b{daypart}\b", original, re.IGNORECASE):
                for incompatible_part in incompatible:
                    if re.search(rf"\b{incompatible_part}\b", adapted, re.IGNORECASE):
                        return True
        
        return False

class CulturalAdaptationSystem:
    """Cultural adaptation system with invariant protection."""
    
    def __init__(self):
        """Initialize the cultural adaptation system."""
        self.invariant_checker = CulturalInvariantChecker()
        
        # Cultural context database
        self.cultural_contexts = {
            "en_US": {
                "formality_levels": ["casual", "neutral", "formal"],
                "politeness_markers": ["please", "thank you", "excuse me"],
                "idiomatic_expressions": {
                    "break a leg": "good luck",
                    "piece of cake": "very easy",
                    "hit the nail on the head": "exactly right"
                }
            },
            "en_GB": {
                "formality_levels": ["casual", "neutral", "formal"],
                "politeness_markers": ["please", "thank you", "pardon me"],
                "idiomatic_expressions": {
                    "break a leg": "good luck",
                    "piece of cake": "very easy",
                    "spot on": "exactly right"
                }
            },
            "es_ES": {
                "formality_levels": ["tú", "usted", "formal"],
                "politeness_markers": ["por favor", "gracias", "perdón"],
                "idiomatic_expressions": {
                    "romper una pierna": "buena suerte",
                    "pan comido": "muy fácil",
                    "dar en el clavo": "exactamente correcto"
                }
            },
            "es_MX": {
                "formality_levels": ["tú", "usted", "formal"],
                "politeness_markers": ["por favor", "gracias", "disculpa"],
                "idiomatic_expressions": {
                    "romper una pierna": "buena suerte",
                    "pan comido": "muy fácil",
                    "dar en el clavo": "exactamente correcto"
                }
            },
            "fr_FR": {
                "formality_levels": ["tu", "vous", "formal"],
                "politeness_markers": ["s'il vous plaît", "merci", "pardon"],
                "idiomatic_expressions": {
                    "casser une jambe": "bonne chance",
                    "du gâteau": "très facile",
                    "mettre dans le mille": "exactement correct"
                }
            },
            "de_DE": {
                "formality_levels": ["du", "Sie", "formal"],
                "politeness_markers": ["bitte", "danke", "entschuldigung"],
                "idiomatic_expressions": {
                    "ein Bein brechen": "viel Glück",
                    "ein Kinderspiel": "sehr einfach",
                    "den Nagel auf den Kopf treffen": "genau richtig"
                }
            },
            "ja_JP": {
                "formality_levels": ["casual", "polite", "honorific"],
                "politeness_markers": ["お願いします", "ありがとう", "すみません"],
                "idiomatic_expressions": {
                    "足を折る": "頑張って",
                    "朝飯前": "とても簡単",
                    "的を射る": "まさに正しい"
                }
            },
            "zh_CN": {
                "formality_levels": ["casual", "polite", "formal"],
                "politeness_markers": ["请", "谢谢", "对不起"],
                "idiomatic_expressions": {
                    "断腿": "祝你好运",
                    "小菜一碟": "很容易",
                    "一针见血": "完全正确"
                }
            }
        }
        
        # Idiomatic expression mappings
        self.idiomatic_mappings = {
            "en": {
                "break a leg": "good luck",
                "piece of cake": "very easy",
                "hit the nail on the head": "exactly right",
                "pull someone's leg": "joke with someone",
                "cost an arm and a leg": "very expensive"
            },
            "es": {
                "romper una pierna": "buena suerte",
                "pan comido": "muy fácil",
                "dar en el clavo": "exactamente correcto",
                "tomar el pelo": "bromear",
                "costar un ojo de la cara": "muy caro"
            },
            "fr": {
                "casser une jambe": "bonne chance",
                "du gâteau": "très facile",
                "mettre dans le mille": "exactement correct",
                "faire marcher": "faire une blague",
                "coûter les yeux de la tête": "très cher"
            },
            "de": {
                "ein Bein brechen": "viel Glück",
                "ein Kinderspiel": "sehr einfach",
                "den Nagel auf den Kopf treffen": "genau richtig",
                "jemanden auf den Arm nehmen": "einen Scherz machen",
                "ein Vermögen kosten": "sehr teuer"
            }
        }
    
    def adapt_text(self, text: str, target_language: Language, 
                   cultural_context: str = None) -> AdaptationResult:
        """Adapt text culturally with invariant protection."""
        
        # Extract invariants before adaptation
        original_invariants = self.invariant_checker.extract_invariants(text)
        
        # Perform cultural adaptation
        adapted_text = text
        changes = []
        
        # Apply idiomatic expression adaptation
        idiomatic_changes = self._adapt_idiomatic_expressions(text, target_language)
        if idiomatic_changes:
            adapted_text = idiomatic_changes["adapted_text"]
            changes.extend(idiomatic_changes["changes"])
        
        # Apply politeness adaptation
        politeness_changes = self._adapt_politeness(adapted_text, target_language, cultural_context)
        if politeness_changes:
            adapted_text = politeness_changes["adapted_text"]
            changes.extend(politeness_changes["changes"])
        
        # Apply cultural norm adaptation
        cultural_changes = self._adapt_cultural_norms(adapted_text, target_language, cultural_context)
        if cultural_changes:
            adapted_text = cultural_changes["adapted_text"]
            changes.extend(cultural_changes["changes"])
        
        # Check for invariant violations
        invariants_ok, violations = self.invariant_checker.check_invariant_violations(
            text, adapted_text
        )
        
        # If invariants are violated, revert to original text
        if not invariants_ok:
            logger.warning(f"Cultural adaptation violated invariants: {violations}")
            adapted_text = text
            changes = []
        
        return AdaptationResult(
            adapted_text=adapted_text,
            changes=changes,
            invariants_checked=True,
            invariants_violated=not invariants_ok,
            violation_details=violations
        )
    
    def _adapt_idiomatic_expressions(self, text: str, target_language: Language) -> Dict[str, Any]:
        """Adapt idiomatic expressions."""
        changes = []
        adapted_text = text
        
        # Get idiomatic mappings for target language
        language_code = target_language.value
        mappings = self.idiomatic_mappings.get(language_code, {})
        
        for idiom, meaning in mappings.items():
            if idiom.lower() in text.lower():
                # Replace idiom with meaning
                adapted_text = re.sub(
                    re.escape(idiom), 
                    meaning, 
                    adapted_text, 
                    flags=re.IGNORECASE
                )
                
                changes.append(AdaptationChange(
                    type=AdaptationType.IDIOMATIC_EXPRESSION,
                    before=idiom,
                    after=meaning,
                    justification=f"Idiomatic expression adapted for {language_code}",
                    confidence=0.9
                ))
        
        return {
            "adapted_text": adapted_text,
            "changes": changes
        }
    
    def _adapt_politeness(self, text: str, target_language: Language, 
                         cultural_context: str = None) -> Dict[str, Any]:
        """Adapt politeness levels."""
        changes = []
        adapted_text = text
        
        # Get cultural context
        context_key = cultural_context or f"{target_language.value}_default"
        context = self.cultural_contexts.get(context_key, {})
        
        # Apply politeness markers based on context
        politeness_markers = context.get("politeness_markers", [])
        
        # Simple politeness adaptation (add "please" for requests)
        if any(word in text.lower() for word in ["send", "give", "bring", "show"]):
            if "please" not in text.lower() and target_language == Language.ENGLISH:
                adapted_text = f"Please {text}"
                changes.append(AdaptationChange(
                    type=AdaptationType.POLITENESS_LEVEL,
                    before=text,
                    after=adapted_text,
                    justification="Added politeness marker for request",
                    confidence=0.8
                ))
        
        return {
            "adapted_text": adapted_text,
            "changes": changes
        }
    
    def _adapt_cultural_norms(self, text: str, target_language: Language, 
                             cultural_context: str = None) -> Dict[str, Any]:
        """Adapt cultural norms."""
        changes = []
        adapted_text = text
        
        # Apply cultural-specific adaptations
        if target_language == Language.JAPANESE:
            # Add honorific markers for formal contexts
            if cultural_context == "ja_JP" and any(word in text for word in ["report", "document", "meeting"]):
                # This is a simplified example - real implementation would be more sophisticated
                pass
        
        return {
            "adapted_text": adapted_text,
            "changes": changes
        }
    
    def get_supported_regions(self) -> List[str]:
        """Get list of supported cultural regions."""
        return list(self.cultural_contexts.keys())
    
    def get_adaptation_types(self) -> List[str]:
        """Get list of supported adaptation types."""
        return [adaptation_type.value for adaptation_type in AdaptationType]

# Factory function
def create_cultural_adaptation_system() -> CulturalAdaptationSystem:
    """Create a cultural adaptation system."""
    return CulturalAdaptationSystem()

#!/usr/bin/env python3
"""
NSM-based translation system.

Implements translation via NSM explications and cross-language primitive mapping.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional, Any
from .explicator import NSMExplicator

logger = logging.getLogger(__name__)


class NSMTranslator:
    """NSM-based translator using explications and primitive mapping."""
    
    def __init__(self):
        """Initialize the NSM translator."""
        self.explicator = NSMExplicator()
        
        # Cross-language primitive mappings
        self.primitive_mappings = self._build_primitive_mappings()
        
    def _build_primitive_mappings(self) -> Dict[str, Dict[str, str]]:
        """Build cross-language primitive mappings."""
        # Basic primitive concepts with cross-language equivalents
        mappings = {
            "AtLocation": {
                "en": "something is in a place",
                "es": "algo está en un lugar", 
                "fr": "quelque chose est dans un lieu"
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
            "UsedFor": {
                "en": "someone can do something with this thing",
                "es": "alguien puede hacer algo con esta cosa",
                "fr": "quelqu'un peut faire quelque chose avec cette chose"
            },
            "HasProperty": {
                "en": "this thing is like this",
                "es": "esta cosa es así",
                "fr": "cette chose est comme cela"
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
        
        return mappings

    def translate_by_primitive(self, primitive: str, target_lang: str) -> str:
        """Translate a primitive concept to target language."""
        return self.explicator.template_for_primitive(primitive, target_lang)

    def legality(self, text: str, lang: str) -> float:
        """Check NSM legality score for text."""
        return self.explicator.legality_score(text, lang)
    
    def detect_primitives_in_text(self, text: str, lang: str) -> List[str]:
        """Detect primitives in text using NSM analysis."""
        # Use the explicator to detect NSM primes
        primes = self.explicator.detect_primes(text, lang)
        
        # Also detect conceptual primitives based on content
        detected_primitives = []
        
        # Import UD detection for broader primitive detection
        try:
            from ..detect.srl_ud_detectors import detect_primitives_multilingual
            ud_primitives = detect_primitives_multilingual(text)
            detected_primitives.extend(ud_primitives)
        except ImportError:
            logger.warning("UD detectors not available")
        
        # Combine NSM primes and UD primitives
        all_primitives = list(set(primes + detected_primitives))
        
        return all_primitives
    
    def generate_explication(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Generate NSM explication for text and translate it."""
        logger.info(f"Generating explication for: {text} ({source_lang} → {target_lang})")
        
        # Detect primitives in source text
        primitives = self.detect_primitives_in_text(text, source_lang)
        
        if not primitives:
            return {
                'success': False,
                'error': 'No primitives detected',
                'source_text': text,
                'source_primitives': [],
                'explication': None,
                'target_explication': None
            }
        
        # Generate explications for each primitive
        source_explications = []
        target_explications = []
        
        for primitive in primitives:
            if primitive in self.primitive_mappings:
                source_exp = self.primitive_mappings[primitive].get(source_lang, "")
                target_exp = self.primitive_mappings[primitive].get(target_lang, "")
                
                if source_exp and target_exp:
                    source_explications.append(source_exp)
                    target_explications.append(target_exp)
        
        # Combine explications
        combined_source = " AND ".join(source_explications) if source_explications else ""
        combined_target = " AND ".join(target_explications) if target_explications else ""
        
        # Check legality
        source_legality = self.legality(combined_source, source_lang) if combined_source else 0.0
        target_legality = self.legality(combined_target, target_lang) if combined_target else 0.0
        
        return {
            'success': bool(combined_target),
            'source_text': text,
            'source_primitives': primitives,
            'source_explication': combined_source,
            'target_explication': combined_target,
            'source_legality': source_legality,
            'target_legality': target_legality,
            'cross_translatable': source_legality > 0.3 and target_legality > 0.3
        }
    
    def translate_via_explications(self, text: str, source_lang: str = "en", 
                                 target_lang: str = "es") -> Dict[str, Any]:
        """Translate text via NSM explications."""
        logger.info(f"NSM translation: {text} ({source_lang} → {target_lang})")
        
        # Generate explications
        explication_result = self.generate_explication(text, source_lang, target_lang)
        
        if not explication_result['success']:
            return explication_result
        
        # For now, return the target explication as the translation
        # In a full implementation, this would surface the explication back to natural language
        translation = explication_result['target_explication']
        
        return {
            'success': True,
            'source_text': text,
            'target_text': translation,
            'translation_method': 'nsm_explication',
            'primitives_used': explication_result['source_primitives'],
            'source_legality': explication_result['source_legality'],
            'target_legality': explication_result['target_legality'],
            'cross_translatable': explication_result['cross_translatable'],
            'explications': {
                'source': explication_result['source_explication'],
                'target': explication_result['target_explication']
            }
        }
    
    def evaluate_translation_quality(self, source_text: str, target_text: str, 
                                   source_lang: str, target_lang: str) -> Dict[str, float]:
        """Evaluate translation quality using NSM metrics."""
        # Check legality of both texts
        source_legality = self.legality(source_text, source_lang)
        target_legality = self.legality(target_text, target_lang)
        
        # Check primitive consistency
        source_primitives = set(self.detect_primitives_in_text(source_text, source_lang))
        target_primitives = set(self.detect_primitives_in_text(target_text, target_lang))
        
        # Calculate primitive overlap
        if source_primitives and target_primitives:
            primitive_overlap = len(source_primitives & target_primitives) / len(source_primitives | target_primitives)
        else:
            primitive_overlap = 0.0
        
        # Overall quality score
        quality_score = (source_legality + target_legality + primitive_overlap) / 3
        
        return {
            'source_legality': source_legality,
            'target_legality': target_legality,
            'primitive_overlap': primitive_overlap,
            'quality_score': quality_score,
            'source_primitives': len(source_primitives),
            'target_primitives': len(target_primitives),
            'shared_primitives': len(source_primitives & target_primitives)
        }
    
    def batch_translate(self, texts: List[str], source_lang: str = "en", 
                       target_lang: str = "es") -> List[Dict[str, Any]]:
        """Translate multiple texts via NSM explications."""
        results = []
        
        for text in texts:
            result = self.translate_via_explications(text, source_lang, target_lang)
            results.append(result)
        
        return results




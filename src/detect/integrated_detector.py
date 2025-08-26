#!/usr/bin/env python3
"""Integrated Detector with Enhanced Patterns and Critical Fixes.

Combines:
- Enhanced UD patterns with critical fixes
- Existing lexical patterns
- MWE detection
- Language-specific optimizations
"""

import logging
from typing import Dict, List, Any, Optional, Set
from src.detect.enhanced_ud_patterns import EnhancedUDPatterns
from src.detect.srl_ud_detectors import detect_primitives_multilingual, detect_primitives_lexical
from src.detect.mwe_tagger import MWETagger
import spacy

logger = logging.getLogger(__name__)


class IntegratedDetector:
    """Integrated detector with enhanced patterns and critical fixes."""
    
    def __init__(self, nlp_models: Dict[str, Any]):
        """Initialize the integrated detector."""
        self.nlp_models = nlp_models
        self.enhanced_patterns = EnhancedUDPatterns(nlp_models)
        self.mwe_tagger = MWETagger()
        
        # Language-specific thresholds
        self.language_thresholds = {
            "en": {"legality": 0.9, "drift": 0.15, "confidence": 0.7},
            "es": {"legality": 0.85, "drift": 0.2, "confidence": 0.65},
            "fr": {"legality": 0.85, "drift": 0.2, "confidence": 0.65}
        }
        
        logger.info("Integrated detector initialized")
    
    def detect_primes_integrated(self, text: str, lang: str = "en") -> List[str]:
        """Detect primes using integrated approach with critical fixes."""
        detected_primes = set()
        
        try:
            # Step 1: Enhanced UD patterns (with critical fixes)
            ud_primes = self.enhanced_patterns.detect_with_enhanced_patterns(text, lang)
            detected_primes.update(ud_primes)
            logger.debug(f"UD patterns detected: {ud_primes}")
            
            # Step 2: MWE detection
            mwes = self.mwe_tagger.detect_mwes(text)
            mwe_primes = self.mwe_tagger.get_primes_from_mwes(mwes)
            detected_primes.update(mwe_primes)
            logger.debug(f"MWE detection: {mwe_primes}")
            
            # Step 3: Enhanced lexical patterns (with language-specific fixes)
            lexical_primes = self._detect_enhanced_lexical(text, lang)
            detected_primes.update(lexical_primes)
            logger.debug(f"Enhanced lexical: {lexical_primes}")
            
            # Step 4: Apply post-processing fixes
            final_primes = self._apply_post_processing_fixes(list(detected_primes), text, lang)
            
            return final_primes
            
        except Exception as e:
            logger.error(f"Error in integrated detection: {e}")
            # Fallback to existing system
            return detect_primitives_multilingual(text)
    
    def _detect_enhanced_lexical(self, text: str, lang: str) -> List[str]:
        """Enhanced lexical detection with language-specific fixes."""
        # Get base lexical detection
        base_primes = detect_primitives_lexical(text)
        
        # Apply language-specific enhancements
        if lang == "es":
            return self._enhance_spanish_lexical(base_primes, text)
        elif lang == "fr":
            return self._enhance_french_lexical(base_primes, text)
        else:
            return base_primes
    
    def _enhance_spanish_lexical(self, base_primes: List[str], text: str) -> List[str]:
        """Enhance Spanish lexical detection with critical fixes."""
        enhanced_primes = base_primes.copy()
        text_lower = text.lower()
        
        # Fix: THIS over-firing prevention
        if "este" in text_lower or "esta" in text_lower or "estos" in text_lower or "estas" in text_lower:
            # Only add THIS if it's pronominal or copular
            if any(phrase in text_lower for phrase in ["esto es", "eso es", "esto está", "eso está"]):
                if "THIS" not in enhanced_primes:
                    enhanced_primes.append("THIS")
            else:
                # Remove THIS if it was added as determiner
                if "THIS" in enhanced_primes:
                    enhanced_primes.remove("THIS")
        
        # Fix: GOOD/BAD false positive prevention
        if "mal" in text_lower:
            # Check if it's used as noun (el mal) vs adjective (está mal)
            if "el mal" in text_lower or "la mal" in text_lower:
                if "BAD" in enhanced_primes:
                    enhanced_primes.remove("BAD")
        
        # Fix: TRUE/FALSE negation scope
        if "no es falso" in text_lower:
            if "FALSE" in enhanced_primes:
                enhanced_primes.remove("FALSE")
            if "TRUE" not in enhanced_primes:
                enhanced_primes.append("TRUE")
        
        # Fix: VERY vs MANY distinction
        if "muchos" in text_lower or "muchas" in text_lower:
            if "MANY" not in enhanced_primes:
                enhanced_primes.append("MANY")
        
        return enhanced_primes
    
    def _enhance_french_lexical(self, base_primes: List[str], text: str) -> List[str]:
        """Enhance French lexical detection with critical fixes."""
        enhanced_primes = base_primes.copy()
        text_lower = text.lower()
        
        # Fix: French negative polarity (personne ≠ PEOPLE)
        if "personne" in text_lower:
            # Check for negative context
            if any(phrase in text_lower for phrase in ["ne personne", "personne ne", "n'y a personne"]):
                if "PEOPLE" in enhanced_primes:
                    enhanced_primes.remove("PEOPLE")
                if "NOT" not in enhanced_primes:
                    enhanced_primes.append("NOT")
        
        # Fix: THIS over-firing prevention
        if "ce" in text_lower or "cet" in text_lower or "cette" in text_lower or "ces" in text_lower:
            # Only add THIS if it's pronominal or copular
            if any(phrase in text_lower for phrase in ["c'est", "ce sont", "cela est", "ça est"]):
                if "THIS" not in enhanced_primes:
                    enhanced_primes.append("THIS")
            else:
                # Remove THIS if it was added as determiner
                if "THIS" in enhanced_primes:
                    enhanced_primes.remove("THIS")
        
        # Fix: GOOD/BAD false positive prevention
        if "bon" in text_lower:
            # Check if it's used as noun (le bon) vs adjective (est bon)
            if "le bon" in text_lower or "la bonne" in text_lower:
                if "GOOD" in enhanced_primes:
                    enhanced_primes.remove("GOOD")
        
        # Fix: TRUE/FALSE negation scope
        if "n'est pas faux" in text_lower or "ne sont pas faux" in text_lower:
            if "FALSE" in enhanced_primes:
                enhanced_primes.remove("FALSE")
            if "TRUE" not in enhanced_primes:
                enhanced_primes.append("TRUE")
        
        # Fix: VERY vs MANY distinction
        if "beaucoup de" in text_lower:
            if "MANY" not in enhanced_primes:
                enhanced_primes.append("MANY")
        
        return enhanced_primes
    
    def _apply_post_processing_fixes(self, primes: List[str], text: str, lang: str) -> List[str]:
        """Apply post-processing fixes to detected primes."""
        fixed_primes = primes.copy()
        
        # Remove duplicates while preserving order
        seen = set()
        unique_primes = []
        for prime in fixed_primes:
            if prime not in seen:
                seen.add(prime)
                unique_primes.append(prime)
        
        # Apply language-specific post-processing
        if lang == "es":
            unique_primes = self._post_process_spanish(unique_primes, text)
        elif lang == "fr":
            unique_primes = self._post_process_french(unique_primes, text)
        
        return unique_primes
    
    def _post_process_spanish(self, primes: List[str], text: str) -> List[str]:
        """Post-process Spanish primes."""
        text_lower = text.lower()
        
        # Ensure proper THIS detection for copular constructions
        if any(phrase in text_lower for phrase in ["esto es", "eso es"]) and "THIS" not in primes:
            primes.append("THIS")
        
        # Ensure proper GOOD detection for predicative adjectives
        if any(phrase in text_lower for phrase in ["es bueno", "está bueno"]) and "GOOD" not in primes:
            primes.append("GOOD")
        
        # Ensure proper TRUE detection for negation flips
        if "no es falso" in text_lower and "TRUE" not in primes:
            primes.append("TRUE")
        
        return primes
    
    def _post_process_french(self, primes: List[str], text: str) -> List[str]:
        """Post-process French primes."""
        text_lower = text.lower()
        
        # Ensure proper THIS detection for contractions
        if "c'est" in text_lower and "THIS" not in primes:
            primes.append("THIS")
        
        # Ensure proper GOOD detection for predicative adjectives
        if any(phrase in text_lower for phrase in ["est bon", "est bonne"]) and "GOOD" not in primes:
            primes.append("GOOD")
        
        # Ensure proper TRUE detection for negation flips
        if "n'est pas faux" in text_lower and "TRUE" not in primes:
            primes.append("TRUE")
        
        return primes
    
    def get_detection_statistics(self, text: str, lang: str = "en") -> Dict[str, Any]:
        """Get detailed detection statistics."""
        stats = {
            "text": text,
            "language": lang,
            "total_primes": 0,
            "ud_primes": [],
            "mwe_primes": [],
            "lexical_primes": [],
            "final_primes": [],
            "processing_time": 0.0
        }
        
        import time
        start_time = time.time()
        
        try:
            # Get individual component results
            ud_primes = self.enhanced_patterns.detect_with_enhanced_patterns(text, lang)
            mwes = self.mwe_tagger.detect_mwes(text)
            mwe_primes = self.mwe_tagger.get_primes_from_mwes(mwes)
            lexical_primes = self._detect_enhanced_lexical(text, lang)
            
            # Get final integrated result
            final_primes = self.detect_primes_integrated(text, lang)
            
            stats.update({
                "ud_primes": ud_primes,
                "mwe_primes": mwe_primes,
                "lexical_primes": lexical_primes,
                "final_primes": final_primes,
                "total_primes": len(final_primes),
                "processing_time": time.time() - start_time
            })
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            stats["error"] = str(e)
        
        return stats

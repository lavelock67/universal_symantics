#!/usr/bin/env python3
"""
Universal Translator Service

This module provides the complete universal translation pipeline:
Source Text → Prime Detection → Prime Generation → Target Text
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

from ..domain.models import Language, NSMPrime
from ..application.services import NSMDetectionService
from ..generation.prime_generator import PrimeGenerator, GenerationStrategy, GenerationResult

logger = logging.getLogger(__name__)

@dataclass
class TranslationResult:
    """Result of universal translation."""
    source_text: str
    target_text: str
    source_language: Language
    target_language: Language
    detected_primes: List[NSMPrime]
    generation_result: GenerationResult
    confidence: float
    processing_time: float
    metadata: Dict

class UniversalTranslator:
    """Universal translator using NSM primes as interlingua."""
    
    def __init__(self):
        """Initialize the universal translator."""
        self.detection_service = NSMDetectionService()
        self.generator = PrimeGenerator()
        logger.info("Universal translator initialized")
    
    def translate(self, source_text: str, source_language: Language, 
                 target_language: Language, generation_strategy: GenerationStrategy = GenerationStrategy.LEXICAL) -> TranslationResult:
        """Translate text from source language to target language using NSM primes.
        
        Args:
            source_text: Text to translate
            source_language: Source language
            target_language: Target language
            generation_strategy: Strategy for text generation
            
        Returns:
            TranslationResult with translation and metadata
        """
        import time
        start_time = time.time()
        
        logger.info(f"Translating from {source_language.value} to {target_language.value}")
        logger.info(f"Source text: '{source_text}'")
        
        # Step 1: Detect primes in source text
        detection_result = self.detection_service.detect_primes(source_text, source_language)
        detected_primes = detection_result.primes
        
        logger.info(f"Detected {len(detected_primes)} primes: {[p.text for p in detected_primes]}")
        
        # Step 2: Generate text in target language from primes
        generation_result = self.generator.generate_text(
            detected_primes, 
            target_language, 
            generation_strategy
        )
        
        target_text = generation_result.text
        
        # Step 3: Calculate overall confidence
        detection_confidence = detection_result.confidence
        generation_confidence = generation_result.confidence
        overall_confidence = (detection_confidence + generation_confidence) / 2
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated target text: '{target_text}'")
        logger.info(f"Overall confidence: {overall_confidence:.2f}")
        
        return TranslationResult(
            source_text=source_text,
            target_text=target_text,
            source_language=source_language,
            target_language=target_language,
            detected_primes=detected_primes,
            generation_result=generation_result,
            confidence=overall_confidence,
            processing_time=processing_time,
            metadata={
                "detection_confidence": detection_confidence,
                "generation_confidence": generation_confidence,
                "prime_count": len(detected_primes),
                "generation_strategy": generation_strategy.value
            }
        )
    
    def decompose_to_semantic_graph(self, source_text: str, detected_primes: List[NSMPrime], 
                                   source_language: Language) -> Dict[str, Any]:
        """Decompose text to semantic graph representation."""
        
        # Create semantic graph from detected primes
        nodes = []
        relationships = []
        
        # Add prime nodes
        for i, prime in enumerate(detected_primes):
            nodes.append({
                "id": f"prime_{i}",
                "type": "prime",
                "text": prime.prime_name,
                "prime_type": prime.prime_type,
                "confidence": prime.confidence
            })
        
        # Add basic relationships based on prime types
        entities = [n for n in nodes if n["prime_type"] in ["SUBSTANTIVE", "RELATIONAL"]]
        actions = [n for n in nodes if n["prime_type"] in ["ACTION", "EVENT"]]
        modifiers = [n for n in nodes if n["prime_type"] in ["EVALUATOR", "DESCRIPTOR"]]
        
        # Create basic semantic structure
        if entities and actions:
            # Entity performs action
            for entity in entities[:2]:  # Limit to first 2 entities
                for action in actions[:1]:  # Limit to first action
                    relationships.append({
                        "source": entity["id"],
                        "target": action["id"],
                        "relation": "performs"
                    })
        
        if actions and len(entities) > 1:
            # Action affects second entity
            for action in actions[:1]:
                for entity in entities[1:2]:
                    relationships.append({
                        "source": action["id"],
                        "target": entity["id"],
                        "relation": "affects"
                    })
        
        return {
            "nodes": nodes,
            "relationships": relationships,
            "source_text": source_text,
            "source_language": source_language.value,
            "prime_count": len(detected_primes)
        }
    
    def translate_batch(self, texts: List[str], source_language: Language, 
                       target_language: Language, generation_strategy: GenerationStrategy = GenerationStrategy.LEXICAL) -> List[TranslationResult]:
        """Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_language: Source language
            target_language: Target language
            generation_strategy: Strategy for text generation
            
        Returns:
            List of TranslationResult objects
        """
        results = []
        for text in texts:
            result = self.translate(text, source_language, target_language, generation_strategy)
            results.append(result)
        return results
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get supported languages for detection and generation."""
        detection_languages = [lang.value for lang in Language]
        generation_languages = [lang.value for lang in self.generator.get_supported_languages()]
        
        return {
            "detection": detection_languages,
            "generation": generation_languages,
            "full_pipeline": list(set(detection_languages) & set(generation_languages))
        }
    
    def get_language_coverage(self, language: Language) -> Dict[str, Dict]:
        """Get coverage statistics for a language."""
        detection_coverage = {
            "supported": language in Language,
            "notes": "Full NSM prime detection support" if language in Language else "Limited support"
        }
        
        generation_coverage = self.generator.get_coverage(language)
        
        return {
            "detection": detection_coverage,
            "generation": generation_coverage
        }
    
    def validate_translation(self, source_text: str, target_text: str, 
                           source_language: Language, target_language: Language) -> Dict:
        """Validate translation quality (basic implementation).
        
        Args:
            source_text: Original source text
            target_text: Generated target text
            source_language: Source language
            target_language: Target language
            
        Returns:
            Validation metrics
        """
        # Basic validation metrics
        source_words = len(source_text.split())
        target_words = len(target_text.split())
        
        # Word count ratio (should be reasonable)
        word_ratio = target_words / source_words if source_words > 0 else 0
        
        # Check if target text is not empty
        has_content = len(target_text.strip()) > 0
        
        # Check if target text is different from source (not just copying)
        is_different = target_text.lower() != source_text.lower()
        
        return {
            "word_count_ratio": word_ratio,
            "has_content": has_content,
            "is_different": is_different,
            "source_word_count": source_words,
            "target_word_count": target_words,
            "overall_valid": has_content and is_different and 0.1 < word_ratio < 10
        }

#!/usr/bin/env python3
"""
Unified Translation Pipeline

This module integrates all components of the universal translator
into a seamless end-to-end translation workflow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time

from ..domain.models import Language, NSMPrime
from ..application.services import NSMDetectionService
from .universal_translator import UniversalTranslator
from ..generation.neural_generator import MultilingualNeuralGenerator, NeuralGenerationResult
from ..generation.prime_generator import PrimeGenerator
from cultural_adaptation_system import CulturalAdaptationSystem

logger = logging.getLogger(__name__)

class TranslationMode(Enum):
    """Translation modes for the unified pipeline."""
    SEMANTIC_DECOMPOSITION = "semantic_decomposition"  # Use semantic decomposition
    NEURAL_GENERATION = "neural_generation"  # Use neural generation
    HYBRID = "hybrid"  # Use both and combine results
    FALLBACK = "fallback"  # Use fallback methods

@dataclass
class TranslationRequest:
    """Request for translation."""
    source_text: str
    source_language: Language
    target_language: Language
    mode: TranslationMode = TranslationMode.HYBRID
    include_metadata: bool = True
    cultural_context: Optional[str] = None

@dataclass
class TranslationResult:
    """Result of translation."""
    source_text: str
    target_text: str
    source_language: Language
    target_language: Language
    confidence: float
    mode: TranslationMode
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class PipelineStep:
    """Represents a step in the translation pipeline."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class UnifiedTranslationPipeline:
    """Unified translation pipeline integrating all components."""
    
    def __init__(self):
        """Initialize the unified translation pipeline."""
        self.detection_service = NSMDetectionService()
        self.universal_translator = UniversalTranslator()
        self.neural_generator = MultilingualNeuralGenerator()
        self.prime_generator = PrimeGenerator()
        self.cultural_adaptation = CulturalAdaptationSystem()
        
        logger.info("Unified translation pipeline initialized")
    
    def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate text using the unified pipeline."""
        
        start_time = time.time()
        pipeline_steps = []
        
        try:
            logger.info(f"Starting translation: {request.source_language} -> {request.target_language}")
            
            # Step 1: Semantic Analysis
            step1 = PipelineStep("semantic_analysis", time.time())
            try:
                detection_result = self.detection_service.detect_primes(
                    request.source_text, 
                    request.source_language
                )
                detected_primes = detection_result.primes
                step1.success = True
                step1.end_time = time.time()
                step1.metadata = {
                    "primes_detected": len(detected_primes),
                    "prime_names": [p.text for p in detected_primes]
                }
                pipeline_steps.append(step1)
                logger.info(f"Semantic analysis completed: {len(detected_primes)} primes detected")
            except Exception as e:
                step1.error = str(e)
                step1.end_time = time.time()
                pipeline_steps.append(step1)
                logger.error(f"Semantic analysis failed: {e}")
                raise
            
            # Step 2: Semantic Decomposition
            step2 = PipelineStep("semantic_decomposition", time.time())
            try:
                semantic_graph = self.universal_translator.decompose_to_semantic_graph(
                    request.source_text,
                    detected_primes,
                    request.source_language
                )
                step2.success = True
                step2.end_time = time.time()
                step2.metadata = {
                    "graph_nodes": len(semantic_graph.get("nodes", [])),
                    "graph_relationships": len(semantic_graph.get("relationships", []))
                }
                pipeline_steps.append(step2)
                logger.info("Semantic decomposition completed")
            except Exception as e:
                step2.error = str(e)
                step2.end_time = time.time()
                pipeline_steps.append(step2)
                logger.error(f"Semantic decomposition failed: {e}")
                raise
            
            # Step 3: Text Generation
            step3 = PipelineStep("text_generation", time.time())
            try:
                if request.mode == TranslationMode.NEURAL_GENERATION:
                    generated_result = self.neural_generator.generate(
                        semantic_graph, 
                        request.target_language
                    )
                    target_text = generated_result.text
                    confidence = generated_result.confidence
                    generation_metadata = generated_result.metadata
                elif request.mode == TranslationMode.SEMANTIC_DECOMPOSITION:
                    # Use prime generator for semantic decomposition approach
                    prime_result = self.prime_generator.generate_text(
                        detected_primes,
                        request.target_language,
                        GenerationStrategy.CONTEXTUAL
                    )
                    target_text = prime_result.text
                    confidence = prime_result.confidence
                    generation_metadata = {"method": "semantic_decomposition"}
                else:  # HYBRID mode
                    # Try neural generation first, fallback to semantic decomposition
                    try:
                        generated_result = self.neural_generator.generate(
                            semantic_graph, 
                            request.target_language
                        )
                        target_text = generated_result.text
                        confidence = generated_result.confidence
                        generation_metadata = generated_result.metadata
                        generation_metadata["method"] = "neural_generation"
                    except Exception as neural_error:
                        logger.warning(f"Neural generation failed, using fallback: {neural_error}")
                        prime_result = self.prime_generator.generate_text(
                            detected_primes,
                            request.target_language,
                            GenerationStrategy.CONTEXTUAL
                        )
                        target_text = prime_result.text
                        confidence = prime_result.confidence * 0.8  # Lower confidence for fallback
                        generation_metadata = {"method": "semantic_decomposition_fallback"}
                
                step3.success = True
                step3.end_time = time.time()
                step3.metadata = generation_metadata
                pipeline_steps.append(step3)
                logger.info(f"Text generation completed: {target_text[:50]}...")
            except Exception as e:
                step3.error = str(e)
                step3.end_time = time.time()
                pipeline_steps.append(step3)
                logger.error(f"Text generation failed: {e}")
                raise
            
            # Step 4: Cultural Adaptation
            step4 = PipelineStep("cultural_adaptation", time.time())
            try:
                if request.cultural_context:
                    adapted_text = self.cultural_adaptation.adapt_text(
                        target_text,
                        request.target_language,
                        request.cultural_context
                    )
                else:
                    # Use default cultural context for target language
                    adapted_text = self.cultural_adaptation.adapt_text(
                        target_text,
                        request.target_language,
                        f"{request.target_language.value}_default"
                    )
                
                step4.success = True
                step4.end_time = time.time()
                step4.metadata = {
                    "original_text": target_text,
                    "adapted_text": adapted_text,
                    "adaptations_applied": True
                }
                pipeline_steps.append(step4)
                logger.info("Cultural adaptation completed")
            except Exception as e:
                step4.error = str(e)
                step4.end_time = time.time()
                pipeline_steps.append(step4)
                logger.warning(f"Cultural adaptation failed, using original text: {e}")
                adapted_text = target_text
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Build metadata
            metadata = {
                "pipeline_steps": [
                    {
                        "name": step.name,
                        "duration": step.end_time - step.start_time if step.end_time else 0,
                        "success": step.success,
                        "error": step.error,
                        "metadata": step.metadata
                    }
                    for step in pipeline_steps
                ],
                "total_processing_time": total_time,
                "translation_mode": request.mode.value,
                "source_language": request.source_language.value,
                "target_language": request.target_language.value
            }
            
            if request.include_metadata:
                metadata.update({
                    "detected_primes": [p.prime_name for p in detected_primes],
                    "semantic_graph": semantic_graph,
                    "generation_metadata": generation_metadata
                })
            
            return TranslationResult(
                source_text=request.source_text,
                target_text=adapted_text,
                source_language=request.source_language,
                target_language=request.target_language,
                confidence=confidence,
                mode=request.mode,
                processing_time=total_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Translation pipeline failed: {e}")
            # Return error result
            return TranslationResult(
                source_text=request.source_text,
                target_text=f"[Translation Error: {str(e)}]",
                source_language=request.source_language,
                target_language=request.target_language,
                confidence=0.0,
                mode=request.mode,
                processing_time=time.time() - start_time,
                metadata={
                    "error": str(e),
                    "pipeline_steps": [
                        {
                            "name": step.name,
                            "duration": step.end_time - step.start_time if step.end_time else 0,
                            "success": step.success,
                            "error": step.error,
                            "metadata": step.metadata
                        }
                        for step in pipeline_steps
                    ]
                }
            )
    
    def translate_simple(self, source_text: str, source_language: Language, 
                        target_language: Language) -> str:
        """Simple translation interface."""
        
        request = TranslationRequest(
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            mode=TranslationMode.HYBRID,
            include_metadata=False
        )
        
        result = self.translate(request)
        return result.target_text
    
    def get_supported_languages(self) -> List[Language]:
        """Get list of supported languages."""
        
        return [
            Language.ENGLISH,
            Language.SPANISH,
            Language.FRENCH,
            Language.GERMAN,
            Language.ITALIAN,
            Language.PORTUGUESE,
            Language.RUSSIAN,
            Language.CHINESE,
            Language.JAPANESE,
            Language.KOREAN
        ]
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation pipeline statistics."""
        
        return {
            "supported_languages": len(self.get_supported_languages()),
            "translation_modes": [mode.value for mode in TranslationMode],
            "components": {
                "detection_service": "NSM Detection Service",
                "universal_translator": "Universal Translator",
                "neural_generator": "Multilingual Neural Generator",
                "prime_generator": "Prime Generator",
                "cultural_adaptation": "Cultural Adaptation System"
            }
        }

# Factory function for creating unified translation pipeline
def create_unified_translation_pipeline() -> UnifiedTranslationPipeline:
    """Create a unified translation pipeline."""
    
    return UnifiedTranslationPipeline()

# Convenience functions for simple translation
def translate_text(source_text: str, source_language: Language, 
                  target_language: Language) -> str:
    """Translate text using the unified pipeline."""
    
    pipeline = create_unified_translation_pipeline()
    return pipeline.translate_simple(source_text, source_language, target_language)

def translate_with_metadata(source_text: str, source_language: Language,
                          target_language: Language) -> TranslationResult:
    """Translate text with full metadata."""
    
    pipeline = create_unified_translation_pipeline()
    request = TranslationRequest(
        source_text=source_text,
        source_language=source_language,
        target_language=target_language,
        mode=TranslationMode.HYBRID,
        include_metadata=True
    )
    
    return pipeline.translate(request)

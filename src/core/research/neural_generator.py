"""
Neural Generation System for NSM

Replaces template-based generation with advanced neural models:
- Fine-tuned T5 for explication-to-surface generation
- MarianMT for cross-lingual generation
- SBERT-based semantic validation
- Context-aware generation with discourse modeling
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from transformers import T5Tokenizer, T5ForConditionalGeneration, MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

from ..domain.models import Language, GenerationResult
from ...shared.config import get_settings
from ...shared.logging import get_logger

logger = get_logger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for neural generation."""
    model_name: str = "t5-base"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 4
    early_stopping: bool = True
    semantic_threshold: float = 0.85
    context_window: int = 3

@dataclass
class SemanticValidationResult:
    """Result of semantic validation."""
    similarity_score: float
    is_valid: bool
    validation_notes: List[str]
    confidence: float

class SemanticValidator:
    """Validates semantic fidelity of generated text using SBERT."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.settings = get_settings()
        logger.info(f"Initialized SemanticValidator with {model_name}")
    
    def validate_generation(self, 
                          explication: str, 
                          generated_text: str,
                          context: Optional[List[str]] = None) -> SemanticValidationResult:
        """Validate that generated text semantically matches the explication."""
        try:
            # Encode explication and generated text
            explication_embedding = self.model.encode([explication])[0]
            generated_embedding = self.model.encode([generated_text])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(explication_embedding, generated_embedding) / (
                np.linalg.norm(explication_embedding) * np.linalg.norm(generated_embedding)
            )
            
            # Context-aware validation if context provided
            context_score = 1.0
            validation_notes = []
            
            if context:
                context_embeddings = self.model.encode(context)
                context_similarities = []
                
                for ctx_emb in context_embeddings:
                    ctx_sim = np.dot(generated_embedding, ctx_emb) / (
                        np.linalg.norm(generated_embedding) * np.linalg.norm(ctx_emb)
                    )
                    context_similarities.append(ctx_sim)
                
                context_score = np.mean(context_similarities)
                if context_score < 0.5:
                    validation_notes.append("Generated text may be contextually inappropriate")
            
            # Combined score
            final_score = (similarity + context_score) / 2
            is_valid = final_score >= self.settings.model.semantic_threshold
            
            if similarity < 0.7:
                validation_notes.append("Low semantic similarity to explication")
            
            confidence = min(final_score, 1.0)
            
            return SemanticValidationResult(
                similarity_score=similarity,
                is_valid=is_valid,
                validation_notes=validation_notes,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Semantic validation failed: {str(e)}")
            return SemanticValidationResult(
                similarity_score=0.0,
                is_valid=False,
                validation_notes=[f"Validation error: {str(e)}"],
                confidence=0.0
            )

class NeuralGenerator:
    """Advanced neural generation system for NSM."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Initialize models
        self._load_models()
        
        # Initialize semantic validator
        self.semantic_validator = SemanticValidator()
        
        self.logger.info("NeuralGenerator initialized successfully")
    
    def _load_models(self):
        """Load neural models for generation."""
        try:
            # T5 for explication-to-surface generation
            self.t5_tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
            
            # MarianMT models for cross-lingual generation
            self.marian_models = {}
            self.marian_tokenizers = {}
            
            language_pairs = {
                Language.ENGLISH: "en",
                Language.SPANISH: "es", 
                Language.FRENCH: "fr"
            }
            
            for lang, code in language_pairs.items():
                if code != "en":  # Skip English as source
                    model_name = f"Helsinki-NLP/opus-mt-en-{code}"
                    try:
                        self.marian_models[lang] = MarianMTModel.from_pretrained(model_name)
                        self.marian_tokenizers[lang] = MarianTokenizer.from_pretrained(model_name)
                        self.logger.info(f"Loaded MarianMT model for {lang.value}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load MarianMT model for {lang.value}: {str(e)}")
            
            # Move models to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.t5_model.to(self.device)
            
            for model in self.marian_models.values():
                model.to(self.device)
                
        except Exception as e:
            self.logger.error(f"Failed to load neural models: {str(e)}")
            raise
    
    def generate_from_explication(self, 
                                explication: str,
                                target_language: Language,
                                context: Optional[List[str]] = None,
                                style: Optional[Dict[str, str]] = None) -> GenerationResult:
        """Generate surface text from NSM explication using neural models."""
        try:
            # Step 1: Generate base surface form using T5
            base_text = self._generate_base_surface(explication, context, style)
            
            # Step 2: Cross-lingual generation if needed
            if target_language != Language.ENGLISH:
                base_text = self._translate_to_target(base_text, target_language)
            
            # Step 3: Semantic validation
            validation = self.semantic_validator.validate_generation(
                explication, base_text, context
            )
            
            # Step 4: Post-processing and refinement
            final_text = self._post_process_generation(base_text, explication, validation)
            
            return GenerationResult(
                generated_text=final_text,
                source_primes=[explication],  # Store explication as source primes
                confidence=validation.confidence,
                processing_time=0.0,  # TODO: Add timing
                target_language=target_language,
                metadata={
                    "explication": explication,
                    "semantic_score": validation.similarity_score,
                    "is_valid": validation.is_valid,
                    "validation_notes": validation.validation_notes,
                    "generation_method": "neural"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Neural generation failed: {str(e)}")
            raise
    
    def _generate_base_surface(self, 
                             explication: str, 
                             context: Optional[List[str]] = None,
                             style: Optional[Dict[str, str]] = None) -> str:
        """Generate base surface form from explication using T5."""
        try:
            # Prepare input with context and style
            input_text = self._prepare_input(explication, context, style)
            
            # Tokenize
            inputs = self.t5_tokenizer(
                input_text,
                max_length=self.config.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=self.config.do_sample,
                    num_beams=self.config.num_beams,
                    early_stopping=self.config.early_stopping,
                    pad_token_id=self.t5_tokenizer.pad_token_id,
                    eos_token_id=self.t5_tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Base surface generation failed: {str(e)}")
            raise
    
    def _translate_to_target(self, text: str, target_language: Language) -> str:
        """Translate text to target language using MarianMT."""
        try:
            if target_language not in self.marian_models:
                self.logger.warning(f"No MarianMT model for {target_language.value}")
                return text
            
            model = self.marian_models[target_language]
            tokenizer = self.marian_tokenizers[target_language]
            
            # Tokenize
            inputs = tokenizer(
                text,
                max_length=self.config.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                    early_stopping=self.config.early_stopping,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            return text  # Fallback to original text
    
    def _prepare_input(self, 
                      explication: str, 
                      context: Optional[List[str]] = None,
                      style: Optional[Dict[str, str]] = None) -> str:
        """Prepare input for T5 generation with context and style."""
        # Base input
        input_parts = [f"Generate surface form from NSM explication: {explication}"]
        
        # Add context if provided
        if context:
            context_text = " ".join(context[-self.config.context_window:])
            input_parts.append(f"Context: {context_text}")
        
        # Add style instructions
        if style:
            style_instructions = []
            if "formality" in style:
                style_instructions.append(f"Formality: {style['formality']}")
            if "directness" in style:
                style_instructions.append(f"Directness: {style['directness']}")
            if "tone" in style:
                style_instructions.append(f"Tone: {style['tone']}")
            
            if style_instructions:
                input_parts.append(f"Style: {', '.join(style_instructions)}")
        
        return " | ".join(input_parts)
    
    def _post_process_generation(self, 
                               text: str, 
                               explication: str,
                               validation: SemanticValidationResult) -> str:
        """Post-process generated text based on validation results."""
        # If semantic validation failed, try to improve the text
        if not validation.is_valid and validation.similarity_score < 0.6:
            # Try regeneration with more specific instructions
            improved_input = f"Generate a more semantically accurate surface form: {explication}"
            
            try:
                inputs = self.t5_tokenizer(
                    improved_input,
                    max_length=self.config.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.t5_model.generate(
                        **inputs,
                        max_length=self.config.max_length,
                        temperature=0.5,  # Lower temperature for more focused generation
                        num_beams=6,  # More beams for better quality
                        early_stopping=True,
                        pad_token_id=self.t5_tokenizer.pad_token_id,
                        eos_token_id=self.t5_tokenizer.eos_token_id
                    )
                
                improved_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return improved_text.strip()
                
            except Exception as e:
                self.logger.warning(f"Post-processing failed: {str(e)}")
                return text
        
        return text
    
    def batch_generate(self, 
                      explications: List[str],
                      target_language: Language,
                      contexts: Optional[List[List[str]]] = None,
                      styles: Optional[List[Dict[str, str]]] = None) -> List[GenerationResult]:
        """Generate multiple surface forms from explications."""
        results = []
        
        for i, explication in enumerate(explications):
            context = contexts[i] if contexts else None
            style = styles[i] if styles else None
            
            try:
                result = self.generate_from_explication(
                    explication, target_language, context, style
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch generation failed for item {i}: {str(e)}")
                # Create error result
                error_result = GenerationResult(
                    generated_text="",
                    explication=explication,
                    target_language=target_language,
                    confidence=0.0,
                    semantic_score=0.0,
                    is_valid=False,
                    validation_notes=[f"Generation error: {str(e)}"],
                    generation_method="neural",
                    processing_time=0.0
                )
                results.append(error_result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "t5_model": self.config.model_name,
            "marian_models": list(self.marian_models.keys()),
            "device": str(self.device),
            "config": {
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
                "semantic_threshold": self.config.semantic_threshold
            }
        }

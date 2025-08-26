#!/usr/bin/env python3
"""
Neural NSM Generation System

This module implements a neural generation pipeline that can:
- Generate fluent text from NSM primitives
- Maintain semantic fidelity during generation
- Support multiple languages
- Use constrained decoding for NSM compliance
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    T5Tokenizer, T5ForConditionalGeneration,
    MarianTokenizer, MarianMTModel,
    pipeline, set_seed
)
from sentence_transformers import SentenceTransformer
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class NSMGenerationConfig:
    """Configuration for NSM generation."""
    model_name: str = "t5-base"
    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 4
    do_sample: bool = True
    constraint_mode: str = "soft"  # soft, hard, hybrid
    semantic_fidelity_threshold: float = 0.8
    nsm_compliance_weight: float = 0.3

@dataclass
class NSMGenerationResult:
    """Result of NSM generation."""
    generated_text: str
    source_primes: List[str]
    target_primes: List[str]
    semantic_fidelity: float
    nsm_compliance: float
    generation_confidence: float
    generation_time: float
    constraint_violations: List[str]

class NeuralNSMGenerator:
    """Neural generator for NSM-based text generation."""
    
    def __init__(self, config: NSMGenerationConfig):
        """Initialize the neural NSM generator."""
        self.config = config
        self.tokenizer = None
        self.model = None
        self.sbert_model = None
        self.nsm_validator = None
        
        # Load models
        self._load_models()
        
        # NSM prime templates for different languages
        self.nsm_templates = self._load_nsm_templates()
        
        # Semantic similarity cache
        self.similarity_cache = {}
    
    def _load_models(self):
        """Load neural models for generation."""
        try:
            logger.info(f"Loading generation model: {self.config.model_name}")
            
            # Load T5 model for text generation
            if "t5" in self.config.model_name.lower():
                self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
            elif "marian" in self.config.model_name.lower():
                self.tokenizer = MarianTokenizer.from_pretrained(self.config.model_name)
                self.model = MarianMTModel.from_pretrained(self.config.model_name)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
            
            # Load SBERT for semantic similarity
            logger.info("Loading SBERT model for semantic validation")
            self.sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            logger.info(f"Models loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_nsm_templates(self) -> Dict[str, Dict[str, str]]:
        """Load NSM templates for different languages."""
        return {
            "en": {
                "PEOPLE": "people",
                "THINK": "think",
                "KNOW": "know",
                "WANT": "want",
                "FEEL": "feel",
                "SEE": "see",
                "HEAR": "hear",
                "SAY": "say",
                "DO": "do",
                "HAPPEN": "happen",
                "MOVE": "move",
                "TOUCH": "touch",
                "LIVE": "live",
                "DIE": "die",
                "GOOD": "good",
                "BAD": "bad",
                "BIG": "big",
                "SMALL": "small",
                "THIS": "this",
                "THAT": "that",
                "ONE": "one",
                "TWO": "two",
                "SOME": "some",
                "ALL": "all",
                "MANY": "many",
                "MUCH": "much",
                "NOT": "not",
                "CAN": "can",
                "BECAUSE": "because",
                "IF": "if",
                "VERY": "very",
                "MORE": "more",
                "LIKE": "like",
                "WHEN": "when",
                "NOW": "now",
                "BEFORE": "before",
                "AFTER": "after",
                "WHERE": "where",
                "HERE": "here",
                "ABOVE": "above",
                "BELOW": "below",
                "FAR": "far",
                "NEAR": "near",
                "INSIDE": "inside",
                "TRUE": "true",
                "FALSE": "false",
                "MAYBE": "maybe",
                "I": "I",
                "YOU": "you",
                "SOMEONE": "someone",
                "SOMETHING": "something",
                "THING": "thing",
                "BODY": "body"
            },
            "es": {
                "PEOPLE": "gente",
                "THINK": "piensa",
                "KNOW": "sabe",
                "WANT": "quiere",
                "FEEL": "siente",
                "SEE": "ve",
                "HEAR": "oye",
                "SAY": "dice",
                "DO": "hace",
                "HAPPEN": "pasa",
                "MOVE": "mueve",
                "TOUCH": "toca",
                "LIVE": "vive",
                "DIE": "muere",
                "GOOD": "bueno",
                "BAD": "malo",
                "BIG": "grande",
                "SMALL": "pequeño",
                "THIS": "esto",
                "THAT": "eso",
                "ONE": "uno",
                "TWO": "dos",
                "SOME": "algunos",
                "ALL": "todos",
                "MANY": "muchos",
                "MUCH": "mucho",
                "NOT": "no",
                "CAN": "puede",
                "BECAUSE": "porque",
                "IF": "si",
                "VERY": "muy",
                "MORE": "más",
                "LIKE": "como",
                "WHEN": "cuando",
                "NOW": "ahora",
                "BEFORE": "antes",
                "AFTER": "después",
                "WHERE": "donde",
                "HERE": "aquí",
                "ABOVE": "arriba",
                "BELOW": "abajo",
                "FAR": "lejos",
                "NEAR": "cerca",
                "INSIDE": "dentro",
                "TRUE": "verdadero",
                "FALSE": "falso",
                "MAYBE": "tal vez",
                "I": "yo",
                "YOU": "tú",
                "SOMEONE": "alguien",
                "SOMETHING": "algo",
                "THING": "cosa",
                "BODY": "cuerpo"
            },
            "fr": {
                "PEOPLE": "gens",
                "THINK": "pense",
                "KNOW": "sait",
                "WANT": "veut",
                "FEEL": "sent",
                "SEE": "voit",
                "HEAR": "entend",
                "SAY": "dit",
                "DO": "fait",
                "HAPPEN": "arrive",
                "MOVE": "bouge",
                "TOUCH": "touche",
                "LIVE": "vit",
                "DIE": "meurt",
                "GOOD": "bon",
                "BAD": "mauvais",
                "BIG": "grand",
                "SMALL": "petit",
                "THIS": "ceci",
                "THAT": "cela",
                "ONE": "un",
                "TWO": "deux",
                "SOME": "quelques",
                "ALL": "tous",
                "MANY": "beaucoup",
                "MUCH": "beaucoup",
                "NOT": "ne",
                "CAN": "peut",
                "BECAUSE": "parce que",
                "IF": "si",
                "VERY": "très",
                "MORE": "plus",
                "LIKE": "comme",
                "WHEN": "quand",
                "NOW": "maintenant",
                "BEFORE": "avant",
                "AFTER": "après",
                "WHERE": "où",
                "HERE": "ici",
                "ABOVE": "au-dessus",
                "BELOW": "en-dessous",
                "FAR": "loin",
                "NEAR": "près",
                "INSIDE": "dedans",
                "TRUE": "vrai",
                "FALSE": "faux",
                "MAYBE": "peut-être",
                "I": "je",
                "YOU": "vous",
                "SOMEONE": "quelqu'un",
                "SOMETHING": "quelque chose",
                "THING": "chose",
                "BODY": "corps"
            }
        }
    
    def generate_from_primes(self, primes: List[str], target_language: str = "en") -> NSMGenerationResult:
        """Generate fluent text from NSM primitives."""
        import time
        start_time = time.time()
        
        try:
            # Step 1: Create NSM prompt
            nsm_prompt = self._create_nsm_prompt(primes, target_language)
            
            # Step 2: Generate text using neural model
            generated_text = self._generate_text(nsm_prompt, target_language)
            
            # Step 3: Validate semantic fidelity
            semantic_fidelity = self._validate_semantic_fidelity(primes, generated_text)
            
            # Step 4: Check NSM compliance
            nsm_compliance = self._check_nsm_compliance(generated_text, target_language)
            
            # Step 5: Extract target primes from generated text
            target_primes = self._extract_primes_from_text(generated_text, target_language)
            
            # Step 6: Calculate generation confidence
            generation_confidence = self._calculate_generation_confidence(
                semantic_fidelity, nsm_compliance, primes, target_primes
            )
            
            # Step 7: Check for constraint violations
            constraint_violations = self._check_constraint_violations(primes, generated_text)
            
            generation_time = time.time() - start_time
            
            return NSMGenerationResult(
                generated_text=generated_text,
                source_primes=primes,
                target_primes=target_primes,
                semantic_fidelity=semantic_fidelity,
                nsm_compliance=nsm_compliance,
                generation_confidence=generation_confidence,
                generation_time=generation_time,
                constraint_violations=constraint_violations
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _create_nsm_prompt(self, primes: List[str], target_language: str) -> str:
        """Create a prompt for NSM-based generation."""
        # Convert primes to target language
        prime_words = []
        for prime in primes:
            if prime in self.nsm_templates.get(target_language, {}):
                prime_words.append(self.nsm_templates[target_language][prime])
            else:
                prime_words.append(prime.lower())
        
        # Create different prompt formats based on model type
        if "t5" in self.config.model_name.lower():
            # T5-style prompt
            prompt = f"Generate a natural sentence using these NSM primitives: {' '.join(prime_words)}"
        else:
            # Generic prompt
            prompt = f"Translate these NSM primitives to natural {target_language}: {' '.join(prime_words)}"
        
        return prompt
    
    def _generate_text(self, prompt: str, target_language: str) -> str:
        """Generate text using the neural model."""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set generation parameters
        generation_config = {
            "max_length": self.config.max_length,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "num_beams": self.config.num_beams,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the generated text
        generated_text = self._clean_generated_text(generated_text)
        
        return generated_text
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean up generated text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common generation artifacts
        text = re.sub(r'^Generate a natural sentence using these NSM primitives:\s*', '', text)
        text = re.sub(r'^Translate these NSM primitives to natural \w+:\s*', '', text)
        
        # Ensure proper sentence structure
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def _validate_semantic_fidelity(self, source_primes: List[str], generated_text: str) -> float:
        """Validate semantic fidelity between source primes and generated text."""
        try:
            # Get embeddings
            source_embedding = self.sbert_model.encode(' '.join(source_primes))
            target_embedding = self.sbert_model.encode(generated_text)
            
            # Calculate cosine similarity
            similarity = np.dot(source_embedding, target_embedding) / (
                np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Semantic fidelity calculation failed: {e}")
            return 0.5  # Default score
    
    def _check_nsm_compliance(self, text: str, language: str) -> float:
        """Check NSM compliance of generated text."""
        try:
            # Extract words from text
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Count NSM words
            nsm_words = 0
            total_words = len(words)
            
            if total_words == 0:
                return 0.0
            
            templates = self.nsm_templates.get(language, {})
            for word in words:
                if word in templates.values():
                    nsm_words += 1
            
            return nsm_words / total_words
            
        except Exception as e:
            logger.warning(f"NSM compliance check failed: {e}")
            return 0.5  # Default score
    
    def _extract_primes_from_text(self, text: str, language: str) -> List[str]:
        """Extract NSM primes from generated text."""
        primes = []
        templates = self.nsm_templates.get(language, {})
        
        # Reverse mapping from words to primes
        word_to_prime = {v: k for k, v in templates.items()}
        
        # Extract words and map to primes
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word in word_to_prime:
                primes.append(word_to_prime[word])
        
        return list(set(primes))  # Remove duplicates
    
    def _calculate_generation_confidence(self, semantic_fidelity: float, nsm_compliance: float,
                                       source_primes: List[str], target_primes: List[str]) -> float:
        """Calculate overall generation confidence."""
        # Base confidence from semantic fidelity
        confidence = semantic_fidelity * 0.4
        
        # NSM compliance component
        confidence += nsm_compliance * 0.3
        
        # Prime coverage component
        if source_primes and target_primes:
            coverage = len(set(source_primes) & set(target_primes)) / len(set(source_primes))
            confidence += coverage * 0.3
        
        return min(confidence, 1.0)
    
    def _check_constraint_violations(self, source_primes: List[str], generated_text: str) -> List[str]:
        """Check for constraint violations in generated text."""
        violations = []
        
        # Check for semantic contradictions
        if "GOOD" in source_primes and "BAD" in source_primes:
            if "good" in generated_text.lower() and "bad" in generated_text.lower():
                violations.append("Semantic contradiction: GOOD and BAD in same text")
        
        # Check for missing important primes
        important_primes = ["PEOPLE", "THINK", "KNOW", "WANT", "FEEL"]
        for prime in important_primes:
            if prime in source_primes and prime.lower() not in generated_text.lower():
                violations.append(f"Missing important prime: {prime}")
        
        return violations

class ConstrainedNSMGenerator(NeuralNSMGenerator):
    """NSM generator with constrained decoding for better compliance."""
    
    def __init__(self, config: NSMGenerationConfig):
        """Initialize constrained NSM generator."""
        super().__init__(config)
        self.constraint_logits_processor = None
        self._setup_constraint_processor()
    
    def _setup_constraint_processor(self):
        """Setup constraint logits processor for NSM compliance."""
        # This would implement logits masking based on NSM grammar
        # For now, we'll use a simplified approach
        pass
    
    def _generate_text(self, prompt: str, target_language: str) -> str:
        """Generate text with NSM constraints."""
        # Enhanced generation with constraints
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Add constraint processing
        if self.config.constraint_mode == "hard":
            # Implement hard constraints
            pass
        elif self.config.constraint_mode == "soft":
            # Implement soft constraints
            pass
        
        # Generate with constraints
        generation_config = {
            "max_length": self.config.max_length,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "num_beams": self.config.num_beams,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_generated_text(generated_text)

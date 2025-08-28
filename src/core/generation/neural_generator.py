#!/usr/bin/env python3
"""
Neural Graph-to-Text Generation

This module implements neural models for converting semantic graphs
back to fluent natural language text, completing the universal translator pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
import time

try:
    import torch
    import transformers
    from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    logging.warning("Neural models not available. Install torch and transformers for neural generation.")

from ..domain.models import Language, NSMPrime
from .prime_generator import GenerationStrategy, GenerationResult

logger = logging.getLogger(__name__)

class NeuralModelType(Enum):
    """Types of neural models for generation."""
    T5 = "t5"
    BART = "bart"
    MT5 = "mt5"  # Multilingual T5
    BLOOM = "bloom"

@dataclass
class NeuralGenerationConfig:
    """Configuration for neural generation."""
    model_type: NeuralModelType
    model_name: str
    max_length: int = 512
    num_beams: int = 4
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True
    early_stopping: bool = True
    length_penalty: float = 1.0
    repetition_penalty: float = 1.2

@dataclass
class NeuralGenerationResult:
    """Result of neural generation."""
    text: str
    confidence: float
    model_type: NeuralModelType
    generation_time: float
    metadata: Dict[str, Any]

class SemanticGraphEncoder:
    """Encodes semantic graphs for neural model input."""
    
    def __init__(self):
        """Initialize the semantic graph encoder."""
        self.node_types = {
            "AGENT": "agent",
            "PATIENT": "patient", 
            "THEME": "theme",
            "GOAL": "goal",
            "LOCATION": "location",
            "TIME": "time",
            "INSTRUMENT": "instrument",
            "MANNER": "manner"
        }
    
    def encode_graph(self, semantic_graph: Dict[str, Any]) -> str:
        """Encode semantic graph to text format for neural model."""
        
        # Extract nodes and relationships
        nodes = semantic_graph.get("nodes", [])
        relationships = semantic_graph.get("relationships", [])
        
        # Build node descriptions
        node_descriptions = []
        for node in nodes:
            if isinstance(node, dict):
                node_type = node.get("type", "entity")
                node_text = node.get("text", "")
                node_descriptions.append(f"{node_type}: {node_text}")
            else:
                node_descriptions.append(str(node))
        
        # Build relationship descriptions
        rel_descriptions = []
        for rel in relationships:
            if isinstance(rel, dict):
                source = rel.get("source", "")
                target = rel.get("target", "")
                relation = rel.get("relation", "relates_to")
                rel_descriptions.append(f"{source} {relation} {target}")
            else:
                rel_descriptions.append(str(rel))
        
        # Combine into structured input
        graph_text = "Graph: "
        if node_descriptions:
            graph_text += "Nodes: " + "; ".join(node_descriptions) + ". "
        if rel_descriptions:
            graph_text += "Relations: " + "; ".join(rel_descriptions) + ". "
        
        return graph_text.strip()
    
    def encode_primes_to_graph(self, primes: List[NSMPrime]) -> str:
        """Encode NSM primes to graph format."""
        
        # Group primes by type
        entities = []
        actions = []
        modifiers = []
        
        for prime in primes:
            if prime.prime_type in ["SUBSTANTIVE", "RELATIONAL"]:
                entities.append(prime.prime_name)
            elif prime.prime_type in ["ACTION", "EVENT"]:
                actions.append(prime.prime_name)
            else:
                modifiers.append(prime.prime_name)
        
        # Build graph description
        graph_parts = []
        
        if entities:
            graph_parts.append(f"Entities: {', '.join(entities)}")
        if actions:
            graph_parts.append(f"Actions: {', '.join(actions)}")
        if modifiers:
            graph_parts.append(f"Modifiers: {', '.join(modifiers)}")
        
        return "Graph: " + "; ".join(graph_parts) + "."

class NeuralGenerator:
    """Neural model for graph-to-text generation."""
    
    def __init__(self, config: NeuralGenerationConfig):
        """Initialize the neural generator."""
        self.config = config
        self.encoder = SemanticGraphEncoder()
        
        if not NEURAL_AVAILABLE:
            logger.warning("Neural models not available. Using fallback generation.")
            self.model = None
            self.tokenizer = None
            return
        
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Failed to load neural model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _load_model(self):
        """Load the specified neural model."""
        
        if self.config.model_type == NeuralModelType.T5:
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        elif self.config.model_type == NeuralModelType.BART:
            self.tokenizer = BartTokenizer.from_pretrained(self.config.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(self.config.model_name)
        elif self.config.model_type == NeuralModelType.MT5:
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Set model to evaluation mode
        if self.model:
            self.model.eval()
        
        logger.info(f"Loaded neural model: {self.config.model_name}")
    
    def generate_from_graph(self, semantic_graph: Dict[str, Any], target_language: Language = Language.ENGLISH) -> NeuralGenerationResult:
        """Generate text from semantic graph."""
        
        start_time = time.time()
        
        try:
            # Encode graph to text
            graph_text = self.encoder.encode_graph(semantic_graph)
            
            if not self.model or not self.tokenizer:
                # Fallback to template generation
                return self._fallback_generation(graph_text, target_language)
            
            # Tokenize input
            inputs = self.tokenizer(
                graph_text,
                max_length=self.config.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    early_stopping=self.config.early_stopping,
                    length_penalty=self.config.length_penalty,
                    repetition_penalty=self.config.repetition_penalty
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(generated_text, graph_text)
            
            generation_time = time.time() - start_time
            
            return NeuralGenerationResult(
                text=generated_text,
                confidence=confidence,
                model_type=self.config.model_type,
                generation_time=generation_time,
                metadata={
                    "input_graph": semantic_graph,
                    "graph_text": graph_text,
                    "model_name": self.config.model_name,
                    "generation_params": {
                        "max_length": self.config.max_length,
                        "num_beams": self.config.num_beams,
                        "temperature": self.config.temperature
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Neural generation failed: {e}")
            return self._fallback_generation(graph_text, target_language)
    
    def generate_from_primes(self, primes: List[NSMPrime], target_language: Language = Language.ENGLISH) -> NeuralGenerationResult:
        """Generate text from NSM primes."""
        
        # Convert primes to graph format
        graph_text = self.encoder.encode_primes_to_graph(primes)
        
        # Create simple graph structure
        semantic_graph = {
            "nodes": [{"type": "prime", "text": prime.prime_name} for prime in primes],
            "relationships": [],
            "source": "nsm_primes"
        }
        
        return self.generate_from_graph(semantic_graph, target_language)
    
    def _fallback_generation(self, graph_text: str, target_language: Language) -> NeuralGenerationResult:
        """Fallback generation when neural model is not available."""
        
        # Simple template-based generation
        if "Entities:" in graph_text and "Actions:" in graph_text:
            # Extract entities and actions
            entities_part = graph_text.split("Actions:")[0].replace("Graph: Entities:", "").strip()
            actions_part = graph_text.split("Actions:")[1].split(";")[0].strip()
            
            entities = [e.strip() for e in entities_part.split(",")]
            actions = [a.strip() for a in actions_part.split(",")]
            
            # Simple template
            if entities and actions:
                text = f"The {entities[0]} {actions[0]}"
                if len(entities) > 1:
                    text += f" the {entities[1]}"
            else:
                text = "Something happened."
        else:
            text = "Something happened."
        
        return NeuralGenerationResult(
            text=text,
            confidence=0.3,  # Low confidence for fallback
            model_type=NeuralModelType.T5,  # Placeholder
            generation_time=0.01,
            metadata={
                "fallback": True,
                "graph_text": graph_text,
                "method": "template_fallback"
            }
        )
    
    def _calculate_confidence(self, generated_text: str, input_text: str) -> float:
        """Calculate confidence score for generated text."""
        
        # Simple heuristics for confidence calculation
        confidence = 0.5  # Base confidence
        
        # Length-based confidence
        if len(generated_text) > 10:
            confidence += 0.2
        
        # Completeness confidence
        if generated_text.endswith(('.', '!', '?')):
            confidence += 0.1
        
        # Content relevance (simple check)
        if any(word in generated_text.lower() for word in input_text.lower().split()):
            confidence += 0.2
        
        return min(confidence, 1.0)

class MultilingualNeuralGenerator:
    """Multilingual neural generator supporting multiple languages."""
    
    def __init__(self):
        """Initialize multilingual neural generator."""
        self.generators = {}
        self.language_models = {
            Language.ENGLISH: NeuralGenerationConfig(
                model_type=NeuralModelType.T5,
                model_name="t5-base"
            ),
            Language.SPANISH: NeuralGenerationConfig(
                model_type=NeuralModelType.MT5,
                model_name="google/mt5-base"
            ),
            Language.FRENCH: NeuralGenerationConfig(
                model_type=NeuralModelType.MT5,
                model_name="google/mt5-base"
            ),
            Language.GERMAN: NeuralGenerationConfig(
                model_type=NeuralModelType.MT5,
                model_name="google/mt5-base"
            ),
            Language.ITALIAN: NeuralGenerationConfig(
                model_type=NeuralModelType.MT5,
                model_name="google/mt5-base"
            )
        }
    
    def get_generator(self, language: Language) -> NeuralGenerator:
        """Get or create generator for specific language."""
        
        if language not in self.generators:
            config = self.language_models.get(language, self.language_models[Language.ENGLISH])
            self.generators[language] = NeuralGenerator(config)
        
        return self.generators[language]
    
    def generate(self, semantic_graph: Dict[str, Any], target_language: Language) -> NeuralGenerationResult:
        """Generate text in target language."""
        
        generator = self.get_generator(target_language)
        return generator.generate_from_graph(semantic_graph, target_language)
    
    def generate_from_primes(self, primes: List[NSMPrime], target_language: Language) -> NeuralGenerationResult:
        """Generate text from primes in target language."""
        
        generator = self.get_generator(target_language)
        return generator.generate_from_primes(primes, target_language)

# Factory function for creating neural generators
def create_neural_generator(model_type: NeuralModelType = NeuralModelType.T5, 
                          model_name: str = "t5-base") -> NeuralGenerator:
    """Create a neural generator with specified configuration."""
    
    config = NeuralGenerationConfig(
        model_type=model_type,
        model_name=model_name
    )
    
    return NeuralGenerator(config)

def create_multilingual_generator() -> MultilingualNeuralGenerator:
    """Create a multilingual neural generator."""
    
    return MultilingualNeuralGenerator()

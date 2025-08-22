#!/usr/bin/env python3
"""
Enhanced NSM Translation with Exponent Surfacing System.

This script implements comprehensive NSM-based translation via explications and exponent surfacing:
1. Advanced primitive detection and explication generation
2. Cross-language explication mapping and validation
3. Exponent surfacing from explications back to natural language
4. Quality assessment and translation optimization
5. Multi-step translation pipeline with validation
6. Surface realization from NSM explications to fluent text
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.nsm.translate import NSMTranslator
    from src.nsm.explicator import NSMExplicator
    from src.nsm.enhanced_explicator import EnhancedNSMExplicator
    from src.table.schema import PeriodicTable
except ImportError as e:
    logger.error(f"Failed to import NSM components: {e}")
    exit(1)


def convert_numpy_types(obj):
    """Convert numpy types and other non-serializable types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_numpy_types(obj.__dict__)
    else:
        return obj


class EnhancedNSMTranslationSurface:
    """Enhanced NSM translation system with exponent surfacing."""
    
    def __init__(self):
        """Initialize the enhanced NSM translation system."""
        self.sbert_model = None
        self.nsm_translator = NSMTranslator()
        self.nsm_explicator = NSMExplicator()
        self.enhanced_explicator = EnhancedNSMExplicator()
        self.languages = ['en', 'es', 'fr']
        
        # Load exponents data
        self.exponents = self._load_exponents()
        
        # Load periodic table
        try:
            with open("data/nsm_periodic_table.json", 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            self.periodic_table = PeriodicTable.from_dict(table_data)
        except Exception as e:
            logger.warning(f"Failed to load periodic table: {e}")
            self.periodic_table = PeriodicTable()
        
        # Enhanced translation templates for exponent surfacing
        self.surface_templates = {
            'AtLocation': {
                'en': {
                    'patterns': [
                        "{thing} is in {place}",
                        "{thing} is at {place}",
                        "{thing} is located in {place}",
                        "you can find {thing} in {place}"
                    ],
                    'slots': ['thing', 'place'],
                    'complexity': 'low'
                },
                'es': {
                    'patterns': [
                        "{thing} está en {place}",
                        "{thing} se encuentra en {place}",
                        "{thing} está ubicado en {place}",
                        "puedes encontrar {thing} en {place}"
                    ],
                    'slots': ['thing', 'place'],
                    'complexity': 'low'
                },
                'fr': {
                    'patterns': [
                        "{thing} est dans {place}",
                        "{thing} se trouve dans {place}",
                        "{thing} est situé dans {place}",
                        "on peut trouver {thing} dans {place}"
                    ],
                    'slots': ['thing', 'place'],
                    'complexity': 'low'
                }
            },
            'HasProperty': {
                'en': {
                    'patterns': [
                        "{thing} is {property}",
                        "{thing} has {property}",
                        "{thing} is characterized by {property}",
                        "the {property} of {thing}"
                    ],
                    'slots': ['thing', 'property'],
                    'complexity': 'medium'
                },
                'es': {
                    'patterns': [
                        "{thing} es {property}",
                        "{thing} tiene {property}",
                        "{thing} se caracteriza por {property}",
                        "la {property} de {thing}"
                    ],
                    'slots': ['thing', 'property'],
                    'complexity': 'medium'
                },
                'fr': {
                    'patterns': [
                        "{thing} est {property}",
                        "{thing} a {property}",
                        "{thing} se caractérise par {property}",
                        "la {property} de {thing}"
                    ],
                    'slots': ['thing', 'property'],
                    'complexity': 'medium'
                }
            },
            'Causes': {
                'en': {
                    'patterns': [
                        "{cause} makes {effect} happen",
                        "{cause} causes {effect}",
                        "{cause} leads to {effect}",
                        "because of {cause}, {effect} happens"
                    ],
                    'slots': ['cause', 'effect'],
                    'complexity': 'high'
                },
                'es': {
                    'patterns': [
                        "{cause} hace que {effect} suceda",
                        "{cause} causa {effect}",
                        "{cause} lleva a {effect}",
                        "debido a {cause}, {effect} sucede"
                    ],
                    'slots': ['cause', 'effect'],
                    'complexity': 'high'
                },
                'fr': {
                    'patterns': [
                        "{cause} fait que {effect} arrive",
                        "{cause} cause {effect}",
                        "{cause} mène à {effect}",
                        "à cause de {cause}, {effect} arrive"
                    ],
                    'slots': ['cause', 'effect'],
                    'complexity': 'high'
                }
            },
            'UsedFor': {
                'en': {
                    'patterns': [
                        "{thing} is used for {purpose}",
                        "people use {thing} to {purpose}",
                        "{thing} serves to {purpose}",
                        "the purpose of {thing} is {purpose}"
                    ],
                    'slots': ['thing', 'purpose'],
                    'complexity': 'medium'
                },
                'es': {
                    'patterns': [
                        "{thing} se usa para {purpose}",
                        "la gente usa {thing} para {purpose}",
                        "{thing} sirve para {purpose}",
                        "el propósito de {thing} es {purpose}"
                    ],
                    'slots': ['thing', 'purpose'],
                    'complexity': 'medium'
                },
                'fr': {
                    'patterns': [
                        "{thing} est utilisé pour {purpose}",
                        "les gens utilisent {thing} pour {purpose}",
                        "{thing} sert à {purpose}",
                        "le but de {thing} est {purpose}"
                    ],
                    'slots': ['thing', 'purpose'],
                    'complexity': 'medium'
                }
            },
            'PartOf': {
                'en': {
                    'patterns': [
                        "{part} is part of {whole}",
                        "{part} belongs to {whole}",
                        "{part} is a component of {whole}",
                        "{whole} contains {part}"
                    ],
                    'slots': ['part', 'whole'],
                    'complexity': 'medium'
                },
                'es': {
                    'patterns': [
                        "{part} es parte de {whole}",
                        "{part} pertenece a {whole}",
                        "{part} es un componente de {whole}",
                        "{whole} contiene {part}"
                    ],
                    'slots': ['part', 'whole'],
                    'complexity': 'medium'
                },
                'fr': {
                    'patterns': [
                        "{part} fait partie de {whole}",
                        "{part} appartient à {whole}",
                        "{part} est un composant de {whole}",
                        "{whole} contient {part}"
                    ],
                    'slots': ['part', 'whole'],
                    'complexity': 'medium'
                }
            },
            'SimilarTo': {
                'en': {
                    'patterns': [
                        "{thing1} is like {thing2}",
                        "{thing1} is similar to {thing2}",
                        "{thing1} resembles {thing2}",
                        "{thing1} and {thing2} are alike"
                    ],
                    'slots': ['thing1', 'thing2'],
                    'complexity': 'low'
                },
                'es': {
                    'patterns': [
                        "{thing1} es como {thing2}",
                        "{thing1} es similar a {thing2}",
                        "{thing1} se parece a {thing2}",
                        "{thing1} y {thing2} son parecidos"
                    ],
                    'slots': ['thing1', 'thing2'],
                    'complexity': 'low'
                },
                'fr': {
                    'patterns': [
                        "{thing1} est comme {thing2}",
                        "{thing1} est similaire à {thing2}",
                        "{thing1} ressemble à {thing2}",
                        "{thing1} et {thing2} se ressemblent"
                    ],
                    'slots': ['thing1', 'thing2'],
                    'complexity': 'low'
                }
            },
            'DifferentFrom': {
                'en': {
                    'patterns': [
                        "{thing1} is different from {thing2}",
                        "{thing1} is not like {thing2}",
                        "{thing1} differs from {thing2}",
                        "{thing1} and {thing2} are different"
                    ],
                    'slots': ['thing1', 'thing2'],
                    'complexity': 'low'
                },
                'es': {
                    'patterns': [
                        "{thing1} es diferente de {thing2}",
                        "{thing1} no es como {thing2}",
                        "{thing1} difiere de {thing2}",
                        "{thing1} y {thing2} son diferentes"
                    ],
                    'slots': ['thing1', 'thing2'],
                    'complexity': 'low'
                },
                'fr': {
                    'patterns': [
                        "{thing1} est différent de {thing2}",
                        "{thing1} n'est pas comme {thing2}",
                        "{thing1} diffère de {thing2}",
                        "{thing1} et {thing2} sont différents"
                    ],
                    'slots': ['thing1', 'thing2'],
                    'complexity': 'low'
                }
            },
            'Not': {
                'en': {
                    'patterns': [
                        "{thing} is not {property}",
                        "it is not the case that {statement}",
                        "{thing} does not {action}",
                        "not {statement}"
                    ],
                    'slots': ['thing', 'property', 'statement', 'action'],
                    'complexity': 'low'
                },
                'es': {
                    'patterns': [
                        "{thing} no es {property}",
                        "no es el caso que {statement}",
                        "{thing} no {action}",
                        "no {statement}"
                    ],
                    'slots': ['thing', 'property', 'statement', 'action'],
                    'complexity': 'low'
                },
                'fr': {
                    'patterns': [
                        "{thing} n'est pas {property}",
                        "il n'est pas le cas que {statement}",
                        "{thing} ne {action} pas",
                        "ne pas {statement}"
                    ],
                    'slots': ['thing', 'property', 'statement', 'action'],
                    'complexity': 'low'
                }
            },
            'Exist': {
                'en': {
                    'patterns': [
                        "{thing} exists",
                        "there is {thing}",
                        "{thing} is real",
                        "we can find {thing}"
                    ],
                    'slots': ['thing'],
                    'complexity': 'low'
                },
                'es': {
                    'patterns': [
                        "{thing} existe",
                        "hay {thing}",
                        "{thing} es real",
                        "podemos encontrar {thing}"
                    ],
                    'slots': ['thing'],
                    'complexity': 'low'
                },
                'fr': {
                    'patterns': [
                        "{thing} existe",
                        "il y a {thing}",
                        "{thing} est réel",
                        "on peut trouver {thing}"
                    ],
                    'slots': ['thing'],
                    'complexity': 'low'
                }
            }
        }
        
        # Default slot fillers for surface realization
        self.default_slots = {
            'en': {
                'thing': 'something',
                'place': 'somewhere',
                'property': 'some quality',
                'cause': 'something',
                'effect': 'something else',
                'purpose': 'some purpose',
                'part': 'a part',
                'whole': 'the whole',
                'thing1': 'one thing',
                'thing2': 'another thing',
                'statement': 'this',
                'action': 'do something'
            },
            'es': {
                'thing': 'algo',
                'place': 'algún lugar',
                'property': 'alguna cualidad',
                'cause': 'algo',
                'effect': 'otra cosa',
                'purpose': 'algún propósito',
                'part': 'una parte',
                'whole': 'el todo',
                'thing1': 'una cosa',
                'thing2': 'otra cosa',
                'statement': 'esto',
                'action': 'hacer algo'
            },
            'fr': {
                'thing': 'quelque chose',
                'place': 'quelque part',
                'property': 'une qualité',
                'cause': 'quelque chose',
                'effect': 'autre chose',
                'purpose': 'un but',
                'part': 'une partie',
                'whole': 'le tout',
                'thing1': 'une chose',
                'thing2': 'une autre chose',
                'statement': 'ceci',
                'action': 'faire quelque chose'
            }
        }
        
        self._load_models()
    
    def _load_exponents(self) -> Dict[str, Any]:
        """Load NSM exponents data."""
        try:
            with open("data/nsm_exponents_en_es_fr.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load exponents: {e}")
            return {"primes": {}, "metadata": {}}
    
    def _load_models(self):
        """Load SBERT model for semantic evaluation."""
        try:
            logger.info("Loading SBERT model for translation quality assessment...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def detect_primitives_enhanced(self, text: str, lang: str) -> List[Dict[str, Any]]:
        """Enhanced primitive detection with confidence scores."""
        primitives = []
        
        # Use enhanced explicator for detection
        detected_primes = self.enhanced_explicator.detect_primes(text, lang)
        
        # Map detected primes to our primitive system
        prime_to_primitive = {
            'BE': 'HasProperty',
            'BE_SOMEWHERE': 'AtLocation',
            'IN': 'AtLocation',
            'NOT': 'Not',
            'BECAUSE': 'Causes',
            'LIKE': 'SimilarTo',
            'DIFFERENT': 'DifferentFrom',
            'PART': 'PartOf',
            'THERE_IS': 'Exist',
            'DO': 'UsedFor',
            'HAPPEN': 'Causes'
        }
        
        for prime in detected_primes:
            if prime in prime_to_primitive:
                primitive = prime_to_primitive[prime]
                confidence = self._calculate_primitive_confidence(text, primitive, lang)
                primitives.append({
                    'primitive': primitive,
                    'prime': prime,
                    'confidence': confidence,
                    'detected_in_text': text
                })
        
        # Also try pattern-based detection
        pattern_primitives = self._detect_via_patterns(text, lang)
        for prim_info in pattern_primitives:
            # Avoid duplicates
            if not any(p['primitive'] == prim_info['primitive'] for p in primitives):
                primitives.append(prim_info)
        
        # Sort by confidence
        primitives.sort(key=lambda x: x['confidence'], reverse=True)
        
        return primitives
    
    def _calculate_primitive_confidence(self, text: str, primitive: str, lang: str) -> float:
        """Calculate confidence score for primitive detection."""
        base_confidence = 0.5
        
        # Check for specific keywords that boost confidence
        keyword_mappings = {
            'AtLocation': {
                'en': ['in', 'at', 'on', 'inside', 'where', 'location', 'place'],
                'es': ['en', 'dentro', 'donde', 'lugar', 'ubicación'],
                'fr': ['dans', 'à', 'sur', 'où', 'endroit', 'lieu']
            },
            'HasProperty': {
                'en': ['is', 'has', 'property', 'characteristic', 'quality'],
                'es': ['es', 'tiene', 'propiedad', 'característica', 'cualidad'],
                'fr': ['est', 'a', 'propriété', 'caractéristique', 'qualité']
            },
            'Causes': {
                'en': ['because', 'cause', 'reason', 'due', 'leads', 'makes'],
                'es': ['porque', 'causa', 'razón', 'debido', 'lleva', 'hace'],
                'fr': ['parce', 'cause', 'raison', 'dû', 'mène', 'fait']
            },
            'UsedFor': {
                'en': ['use', 'for', 'purpose', 'function', 'tool'],
                'es': ['usar', 'para', 'propósito', 'función', 'herramienta'],
                'fr': ['utiliser', 'pour', 'but', 'fonction', 'outil']
            }
        }
        
        keywords = keyword_mappings.get(primitive, {}).get(lang, [])
        text_lower = text.lower()
        
        # Boost confidence based on keyword presence
        keyword_hits = sum(1 for keyword in keywords if keyword in text_lower)
        keyword_boost = min(0.4, keyword_hits * 0.1)
        
        # Check explication quality
        explication = self.enhanced_explicator.generate_explication(text, lang, primitive)
        legality_score = self.nsm_explicator.legality_score(explication, lang)
        legality_boost = legality_score * 0.3
        
        return min(1.0, base_confidence + keyword_boost + legality_boost)
    
    def _detect_via_patterns(self, text: str, lang: str) -> List[Dict[str, Any]]:
        """Detect primitives via pattern matching."""
        patterns = []
        text_lower = text.lower()
        
        # Simple pattern detection
        if lang == 'en':
            if any(word in text_lower for word in ['in', 'at', 'on']):
                patterns.append({
                    'primitive': 'AtLocation',
                    'prime': 'IN',
                    'confidence': 0.7,
                    'detected_in_text': text
                })
            if any(word in text_lower for word in ['because', 'due to', 'cause']):
                patterns.append({
                    'primitive': 'Causes',
                    'prime': 'BECAUSE',
                    'confidence': 0.8,
                    'detected_in_text': text
                })
            if any(word in text_lower for word in ['like', 'similar']):
                patterns.append({
                    'primitive': 'SimilarTo',
                    'prime': 'LIKE',
                    'confidence': 0.7,
                    'detected_in_text': text
                })
        
        return patterns
    
    def generate_enhanced_explication(self, text: str, lang: str) -> Dict[str, Any]:
        """Generate enhanced NSM explication with metadata."""
        # Detect primitives
        primitives = self.detect_primitives_enhanced(text, lang)
        
        if not primitives:
            return {
                'success': False,
                'error': 'No primitives detected',
                'text': text,
                'language': lang,
                'explication': None,
                'legality_score': 0.0
            }
        
        # Generate explication for the highest confidence primitive
        best_primitive = primitives[0]
        primitive_name = best_primitive['primitive']
        
        # Get template-based explication
        template_explication = self.enhanced_explicator.template_for_primitive(primitive_name, lang)
        
        # Calculate legality
        legality_score = self.nsm_explicator.legality_score(template_explication, lang)
        
        return {
            'success': True,
            'text': text,
            'language': lang,
            'detected_primitives': primitives,
            'primary_primitive': best_primitive,
            'explication': template_explication,
            'legality_score': legality_score,
            'metadata': {
                'confidence': best_primitive['confidence'],
                'detection_method': 'enhanced_pattern_detection'
            }
        }
    
    def surface_explication_to_text(self, explication: str, primitive: str, lang: str, 
                                   slots: Dict[str, str] = None) -> Dict[str, Any]:
        """Surface NSM explication back to natural language text."""
        if primitive not in self.surface_templates:
            return {
                'success': False,
                'error': f'No surface template for primitive: {primitive}',
                'explication': explication,
                'surfaced_text': explication  # Fallback to explication
            }
        
        templates = self.surface_templates[primitive].get(lang, {})
        patterns = templates.get('patterns', [])
        
        if not patterns:
            return {
                'success': False,
                'error': f'No patterns for primitive {primitive} in language {lang}',
                'explication': explication,
                'surfaced_text': explication
            }
        
        # Use provided slots or defaults
        if slots is None:
            slots = self.default_slots.get(lang, {})
        
        # Try each pattern and choose the best one
        surfaced_options = []
        
        for pattern in patterns:
            try:
                # Fill in the slots
                surfaced_text = pattern.format(**slots)
                
                # Calculate quality of surfacing
                quality_score = self._evaluate_surfacing_quality(
                    explication, surfaced_text, lang
                )
                
                surfaced_options.append({
                    'text': surfaced_text,
                    'pattern': pattern,
                    'quality': quality_score,
                    'slots_used': slots
                })
            except KeyError as e:
                # Skip patterns that require missing slots
                continue
        
        if not surfaced_options:
            return {
                'success': False,
                'error': 'No patterns could be applied with available slots',
                'explication': explication,
                'surfaced_text': explication
            }
        
        # Choose the best option
        best_option = max(surfaced_options, key=lambda x: x['quality'])
        
        return {
            'success': True,
            'explication': explication,
            'surfaced_text': best_option['text'],
            'pattern_used': best_option['pattern'],
            'quality_score': best_option['quality'],
            'slots_used': best_option['slots_used'],
            'all_options': surfaced_options
        }
    
    def _evaluate_surfacing_quality(self, explication: str, surfaced_text: str, lang: str) -> float:
        """Evaluate the quality of explication surfacing."""
        if not self.sbert_model:
            return 0.5  # Default score if no model
        
        try:
            # Semantic similarity between explication and surfaced text
            embeddings = self.sbert_model.encode([explication, surfaced_text])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            # Fluency check (simple heuristics)
            fluency_score = self._evaluate_fluency(surfaced_text, lang)
            
            # Combined quality score
            quality = 0.6 * float(similarity) + 0.4 * fluency_score
            
            return min(1.0, max(0.0, quality))
        except Exception as e:
            logger.warning(f"Quality evaluation failed: {e}")
            return 0.5
    
    def _evaluate_fluency(self, text: str, lang: str) -> float:
        """Evaluate text fluency using simple heuristics."""
        # Basic fluency checks
        fluency_score = 1.0
        
        # Check for proper sentence structure
        if not text.strip():
            return 0.0
        
        # Check for reasonable length
        words = text.split()
        if len(words) < 2:
            fluency_score *= 0.5
        elif len(words) > 20:
            fluency_score *= 0.8
        
        # Check for proper capitalization
        if not text[0].isupper():
            fluency_score *= 0.9
        
        # Check for repetitive patterns
        if len(set(words)) < len(words) * 0.5:
            fluency_score *= 0.7
        
        # Language-specific checks
        if lang == 'en':
            # Check for proper articles
            if any(word in text.lower() for word in ['a', 'an', 'the']):
                fluency_score *= 1.1
        elif lang == 'es':
            # Check for proper articles
            if any(word in text.lower() for word in ['el', 'la', 'un', 'una']):
                fluency_score *= 1.1
        elif lang == 'fr':
            # Check for proper articles
            if any(word in text.lower() for word in ['le', 'la', 'un', 'une']):
                fluency_score *= 1.1
        
        return min(1.0, fluency_score)
    
    def translate_with_surfacing(self, text: str, source_lang: str, target_lang: str,
                               custom_slots: Dict[str, str] = None) -> Dict[str, Any]:
        """Complete translation pipeline with explication and surfacing."""
        logger.info(f"Enhanced NSM translation with surfacing: {text} ({source_lang} → {target_lang})")
        
        # Step 1: Generate source explication
        source_result = self.generate_enhanced_explication(text, source_lang)
        
        if not source_result['success']:
            return {
                'success': False,
                'error': source_result['error'],
                'translation_step': 'source_explication',
                'source_text': text
            }
        
        # Step 2: Map explication to target language
        source_primitive = source_result['primary_primitive']['primitive']
        target_explication = self.enhanced_explicator.template_for_primitive(source_primitive, target_lang)
        target_legality = self.nsm_explicator.legality_score(target_explication, target_lang)
        
        # Step 3: Surface target explication to natural language
        surface_result = self.surface_explication_to_text(
            target_explication, source_primitive, target_lang, custom_slots
        )
        
        # Step 4: Evaluate translation quality
        translation_quality = self._evaluate_translation_quality(
            text, surface_result['surfaced_text'], source_lang, target_lang
        )
        
        return {
            'success': True,
            'source_text': text,
            'target_text': surface_result['surfaced_text'],
            'source_language': source_lang,
            'target_language': target_lang,
            'translation_method': 'nsm_explication_surfacing',
            'pipeline_steps': {
                'source_explication': source_result,
                'target_explication': {
                    'explication': target_explication,
                    'legality_score': target_legality,
                    'primitive': source_primitive
                },
                'surfacing': surface_result
            },
            'quality_assessment': translation_quality,
            'metadata': {
                'primitive_used': source_primitive,
                'source_confidence': source_result['primary_primitive']['confidence'],
                'surfacing_quality': surface_result.get('quality_score', 0.0),
                'cross_translatable': source_result['legality_score'] > 0.3 and target_legality > 0.3
            }
        }
    
    def _evaluate_translation_quality(self, source_text: str, target_text: str,
                                    source_lang: str, target_lang: str) -> Dict[str, float]:
        """Evaluate translation quality using multiple metrics."""
        quality_metrics = {}
        
        if self.sbert_model:
            try:
                # Semantic similarity
                embeddings = self.sbert_model.encode([source_text, target_text])
                semantic_similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                quality_metrics['semantic_similarity'] = float(semantic_similarity)
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
                quality_metrics['semantic_similarity'] = 0.0
        else:
            quality_metrics['semantic_similarity'] = 0.0
        
        # Fluency assessment
        quality_metrics['target_fluency'] = self._evaluate_fluency(target_text, target_lang)
        quality_metrics['source_fluency'] = self._evaluate_fluency(source_text, source_lang)
        
        # Length ratio (reasonable translations should have similar lengths)
        source_words = len(source_text.split())
        target_words = len(target_text.split())
        if source_words > 0:
            length_ratio = target_words / source_words
            # Penalize extreme length differences
            length_score = 1.0 - min(0.5, abs(1.0 - length_ratio))
            quality_metrics['length_consistency'] = length_score
        else:
            quality_metrics['length_consistency'] = 0.0
        
        # Overall quality score
        quality_metrics['overall_quality'] = (
            0.4 * quality_metrics['semantic_similarity'] +
            0.3 * quality_metrics['target_fluency'] +
            0.2 * quality_metrics['source_fluency'] +
            0.1 * quality_metrics['length_consistency']
        )
        
        return quality_metrics
    
    def evaluate_translation_dataset(self, dataset_path: str = None) -> Dict[str, Any]:
        """Evaluate translation performance on a dataset."""
        if dataset_path is None:
            # Try to find suitable test data
            possible_paths = [
                "data/parallel_test_data_1k.json",
                "data/expanded_parallel_test_data.json",
                "data/parallel_test_data.json"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    dataset_path = path
                    break
        
        if not dataset_path or not Path(dataset_path).exists():
            logger.error("No suitable translation test dataset found")
            return {}
        
        logger.info(f"Evaluating translation performance on dataset: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Extract test data
        test_entries = []
        if 'entries' in dataset:
            for entry in dataset['entries']:
                if 'translations' in entry:
                    test_entries.append(entry)
        
        # Perform translations and evaluate
        results = {
            'total_entries': len(test_entries),
            'language_pairs': {},
            'primitive_performance': {},
            'overall_metrics': {
                'avg_semantic_similarity': 0.0,
                'avg_target_fluency': 0.0,
                'avg_overall_quality': 0.0,
                'success_rate': 0.0
            }
        }
        
        all_qualities = []
        successful_translations = 0
        
        for entry in test_entries[:50]:  # Limit for testing
            source_text = entry.get('text', '')
            source_lang = entry.get('language', 'en')
            
            # Try translating to other languages
            for target_lang in self.languages:
                if target_lang != source_lang:
                    try:
                        translation_result = self.translate_with_surfacing(
                            source_text, source_lang, target_lang
                        )
                        
                        if translation_result['success']:
                            successful_translations += 1
                            quality = translation_result['quality_assessment']
                            all_qualities.append(quality)
                            
                            # Update language pair statistics
                            pair_key = f"{source_lang}→{target_lang}"
                            if pair_key not in results['language_pairs']:
                                results['language_pairs'][pair_key] = {
                                    'translations': 0,
                                    'avg_quality': 0.0,
                                    'qualities': []
                                }
                            
                            results['language_pairs'][pair_key]['translations'] += 1
                            results['language_pairs'][pair_key]['qualities'].append(quality['overall_quality'])
                            
                            # Update primitive performance
                            primitive = translation_result['metadata']['primitive_used']
                            if primitive not in results['primitive_performance']:
                                results['primitive_performance'][primitive] = {
                                    'translations': 0,
                                    'avg_quality': 0.0,
                                    'qualities': []
                                }
                            
                            results['primitive_performance'][primitive]['translations'] += 1
                            results['primitive_performance'][primitive]['qualities'].append(quality['overall_quality'])
                    
                    except Exception as e:
                        logger.warning(f"Translation failed for {source_text}: {e}")
        
        # Calculate averages
        if all_qualities:
            results['overall_metrics']['avg_semantic_similarity'] = np.mean([q['semantic_similarity'] for q in all_qualities])
            results['overall_metrics']['avg_target_fluency'] = np.mean([q['target_fluency'] for q in all_qualities])
            results['overall_metrics']['avg_overall_quality'] = np.mean([q['overall_quality'] for q in all_qualities])
        
        total_attempts = len(test_entries) * (len(self.languages) - 1)
        results['overall_metrics']['success_rate'] = successful_translations / total_attempts if total_attempts > 0 else 0.0
        
        # Calculate language pair averages
        for pair_key, pair_data in results['language_pairs'].items():
            if pair_data['qualities']:
                pair_data['avg_quality'] = np.mean(pair_data['qualities'])
        
        # Calculate primitive averages
        for primitive, prim_data in results['primitive_performance'].items():
            if prim_data['qualities']:
                prim_data['avg_quality'] = np.mean(prim_data['qualities'])
        
        return results


def main():
    """Main function to run enhanced NSM translation with surfacing."""
    logger.info("Starting enhanced NSM translation with exponent surfacing...")
    
    # Initialize enhanced translator
    translator = EnhancedNSMTranslationSurface()
    
    # Test translation examples
    test_examples = [
        {"text": "The book is on the table", "source": "en", "target": "es"},
        {"text": "The car is in the garage", "source": "en", "target": "fr"},
        {"text": "Esta cosa es como otra cosa", "source": "es", "target": "en"},
        {"text": "Cette chose fait partie de l'ensemble", "source": "fr", "target": "en"}
    ]
    
    # Perform test translations
    translation_results = []
    for example in test_examples:
        result = translator.translate_with_surfacing(
            example["text"], example["source"], example["target"]
        )
        translation_results.append(result)
        
        print(f"\nTranslation: {example['text']} ({example['source']} → {example['target']})")
        if result['success']:
            print(f"Result: {result['target_text']}")
            print(f"Quality: {result['quality_assessment']['overall_quality']:.3f}")
            print(f"Primitive: {result['metadata']['primitive_used']}")
        else:
            print(f"Failed: {result['error']}")
    
    # Evaluate on dataset
    dataset_results = translator.evaluate_translation_dataset()
    
    # Save results
    output_path = "data/nsm_translation_surface_enhanced_report.json"
    report = {
        "metadata": {
            "report_type": "enhanced_NSM_translation_surface_report",
            "timestamp": "2025-08-22",
            "enhanced_features": [
                "exponent_surfacing",
                "enhanced_primitive_detection",
                "quality_assessment",
                "fluency_evaluation",
                "multi_step_pipeline"
            ]
        },
        "test_translations": translation_results,
        "dataset_evaluation": dataset_results,
        "summary": {
            "translation_examples": len(translation_results),
            "successful_examples": sum(1 for r in translation_results if r['success']),
            "dataset_success_rate": dataset_results.get('overall_metrics', {}).get('success_rate', 0.0),
            "avg_translation_quality": dataset_results.get('overall_metrics', {}).get('avg_overall_quality', 0.0)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(report), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced translation report saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ENHANCED NSM TRANSLATION WITH SURFACING SUMMARY")
    print("="*80)
    print(f"Test Examples: {len(translation_results)}")
    print(f"Successful: {sum(1 for r in translation_results if r['success'])}/{len(translation_results)}")
    if dataset_results:
        print(f"Dataset Success Rate: {dataset_results.get('overall_metrics', {}).get('success_rate', 0.0):.1%}")
        print(f"Average Quality: {dataset_results.get('overall_metrics', {}).get('avg_overall_quality', 0.0):.3f}")
        print("\nLanguage Pair Performance:")
        for pair, data in dataset_results.get('language_pairs', {}).items():
            print(f"  {pair}: {data['avg_quality']:.3f} avg quality ({data['translations']} translations)")
        print("\nPrimitive Performance:")
        for primitive, data in dataset_results.get('primitive_performance', {}).items():
            print(f"  {primitive}: {data['avg_quality']:.3f} avg quality ({data['translations']} translations)")
    print("="*80)


if __name__ == "__main__":
    main()

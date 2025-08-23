#!/usr/bin/env python3
"""
NSM Translation System Re-evaluation.

This script re-evaluates the NSM translation system using the improved
primitive detection to measure honest improvement in translation quality.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import time
from collections import defaultdict, Counter
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class NSMTranslationReEvaluator:
    """Re-evaluation of NSM translation system with improved primitive detection."""
    
    def __init__(self):
        """Initialize the NSM translation re-evaluator."""
        self.sbert_model = None
        self._load_semantic_model()
        
        # Test translation pairs for evaluation
        self.test_pairs = [
            # Basic translations
            {
                "source": "The cat is not on the mat",
                "target": "El gato no está en la alfombra",
                "expected_primitives": ["NOT", "AtLocation"]
            },
            {
                "source": "I do not like this weather",
                "target": "No me gusta este tiempo",
                "expected_primitives": ["NOT", "LIKE"]
            },
            {
                "source": "All children can play here",
                "target": "Todos los niños pueden jugar aquí",
                "expected_primitives": ["ALL", "CAN", "HERE"]
            },
            {
                "source": "Some people want to help",
                "target": "Algunas personas quieren ayudar",
                "expected_primitives": ["SOME", "WANT"]
            },
            {
                "source": "The weather is good today",
                "target": "El tiempo está bueno hoy",
                "expected_primitives": ["GOOD"]
            },
            
            # Complex translations
            {
                "source": "I think you are right about this",
                "target": "Creo que tienes razón sobre esto",
                "expected_primitives": ["THINK", "RIGHT"]
            },
            {
                "source": "The rain caused the flood",
                "target": "La lluvia causó la inundación",
                "expected_primitives": ["BECAUSE", "Causes"]
            },
            {
                "source": "This is similar to that",
                "target": "Esto es similar a eso",
                "expected_primitives": ["LIKE", "SimilarTo"]
            },
            {
                "source": "Many people understand this",
                "target": "Mucha gente entiende esto",
                "expected_primitives": ["MANY", "THINK"]
            },
            {
                "source": "She cannot believe the story",
                "target": "Ella no puede creer la historia",
                "expected_primitives": ["NOT", "CAN", "THINK"]
            },
            
            # Negation and modality
            {
                "source": "I cannot do this",
                "target": "No puedo hacer esto",
                "expected_primitives": ["NOT", "CAN", "DO"]
            },
            {
                "source": "She must not go there",
                "target": "Ella no debe ir allí",
                "expected_primitives": ["NOT", "CAN", "WHERE"]
            },
            {
                "source": "They should not eat that",
                "target": "No deberían comer eso",
                "expected_primitives": ["NOT", "CAN"]
            },
            {
                "source": "We might not understand",
                "target": "Podríamos no entender",
                "expected_primitives": ["NOT", "CAN", "THINK"]
            },
            {
                "source": "You will not believe this",
                "target": "No creerás esto",
                "expected_primitives": ["NOT", "THINK"]
            },
            
            # Quantifiers and evaluators
            {
                "source": "Few understand the problem",
                "target": "Pocos entienden el problema",
                "expected_primitives": ["FEW", "THINK"]
            },
            {
                "source": "Most agree with the decision",
                "target": "La mayoría está de acuerdo con la decisión",
                "expected_primitives": ["MOST", "LIKE"]
            },
            {
                "source": "Several options exist",
                "target": "Existen varias opciones",
                "expected_primitives": ["SOME"]
            },
            {
                "source": "Various solutions are available",
                "target": "Varias soluciones están disponibles",
                "expected_primitives": ["SOME"]
            },
            {
                "source": "Numerous attempts were made",
                "target": "Se hicieron numerosos intentos",
                "expected_primitives": ["MANY"]
            },
            
            # Mental predicates
            {
                "source": "I know the answer",
                "target": "Sé la respuesta",
                "expected_primitives": ["THINK"]
            },
            {
                "source": "She believes the story",
                "target": "Ella cree la historia",
                "expected_primitives": ["THINK"]
            },
            {
                "source": "They understand the concept",
                "target": "Ellos entienden el concepto",
                "expected_primitives": ["THINK"]
            },
            {
                "source": "We remember the event",
                "target": "Recordamos el evento",
                "expected_primitives": ["THINK"]
            },
            {
                "source": "He forgets the details",
                "target": "Él olvida los detalles",
                "expected_primitives": ["THINK"]
            },
            
            # Action predicates
            {
                "source": "I do the work",
                "target": "Hago el trabajo",
                "expected_primitives": ["DO"]
            },
            {
                "source": "She makes the decision",
                "target": "Ella toma la decisión",
                "expected_primitives": ["DO"]
            },
            {
                "source": "They take the action",
                "target": "Ellos toman la acción",
                "expected_primitives": ["DO"]
            },
            {
                "source": "We perform the task",
                "target": "Realizamos la tarea",
                "expected_primitives": ["DO"]
            },
            {
                "source": "He executes the plan",
                "target": "Él ejecuta el plan",
                "expected_primitives": ["DO"]
            },
            
            # Spatial and temporal
            {
                "source": "The book is on the table",
                "target": "El libro está en la mesa",
                "expected_primitives": ["AtLocation"]
            },
            {
                "source": "The car is in the garage",
                "target": "El carro está en el garaje",
                "expected_primitives": ["AtLocation"]
            },
            {
                "source": "The bird is under the tree",
                "target": "El pájaro está debajo del árbol",
                "expected_primitives": ["AtLocation"]
            },
            {
                "source": "The plane is above the clouds",
                "target": "El avión está sobre las nubes",
                "expected_primitives": ["AtLocation"]
            },
            {
                "source": "The fish is in the water",
                "target": "El pez está en el agua",
                "expected_primitives": ["AtLocation"]
            },
            
            # Causation and similarity
            {
                "source": "The fire destroyed the building",
                "target": "El fuego destruyó el edificio",
                "expected_primitives": ["Causes"]
            },
            {
                "source": "The medicine cured the disease",
                "target": "La medicina curó la enfermedad",
                "expected_primitives": ["Causes"]
            },
            {
                "source": "They are different from us",
                "target": "Ellos son diferentes de nosotros",
                "expected_primitives": ["LIKE", "SimilarTo"]
            },
            {
                "source": "The colors match perfectly",
                "target": "Los colores coinciden perfectamente",
                "expected_primitives": ["LIKE", "SimilarTo"]
            },
            {
                "source": "This is similar to that",
                "target": "Esto es similar a eso",
                "expected_primitives": ["LIKE", "SimilarTo"]
            }
        ]
    
    def _load_semantic_model(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for semantic similarity...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def detect_primitives_unified(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Detect primitives using the unified detection system."""
        detected_primitives = []
        text_lower = text.lower()
        
        # NSM Primitives detection patterns
        nsm_patterns = {
            'NOT': [r'\b(not|no|never|none|nobody|nothing|nowhere|neither|nor)\b'],
            'CAN': [r'\b(can|could|might|may|must|should|will|would|shall)\b'],
            'WANT': [r'\b(want|wish|desire|hope|need|require)\b'],
            'THINK': [r'\b(think|believe|know|understand|remember|forget)\b'],
            'DO': [r'\b(do|does|did|doing|done)\b'],
            'HAPPEN': [r'\b(happen|happens|happened|happening)\b'],
            'BECAUSE': [r'\b(because|since|as|due to|result in|lead to|cause|caused)\b'],
            'LIKE': [r'\b(like|similar|same|different|alike|unlike|resemble|match)\b'],
            'ALL': [r'\b(all|every|each|entire|complete|total)\b'],
            'SOME': [r'\b(some|any|several|various|numerous|countless)\b'],
            'MANY': [r'\b(many|much|numerous|countless|plenty|lots)\b'],
            'FEW': [r'\b(few|little|scarce|rare|minimal|limited)\b'],
            'GOOD': [r'\b(good|nice|terrible|wonderful|awful|excellent|horrible)\b'],
            'BAD': [r'\b(bad|terrible|awful|horrible|dreadful|atrocious)\b'],
            'BIG': [r'\b(big|large|little|huge|tiny|long|short|wide|narrow)\b'],
            'SMALL': [r'\b(small|little|tiny|minute|microscopic|minuscule)\b'],
            'HERE': [r'\b(here|this place|this location|where i am|where we are)\b'],
            'WHERE': [r'\b(where|location|place|position|site|spot)\b'],
            'NOW': [r'\b(now|then|when|before|after|during|while|since|until)\b'],
            'BEFORE': [r'\b(before|prior|earlier|previously|ahead|in front)\b'],
            'AFTER': [r'\b(after|later|subsequently|following|behind|afterward)\b'],
            'RIGHT': [r'\b(right|correct|true|accurate|proper|appropriate)\b']
        }
        
        # ConceptNet Relations detection patterns
        conceptnet_patterns = {
            'AtLocation': [r'\b(at|in|on|under|over|beside|near|far|inside|outside)\b'],
            'Causes': [r'\b(cause|caused|because|since|as|result|effect|consequence)\b'],
            'SimilarTo': [r'\b(similar|like|same|different|alike|unlike|resemble|match)\b'],
            'PartOf': [r'\b(part of|piece of|component of|element of|member of)\b'],
            'UsedFor': [r'\b(used for|purpose|function|serve|designed for|intended for)\b'],
            'CapableOf': [r'\b(can do|able to|capable of|know how to|skilled at)\b'],
            'Desires': [r'\b(want|desire|wish|hope|long for|yearn for)\b'],
            'HasProperty': [r'\b(has|property|characteristic|feature|attribute|quality)\b'],
            'IsA': [r'\b(is a|type of|kind of|sort of|category of|class of)\b']
        }
        
        # Detect NSM primitives
        for primitive, patterns in nsm_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    detected_primitives.append({
                        'primitive': primitive,
                        'type': 'NSM',
                        'confidence': 0.8,
                        'method': 'pattern',
                        'evidence': match.group(0),
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # Detect ConceptNet relations
        for relation, patterns in conceptnet_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    detected_primitives.append({
                        'primitive': relation,
                        'type': 'ConceptNet',
                        'confidence': 0.7,
                        'method': 'pattern',
                        'evidence': match.group(0),
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # Semantic detection (if model available)
        if self.sbert_model:
            semantic_primitives = self._detect_semantic_primitives(text, language)
            detected_primitives.extend(semantic_primitives)
        
        return {
            'text': text,
            'language': language,
            'detected_primitives': detected_primitives,
            'total_count': len(detected_primitives),
            'nsm_count': sum(1 for p in detected_primitives if p['type'] == 'NSM'),
            'conceptnet_count': sum(1 for p in detected_primitives if p['type'] == 'ConceptNet'),
            'average_confidence': np.mean([p['confidence'] for p in detected_primitives]) if detected_primitives else 0.0
        }
    
    def _detect_semantic_primitives(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Detect primitives using semantic similarity."""
        if not self.sbert_model:
            return []
        
        semantic_primitives = []
        
        # Semantic descriptions for key primitives
        primitive_descriptions = {
            'NOT': 'negation, not, no, never',
            'BECAUSE': 'causation, because, since, as',
            'LIKE': 'similarity, like, similar, same',
            'CAN': 'ability, can, able to, capable of',
            'WANT': 'desire, want, wish, desire',
            'ALL': 'universal quantifier, all, every, each',
            'SOME': 'existential quantifier, some, any',
            'MANY': 'large quantity, many, much, numerous',
            'FEW': 'small quantity, few, little, scarce',
            'GOOD': 'positive evaluation, good, nice, excellent',
            'BAD': 'negative evaluation, bad, terrible, awful',
            'BIG': 'large size, big, large, huge',
            'SMALL': 'small size, small, little, tiny',
            'AtLocation': 'spatial location, at, in, on, place',
            'Causes': 'causal relation, cause, because, result',
            'SimilarTo': 'similarity relation, similar, like, same',
            'PartOf': 'part-whole relation, part of, component',
            'CapableOf': 'ability relation, can, able to, capable',
            'Desires': 'desire relation, want, desire, wish',
            'HasProperty': 'property relation, has, property, characteristic',
            'IsA': 'taxonomic relation, is a, type of, kind of',
            'UsedFor': 'purpose relation, used for, purpose, function'
        }
        
        try:
            text_embedding = self.sbert_model.encode([text])[0]
            
            for primitive, description in primitive_descriptions.items():
                desc_embedding = self.sbert_model.encode([description])[0]
                
                similarity = np.dot(text_embedding, desc_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(desc_embedding)
                )
                
                if similarity > 0.3:  # Lower threshold for semantic detection
                    semantic_primitives.append({
                        'primitive': primitive,
                        'type': 'NSM' if primitive in ['NOT', 'CAN', 'MUST', 'WANT', 'THINK', 'KNOW', 'FEEL', 'SEE', 'HEAR', 'DO', 'HAPPEN', 'BECAUSE', 'LIKE', 'ALL', 'SOME', 'MANY', 'FEW', 'GOOD', 'BAD', 'BIG', 'SMALL', 'HERE', 'WHERE', 'NOW', 'BEFORE', 'AFTER', 'RIGHT'] else 'ConceptNet',
                        'confidence': float(similarity),
                        'method': 'semantic',
                        'evidence': description,
                        'start': -1,
                        'end': -1
                    })
        
        except Exception as e:
            logger.warning(f"Semantic detection failed: {e}")
        
        return semantic_primitives
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.sbert_model:
            return 0.0
        
        try:
            embeddings = self.sbert_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def evaluate_translation_quality(self) -> Dict[str, Any]:
        """Evaluate translation quality using improved primitive detection."""
        logger.info("Starting NSM translation quality evaluation...")
        
        results = []
        total_pairs = len(self.test_pairs)
        
        for i, pair in enumerate(self.test_pairs):
            logger.info(f"Processing pair {i+1}/{total_pairs}: {pair['source'][:50]}...")
            
            # Detect primitives in source and target
            source_detection = self.detect_primitives_unified(pair['source'], "en")
            target_detection = self.detect_primitives_unified(pair['target'], "es")
            
            # Calculate semantic similarity
            semantic_similarity = self.calculate_semantic_similarity(pair['source'], pair['target'])
            
            # Analyze primitive preservation
            source_primitives = set(p['primitive'] for p in source_detection['detected_primitives'])
            target_primitives = set(p['primitive'] for p in target_detection['detected_primitives'])
            expected_primitives = set(pair['expected_primitives'])
            
            # Calculate metrics
            primitive_preservation = len(source_primitives.intersection(target_primitives)) / len(source_primitives) if source_primitives else 0
            expected_coverage = len(source_primitives.intersection(expected_primitives)) / len(expected_primitives) if expected_primitives else 0
            target_coverage = len(target_primitives.intersection(expected_primitives)) / len(expected_primitives) if expected_primitives else 0
            
            result = {
                'pair_id': i + 1,
                'source': pair['source'],
                'target': pair['target'],
                'expected_primitives': list(expected_primitives),
                'source_detection': source_detection,
                'target_detection': target_detection,
                'semantic_similarity': semantic_similarity,
                'primitive_preservation': primitive_preservation,
                'expected_coverage': expected_coverage,
                'target_coverage': target_coverage,
                'source_primitives': list(source_primitives),
                'target_primitives': list(target_primitives)
            }
            
            results.append(result)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(results)
        
        return {
            'evaluation_timestamp': time.time(),
            'total_pairs': total_pairs,
            'results': results,
            'metrics': metrics
        }
    
    def _calculate_comprehensive_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics from translation results."""
        
        # Basic metrics
        avg_semantic_similarity = np.mean([r['semantic_similarity'] for r in results])
        avg_primitive_preservation = np.mean([r['primitive_preservation'] for r in results])
        avg_expected_coverage = np.mean([r['expected_coverage'] for r in results])
        avg_target_coverage = np.mean([r['target_coverage'] for r in results])
        
        # Quality distribution
        high_quality = sum(1 for r in results if r['semantic_similarity'] >= 0.8)
        medium_quality = sum(1 for r in results if 0.6 <= r['semantic_similarity'] < 0.8)
        low_quality = sum(1 for r in results if r['semantic_similarity'] < 0.6)
        
        # Primitive preservation distribution
        high_preservation = sum(1 for r in results if r['primitive_preservation'] >= 0.8)
        medium_preservation = sum(1 for r in results if 0.5 <= r['primitive_preservation'] < 0.8)
        low_preservation = sum(1 for r in results if r['primitive_preservation'] < 0.5)
        
        # Primitive frequency analysis
        all_source_primitives = []
        all_target_primitives = []
        for r in results:
            all_source_primitives.extend(r['source_primitives'])
            all_target_primitives.extend(r['target_primitives'])
        
        source_primitive_counts = Counter(all_source_primitives)
        target_primitive_counts = Counter(all_target_primitives)
        
        # Cross-language primitive alignment
        alignment_scores = []
        for r in results:
            source_set = set(r['source_primitives'])
            target_set = set(r['target_primitives'])
            if source_set:
                alignment_score = len(source_set.intersection(target_set)) / len(source_set)
                alignment_scores.append(alignment_score)
        
        avg_alignment_score = np.mean(alignment_scores) if alignment_scores else 0
        
        # Comparison with previous metrics (from sanity checks)
        previous_metrics = {
            'semantic_similarity': 0.785,  # Previous baseline
            'primitive_preservation': 0.0,  # Not measured before
            'translation_quality': 0.0  # Not measured before
        }
        
        improvement_metrics = {
            'semantic_similarity_improvement': avg_semantic_similarity - previous_metrics['semantic_similarity'],
            'semantic_similarity_improvement_percentage': ((avg_semantic_similarity - previous_metrics['semantic_similarity']) / previous_metrics['semantic_similarity']) * 100 if previous_metrics['semantic_similarity'] > 0 else 0,
            'primitive_preservation_new': avg_primitive_preservation,
            'alignment_score_new': avg_alignment_score
        }
        
        return {
            'avg_semantic_similarity': avg_semantic_similarity,
            'avg_primitive_preservation': avg_primitive_preservation,
            'avg_expected_coverage': avg_expected_coverage,
            'avg_target_coverage': avg_target_coverage,
            'avg_alignment_score': avg_alignment_score,
            'quality_distribution': {
                'high': high_quality,
                'medium': medium_quality,
                'low': low_quality
            },
            'preservation_distribution': {
                'high': high_preservation,
                'medium': medium_preservation,
                'low': low_preservation
            },
            'source_primitive_counts': dict(source_primitive_counts.most_common(20)),
            'target_primitive_counts': dict(target_primitive_counts.most_common(20)),
            'previous_metrics': previous_metrics,
            'improvement_metrics': improvement_metrics
        }


def main():
    """Main function to run NSM translation re-evaluation."""
    logger.info("Starting NSM translation system re-evaluation...")
    
    # Initialize evaluator
    evaluator = NSMTranslationReEvaluator()
    
    # Run evaluation
    evaluation_results = evaluator.evaluate_translation_quality()
    
    # Display results
    metrics = evaluation_results['metrics']
    
    print("="*80)
    print("NSM TRANSLATION SYSTEM RE-EVALUATION RESULTS")
    print("="*80)
    
    print(f"Total Translation Pairs: {evaluation_results['total_pairs']}")
    print(f"Average Semantic Similarity: {metrics['avg_semantic_similarity']:.3f}")
    print(f"Average Primitive Preservation: {metrics['avg_primitive_preservation']:.3f}")
    print(f"Average Expected Coverage: {metrics['avg_expected_coverage']:.3f}")
    print(f"Average Target Coverage: {metrics['avg_target_coverage']:.3f}")
    print(f"Average Alignment Score: {metrics['avg_alignment_score']:.3f}")
    
    print(f"\nQUALITY DISTRIBUTION:")
    for level, count in metrics['quality_distribution'].items():
        print(f"  {level}: {count}")
    
    print(f"\nPRESERVATION DISTRIBUTION:")
    for level, count in metrics['preservation_distribution'].items():
        print(f"  {level}: {count}")
    
    print(f"\nIMPROVEMENT COMPARISON:")
    print(f"Previous Semantic Similarity: {metrics['previous_metrics']['semantic_similarity']:.3f}")
    print(f"New Semantic Similarity: {metrics['avg_semantic_similarity']:.3f}")
    print(f"Improvement: {metrics['improvement_metrics']['semantic_similarity_improvement_percentage']:.1f}%")
    
    print(f"\nNEW METRICS:")
    print(f"Primitive Preservation: {metrics['improvement_metrics']['primitive_preservation_new']:.3f}")
    print(f"Cross-language Alignment: {metrics['improvement_metrics']['alignment_score_new']:.3f}")
    
    print(f"\nTOP SOURCE PRIMITIVES:")
    for primitive, count in list(metrics['source_primitive_counts'].items())[:10]:
        print(f"  {primitive}: {count}")
    
    print(f"\nTOP TARGET PRIMITIVES:")
    for primitive, count in list(metrics['target_primitive_counts'].items())[:10]:
        print(f"  {primitive}: {count}")
    
    # Save results
    output_path = "data/nsm_translation_re_evaluation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(evaluation_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"NSM translation re-evaluation results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("NSM translation system re-evaluation completed!")
    print("="*80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Metrics Pipeline Re-evaluation.

This script re-evaluates the metrics pipeline using the unified primitive detection
system to measure the honest improvement from the core logic bug fix.
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


class UnifiedMetricsCollector:
    """Re-evaluation of metrics using unified primitive detection."""
    
    def __init__(self):
        """Initialize the unified metrics collector."""
        self.sbert_model = None
        self._load_semantic_model()
        
        # Test texts for evaluation
        self.test_texts = [
            # Basic primitive detection
            "The cat is not on the mat",
            "I do not like this weather",
            "She does not work here",
            "All children can play here",
            "Some people want to help",
            "The weather is good today",
            "This book is big and heavy",
            "I think you are right",
            "The rain caused the flood",
            "This is similar to that",
            
            # Cross-language examples
            "El gato no está en la alfombra",  # Spanish
            "Le chat n'est pas sur le tapis",  # French
            "Yo no me gusta este tiempo",      # Spanish
            "Je n'aime pas ce temps",          # French
            "Todos los niños pueden jugar aquí", # Spanish
            "Tous les enfants peuvent jouer ici", # French
            
            # Complex examples
            "The cat is here on the mat",
            "I am here in the room",
            "The meeting is here at the office",
            "Where is the cat?",
            "Here is the answer",
            "The location is here",
            "The place is here",
            
            # Negation and modality
            "I cannot do this",
            "She must not go there",
            "They should not eat that",
            "We might not understand",
            "You will not believe this",
            "He would not say that",
            
            # Quantifiers and evaluators
            "Many people think this",
            "Few understand the problem",
            "Most agree with the decision",
            "Several options exist",
            "Various solutions are available",
            "Numerous attempts were made",
            
            # Mental predicates
            "I know the answer",
            "She believes the story",
            "They understand the concept",
            "We remember the event",
            "He forgets the details",
            "You learn the lesson",
            
            # Action predicates
            "I do the work",
            "She makes the decision",
            "They take the action",
            "We perform the task",
            "He executes the plan",
            "You complete the job",
            
            # Spatial and temporal
            "The book is on the table",
            "The car is in the garage",
            "The bird is under the tree",
            "The plane is above the clouds",
            "The fish is in the water",
            "The key is on the table",
            
            # Causation and similarity
            "The rain caused the flood",
            "The fire destroyed the building",
            "The medicine cured the disease",
            "This is similar to that",
            "They are different from us",
            "The colors match perfectly",
            
            # Complex combinations
            "I do not think that many people understand this complex situation",
            "She cannot believe that all children should not play here",
            "They might not know where the big book is located",
            "We should not forget that some things are similar to others",
            "You will not find that most solutions are easy to implement"
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
            'AFTER': [r'\b(after|later|subsequently|following|behind|afterward)\b']
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
                        'type': 'NSM' if primitive in ['NOT', 'CAN', 'MUST', 'WANT', 'THINK', 'KNOW', 'FEEL', 'SEE', 'HEAR', 'DO', 'HAPPEN', 'BECAUSE', 'LIKE', 'ALL', 'SOME', 'MANY', 'FEW', 'GOOD', 'BAD', 'BIG', 'SMALL', 'HERE', 'WHERE', 'NOW', 'BEFORE', 'AFTER'] else 'ConceptNet',
                        'confidence': float(similarity),
                        'method': 'semantic',
                        'evidence': description,
                        'start': -1,
                        'end': -1
                    })
        
        except Exception as e:
            logger.warning(f"Semantic detection failed: {e}")
        
        return semantic_primitives
    
    def evaluate_metrics(self) -> Dict[str, Any]:
        """Evaluate comprehensive metrics using unified detection."""
        logger.info("Starting unified metrics evaluation...")
        
        results = []
        total_texts = len(self.test_texts)
        
        for i, text in enumerate(self.test_texts):
            logger.info(f"Processing text {i+1}/{total_texts}: {text[:50]}...")
            
            # Detect primitives using unified system
            detection_result = self.detect_primitives_unified(text)
            results.append(detection_result)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(results)
        
        return {
            'evaluation_timestamp': time.time(),
            'total_texts': total_texts,
            'results': results,
            'metrics': metrics
        }
    
    def _calculate_comprehensive_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics from detection results."""
        
        # Basic counts
        total_detections = sum(r['total_count'] for r in results)
        total_nsm = sum(r['nsm_count'] for r in results)
        total_conceptnet = sum(r['conceptnet_count'] for r in results)
        
        # Detection rates
        texts_with_detections = sum(1 for r in results if r['total_count'] > 0)
        detection_rate = texts_with_detections / len(results) if results else 0
        
        # Average detections per text
        avg_detections_per_text = total_detections / len(results) if results else 0
        avg_nsm_per_text = total_nsm / len(results) if results else 0
        avg_conceptnet_per_text = total_conceptnet / len(results) if results else 0
        
        # Confidence analysis
        all_confidences = []
        for r in results:
            all_confidences.extend([p['confidence'] for p in r['detected_primitives']])
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        confidence_distribution = {
            'high': sum(1 for c in all_confidences if c >= 0.8),
            'medium': sum(1 for c in all_confidences if 0.6 <= c < 0.8),
            'low': sum(1 for c in all_confidences if c < 0.6)
        }
        
        # Method analysis
        method_counts = defaultdict(int)
        for r in results:
            for p in r['detected_primitives']:
                method_counts[p['method']] += 1
        
        # Primitive frequency analysis
        primitive_counts = Counter()
        for r in results:
            for p in r['detected_primitives']:
                primitive_counts[p['primitive']] += 1
        
        # Cross-language analysis
        english_results = [r for r in results if r['language'] == 'en']
        spanish_results = [r for r in results if r['language'] == 'es']
        french_results = [r for r in results if r['language'] == 'fr']
        
        cross_language_metrics = {
            'english_detection_rate': len([r for r in english_results if r['total_count'] > 0]) / len(english_results) if english_results else 0,
            'spanish_detection_rate': len([r for r in spanish_results if r['total_count'] > 0]) / len(spanish_results) if spanish_results else 0,
            'french_detection_rate': len([r for r in french_results if r['total_count'] > 0]) / len(french_results) if french_results else 0
        }
        
        # Comparison with previous metrics
        previous_metrics = {
            'detection_rate': 0.347,  # Previous 34.7%
            'avg_detections_per_text': 1.2,
            'total_detections': 0
        }
        
        improvement_metrics = {
            'detection_rate_improvement': detection_rate - previous_metrics['detection_rate'],
            'detection_rate_improvement_percentage': ((detection_rate - previous_metrics['detection_rate']) / previous_metrics['detection_rate']) * 100 if previous_metrics['detection_rate'] > 0 else 0,
            'avg_detections_improvement': avg_detections_per_text - previous_metrics['avg_detections_per_text'],
            'avg_detections_improvement_percentage': ((avg_detections_per_text - previous_metrics['avg_detections_per_text']) / previous_metrics['avg_detections_per_text']) * 100 if previous_metrics['avg_detections_per_text'] > 0 else 0
        }
        
        return {
            'detection_rate': detection_rate,
            'avg_detections_per_text': avg_detections_per_text,
            'avg_nsm_per_text': avg_nsm_per_text,
            'avg_conceptnet_per_text': avg_conceptnet_per_text,
            'total_detections': total_detections,
            'total_nsm': total_nsm,
            'total_conceptnet': total_conceptnet,
            'avg_confidence': avg_confidence,
            'confidence_distribution': confidence_distribution,
            'method_distribution': dict(method_counts),
            'top_primitives': dict(primitive_counts.most_common(20)),
            'cross_language_metrics': cross_language_metrics,
            'previous_metrics': previous_metrics,
            'improvement_metrics': improvement_metrics
        }


def main():
    """Main function to run unified metrics re-evaluation."""
    logger.info("Starting unified metrics pipeline re-evaluation...")
    
    # Initialize collector
    collector = UnifiedMetricsCollector()
    
    # Run evaluation
    evaluation_results = collector.evaluate_metrics()
    
    # Display results
    metrics = evaluation_results['metrics']
    
    print("="*80)
    print("UNIFIED METRICS PIPELINE RE-EVALUATION RESULTS")
    print("="*80)
    
    print(f"Total Texts Evaluated: {evaluation_results['total_texts']}")
    print(f"Detection Rate: {metrics['detection_rate']:.1%}")
    print(f"Average Detections per Text: {metrics['avg_detections_per_text']:.2f}")
    print(f"Total Detections: {metrics['total_detections']}")
    print(f"NSM Detections: {metrics['total_nsm']}")
    print(f"ConceptNet Detections: {metrics['total_conceptnet']}")
    print(f"Average Confidence: {metrics['avg_confidence']:.3f}")
    
    print(f"\nIMPROVEMENT COMPARISON:")
    print(f"Previous Detection Rate: {metrics['previous_metrics']['detection_rate']:.1%}")
    print(f"New Detection Rate: {metrics['detection_rate']:.1%}")
    print(f"Improvement: +{metrics['improvement_metrics']['detection_rate_improvement_percentage']:.1f}%")
    
    print(f"\nPrevious Avg Detections: {metrics['previous_metrics']['avg_detections_per_text']:.2f}")
    print(f"New Avg Detections: {metrics['avg_detections_per_text']:.2f}")
    print(f"Improvement: +{metrics['improvement_metrics']['avg_detections_improvement_percentage']:.1f}%")
    
    print(f"\nCONFIDENCE DISTRIBUTION:")
    for level, count in metrics['confidence_distribution'].items():
        print(f"  {level}: {count}")
    
    print(f"\nMETHOD DISTRIBUTION:")
    for method, count in metrics['method_distribution'].items():
        print(f"  {method}: {count}")
    
    print(f"\nTOP PRIMITIVES:")
    for primitive, count in list(metrics['top_primitives'].items())[:10]:
        print(f"  {primitive}: {count}")
    
    print(f"\nCROSS-LANGUAGE METRICS:")
    for lang, rate in metrics['cross_language_metrics'].items():
        print(f"  {lang}: {rate:.1%}")
    
    # Save results
    output_path = "data/unified_metrics_re_evaluation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(evaluation_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Unified metrics re-evaluation results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Unified metrics pipeline re-evaluation completed!")
    print("="*80)


if __name__ == "__main__":
    main()

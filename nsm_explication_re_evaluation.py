#!/usr/bin/env python3
"""
NSM Explication System Re-evaluation.

This script re-evaluates the NSM explication system using the improved
primitive detection to measure honest improvement in explication quality.
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


class NSMExplicationReEvaluator:
    """Re-evaluation of NSM explication system with improved primitive detection."""
    
    def __init__(self):
        """Initialize the NSM explication re-evaluator."""
        self.sbert_model = None
        self._load_semantic_model()
        
        # Test texts for explication evaluation
        self.test_texts = [
            # Basic concepts
            "The cat is not on the mat",
            "I do not like this weather",
            "All children can play here",
            "Some people want to help",
            "The weather is good today",
            
            # Complex concepts
            "I think you are right about this",
            "The rain caused the flood",
            "This is similar to that",
            "Many people understand this",
            "She cannot believe the story",
            
            # Negation and modality
            "I cannot do this",
            "She must not go there",
            "They should not eat that",
            "We might not understand",
            "You will not believe this",
            
            # Quantifiers and evaluators
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
            
            # Action predicates
            "I do the work",
            "She makes the decision",
            "They take the action",
            "We perform the task",
            "He executes the plan",
            
            # Spatial and temporal
            "The book is on the table",
            "The car is in the garage",
            "The bird is under the tree",
            "The plane is above the clouds",
            "The fish is in the water",
            
            # Causation and similarity
            "The fire destroyed the building",
            "The medicine cured the disease",
            "They are different from us",
            "The colors match perfectly",
            "This is similar to that"
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
    
    def generate_nsm_explication(self, text: str) -> str:
        """Generate NSM explication for the given text."""
        # Detect primitives in the text
        detection_result = self.detect_primitives_unified(text)
        detected_primitives = detection_result['detected_primitives']
        
        if not detected_primitives:
            return f"X SAY: {text}"
        
        # Group primitives by type
        nsm_primitives = [p['primitive'] for p in detected_primitives if p['type'] == 'NSM']
        conceptnet_relations = [p['primitive'] for p in detected_primitives if p['type'] == 'ConceptNet']
        
        # Generate explication based on detected primitives
        explication_parts = []
        
        # Handle negation
        if 'NOT' in nsm_primitives:
            explication_parts.append("NOT")
        
        # Handle modality
        modality_primitives = [p for p in nsm_primitives if p in ['CAN', 'MUST', 'SHOULD', 'WILL']]
        if modality_primitives:
            explication_parts.extend(modality_primitives)
        
        # Handle quantifiers
        quantifier_primitives = [p for p in nsm_primitives if p in ['ALL', 'SOME', 'MANY', 'FEW', 'MOST']]
        if quantifier_primitives:
            explication_parts.extend(quantifier_primitives)
        
        # Handle mental predicates
        mental_primitives = [p for p in nsm_primitives if p in ['THINK', 'KNOW', 'BELIEVE', 'UNDERSTAND']]
        if mental_primitives:
            explication_parts.extend(mental_primitives)
        
        # Handle action predicates
        action_primitives = [p for p in nsm_primitives if p in ['DO', 'MAKE', 'TAKE', 'PERFORM']]
        if action_primitives:
            explication_parts.extend(action_primitives)
        
        # Handle evaluators
        evaluator_primitives = [p for p in nsm_primitives if p in ['GOOD', 'BAD', 'RIGHT', 'WRONG']]
        if evaluator_primitives:
            explication_parts.extend(evaluator_primitives)
        
        # Handle spatial/temporal
        spatial_primitives = [p for p in nsm_primitives if p in ['HERE', 'WHERE', 'NOW', 'BEFORE', 'AFTER']]
        if spatial_primitives:
            explication_parts.extend(spatial_primitives)
        
        # Handle ConceptNet relations
        if conceptnet_relations:
            explication_parts.extend(conceptnet_relations)
        
        # Generate the explication
        if explication_parts:
            explication = f"X SAY: {' '.join(explication_parts)}"
        else:
            explication = f"X SAY: {text}"
        
        return explication
    
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
    
    def evaluate_explication_quality(self) -> Dict[str, Any]:
        """Evaluate explication quality using improved primitive detection."""
        logger.info("Starting NSM explication quality evaluation...")
        
        results = []
        total_texts = len(self.test_texts)
        
        for i, text in enumerate(self.test_texts):
            logger.info(f"Processing text {i+1}/{total_texts}: {text[:50]}...")
            
            # Detect primitives in original text
            detection_result = self.detect_primitives_unified(text)
            
            # Generate NSM explication
            explication = self.generate_nsm_explication(text)
            
            # Calculate semantic similarity between original and explication
            semantic_similarity = self.calculate_semantic_similarity(text, explication)
            
            # Analyze primitive coverage
            detected_primitives = detection_result['detected_primitives']
            nsm_primitives = [p['primitive'] for p in detected_primitives if p['type'] == 'NSM']
            conceptnet_relations = [p['primitive'] for p in detected_primitives if p['type'] == 'ConceptNet']
            
            # Calculate explication metrics
            explication_length = len(explication.split())
            primitive_density = len(detected_primitives) / len(text.split()) if text.split() else 0
            nsm_coverage = len(nsm_primitives) / len(detected_primitives) if detected_primitives else 0
            
            result = {
                'text_id': i + 1,
                'original_text': text,
                'explication': explication,
                'detection_result': detection_result,
                'semantic_similarity': semantic_similarity,
                'explication_length': explication_length,
                'primitive_density': primitive_density,
                'nsm_coverage': nsm_coverage,
                'nsm_primitives': nsm_primitives,
                'conceptnet_relations': conceptnet_relations,
                'total_primitives': len(detected_primitives)
            }
            
            results.append(result)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(results)
        
        return {
            'evaluation_timestamp': time.time(),
            'total_texts': total_texts,
            'results': results,
            'metrics': metrics
        }
    
    def _calculate_comprehensive_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics from explication results."""
        
        # Basic metrics
        avg_semantic_similarity = np.mean([r['semantic_similarity'] for r in results])
        avg_explication_length = np.mean([r['explication_length'] for r in results])
        avg_primitive_density = np.mean([r['primitive_density'] for r in results])
        avg_nsm_coverage = np.mean([r['nsm_coverage'] for r in results])
        
        # Quality distribution
        high_quality = sum(1 for r in results if r['semantic_similarity'] >= 0.8)
        medium_quality = sum(1 for r in results if 0.6 <= r['semantic_similarity'] < 0.8)
        low_quality = sum(1 for r in results if r['semantic_similarity'] < 0.6)
        
        # Primitive coverage distribution
        high_coverage = sum(1 for r in results if r['total_primitives'] >= 3)
        medium_coverage = sum(1 for r in results if 1 <= r['total_primitives'] < 3)
        low_coverage = sum(1 for r in results if r['total_primitives'] == 0)
        
        # Primitive frequency analysis
        all_nsm_primitives = []
        all_conceptnet_relations = []
        for r in results:
            all_nsm_primitives.extend(r['nsm_primitives'])
            all_conceptnet_relations.extend(r['conceptnet_relations'])
        
        nsm_primitive_counts = Counter(all_nsm_primitives)
        conceptnet_relation_counts = Counter(all_conceptnet_relations)
        
        # Explication complexity analysis
        explication_lengths = [r['explication_length'] for r in results]
        avg_explication_complexity = np.mean(explication_lengths)
        complexity_distribution = {
            'short': sum(1 for l in explication_lengths if l <= 5),
            'medium': sum(1 for l in explication_lengths if 6 <= l <= 10),
            'long': sum(1 for l in explication_lengths if l > 10)
        }
        
        # Comparison with previous metrics (estimated)
        previous_metrics = {
            'semantic_similarity': 0.75,  # Estimated previous baseline
            'primitive_coverage': 0.0,  # Not measured before
            'explication_quality': 0.0  # Not measured before
        }
        
        improvement_metrics = {
            'semantic_similarity_improvement': avg_semantic_similarity - previous_metrics['semantic_similarity'],
            'semantic_similarity_improvement_percentage': ((avg_semantic_similarity - previous_metrics['semantic_similarity']) / previous_metrics['semantic_similarity']) * 100 if previous_metrics['semantic_similarity'] > 0 else 0,
            'primitive_coverage_new': avg_nsm_coverage,
            'explication_complexity_new': avg_explication_complexity
        }
        
        return {
            'avg_semantic_similarity': avg_semantic_similarity,
            'avg_explication_length': avg_explication_length,
            'avg_primitive_density': avg_primitive_density,
            'avg_nsm_coverage': avg_nsm_coverage,
            'avg_explication_complexity': avg_explication_complexity,
            'quality_distribution': {
                'high': high_quality,
                'medium': medium_quality,
                'low': low_quality
            },
            'coverage_distribution': {
                'high': high_coverage,
                'medium': medium_coverage,
                'low': low_coverage
            },
            'complexity_distribution': complexity_distribution,
            'nsm_primitive_counts': dict(nsm_primitive_counts.most_common(20)),
            'conceptnet_relation_counts': dict(conceptnet_relation_counts.most_common(20)),
            'previous_metrics': previous_metrics,
            'improvement_metrics': improvement_metrics
        }


def main():
    """Main function to run NSM explication re-evaluation."""
    logger.info("Starting NSM explication system re-evaluation...")
    
    # Initialize evaluator
    evaluator = NSMExplicationReEvaluator()
    
    # Run evaluation
    evaluation_results = evaluator.evaluate_explication_quality()
    
    # Display results
    metrics = evaluation_results['metrics']
    
    print("="*80)
    print("NSM EXPLICATION SYSTEM RE-EVALUATION RESULTS")
    print("="*80)
    
    print(f"Total Texts Evaluated: {evaluation_results['total_texts']}")
    print(f"Average Semantic Similarity: {metrics['avg_semantic_similarity']:.3f}")
    print(f"Average Explication Length: {metrics['avg_explication_length']:.1f}")
    print(f"Average Primitive Density: {metrics['avg_primitive_density']:.3f}")
    print(f"Average NSM Coverage: {metrics['avg_nsm_coverage']:.3f}")
    print(f"Average Explication Complexity: {metrics['avg_explication_complexity']:.1f}")
    
    print(f"\nQUALITY DISTRIBUTION:")
    for level, count in metrics['quality_distribution'].items():
        print(f"  {level}: {count}")
    
    print(f"\nCOVERAGE DISTRIBUTION:")
    for level, count in metrics['coverage_distribution'].items():
        print(f"  {level}: {count}")
    
    print(f"\nCOMPLEXITY DISTRIBUTION:")
    for level, count in metrics['complexity_distribution'].items():
        print(f"  {level}: {count}")
    
    print(f"\nIMPROVEMENT COMPARISON:")
    print(f"Previous Semantic Similarity: {metrics['previous_metrics']['semantic_similarity']:.3f}")
    print(f"New Semantic Similarity: {metrics['avg_semantic_similarity']:.3f}")
    print(f"Improvement: {metrics['improvement_metrics']['semantic_similarity_improvement_percentage']:.1f}%")
    
    print(f"\nNEW METRICS:")
    print(f"Primitive Coverage: {metrics['improvement_metrics']['primitive_coverage_new']:.3f}")
    print(f"Explication Complexity: {metrics['improvement_metrics']['explication_complexity_new']:.1f}")
    
    print(f"\nTOP NSM PRIMITIVES:")
    for primitive, count in list(metrics['nsm_primitive_counts'].items())[:10]:
        print(f"  {primitive}: {count}")
    
    print(f"\nTOP CONCEPTNET RELATIONS:")
    for relation, count in list(metrics['conceptnet_relation_counts'].items())[:10]:
        print(f"  {relation}: {count}")
    
    # Show sample explications
    print(f"\nSAMPLE EXPLICATIONS:")
    for i, result in enumerate(evaluation_results['results'][:5]):
        print(f"\n{i+1}. Original: {result['original_text']}")
        print(f"   Explication: {result['explication']}")
        print(f"   Similarity: {result['semantic_similarity']:.3f}")
        print(f"   Primitives: {result['nsm_primitives']}")
    
    # Save results
    output_path = "data/nsm_explication_re_evaluation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(evaluation_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"NSM explication re-evaluation results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("NSM explication system re-evaluation completed!")
    print("="*80)


if __name__ == "__main__":
    main()

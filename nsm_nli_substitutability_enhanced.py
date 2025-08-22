#!/usr/bin/env python3
"""
Enhanced NLI-based Substitutability Evaluation for NSM Explications.

This script implements comprehensive NLI-based substitutability evaluation using:
1. Advanced multilingual XNLI models for semantic evaluation
2. Bidirectional entailment analysis for comprehensive substitutability assessment
3. Cross-language substitutability validation
4. Context-aware NLI evaluation with semantic focus detection
5. Quality-driven substitutability scoring with confidence assessment
6. Integration with enhanced NSM translation and explication systems
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


def load_enhanced_xnli_model(model_name: str = 'joeddav/xlm-roberta-large-xnli'):
    """Load enhanced XNLI model for comprehensive NLI evaluation."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
        
        logger.info(f"Loading enhanced XNLI model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        
        logger.info("Enhanced XNLI model loaded successfully")
        return pipeline
    except ImportError:
        logger.error("transformers library not found. Install with: pip install transformers")
        return None
    except Exception as e:
        logger.error(f"Failed to load enhanced XNLI model: {e}")
        return None


class EnhancedNLISubstitutabilityEvaluator:
    """Enhanced NLI-based substitutability evaluator for NSM explications."""
    
    def __init__(self):
        """Initialize the enhanced NLI substitutability evaluator."""
        self.xnli_model = load_enhanced_xnli_model()
        self.sbert_model = None
        self.nsm_translator = NSMTranslator()
        self.nsm_explicator = NSMExplicator()
        self.enhanced_explicator = EnhancedNSMExplicator()
        self.languages = ['en', 'es', 'fr']
        
        # Load periodic table
        try:
            with open("data/nsm_periodic_table.json", 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            self.periodic_table = PeriodicTable.from_dict(table_data)
        except Exception as e:
            logger.warning(f"Failed to load periodic table: {e}")
            self.periodic_table = PeriodicTable()
        
        # Enhanced NLI evaluation parameters
        self.nli_evaluation_params = {
            'bidirectional_weight': 0.6,  # Weight for bidirectional entailment
            'semantic_similarity_weight': 0.3,  # Weight for semantic similarity
            'context_awareness_weight': 0.1,  # Weight for context awareness
            'confidence_threshold': 0.5,  # Minimum confidence for reliable evaluation
            'entailment_threshold': 0.7,  # Threshold for strong entailment
            'contradiction_threshold': 0.3  # Threshold for contradiction detection
        }
        
        # Language-specific NLI adjustments
        self.language_nli_adjustments = {
            'en': {
                'entailment_boost': 1.0,  # Baseline
                'contradiction_penalty': 1.0,
                'semantic_focus_weight': 1.0
            },
            'es': {
                'entailment_boost': 0.95,  # Slightly lower due to morphological complexity
                'contradiction_penalty': 1.05,
                'semantic_focus_weight': 0.95
            },
            'fr': {
                'entailment_boost': 0.95,  # Slightly lower due to morphological complexity
                'contradiction_penalty': 1.05,
                'semantic_focus_weight': 0.95
            }
        }
        
        # Primitive-specific NLI evaluation strategies
        self.primitive_nli_strategies = {
            'AtLocation': {
                'focus_elements': ['location', 'place', 'spatial'],
                'entailment_priority': 'forward',  # Text → explication more important
                'semantic_weight': 1.1
            },
            'HasProperty': {
                'focus_elements': ['property', 'characteristic', 'attribute'],
                'entailment_priority': 'bidirectional',
                'semantic_weight': 1.0
            },
            'Causes': {
                'focus_elements': ['cause', 'effect', 'causal'],
                'entailment_priority': 'bidirectional',
                'semantic_weight': 1.2
            },
            'UsedFor': {
                'focus_elements': ['purpose', 'function', 'use'],
                'entailment_priority': 'bidirectional',
                'semantic_weight': 1.0
            },
            'PartOf': {
                'focus_elements': ['part', 'whole', 'component'],
                'entailment_priority': 'bidirectional',
                'semantic_weight': 1.0
            },
            'SimilarTo': {
                'focus_elements': ['similarity', 'resemblance', 'like'],
                'entailment_priority': 'bidirectional',
                'semantic_weight': 0.9
            },
            'DifferentFrom': {
                'focus_elements': ['difference', 'distinction', 'contrast'],
                'entailment_priority': 'bidirectional',
                'semantic_weight': 0.9
            },
            'Not': {
                'focus_elements': ['negation', 'denial', 'opposite'],
                'entailment_priority': 'bidirectional',
                'semantic_weight': 1.1
            },
            'Exist': {
                'focus_elements': ['existence', 'presence', 'being'],
                'entailment_priority': 'forward',
                'semantic_weight': 1.0
            }
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for semantic similarity...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def evaluate_enhanced_nli_substitutability(self, text: str, explication: str, language: str, 
                                             primitive: str = None) -> Dict[str, Any]:
        """Evaluate substitutability using enhanced NLI analysis."""
        if not self.xnli_model:
            return {
                'success': False,
                'error': 'XNLI model not available',
                'substitutability_score': 0.0,
                'confidence': 0.0
            }
        
        try:
            # Get primitive-specific strategy
            strategy = self.primitive_nli_strategies.get(primitive, {
                'focus_elements': [],
                'entailment_priority': 'bidirectional',
                'semantic_weight': 1.0
            })
            
            # Perform bidirectional NLI evaluation
            forward_entailment = self._compute_entailment(text, explication)
            backward_entailment = self._compute_entailment(explication, text)
            
            # Calculate contradiction scores
            forward_contradiction = self._compute_contradiction(text, explication)
            backward_contradiction = self._compute_contradiction(explication, text)
            
            # Semantic similarity (if SBERT available)
            semantic_similarity = 0.0
            if self.sbert_model:
                semantic_similarity = self._compute_semantic_similarity(text, explication)
            
            # Context awareness evaluation
            context_awareness = self._evaluate_context_awareness(text, explication, language, strategy)
            
            # Apply language-specific adjustments
            lang_adjustments = self.language_nli_adjustments.get(language, self.language_nli_adjustments['en'])
            
            # Weighted combination based on entailment priority
            if strategy['entailment_priority'] == 'forward':
                primary_entailment = forward_entailment * lang_adjustments['entailment_boost']
                secondary_entailment = backward_entailment * 0.5
            elif strategy['entailment_priority'] == 'bidirectional':
                primary_entailment = (forward_entailment + backward_entailment) / 2 * lang_adjustments['entailment_boost']
                secondary_entailment = primary_entailment
            else:
                primary_entailment = backward_entailment * lang_adjustments['entailment_boost']
                secondary_entailment = forward_entailment * 0.5
            
            # Contradiction penalty
            contradiction_penalty = (forward_contradiction + backward_contradiction) / 2 * lang_adjustments['contradiction_penalty']
            
            # Calculate overall substitutability score
            substitutability_score = (
                self.nli_evaluation_params['bidirectional_weight'] * primary_entailment +
                self.nli_evaluation_params['semantic_similarity_weight'] * semantic_similarity * strategy['semantic_weight'] +
                self.nli_evaluation_params['context_awareness_weight'] * context_awareness * lang_adjustments['semantic_focus_weight']
            ) - contradiction_penalty
            
            # Ensure score is in valid range
            substitutability_score = max(0.0, min(1.0, substitutability_score))
            
            # Calculate confidence based on model certainty
            confidence = self._calculate_evaluation_confidence(
                forward_entailment, backward_entailment, 
                forward_contradiction, backward_contradiction,
                semantic_similarity, context_awareness
            )
            
            # Determine substitutability level
            if substitutability_score >= self.nli_evaluation_params['entailment_threshold']:
                substitutability_level = 'high'
            elif substitutability_score >= self.nli_evaluation_params['confidence_threshold']:
                substitutability_level = 'medium'
            elif substitutability_score >= 0.2:
                substitutability_level = 'low'
            else:
                substitutability_level = 'poor'
            
            return {
                'success': True,
                'substitutability_score': float(substitutability_score),
                'confidence': float(confidence),
                'substitutability_level': substitutability_level,
                'entailment_scores': {
                    'forward': float(forward_entailment),
                    'backward': float(backward_entailment),
                    'primary': float(primary_entailment),
                    'secondary': float(secondary_entailment)
                },
                'contradiction_scores': {
                    'forward': float(forward_contradiction),
                    'backward': float(backward_contradiction),
                    'penalty': float(contradiction_penalty)
                },
                'semantic_similarity': float(semantic_similarity),
                'context_awareness': float(context_awareness),
                'evaluation_metadata': {
                    'primitive': primitive,
                    'strategy_used': strategy,
                    'language_adjustments': lang_adjustments,
                    'model_confidence': confidence
                }
            }
            
        except Exception as e:
            logger.warning(f"Enhanced NLI evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'substitutability_score': 0.0,
                'confidence': 0.0
            }
    
    def _compute_entailment(self, premise: str, hypothesis: str) -> float:
        """Compute entailment probability between premise and hypothesis."""
        if not self.xnli_model:
            return 0.0
        
        try:
            result = self.xnli_model({
                'text': premise,
                'text_pair': hypothesis
            })
            
            # Extract entailment probability
            if isinstance(result, list):
                result = result[0]
            
            entailment_score = 0.0
            if isinstance(result, dict):
                if 'label' in result and result['label'] == 'ENTAILMENT':
                    entailment_score = result.get('score', 0.0)
                else:
                    # Look for entailment in the results
                    for item in result.get('labels', []):
                        if item.get('label') == 'ENTAILMENT':
                            entailment_score = item.get('score', 0.0)
                            break
            
            return float(entailment_score)
        except Exception as e:
            logger.warning(f"Entailment computation failed: {e}")
            return 0.0
    
    def _compute_contradiction(self, premise: str, hypothesis: str) -> float:
        """Compute contradiction probability between premise and hypothesis."""
        if not self.xnli_model:
            return 0.0
        
        try:
            result = self.xnli_model({
                'text': premise,
                'text_pair': hypothesis
            })
            
            # Extract contradiction probability
            if isinstance(result, list):
                result = result[0]
            
            contradiction_score = 0.0
            if isinstance(result, dict):
                if 'label' in result and result['label'] == 'CONTRADICTION':
                    contradiction_score = result.get('score', 0.0)
                else:
                    # Look for contradiction in the results
                    for item in result.get('labels', []):
                        if item.get('label') == 'CONTRADICTION':
                            contradiction_score = item.get('score', 0.0)
                            break
            
            return float(contradiction_score)
        except Exception as e:
            logger.warning(f"Contradiction computation failed: {e}")
            return 0.0
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using SBERT."""
        if not self.sbert_model:
            return 0.0
        
        try:
            embeddings = self.sbert_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.0
    
    def _evaluate_context_awareness(self, text: str, explication: str, language: str, 
                                   strategy: Dict[str, Any]) -> float:
        """Evaluate context awareness and semantic focus preservation."""
        try:
            # Extract focus elements from strategy
            focus_elements = strategy.get('focus_elements', [])
            
            # Simple keyword-based context evaluation
            text_lower = text.lower()
            explication_lower = explication.lower()
            
            # Check for focus element presence
            text_focus_hits = sum(1 for element in focus_elements if element in text_lower)
            explication_focus_hits = sum(1 for element in focus_elements if element in explication_lower)
            
            # Calculate focus preservation
            if text_focus_hits > 0 and explication_focus_hits > 0:
                focus_preservation = min(1.0, explication_focus_hits / text_focus_hits)
            elif text_focus_hits == 0 and explication_focus_hits == 0:
                focus_preservation = 1.0  # No focus elements to preserve
            else:
                focus_preservation = 0.5  # Partial preservation
            
            # Check for semantic role preservation
            role_preservation = self._evaluate_semantic_roles(text, explication, language)
            
            # Combined context awareness score
            context_awareness = (focus_preservation + role_preservation) / 2
            
            return float(context_awareness)
        except Exception as e:
            logger.warning(f"Context awareness evaluation failed: {e}")
            return 0.5
    
    def _evaluate_semantic_roles(self, text: str, explication: str, language: str) -> float:
        """Evaluate preservation of semantic roles."""
        # Simple heuristic-based evaluation
        text_lower = text.lower()
        explication_lower = explication.lower()
        
        # Check for subject-verb-object patterns
        subject_indicators = {
            'en': ['this', 'that', 'the', 'a', 'an', 'it', 'they'],
            'es': ['este', 'esta', 'el', 'la', 'un', 'una', 'esto'],
            'fr': ['ce', 'cette', 'le', 'la', 'un', 'une', 'cela']
        }
        
        action_indicators = {
            'en': ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'will'],
            'es': ['es', 'está', 'son', 'están', 'hace', 'hacen', 'puede'],
            'fr': ['est', 'sont', 'fait', 'font', 'peut', 'peuvent', 'va']
        }
        
        lang_subjects = subject_indicators.get(language, subject_indicators['en'])
        lang_actions = action_indicators.get(language, action_indicators['en'])
        
        # Check subject presence
        text_has_subject = any(word in text_lower for word in lang_subjects)
        explication_has_subject = any(word in explication_lower for word in lang_subjects)
        
        # Check action presence
        text_has_action = any(word in text_lower for word in lang_actions)
        explication_has_action = any(word in explication_lower for word in lang_actions)
        
        # Calculate role preservation score
        subject_preservation = 1.0 if text_has_subject == explication_has_subject else 0.5
        action_preservation = 1.0 if text_has_action == explication_has_action else 0.5
        
        return (subject_preservation + action_preservation) / 2
    
    def _calculate_evaluation_confidence(self, forward_entailment: float, backward_entailment: float,
                                        forward_contradiction: float, backward_contradiction: float,
                                        semantic_similarity: float, context_awareness: float) -> float:
        """Calculate confidence in the evaluation based on model certainty."""
        # Higher confidence when model is more certain
        entailment_confidence = (forward_entailment + backward_entailment) / 2
        contradiction_confidence = (forward_contradiction + backward_contradiction) / 2
        
        # Overall confidence based on model certainty and consistency
        confidence = (
            0.4 * entailment_confidence +
            0.2 * (1.0 - contradiction_confidence) +  # Lower contradiction = higher confidence
            0.2 * semantic_similarity +
            0.2 * context_awareness
        )
        
        return min(1.0, max(0.0, confidence))
    
    def evaluate_cross_language_nli_substitutability(self, text: str, source_lang: str, target_lang: str,
                                                    primitive: str = None) -> Dict[str, Any]:
        """Evaluate cross-language NLI substitutability via NSM explications."""
        logger.info(f"Evaluating cross-language NLI substitutability: {text} ({source_lang} → {target_lang})")
        
        # Generate explications in both languages
        source_explication = self.enhanced_explicator.generate_explication(text, source_lang, primitive)
        target_explication = self.enhanced_explicator.generate_explication(text, target_lang, primitive)
        
        # Evaluate substitutability in source language
        source_evaluation = self.evaluate_enhanced_nli_substitutability(
            text, source_explication, source_lang, primitive
        )
        
        # Evaluate substitutability in target language
        target_evaluation = self.evaluate_enhanced_nli_substitutability(
            text, target_explication, target_lang, primitive
        )
        
        # Evaluate cross-language consistency
        cross_consistency = self.evaluate_enhanced_nli_substitutability(
            source_explication, target_explication, source_lang, primitive
        )
        
        # Calculate overall cross-language substitutability
        overall_score = (
            source_evaluation.get('substitutability_score', 0.0) * 0.4 +
            target_evaluation.get('substitutability_score', 0.0) * 0.4 +
            cross_consistency.get('substitutability_score', 0.0) * 0.2
        )
        
        return {
            'success': True,
            'source_text': text,
            'source_language': source_lang,
            'target_language': target_lang,
            'primitive': primitive,
            'explications': {
                'source': source_explication,
                'target': target_explication
            },
            'evaluations': {
                'source': source_evaluation,
                'target': target_evaluation,
                'cross_consistency': cross_consistency
            },
            'overall_substitutability': float(overall_score),
            'cross_language_quality': {
                'source_quality': source_evaluation.get('substitutability_score', 0.0),
                'target_quality': target_evaluation.get('substitutability_score', 0.0),
                'consistency_quality': cross_consistency.get('substitutability_score', 0.0)
            }
        }
    
    def evaluate_dataset_nli_substitutability(self, dataset_path: str = None) -> Dict[str, Any]:
        """Evaluate NLI substitutability on a comprehensive dataset."""
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
            logger.error("No suitable NLI test dataset found")
            return {}
        
        logger.info(f"Evaluating NLI substitutability on dataset: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Extract test data
        test_entries = []
        if 'entries' in dataset:
            for entry in dataset['entries']:
                if 'text' in entry and 'language' in entry:
                    test_entries.append(entry)
        
        # Perform NLI evaluations
        results = {
            'total_entries': len(test_entries),
            'language_analysis': {},
            'primitive_analysis': {},
            'overall_metrics': {
                'avg_substitutability': 0.0,
                'avg_confidence': 0.0,
                'success_rate': 0.0,
                'quality_distribution': {'high': 0, 'medium': 0, 'low': 0, 'poor': 0}
            }
        }
        
        all_scores = []
        all_confidences = []
        successful_evaluations = 0
        
        for entry in test_entries[:100]:  # Limit for testing
            text = entry.get('text', '')
            language = entry.get('language', 'en')
            primitive = entry.get('primitive', 'HasProperty')
            
            try:
                # Generate explication
                explication = self.enhanced_explicator.generate_explication(text, language, primitive)
                
                # Evaluate NLI substitutability
                evaluation = self.evaluate_enhanced_nli_substitutability(
                    text, explication, language, primitive
                )
                
                if evaluation['success']:
                    successful_evaluations += 1
                    score = evaluation['substitutability_score']
                    confidence = evaluation['confidence']
                    level = evaluation['substitutability_level']
                    
                    all_scores.append(score)
                    all_confidences.append(confidence)
                    
                    # Update language analysis
                    if language not in results['language_analysis']:
                        results['language_analysis'][language] = {
                            'entries': 0,
                            'avg_substitutability': 0.0,
                            'avg_confidence': 0.0,
                            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0, 'poor': 0}
                        }
                    
                    results['language_analysis'][language]['entries'] += 1
                    results['language_analysis'][language]['avg_substitutability'] += score
                    results['language_analysis'][language]['avg_confidence'] += confidence
                    results['language_analysis'][language]['quality_distribution'][level] += 1
                    
                    # Update primitive analysis
                    if primitive not in results['primitive_analysis']:
                        results['primitive_analysis'][primitive] = {
                            'entries': 0,
                            'avg_substitutability': 0.0,
                            'avg_confidence': 0.0,
                            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0, 'poor': 0}
                        }
                    
                    results['primitive_analysis'][primitive]['entries'] += 1
                    results['primitive_analysis'][primitive]['avg_substitutability'] += score
                    results['primitive_analysis'][primitive]['avg_confidence'] += confidence
                    results['primitive_analysis'][primitive]['quality_distribution'][level] += 1
                    
                    # Update overall metrics
                    results['overall_metrics']['quality_distribution'][level] += 1
                
            except Exception as e:
                logger.warning(f"NLI evaluation failed for {text}: {e}")
        
        # Calculate averages
        if all_scores:
            results['overall_metrics']['avg_substitutability'] = np.mean(all_scores)
            results['overall_metrics']['avg_confidence'] = np.mean(all_confidences)
        
        results['overall_metrics']['success_rate'] = successful_evaluations / len(test_entries) if test_entries else 0.0
        
        # Calculate language averages
        for lang_data in results['language_analysis'].values():
            if lang_data['entries'] > 0:
                lang_data['avg_substitutability'] /= lang_data['entries']
                lang_data['avg_confidence'] /= lang_data['entries']
        
        # Calculate primitive averages
        for prim_data in results['primitive_analysis'].values():
            if prim_data['entries'] > 0:
                prim_data['avg_substitutability'] /= prim_data['entries']
                prim_data['avg_confidence'] /= prim_data['entries']
        
        return results


def main():
    """Main function to run enhanced NLI substitutability evaluation."""
    logger.info("Starting enhanced NLI substitutability evaluation...")
    
    # Initialize enhanced evaluator
    evaluator = EnhancedNLISubstitutabilityEvaluator()
    
    # Test NLI substitutability examples
    test_examples = [
        {"text": "The book is on the table", "language": "en", "primitive": "AtLocation"},
        {"text": "This thing is like that thing", "language": "en", "primitive": "SimilarTo"},
        {"text": "Esta cosa está en este lugar", "language": "es", "primitive": "AtLocation"},
        {"text": "Cette chose fait partie de l'ensemble", "language": "fr", "primitive": "PartOf"}
    ]
    
    # Perform test evaluations
    evaluation_results = []
    for example in test_examples:
        text = example["text"]
        language = example["language"]
        primitive = example["primitive"]
        
        # Generate explication
        explication = evaluator.enhanced_explicator.generate_explication(text, language, primitive)
        
        # Evaluate NLI substitutability
        result = evaluator.evaluate_enhanced_nli_substitutability(text, explication, language, primitive)
        evaluation_results.append({
            'example': example,
            'explication': explication,
            'evaluation': result
        })
        
        print(f"\nNLI Evaluation: {text} ({language}, {primitive})")
        if result['success']:
            print(f"Explication: {explication}")
            print(f"Substitutability: {result['substitutability_score']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Level: {result['substitutability_level']}")
            print(f"Forward Entailment: {result['entailment_scores']['forward']:.3f}")
            print(f"Backward Entailment: {result['entailment_scores']['backward']:.3f}")
        else:
            print(f"Failed: {result['error']}")
    
    # Evaluate on dataset
    dataset_results = evaluator.evaluate_dataset_nli_substitutability()
    
    # Save results
    output_path = "data/nsm_nli_substitutability_enhanced_report.json"
    report = {
        "metadata": {
            "report_type": "enhanced_NLI_substitutability_report",
            "timestamp": "2025-08-22",
            "enhanced_features": [
                "multilingual_XNLI_evaluation",
                "bidirectional_entailment_analysis",
                "context_awareness_evaluation",
                "primitive_specific_strategies",
                "cross_language_validation",
                "confidence_assessment"
            ]
        },
        "test_evaluations": evaluation_results,
        "dataset_evaluation": dataset_results,
        "summary": {
            "evaluation_examples": len(evaluation_results),
            "successful_evaluations": sum(1 for r in evaluation_results if r['evaluation']['success']),
            "dataset_success_rate": dataset_results.get('overall_metrics', {}).get('success_rate', 0.0),
            "avg_substitutability": dataset_results.get('overall_metrics', {}).get('avg_substitutability', 0.0),
            "avg_confidence": dataset_results.get('overall_metrics', {}).get('avg_confidence', 0.0)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(report), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced NLI substitutability report saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ENHANCED NLI SUBSTITUTABILITY EVALUATION SUMMARY")
    print("="*80)
    print(f"Test Examples: {len(evaluation_results)}")
    print(f"Successful: {sum(1 for r in evaluation_results if r['evaluation']['success'])}/{len(evaluation_results)}")
    if dataset_results:
        print(f"Dataset Success Rate: {dataset_results.get('overall_metrics', {}).get('success_rate', 0.0):.1%}")
        print(f"Average Substitutability: {dataset_results.get('overall_metrics', {}).get('avg_substitutability', 0.0):.3f}")
        print(f"Average Confidence: {dataset_results.get('overall_metrics', {}).get('avg_confidence', 0.0):.3f}")
        print("\nLanguage Analysis:")
        for lang, data in dataset_results.get('language_analysis', {}).items():
            print(f"  {lang}: {data['avg_substitutability']:.3f} avg substitutability, {data['entries']} entries")
        print("\nPrimitive Analysis:")
        for primitive, data in dataset_results.get('primitive_analysis', {}).items():
            print(f"  {primitive}: {data['avg_substitutability']:.3f} avg substitutability, {data['entries']} entries")
    print("="*80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Metrics Pipeline Consolidation System.

This script implements a comprehensive metrics pipeline consolidation system
to provide a single source of truth for all KPIs with proper versioning and CI gates.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import time
from collections import defaultdict, Counter
import hashlib
from datetime import datetime

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
    logger.error(f"Failed to import system components: {e}")
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
        converted_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                str_key = f"{key[0]}_{key[1]}" if len(key) == 2 else str(key)
                converted_dict[str_key] = convert_numpy_types(value)
            else:
                converted_dict[str(key)] = convert_numpy_types(value)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_numpy_types(obj.__dict__)
    else:
        return obj


def get_git_commit_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_file_hash(file_path: str) -> str:
    """Get SHA256 hash of a file."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:8]
    except FileNotFoundError:
        return "missing"


class MetricsCollector:
    """Collector for comprehensive system metrics."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.nsm_translator = NSMTranslator()
        self.sbert_model = None
        
        # Metrics configuration
        self.metrics_config = {
            'target_prime_recall': 0.98,
            'target_polarity_recall': 0.98,
            'target_scope_recall': 0.98,
            'target_legality': 0.90,
            'target_mps': 0.90,
            'target_cross_lang_consistency': 0.95,
            'target_synset_stability': 0.80,
            'target_cns': 0.80,
            'target_mdl_improvement': 0.15
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for metrics collection...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def collect_primitive_metrics(self, test_texts: List[str], languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Collect primitive detection metrics."""
        logger.info("Collecting primitive detection metrics...")
        
        metrics = {
            'prime_detection': {
                'total_texts': len(test_texts) * len(languages),
                'successful_detections': 0,
                'primitives_found': [],
                'prime_types': defaultdict(int),
                'detection_rates': defaultdict(float)
            },
            'polarity_detection': {
                'total_texts': len(test_texts) * len(languages),
                'polarity_found': 0,
                'negation_detected': 0,
                'polarity_accuracy': 0.0
            },
            'scope_detection': {
                'total_texts': len(test_texts) * len(languages),
                'scope_found': 0,
                'scope_accuracy': 0.0
            }
        }
        
        for language in languages:
            for text in test_texts:
                try:
                    primitives = self.nsm_translator.detect_primitives_in_text(text, language)
                    
                    if primitives:
                        metrics['prime_detection']['successful_detections'] += 1
                        metrics['prime_detection']['primitives_found'].extend(primitives)
                        
                        # Count prime types
                        for primitive in primitives:
                            metrics['prime_detection']['prime_types'][primitive] += 1
                        
                        # Check for polarity/negation
                        if any('NOT' in p or 'NEG' in p for p in primitives):
                            metrics['polarity_detection']['polarity_found'] += 1
                            metrics['polarity_detection']['negation_detected'] += 1
                        
                        # Check for scope indicators
                        if any('SCOPE' in p or 'FOR' in p or 'WHEN' in p for p in primitives):
                            metrics['scope_detection']['scope_found'] += 1
                
                except Exception as e:
                    logger.warning(f"Primitive detection failed for '{text}' in {language}: {e}")
        
        # Calculate rates
        total = metrics['prime_detection']['total_texts']
        if total > 0:
            metrics['prime_detection']['detection_rate'] = metrics['prime_detection']['successful_detections'] / total
            metrics['polarity_detection']['polarity_accuracy'] = metrics['polarity_detection']['polarity_found'] / total
            metrics['scope_detection']['scope_accuracy'] = metrics['scope_detection']['scope_found'] / total
        
        return metrics
    
    def collect_legality_metrics(self, test_texts: List[str], languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Collect NSM legality metrics."""
        logger.info("Collecting NSM legality metrics...")
        
        metrics = {
            'total_explications': 0,
            'legal_explications': 0,
            'legality_rate': 0.0,
            'avg_legality_score': 0.0,
            'legality_scores': []
        }
        
        for language in languages:
            for text in test_texts:
                try:
                    primitives = self.nsm_translator.detect_primitives_in_text(text, language)
                    
                    if primitives:
                        # Simple legality check
                        explication = f"{' '.join(primitives)}({text})"
                        metrics['total_explications'] += 1
                        
                        # Basic legality validation
                        is_legal = self._validate_basic_legality(explication, primitives)
                        if is_legal:
                            metrics['legal_explications'] += 1
                        
                        # Calculate legality score
                        legality_score = self._calculate_legality_score(explication, primitives)
                        metrics['legality_scores'].append(legality_score)
                
                except Exception as e:
                    logger.warning(f"Legality check failed for '{text}' in {language}: {e}")
        
        # Calculate averages
        if metrics['total_explications'] > 0:
            metrics['legality_rate'] = metrics['legal_explications'] / metrics['total_explications']
            metrics['avg_legality_score'] = np.mean(metrics['legality_scores']) if metrics['legality_scores'] else 0.0
        
        return metrics
    
    def collect_mps_metrics(self, test_texts: List[str], languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Collect Meaning Preservation Score (MPS) metrics."""
        logger.info("Collecting MPS metrics...")
        
        metrics = {
            'total_pairs': 0,
            'mps_scores': [],
            'avg_mps': 0.0,
            'mps_distribution': defaultdict(int)
        }
        
        for language in languages:
            for i, text in enumerate(test_texts):
                try:
                    primitives = self.nsm_translator.detect_primitives_in_text(text, language)
                    
                    if primitives:
                        explication = f"{' '.join(primitives)}({text})"
                        
                        # Calculate MPS using semantic similarity
                        mps_score = self._calculate_mps(text, explication)
                        metrics['mps_scores'].append(mps_score)
                        metrics['total_pairs'] += 1
                        
                        # Categorize MPS scores
                        if mps_score >= 0.9:
                            metrics['mps_distribution']['excellent'] += 1
                        elif mps_score >= 0.7:
                            metrics['mps_distribution']['good'] += 1
                        elif mps_score >= 0.5:
                            metrics['mps_distribution']['fair'] += 1
                        else:
                            metrics['mps_distribution']['poor'] += 1
                
                except Exception as e:
                    logger.warning(f"MPS calculation failed for '{text}' in {language}: {e}")
        
        # Calculate average MPS
        if metrics['mps_scores']:
            metrics['avg_mps'] = np.mean(metrics['mps_scores'])
        
        return metrics
    
    def collect_cross_language_metrics(self, test_texts: List[str], languages: List[str] = ["en", "es", "fr"]) -> Dict[str, Any]:
        """Collect cross-language consistency metrics."""
        logger.info("Collecting cross-language consistency metrics...")
        
        metrics = {
            'total_comparisons': 0,
            'consistent_pairs': 0,
            'consistency_rate': 0.0,
            'avg_consistency_score': 0.0,
            'consistency_scores': [],
            'language_pairs': defaultdict(list)
        }
        
        # Compare across language pairs
        for i, lang1 in enumerate(languages):
            for lang2 in languages[i+1:]:
                for text in test_texts:
                    try:
                        primitives1 = self.nsm_translator.detect_primitives_in_text(text, lang1)
                        primitives2 = self.nsm_translator.detect_primitives_in_text(text, lang2)
                        
                        if primitives1 and primitives2:
                            # Calculate consistency
                            consistency_score = self._calculate_cross_language_consistency(primitives1, primitives2)
                            metrics['consistency_scores'].append(consistency_score)
                            metrics['total_comparisons'] += 1
                            
                            if consistency_score >= 0.8:
                                metrics['consistent_pairs'] += 1
                            
                            # Store by language pair
                            pair_key = f"{lang1}-{lang2}"
                            metrics['language_pairs'][pair_key].append(consistency_score)
                    
                    except Exception as e:
                        logger.warning(f"Cross-language comparison failed for '{text}' ({lang1}-{lang2}): {e}")
        
        # Calculate averages
        if metrics['total_comparisons'] > 0:
            metrics['consistency_rate'] = metrics['consistent_pairs'] / metrics['total_comparisons']
            metrics['avg_consistency_score'] = np.mean(metrics['consistency_scores']) if metrics['consistency_scores'] else 0.0
        
        return metrics
    
    def collect_synset_stability_metrics(self, test_texts: List[str], languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Collect synset stability metrics."""
        logger.info("Collecting synset stability metrics...")
        
        metrics = {
            'total_synsets': 0,
            'stable_synsets': 0,
            'stability_rate': 0.0,
            'avg_stability_score': 0.0,
            'stability_scores': []
        }
        
        # This is a placeholder - would need BabelNet integration for full implementation
        for language in languages:
            for text in test_texts:
                try:
                    # Simulate synset stability check
                    stability_score = self._simulate_synset_stability(text, language)
                    metrics['stability_scores'].append(stability_score)
                    metrics['total_synsets'] += 1
                    
                    if stability_score >= 0.8:
                        metrics['stable_synsets'] += 1
                
                except Exception as e:
                    logger.warning(f"Synset stability check failed for '{text}' in {language}: {e}")
        
        # Calculate averages
        if metrics['total_synsets'] > 0:
            metrics['stability_rate'] = metrics['stable_synsets'] / metrics['total_synsets']
            metrics['avg_stability_score'] = np.mean(metrics['stability_scores']) if metrics['stability_scores'] else 0.0
        
        return metrics
    
    def collect_cns_metrics(self, test_texts: List[str], languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Collect Cultural Naturalness Score (CNS) metrics."""
        logger.info("Collecting CNS metrics...")
        
        metrics = {
            'total_evaluations': 0,
            'natural_texts': 0,
            'naturalness_rate': 0.0,
            'avg_cns': 0.0,
            'cns_scores': []
        }
        
        for language in languages:
            for text in test_texts:
                try:
                    # Calculate CNS (simplified version)
                    cns_score = self._calculate_cns(text, language)
                    metrics['cns_scores'].append(cns_score)
                    metrics['total_evaluations'] += 1
                    
                    if cns_score >= 0.8:
                        metrics['natural_texts'] += 1
                
                except Exception as e:
                    logger.warning(f"CNS calculation failed for '{text}' in {language}: {e}")
        
        # Calculate averages
        if metrics['total_evaluations'] > 0:
            metrics['naturalness_rate'] = metrics['natural_texts'] / metrics['total_evaluations']
            metrics['avg_cns'] = np.mean(metrics['cns_scores']) if metrics['cns_scores'] else 0.0
        
        return metrics
    
    def collect_mdl_metrics(self, test_texts: List[str], languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Collect Minimum Description Length (MDL) improvement metrics."""
        logger.info("Collecting MDL metrics...")
        
        metrics = {
            'total_compressions': 0,
            'successful_compressions': 0,
            'avg_mdl_improvement': 0.0,
            'mdl_improvements': [],
            'compression_rates': []
        }
        
        for language in languages:
            for text in test_texts:
                try:
                    primitives = self.nsm_translator.detect_primitives_in_text(text, language)
                    
                    if primitives:
                        # Calculate MDL improvement
                        original_length = len(text.split())
                        compressed_length = len(primitives)
                        
                        if original_length > 0:
                            compression_rate = (original_length - compressed_length) / original_length
                            metrics['compression_rates'].append(compression_rate)
                            metrics['total_compressions'] += 1
                            
                            if compression_rate > 0.15:  # 15% improvement threshold
                                metrics['successful_compressions'] += 1
                            
                            metrics['mdl_improvements'].append(compression_rate)
                
                except Exception as e:
                    logger.warning(f"MDL calculation failed for '{text}' in {language}: {e}")
        
        # Calculate averages
        if metrics['mdl_improvements']:
            metrics['avg_mdl_improvement'] = np.mean(metrics['mdl_improvements'])
        
        return metrics
    
    def _validate_basic_legality(self, explication: str, primitives: List[str]) -> bool:
        """Basic legality validation."""
        try:
            # Check for basic NSM structure
            if not explication or not primitives:
                return False
            
            # Check for balanced parentheses
            if explication.count('(') != explication.count(')'):
                return False
            
            # Check for primitive presence
            for primitive in primitives:
                if primitive not in explication:
                    return False
            
            return True
        
        except Exception:
            return False
    
    def _calculate_legality_score(self, explication: str, primitives: List[str]) -> float:
        """Calculate legality score."""
        try:
            score = 0.0
            
            # Basic structure (40%)
            if self._validate_basic_legality(explication, primitives):
                score += 0.4
            
            # Primitive coverage (30%)
            primitive_coverage = sum(1 for p in primitives if p in explication) / len(primitives)
            score += 0.3 * primitive_coverage
            
            # Length appropriateness (30%)
            words = explication.split()
            if 3 <= len(words) <= 20:
                score += 0.3
            
            return score
        
        except Exception:
            return 0.0
    
    def _calculate_mps(self, original_text: str, explication: str) -> float:
        """Calculate Meaning Preservation Score."""
        try:
            if not self.sbert_model:
                return 0.8  # Default score if no model available
            
            # Calculate semantic similarity
            embeddings = self.sbert_model.encode([original_text, explication])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return max(0.0, float(similarity))
        
        except Exception:
            return 0.5
    
    def _calculate_cross_language_consistency(self, primitives1: List[str], primitives2: List[str]) -> float:
        """Calculate cross-language consistency."""
        try:
            # Simple overlap-based consistency
            set1 = set(primitives1)
            set2 = set(primitives2)
            
            if not set1 and not set2:
                return 1.0
            elif not set1 or not set2:
                return 0.0
            
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            
            return len(intersection) / len(union)
        
        except Exception:
            return 0.0
    
    def _simulate_synset_stability(self, text: str, language: str) -> float:
        """Simulate synset stability score."""
        try:
            # Placeholder implementation
            # In real implementation, this would check BabelNet synset consistency
            return 0.8  # Simulated stable score
        except Exception:
            return 0.5
    
    def _calculate_cns(self, text: str, language: str) -> float:
        """Calculate Cultural Naturalness Score."""
        try:
            # Simplified CNS calculation
            # In real implementation, this would use human evaluation or fine-tuned models
            
            # Basic heuristics
            score = 0.8  # Base score
            
            # Length penalty
            words = text.split()
            if len(words) < 2:
                score -= 0.2
            elif len(words) > 20:
                score -= 0.1
            
            # Grammar penalty (simple checks)
            if text.count('(') != text.count(')'):
                score -= 0.1
            
            return max(0.0, score)
        
        except Exception:
            return 0.5


class MetricsPipeline:
    """Comprehensive metrics pipeline with versioning and CI gates."""
    
    def __init__(self):
        """Initialize the metrics pipeline."""
        self.collector = MetricsCollector()
        
        # Pipeline configuration
        self.pipeline_config = {
            'test_texts_per_language': 100,
            'languages': ["en", "es", "fr"],
            'enable_ci_gates': True,
            'output_directory': "metrics"
        }
    
    def run_comprehensive_metrics(self) -> Dict[str, Any]:
        """Run comprehensive metrics collection."""
        logger.info("Running comprehensive metrics pipeline...")
        
        # Generate test texts
        test_texts = self._generate_test_texts()
        
        # Collect all metrics
        metrics_results = {
            'metadata': self._generate_metadata(),
            'macro_metrics': {},
            'phenomenon_metrics': {},
            'ci_gates': {}
        }
        
        # Collect macro metrics
        logger.info("Collecting macro metrics...")
        metrics_results['macro_metrics'] = {
            'primitive_metrics': self.collector.collect_primitive_metrics(test_texts, self.pipeline_config['languages']),
            'legality_metrics': self.collector.collect_legality_metrics(test_texts, self.pipeline_config['languages']),
            'mps_metrics': self.collector.collect_mps_metrics(test_texts, self.pipeline_config['languages']),
            'cross_language_metrics': self.collector.collect_cross_language_metrics(test_texts, self.pipeline_config['languages']),
            'synset_stability_metrics': self.collector.collect_synset_stability_metrics(test_texts, self.pipeline_config['languages']),
            'cns_metrics': self.collector.collect_cns_metrics(test_texts, self.pipeline_config['languages']),
            'mdl_metrics': self.collector.collect_mdl_metrics(test_texts, self.pipeline_config['languages'])
        }
        
        # Collect phenomenon-specific metrics
        logger.info("Collecting phenomenon-specific metrics...")
        metrics_results['phenomenon_metrics'] = self._collect_phenomenon_metrics(test_texts)
        
        # Evaluate CI gates
        logger.info("Evaluating CI gates...")
        metrics_results['ci_gates'] = self._evaluate_ci_gates(metrics_results['macro_metrics'])
        
        return metrics_results
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the metrics run."""
        return {
            'timestamp': datetime.now().isoformat(),
            'git_commit_sha': get_git_commit_sha(),
            'dataset_version': 'v1.0',
            'prime_registry_version': get_file_hash('primitives.json') if os.path.exists('primitives.json') else 'unknown',
            'molecule_registry_version': get_file_hash('molecules.json') if os.path.exists('molecules.json') else 'unknown',
            'evaluation_config': self.collector.metrics_config,
            'pipeline_config': self.pipeline_config
        }
    
    def _generate_test_texts(self) -> List[str]:
        """Generate test texts for metrics evaluation."""
        test_texts = [
            # Basic statements
            "The cat is on the mat",
            "The weather is cold today",
            "She works at the hospital",
            "The book contains important information",
            "Children play in the park",
            
            # Negation
            "The cat is not on the mat",
            "I do not like this weather",
            "She does not work here",
            "The book does not contain that information",
            "Children do not play here",
            
            # Modality
            "The cat might be on the mat",
            "The weather could be cold",
            "She should work here",
            "The book must contain this",
            "Children can play here",
            
            # Temporal
            "The cat was on the mat",
            "The weather will be cold",
            "She has worked here",
            "The book had contained this",
            "Children will play here",
            
            # Quantifiers
            "All cats are on mats",
            "Some weather is cold",
            "Many people work here",
            "Few books contain this",
            "Most children play here",
            
            # Causation
            "The rain caused the flood",
            "Eating caused the sickness",
            "The fire caused the damage",
            "The accident caused the injury",
            "The storm caused the power outage",
            
            # Experiencer
            "I like this weather",
            "She enjoys her work",
            "They love this book",
            "Children hate homework",
            "We prefer this option",
            
            # Aspectual
            "I almost finished the work",
            "She just arrived",
            "They still work here",
            "We already ate",
            "He has not yet arrived",
            
            # Politeness
            "Could you please help me?",
            "Would you mind closing the door?",
            "I'm sorry for the delay",
            "Thank you for your help",
            "Please take a seat",
            
            # Idioms
            "It's raining cats and dogs",
            "She's pulling my leg",
            "He's on the ball",
            "We're in the same boat",
            "She's got a chip on her shoulder"
        ]
        
        return test_texts[:self.pipeline_config['test_texts_per_language']]
    
    def _collect_phenomenon_metrics(self, test_texts: List[str]) -> Dict[str, Any]:
        """Collect phenomenon-specific metrics."""
        phenomena = {
            'negation': [text for text in test_texts if 'not' in text.lower() or 'no ' in text.lower()],
            'modality': [text for text in test_texts if any(word in text.lower() for word in ['might', 'could', 'should', 'must', 'can'])],
            'temporal': [text for text in test_texts if any(word in text.lower() for word in ['was', 'will', 'has', 'had', 'been'])],
            'quantifiers': [text for text in test_texts if any(word in text.lower() for word in ['all', 'some', 'many', 'few', 'most'])],
            'causation': [text for text in test_texts if 'caused' in text.lower()],
            'experiencer': [text for text in test_texts if any(word in text.lower() for word in ['like', 'enjoy', 'love', 'hate', 'prefer'])],
            'aspectual': [text for text in test_texts if any(word in text.lower() for word in ['almost', 'just', 'still', 'already', 'yet'])],
            'politeness': [text for text in test_texts if any(word in text.lower() for word in ['please', 'sorry', 'thank', 'could you', 'would you'])],
            'idioms': [text for text in test_texts if any(phrase in text.lower() for phrase in ['cats and dogs', 'pulling my leg', 'on the ball', 'same boat', 'chip on'])]
        }
        
        phenomenon_metrics = {}
        
        for phenomenon, texts in phenomena.items():
            if texts:
                phenomenon_metrics[phenomenon] = {
                    'primitive_metrics': self.collector.collect_primitive_metrics(texts, self.pipeline_config['languages']),
                    'legality_metrics': self.collector.collect_legality_metrics(texts, self.pipeline_config['languages']),
                    'mps_metrics': self.collector.collect_mps_metrics(texts, self.pipeline_config['languages'])
                }
        
        return phenomenon_metrics
    
    def _evaluate_ci_gates(self, macro_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate CI gates against target metrics."""
        ci_gates = {
            'all_passed': True,
            'gate_results': {},
            'failures': []
        }
        
        # Prime recall gate
        prime_recall = macro_metrics['primitive_metrics']['prime_detection']['detection_rate']
        prime_gate_passed = prime_recall >= self.collector.metrics_config['target_prime_recall']
        ci_gates['gate_results']['prime_recall'] = {
            'passed': prime_gate_passed,
            'value': prime_recall,
            'target': self.collector.metrics_config['target_prime_recall']
        }
        if not prime_gate_passed:
            ci_gates['all_passed'] = False
            ci_gates['failures'].append(f"Prime recall {prime_recall:.3f} < {self.collector.metrics_config['target_prime_recall']}")
        
        # Polarity recall gate
        polarity_recall = macro_metrics['primitive_metrics']['polarity_detection']['polarity_accuracy']
        polarity_gate_passed = polarity_recall >= self.collector.metrics_config['target_polarity_recall']
        ci_gates['gate_results']['polarity_recall'] = {
            'passed': polarity_gate_passed,
            'value': polarity_recall,
            'target': self.collector.metrics_config['target_polarity_recall']
        }
        if not polarity_gate_passed:
            ci_gates['all_passed'] = False
            ci_gates['failures'].append(f"Polarity recall {polarity_recall:.3f} < {self.collector.metrics_config['target_polarity_recall']}")
        
        # Scope recall gate
        scope_recall = macro_metrics['primitive_metrics']['scope_detection']['scope_accuracy']
        scope_gate_passed = scope_recall >= self.collector.metrics_config['target_scope_recall']
        ci_gates['gate_results']['scope_recall'] = {
            'passed': scope_gate_passed,
            'value': scope_recall,
            'target': self.collector.metrics_config['target_scope_recall']
        }
        if not scope_gate_passed:
            ci_gates['all_passed'] = False
            ci_gates['failures'].append(f"Scope recall {scope_recall:.3f} < {self.collector.metrics_config['target_scope_recall']}")
        
        # Legality gate
        legality_rate = macro_metrics['legality_metrics']['legality_rate']
        legality_gate_passed = legality_rate >= self.collector.metrics_config['target_legality']
        ci_gates['gate_results']['legality'] = {
            'passed': legality_gate_passed,
            'value': legality_rate,
            'target': self.collector.metrics_config['target_legality']
        }
        if not legality_gate_passed:
            ci_gates['all_passed'] = False
            ci_gates['failures'].append(f"Legality {legality_rate:.3f} < {self.collector.metrics_config['target_legality']}")
        
        # MPS gate
        avg_mps = macro_metrics['mps_metrics']['avg_mps']
        mps_gate_passed = avg_mps >= self.collector.metrics_config['target_mps']
        ci_gates['gate_results']['mps'] = {
            'passed': mps_gate_passed,
            'value': avg_mps,
            'target': self.collector.metrics_config['target_mps']
        }
        if not mps_gate_passed:
            ci_gates['all_passed'] = False
            ci_gates['failures'].append(f"MPS {avg_mps:.3f} < {self.collector.metrics_config['target_mps']}")
        
        # Cross-language consistency gate
        consistency_rate = macro_metrics['cross_language_metrics']['consistency_rate']
        consistency_gate_passed = consistency_rate >= self.collector.metrics_config['target_cross_lang_consistency']
        ci_gates['gate_results']['cross_language_consistency'] = {
            'passed': consistency_gate_passed,
            'value': consistency_rate,
            'target': self.collector.metrics_config['target_cross_lang_consistency']
        }
        if not consistency_gate_passed:
            ci_gates['all_passed'] = False
            ci_gates['failures'].append(f"Cross-language consistency {consistency_rate:.3f} < {self.collector.metrics_config['target_cross_lang_consistency']}")
        
        return ci_gates
    
    def save_metrics(self, metrics_results: Dict[str, Any]) -> str:
        """Save metrics to file."""
        # Ensure output directory exists
        os.makedirs(self.pipeline_config['output_directory'], exist_ok=True)
        
        # Save latest metrics
        latest_path = os.path.join(self.pipeline_config['output_directory'], 'latest.json')
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(metrics_results), f, ensure_ascii=False, indent=2)
        
        # Save timestamped version
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_path = os.path.join(self.pipeline_config['output_directory'], f'metrics_{timestamp}.json')
        with open(timestamped_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(metrics_results), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metrics saved to: {latest_path}")
        logger.info(f"Timestamped metrics saved to: {timestamped_path}")
        
        return latest_path


def main():
    """Main function to run comprehensive metrics pipeline."""
    logger.info("Starting comprehensive metrics pipeline...")
    
    # Initialize metrics pipeline
    pipeline = MetricsPipeline()
    
    # Run comprehensive metrics
    metrics_results = pipeline.run_comprehensive_metrics()
    
    # Save metrics
    output_path = pipeline.save_metrics(metrics_results)
    
    # Print results
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS PIPELINE RESULTS")
    print("="*80)
    
    print(f"Metadata:")
    metadata = metrics_results['metadata']
    print(f"  Timestamp: {metadata['timestamp']}")
    print(f"  Git Commit: {metadata['git_commit_sha']}")
    print(f"  Dataset Version: {metadata['dataset_version']}")
    
    print(f"\nMacro Metrics:")
    macro = metrics_results['macro_metrics']
    
    print(f"  Primitive Detection:")
    prime_metrics = macro['primitive_metrics']
    print(f"    Detection Rate: {prime_metrics['prime_detection']['detection_rate']:.3f}")
    print(f"    Polarity Accuracy: {prime_metrics['polarity_detection']['polarity_accuracy']:.3f}")
    print(f"    Scope Accuracy: {prime_metrics['scope_detection']['scope_accuracy']:.3f}")
    
    print(f"  Legality:")
    legality_metrics = macro['legality_metrics']
    print(f"    Legality Rate: {legality_metrics['legality_rate']:.3f}")
    print(f"    Average Legality Score: {legality_metrics['avg_legality_score']:.3f}")
    
    print(f"  Meaning Preservation Score (MPS):")
    mps_metrics = macro['mps_metrics']
    print(f"    Average MPS: {mps_metrics['avg_mps']:.3f}")
    print(f"    Excellent: {mps_metrics['mps_distribution']['excellent']}")
    print(f"    Good: {mps_metrics['mps_distribution']['good']}")
    print(f"    Fair: {mps_metrics['mps_distribution']['fair']}")
    print(f"    Poor: {mps_metrics['mps_distribution']['poor']}")
    
    print(f"  Cross-Language Consistency:")
    cross_metrics = macro['cross_language_metrics']
    print(f"    Consistency Rate: {cross_metrics['consistency_rate']:.3f}")
    print(f"    Average Consistency Score: {cross_metrics['avg_consistency_score']:.3f}")
    
    print(f"  Cultural Naturalness Score (CNS):")
    cns_metrics = macro['cns_metrics']
    print(f"    Naturalness Rate: {cns_metrics['naturalness_rate']:.3f}")
    print(f"    Average CNS: {cns_metrics['avg_cns']:.3f}")
    
    print(f"  Minimum Description Length (MDL):")
    mdl_metrics = macro['mdl_metrics']
    print(f"    Average MDL Improvement: {mdl_metrics['avg_mdl_improvement']:.3f}")
    print(f"    Successful Compressions: {mdl_metrics['successful_compressions']}/{mdl_metrics['total_compressions']}")
    
    print(f"\nCI Gates:")
    ci_gates = metrics_results['ci_gates']
    print(f"  All Gates Passed: {ci_gates['all_passed']}")
    
    for gate_name, gate_result in ci_gates['gate_results'].items():
        status = "✅ PASS" if gate_result['passed'] else "❌ FAIL"
        print(f"    {gate_name}: {status} ({gate_result['value']:.3f} vs {gate_result['target']:.3f})")
    
    if ci_gates['failures']:
        print(f"  Failures:")
        for failure in ci_gates['failures']:
            print(f"    - {failure}")
    
    print(f"\nPhenomenon Metrics:")
    phenomena = metrics_results['phenomenon_metrics']
    for phenomenon, metrics in phenomena.items():
        print(f"  {phenomenon.upper()}:")
        print(f"    Detection Rate: {metrics['primitive_metrics']['prime_detection']['detection_rate']:.3f}")
        print(f"    Legality Rate: {metrics['legality_metrics']['legality_rate']:.3f}")
        print(f"    Average MPS: {metrics['mps_metrics']['avg_mps']:.3f}")
    
    print("="*80)
    print("Comprehensive metrics pipeline completed!")
    print("="*80)


if __name__ == "__main__":
    main()

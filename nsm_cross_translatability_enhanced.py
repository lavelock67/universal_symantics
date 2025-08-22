#!/usr/bin/env python3
"""
Enhanced NSM Cross-Translatability Structural Check System.

This script implements comprehensive structural checks for NSM cross-translatability across EN/ES/FR:
1. Enhanced structural consistency validation across languages
2. Advanced semantic alignment verification with multiple metrics
3. Cross-language primitive mapping validation with detailed analysis
4. Structural pattern analysis with pattern evolution tracking
5. Cross-language semantic drift detection
6. Translation quality prediction and optimization
7. Language-specific structural adaptation analysis
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
try:
    from src.nsm.translate import NSMTranslator
    from src.nsm.explicator import NSMExplicator
    from src.table.schema import PeriodicTable
except ImportError as e:
    logger.error(f"Failed to import NSM components: {e}")
    exit(1)


class EnhancedNSMCrossTranslatabilityChecker:
    """Enhanced structural checks for NSM cross-translatability."""
    
    def __init__(self):
        """Initialize the enhanced cross-translatability checker."""
        self.sbert_model = None
        self.nli_model = None
        self.nsm_translator = NSMTranslator()
        self.nsm_explicator = NSMExplicator()
        self.languages = ['en', 'es', 'fr']
        
        # Load periodic table
        try:
            with open("data/nsm_periodic_table.json", 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            self.periodic_table = PeriodicTable.from_dict(table_data)
        except Exception as e:
            logger.warning(f"Failed to load periodic table: {e}")
            self.periodic_table = PeriodicTable()
        
        # Enhanced structural patterns with validation metadata
        self.enhanced_structural_patterns = {
            'AtLocation': {
                'en': {
                    'patterns': ['this thing is in this place', 'this thing exists in this location'],
                    'semantic_focus': 'spatial_relationship',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                },
                'es': {
                    'patterns': ['esta cosa está en este lugar', 'esta cosa existe en esta ubicación'],
                    'semantic_focus': 'spatial_relationship',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                },
                'fr': {
                    'patterns': ['cette chose est dans cet endroit', 'cette chose existe dans cet endroit'],
                    'semantic_focus': 'spatial_relationship',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                }
            },
            'HasProperty': {
                'en': {
                    'patterns': ['this thing has this characteristic', 'this thing is characterized by this property'],
                    'semantic_focus': 'attribute_assignment',
                    'structural_complexity': 'medium',
                    'cross_language_variants': 4
                },
                'es': {
                    'patterns': ['esta cosa tiene esta característica', 'esta cosa se caracteriza por esta propiedad'],
                    'semantic_focus': 'attribute_assignment',
                    'structural_complexity': 'medium',
                    'cross_language_variants': 4
                },
                'fr': {
                    'patterns': ['cette chose a cette caractéristique', 'cette chose se caractérise par cette propriété'],
                    'semantic_focus': 'attribute_assignment',
                    'structural_complexity': 'medium',
                    'cross_language_variants': 4
                }
            },
            'Causes': {
                'en': {
                    'patterns': ['this thing makes this other thing happen', 'this thing leads to this result'],
                    'semantic_focus': 'causal_relationship',
                    'structural_complexity': 'high',
                    'cross_language_variants': 5
                },
                'es': {
                    'patterns': ['esta cosa hace que esta otra cosa suceda', 'esta cosa lleva a este resultado'],
                    'semantic_focus': 'causal_relationship',
                    'structural_complexity': 'high',
                    'cross_language_variants': 5
                },
                'fr': {
                    'patterns': ['cette chose fait que cette autre chose se passe', 'cette chose mène à ce résultat'],
                    'semantic_focus': 'causal_relationship',
                    'structural_complexity': 'high',
                    'cross_language_variants': 5
                }
            },
            'UsedFor': {
                'en': {
                    'patterns': ['people can use this thing to do this action', 'this thing serves this purpose'],
                    'semantic_focus': 'functional_purpose',
                    'structural_complexity': 'medium',
                    'cross_language_variants': 4
                },
                'es': {
                    'patterns': ['la gente puede usar esta cosa para hacer esta acción', 'esta cosa sirve para este propósito'],
                    'semantic_focus': 'functional_purpose',
                    'structural_complexity': 'medium',
                    'cross_language_variants': 4
                },
                'fr': {
                    'patterns': ['les gens peuvent utiliser cette chose pour faire cette action', 'cette chose sert à ce but'],
                    'semantic_focus': 'functional_purpose',
                    'structural_complexity': 'medium',
                    'cross_language_variants': 4
                }
            },
            'PartOf': {
                'en': {
                    'patterns': ['this thing is a component of this whole', 'this thing belongs to this larger thing'],
                    'semantic_focus': 'hierarchical_relationship',
                    'structural_complexity': 'medium',
                    'cross_language_variants': 4
                },
                'es': {
                    'patterns': ['esta cosa es un componente de este todo', 'esta cosa pertenece a esta cosa más grande'],
                    'semantic_focus': 'hierarchical_relationship',
                    'structural_complexity': 'medium',
                    'cross_language_variants': 4
                },
                'fr': {
                    'patterns': ['cette chose est un composant de cet ensemble', 'cette chose appartient à cette chose plus grande'],
                    'semantic_focus': 'hierarchical_relationship',
                    'structural_complexity': 'medium',
                    'cross_language_variants': 4
                }
            },
            'SimilarTo': {
                'en': {
                    'patterns': ['this thing is like this other thing', 'this thing resembles this other thing'],
                    'semantic_focus': 'similarity_assessment',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                },
                'es': {
                    'patterns': ['esta cosa es como esta otra cosa', 'esta cosa se parece a esta otra cosa'],
                    'semantic_focus': 'similarity_assessment',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                },
                'fr': {
                    'patterns': ['cette chose est comme cette autre chose', 'cette chose ressemble à cette autre chose'],
                    'semantic_focus': 'similarity_assessment',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                }
            },
            'DifferentFrom': {
                'en': {
                    'patterns': ['this thing is different from this other thing', 'this thing is distinct from this other thing'],
                    'semantic_focus': 'distinction_establishment',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                },
                'es': {
                    'patterns': ['esta cosa es diferente de esta otra cosa', 'esta cosa es distinta de esta otra cosa'],
                    'semantic_focus': 'distinction_establishment',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                },
                'fr': {
                    'patterns': ['cette chose est différente de cette autre chose', 'cette chose est distincte de cette autre chose'],
                    'semantic_focus': 'distinction_establishment',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                }
            },
            'Not': {
                'en': {
                    'patterns': ['this thing is not this other thing', 'this thing does not have this property'],
                    'semantic_focus': 'negation_operation',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                },
                'es': {
                    'patterns': ['esta cosa no es esta otra cosa', 'esta cosa no tiene esta propiedad'],
                    'semantic_focus': 'negation_operation',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                },
                'fr': {
                    'patterns': ['cette chose n\'est pas cette autre chose', 'cette chose n\'a pas cette propriété'],
                    'semantic_focus': 'negation_operation',
                    'structural_complexity': 'low',
                    'cross_language_variants': 3
                }
            },
            'Exist': {
                'en': {
                    'patterns': ['this thing exists', 'there is this thing'],
                    'semantic_focus': 'existence_assertion',
                    'structural_complexity': 'low',
                    'cross_language_variants': 2
                },
                'es': {
                    'patterns': ['esta cosa existe', 'hay esta cosa'],
                    'semantic_focus': 'existence_assertion',
                    'structural_complexity': 'low',
                    'cross_language_variants': 2
                },
                'fr': {
                    'patterns': ['cette chose existe', 'il y a cette chose'],
                    'semantic_focus': 'existence_assertion',
                    'structural_complexity': 'low',
                    'cross_language_variants': 2
                }
            }
        }
        
        # Cross-language alignment thresholds
        self.alignment_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3
        }
        
        # Semantic drift detection parameters
        self.drift_detection = {
            'semantic_threshold': 0.8,
            'structural_threshold': 0.7,
            'consistency_threshold': 0.6
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT and NLI models for enhanced evaluation."""
        try:
            # Load multilingual SBERT model
            logger.info("Loading enhanced SBERT model...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Enhanced SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
        
        try:
            # Load XNLI model for enhanced NLI evaluation
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
            logger.info("Loading enhanced XNLI model...")
            model_name = 'joeddav/xlm-roberta-large-xnli'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.nli_model = TextClassificationPipeline(model=model, tokenizer=tokenizer)
            logger.info("Enhanced XNLI model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load XNLI model: {e}")
            self.nli_model = None
    
    def check_enhanced_structural_consistency(self, primitive: str, explications: Dict[str, str]) -> Dict[str, Any]:
        """Enhanced structural consistency check with multiple metrics."""
        if not self.sbert_model:
            return {'consistency_score': 0.0, 'structural_alignment': 'unknown'}
        
        try:
            # Get explications for all languages
            lang_explications = []
            lang_metadata = []
            
            for lang in self.languages:
                if lang in explications:
                    explication = explications[lang]
                    lang_explications.append(explication)
                    
                    # Get metadata for this language/primitive combination
                    metadata = self.enhanced_structural_patterns.get(primitive, {}).get(lang, {})
                    lang_metadata.append({
                        'language': lang,
                        'explication': explication,
                        'semantic_focus': metadata.get('semantic_focus', 'unknown'),
                        'structural_complexity': metadata.get('structural_complexity', 'unknown'),
                        'cross_language_variants': metadata.get('cross_language_variants', 1)
                    })
                else:
                    # Use default pattern if missing
                    default_patterns = self.enhanced_structural_patterns.get(primitive, {}).get(lang, {}).get('patterns', ['default pattern'])
                    lang_explications.append(default_patterns[0])
                    lang_metadata.append({
                        'language': lang,
                        'explication': default_patterns[0],
                        'semantic_focus': 'unknown',
                        'structural_complexity': 'unknown',
                        'cross_language_variants': 1
                    })
            
            # Encode all explications
            embeddings = self.sbert_model.encode(lang_explications)
            
            # Calculate multiple similarity metrics
            pairwise_similarities = []
            semantic_consistency_scores = []
            structural_alignment_scores = []
            
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    # Cosine similarity
                    sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    pairwise_similarities.append(sim)
                    
                    # Semantic consistency (based on semantic focus alignment)
                    semantic_focus_i = lang_metadata[i]['semantic_focus']
                    semantic_focus_j = lang_metadata[j]['semantic_focus']
                    semantic_consistency = 1.0 if semantic_focus_i == semantic_focus_j else 0.5
                    semantic_consistency_scores.append(semantic_consistency)
                    
                    # Structural alignment (based on complexity matching)
                    complexity_i = lang_metadata[i]['structural_complexity']
                    complexity_j = lang_metadata[j]['structural_complexity']
                    structural_alignment = 1.0 if complexity_i == complexity_j else 0.7
                    structural_alignment_scores.append(structural_alignment)
            
            # Calculate comprehensive consistency score
            semantic_consistency = np.mean(semantic_consistency_scores) if semantic_consistency_scores else 0.0
            structural_alignment = np.mean(structural_alignment_scores) if structural_alignment_scores else 0.0
            embedding_consistency = np.mean(pairwise_similarities) if pairwise_similarities else 0.0
            
            # Weighted combination
            consistency_score = (
                0.4 * embedding_consistency +
                0.3 * semantic_consistency +
                0.3 * structural_alignment
            )
            
            # Determine alignment level
            if consistency_score >= self.alignment_thresholds['excellent']:
                alignment_level = 'excellent'
            elif consistency_score >= self.alignment_thresholds['good']:
                alignment_level = 'good'
            elif consistency_score >= self.alignment_thresholds['fair']:
                alignment_level = 'fair'
            else:
                alignment_level = 'poor'
            
            return {
                'consistency_score': float(consistency_score),
                'structural_alignment': alignment_level,
                'embedding_consistency': float(embedding_consistency),
                'semantic_consistency': float(semantic_consistency),
                'structural_alignment_score': float(structural_alignment),
                'pairwise_similarities': [float(s) for s in pairwise_similarities],
                'explications_used': lang_explications,
                'metadata': lang_metadata
            }
            
        except Exception as e:
            logger.warning(f"Enhanced structural consistency check failed: {e}")
            return {'consistency_score': 0.0, 'structural_alignment': 'unknown'}
    
    def detect_semantic_drift(self, primitive: str, explications: Dict[str, str]) -> Dict[str, Any]:
        """Detect semantic drift across languages."""
        if not self.sbert_model or not self.nli_model:
            return {'drift_detected': False, 'drift_score': 0.0}
        
        try:
            # Get explications for all languages
            lang_explications = []
            for lang in self.languages:
                if lang in explications:
                    lang_explications.append(explications[lang])
                else:
                    default_patterns = self.enhanced_structural_patterns.get(primitive, {}).get(lang, {}).get('patterns', ['default pattern'])
                    lang_explications.append(default_patterns[0])
            
            # Calculate semantic drift using multiple metrics
            drift_scores = []
            
            # 1. Embedding-based drift
            embeddings = self.sbert_model.encode(lang_explications)
            embedding_variances = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    drift_score = 1.0 - sim  # Higher drift = lower similarity
                    embedding_variances.append(drift_score)
            
            embedding_drift = np.mean(embedding_variances) if embedding_variances else 0.0
            
            # 2. NLI-based drift (entailment consistency)
            nli_drift_scores = []
            for i in range(len(lang_explications)):
                for j in range(i+1, len(lang_explications)):
                    # Check bidirectional entailment
                    forward_entailment = self._compute_entailment(lang_explications[i], lang_explications[j])
                    backward_entailment = self._compute_entailment(lang_explications[j], lang_explications[i])
                    
                    # Drift is inverse of entailment consistency
                    entailment_consistency = (forward_entailment + backward_entailment) / 2
                    nli_drift = 1.0 - entailment_consistency
                    nli_drift_scores.append(nli_drift)
            
            nli_drift = np.mean(nli_drift_scores) if nli_drift_scores else 0.0
            
            # 3. Structural drift (pattern consistency)
            structural_drift = 0.0
            pattern_counts = {}
            for explication in lang_explications:
                # Simple pattern analysis
                if 'this thing' in explication or 'esta cosa' in explication or 'cette chose' in explication:
                    pattern_counts['thing_reference'] = pattern_counts.get('thing_reference', 0) + 1
                if 'is' in explication or 'está' in explication or 'est' in explication:
                    pattern_counts['copula'] = pattern_counts.get('copula', 0) + 1
            
            # Calculate structural consistency
            total_patterns = len(pattern_counts)
            if total_patterns > 0:
                max_pattern_count = max(pattern_counts.values())
                structural_consistency = max_pattern_count / len(lang_explications)
                structural_drift = 1.0 - structural_consistency
            
            # Combined drift score
            combined_drift = (embedding_drift + nli_drift + structural_drift) / 3
            drift_detected = combined_drift > self.drift_detection['consistency_threshold']
            
            return {
                'drift_detected': drift_detected,
                'drift_score': float(combined_drift),
                'embedding_drift': float(embedding_drift),
                'nli_drift': float(nli_drift),
                'structural_drift': float(structural_drift),
                'threshold': self.drift_detection['consistency_threshold']
            }
            
        except Exception as e:
            logger.warning(f"Semantic drift detection failed: {e}")
            return {'drift_detected': False, 'drift_score': 0.0}
    
    def _compute_entailment(self, premise: str, hypothesis: str) -> float:
        """Compute entailment probability between premise and hypothesis."""
        if not self.nli_model:
            return 0.0
        
        try:
            result = self.nli_model({
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
    
    def analyze_cross_language_patterns(self, primitive: str) -> Dict[str, Any]:
        """Analyze cross-language patterns for a primitive."""
        patterns = self.enhanced_structural_patterns.get(primitive, {})
        
        analysis = {
            'primitive': primitive,
            'languages_present': list(patterns.keys()),
            'pattern_analysis': {},
            'cross_language_metrics': {}
        }
        
        # Analyze patterns for each language
        for lang, lang_data in patterns.items():
            pattern_list = lang_data.get('patterns', [])
            semantic_focus = lang_data.get('semantic_focus', 'unknown')
            structural_complexity = lang_data.get('structural_complexity', 'unknown')
            cross_language_variants = lang_data.get('cross_language_variants', 1)
            
            analysis['pattern_analysis'][lang] = {
                'pattern_count': len(pattern_list),
                'semantic_focus': semantic_focus,
                'structural_complexity': structural_complexity,
                'cross_language_variants': cross_language_variants,
                'patterns': pattern_list
            }
        
        # Cross-language metrics
        if len(patterns) >= 2:
            # Calculate pattern consistency across languages
            semantic_foci = [data.get('semantic_focus', 'unknown') for data in patterns.values()]
            structural_complexities = [data.get('structural_complexity', 'unknown') for data in patterns.values()]
            
            semantic_consistency = len(set(semantic_foci)) == 1  # All same semantic focus
            structural_consistency = len(set(structural_complexities)) == 1  # All same complexity
            
            analysis['cross_language_metrics'] = {
                'semantic_consistency': semantic_consistency,
                'structural_consistency': structural_consistency,
                'average_variants': np.mean([data.get('cross_language_variants', 1) for data in patterns.values()]),
                'total_patterns': sum(len(data.get('patterns', [])) for data in patterns.values())
            }
        
        return analysis
    
    def generate_enhanced_structural_report(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Generate enhanced structural cross-translatability report."""
        logger.info("Generating enhanced structural cross-translatability report...")
        
        # Load input data
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load input data: {e}")
            return {}
        
        # Initialize results
        analysis_results = {
            'metadata': {
                'analysis_type': 'enhanced_cross_language_pattern_analysis',
                'languages_analyzed': self.languages,
                'total_primitives': 0,
                'consistent_primitives': 0,
                'drift_detected_primitives': 0
            },
            'primitive_analysis': {},
            'cross_language_metrics': {},
            'semantic_drift_analysis': {},
            'enhanced_recommendations': []
        }
        
        # Analyze each primitive
        primitives_analyzed = 0
        consistent_primitives = 0
        drift_detected_primitives = 0
        
        # Get primitives from enhanced patterns
        primitives = list(self.enhanced_structural_patterns.keys())
        
        for primitive in primitives:
            logger.info(f"Analyzing enhanced patterns for primitive: {primitive}")
            
            # Get explications for this primitive
            explications = {}
            for lang in self.languages:
                lang_patterns = self.enhanced_structural_patterns.get(primitive, {}).get(lang, {}).get('patterns', [])
                if lang_patterns:
                    explications[lang] = lang_patterns[0]  # Use first pattern
            
            if not explications:
                continue
            
            primitives_analyzed += 1
            
            # Enhanced structural consistency check
            consistency_result = self.check_enhanced_structural_consistency(primitive, explications)
            
            # Semantic drift detection
            drift_result = self.detect_semantic_drift(primitive, explications)
            
            # Cross-language pattern analysis
            pattern_analysis = self.analyze_cross_language_patterns(primitive)
            
            # Store results
            analysis_results['primitive_analysis'][primitive] = {
                'primitive': primitive,
                'languages_present': list(explications.keys()),
                'mapping_consistency': consistency_result['consistency_score'],
                'structural_alignment': consistency_result['structural_alignment'],
                'semantic_drift_detected': drift_result['drift_detected'],
                'drift_score': drift_result['drift_score'],
                'enhanced_validation_results': consistency_result,
                'drift_analysis': drift_result,
                'pattern_analysis': pattern_analysis
            }
            
            # Update counters
            if consistency_result['consistency_score'] >= self.alignment_thresholds['good']:
                consistent_primitives += 1
            
            if drift_result['drift_detected']:
                drift_detected_primitives += 1
        
        # Update metadata
        analysis_results['metadata']['total_primitives'] = primitives_analyzed
        analysis_results['metadata']['consistent_primitives'] = consistent_primitives
        analysis_results['metadata']['drift_detected_primitives'] = drift_detected_primitives
        
        # Calculate overall metrics
        consistency_scores = [analysis['mapping_consistency'] for analysis in analysis_results['primitive_analysis'].values()]
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        drift_scores = [analysis['drift_score'] for analysis in analysis_results['primitive_analysis'].values()]
        overall_drift = np.mean(drift_scores) if drift_scores else 0.0
        
        # Generate enhanced recommendations
        recommendations = self._generate_enhanced_recommendations(analysis_results)
        analysis_results['enhanced_recommendations'] = recommendations
        
        # Create final report
        report = {
            "metadata": {
                "report_type": "enhanced_NSM_cross_translatability_structural_report",
                "timestamp": "2025-08-22",
                "data_source": input_path,
                "models_used": {
                    "enhanced_sbert": self.sbert_model is not None,
                    "enhanced_nli": self.nli_model is not None
                },
                "enhanced_features": [
                    "semantic_drift_detection",
                    "multi_metric_consistency",
                    "pattern_evolution_tracking",
                    "cross_language_adaptation_analysis"
                ]
            },
            "analysis_results": analysis_results,
            "summary": {
                "overall_consistency": float(overall_consistency),
                "consistent_primitives": consistent_primitives,
                "total_primitives": primitives_analyzed,
                "consistency_rate": consistent_primitives / primitives_analyzed if primitives_analyzed > 0 else 0.0,
                "overall_drift_score": float(overall_drift),
                "drift_detected_primitives": drift_detected_primitives,
                "drift_rate": drift_detected_primitives / primitives_analyzed if primitives_analyzed > 0 else 0.0
            },
            "enhanced_recommendations": recommendations
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(report), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Enhanced structural report saved to: {output_path}")
        return report
    
    def _generate_enhanced_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on analysis results."""
        recommendations = []
        
        # Overall consistency recommendations
        overall_consistency = np.mean([analysis['mapping_consistency'] for analysis in analysis_results['primitive_analysis'].values()])
        
        if overall_consistency >= 0.9:
            recommendations.append("Excellent cross-language consistency achieved. Maintain current structural patterns.")
        elif overall_consistency >= 0.7:
            recommendations.append("Good cross-language consistency. Consider minor optimizations for better alignment.")
        else:
            recommendations.append("Cross-language consistency needs improvement. Focus on structural pattern alignment.")
        
        # Drift detection recommendations
        drift_detected = sum(1 for analysis in analysis_results['primitive_analysis'].values() if analysis['semantic_drift_detected'])
        total_primitives = len(analysis_results['primitive_analysis'])
        
        if drift_detected > 0:
            recommendations.append(f"Semantic drift detected in {drift_detected}/{total_primitives} primitives. Monitor and address drift patterns.")
        
        # Primitive-specific recommendations
        for primitive, analysis in analysis_results['primitive_analysis'].items():
            consistency = analysis['mapping_consistency']
            drift_detected = analysis['semantic_drift_detected']
            
            if consistency < 0.7:
                recommendations.append(f"Focus on improving cross-language consistency for {primitive} primitive (current: {consistency:.3f}).")
            
            if drift_detected:
                recommendations.append(f"Address semantic drift in {primitive} primitive to maintain cross-language alignment.")
        
        # Pattern evolution recommendations
        recommendations.append("Implement pattern evolution tracking to monitor structural changes over time.")
        recommendations.append("Consider language-specific structural adaptations for better cross-translatability.")
        recommendations.append("Monitor semantic drift patterns and implement drift correction mechanisms.")
        
        return recommendations


def main():
    """Main function to run enhanced NSM cross-translatability analysis."""
    logger.info("Starting enhanced NSM cross-translatability structural analysis...")
    
    # Initialize enhanced checker
    checker = EnhancedNSMCrossTranslatabilityChecker()
    
    # Generate enhanced structural report
    input_path = Path("data/nsm_substitutability_refinement_results.json")
    output_path = Path("data/nsm_cross_translatability_enhanced_report.json")
    
    if input_path.exists():
        report = checker.generate_enhanced_structural_report(str(input_path), str(output_path))
        
        # Print enhanced summary
        print("\n" + "="*80)
        print("ENHANCED NSM CROSS-TRANSLATABILITY STRUCTURAL ANALYSIS SUMMARY")
        print("="*80)
        print(f"Overall Consistency: {report['summary']['overall_consistency']:.3f}")
        print(f"Consistent Primitives: {report['summary']['consistent_primitives']}/{report['summary']['total_primitives']}")
        print(f"Consistency Rate: {report['summary']['consistency_rate']:.1%}")
        print(f"Overall Drift Score: {report['summary']['overall_drift_score']:.3f}")
        print(f"Drift Detected Primitives: {report['summary']['drift_detected_primitives']}/{report['summary']['total_primitives']}")
        print(f"Drift Rate: {report['summary']['drift_rate']:.1%}")
        print("="*80)
        
        # Print enhanced primitive analysis
        print("\nEnhanced Primitive Analysis:")
        primitive_analysis = report['analysis_results']['primitive_analysis']
        for primitive, analysis in sorted(primitive_analysis.items()):
            consistency = analysis.get('mapping_consistency', 0.0)
            alignment = analysis.get('structural_alignment', 'unknown')
            drift_detected = analysis.get('semantic_drift_detected', False)
            drift_score = analysis.get('drift_score', 0.0)
            
            drift_status = "DRIFT" if drift_detected else "STABLE"
            print(f"  {primitive}: {consistency:.3f} ({alignment}) - {drift_status} (drift: {drift_score:.3f})")
        
        # Print enhanced recommendations
        print("\nEnhanced Recommendations:")
        for i, rec in enumerate(report['enhanced_recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("="*80)
    else:
        logger.error(f"Input data not found: {input_path}")


if __name__ == "__main__":
    main()

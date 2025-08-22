#!/usr/bin/env python3
"""
Detection Evidence Logger System.

This script implements comprehensive evidence logging for primitive detection
to ensure every detection is auditable and trustworthy.
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
from dataclasses import dataclass, asdict
from enum import Enum

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DetectionType(Enum):
    """Types of primitive detection."""
    NSM = "nsm"
    CONCEPTNET = "conceptnet"


class DetectionMethod(Enum):
    """Methods used for primitive detection."""
    PATTERN = "pattern"
    UD_DEPENDENCY = "ud_dependency"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    WORD_BASED = "word_based"
    PHRASE_BASED = "phrase_based"


@dataclass
class DetectionEvidence:
    """Evidence for a primitive detection."""
    primitive: str
    detection_type: DetectionType
    detection_method: DetectionMethod
    confidence: float
    source_text: str
    language: str
    evidence_text: str
    evidence_type: str  # "dep_path", "token", "nearest_synset", "pattern_match", etc.
    evidence_details: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'primitive': self.primitive,
            'detection_type': self.detection_type.value,
            'detection_method': self.detection_method.value,
            'confidence': self.confidence,
            'source_text': self.source_text,
            'language': self.language,
            'evidence_text': self.evidence_text,
            'evidence_type': self.evidence_type,
            'evidence_details': self.evidence_details,
            'timestamp': self.timestamp
        }


class DetectionEvidenceLogger:
    """Comprehensive detection evidence logging system."""
    
    def __init__(self, log_file_path: str = "data/detection_evidence_log.json"):
        """Initialize the evidence logger."""
        self.log_file_path = log_file_path
        self.evidence_log = []
        self.detection_stats = defaultdict(int)
        
        # Load existing log if it exists
        self._load_existing_log()
    
    def _load_existing_log(self):
        """Load existing evidence log if it exists."""
        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.evidence_log = data.get('evidence_log', [])
                    self.detection_stats = defaultdict(int, data.get('detection_stats', {}))
                logger.info(f"Loaded existing evidence log with {len(self.evidence_log)} entries")
        except Exception as e:
            logger.warning(f"Failed to load existing evidence log: {e}")
            self.evidence_log = []
            self.detection_stats = defaultdict(int)
    
    def log_detection(self, evidence: DetectionEvidence) -> None:
        """Log a detection with full evidence."""
        # Validate evidence
        if not self._validate_evidence(evidence):
            logger.warning(f"Invalid evidence detected, skipping: {evidence.primitive}")
            return
        
        # Add to log
        self.evidence_log.append(evidence.to_dict())
        
        # Update statistics
        self.detection_stats[f"{evidence.detection_type.value}_{evidence.detection_method.value}"] += 1
        self.detection_stats[f"total_{evidence.detection_type.value}"] += 1
        self.detection_stats["total_detections"] += 1
        
        # Save periodically
        if len(self.evidence_log) % 100 == 0:
            self._save_log()
    
    def _validate_evidence(self, evidence: DetectionEvidence) -> bool:
        """Validate detection evidence."""
        # Check required fields
        if not evidence.primitive or not evidence.source_text:
            return False
        
        # Check confidence range
        if not (0.0 <= evidence.confidence <= 1.0):
            return False
        
        # Check evidence text is not empty
        if not evidence.evidence_text:
            return False
        
        # Check evidence details is a dict
        if not isinstance(evidence.evidence_details, dict):
            return False
        
        return True
    
    def _save_log(self) -> None:
        """Save evidence log to file."""
        try:
            log_data = {
                'metadata': {
                    'version': '1.0.0',
                    'last_updated': time.time(),
                    'total_entries': len(self.evidence_log)
                },
                'evidence_log': self.evidence_log,
                'detection_stats': dict(self.detection_stats)
            }
            
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved evidence log with {len(self.evidence_log)} entries")
        except Exception as e:
            logger.error(f"Failed to save evidence log: {e}")
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary statistics of detections."""
        if not self.evidence_log:
            return {'total_detections': 0}
        
        # Calculate statistics
        total_detections = len(self.evidence_log)
        nsm_detections = sum(1 for e in self.evidence_log if e['detection_type'] == 'nsm')
        conceptnet_detections = sum(1 for e in self.evidence_log if e['detection_type'] == 'conceptnet')
        
        # Method distribution
        method_dist = defaultdict(int)
        for e in self.evidence_log:
            method_dist[e['detection_method']] += 1
        
        # Confidence statistics
        confidences = [e['confidence'] for e in self.evidence_log]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Evidence type distribution
        evidence_type_dist = defaultdict(int)
        for e in self.evidence_log:
            evidence_type_dist[e['evidence_type']] += 1
        
        return {
            'total_detections': total_detections,
            'nsm_detections': nsm_detections,
            'conceptnet_detections': conceptnet_detections,
            'method_distribution': dict(method_dist),
            'evidence_type_distribution': dict(evidence_type_dist),
            'average_confidence': avg_confidence,
            'confidence_distribution': {
                'high': sum(1 for c in confidences if c >= 0.8),
                'medium': sum(1 for c in confidences if 0.6 <= c < 0.8),
                'low': sum(1 for c in confidences if c < 0.6)
            }
        }
    
    def get_detections_by_primitive(self, primitive: str) -> List[Dict[str, Any]]:
        """Get all detections for a specific primitive."""
        return [e for e in self.evidence_log if e['primitive'] == primitive]
    
    def get_detections_by_method(self, method: DetectionMethod) -> List[Dict[str, Any]]:
        """Get all detections using a specific method."""
        return [e for e in self.evidence_log if e['detection_method'] == method.value]
    
    def get_detections_by_type(self, detection_type: DetectionType) -> List[Dict[str, Any]]:
        """Get all detections of a specific type."""
        return [e for e in self.evidence_log if e['detection_type'] == detection_type.value]
    
    def export_audit_report(self, output_path: str = "data/detection_audit_report.json") -> None:
        """Export comprehensive audit report."""
        summary = self.get_detection_summary()
        
        # Add detailed analysis
        audit_data = {
            'summary': summary,
            'detection_analysis': self._analyze_detections(),
            'evidence_quality': self._analyze_evidence_quality(),
            'method_performance': self._analyze_method_performance(),
            'primitive_coverage': self._analyze_primitive_coverage()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported audit report to {output_path}")
    
    def _analyze_detections(self) -> Dict[str, Any]:
        """Analyze detection patterns and quality."""
        if not self.evidence_log:
            return {}
        
        # Primitive frequency analysis
        primitive_counts = Counter(e['primitive'] for e in self.evidence_log)
        
        # Confidence analysis by primitive
        confidence_by_primitive = defaultdict(list)
        for e in self.evidence_log:
            confidence_by_primitive[e['primitive']].append(e['confidence'])
        
        avg_confidence_by_primitive = {
            prim: np.mean(confs) for prim, confs in confidence_by_primitive.items()
        }
        
        return {
            'primitive_frequency': dict(primitive_counts.most_common(20)),
            'average_confidence_by_primitive': avg_confidence_by_primitive,
            'detection_timeline': {
                'earliest': min(e['timestamp'] for e in self.evidence_log),
                'latest': max(e['timestamp'] for e in self.evidence_log),
                'total_duration': max(e['timestamp'] for e in self.evidence_log) - min(e['timestamp'] for e in self.evidence_log)
            }
        }
    
    def _analyze_evidence_quality(self) -> Dict[str, Any]:
        """Analyze evidence quality and consistency."""
        if not self.evidence_log:
            return {}
        
        # Evidence type analysis
        evidence_type_counts = Counter(e['evidence_type'] for e in self.evidence_log)
        
        # Evidence text length analysis
        evidence_lengths = [len(e['evidence_text']) for e in self.evidence_log]
        
        # Evidence details analysis
        detail_keys = set()
        for e in self.evidence_log:
            detail_keys.update(e['evidence_details'].keys())
        
        return {
            'evidence_type_distribution': dict(evidence_type_counts),
            'evidence_text_length': {
                'average': np.mean(evidence_lengths),
                'median': np.median(evidence_lengths),
                'min': min(evidence_lengths),
                'max': max(evidence_lengths)
            },
            'evidence_detail_keys': list(detail_keys),
            'evidence_quality_indicators': {
                'has_evidence_text': sum(1 for e in self.evidence_log if e['evidence_text']),
                'has_evidence_details': sum(1 for e in self.evidence_log if e['evidence_details']),
                'has_confidence': sum(1 for e in self.evidence_log if 'confidence' in e)
            }
        }
    
    def _analyze_method_performance(self) -> Dict[str, Any]:
        """Analyze performance of different detection methods."""
        if not self.evidence_log:
            return {}
        
        method_stats = defaultdict(lambda: {'count': 0, 'confidences': [], 'primitives': set()})
        
        for e in self.evidence_log:
            method = e['detection_method']
            method_stats[method]['count'] += 1
            method_stats[method]['confidences'].append(e['confidence'])
            method_stats[method]['primitives'].add(e['primitive'])
        
        # Calculate method performance metrics
        method_performance = {}
        for method, stats in method_stats.items():
            method_performance[method] = {
                'count': stats['count'],
                'average_confidence': np.mean(stats['confidences']),
                'unique_primitives': len(stats['primitives']),
                'primitive_coverage': list(stats['primitives'])
            }
        
        return method_performance
    
    def _analyze_primitive_coverage(self) -> Dict[str, Any]:
        """Analyze primitive coverage and distribution."""
        if not self.evidence_log:
            return {}
        
        # NSM vs ConceptNet distribution
        type_counts = Counter(e['detection_type'] for e in self.evidence_log)
        
        # Primitive distribution by type
        nsm_primitives = set(e['primitive'] for e in self.evidence_log if e['detection_type'] == 'nsm')
        conceptnet_primitives = set(e['primitive'] for e in self.evidence_log if e['detection_type'] == 'conceptnet')
        
        return {
            'type_distribution': dict(type_counts),
            'nsm_primitive_coverage': {
                'count': len(nsm_primitives),
                'primitives': list(nsm_primitives)
            },
            'conceptnet_primitive_coverage': {
                'count': len(conceptnet_primitives),
                'primitives': list(conceptnet_primitives)
            },
            'overall_coverage': {
                'total_unique_primitives': len(nsm_primitives | conceptnet_primitives),
                'nsm_percentage': len(nsm_primitives) / len(nsm_primitives | conceptnet_primitives) if (nsm_primitives | conceptnet_primitives) else 0
            }
        }


class EnhancedUnifiedPrimitiveDetector:
    """Enhanced unified primitive detector with evidence logging."""
    
    def __init__(self, evidence_logger: DetectionEvidenceLogger):
        """Initialize the enhanced detector."""
        self.evidence_logger = evidence_logger
        self.sbert_model = None
        
        # Load detection patterns with evidence tracking
        self.detection_patterns = self._load_enhanced_patterns()
        
        # Load semantic model
        self._load_semantic_model()
    
    def _load_enhanced_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load detection patterns with evidence tracking."""
        return {
            # NSM Primitives with evidence tracking
            'nsm_negation': [
                {
                    'pattern': r'\b(not|no|never|none|nobody|nothing|nowhere|neither|nor)\b',
                    'evidence_type': 'lexical_match',
                    'primitives': ['NOT']
                },
                {
                    'pattern': r'\b(doesn\'t|don\'t|didn\'t|won\'t|can\'t|couldn\'t|shouldn\'t|wouldn\'t)\b',
                    'evidence_type': 'contraction_match',
                    'primitives': ['NOT']
                }
            ],
            'nsm_modality': [
                {
                    'pattern': r'\b(can|could|might|may|must|should|will|would|shall)\b',
                    'evidence_type': 'modal_verb',
                    'primitives': ['CAN', 'MUST', 'SHOULD', 'MIGHT', 'WILL']
                }
            ],
            'nsm_quantifier': [
                {
                    'pattern': r'\b(all|some|many|few|most|several|various|numerous|countless)\b',
                    'evidence_type': 'quantifier_word',
                    'primitives': ['ALL', 'SOME', 'MANY', 'FEW', 'MOST']
                }
            ],
            'nsm_evaluator': [
                {
                    'pattern': r'\b(good|bad|nice|terrible|wonderful|awful|excellent|horrible)\b',
                    'evidence_type': 'evaluator_word',
                    'primitives': ['GOOD', 'BAD']
                }
            ],
            'nsm_descriptor': [
                {
                    'pattern': r'\b(big|small|large|little|huge|tiny|long|short|wide|narrow)\b',
                    'evidence_type': 'descriptor_word',
                    'primitives': ['BIG', 'SMALL', 'LONG', 'SHORT']
                }
            ],
            'nsm_mental': [
                {
                    'pattern': r'\b(think|know|want|feel|see|hear|believe|understand|remember|forget)\b',
                    'evidence_type': 'mental_verb',
                    'primitives': ['THINK', 'KNOW', 'WANT', 'FEEL', 'SEE', 'HEAR']
                }
            ],
            'nsm_action': [
                {
                    'pattern': r'\b(do|happen|move|touch|go|come|stay|leave|arrive|depart)\b',
                    'evidence_type': 'action_verb',
                    'primitives': ['DO', 'HAPPEN', 'MOVE', 'TOUCH']
                }
            ],
            'nsm_location': [
                {
                    'pattern': r'\b(here|there|where|above|below|near|far|inside|outside|beside)\b',
                    'evidence_type': 'location_word',
                    'primitives': ['HERE', 'WHERE', 'ABOVE', 'BELOW', 'NEAR', 'FAR']
                }
            ],
            'nsm_time': [
                {
                    'pattern': r'\b(now|then|when|before|after|during|while|since|until|always|never)\b',
                    'evidence_type': 'temporal_word',
                    'primitives': ['NOW', 'WHEN', 'BEFORE', 'AFTER']
                }
            ],
            'nsm_causation': [
                {
                    'pattern': r'\b(because|since|as|due to|result in|lead to|cause|caused)\b',
                    'evidence_type': 'causal_word',
                    'primitives': ['BECAUSE']
                }
            ],
            'nsm_similarity': [
                {
                    'pattern': r'\b(like|similar|same|different|alike|unlike|resemble|match)\b',
                    'evidence_type': 'similarity_word',
                    'primitives': ['LIKE', 'SIMILAR', 'DIFFERENT']
                }
            ],
            
            # ConceptNet Relations with evidence tracking
            'conceptnet_spatial': [
                {
                    'pattern': r'\b(at|in|on|under|over|beside|near|far|inside|outside)\b',
                    'evidence_type': 'spatial_preposition',
                    'primitives': ['AtLocation']
                }
            ],
            'conceptnet_causal': [
                {
                    'pattern': r'\b(cause|caused|because|since|as|result|effect|consequence)\b',
                    'evidence_type': 'causal_word',
                    'primitives': ['Causes']
                }
            ],
            'conceptnet_similarity': [
                {
                    'pattern': r'\b(similar|like|same|different|alike|unlike|resemble|match)\b',
                    'evidence_type': 'similarity_word',
                    'primitives': ['SimilarTo']
                }
            ],
            'conceptnet_structural': [
                {
                    'pattern': r'\b(part of|piece of|component of|element of|member of)\b',
                    'evidence_type': 'part_whole_phrase',
                    'primitives': ['PartOf']
                }
            ],
            'conceptnet_functional': [
                {
                    'pattern': r'\b(used for|purpose|function|serve|designed for|intended for)\b',
                    'evidence_type': 'purpose_phrase',
                    'primitives': ['UsedFor']
                }
            ],
            'conceptnet_cognitive': [
                {
                    'pattern': r'\b(can do|able to|capable of|know how to|skilled at)\b',
                    'evidence_type': 'ability_phrase',
                    'primitives': ['CapableOf']
                }
            ]
        }
    
    def _load_semantic_model(self):
        """Load SBERT model for semantic detection."""
        try:
            logger.info("Loading SBERT model for semantic primitive detection...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def detect_primitives_with_evidence(self, text: str, language: str = "en") -> List[DetectionEvidence]:
        """Detect primitives with comprehensive evidence logging."""
        detected_evidence = []
        
        # Pattern-based detection with evidence
        pattern_evidence = self._detect_via_patterns_with_evidence(text, language)
        detected_evidence.extend(pattern_evidence)
        
        # Semantic detection with evidence
        if self.sbert_model:
            semantic_evidence = self._detect_via_semantics_with_evidence(text, language)
            detected_evidence.extend(semantic_evidence)
        
        # Log all evidence
        for evidence in detected_evidence:
            self.evidence_logger.log_detection(evidence)
        
        return detected_evidence
    
    def _detect_via_patterns_with_evidence(self, text: str, language: str) -> List[DetectionEvidence]:
        """Detect primitives using pattern matching with evidence."""
        detected_evidence = []
        text_lower = text.lower()
        
        for pattern_category, patterns in self.detection_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                evidence_type = pattern_info['evidence_type']
                primitives = pattern_info['primitives']
                
                match = re.search(pattern, text_lower)
                if match:
                    matched_text = match.group(0)
                    
                    for primitive in primitives:
                        evidence = DetectionEvidence(
                            primitive=primitive,
                            detection_type=DetectionType.NSM if primitive in ['NOT', 'CAN', 'MUST', 'WANT', 'THINK', 'KNOW', 'FEEL', 'SEE', 'HEAR', 'DO', 'HAPPEN', 'BECAUSE', 'LIKE', 'ALL', 'SOME', 'MANY', 'FEW', 'GOOD', 'BAD', 'BIG', 'SMALL', 'HERE', 'WHERE', 'NOW', 'BEFORE', 'AFTER'] else DetectionType.CONCEPTNET,
                            detection_method=DetectionMethod.PATTERN,
                            confidence=0.7,
                            source_text=text,
                            language=language,
                            evidence_text=matched_text,
                            evidence_type=evidence_type,
                            evidence_details={
                                'pattern': pattern,
                                'pattern_category': pattern_category,
                                'match_start': match.start(),
                                'match_end': match.end(),
                                'full_match': matched_text
                            },
                            timestamp=time.time()
                        )
                        detected_evidence.append(evidence)
        
        return detected_evidence
    
    def _detect_via_semantics_with_evidence(self, text: str, language: str) -> List[DetectionEvidence]:
        """Detect primitives using semantic similarity with evidence."""
        if not self.sbert_model:
            return []
        
        detected_evidence = []
        
        try:
            # Semantic descriptions for primitives
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
            
            # Calculate semantic similarity
            text_embedding = self.sbert_model.encode([text])[0]
            
            for primitive, description in primitive_descriptions.items():
                desc_embedding = self.sbert_model.encode([description])[0]
                
                similarity = np.dot(text_embedding, desc_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(desc_embedding)
                )
                
                if similarity > 0.3:  # Lower threshold for semantic detection
                    evidence = DetectionEvidence(
                        primitive=primitive,
                        detection_type=DetectionType.NSM if primitive in ['NOT', 'CAN', 'MUST', 'WANT', 'THINK', 'KNOW', 'FEEL', 'SEE', 'HEAR', 'DO', 'HAPPEN', 'BECAUSE', 'LIKE', 'ALL', 'SOME', 'MANY', 'FEW', 'GOOD', 'BAD', 'BIG', 'SMALL', 'HERE', 'WHERE', 'NOW', 'BEFORE', 'AFTER'] else DetectionType.CONCEPTNET,
                        detection_method=DetectionMethod.SEMANTIC,
                        confidence=float(similarity),
                        source_text=text,
                        language=language,
                        evidence_text=description,
                        evidence_type='semantic_similarity',
                        evidence_details={
                            'similarity_score': float(similarity),
                            'description': description,
                            'threshold': 0.3
                        },
                        timestamp=time.time()
                    )
                    detected_evidence.append(evidence)
        
        except Exception as e:
            logger.warning(f"Semantic detection failed: {e}")
        
        return detected_evidence


def main():
    """Main function to test the detection evidence logging system."""
    logger.info("Starting detection evidence logging system test...")
    
    # Initialize evidence logger
    evidence_logger = DetectionEvidenceLogger()
    
    # Initialize enhanced detector
    detector = EnhancedUnifiedPrimitiveDetector(evidence_logger)
    
    # Test texts
    test_texts = [
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
        "The part of the machine",
        "This tool is used for cutting",
        "He can do many things",
        "She wants to learn",
        "The cat is here now"
    ]
    
    # Run detection with evidence logging
    total_detections = 0
    for text in test_texts:
        evidence_list = detector.detect_primitives_with_evidence(text, "en")
        total_detections += len(evidence_list)
        
        print(f"\nText: {text}")
        print(f"Detections: {len(evidence_list)}")
        for evidence in evidence_list[:3]:  # Show first 3 detections
            print(f"  - {evidence.primitive} ({evidence.detection_method.value}, {evidence.evidence_type}, conf: {evidence.confidence:.3f})")
    
    # Save log
    evidence_logger._save_log()
    
    # Generate summary
    summary = evidence_logger.get_detection_summary()
    
    print(f"\n" + "="*80)
    print("DETECTION EVIDENCE LOGGING SUMMARY")
    print("="*80)
    print(f"Total Detections: {summary['total_detections']}")
    print(f"NSM Detections: {summary['nsm_detections']}")
    print(f"ConceptNet Detections: {summary['conceptnet_detections']}")
    print(f"Average Confidence: {summary['average_confidence']:.3f}")
    
    print(f"\nMethod Distribution:")
    for method, count in summary['method_distribution'].items():
        print(f"  {method}: {count}")
    
    print(f"\nEvidence Type Distribution:")
    for ev_type, count in summary['evidence_type_distribution'].items():
        print(f"  {ev_type}: {count}")
    
    print(f"\nConfidence Distribution:")
    for level, count in summary['confidence_distribution'].items():
        print(f"  {level}: {count}")
    
    # Export audit report
    evidence_logger.export_audit_report()
    
    print(f"\n" + "="*80)
    print("Detection evidence logging test completed!")
    print("="*80)


if __name__ == "__main__":
    main()

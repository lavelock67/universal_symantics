#!/usr/bin/env python3
"""
Mining Sprint Operationalization System.

This script implements the mining sprint to operationalize idea-prime mining
and turn recurring friction patterns into actionable molecules/primes as
specified in the ChatGPT5 roadmap.
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


class FrictionType(Enum):
    """Types of recurring friction patterns."""
    SEMANTIC_AMBIGUITY = "semantic_ambiguity"
    CROSS_LANGUAGE_MISMATCH = "cross_language_mismatch"
    PRIMITIVE_DETECTION_FAILURE = "primitive_detection_failure"
    TRANSLATION_QUALITY_ISSUE = "translation_quality_issue"
    EXPLICATION_COMPLEXITY = "explication_complexity"
    GRAMMAR_VIOLATION = "grammar_violation"
    CONTEXT_DEPENDENCY = "context_dependency"
    CULTURAL_NUANCE = "cultural_nuance"
    TEMPORAL_ASPECT = "temporal_aspect"
    MODALITY_CONFUSION = "modality_confusion"


class MoleculeType(Enum):
    """Types of molecules/primes derived from friction patterns."""
    DISAMBIGUATION_MOLECULE = "disambiguation_molecule"
    CROSS_LANGUAGE_MOLECULE = "cross_language_molecule"
    DETECTION_ENHANCEMENT_MOLECULE = "detection_enhancement_molecule"
    QUALITY_IMPROVEMENT_MOLECULE = "quality_improvement_molecule"
    COMPLEXITY_REDUCTION_MOLECULE = "complexity_reduction_molecule"
    GRAMMAR_CORRECTION_MOLECULE = "grammar_correction_molecule"
    CONTEXT_ADAPTATION_MOLECULE = "context_adaptation_molecule"
    CULTURAL_ADAPTATION_MOLECULE = "cultural_adaptation_molecule"
    TEMPORAL_MOLECULE = "temporal_molecule"
    MODALITY_MOLECULE = "modality_molecule"


@dataclass
class FrictionPattern:
    """A recurring friction pattern identified in the system."""
    pattern_id: str
    friction_type: FrictionType
    description: str
    examples: List[str]
    frequency: int
    impact_score: float
    affected_components: List[str]
    detection_triggers: List[str]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'pattern_id': self.pattern_id,
            'friction_type': self.friction_type.value,
            'description': self.description,
            'examples': self.examples,
            'frequency': self.frequency,
            'impact_score': self.impact_score,
            'affected_components': self.affected_components,
            'detection_triggers': self.detection_triggers,
            'timestamp': self.timestamp
        }


@dataclass
class FrictionMolecule:
    """A molecule/prime derived from a friction pattern."""
    molecule_id: str
    source_pattern: str
    molecule_type: MoleculeType
    name: str
    description: str
    implementation: Dict[str, Any]
    confidence: float
    expected_impact: float
    validation_tests: List[str]
    dependencies: List[str]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'molecule_id': self.molecule_id,
            'source_pattern': self.source_pattern,
            'molecule_type': self.molecule_type.value,
            'name': self.name,
            'description': self.description,
            'implementation': self.implementation,
            'confidence': self.confidence,
            'expected_impact': self.expected_impact,
            'validation_tests': self.validation_tests,
            'dependencies': self.dependencies,
            'timestamp': self.timestamp
        }


class FrictionPatternDetector:
    """Detects recurring friction patterns in the system."""
    
    def __init__(self):
        """Initialize the friction pattern detector."""
        self.sbert_model = None
        self._load_semantic_model()
        
        # Friction detection patterns
        self.friction_patterns = {
            FrictionType.SEMANTIC_AMBIGUITY: {
                'triggers': ['multiple_senses', 'context_dependent', 'polysemous'],
                'examples': ['bank', 'run', 'light', 'fair']
            },
            FrictionType.CROSS_LANGUAGE_MISMATCH: {
                'triggers': ['translation_failure', 'semantic_drift', 'cultural_mismatch'],
                'examples': ['polite_requests', 'temporal_expressions', 'spatial_relations']
            },
            FrictionType.PRIMITIVE_DETECTION_FAILURE: {
                'triggers': ['low_confidence', 'missing_primitives', 'false_positives'],
                'examples': ['complex_phrases', 'idiomatic_expressions', 'technical_terms']
            },
            FrictionType.TRANSLATION_QUALITY_ISSUE: {
                'triggers': ['low_similarity', 'grammar_errors', 'fluency_issues'],
                'examples': ['long_sentences', 'complex_structures', 'domain_specific']
            },
            FrictionType.EXPLICATION_COMPLEXITY: {
                'triggers': ['high_complexity', 'poor_readability', 'structural_issues'],
                'examples': ['nested_structures', 'multiple_clauses', 'abstract_concepts']
            }
        }
    
    def _load_semantic_model(self):
        """Load SBERT model for semantic analysis."""
        try:
            logger.info("Loading SBERT model for friction pattern detection...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def detect_friction_patterns(self, system_logs: List[Dict[str, Any]]) -> List[FrictionPattern]:
        """Detect recurring friction patterns from system logs."""
        logger.info(f"Detecting friction patterns from {len(system_logs)} system logs")
        
        friction_patterns = []
        pattern_counts = defaultdict(int)
        pattern_examples = defaultdict(list)
        pattern_components = defaultdict(set)
        
        # Analyze system logs for friction patterns
        for log_entry in system_logs:
            error_type = log_entry.get('error_type', '')
            error_message = log_entry.get('error_message', '')
            component = log_entry.get('component', '')
            timestamp = log_entry.get('timestamp', time.time())
            
            # Detect friction type based on error patterns
            friction_type = self._classify_friction_type(error_type, error_message)
            
            if friction_type:
                pattern_key = f"{friction_type.value}_{error_type}"
                pattern_counts[pattern_key] += 1
                pattern_examples[pattern_key].append(error_message)
                pattern_components[pattern_key].add(component)
        
        # Create friction pattern objects
        for pattern_key, count in pattern_counts.items():
            if count >= 3:  # Minimum frequency threshold
                friction_type_str = pattern_key.split('_')[0]
                friction_type = FrictionType(friction_type_str)
                
                # Calculate impact score
                impact_score = self._calculate_impact_score(count, pattern_examples[pattern_key])
                
                # Create friction pattern
                pattern = FrictionPattern(
                    pattern_id=pattern_key,
                    friction_type=friction_type,
                    description=f"Recurring {friction_type.value} in {', '.join(pattern_components[pattern_key])}",
                    examples=pattern_examples[pattern_key][:5],  # Top 5 examples
                    frequency=count,
                    impact_score=impact_score,
                    affected_components=list(pattern_components[pattern_key]),
                    detection_triggers=self.friction_patterns[friction_type]['triggers'],
                    timestamp=time.time()
                )
                friction_patterns.append(pattern)
        
        # Sort by impact score
        friction_patterns.sort(key=lambda x: x.impact_score, reverse=True)
        
        logger.info(f"Detected {len(friction_patterns)} friction patterns")
        return friction_patterns
    
    def _classify_friction_type(self, error_type: str, error_message: str) -> Optional[FrictionType]:
        """Classify friction type based on error information."""
        error_lower = error_message.lower()
        
        # Semantic ambiguity
        if any(trigger in error_lower for trigger in ['ambiguous', 'multiple senses', 'context']):
            return FrictionType.SEMANTIC_AMBIGUITY
        
        # Cross-language mismatch
        if any(trigger in error_lower for trigger in ['translation', 'language', 'cross-lingual']):
            return FrictionType.CROSS_LANGUAGE_MISMATCH
        
        # Primitive detection failure
        if any(trigger in error_lower for trigger in ['detection', 'primitive', 'confidence']):
            return FrictionType.PRIMITIVE_DETECTION_FAILURE
        
        # Translation quality issue
        if any(trigger in error_lower for trigger in ['quality', 'similarity', 'fluency']):
            return FrictionType.TRANSLATION_QUALITY_ISSUE
        
        # Explication complexity
        if any(trigger in error_lower for trigger in ['complexity', 'readability', 'structure']):
            return FrictionType.EXPLICATION_COMPLEXITY
        
        return None
    
    def _calculate_impact_score(self, frequency: int, examples: List[str]) -> float:
        """Calculate impact score for a friction pattern."""
        # Base score from frequency
        base_score = min(frequency / 10.0, 1.0)
        
        # Complexity penalty from examples
        avg_length = np.mean([len(ex) for ex in examples])
        complexity_penalty = min(avg_length / 100.0, 0.3)
        
        return base_score - complexity_penalty


class MoleculeGenerator:
    """Generates molecules/primes from friction patterns."""
    
    def __init__(self):
        """Initialize the molecule generator."""
        self.molecule_templates = {
            MoleculeType.DISAMBIGUATION_MOLECULE: {
                'name_template': 'Disambiguate_{pattern}',
                'description_template': 'Resolve semantic ambiguity in {pattern}',
                'implementation_template': {
                    'method': 'context_aware_disambiguation',
                    'confidence_threshold': 0.8,
                    'fallback_strategy': 'most_frequent_sense'
                }
            },
            MoleculeType.CROSS_LANGUAGE_MOLECULE: {
                'name_template': 'CrossLang_{pattern}',
                'description_template': 'Handle cross-language mismatch in {pattern}',
                'implementation_template': {
                    'method': 'semantic_alignment',
                    'alignment_threshold': 0.7,
                    'cultural_adaptation': True
                }
            },
            MoleculeType.DETECTION_ENHANCEMENT_MOLECULE: {
                'name_template': 'Detect_{pattern}',
                'description_template': 'Enhance primitive detection for {pattern}',
                'implementation_template': {
                    'method': 'pattern_enhancement',
                    'confidence_boost': 0.2,
                    'fallback_detection': True
                }
            },
            MoleculeType.QUALITY_IMPROVEMENT_MOLECULE: {
                'name_template': 'Quality_{pattern}',
                'description_template': 'Improve translation quality for {pattern}',
                'implementation_template': {
                    'method': 'quality_enhancement',
                    'similarity_threshold': 0.8,
                    'fluency_check': True
                }
            },
            MoleculeType.COMPLEXITY_REDUCTION_MOLECULE: {
                'name_template': 'Simplify_{pattern}',
                'description_template': 'Reduce complexity in {pattern}',
                'implementation_template': {
                    'method': 'complexity_reduction',
                    'max_complexity': 5,
                    'simplification_rules': True
                }
            }
        }
    
    def generate_molecules(self, friction_patterns: List[FrictionPattern]) -> List[FrictionMolecule]:
        """Generate molecules from friction patterns."""
        logger.info(f"Generating molecules from {len(friction_patterns)} friction patterns")
        
        molecules = []
        
        for pattern in friction_patterns:
            # Map friction type to molecule type
            molecule_type = self._map_friction_to_molecule(pattern.friction_type)
            
            if molecule_type:
                # Generate molecule using template
                molecule = self._generate_molecule_from_template(pattern, molecule_type)
                molecules.append(molecule)
        
        # Sort by expected impact
        molecules.sort(key=lambda x: x.expected_impact, reverse=True)
        
        logger.info(f"Generated {len(molecules)} molecules")
        return molecules
    
    def _map_friction_to_molecule(self, friction_type: FrictionType) -> Optional[MoleculeType]:
        """Map friction type to molecule type."""
        mapping = {
            FrictionType.SEMANTIC_AMBIGUITY: MoleculeType.DISAMBIGUATION_MOLECULE,
            FrictionType.CROSS_LANGUAGE_MISMATCH: MoleculeType.CROSS_LANGUAGE_MOLECULE,
            FrictionType.PRIMITIVE_DETECTION_FAILURE: MoleculeType.DETECTION_ENHANCEMENT_MOLECULE,
            FrictionType.TRANSLATION_QUALITY_ISSUE: MoleculeType.QUALITY_IMPROVEMENT_MOLECULE,
            FrictionType.EXPLICATION_COMPLEXITY: MoleculeType.COMPLEXITY_REDUCTION_MOLECULE
        }
        return mapping.get(friction_type)
    
    def _generate_molecule_from_template(self, pattern: FrictionPattern, 
                                       molecule_type: MoleculeType) -> FrictionMolecule:
        """Generate molecule using template."""
        template = self.molecule_templates[molecule_type]
        
        # Generate molecule ID
        molecule_id = f"{molecule_type.value}_{pattern.pattern_id}"
        
        # Generate name and description
        name = template['name_template'].format(pattern=pattern.pattern_id.split('_')[-1])
        description = template['description_template'].format(pattern=pattern.description)
        
        # Get implementation template
        implementation = template['implementation_template'].copy()
        
        # Calculate confidence and expected impact
        confidence = min(pattern.impact_score * 0.8, 0.95)
        expected_impact = pattern.impact_score * 0.7
        
        # Generate validation tests
        validation_tests = self._generate_validation_tests(pattern, molecule_type)
        
        # Determine dependencies
        dependencies = self._determine_dependencies(molecule_type)
        
        return FrictionMolecule(
            molecule_id=molecule_id,
            source_pattern=pattern.pattern_id,
            molecule_type=molecule_type,
            name=name,
            description=description,
            implementation=implementation,
            confidence=confidence,
            expected_impact=expected_impact,
            validation_tests=validation_tests,
            dependencies=dependencies,
            timestamp=time.time()
        )
    
    def _generate_validation_tests(self, pattern: FrictionPattern, 
                                 molecule_type: MoleculeType) -> List[str]:
        """Generate validation tests for molecule."""
        tests = []
        
        # Add pattern-specific tests
        for example in pattern.examples[:3]:  # Use first 3 examples
            tests.append(f"test_{molecule_type.value}_{pattern.pattern_id}_{hash(example) % 1000}")
        
        # Add general validation tests
        tests.extend([
            f"test_{molecule_type.value}_confidence_threshold",
            f"test_{molecule_type.value}_performance_improvement",
            f"test_{molecule_type.value}_regression_prevention"
        ])
        
        return tests
    
    def _determine_dependencies(self, molecule_type: MoleculeType) -> List[str]:
        """Determine dependencies for molecule type."""
        dependencies = {
            MoleculeType.DISAMBIGUATION_MOLECULE: ['context_analyzer', 'sense_disambiguator'],
            MoleculeType.CROSS_LANGUAGE_MOLECULE: ['semantic_aligner', 'cultural_adapter'],
            MoleculeType.DETECTION_ENHANCEMENT_MOLECULE: ['pattern_enhancer', 'confidence_booster'],
            MoleculeType.QUALITY_IMPROVEMENT_MOLECULE: ['quality_enhancer', 'fluency_checker'],
            MoleculeType.COMPLEXITY_REDUCTION_MOLECULE: ['complexity_analyzer', 'simplifier']
        }
        return dependencies.get(molecule_type, [])


class MiningSprintSystem:
    """Comprehensive mining sprint system for operationalizing idea-prime mining."""
    
    def __init__(self):
        """Initialize the mining sprint system."""
        self.pattern_detector = FrictionPatternDetector()
        self.molecule_generator = MoleculeGenerator()
        
        # System parameters
        self.sprint_params = {
            'min_pattern_frequency': 3,
            'min_impact_score': 0.3,
            'max_molecules_per_pattern': 2,
            'validation_threshold': 0.7
        }
    
    def run_mining_sprint(self, system_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive mining sprint analysis."""
        logger.info("Starting mining sprint operationalization...")
        
        sprint_results = {
            'sprint_configuration': {
                'min_pattern_frequency': self.sprint_params['min_pattern_frequency'],
                'min_impact_score': self.sprint_params['min_impact_score'],
                'timestamp': time.time()
            },
            'friction_patterns': [],
            'generated_molecules': [],
            'sprint_analysis': {},
            'operationalization_plan': [],
            'recommendations': []
        }
        
        # Detect friction patterns
        friction_patterns = self.pattern_detector.detect_friction_patterns(system_logs)
        sprint_results['friction_patterns'] = [pattern.to_dict() for pattern in friction_patterns]
        
        # Generate molecules
        molecules = self.molecule_generator.generate_molecules(friction_patterns)
        sprint_results['generated_molecules'] = [molecule.to_dict() for molecule in molecules]
        
        # Analyze sprint results
        sprint_results['sprint_analysis'] = self._analyze_sprint_results(
            friction_patterns, molecules
        )
        
        # Generate operationalization plan
        sprint_results['operationalization_plan'] = self._generate_operationalization_plan(
            friction_patterns, molecules
        )
        
        # Generate recommendations
        sprint_results['recommendations'] = self._generate_sprint_recommendations(
            sprint_results['sprint_analysis']
        )
        
        return sprint_results
    
    def _analyze_sprint_results(self, friction_patterns: List[FrictionPattern], 
                              molecules: List[FrictionMolecule]) -> Dict[str, Any]:
        """Analyze mining sprint results."""
        analysis = {
            'total_patterns': len(friction_patterns),
            'total_molecules': len(molecules),
            'pattern_distribution': defaultdict(int),
            'molecule_distribution': defaultdict(int),
            'impact_analysis': {},
            'operationalization_metrics': {}
        }
        
        # Analyze pattern distribution
        for pattern in friction_patterns:
            analysis['pattern_distribution'][pattern.friction_type.value] += 1
        
        # Analyze molecule distribution
        for molecule in molecules:
            analysis['molecule_distribution'][molecule.molecule_type.value] += 1
        
        # Impact analysis
        total_impact = sum(pattern.impact_score for pattern in friction_patterns)
        avg_impact = total_impact / len(friction_patterns) if friction_patterns else 0
        
        analysis['impact_analysis'] = {
            'total_impact': total_impact,
            'average_impact': avg_impact,
            'high_impact_patterns': len([p for p in friction_patterns if p.impact_score > 0.7]),
            'medium_impact_patterns': len([p for p in friction_patterns if 0.4 <= p.impact_score <= 0.7]),
            'low_impact_patterns': len([p for p in friction_patterns if p.impact_score < 0.4])
        }
        
        # Operationalization metrics
        total_expected_impact = sum(molecule.expected_impact for molecule in molecules)
        avg_confidence = np.mean([molecule.confidence for molecule in molecules]) if molecules else 0
        
        analysis['operationalization_metrics'] = {
            'total_expected_impact': total_expected_impact,
            'average_confidence': avg_confidence,
            'high_confidence_molecules': len([m for m in molecules if m.confidence > 0.8]),
            'ready_for_implementation': len([m for m in molecules if m.confidence > 0.7 and m.expected_impact > 0.5])
        }
        
        return analysis
    
    def _generate_operationalization_plan(self, friction_patterns: List[FrictionPattern], 
                                        molecules: List[FrictionMolecule]) -> List[Dict[str, Any]]:
        """Generate operationalization plan for molecules."""
        plan = []
        
        # Sort molecules by readiness (confidence * expected_impact)
        ready_molecules = [m for m in molecules if m.confidence > 0.7 and m.expected_impact > 0.5]
        ready_molecules.sort(key=lambda x: x.confidence * x.expected_impact, reverse=True)
        
        for i, molecule in enumerate(ready_molecules[:10]):  # Top 10 ready molecules
            implementation_plan = {
                'priority': i + 1,
                'molecule_id': molecule.molecule_id,
                'molecule_name': molecule.name,
                'implementation_steps': [
                    f"Implement {molecule.implementation['method']}",
                    f"Set up validation tests: {', '.join(molecule.validation_tests[:3])}",
                    f"Configure dependencies: {', '.join(molecule.dependencies)}",
                    f"Deploy with confidence threshold: {molecule.confidence:.2f}",
                    f"Monitor expected impact: {molecule.expected_impact:.2f}"
                ],
                'estimated_effort': 'medium' if molecule.confidence > 0.8 else 'high',
                'expected_timeline': '1-2 weeks' if molecule.confidence > 0.8 else '2-4 weeks',
                'success_criteria': [
                    f"Confidence threshold met: {molecule.confidence:.2f}",
                    f"Impact improvement: {molecule.expected_impact:.2f}",
                    f"All validation tests pass",
                    f"No regression in existing functionality"
                ]
            }
            plan.append(implementation_plan)
        
        return plan
    
    def _generate_sprint_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations from sprint analysis."""
        recommendations = []
        
        # Pattern-based recommendations
        if analysis['impact_analysis']['high_impact_patterns'] > 0:
            recommendations.append(
                f"Focus on {analysis['impact_analysis']['high_impact_patterns']} high-impact patterns first"
            )
        
        if analysis['operationalization_metrics']['ready_for_implementation'] > 0:
            recommendations.append(
                f"Implement {analysis['operationalization_metrics']['ready_for_implementation']} ready molecules immediately"
            )
        
        # Distribution-based recommendations
        if analysis['pattern_distribution']:
            top_pattern_type = max(analysis['pattern_distribution'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Prioritize {top_pattern_type} friction patterns for systematic improvement")
        
        if analysis['molecule_distribution']:
            top_molecule_type = max(analysis['molecule_distribution'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Develop expertise in {top_molecule_type} molecule generation")
        
        # Quality-based recommendations
        if analysis['operationalization_metrics']['average_confidence'] < 0.7:
            recommendations.append("Improve molecule generation confidence through better pattern analysis")
        
        if analysis['operationalization_metrics']['total_expected_impact'] < 5.0:
            recommendations.append("Focus on higher-impact friction patterns for better ROI")
        
        # Default recommendations if no patterns detected
        if not analysis['pattern_distribution']:
            recommendations.extend([
                "Increase system monitoring to capture more friction patterns",
                "Lower detection thresholds to identify emerging issues",
                "Expand error logging to include more detailed context"
            ])
        
        return recommendations


def main():
    """Main function to run mining sprint operationalization."""
    logger.info("Starting mining sprint operationalization...")
    
    # Initialize mining sprint system
    sprint_system = MiningSprintSystem()
    
    # Generate sample system logs for demonstration
    sample_logs = [
        {'error_type': 'semantic_ambiguity', 'error_message': 'Multiple senses detected for "bank"', 'component': 'nsm_detector', 'timestamp': time.time()},
        {'error_type': 'semantic_ambiguity', 'error_message': 'Context-dependent meaning for "run"', 'component': 'nsm_detector', 'timestamp': time.time()},
        {'error_type': 'cross_language_mismatch', 'error_message': 'Translation quality below threshold', 'component': 'nsm_translator', 'timestamp': time.time()},
        {'error_type': 'cross_language_mismatch', 'error_message': 'Semantic drift detected in translation', 'component': 'nsm_translator', 'timestamp': time.time()},
        {'error_type': 'primitive_detection_failure', 'error_message': 'Low confidence in primitive detection', 'component': 'primitive_detector', 'timestamp': time.time()},
        {'error_type': 'primitive_detection_failure', 'error_message': 'Missing primitives in complex phrase', 'component': 'primitive_detector', 'timestamp': time.time()},
        {'error_type': 'translation_quality_issue', 'error_message': 'Semantic similarity below threshold', 'component': 'quality_evaluator', 'timestamp': time.time()},
        {'error_type': 'translation_quality_issue', 'error_message': 'Fluency issues in generated text', 'component': 'quality_evaluator', 'timestamp': time.time()},
        {'error_type': 'explication_complexity', 'error_message': 'High complexity in NSM explication', 'component': 'nsm_explicator', 'timestamp': time.time()},
        {'error_type': 'explication_complexity', 'error_message': 'Poor readability in generated explication', 'component': 'nsm_explicator', 'timestamp': time.time()}
    ]
    
    # Run mining sprint
    sprint_results = sprint_system.run_mining_sprint(sample_logs)
    
    # Print results
    print("\n" + "="*80)
    print("MINING SPRINT OPERATIONALIZATION RESULTS")
    print("="*80)
    
    print(f"Sprint Configuration:")
    print(f"  Min Pattern Frequency: {sprint_results['sprint_configuration']['min_pattern_frequency']}")
    print(f"  Min Impact Score: {sprint_results['sprint_configuration']['min_impact_score']}")
    
    print(f"\nFriction Pattern Analysis:")
    analysis = sprint_results['sprint_analysis']
    print(f"  Total Patterns: {analysis['total_patterns']}")
    print(f"  Total Molecules: {analysis['total_molecules']}")
    print(f"  High Impact Patterns: {analysis['impact_analysis']['high_impact_patterns']}")
    print(f"  Ready for Implementation: {analysis['operationalization_metrics']['ready_for_implementation']}")
    
    print(f"\nPattern Distribution:")
    for pattern_type, count in analysis['pattern_distribution'].items():
        print(f"  {pattern_type}: {count}")
    
    print(f"\nMolecule Distribution:")
    for molecule_type, count in analysis['molecule_distribution'].items():
        print(f"  {molecule_type}: {count}")
    
    print(f"\nOperationalization Plan:")
    for plan_item in sprint_results['operationalization_plan'][:3]:  # Show top 3
        print(f"  Priority {plan_item['priority']}: {plan_item['molecule_name']}")
        print(f"    Effort: {plan_item['estimated_effort']}, Timeline: {plan_item['expected_timeline']}")
    
    print(f"\nRecommendations:")
    for i, recommendation in enumerate(sprint_results['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    # Save results
    output_path = Path("data/mining_sprint_operationalization.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(convert_numpy_types(sprint_results), f, indent=2)
    
    logger.info(f"Mining sprint results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Mining sprint operationalization completed!")
    print("="*80)


if __name__ == "__main__":
    main()

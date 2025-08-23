#!/usr/bin/env python3
"""
Enhanced NSM Grammar Molecules Refinement System.

This script implements a comprehensive NSM grammar molecules refinement system
to tighten legality micro-grammar and improve explication generation quality.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
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
        # Convert tuple keys to strings
        converted_dict = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                # Convert tuple key to string representation
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


class NSMGrammarRule:
    """NSM grammar rule for legality validation."""
    
    def __init__(self, name: str, pattern: str, constraints: Dict[str, Any], 
                 primitive_types: List[str], confidence: float = 1.0):
        """Initialize an NSM grammar rule."""
        self.name = name
        self.pattern = pattern
        self.constraints = constraints
        self.primitive_types = primitive_types
        self.confidence = confidence
        
        # Compile pattern
        self.compiled_pattern = re.compile(pattern, re.IGNORECASE)
    
    def validate(self, explication: str, primitives: List[str]) -> Dict[str, Any]:
        """Validate an explication against this grammar rule."""
        try:
            # Check pattern match
            pattern_match = bool(self.compiled_pattern.search(explication))
            
            # Check primitive type constraints
            primitive_valid = self._check_primitive_constraints(primitives)
            
            # Check additional constraints
            constraint_valid = self._check_constraints(explication, primitives)
            
            # Calculate overall validity
            validity_score = 0.0
            if pattern_match:
                validity_score += 0.4
            if primitive_valid:
                validity_score += 0.3
            if constraint_valid:
                validity_score += 0.3
            
            return {
                'rule_name': self.name,
                'pattern_match': pattern_match,
                'primitive_valid': primitive_valid,
                'constraint_valid': constraint_valid,
                'validity_score': validity_score,
                'confidence': self.confidence,
                'overall_valid': validity_score >= 0.7
            }
        
        except Exception as e:
            logger.warning(f"Grammar rule validation failed for {self.name}: {e}")
            return {
                'rule_name': self.name,
                'pattern_match': False,
                'primitive_valid': False,
                'constraint_valid': False,
                'validity_score': 0.0,
                'confidence': 0.0,
                'overall_valid': False,
                'error': str(e)
            }
    
    def _check_primitive_constraints(self, primitives: List[str]) -> bool:
        """Check if primitives meet the rule's constraints."""
        if not self.primitive_types:
            return True
        
        # Check if at least one primitive matches the required types
        for primitive in primitives:
            if any(primitive_type.lower() in primitive.lower() for primitive_type in self.primitive_types):
                return True
        
        return False
    
    def _check_constraints(self, explication: str, primitives: List[str]) -> bool:
        """Check additional constraints."""
        try:
            # Length constraint
            if 'min_length' in self.constraints:
                if len(explication.split()) < self.constraints['min_length']:
                    return False
            
            if 'max_length' in self.constraints:
                if len(explication.split()) > self.constraints['max_length']:
                    return False
            
            # Primitive count constraint
            if 'min_primitives' in self.constraints:
                if len(primitives) < self.constraints['min_primitives']:
                    return False
            
            if 'max_primitives' in self.constraints:
                if len(primitives) > self.constraints['max_primitives']:
                    return False
            
            # Structure constraint
            if 'required_structure' in self.constraints:
                required = self.constraints['required_structure']
                if required == 'simple' and 'AND' in explication:
                    return False
                elif required == 'complex' and 'AND' not in explication:
                    return False
            
            return True
        
        except Exception as e:
            logger.warning(f"Constraint checking failed: {e}")
            return False


class NSMGrammarRuleRegistry:
    """Registry for NSM grammar rules."""
    
    def __init__(self):
        """Initialize the NSM grammar rule registry."""
        self.rules: Dict[str, NSMGrammarRule] = {}
        self.rule_categories: Dict[str, List[str]] = defaultdict(list)
        
        # Load base grammar rules
        self._load_base_rules()
    
    def _load_base_rules(self):
        """Load base NSM grammar rules."""
        base_rules = [
            # Simple explication rules
            {
                'name': 'simple_explication',
                'pattern': r'^[A-Za-z]+\([^)]+\)$',
                'constraints': {
                    'min_length': 2,
                    'max_length': 10,
                    'min_primitives': 1,
                    'max_primitives': 1,
                    'required_structure': 'simple'
                },
                'primitive_types': ['HASPROPERTY', 'ATLOCATION', 'SIMILARTO'],
                'confidence': 0.9
            },
            
            # Complex explication rules
            {
                'name': 'complex_explication',
                'pattern': r'.*AND.*',
                'constraints': {
                    'min_length': 5,
                    'max_length': 20,
                    'min_primitives': 2,
                    'max_primitives': 5,
                    'required_structure': 'complex'
                },
                'primitive_types': ['HASPROPERTY', 'ATLOCATION', 'SIMILARTO', 'USEDFOR', 'CONTAINS'],
                'confidence': 0.8
            },
            
            # Property explication rules
            {
                'name': 'property_explication',
                'pattern': r'.*HASPROPERTY.*',
                'constraints': {
                    'min_length': 3,
                    'max_length': 15,
                    'min_primitives': 1,
                    'max_primitives': 3
                },
                'primitive_types': ['HASPROPERTY'],
                'confidence': 0.9
            },
            
            # Location explication rules
            {
                'name': 'location_explication',
                'pattern': r'.*ATLOCATION.*',
                'constraints': {
                    'min_length': 3,
                    'max_length': 15,
                    'min_primitives': 1,
                    'max_primitives': 3
                },
                'primitive_types': ['ATLOCATION'],
                'confidence': 0.9
            },
            
            # Comparison explication rules
            {
                'name': 'comparison_explication',
                'pattern': r'.*SIMILARTO.*',
                'constraints': {
                    'min_length': 3,
                    'max_length': 15,
                    'min_primitives': 1,
                    'max_primitives': 3
                },
                'primitive_types': ['SIMILARTO'],
                'confidence': 0.8
            },
            
            # Action explication rules
            {
                'name': 'action_explication',
                'pattern': r'.*USEDFOR.*',
                'constraints': {
                    'min_length': 3,
                    'max_length': 15,
                    'min_primitives': 1,
                    'max_primitives': 3
                },
                'primitive_types': ['USEDFOR'],
                'confidence': 0.8
            },
            
            # Containment explication rules
            {
                'name': 'containment_explication',
                'pattern': r'.*CONTAINS.*',
                'constraints': {
                    'min_length': 3,
                    'max_length': 15,
                    'min_primitives': 1,
                    'max_primitives': 3
                },
                'primitive_types': ['CONTAINS'],
                'confidence': 0.8
            },
            
            # Causality explication rules
            {
                'name': 'causality_explication',
                'pattern': r'.*CAUSES.*',
                'constraints': {
                    'min_length': 4,
                    'max_length': 20,
                    'min_primitives': 2,
                    'max_primitives': 4
                },
                'primitive_types': ['CAUSES'],
                'confidence': 0.7
            },
            
            # Temporal explication rules
            {
                'name': 'temporal_explication',
                'pattern': r'.*PAST.*|.*FUTURE.*|.*PRESENT.*',
                'constraints': {
                    'min_length': 3,
                    'max_length': 15,
                    'min_primitives': 1,
                    'max_primitives': 3
                },
                'primitive_types': ['PAST', 'FUTURE', 'PRESENT'],
                'confidence': 0.8
            },
            
            # Negation explication rules
            {
                'name': 'negation_explication',
                'pattern': r'.*NOT.*',
                'constraints': {
                    'min_length': 3,
                    'max_length': 15,
                    'min_primitives': 2,
                    'max_primitives': 4
                },
                'primitive_types': ['NOT'],
                'confidence': 0.8
            }
        ]
        
        for rule_data in base_rules:
            rule = NSMGrammarRule(**rule_data)
            self.add_rule(rule)
    
    def add_rule(self, rule: NSMGrammarRule):
        """Add a rule to the registry."""
        self.rules[rule.name] = rule
        
        # Categorize by primitive types
        for primitive_type in rule.primitive_types:
            self.rule_categories[primitive_type].append(rule.name)
    
    def get_rules_by_primitive_type(self, primitive_type: str) -> List[NSMGrammarRule]:
        """Get rules by primitive type."""
        rule_names = self.rule_categories.get(primitive_type, [])
        return [self.rules[name] for name in rule_names if name in self.rules]
    
    def get_all_rules(self) -> List[NSMGrammarRule]:
        """Get all rules."""
        return list(self.rules.values())


class NSMGrammarValidator:
    """NSM grammar validator for explication legality."""
    
    def __init__(self):
        """Initialize the NSM grammar validator."""
        self.rule_registry = NSMGrammarRuleRegistry()
        self.sbert_model = None
        
        # Validation parameters
        self.validation_params = {
            'min_validity_score': 0.7,
            'min_rule_confidence': 0.6,
            'enable_semantic_validation': True,
            'semantic_threshold': 0.6
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model for semantic validation."""
        try:
            logger.info("Loading SBERT model for NSM grammar validation...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def validate_explication(self, explication: str, primitives: List[str], 
                           original_text: str = "") -> Dict[str, Any]:
        """Validate an NSM explication for legality."""
        try:
            # Apply all grammar rules
            rule_validations = []
            for rule in self.rule_registry.get_all_rules():
                validation = rule.validate(explication, primitives)
                rule_validations.append(validation)
            
            # Calculate overall validity
            valid_rules = [v for v in rule_validations if v['overall_valid']]
            overall_validity = len(valid_rules) / len(rule_validations) if rule_validations else 0.0
            
            # Calculate average validity score
            avg_validity_score = np.mean([v['validity_score'] for v in rule_validations]) if rule_validations else 0.0
            
            # Semantic validation
            semantic_score = 0.5
            if self.validation_params['enable_semantic_validation'] and self.sbert_model and original_text:
                semantic_score = self._calculate_semantic_coherence(explication, original_text)
            
            # Overall legality score
            legality_score = (avg_validity_score * 0.6 + semantic_score * 0.4)
            
            return {
                'explication': explication,
                'primitives': primitives,
                'rule_validations': rule_validations,
                'valid_rules': len(valid_rules),
                'total_rules': len(rule_validations),
                'overall_validity': overall_validity,
                'avg_validity_score': avg_validity_score,
                'semantic_score': semantic_score,
                'legality_score': legality_score,
                'is_legal': legality_score >= self.validation_params['min_validity_score']
            }
        
        except Exception as e:
            logger.warning(f"NSM grammar validation failed: {e}")
            return {
                'explication': explication,
                'primitives': primitives,
                'rule_validations': [],
                'valid_rules': 0,
                'total_rules': 0,
                'overall_validity': 0.0,
                'avg_validity_score': 0.0,
                'semantic_score': 0.0,
                'legality_score': 0.0,
                'is_legal': False,
                'error': str(e)
            }
    
    def _calculate_semantic_coherence(self, explication: str, original_text: str) -> float:
        """Calculate semantic coherence between explication and original text."""
        try:
            if not self.sbert_model:
                return 0.5
            
            # Calculate semantic similarity
            embeddings = self.sbert_model.encode([explication, original_text])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return max(0.0, float(similarity))
        
        except Exception as e:
            logger.warning(f"Semantic coherence calculation failed: {e}")
            return 0.5


class NSMGrammarRefiner:
    """NSM grammar refiner for improving explication quality."""
    
    def __init__(self):
        """Initialize the NSM grammar refiner."""
        self.validator = NSMGrammarValidator()
        self.nsm_translator = NSMTranslator()
        
        # Refinement parameters
        self.refinement_params = {
            'max_refinement_attempts': 5,
            'min_improvement_threshold': 0.1,
            'enable_structure_refinement': True,
            'enable_primitive_refinement': True
        }
    
    def refine_explication(self, explication: str, primitives: List[str], 
                          original_text: str) -> Dict[str, Any]:
        """Refine an NSM explication to improve legality."""
        try:
            # Initial validation
            initial_validation = self.validator.validate_explication(explication, primitives, original_text)
            
            if initial_validation['is_legal']:
                return {
                    'original_explication': explication,
                    'refined_explication': explication,
                    'improvement': 0.0,
                    'refinement_applied': False,
                    'validation': initial_validation
                }
            
            # Attempt refinements
            best_explication = explication
            best_score = initial_validation['legality_score']
            refinements_applied = []
            
            for attempt in range(self.refinement_params['max_refinement_attempts']):
                # Structure refinement
                if self.refinement_params['enable_structure_refinement']:
                    structure_refined = self._refine_structure(best_explication, primitives)
                    if structure_refined != best_explication:
                        structure_validation = self.validator.validate_explication(
                            structure_refined, primitives, original_text
                        )
                        if structure_validation['legality_score'] > best_score + self.refinement_params['min_improvement_threshold']:
                            best_explication = structure_refined
                            best_score = structure_validation['legality_score']
                            refinements_applied.append(f"structure_refinement_{attempt}")
                
                # Primitive refinement
                if self.refinement_params['enable_primitive_refinement']:
                    primitive_refined = self._refine_primitives(best_explication, primitives)
                    if primitive_refined != best_explication:
                        primitive_validation = self.validator.validate_explication(
                            primitive_refined, primitives, original_text
                        )
                        if primitive_validation['legality_score'] > best_score + self.refinement_params['min_improvement_threshold']:
                            best_explication = primitive_refined
                            best_score = primitive_validation['legality_score']
                            refinements_applied.append(f"primitive_refinement_{attempt}")
            
            # Final validation
            final_validation = self.validator.validate_explication(best_explication, primitives, original_text)
            
            return {
                'original_explication': explication,
                'refined_explication': best_explication,
                'improvement': final_validation['legality_score'] - initial_validation['legality_score'],
                'refinement_applied': len(refinements_applied) > 0,
                'refinements_applied': refinements_applied,
                'initial_validation': initial_validation,
                'final_validation': final_validation
            }
        
        except Exception as e:
            logger.warning(f"NSM grammar refinement failed: {e}")
            return {
                'original_explication': explication,
                'refined_explication': explication,
                'improvement': 0.0,
                'refinement_applied': False,
                'error': str(e)
            }
    
    def _refine_structure(self, explication: str, primitives: List[str]) -> str:
        """Refine the structure of an explication."""
        try:
            # Remove excessive AND patterns
            if explication.count('AND') > 2:
                parts = explication.split('AND')
                # Keep only the first two parts
                refined = ' AND '.join(parts[:2])
                return refined
            
            # Add missing parentheses
            if not explication.endswith(')'):
                if '(' in explication:
                    refined = explication + ')'
                    return refined
            
            # Simplify complex patterns
            if len(explication.split()) > 15:
                words = explication.split()
                refined = ' '.join(words[:10]) + '...'
                return refined
            
            return explication
        
        except Exception as e:
            logger.warning(f"Structure refinement failed: {e}")
            return explication
    
    def _refine_primitives(self, explication: str, primitives: List[str]) -> str:
        """Refine the primitives in an explication."""
        try:
            # Ensure all primitives are properly capitalized
            refined = explication
            for primitive in primitives:
                if primitive.lower() in refined.lower():
                    # Replace with properly capitalized version
                    refined = re.sub(primitive.lower(), primitive, refined, flags=re.IGNORECASE)
            
            # Remove duplicate primitives
            primitive_pattern = r'\b([A-Z]+)\b'
            found_primitives = re.findall(primitive_pattern, refined)
            unique_primitives = list(dict.fromkeys(found_primitives))  # Preserve order
            
            if len(unique_primitives) < len(found_primitives):
                # Reconstruct with unique primitives
                parts = refined.split('AND')
                if len(parts) > 1:
                    refined = ' AND '.join(unique_primitives[:2]) + '(' + parts[-1].split('(')[-1]
            
            return refined
        
        except Exception as e:
            logger.warning(f"Primitive refinement failed: {e}")
            return explication


class EnhancedNSMGrammarRefinementSystem:
    """Enhanced NSM grammar refinement system with comprehensive analysis."""
    
    def __init__(self):
        """Initialize the enhanced NSM grammar refinement system."""
        self.validator = NSMGrammarValidator()
        self.refiner = NSMGrammarRefiner()
        
        # System parameters
        self.system_params = {
            'enable_validation': True,
            'enable_refinement': True,
            'min_legality_threshold': 0.7,
            'max_refinement_iterations': 3
        }
    
    def run_grammar_refinement_analysis(self, test_texts: List[str], 
                                      languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Run comprehensive NSM grammar refinement analysis."""
        logger.info(f"Running NSM grammar refinement analysis for {len(test_texts)} texts")
        
        analysis_results = {
            'test_configuration': {
                'num_test_texts': len(test_texts),
                'languages': languages,
                'timestamp': time.time()
            },
            'validation_results': [],
            'refinement_results': [],
            'grammar_analysis': {},
            'recommendations': []
        }
        
        # Run validation and refinement on test texts
        for language in languages:
            for text in test_texts:
                # Generate explication
                primitives = self.refiner.nsm_translator.detect_primitives_in_text(text, language)
                
                if primitives:
                    # Simple explication
                    explication = f"{' '.join(primitives)}({text})"
                    
                    # Validate explication
                    validation_result = self.validator.validate_explication(explication, primitives, text)
                    analysis_results['validation_results'].append(validation_result)
                    
                    # Refine explication if needed
                    refinement_result = self.refiner.refine_explication(explication, primitives, text)
                    analysis_results['refinement_results'].append(refinement_result)
        
        # Analyze results
        analysis_results['grammar_analysis'] = self._analyze_grammar_results(
            analysis_results['validation_results'],
            analysis_results['refinement_results']
        )
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_refinement_recommendations(
            analysis_results['grammar_analysis']
        )
        
        return analysis_results
    
    def _analyze_grammar_results(self, validation_results: List[Dict[str, Any]], 
                               refinement_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze grammar validation and refinement results."""
        analysis = {
            'total_explications': len(validation_results),
            'legal_explications': 0,
            'refined_explications': 0,
            'avg_legality_score': 0.0,
            'avg_improvement': 0.0,
            'rule_performance': defaultdict(lambda: {'matches': 0, 'valid': 0}),
            'refinement_effectiveness': {
                'successful_refinements': 0,
                'avg_improvement': 0.0,
                'refinement_types': defaultdict(int)
            }
        }
        
        legality_scores = []
        improvements = []
        
        for validation_result in validation_results:
            if validation_result.get('is_legal', False):
                analysis['legal_explications'] += 1
            
            legality_score = validation_result.get('legality_score', 0.0)
            legality_scores.append(legality_score)
            
            # Analyze rule performance
            for rule_validation in validation_result.get('rule_validations', []):
                rule_name = rule_validation.get('rule_name', 'unknown')
                analysis['rule_performance'][rule_name]['matches'] += 1
                if rule_validation.get('overall_valid', False):
                    analysis['rule_performance'][rule_name]['valid'] += 1
        
        for refinement_result in refinement_results:
            if refinement_result.get('refinement_applied', False):
                analysis['refined_explications'] += 1
                analysis['refinement_effectiveness']['successful_refinements'] += 1
            
            improvement = refinement_result.get('improvement', 0.0)
            improvements.append(improvement)
            
            # Analyze refinement types
            for refinement_type in refinement_result.get('refinements_applied', []):
                analysis['refinement_effectiveness']['refinement_types'][refinement_type] += 1
        
        # Calculate averages
        if legality_scores:
            analysis['avg_legality_score'] = np.mean(legality_scores)
        
        if improvements:
            analysis['avg_improvement'] = np.mean(improvements)
            analysis['refinement_effectiveness']['avg_improvement'] = np.mean(improvements)
        
        # Calculate rule success rates
        for rule_name, stats in analysis['rule_performance'].items():
            if stats['matches'] > 0:
                stats['success_rate'] = stats['valid'] / stats['matches']
            else:
                stats['success_rate'] = 0.0
        
        return analysis
    
    def _generate_refinement_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for grammar refinement."""
        recommendations = []
        
        # Legality recommendations
        legality_rate = analysis['legal_explications'] / analysis['total_explications']
        if legality_rate < 0.8:
            recommendations.append(f"Low legality rate ({legality_rate:.1%}) - improve grammar rules")
        
        # Rule performance recommendations
        for rule_name, stats in analysis['rule_performance'].items():
            if stats['success_rate'] < 0.5:
                recommendations.append(f"Rule '{rule_name}' has low success rate ({stats['success_rate']:.1%}) - consider refinement")
        
        # Refinement effectiveness recommendations
        refinement_rate = analysis['refined_explications'] / analysis['total_explications']
        if refinement_rate > 0.5:
            recommendations.append("High refinement rate - consider improving initial explication generation")
        
        if analysis['avg_improvement'] < 0.1:
            recommendations.append("Low average improvement - consider enhancing refinement strategies")
        
        # Grammar score recommendations
        if analysis['avg_legality_score'] < 0.6:
            recommendations.append("Low average legality score - improve grammar validation")
        
        return recommendations


def main():
    """Main function to run enhanced NSM grammar refinement analysis."""
    logger.info("Starting enhanced NSM grammar refinement analysis...")
    
    # Initialize NSM grammar refinement system
    grammar_system = EnhancedNSMGrammarRefinementSystem()
    
    # Test texts
    test_texts = [
        "The red car is parked near the building",
        "The cat is on the mat",
        "This is similar to that",
        "The book contains important information",
        "The weather is cold today",
        "She works at the hospital",
        "The movie was very long",
        "I need to buy groceries",
        "Children play in the park",
        "The restaurant serves Italian food"
    ]
    
    # Run grammar refinement analysis
    analysis_results = grammar_system.run_grammar_refinement_analysis(test_texts, ["en"])
    
    # Print results
    print("\n" + "="*80)
    print("ENHANCED NSM GRAMMAR REFINEMENT ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Texts: {analysis_results['test_configuration']['num_test_texts']}")
    print(f"  Languages: {analysis_results['test_configuration']['languages']}")
    
    print(f"\nGrammar Analysis:")
    analysis = analysis_results['grammar_analysis']
    print(f"  Total Explications: {analysis['total_explications']}")
    print(f"  Legal Explications: {analysis['legal_explications']}")
    print(f"  Legality Rate: {analysis['legal_explications']/analysis['total_explications']:.1%}")
    print(f"  Average Legality Score: {analysis['avg_legality_score']:.3f}")
    print(f"  Refined Explications: {analysis['refined_explications']}")
    print(f"  Average Improvement: {analysis['avg_improvement']:.3f}")
    
    print(f"\nRule Performance:")
    for rule_name, stats in sorted(analysis['rule_performance'].items(), key=lambda x: x[1]['success_rate'], reverse=True):
        print(f"  {rule_name}:")
        print(f"    Matches: {stats['matches']}")
        print(f"    Valid: {stats['valid']}")
        print(f"    Success Rate: {stats['success_rate']:.1%}")
    
    print(f"\nRefinement Effectiveness:")
    refinement_eff = analysis['refinement_effectiveness']
    print(f"  Successful Refinements: {refinement_eff['successful_refinements']}")
    print(f"  Average Improvement: {refinement_eff['avg_improvement']:.3f}")
    print(f"  Refinement Types:")
    for refinement_type, count in refinement_eff['refinement_types'].items():
        print(f"    {refinement_type}: {count}")
    
    print(f"\nExample Validations:")
    for i, result in enumerate(analysis_results['validation_results'][:3]):
        explication = result['explication']
        legality_score = result.get('legality_score', 0.0)
        is_legal = result.get('is_legal', False)
        
        print(f"  {i+1}. Explication: {explication}")
        print(f"     Legality Score: {legality_score:.3f}")
        print(f"     Is Legal: {is_legal}")
    
    print(f"\nExample Refinements:")
    for i, result in enumerate(analysis_results['refinement_results'][:3]):
        original = result['original_explication']
        refined = result['refined_explication']
        improvement = result.get('improvement', 0.0)
        
        print(f"  {i+1}. Original: {original}")
        print(f"     Refined: {refined}")
        print(f"     Improvement: {improvement:.3f}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_path = "data/nsm_grammar_molecules_refinement_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(analysis_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced NSM grammar refinement report saved to: {output_path}")
    
    print("="*80)
    print("Enhanced NSM grammar refinement analysis completed!")
    print("="*80)


if __name__ == "__main__":
    main()

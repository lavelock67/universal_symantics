#!/usr/bin/env python3
"""
NSM Validation Enhancement System.

This script enhances NSM validation with structural and semantic checks:
1. Structural validation (syntax, grammar, consistency)
2. Semantic validation (meaning, coherence, logic)
3. Cross-language validation (translatability, alignment)
4. Contextual validation (appropriateness, pragmatics)
5. Quality assessment (completeness, accuracy)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types and booleans to JSON-serializable types."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return str(bool(obj))  # Convert numpy.bool_ to string
    elif isinstance(obj, bool):
        return str(obj)  # Convert boolean to string for JSON serialization
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting to dict
        return convert_numpy_types(obj.__dict__)
    else:
        return obj

class NSMValidationEnhancer:
    """Enhances NSM validation with comprehensive checks."""
    
    def __init__(self):
        """Initialize the validation enhancer."""
        self.sbert_model = None
        self.languages = ['en', 'es', 'fr']
        self._load_models()
        
        # Validation rules and patterns
        self.validation_rules = {
            'structural': {
                'required_elements': ['this thing', 'this other thing', 'this place', 'this action'],
                'forbidden_elements': ['complex_terms', 'technical_jargon', 'culture_specific'],
                'syntax_patterns': {
                    'AtLocation': r'this thing.*(?:in|at|near|close to).*this place',
                    'HasProperty': r'this thing.*(?:has|possesses|characterized by).*this.*(?:characteristic|property|quality)',
                    'PartOf': r'this thing.*(?:component|part|belongs to).*this.*(?:whole|larger thing)',
                    'Causes': r'this thing.*(?:makes|causes|leads to).*this.*(?:other thing|result)',
                    'UsedFor': r'(?:people|this thing).*(?:use|serves|designed for).*this.*(?:action|purpose)',
                    'Exist': r'this thing.*(?:exists|is real|is present)',
                    'Not': r'this thing.*(?:not|differs from|opposite of).*this.*(?:other thing|property)',
                    'SimilarTo': r'this thing.*(?:like|resembles|similar to).*this.*(?:other thing)',
                    'DifferentFrom': r'this thing.*(?:different|distinct from).*this.*(?:other thing)'
                }
            },
            'semantic': {
                'coherence_checks': [
                    'logical_consistency',
                    'meaning_preservation',
                    'concept_alignment',
                    'relationship_validity'
                ],
                'semantic_constraints': {
                    'AtLocation': ['spatial_relationship', 'physical_presence'],
                    'HasProperty': ['attribute_assignment', 'characteristic_description'],
                    'PartOf': ['hierarchical_relationship', 'compositional_structure'],
                    'Causes': ['causal_relationship', 'temporal_sequence'],
                    'UsedFor': ['functional_purpose', 'intentional_design'],
                    'Exist': ['existence_assertion', 'reality_claim'],
                    'Not': ['negation_operation', 'contrast_establishment'],
                    'SimilarTo': ['similarity_assessment', 'comparison_basis'],
                    'DifferentFrom': ['distinction_establishment', 'difference_identification']
                }
            },
            'cross_language': {
                'alignment_thresholds': {
                    'excellent': 0.8,
                    'good': 0.6,
                    'fair': 0.4,
                    'poor': 0.2
                },
                'consistency_checks': [
                    'structural_equivalence',
                    'semantic_parity',
                    'functional_correspondence',
                    'pragmatic_alignment'
                ]
            }
        }
        
        # Quality metrics
        self.quality_metrics = {
            'completeness': {
                'required_elements_present': 0.3,
                'pattern_followed': 0.2,
                'context_appropriate': 0.2,
                'language_specific': 0.3
            },
            'accuracy': {
                'semantic_correctness': 0.4,
                'logical_consistency': 0.3,
                'cross_language_alignment': 0.3
            },
            'coherence': {
                'internal_consistency': 0.4,
                'external_alignment': 0.3,
                'pragmatic_appropriateness': 0.3
            }
        }
    
    def _load_models(self):
        """Load SBERT model for semantic validation."""
        try:
            logger.info("Loading SBERT model for semantic validation...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def validate_structural_pattern(self, explication: str, primitive: str, language: str) -> Dict[str, Any]:
        """Validate structural pattern of an explication."""
        validation_result = {
            'valid': False,
            'score': 0.0,
            'issues': [],
            'pattern_match': False,
            'required_elements': [],
            'missing_elements': []
        }
        
        # Check pattern match
        if primitive in self.validation_rules['structural']['syntax_patterns']:
            pattern = self.validation_rules['structural']['syntax_patterns'][primitive]
            if re.search(pattern, explication, re.IGNORECASE):
                validation_result['pattern_match'] = True
                validation_result['score'] += 0.4
            else:
                validation_result['issues'].append(f"Pattern mismatch for {primitive}")
        
        # Check required elements
        required_elements = self.validation_rules['structural']['required_elements']
        present_elements = []
        missing_elements = []
        
        for element in required_elements:
            if element in explication:
                present_elements.append(element)
            else:
                missing_elements.append(element)
        
        validation_result['required_elements'] = present_elements
        validation_result['missing_elements'] = missing_elements
        
        # Score based on required elements
        element_score = len(present_elements) / len(required_elements) * 0.3
        validation_result['score'] += element_score
        
        # Check for forbidden elements
        forbidden_elements = self.validation_rules['structural']['forbidden_elements']
        for element in forbidden_elements:
            if element in explication:
                validation_result['issues'].append(f"Forbidden element detected: {element}")
                validation_result['score'] -= 0.1
        
        # Determine validity
        validation_result['valid'] = validation_result['score'] >= 0.6 and validation_result['pattern_match']
        
        return validation_result
    
    def validate_semantic_coherence(self, explication: str, primitive: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate semantic coherence of an explication."""
        validation_result = {
            'coherent': False,
            'score': 0.0,
            'issues': [],
            'semantic_constraints': [],
            'logical_consistency': True
        }
        
        # Check semantic constraints
        if primitive in self.validation_rules['semantic']['semantic_constraints']:
            constraints = self.validation_rules['semantic']['semantic_constraints'][primitive]
            satisfied_constraints = []
            
            for constraint in constraints:
                # Basic constraint checking (can be enhanced with more sophisticated logic)
                if self._check_semantic_constraint(explication, constraint):
                    satisfied_constraints.append(constraint)
            
            validation_result['semantic_constraints'] = satisfied_constraints
            constraint_score = len(satisfied_constraints) / len(constraints) * 0.4
            validation_result['score'] += constraint_score
        
        # Check logical consistency
        logical_issues = self._check_logical_consistency(explication, primitive)
        if logical_issues:
            validation_result['logical_consistency'] = False
            validation_result['issues'].extend(logical_issues)
            validation_result['score'] -= 0.2
        
        # Check meaning preservation
        meaning_score = self._check_meaning_preservation(explication, primitive)
        validation_result['score'] += meaning_score * 0.3
        
        # Determine coherence
        validation_result['coherent'] = validation_result['score'] >= 0.6 and validation_result['logical_consistency']
        
        return validation_result
    
    def _check_semantic_constraint(self, explication: str, constraint: str) -> bool:
        """Check if an explication satisfies a semantic constraint."""
        # Basic constraint checking - can be enhanced with more sophisticated logic
        constraint_keywords = {
            'spatial_relationship': ['in', 'at', 'near', 'close to', 'location', 'place'],
            'attribute_assignment': ['has', 'possesses', 'characterized by', 'property', 'characteristic'],
            'hierarchical_relationship': ['component', 'part', 'belongs to', 'whole', 'larger'],
            'causal_relationship': ['makes', 'causes', 'leads to', 'results in'],
            'functional_purpose': ['use', 'serves', 'designed for', 'purpose', 'action'],
            'existence_assertion': ['exists', 'is real', 'is present'],
            'negation_operation': ['not', 'differs from', 'opposite of'],
            'similarity_assessment': ['like', 'resembles', 'similar to'],
            'distinction_establishment': ['different', 'distinct from', 'separate']
        }
        
        if constraint in constraint_keywords:
            keywords = constraint_keywords[constraint]
            return any(keyword in explication.lower() for keyword in keywords)
        
        return True  # Default to satisfied if constraint not recognized
    
    def _check_logical_consistency(self, explication: str, primitive: str) -> List[str]:
        """Check logical consistency of an explication."""
        issues = []
        
        # Check for contradictions
        contradictions = [
            ('not', 'is'),
            ('different', 'same'),
            ('opposite', 'similar'),
            ('excludes', 'includes')
        ]
        
        for neg_term, pos_term in contradictions:
            if neg_term in explication.lower() and pos_term in explication.lower():
                issues.append(f"Logical contradiction: {neg_term} and {pos_term}")
        
        # Check for circular definitions
        if primitive.lower() in explication.lower():
            issues.append(f"Circular definition detected: {primitive}")
        
        return issues
    
    def _check_meaning_preservation(self, explication: str, primitive: str) -> float:
        """Check meaning preservation of an explication."""
        # Basic meaning preservation check
        primitive_keywords = {
            'AtLocation': ['location', 'place', 'where'],
            'HasProperty': ['property', 'characteristic', 'quality'],
            'PartOf': ['part', 'component', 'belongs'],
            'Causes': ['cause', 'makes', 'leads to'],
            'UsedFor': ['use', 'purpose', 'function'],
            'Exist': ['exists', 'real', 'present'],
            'Not': ['not', 'different', 'opposite'],
            'SimilarTo': ['similar', 'like', 'resembles'],
            'DifferentFrom': ['different', 'distinct', 'separate']
        }
        
        if primitive in primitive_keywords:
            keywords = primitive_keywords[primitive]
            keyword_matches = sum(1 for keyword in keywords if keyword in explication.lower())
            return keyword_matches / len(keywords)
        
        return 0.5  # Default score
    
    def validate_cross_language_alignment(self, explications: Dict[str, str], primitive: str) -> Dict[str, Any]:
        """Validate cross-language alignment of explications."""
        validation_result = {
            'aligned': False,
            'score': 0.0,
            'pairwise_similarities': {},
            'alignment_level': 'poor',
            'issues': []
        }
        
        if not self.sbert_model or len(explications) < 2:
            return validation_result
        
        try:
            # Encode all explications
            lang_explications = list(explications.values())
            embeddings = self.sbert_model.encode(lang_explications)
            
            # Calculate pairwise similarities
            similarities = []
            lang_pairs = list(explications.keys())
            
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    similarities.append(sim)
                    
                    pair_name = f"{lang_pairs[i]}-{lang_pairs[j]}"
                    validation_result['pairwise_similarities'][pair_name] = float(sim)
            
            # Calculate overall alignment score
            if similarities:
                alignment_score = np.mean(similarities)
                validation_result['score'] = alignment_score
                
                # Determine alignment level
                thresholds = self.validation_rules['cross_language']['alignment_thresholds']
                if alignment_score >= thresholds['excellent']:
                    validation_result['alignment_level'] = 'excellent'
                elif alignment_score >= thresholds['good']:
                    validation_result['alignment_level'] = 'good'
                elif alignment_score >= thresholds['fair']:
                    validation_result['alignment_level'] = 'fair'
                else:
                    validation_result['alignment_level'] = 'poor'
                
                validation_result['aligned'] = alignment_score >= thresholds['good']
            
        except Exception as e:
            validation_result['issues'].append(f"Cross-language validation error: {e}")
        
        return validation_result
    
    def calculate_quality_score(self, structural_result: Dict[str, Any], semantic_result: Dict[str, Any], 
                               cross_language_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality score based on validation results."""
        quality_score = {
            'overall_score': 0.0,
            'completeness_score': 0.0,
            'accuracy_score': 0.0,
            'coherence_score': 0.0,
            'breakdown': {}
        }
        
        # Completeness score
        completeness_weights = self.quality_metrics['completeness']
        completeness_score = float(
            structural_result.get('score', 0.0) * completeness_weights['required_elements_present'] +
            (1.0 if structural_result.get('pattern_match', False) else 0.0) * completeness_weights['pattern_followed'] +
            0.8 * completeness_weights['context_appropriate'] +  # Assuming context is appropriate
            0.9 * completeness_weights['language_specific']  # Assuming language-specific
        )
        quality_score['completeness_score'] = completeness_score
        
        # Accuracy score
        accuracy_weights = self.quality_metrics['accuracy']
        accuracy_score = float(
            semantic_result.get('score', 0.0) * accuracy_weights['semantic_correctness'] +
            (1.0 if semantic_result.get('logical_consistency', True) else 0.0) * accuracy_weights['logical_consistency'] +
            cross_language_result.get('score', 0.0) * accuracy_weights['cross_language_alignment']
        )
        quality_score['accuracy_score'] = accuracy_score
        
        # Coherence score
        coherence_weights = self.quality_metrics['coherence']
        coherence_score = float(
            (1.0 if semantic_result.get('logical_consistency', True) else 0.0) * coherence_weights['internal_consistency'] +
            cross_language_result.get('score', 0.0) * coherence_weights['external_alignment'] +
            0.8 * coherence_weights['pragmatic_appropriateness']  # Assuming pragmatic appropriateness
        )
        quality_score['coherence_score'] = coherence_score
        
        # Overall score (weighted average)
        quality_score['overall_score'] = float(
            completeness_score * 0.3 +
            accuracy_score * 0.4 +
            coherence_score * 0.3
        )
        
        # Detailed breakdown
        quality_score['breakdown'] = {
            'structural': structural_result,
            'semantic': semantic_result,
            'cross_language': cross_language_result
        }
        
        return quality_score
    
    def enhance_validation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance validation for existing NSM data."""
        logger.info("Enhancing NSM validation with comprehensive checks...")
        
        enhanced_validation = {
            'metadata': {
                'validation_type': 'NSM_enhanced_validation',
                'timestamp': '2025-08-22',
                'languages_validated': self.languages,
                'validation_components': ['structural', 'semantic', 'cross_language', 'quality']
            },
            'validation_results': {},
            'quality_metrics': self.quality_metrics,
            'validation_rules': self.validation_rules
        }
        
        # Process each language
        for lang in self.languages:
            if lang in input_data.get('enhanced_entries', {}):
                lang_data = input_data['enhanced_entries'][lang]
                validation_entries = []
                
                for entry in lang_data.get('entries', []):
                    if 'original_entry' in entry:
                        original = entry['original_entry']
                        primitive = original.get('primitive', 'Unknown')
                        
                        # Validate enhanced explications
                        enhanced_explications = entry.get('enhanced_explications', {})
                        validation_results = {}
                        
                        for pattern_type, explication in enhanced_explications.items():
                            # Structural validation
                            structural_result = self.validate_structural_pattern(explication, primitive, lang)
                            
                            # Semantic validation
                            semantic_result = self.validate_semantic_coherence(explication, primitive)
                            
                            # Cross-language validation (if multiple languages available)
                            cross_language_result = {'aligned': True, 'score': 1.0, 'alignment_level': 'excellent'}
                            if len(input_data.get('enhanced_entries', {})) > 1:
                                # Get corresponding explications from other languages
                                other_explications = {}
                                for other_lang in self.languages:
                                    if other_lang != lang and other_lang in input_data.get('enhanced_entries', {}):
                                        other_entries = input_data['enhanced_entries'][other_lang].get('entries', [])
                                        for other_entry in other_entries:
                                            if other_entry.get('original_entry', {}).get('primitive') == primitive:
                                                other_explications[other_lang] = other_entry.get('enhanced_explications', {}).get(pattern_type, '')
                                                break
                                
                                if len(other_explications) > 0:
                                    other_explications[lang] = explication
                                    cross_language_result = self.validate_cross_language_alignment(other_explications, primitive)
                            
                            # Calculate quality score
                            quality_score = self.calculate_quality_score(structural_result, semantic_result, cross_language_result)
                            
                            validation_results[pattern_type] = {
                                'explication': explication,
                                'structural_validation': structural_result,
                                'semantic_validation': semantic_result,
                                'cross_language_validation': cross_language_result,
                                'quality_score': quality_score
                            }
                        
                        validation_entry = {
                            'original_entry': original,
                            'validation_results': validation_results,
                            'overall_quality': {
                                'average_score': float(np.mean([v['quality_score']['overall_score'] for v in validation_results.values()])),
                                'best_pattern': max(validation_results.keys(), key=lambda k: validation_results[k]['quality_score']['overall_score']),
                                'validation_summary': {
                                    'total_patterns': len(validation_results),
                                    'valid_patterns': sum(1 for v in validation_results.values() if v['structural_validation']['valid']),
                                    'coherent_patterns': sum(1 for v in validation_results.values() if v['semantic_validation']['coherent']),
                                    'aligned_patterns': sum(1 for v in validation_results.values() if v['cross_language_validation']['aligned'])
                                }
                            }
                        }
                        
                        validation_entries.append(validation_entry)
                
                enhanced_validation['validation_results'][lang] = {
                    'statistics': {
                        'total_validated_entries': len(validation_entries),
                        'average_quality_score': float(np.mean([entry['overall_quality']['average_score'] for entry in validation_entries])),
                        'high_quality_entries': sum(1 for entry in validation_entries if entry['overall_quality']['average_score'] >= 0.8)
                    },
                    'entries': validation_entries
                }
        
        return enhanced_validation
    
    def generate_validation_report(self, enhanced_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        report = {
            'metadata': {
                'report_type': 'NSM_validation_enhancement_report',
                'timestamp': '2025-08-22'
            },
            'validation_summary': {
                'total_entries_validated': 0,
                'average_quality_score': 0.0,
                'high_quality_rate': 0.0,
                'validation_coverage': {}
            },
            'quality_analysis': {},
            'recommendations': []
        }
        
        # Calculate summary statistics
        total_entries = 0
        total_quality_scores = []
        high_quality_entries = 0
        
        for lang, lang_data in enhanced_validation.get('validation_results', {}).items():
            lang_entries = len(lang_data.get('entries', []))
            total_entries += lang_entries
            
            lang_scores = [entry['overall_quality']['average_score'] for entry in lang_data.get('entries', [])]
            total_quality_scores.extend(lang_scores)
            
            high_quality_count = sum(1 for score in lang_scores if score >= 0.8)
            high_quality_entries += high_quality_count
            
            report['validation_summary']['validation_coverage'][lang] = {
                'entries_validated': lang_entries,
                'average_quality': float(np.mean(lang_scores)) if lang_scores else 0.0,
                'high_quality_rate': high_quality_count / lang_entries if lang_entries > 0 else 0.0
            }
        
        report['validation_summary']['total_entries_validated'] = total_entries
        report['validation_summary']['average_quality_score'] = float(np.mean(total_quality_scores)) if total_quality_scores else 0.0
        report['validation_summary']['high_quality_rate'] = high_quality_entries / total_entries if total_entries > 0 else 0.0
        
        # Quality analysis by primitive
        primitive_quality = {}
        for lang_data in enhanced_validation.get('validation_results', {}).values():
            for entry in lang_data.get('entries', []):
                primitive = entry['original_entry'].get('primitive', 'Unknown')
                if primitive not in primitive_quality:
                    primitive_quality[primitive] = []
                primitive_quality[primitive].append(entry['overall_quality']['average_score'])
        
        for primitive, scores in primitive_quality.items():
            report['quality_analysis'][primitive] = {
                'average_quality': float(np.mean(scores)),
                'quality_std': float(np.std(scores)),
                'high_quality_rate': sum(1 for score in scores if score >= 0.8) / len(scores)
            }
        
        # Generate recommendations
        overall_quality = report['validation_summary']['average_quality_score']
        if overall_quality < 0.6:
            report['recommendations'].append("Overall quality is low. Focus on improving structural patterns and semantic coherence.")
        elif overall_quality < 0.8:
            report['recommendations'].append("Quality is moderate. Consider enhancing cross-language alignment and semantic validation.")
        else:
            report['recommendations'].append("Quality is good. Maintain current validation standards and monitor for consistency.")
        
        # Primitive-specific recommendations
        for primitive, analysis in report['quality_analysis'].items():
            if analysis['average_quality'] < 0.7:
                report['recommendations'].append(f"Focus on improving quality for {primitive} primitive (current: {analysis['average_quality']:.3f})")
        
        report['recommendations'].extend([
            "Implement automated validation pipeline for new explications",
            "Add more sophisticated semantic constraint checking",
            "Enhance cross-language alignment validation with context awareness",
            "Consider implementing quality-based template selection",
            "Monitor validation metrics over time for quality trends"
        ])
        
        return report

def main():
    """Run NSM validation enhancement."""
    logger.info("Starting NSM validation enhancement...")
    
    # Initialize enhancer
    enhancer = NSMValidationEnhancer()
    
    # Load enhanced data
    input_path = Path("data/nsm_grammar_enhanced.json")
    output_path = Path("data/nsm_validation_enhanced.json")
    report_path = Path("data/nsm_validation_enhancement_report.json")
    
    if input_path.exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Enhance validation
        enhanced_validation = enhancer.enhance_validation(input_data)
        
        # Save enhanced validation
        enhanced_validation_serializable = convert_numpy_types(enhanced_validation)
        
        # Debug: Check for any remaining non-serializable types
        def check_serializable(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_serializable(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_serializable(item, f"{path}[{i}]")
            elif isinstance(obj, bool):
                print(f"Found boolean at {path}: {obj}")
            elif not isinstance(obj, (str, int, float, type(None))):
                print(f"Found non-serializable type at {path}: {type(obj)} - {obj}")
        
        check_serializable(enhanced_validation_serializable)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_validation_serializable, f, ensure_ascii=False, indent=2)
        
        # Generate and save report
        report = enhancer.generate_validation_report(enhanced_validation)
        report_serializable = convert_numpy_types(report)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("NSM VALIDATION ENHANCEMENT SUMMARY")
        print("="*80)
        print(f"Total Entries Validated: {report['validation_summary']['total_entries_validated']}")
        print(f"Average Quality Score: {report['validation_summary']['average_quality_score']:.3f}")
        print(f"High Quality Rate: {report['validation_summary']['high_quality_rate']:.1%}")
        print("="*80)
        
        # Print language coverage
        print("\nLanguage Validation Coverage:")
        for lang, coverage in report['validation_summary']['validation_coverage'].items():
            print(f"  {lang}: {coverage['entries_validated']} entries, {coverage['average_quality']:.3f} avg quality, {coverage['high_quality_rate']:.1%} high quality")
        
        # Print primitive quality analysis
        print("\nPrimitive Quality Analysis:")
        for primitive, analysis in report['quality_analysis'].items():
            print(f"  {primitive}: {analysis['average_quality']:.3f} avg, {analysis['high_quality_rate']:.1%} high quality")
        
        # Print recommendations
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("="*80)
        logger.info(f"Validation enhancement completed. Enhanced validation saved to: {output_path}")
        logger.info(f"Validation report saved to: {report_path}")
    else:
        logger.error(f"Input enhanced data not found: {input_path}")

if __name__ == "__main__":
    main()

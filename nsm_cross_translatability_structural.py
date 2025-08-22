#!/usr/bin/env python3
"""
NSM Cross-Translatability Structural Check System.

This script implements structural checks for NSM cross-translatability across EN/ES/FR:
1. Structural consistency validation across languages
2. Semantic alignment verification
3. Cross-language primitive mapping validation
4. Structural pattern analysis
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NSMCrossTranslatabilityChecker:
    """Implements structural checks for NSM cross-translatability."""
    
    def __init__(self):
        """Initialize the cross-translatability checker."""
        self.sbert_model = None
        self.languages = ['en', 'es', 'fr']
        self._load_models()
        
        # Structural patterns for validation
        self.structural_patterns = {
            'AtLocation': {
                'en': ['this thing is in this place', 'this thing exists in this location'],
                'es': ['esta cosa está en este lugar', 'esta cosa existe en esta ubicación'],
                'fr': ['cette chose est dans cet endroit', 'cette chose existe dans cet endroit']
            },
            'HasProperty': {
                'en': ['this thing has this characteristic', 'this thing is characterized by this property'],
                'es': ['esta cosa tiene esta característica', 'esta cosa se caracteriza por esta propiedad'],
                'fr': ['cette chose a cette caractéristique', 'cette chose se caractérise par cette propriété']
            },
            'PartOf': {
                'en': ['this thing is a component of this whole', 'this thing belongs to this larger thing'],
                'es': ['esta cosa es un componente de este todo', 'esta cosa pertenece a esta cosa más grande'],
                'fr': ['cette chose est un composant de cet ensemble', 'cette chose appartient à cette chose plus grande']
            },
            'Causes': {
                'en': ['this thing makes this other thing happen', 'this thing leads to this result'],
                'es': ['esta cosa hace que esta otra cosa suceda', 'esta cosa lleva a este resultado'],
                'fr': ['cette chose fait que cette autre chose se passe', 'cette chose mène à ce résultat']
            },
            'UsedFor': {
                'en': ['people can use this thing to do this action', 'this thing serves this purpose'],
                'es': ['la gente puede usar esta cosa para hacer esta acción', 'esta cosa sirve para este propósito'],
                'fr': ['les gens peuvent utiliser cette chose pour faire cette action', 'cette chose sert à ce but']
            }
        }
    
    def _load_models(self):
        """Load SBERT model for semantic similarity."""
        try:
            logger.info("Loading SBERT model for cross-translatability checks...")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SBERT model: {e}")
            self.sbert_model = None
    
    def check_structural_consistency(self, primitive: str, explications: Dict[str, str]) -> Dict[str, Any]:
        """Check structural consistency of explications across languages."""
        if not self.sbert_model:
            return {'consistency_score': 0.0, 'structural_alignment': 'unknown'}
        
        try:
            # Get explications for all languages
            lang_explications = []
            for lang in self.languages:
                if lang in explications:
                    lang_explications.append(explications[lang])
                else:
                    # Use default pattern if missing
                    default_patterns = self.structural_patterns.get(primitive, {}).get(lang, ['default pattern'])
                    lang_explications.append(default_patterns[0])
            
            # Encode all explications
            embeddings = self.sbert_model.encode(lang_explications)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    similarities.append(sim)
            
            # Calculate consistency score
            consistency_score = np.mean(similarities) if similarities else 0.0
            
            # Determine structural alignment
            if consistency_score >= 0.8:
                alignment = 'excellent'
            elif consistency_score >= 0.6:
                alignment = 'good'
            elif consistency_score >= 0.4:
                alignment = 'fair'
            else:
                alignment = 'poor'
            
            return {
                'consistency_score': float(consistency_score),
                'structural_alignment': alignment,
                'pairwise_similarities': [float(s) for s in similarities],
                'explications_used': lang_explications
            }
        except Exception as e:
            logger.warning(f"Structural consistency check failed: {e}")
            return {'consistency_score': 0.0, 'structural_alignment': 'error'}
    
    def validate_primitive_mapping(self, primitive: str, language_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that primitive mapping is consistent across languages."""
        mapping_validation = {
            'primitive': primitive,
            'languages_present': [],
            'mapping_consistency': 0.0,
            'structural_patterns': {},
            'validation_results': {}
        }
        
        # Check which languages have this primitive
        for lang in self.languages:
            if lang in language_data:
                mapping_validation['languages_present'].append(lang)
        
        # Validate structural patterns for each language
        for lang in mapping_validation['languages_present']:
            lang_entries = language_data[lang].get('entries', [])
            primitive_entries = [entry for entry in lang_entries if entry.get('primitive') == primitive]
            
            if primitive_entries:
                # Get the best explication from refined results
                best_explication = None
                best_score = 0.0
                
                for entry in primitive_entries:
                    if 'best_refined_explication' in entry:
                        explication = entry['best_refined_explication']
                        score = entry.get('best_refined_scores', {}).get('combined_score', 0.0)
                        if score > best_score:
                            best_score = score
                            best_explication = explication
                
                mapping_validation['structural_patterns'][lang] = {
                    'explication': best_explication,
                    'score': best_score,
                    'entry_count': len(primitive_entries)
                }
        
        # Calculate mapping consistency
        if len(mapping_validation['languages_present']) >= 2:
            explications = {}
            for lang, pattern_data in mapping_validation['structural_patterns'].items():
                explications[lang] = pattern_data['explication']
            
            consistency_result = self.check_structural_consistency(primitive, explications)
            mapping_validation['mapping_consistency'] = consistency_result['consistency_score']
            mapping_validation['validation_results'] = consistency_result
        
        return mapping_validation
    
    def analyze_cross_language_patterns(self, refined_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-language patterns in the refined data."""
        logger.info("Analyzing cross-language patterns...")
        
        analysis_results = {
            'metadata': {
                'analysis_type': 'cross_language_pattern_analysis',
                'languages_analyzed': self.languages,
                'total_primitives': 0,
                'consistent_primitives': 0
            },
            'primitive_analysis': {},
            'language_comparison': {},
            'overall_consistency': 0.0
        }
        
        # Get all unique primitives
        all_primitives = set()
        for lang in self.languages:
            if lang in refined_data.get('per_language', {}):
                lang_entries = refined_data['per_language'][lang].get('entries', [])
                for entry in lang_entries:
                    if 'original_entry' in entry:
                        primitive = entry['original_entry'].get('primitive', 'Unknown')
                        all_primitives.add(primitive)
        
        analysis_results['metadata']['total_primitives'] = len(all_primitives)
        
        # Analyze each primitive
        primitive_consistencies = []
        for primitive in all_primitives:
            # Create language data structure for validation
            language_data = {}
            for lang in self.languages:
                if lang in refined_data.get('per_language', {}):
                    lang_entries = refined_data['per_language'][lang].get('entries', [])
                    language_data[lang] = {'entries': lang_entries}
            
            # Validate primitive mapping
            validation_result = self.validate_primitive_mapping(primitive, language_data)
            analysis_results['primitive_analysis'][primitive] = validation_result
            
            if validation_result['mapping_consistency'] > 0.6:
                analysis_results['metadata']['consistent_primitives'] += 1
            
            primitive_consistencies.append(validation_result['mapping_consistency'])
        
        # Calculate overall consistency
        if primitive_consistencies:
            analysis_results['overall_consistency'] = np.mean(primitive_consistencies)
        
        # Language comparison analysis
        for lang in self.languages:
            if lang in refined_data.get('per_language', {}):
                lang_data = refined_data['per_language'][lang]
                analysis_results['language_comparison'][lang] = {
                    'total_entries': lang_data.get('statistics', {}).get('total_entries', 0),
                    'average_improvement': lang_data.get('statistics', {}).get('average_improvement', 0.0),
                    'positive_improvements': lang_data.get('statistics', {}).get('positive_improvements', 0)
                }
        
        return analysis_results
    
    def generate_structural_report(self, refined_data_path: str, output_path: str):
        """Generate a comprehensive structural cross-translatability report."""
        logger.info(f"Generating structural cross-translatability report...")
        
        # Load refined data
        with open(refined_data_path, 'r', encoding='utf-8') as f:
            refined_data = json.load(f)
        
        # Analyze cross-language patterns
        analysis_results = self.analyze_cross_language_patterns(refined_data)
        
        # Generate detailed report
        report = {
            'metadata': {
                'report_type': 'NSM_cross_translatability_structural_report',
                'timestamp': '2025-08-22',
                'data_source': refined_data_path,
                'models_used': {
                    'sbert': self.sbert_model is not None
                }
            },
            'analysis_results': analysis_results,
            'summary': {
                'overall_consistency': analysis_results['overall_consistency'],
                'consistent_primitives': analysis_results['metadata']['consistent_primitives'],
                'total_primitives': analysis_results['metadata']['total_primitives'],
                'consistency_rate': analysis_results['metadata']['consistent_primitives'] / analysis_results['metadata']['total_primitives'] if analysis_results['metadata']['total_primitives'] > 0 else 0.0
            },
            'recommendations': self._generate_recommendations(analysis_results)
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Structural report saved to: {output_path}")
        return report
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        overall_consistency = analysis_results.get('overall_consistency', 0.0)
        total_primitives = analysis_results.get('metadata', {}).get('total_primitives', 0)
        consistent_primitives = analysis_results.get('metadata', {}).get('consistent_primitives', 0)
        
        # Overall consistency recommendations
        if overall_consistency < 0.5:
            recommendations.append("Overall cross-language consistency is low. Focus on improving structural alignment.")
        elif overall_consistency < 0.7:
            recommendations.append("Cross-language consistency is moderate. Consider refining templates for better alignment.")
        else:
            recommendations.append("Cross-language consistency is good. Maintain current structural patterns.")
        
        # Primitive-specific recommendations
        primitive_analysis = analysis_results.get('primitive_analysis', {})
        low_consistency_primitives = []
        
        for primitive, analysis in primitive_analysis.items():
            consistency = analysis.get('mapping_consistency', 0.0)
            if consistency < 0.5:
                low_consistency_primitives.append(primitive)
        
        if low_consistency_primitives:
            recommendations.append(f"Focus on improving consistency for primitives: {', '.join(low_consistency_primitives[:5])}")
        
        # Language-specific recommendations
        language_comparison = analysis_results.get('language_comparison', {})
        for lang, data in language_comparison.items():
            avg_improvement = data.get('average_improvement', 0.0)
            if avg_improvement < 0.05:
                recommendations.append(f"Consider enhancing templates for {lang} language (low improvement rate: {avg_improvement:.3f})")
        
        # General recommendations
        if consistent_primitives / total_primitives < 0.7:
            recommendations.append("Less than 70% of primitives show good cross-language consistency. Review structural patterns.")
        
        recommendations.append("Consider implementing language-specific template optimization for better cross-translatability.")
        recommendations.append("Monitor cross-language consistency as new primitives are added to the system.")
        
        return recommendations

def main():
    """Run NSM cross-translatability structural analysis."""
    logger.info("Starting NSM cross-translatability structural analysis...")
    
    # Initialize checker
    checker = NSMCrossTranslatabilityChecker()
    
    # Generate structural report
    input_path = Path("data/nsm_substitutability_refined.json")
    output_path = Path("data/nsm_cross_translatability_structural_report.json")
    
    if input_path.exists():
        report = checker.generate_structural_report(str(input_path), str(output_path))
        
        # Print summary
        print("\n" + "="*80)
        print("NSM CROSS-TRANSLATABILITY STRUCTURAL ANALYSIS SUMMARY")
        print("="*80)
        print(f"Overall Consistency: {report['summary']['overall_consistency']:.3f}")
        print(f"Consistent Primitives: {report['summary']['consistent_primitives']}/{report['summary']['total_primitives']}")
        print(f"Consistency Rate: {report['summary']['consistency_rate']:.1%}")
        print("="*80)
        
        # Print primitive analysis
        print("\nPrimitive Consistency Analysis:")
        primitive_analysis = report['analysis_results']['primitive_analysis']
        for primitive, analysis in sorted(primitive_analysis.items()):
            consistency = analysis.get('mapping_consistency', 0.0)
            alignment = analysis.get('validation_results', {}).get('structural_alignment', 'unknown')
            print(f"  {primitive}: {consistency:.3f} ({alignment})")
        
        # Print recommendations
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("="*80)
    else:
        logger.error(f"Input refined data not found: {input_path}")

if __name__ == "__main__":
    main()

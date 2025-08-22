#!/usr/bin/env python3
"""
NSM Primitive Alignment System.

This script addresses the fundamental mismatch between ConceptNet relations and NSM primitives
by creating an honest alignment system that maps between the two approaches.
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


class NSMPrimitiveMapper:
    """Mapper between NSM primitives and ConceptNet relations."""
    
    def __init__(self):
        """Initialize the NSM primitive mapper."""
        
        # NSM primitive definitions (from our analysis)
        self.nsm_primitives = {
            # Substantives
            "I": {"category": "substantive", "arity": 1, "description": "First person singular"},
            "YOU": {"category": "substantive", "arity": 1, "description": "Second person"},
            "SOMEONE": {"category": "substantive", "arity": 1, "description": "Person reference"},
            "PEOPLE": {"category": "substantive", "arity": 1, "description": "Multiple persons"},
            "SOMETHING": {"category": "substantive", "arity": 1, "description": "Thing reference"},
            "BODY": {"category": "substantive", "arity": 1, "description": "Physical body"},
            
            # Determiners
            "THIS": {"category": "determiner", "arity": 1, "description": "Proximal demonstrative"},
            "THE_SAME": {"category": "determiner", "arity": 2, "description": "Identity relation"},
            "OTHER": {"category": "determiner", "arity": 1, "description": "Different from current"},
            
            # Quantifiers
            "ONE": {"category": "quantifier", "arity": 1, "description": "Singular quantity"},
            "TWO": {"category": "quantifier", "arity": 1, "description": "Dual quantity"},
            "SOME": {"category": "quantifier", "arity": 1, "description": "Indefinite quantity"},
            "ALL": {"category": "quantifier", "arity": 1, "description": "Universal quantity"},
            "MANY": {"category": "quantifier", "arity": 1, "description": "Large quantity"},
            "MUCH": {"category": "quantifier", "arity": 1, "description": "Large amount"},
            "FEW": {"category": "quantifier", "arity": 1, "description": "Small quantity"},
            "LITTLE": {"category": "quantifier", "arity": 1, "description": "Small amount"},
            
            # Evaluators
            "GOOD": {"category": "evaluator", "arity": 1, "description": "Positive evaluation"},
            "BAD": {"category": "evaluator", "arity": 1, "description": "Negative evaluation"},
            
            # Descriptors
            "BIG": {"category": "descriptor", "arity": 1, "description": "Large size"},
            "SMALL": {"category": "descriptor", "arity": 1, "description": "Small size"},
            "LONG": {"category": "descriptor", "arity": 1, "description": "Extended dimension"},
            "SHORT": {"category": "descriptor", "arity": 1, "description": "Reduced dimension"},
            
            # Mental predicates
            "THINK": {"category": "mental", "arity": 2, "description": "Cognitive process"},
            "KNOW": {"category": "mental", "arity": 2, "description": "Knowledge state"},
            "WANT": {"category": "mental", "arity": 2, "description": "Desire state"},
            "FEEL": {"category": "mental", "arity": 1, "description": "Emotional state"},
            "SEE": {"category": "mental", "arity": 2, "description": "Visual perception"},
            "HEAR": {"category": "mental", "arity": 2, "description": "Auditory perception"},
            
            # Speech
            "SAY": {"category": "speech", "arity": 2, "description": "Verbal communication"},
            "WORDS": {"category": "speech", "arity": 1, "description": "Verbal content"},
            "TRUE": {"category": "speech", "arity": 1, "description": "Truth value"},
            
            # Actions, events, movement, contact
            "DO": {"category": "action", "arity": 2, "description": "Action performance"},
            "HAPPEN": {"category": "action", "arity": 1, "description": "Event occurrence"},
            "MOVE": {"category": "action", "arity": 2, "description": "Spatial movement"},
            "TOUCH": {"category": "action", "arity": 2, "description": "Physical contact"},
            
            # Location, existence, possession, specification
            "BE_SOMEWHERE": {"category": "location", "arity": 2, "description": "Spatial location"},
            "THERE_IS": {"category": "existence", "arity": 1, "description": "Existence assertion"},
            "HAVE": {"category": "possession", "arity": 2, "description": "Possession relation"},
            "BE_SOMEONE": {"category": "specification", "arity": 2, "description": "Identity specification"},
            "BE_SOMETHING": {"category": "specification", "arity": 2, "description": "Property specification"},
            "BE_LIKE": {"category": "specification", "arity": 2, "description": "Similarity specification"},
            
            # Life and death
            "LIVE": {"category": "life", "arity": 2, "description": "Life state"},
            "DIE": {"category": "life", "arity": 1, "description": "Death event"},
            
            # Time
            "WHEN_TIME": {"category": "time", "arity": 2, "description": "Temporal location"},
            "NOW": {"category": "time", "arity": 1, "description": "Present time"},
            "BEFORE": {"category": "time", "arity": 2, "description": "Temporal precedence"},
            "AFTER": {"category": "time", "arity": 2, "description": "Temporal succession"},
            "A_LONG_TIME": {"category": "time", "arity": 1, "description": "Extended duration"},
            "A_SHORT_TIME": {"category": "time", "arity": 1, "description": "Brief duration"},
            "FOR_SOME_TIME": {"category": "time", "arity": 1, "description": "Indefinite duration"},
            "MOMENT": {"category": "time", "arity": 1, "description": "Instantaneous time"},
            
            # Space
            "WHERE": {"category": "space", "arity": 2, "description": "Spatial location"},
            "HERE": {"category": "space", "arity": 1, "description": "Proximal location"},
            "ABOVE": {"category": "space", "arity": 2, "description": "Vertical position"},
            "BELOW": {"category": "space", "arity": 2, "description": "Vertical position"},
            "FAR": {"category": "space", "arity": 1, "description": "Distant location"},
            "NEAR": {"category": "space", "arity": 1, "description": "Proximal location"},
            "SIDE": {"category": "space", "arity": 1, "description": "Lateral position"},
            "INSIDE": {"category": "space", "arity": 2, "description": "Containment relation"},
            
            # Logical concepts
            "NOT": {"category": "logical", "arity": 1, "description": "Negation"},
            "MAYBE": {"category": "logical", "arity": 1, "description": "Possibility"},
            "CAN": {"category": "logical", "arity": 2, "description": "Ability"},
            "BECAUSE": {"category": "logical", "arity": 2, "description": "Causation"},
            "IF": {"category": "logical", "arity": 2, "description": "Conditional"},
            
            # Augmentors
            "VERY": {"category": "augmentor", "arity": 1, "description": "Intensification"},
            "MORE": {"category": "augmentor", "arity": 2, "description": "Comparative"},
            "LIKE": {"category": "augmentor", "arity": 2, "description": "Similarity"},
            
            # Taxonomy & Partonomy
            "KIND": {"category": "taxonomy", "arity": 2, "description": "Taxonomic relation"},
            "PART": {"category": "partonomy", "arity": 2, "description": "Part-whole relation"},
            
            # Similarity
            "SIMILAR": {"category": "similarity", "arity": 2, "description": "Similarity relation"},
            "DIFFERENT": {"category": "similarity", "arity": 2, "description": "Difference relation"},
            
            # Intensifier
            "REALLY": {"category": "intensifier", "arity": 1, "description": "Emphasis"}
        }
        
        # ConceptNet relation definitions (from primitives.json)
        self.conceptnet_relations = {
            "Antonym": {"category": "logical", "arity": 2, "description": "Opposite meaning"},
            "AtLocation": {"category": "spatial", "arity": 2, "description": "Spatial location"},
            "CapableOf": {"category": "cognitive", "arity": 2, "description": "Ability relation"},
            "Causes": {"category": "causal", "arity": 2, "description": "Causal relation"},
            "CausesDesire": {"category": "informational", "arity": 2, "description": "Desire causation"},
            "CreatedBy": {"category": "causal", "arity": 2, "description": "Creation relation"},
            "DefinedAs": {"category": "informational", "arity": 2, "description": "Definition relation"},
            "DerivedFrom": {"category": "causal", "arity": 2, "description": "Derivation relation"},
            "Desires": {"category": "cognitive", "arity": 2, "description": "Desire relation"},
            "DistinctFrom": {"category": "logical", "arity": 2, "description": "Distinction relation"},
            "Entails": {"category": "logical", "arity": 2, "description": "Entailment relation"},
            "EtymologicallyDerivedFrom": {"category": "causal", "arity": 2, "description": "Etymological relation"},
            "FormOf": {"category": "structural", "arity": 2, "description": "Form relation"},
            "HasA": {"category": "structural", "arity": 2, "description": "Possession relation"},
            "HasContext": {"category": "informational", "arity": 2, "description": "Context relation"},
            "HasFirstSubevent": {"category": "structural", "arity": 2, "description": "Subevent relation"},
            "HasPrerequisite": {"category": "causal", "arity": 2, "description": "Prerequisite relation"},
            "HasProperty": {"category": "structural", "arity": 2, "description": "Property relation"},
            "HasSubevent": {"category": "structural", "arity": 2, "description": "Subevent relation"},
            "IsA": {"category": "structural", "arity": 2, "description": "Taxonomic relation"},
            "MadeOf": {"category": "structural", "arity": 2, "description": "Material relation"},
            "MannerOf": {"category": "structural", "arity": 2, "description": "Manner relation"},
            "MotivatedByGoal": {"category": "cognitive", "arity": 2, "description": "Goal motivation"},
            "NotDesires": {"category": "cognitive", "arity": 2, "description": "Negative desire"},
            "ObstructedBy": {"category": "causal", "arity": 2, "description": "Obstruction relation"},
            "PartOf": {"category": "structural", "arity": 2, "description": "Part-whole relation"},
            "ReceivesAction": {"category": "structural", "arity": 2, "description": "Action reception"},
            "RelatedTo": {"category": "informational", "arity": 2, "description": "General relation"},
            "SimilarTo": {"category": "logical", "arity": 2, "description": "Similarity relation"},
            "SymbolOf": {"category": "informational", "arity": 2, "description": "Symbol relation"},
            "Synonym": {"category": "logical", "arity": 2, "description": "Synonymy relation"},
            "UsedFor": {"category": "functional", "arity": 2, "description": "Purpose relation"}
        }
        
        # Create mapping between NSM primitives and ConceptNet relations
        self.nsm_to_conceptnet_mapping = self._create_nsm_to_conceptnet_mapping()
        self.conceptnet_to_nsm_mapping = self._create_conceptnet_to_nsm_mapping()
    
    def _create_nsm_to_conceptnet_mapping(self) -> Dict[str, List[str]]:
        """Create mapping from NSM primitives to ConceptNet relations."""
        mapping = {
            # Direct mappings
            "NOT": ["Antonym", "DistinctFrom"],
            "BECAUSE": ["Causes", "HasPrerequisite"],
            "LIKE": ["SimilarTo"],
            "SIMILAR": ["SimilarTo"],
            "DIFFERENT": ["DistinctFrom", "Antonym"],
            "PART": ["PartOf"],
            "KIND": ["IsA"],
            "HAVE": ["HasA", "HasProperty"],
            "THERE_IS": ["RelatedTo"],
            "CAN": ["CapableOf"],
            "WANT": ["Desires"],
            "THINK": ["RelatedTo"],  # No direct mapping
            "KNOW": ["RelatedTo"],   # No direct mapping
            "FEEL": ["RelatedTo"],   # No direct mapping
            "SEE": ["RelatedTo"],    # No direct mapping
            "SAY": ["RelatedTo"],    # No direct mapping
            "DO": ["UsedFor", "CapableOf"],
            "HAPPEN": ["Causes", "HasSubevent"],
            "MOVE": ["AtLocation"],
            "TOUCH": ["AtLocation"],
            "BE_SOMEWHERE": ["AtLocation"],
            "BE_SOMEONE": ["IsA"],
            "BE_SOMETHING": ["IsA", "HasProperty"],
            "BE_LIKE": ["SimilarTo"],
            "LIVE": ["AtLocation"],
            "DIE": ["Causes"],
            "WHEN_TIME": ["HasContext"],
            "NOW": ["HasContext"],
            "BEFORE": ["HasPrerequisite"],
            "AFTER": ["Causes"],
            "WHERE": ["AtLocation"],
            "HERE": ["AtLocation"],
            "ABOVE": ["AtLocation"],
            "BELOW": ["AtLocation"],
            "FAR": ["AtLocation"],
            "NEAR": ["AtLocation"],
            "SIDE": ["AtLocation"],
            "INSIDE": ["AtLocation"],
            "MAYBE": ["RelatedTo"],  # No direct mapping
            "IF": ["HasPrerequisite"],
            "VERY": ["RelatedTo"],   # No direct mapping
            "MORE": ["RelatedTo"],   # No direct mapping
            "REALLY": ["RelatedTo"], # No direct mapping
            
            # Quantifiers (no direct mappings)
            "ALL": ["RelatedTo"],
            "SOME": ["RelatedTo"],
            "MANY": ["RelatedTo"],
            "FEW": ["RelatedTo"],
            "LITTLE": ["RelatedTo"],
            "ONE": ["RelatedTo"],
            "TWO": ["RelatedTo"],
            
            # Evaluators (no direct mappings)
            "GOOD": ["RelatedTo"],
            "BAD": ["RelatedTo"],
            
            # Descriptors (no direct mappings)
            "BIG": ["RelatedTo"],
            "SMALL": ["RelatedTo"],
            "LONG": ["RelatedTo"],
            "SHORT": ["RelatedTo"],
            
            # Substantives (no direct mappings)
            "I": ["RelatedTo"],
            "YOU": ["RelatedTo"],
            "SOMEONE": ["RelatedTo"],
            "PEOPLE": ["RelatedTo"],
            "SOMETHING": ["RelatedTo"],
            "BODY": ["RelatedTo"],
            
            # Determiners (no direct mappings)
            "THIS": ["RelatedTo"],
            "THE_SAME": ["Synonym"],
            "OTHER": ["DistinctFrom"],
            
            # Time concepts (no direct mappings)
            "A_LONG_TIME": ["RelatedTo"],
            "A_SHORT_TIME": ["RelatedTo"],
            "FOR_SOME_TIME": ["RelatedTo"],
            "MOMENT": ["RelatedTo"],
            
            # Speech concepts (no direct mappings)
            "WORDS": ["RelatedTo"],
            "TRUE": ["RelatedTo"],
            
            # Life concepts (no direct mappings)
            "LIVE": ["AtLocation"],
            "DIE": ["Causes"]
        }
        
        return mapping
    
    def _create_conceptnet_to_nsm_mapping(self) -> Dict[str, List[str]]:
        """Create mapping from ConceptNet relations to NSM primitives."""
        mapping = {}
        
        for nsm_prim, conceptnet_rels in self.nsm_to_conceptnet_mapping.items():
            for conceptnet_rel in conceptnet_rels:
                if conceptnet_rel not in mapping:
                    mapping[conceptnet_rel] = []
                mapping[conceptnet_rel].append(nsm_prim)
        
        return mapping
    
    def map_nsm_to_conceptnet(self, nsm_primitive: str) -> List[str]:
        """Map NSM primitive to ConceptNet relations."""
        return self.nsm_to_conceptnet_mapping.get(nsm_primitive, ["RelatedTo"])
    
    def map_conceptnet_to_nsm(self, conceptnet_relation: str) -> List[str]:
        """Map ConceptNet relation to NSM primitives."""
        return self.conceptnet_to_nsm_mapping.get(conceptnet_relation, [])
    
    def get_nsm_primitive_info(self, primitive: str) -> Dict[str, Any]:
        """Get information about an NSM primitive."""
        return self.nsm_primitives.get(primitive, {})
    
    def get_conceptnet_relation_info(self, relation: str) -> Dict[str, Any]:
        """Get information about a ConceptNet relation."""
        return self.conceptnet_relations.get(relation, {})
    
    def list_nsm_primitives(self) -> List[str]:
        """List all NSM primitives."""
        return list(self.nsm_primitives.keys())
    
    def list_conceptnet_relations(self) -> List[str]:
        """List all ConceptNet relations."""
        return list(self.conceptnet_relations.keys())


class NSMPrimitiveAlignmentSystem:
    """System for aligning NSM primitives with ConceptNet relations."""
    
    def __init__(self):
        """Initialize the alignment system."""
        self.mapper = NSMPrimitiveMapper()
        self.nsm_translator = NSMTranslator()
        
        # System configuration
        self.system_config = {
            'enable_mapping_analysis': True,
            'enable_detection_alignment': True,
            'enable_validation': True
        }
    
    def analyze_primitive_mismatch(self) -> Dict[str, Any]:
        """Analyze the mismatch between NSM primitives and ConceptNet relations."""
        logger.info("Analyzing primitive system mismatch...")
        
        analysis = {
            'nsm_primitives_count': len(self.mapper.list_nsm_primitives()),
            'conceptnet_relations_count': len(self.mapper.list_conceptnet_relations()),
            'mapping_coverage': {},
            'unmapped_nsm': [],
            'unmapped_conceptnet': [],
            'mapping_quality': {},
            'recommendations': []
        }
        
        # Analyze mapping coverage
        nsm_primitives = self.mapper.list_nsm_primitives()
        conceptnet_relations = self.mapper.list_conceptnet_relations()
        
        # Check NSM to ConceptNet mapping
        mapped_nsm = 0
        for nsm_prim in nsm_primitives:
            conceptnet_mappings = self.mapper.map_nsm_to_conceptnet(nsm_prim)
            if conceptnet_mappings and conceptnet_mappings != ["RelatedTo"]:
                mapped_nsm += 1
            else:
                analysis['unmapped_nsm'].append(nsm_prim)
        
        # Check ConceptNet to NSM mapping
        mapped_conceptnet = 0
        for conceptnet_rel in conceptnet_relations:
            nsm_mappings = self.mapper.map_conceptnet_to_nsm(conceptnet_rel)
            if nsm_mappings:
                mapped_conceptnet += 1
            else:
                analysis['unmapped_conceptnet'].append(conceptnet_rel)
        
        # Calculate coverage
        analysis['mapping_coverage'] = {
            'nsm_to_conceptnet': mapped_nsm / len(nsm_primitives) if nsm_primitives else 0,
            'conceptnet_to_nsm': mapped_conceptnet / len(conceptnet_relations) if conceptnet_relations else 0
        }
        
        # Analyze mapping quality
        analysis['mapping_quality'] = {
            'direct_mappings': 0,
            'indirect_mappings': 0,
            'fallback_mappings': 0
        }
        
        for nsm_prim in nsm_primitives:
            conceptnet_mappings = self.mapper.map_nsm_to_conceptnet(nsm_prim)
            if len(conceptnet_mappings) == 1 and conceptnet_mappings[0] != "RelatedTo":
                analysis['mapping_quality']['direct_mappings'] += 1
            elif len(conceptnet_mappings) > 1:
                analysis['mapping_quality']['indirect_mappings'] += 1
            elif conceptnet_mappings == ["RelatedTo"]:
                analysis['mapping_quality']['fallback_mappings'] += 1
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_mismatch_recommendations(analysis)
        
        return analysis
    
    def _generate_mismatch_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for addressing the primitive mismatch."""
        recommendations = []
        
        # Coverage recommendations
        nsm_coverage = analysis['mapping_coverage']['nsm_to_conceptnet']
        conceptnet_coverage = analysis['mapping_coverage']['conceptnet_to_nsm']
        
        if nsm_coverage < 0.5:
            recommendations.append(f"Low NSM to ConceptNet mapping coverage ({nsm_coverage:.1%}) - many NSM primitives lack proper mappings")
        
        if conceptnet_coverage < 0.5:
            recommendations.append(f"Low ConceptNet to NSM mapping coverage ({conceptnet_coverage:.1%}) - many ConceptNet relations lack NSM equivalents")
        
        # Unmapped primitives
        unmapped_nsm_count = len(analysis['unmapped_nsm'])
        if unmapped_nsm_count > 0:
            recommendations.append(f"{unmapped_nsm_count} NSM primitives lack ConceptNet mappings - need to add these relations")
        
        unmapped_conceptnet_count = len(analysis['unmapped_conceptnet'])
        if unmapped_conceptnet_count > 0:
            recommendations.append(f"{unmapped_conceptnet_count} ConceptNet relations lack NSM mappings - need to add NSM primitives")
        
        # Mapping quality
        fallback_count = analysis['mapping_quality']['fallback_mappings']
        if fallback_count > 0:
            recommendations.append(f"{fallback_count} NSM primitives only have fallback 'RelatedTo' mappings - need specific mappings")
        
        # Overall assessment
        if nsm_coverage < 0.3:
            recommendations.append("CRITICAL: Very low mapping coverage indicates fundamental system mismatch")
        elif nsm_coverage < 0.7:
            recommendations.append("IMPORTANT: Moderate mapping coverage - significant work needed")
        else:
            recommendations.append("Good mapping coverage - focus on remaining edge cases")
        
        return recommendations
    
    def create_aligned_primitive_database(self) -> Dict[str, Any]:
        """Create an aligned primitive database that combines both approaches."""
        logger.info("Creating aligned primitive database...")
        
        aligned_database = {
            'version': '1.0.0',
            'description': 'Aligned primitive database combining NSM primitives and ConceptNet relations',
            'primitives': {},
            'mappings': {},
            'categories': {}
        }
        
        # Add NSM primitives
        for nsm_prim, info in self.mapper.nsm_primitives.items():
            aligned_database['primitives'][nsm_prim] = {
                'type': 'nsm',
                'category': info['category'],
                'arity': info['arity'],
                'description': info['description'],
                'conceptnet_mappings': self.mapper.map_nsm_to_conceptnet(nsm_prim)
            }
        
        # Add ConceptNet relations
        for conceptnet_rel, info in self.mapper.conceptnet_relations.items():
            if conceptnet_rel not in aligned_database['primitives']:
                aligned_database['primitives'][conceptnet_rel] = {
                    'type': 'conceptnet',
                    'category': info['category'],
                    'arity': info['arity'],
                    'description': info['description'],
                    'nsm_mappings': self.mapper.map_conceptnet_to_nsm(conceptnet_rel)
                }
        
        # Add mappings
        aligned_database['mappings'] = {
            'nsm_to_conceptnet': self.mapper.nsm_to_conceptnet_mapping,
            'conceptnet_to_nsm': self.mapper.conceptnet_to_nsm_mapping
        }
        
        # Add categories
        aligned_database['categories'] = {
            'nsm_categories': list(set(info['category'] for info in self.mapper.nsm_primitives.values())),
            'conceptnet_categories': list(set(info['category'] for info in self.mapper.conceptnet_relations.values()))
        }
        
        return aligned_database
    
    def run_alignment_analysis(self) -> Dict[str, Any]:
        """Run comprehensive alignment analysis."""
        logger.info("Running NSM primitive alignment analysis...")
        
        analysis_results = {
            'mismatch_analysis': {},
            'aligned_database': {},
            'recommendations': []
        }
        
        # Analyze mismatch
        if self.system_config['enable_mapping_analysis']:
            analysis_results['mismatch_analysis'] = self.analyze_primitive_mismatch()
        
        # Create aligned database
        if self.system_config['enable_detection_alignment']:
            analysis_results['aligned_database'] = self.create_aligned_primitive_database()
        
        # Generate overall recommendations
        analysis_results['recommendations'] = self._generate_alignment_recommendations(analysis_results)
        
        return analysis_results
    
    def _generate_alignment_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate overall alignment recommendations."""
        recommendations = []
        
        mismatch_analysis = analysis_results.get('mismatch_analysis', {})
        aligned_database = analysis_results.get('aligned_database', {})
        
        # Add mismatch recommendations
        recommendations.extend(mismatch_analysis.get('recommendations', []))
        
        # Database recommendations
        if aligned_database:
            primitive_count = len(aligned_database.get('primitives', {}))
            recommendations.append(f"Created aligned database with {primitive_count} primitives")
            recommendations.append("Use aligned database for consistent primitive detection across both systems")
        
        # System recommendations
        recommendations.append("Implement detection system that can handle both NSM primitives and ConceptNet relations")
        recommendations.append("Create unified primitive detection interface that maps between approaches")
        recommendations.append("Update existing detection code to use aligned primitive database")
        
        return recommendations


def main():
    """Main function to run NSM primitive alignment analysis."""
    logger.info("Starting NSM primitive alignment analysis...")
    
    # Initialize alignment system
    alignment_system = NSMPrimitiveAlignmentSystem()
    
    # Run alignment analysis
    analysis_results = alignment_system.run_alignment_analysis()
    
    # Print results
    print("\n" + "="*80)
    print("NSM PRIMITIVE ALIGNMENT ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Mismatch Analysis:")
    mismatch = analysis_results['mismatch_analysis']
    print(f"  NSM Primitives: {mismatch['nsm_primitives_count']}")
    print(f"  ConceptNet Relations: {mismatch['conceptnet_relations_count']}")
    print(f"  NSM to ConceptNet Coverage: {mismatch['mapping_coverage']['nsm_to_conceptnet']:.1%}")
    print(f"  ConceptNet to NSM Coverage: {mismatch['mapping_coverage']['conceptnet_to_nsm']:.1%}")
    
    print(f"\nMapping Quality:")
    quality = mismatch['mapping_quality']
    print(f"  Direct Mappings: {quality['direct_mappings']}")
    print(f"  Indirect Mappings: {quality['indirect_mappings']}")
    print(f"  Fallback Mappings: {quality['fallback_mappings']}")
    
    print(f"\nUnmapped NSM Primitives ({len(mismatch['unmapped_nsm'])}):")
    for i, primitive in enumerate(mismatch['unmapped_nsm'][:10]):  # Show first 10
        print(f"  {i+1}. {primitive}")
    if len(mismatch['unmapped_nsm']) > 10:
        print(f"  ... and {len(mismatch['unmapped_nsm']) - 10} more")
    
    print(f"\nUnmapped ConceptNet Relations ({len(mismatch['unmapped_conceptnet'])}):")
    for i, relation in enumerate(mismatch['unmapped_conceptnet'][:10]):  # Show first 10
        print(f"  {i+1}. {relation}")
    if len(mismatch['unmapped_conceptnet']) > 10:
        print(f"  ... and {len(mismatch['unmapped_conceptnet']) - 10} more")
    
    print(f"\nAligned Database:")
    aligned_db = analysis_results['aligned_database']
    if aligned_db:
        print(f"  Total Primitives: {len(aligned_db.get('primitives', {}))}")
        print(f"  NSM Categories: {len(aligned_db.get('categories', {}).get('nsm_categories', []))}")
        print(f"  ConceptNet Categories: {len(aligned_db.get('categories', {}).get('conceptnet_categories', []))}")
    
    print(f"\nExample Mappings:")
    mapper = alignment_system.mapper
    example_nsm_primitives = ["NOT", "BECAUSE", "LIKE", "PART", "CAN", "WANT"]
    for nsm_prim in example_nsm_primitives:
        conceptnet_mappings = mapper.map_nsm_to_conceptnet(nsm_prim)
        print(f"  {nsm_prim} → {conceptnet_mappings}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_path = "data/nsm_primitive_alignment_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(analysis_results), f, ensure_ascii=False, indent=2)
    
    # Save aligned database
    if aligned_db:
        db_path = "data/aligned_primitives.json"
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(aligned_db), f, ensure_ascii=False, indent=2)
        logger.info(f"Aligned primitive database saved to: {db_path}")
    
    logger.info(f"NSM primitive alignment report saved to: {output_path}")
    
    print("="*80)
    print("NSM primitive alignment analysis completed!")
    print("="*80)


if __name__ == "__main__":
    main()

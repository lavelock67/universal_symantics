#!/usr/bin/env python3
"""
Executable Interlingua (EIL) Specification System.

This script implements an Executable Interlingua specification system to compile
NSM meanings to a typed IR for reasoning without token-level guessing.
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
from enum import Enum
from dataclasses import dataclass, asdict

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


class EILPredicateType(Enum):
    """EIL predicate types for typed reasoning."""
    EVENT = "event"
    ROLE = "role"
    TIME = "time"
    POLARITY = "polarity"
    MODALITY = "modality"
    CAUSATION = "causation"
    QUANTIFIER = "quantifier"
    ASPECTUAL = "aspectual"
    EXPERIENCER = "experiencer"
    PROPERTY = "property"
    LOCATION = "location"
    COMPARISON = "comparison"
    POSSESSION = "possession"
    TEMPORAL = "temporal"
    NEGATION = "negation"


class EILTimeType(Enum):
    """EIL time types for temporal reasoning."""
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    RECENT_PAST = "recent_past"
    DISTANT_PAST = "distant_past"
    IMMEDIATE_FUTURE = "immediate_future"
    DISTANT_FUTURE = "distant_future"
    DURATION = "duration"
    POINT = "point"
    INTERVAL = "interval"


class EILModalityType(Enum):
    """EIL modality types for modal reasoning."""
    NECESSITY = "necessity"
    POSSIBILITY = "possibility"
    PROBABILITY = "probability"
    OBLIGATION = "obligation"
    PERMISSION = "permission"
    ABILITY = "ability"
    INTENTION = "intention"
    DESIRE = "desire"


@dataclass
class EILPredicate:
    """EIL predicate with typed arguments."""
    name: str
    predicate_type: EILPredicateType
    arguments: List[str]
    confidence: float
    source: str  # NSM, UMR, or BMR
    sense_ids: List[str] = None
    
    def __post_init__(self):
        if self.sense_ids is None:
            self.sense_ids = []


@dataclass
class EILFact:
    """EIL fact representing a semantic proposition."""
    predicates: List[EILPredicate]
    scope: Dict[str, Any]
    confidence: float
    source_text: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EILRule:
    """EIL rule for reasoning."""
    name: str
    antecedent: List[EILPredicate]
    consequent: List[EILPredicate]
    confidence: float
    rule_type: str
    description: str = ""


class EILSpecification:
    """EIL specification with type definitions and reasoning rules."""
    
    def __init__(self):
        """Initialize the EIL specification."""
        self.predicate_types = EILPredicateType
        self.time_types = EILTimeType
        self.modality_types = EILModalityType
        
        # Type definitions
        self.type_definitions = {
            'event': {
                'description': 'Events and actions',
                'arguments': ['agent', 'patient', 'instrument', 'location', 'time'],
                'examples': ['DO', 'HAPPEN', 'CAUSE', 'STOP', 'START']
            },
            'role': {
                'description': 'Semantic roles',
                'arguments': ['entity', 'event'],
                'examples': ['AGENT', 'PATIENT', 'EXPERIENCER', 'THEME', 'GOAL']
            },
            'time': {
                'description': 'Temporal relations',
                'arguments': ['event', 'reference_time'],
                'examples': ['PAST', 'FUTURE', 'PRESENT', 'BEFORE', 'AFTER']
            },
            'polarity': {
                'description': 'Polarity and negation',
                'arguments': ['proposition'],
                'examples': ['NOT', 'NEG', 'POSITIVE', 'NEGATIVE']
            },
            'modality': {
                'description': 'Modal operators',
                'arguments': ['proposition', 'degree'],
                'examples': ['CAN', 'MUST', 'SHOULD', 'MIGHT', 'WILL']
            },
            'causation': {
                'description': 'Causal relations',
                'arguments': ['cause', 'effect'],
                'examples': ['CAUSE', 'BECAUSE', 'RESULT', 'LEAD_TO']
            },
            'quantifier': {
                'description': 'Quantification',
                'arguments': ['variable', 'domain', 'condition'],
                'examples': ['ALL', 'SOME', 'MANY', 'FEW', 'MOST']
            },
            'aspectual': {
                'description': 'Aspectual operators',
                'arguments': ['event', 'aspect'],
                'examples': ['ALMOST', 'JUST', 'STILL', 'ALREADY', 'YET']
            },
            'experiencer': {
                'description': 'Experiencer relations',
                'arguments': ['experiencer', 'stimulus'],
                'examples': ['LIKE', 'HATE', 'ENJOY', 'PREFER', 'FEEL']
            },
            'property': {
                'description': 'Property attribution',
                'arguments': ['entity', 'property'],
                'examples': ['HASPROPERTY', 'IS', 'HAVE', 'POSSESS']
            },
            'location': {
                'description': 'Spatial relations',
                'arguments': ['entity', 'location'],
                'examples': ['ATLOCATION', 'IN', 'ON', 'NEAR', 'FAR']
            },
            'comparison': {
                'description': 'Comparison relations',
                'arguments': ['entity1', 'entity2', 'dimension'],
                'examples': ['SIMILARTO', 'DIFFERENTFROM', 'BIGGER', 'SMALLER']
            },
            'possession': {
                'description': 'Possession relations',
                'arguments': ['possessor', 'possessed'],
                'examples': ['HAVE', 'OWN', 'BELONGTO', 'POSSESS']
            },
            'temporal': {
                'description': 'Temporal operators',
                'arguments': ['event', 'time_expression'],
                'examples': ['WHEN', 'WHILE', 'DURING', 'UNTIL', 'SINCE']
            },
            'negation': {
                'description': 'Negation operators',
                'arguments': ['proposition'],
                'examples': ['NOT', 'NEVER', 'NO', 'NONE', 'NEG']
            }
        }
        
        # Load reasoning rules
        self.reasoning_rules = self._load_reasoning_rules()
    
    def _load_reasoning_rules(self) -> List[EILRule]:
        """Load base reasoning rules."""
        rules = [
            # Negation rules
            EILRule(
                name="double_negation",
                antecedent=[
                    EILPredicate("NOT", EILPredicateType.NEGATION, ["NOT", "P"], 1.0, "EIL")
                ],
                consequent=[
                    EILPredicate("P", EILPredicateType.PROPERTY, ["P"], 1.0, "EIL")
                ],
                confidence=1.0,
                rule_type="negation",
                description="Double negation elimination"
            ),
            
            # Modality rules
            EILRule(
                name="necessity_implies_possibility",
                antecedent=[
                    EILPredicate("MUST", EILModalityType.NECESSITY, ["P"], 1.0, "EIL")
                ],
                consequent=[
                    EILPredicate("CAN", EILModalityType.POSSIBILITY, ["P"], 1.0, "EIL")
                ],
                confidence=0.9,
                rule_type="modality",
                description="Necessity implies possibility"
            ),
            
            # Causation rules
            EILRule(
                name="cause_implies_effect",
                antecedent=[
                    EILPredicate("CAUSE", EILPredicateType.CAUSATION, ["A", "B"], 1.0, "EIL"),
                    EILPredicate("HAPPEN", EILPredicateType.EVENT, ["A"], 1.0, "EIL")
                ],
                consequent=[
                    EILPredicate("HAPPEN", EILPredicateType.EVENT, ["B"], 0.8, "EIL")
                ],
                confidence=0.8,
                rule_type="causation",
                description="If A causes B and A happens, then B happens"
            ),
            
            # Temporal rules
            EILRule(
                name="past_implies_not_future",
                antecedent=[
                    EILPredicate("PAST", EILTimeType.PAST, ["E"], 1.0, "EIL")
                ],
                consequent=[
                    EILPredicate("NOT", EILPredicateType.NEGATION, ["FUTURE", "E"], 1.0, "EIL")
                ],
                confidence=1.0,
                rule_type="temporal",
                description="Past events are not future events"
            ),
            
            # Quantifier rules
            EILRule(
                name="all_implies_some",
                antecedent=[
                    EILPredicate("ALL", EILPredicateType.QUANTIFIER, ["X", "P"], 1.0, "EIL")
                ],
                consequent=[
                    EILPredicate("SOME", EILPredicateType.QUANTIFIER, ["X", "P"], 1.0, "EIL")
                ],
                confidence=1.0,
                rule_type="quantifier",
                description="All implies some"
            ),
            
            # Aspectual rules
            EILRule(
                name="almost_implies_not_complete",
                antecedent=[
                    EILPredicate("ALMOST", EILPredicateType.ASPECTUAL, ["E"], 1.0, "EIL")
                ],
                consequent=[
                    EILPredicate("NOT", EILPredicateType.NEGATION, ["COMPLETE", "E"], 1.0, "EIL")
                ],
                confidence=0.9,
                rule_type="aspectual",
                description="Almost implies not complete"
            ),
            
            # Experiencer rules
            EILRule(
                name="like_implies_not_hate",
                antecedent=[
                    EILPredicate("LIKE", EILPredicateType.EXPERIENCER, ["A", "B"], 1.0, "EIL")
                ],
                consequent=[
                    EILPredicate("NOT", EILPredicateType.NEGATION, ["HATE", "A", "B"], 1.0, "EIL")
                ],
                confidence=0.9,
                rule_type="experiencer",
                description="Liking implies not hating"
            ),
            
            # Property rules
            EILRule(
                name="property_inheritance",
                antecedent=[
                    EILPredicate("IS", EILPredicateType.PROPERTY, ["A", "B"], 1.0, "EIL"),
                    EILPredicate("HASPROPERTY", EILPredicateType.PROPERTY, ["B", "P"], 1.0, "EIL")
                ],
                consequent=[
                    EILPredicate("HASPROPERTY", EILPredicateType.PROPERTY, ["A", "P"], 0.8, "EIL")
                ],
                confidence=0.8,
                rule_type="property",
                description="Property inheritance through IS relation"
            ),
            
            # Location rules
            EILRule(
                name="location_transitivity",
                antecedent=[
                    EILPredicate("IN", EILPredicateType.LOCATION, ["A", "B"], 1.0, "EIL"),
                    EILPredicate("IN", EILPredicateType.LOCATION, ["B", "C"], 1.0, "EIL")
                ],
                consequent=[
                    EILPredicate("IN", EILPredicateType.LOCATION, ["A", "C"], 0.9, "EIL")
                ],
                confidence=0.9,
                rule_type="location",
                description="Location transitivity"
            ),
            
            # Comparison rules
            EILRule(
                name="similarity_symmetry",
                antecedent=[
                    EILPredicate("SIMILARTO", EILPredicateType.COMPARISON, ["A", "B"], 1.0, "EIL")
                ],
                consequent=[
                    EILPredicate("SIMILARTO", EILPredicateType.COMPARISON, ["B", "A"], 1.0, "EIL")
                ],
                confidence=1.0,
                rule_type="comparison",
                description="Similarity is symmetric"
            )
        ]
        
        return rules


class EILCompiler:
    """Compiler from NSM/UMR/BMR to EIL."""
    
    def __init__(self):
        """Initialize the EIL compiler."""
        self.specification = EILSpecification()
        self.nsm_translator = NSMTranslator()
        
        # Compilation mappings
        self.nsm_to_eil_mappings = {
            # Basic predicates
            'DO': ('DO', EILPredicateType.EVENT),
            'HAPPEN': ('HAPPEN', EILPredicateType.EVENT),
            'CAUSE': ('CAUSE', EILPredicateType.CAUSATION),
            'STOP': ('STOP', EILPredicateType.EVENT),
            'START': ('START', EILPredicateType.EVENT),
            
            # Roles
            'AGENT': ('AGENT', EILPredicateType.ROLE),
            'PATIENT': ('PATIENT', EILPredicateType.ROLE),
            'EXPERIENCER': ('EXPERIENCER', EILPredicateType.ROLE),
            'THEME': ('THEME', EILPredicateType.ROLE),
            'GOAL': ('GOAL', EILPredicateType.ROLE),
            
            # Time
            'PAST': ('PAST', EILTimeType.PAST),
            'FUTURE': ('FUTURE', EILTimeType.FUTURE),
            'PRESENT': ('PRESENT', EILTimeType.PRESENT),
            'BEFORE': ('BEFORE', EILPredicateType.TIME),
            'AFTER': ('AFTER', EILPredicateType.TIME),
            
            # Polarity
            'NOT': ('NOT', EILPredicateType.NEGATION),
            'NEG': ('NEG', EILPredicateType.NEGATION),
            'POSITIVE': ('POSITIVE', EILPredicateType.POLARITY),
            'NEGATIVE': ('NEGATIVE', EILPredicateType.POLARITY),
            
            # Modality
            'CAN': ('CAN', EILModalityType.ABILITY),
            'MUST': ('MUST', EILModalityType.NECESSITY),
            'SHOULD': ('SHOULD', EILModalityType.OBLIGATION),
            'MIGHT': ('MIGHT', EILModalityType.POSSIBILITY),
            'WILL': ('WILL', EILModalityType.INTENTION),
            
            # Quantifiers
            'ALL': ('ALL', EILPredicateType.QUANTIFIER),
            'SOME': ('SOME', EILPredicateType.QUANTIFIER),
            'MANY': ('MANY', EILPredicateType.QUANTIFIER),
            'FEW': ('FEW', EILPredicateType.QUANTIFIER),
            'MOST': ('MOST', EILPredicateType.QUANTIFIER),
            
            # Aspectual
            'ALMOST': ('ALMOST', EILPredicateType.ASPECTUAL),
            'JUST': ('JUST', EILPredicateType.ASPECTUAL),
            'STILL': ('STILL', EILPredicateType.ASPECTUAL),
            'ALREADY': ('ALREADY', EILPredicateType.ASPECTUAL),
            'YET': ('YET', EILPredicateType.ASPECTUAL),
            
            # Experiencer
            'LIKE': ('LIKE', EILPredicateType.EXPERIENCER),
            'HATE': ('HATE', EILPredicateType.EXPERIENCER),
            'ENJOY': ('ENJOY', EILPredicateType.EXPERIENCER),
            'PREFER': ('PREFER', EILPredicateType.EXPERIENCER),
            'FEEL': ('FEEL', EILPredicateType.EXPERIENCER),
            
            # Properties
            'HASPROPERTY': ('HASPROPERTY', EILPredicateType.PROPERTY),
            'IS': ('IS', EILPredicateType.PROPERTY),
            'HAVE': ('HAVE', EILPredicateType.PROPERTY),
            'POSSESS': ('POSSESS', EILPredicateType.PROPERTY),
            
            # Location
            'ATLOCATION': ('ATLOCATION', EILPredicateType.LOCATION),
            'IN': ('IN', EILPredicateType.LOCATION),
            'ON': ('ON', EILPredicateType.LOCATION),
            'NEAR': ('NEAR', EILPredicateType.LOCATION),
            'FAR': ('FAR', EILPredicateType.LOCATION),
            
            # Comparison
            'SIMILARTO': ('SIMILARTO', EILPredicateType.COMPARISON),
            'DIFFERENTFROM': ('DIFFERENTFROM', EILPredicateType.COMPARISON),
            'BIGGER': ('BIGGER', EILPredicateType.COMPARISON),
            'SMALLER': ('SMALLER', EILPredicateType.COMPARISON),
            
            # Possession
            'OWN': ('OWN', EILPredicateType.POSSESSION),
            'BELONGTO': ('BELONGTO', EILPredicateType.POSSESSION),
            
            # Temporal
            'WHEN': ('WHEN', EILPredicateType.TEMPORAL),
            'WHILE': ('WHILE', EILPredicateType.TEMPORAL),
            'DURING': ('DURING', EILPredicateType.TEMPORAL),
            'UNTIL': ('UNTIL', EILPredicateType.TEMPORAL),
            'SINCE': ('SINCE', EILPredicateType.TEMPORAL)
        }
    
    def compile_nsm_to_eil(self, text: str, primitives: List[str], 
                          language: str = "en") -> EILFact:
        """Compile NSM primitives to EIL facts."""
        try:
            predicates = []
            
            for primitive in primitives:
                # Map NSM primitive to EIL predicate
                if primitive in self.nsm_to_eil_mappings:
                    eil_name, eil_type = self.nsm_to_eil_mappings[primitive]
                    
                    # Create EIL predicate
                    predicate = EILPredicate(
                        name=eil_name,
                        predicate_type=eil_type,
                        arguments=[primitive, text],  # Simplified arguments
                        confidence=0.8,  # Default confidence
                        source="NSM",
                        sense_ids=[]
                    )
                    predicates.append(predicate)
                else:
                    # Unknown primitive - create generic predicate
                    predicate = EILPredicate(
                        name=primitive,
                        predicate_type=EILPredicateType.PROPERTY,
                        arguments=[primitive, text],
                        confidence=0.5,  # Lower confidence for unknown
                        source="NSM",
                        sense_ids=[]
                    )
                    predicates.append(predicate)
            
            # Create EIL fact
            fact = EILFact(
                predicates=predicates,
                scope={'text': text, 'language': language},
                confidence=0.8,
                source_text=text,
                metadata={'compilation_method': 'nsm_to_eil'}
            )
            
            return fact
        
        except Exception as e:
            logger.warning(f"NSM to EIL compilation failed for '{text}': {e}")
            # Return empty fact on error
            return EILFact(
                predicates=[],
                scope={'text': text, 'language': language},
                confidence=0.0,
                source_text=text,
                metadata={'error': str(e)}
            )
    
    def compile_umr_to_eil(self, umr_data: Dict[str, Any]) -> EILFact:
        """Compile UMR data to EIL facts."""
        try:
            predicates = []
            
            # Extract UMR nodes and edges
            nodes = umr_data.get('nodes', {})
            edges = umr_data.get('edges', [])
            
            # Convert UMR nodes to EIL predicates
            for node_id, node_data in nodes.items():
                node_type = node_data.get('type', 'entity')
                node_label = node_data.get('label', node_id)
                
                # Map UMR node types to EIL predicate types
                if node_type == 'event':
                    predicate_type = EILPredicateType.EVENT
                elif node_type == 'entity':
                    predicate_type = EILPredicateType.PROPERTY
                elif node_type == 'time':
                    predicate_type = EILTimeType.POINT
                else:
                    predicate_type = EILPredicateType.PROPERTY
                
                predicate = EILPredicate(
                    name=node_label,
                    predicate_type=predicate_type,
                    arguments=[node_id, node_label],
                    confidence=0.8,
                    source="UMR",
                    sense_ids=node_data.get('sense_ids', [])
                )
                predicates.append(predicate)
            
            # Convert UMR edges to EIL predicates
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                relation = edge.get('relation', 'RELATED')
                
                predicate = EILPredicate(
                    name=relation,
                    predicate_type=EILPredicateType.PROPERTY,
                    arguments=[source, target],
                    confidence=0.8,
                    source="UMR",
                    sense_ids=edge.get('sense_ids', [])
                )
                predicates.append(predicate)
            
            # Create EIL fact
            fact = EILFact(
                predicates=predicates,
                scope={'umr_data': umr_data},
                confidence=0.8,
                source_text=umr_data.get('text', ''),
                metadata={'compilation_method': 'umr_to_eil'}
            )
            
            return fact
        
        except Exception as e:
            logger.warning(f"UMR to EIL compilation failed: {e}")
            return EILFact(
                predicates=[],
                scope={'umr_data': umr_data},
                confidence=0.0,
                source_text='',
                metadata={'error': str(e)}
            )


class EILReasoner:
    """EIL reasoner for executing logical inference."""
    
    def __init__(self):
        """Initialize the EIL reasoner."""
        self.specification = EILSpecification()
        self.rules = self.specification.reasoning_rules
        
        # Reasoning parameters
        self.reasoning_params = {
            'max_inference_steps': 10,
            'confidence_threshold': 0.7,
            'enable_backward_chaining': True,
            'enable_forward_chaining': True
        }
    
    def reason_over_facts(self, facts: List[EILFact]) -> Dict[str, Any]:
        """Reason over EIL facts using the rule set."""
        try:
            # Extract all predicates from facts
            all_predicates = []
            for fact in facts:
                all_predicates.extend(fact.predicates)
            
            # Apply reasoning rules
            inferences = []
            proof_trees = []
            
            for rule in self.rules:
                # Check if rule antecedent matches any predicates
                rule_inferences = self._apply_rule(rule, all_predicates)
                if rule_inferences:
                    inferences.extend(rule_inferences)
                    
                    # Create proof tree
                    proof_tree = {
                        'rule': rule.name,
                        'antecedent': [asdict(p) for p in rule.antecedent],
                        'consequent': [asdict(p) for p in rule.consequent],
                        'confidence': rule.confidence,
                        'inferences': [asdict(inf) for inf in rule_inferences]
                    }
                    proof_trees.append(proof_tree)
            
            return {
                'original_facts': len(facts),
                'original_predicates': len(all_predicates),
                'inferences_made': len(inferences),
                'proof_trees': proof_trees,
                'inferences': [asdict(inf) for inf in inferences],
                'reasoning_success': len(inferences) > 0
            }
        
        except Exception as e:
            logger.warning(f"EIL reasoning failed: {e}")
            return {
                'original_facts': len(facts),
                'original_predicates': 0,
                'inferences_made': 0,
                'proof_trees': [],
                'inferences': [],
                'reasoning_success': False,
                'error': str(e)
            }
    
    def _apply_rule(self, rule: EILRule, predicates: List[EILPredicate]) -> List[EILPredicate]:
        """Apply a single rule to predicates."""
        try:
            inferences = []
            
            # Check if all antecedent predicates match
            antecedent_matches = []
            for antecedent in rule.antecedent:
                matches = []
                for predicate in predicates:
                    if self._predicates_match(antecedent, predicate):
                        matches.append(predicate)
                antecedent_matches.append(matches)
            
            # If all antecedents have matches, create inferences
            if all(matches for matches in antecedent_matches):
                for consequent in rule.consequent:
                    # Create inferred predicate
                    inferred = EILPredicate(
                        name=consequent.name,
                        predicate_type=consequent.predicate_type,
                        arguments=consequent.arguments.copy(),
                        confidence=rule.confidence,
                        source="EIL_REASONER",
                        sense_ids=consequent.sense_ids.copy()
                    )
                    inferences.append(inferred)
            
            return inferences
        
        except Exception as e:
            logger.warning(f"Rule application failed for {rule.name}: {e}")
            return []
    
    def _predicates_match(self, pattern: EILPredicate, predicate: EILPredicate) -> bool:
        """Check if a predicate matches a pattern."""
        try:
            # Check name match
            if pattern.name != predicate.name:
                return False
            
            # Check type match
            if pattern.predicate_type != predicate.predicate_type:
                return False
            
            # Check argument count
            if len(pattern.arguments) != len(predicate.arguments):
                return False
            
            return True
        
        except Exception:
            return False


class ExecutableInterlinguaSystem:
    """Comprehensive Executable Interlingua system."""
    
    def __init__(self):
        """Initialize the EIL system."""
        self.compiler = EILCompiler()
        self.reasoner = EILReasoner()
        
        # System parameters
        self.system_params = {
            'enable_nsm_compilation': True,
            'enable_umr_compilation': True,
            'enable_reasoning': True,
            'enable_proof_trees': True
        }
    
    def process_text_to_eil(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Process text through NSM to EIL pipeline."""
        try:
            # Detect NSM primitives
            primitives = self.compiler.nsm_translator.detect_primitives_in_text(text, language)
            
            if primitives:
                # Compile to EIL
                eil_fact = self.compiler.compile_nsm_to_eil(text, primitives, language)
                
                # Reason over EIL facts
                reasoning_result = self.reasoner.reason_over_facts([eil_fact])
                
                return {
                    'original_text': text,
                    'primitives': primitives,
                    'eil_fact': asdict(eil_fact),
                    'reasoning_result': reasoning_result,
                    'processing_success': True
                }
            else:
                return {
                    'original_text': text,
                    'primitives': [],
                    'eil_fact': None,
                    'reasoning_result': None,
                    'processing_success': False,
                    'error': 'No primitives detected'
                }
        
        except Exception as e:
            logger.warning(f"EIL processing failed for '{text}': {e}")
            return {
                'original_text': text,
                'primitives': [],
                'eil_fact': None,
                'reasoning_result': None,
                'processing_success': False,
                'error': str(e)
            }
    
    def run_eil_analysis(self, test_texts: List[str], languages: List[str] = ["en"]) -> Dict[str, Any]:
        """Run comprehensive EIL analysis."""
        logger.info(f"Running EIL analysis for {len(test_texts)} texts")
        
        analysis_results = {
            'test_configuration': {
                'num_test_texts': len(test_texts),
                'languages': languages,
                'timestamp': time.time()
            },
            'processing_results': [],
            'eil_analysis': {},
            'recommendations': []
        }
        
        # Process test texts
        for language in languages:
            for text in test_texts:
                result = self.process_text_to_eil(text, language)
                analysis_results['processing_results'].append(result)
        
        # Analyze results
        analysis_results['eil_analysis'] = self._analyze_eil_results(
            analysis_results['processing_results']
        )
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_eil_recommendations(
            analysis_results['eil_analysis']
        )
        
        return analysis_results
    
    def _analyze_eil_results(self, processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze EIL processing results."""
        analysis = {
            'total_texts': len(processing_results),
            'successful_compilations': 0,
            'successful_reasoning': 0,
            'avg_confidence': 0.0,
            'predicate_distribution': defaultdict(int),
            'reasoning_success_rate': 0.0,
            'proof_tree_stats': {
                'total_proof_trees': 0,
                'avg_proof_tree_size': 0.0,
                'rule_usage': defaultdict(int)
            }
        }
        
        confidences = []
        total_proof_trees = 0
        proof_tree_sizes = []
        
        for result in processing_results:
            if result.get('processing_success', False):
                analysis['successful_compilations'] += 1
                
                # Analyze EIL fact
                eil_fact = result.get('eil_fact')
                if eil_fact:
                    confidence = eil_fact.get('confidence', 0.0)
                    confidences.append(confidence)
                    
                    # Analyze predicates
                    for predicate in eil_fact.get('predicates', []):
                        predicate_type = predicate.get('predicate_type', 'unknown')
                        analysis['predicate_distribution'][predicate_type] += 1
                
                # Analyze reasoning
                reasoning_result = result.get('reasoning_result')
                if reasoning_result and reasoning_result.get('reasoning_success', False):
                    analysis['successful_reasoning'] += 1
                    
                    # Analyze proof trees
                    proof_trees = reasoning_result.get('proof_trees', [])
                    total_proof_trees += len(proof_trees)
                    
                    for proof_tree in proof_trees:
                        proof_tree_sizes.append(len(proof_tree.get('inferences', [])))
                        rule_name = proof_tree.get('rule', 'unknown')
                        analysis['proof_tree_stats']['rule_usage'][rule_name] += 1
        
        # Calculate averages
        if confidences:
            analysis['avg_confidence'] = np.mean(confidences)
        
        if analysis['successful_compilations'] > 0:
            analysis['reasoning_success_rate'] = analysis['successful_reasoning'] / analysis['successful_compilations']
        
        analysis['proof_tree_stats']['total_proof_trees'] = total_proof_trees
        if proof_tree_sizes:
            analysis['proof_tree_stats']['avg_proof_tree_size'] = np.mean(proof_tree_sizes)
        
        return analysis
    
    def _generate_eil_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for EIL system."""
        recommendations = []
        
        # Compilation recommendations
        compilation_rate = analysis['successful_compilations'] / analysis['total_texts']
        if compilation_rate < 0.8:
            recommendations.append(f"Low compilation rate ({compilation_rate:.1%}) - improve NSM primitive detection")
        
        # Reasoning recommendations
        if analysis['reasoning_success_rate'] < 0.5:
            recommendations.append("Low reasoning success rate - expand rule set or improve predicate matching")
        
        # Confidence recommendations
        if analysis['avg_confidence'] < 0.7:
            recommendations.append("Low average confidence - improve predicate mapping and confidence estimation")
        
        # Proof tree recommendations
        if analysis['proof_tree_stats']['total_proof_trees'] < 5:
            recommendations.append("Few proof trees generated - add more applicable reasoning rules")
        
        # Predicate distribution recommendations
        if len(analysis['predicate_distribution']) < 5:
            recommendations.append("Limited predicate diversity - expand NSM to EIL mappings")
        
        return recommendations


def main():
    """Main function to run EIL specification analysis."""
    logger.info("Starting Executable Interlingua specification analysis...")
    
    # Initialize EIL system
    eil_system = ExecutableInterlinguaSystem()
    
    # Test texts
    test_texts = [
        "The cat is not on the mat",
        "I can see the bird",
        "She must go to work",
        "The rain caused the flood",
        "All dogs are animals",
        "I almost finished the work",
        "I like this weather",
        "The book is on the table",
        "This is similar to that",
        "I have a car"
    ]
    
    # Run EIL analysis
    analysis_results = eil_system.run_eil_analysis(test_texts, ["en"])
    
    # Print results
    print("\n" + "="*80)
    print("EXECUTABLE INTERLINGUA SPECIFICATION ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Texts: {analysis_results['test_configuration']['num_test_texts']}")
    print(f"  Languages: {analysis_results['test_configuration']['languages']}")
    
    print(f"\nEIL Analysis:")
    analysis = analysis_results['eil_analysis']
    print(f"  Total Texts: {analysis['total_texts']}")
    print(f"  Successful Compilations: {analysis['successful_compilations']}")
    print(f"  Compilation Rate: {analysis['successful_compilations']/analysis['total_texts']:.1%}")
    print(f"  Successful Reasoning: {analysis['successful_reasoning']}")
    print(f"  Reasoning Success Rate: {analysis['reasoning_success_rate']:.1%}")
    print(f"  Average Confidence: {analysis['avg_confidence']:.3f}")
    
    print(f"\nPredicate Distribution:")
    for predicate_type, count in sorted(analysis['predicate_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {predicate_type}: {count}")
    
    print(f"\nProof Tree Statistics:")
    proof_stats = analysis['proof_tree_stats']
    print(f"  Total Proof Trees: {proof_stats['total_proof_trees']}")
    print(f"  Average Proof Tree Size: {proof_stats['avg_proof_tree_size']:.2f}")
    print(f"  Rule Usage:")
    for rule, count in sorted(proof_stats['rule_usage'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {rule}: {count}")
    
    print(f"\nExample Processing Results:")
    for i, result in enumerate(analysis_results['processing_results'][:3]):
        text = result['original_text']
        primitives = result.get('primitives', [])
        success = result.get('processing_success', False)
        reasoning_success = result.get('reasoning_result', {}).get('reasoning_success', False)
        
        print(f"  {i+1}. Text: {text}")
        print(f"     Primitives: {primitives}")
        print(f"     Compilation Success: {success}")
        print(f"     Reasoning Success: {reasoning_success}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    output_path = "data/executable_interlingua_spec_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(analysis_results), f, ensure_ascii=False, indent=2)
    
    logger.info(f"EIL specification report saved to: {output_path}")
    
    print("="*80)
    print("Executable Interlingua specification analysis completed!")
    print("="*80)


if __name__ == "__main__":
    main()

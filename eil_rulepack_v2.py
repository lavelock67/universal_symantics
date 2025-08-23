#!/usr/bin/env python3
"""
EIL Rulepack v2 System.

This script implements the EIL rulepack v2 with reasoning rules as specified
in ChatGPT5's feedback:
- Negation scope over locatives
- Desire/modality chain
- Aspect/mood molecules
- Experiencer mapping
- Quantifier skeletons with monotonicity guards
- Causation defaults with counterfactual scaffold
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
import re
from dataclasses import dataclass, asdict
from enum import Enum
import time
from collections import defaultdict

# Load environment variables
from dotenv import load_dotenv
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


class RuleType(Enum):
    """Types of EIL reasoning rules."""
    NEGATION_SCOPE = "negation_scope"
    DESIRE_MODALITY_CHAIN = "desire_modality_chain"
    ASPECT_MOOD_MOLECULE = "aspect_mood_molecule"
    EXPERIENCER_MAPPING = "experiencer_mapping"
    QUANTIFIER_SKELETON = "quantifier_skeleton"
    CAUSATION_DEFAULT = "causation_default"


@dataclass
class EILRule:
    """An EIL reasoning rule."""
    rule_id: str
    rule_type: RuleType
    name: str
    description: str
    antecedent: str
    consequent: str
    confidence: float
    conditions: List[str]
    examples: List[Dict[str, str]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'rule_id': self.rule_id,
            'rule_type': self.rule_type.value,
            'name': self.name,
            'description': self.description,
            'antecedent': self.antecedent,
            'consequent': self.consequent,
            'confidence': self.confidence,
            'conditions': self.conditions,
            'examples': self.examples
        }


@dataclass
class EILFact:
    """An EIL fact for reasoning."""
    fact_id: str
    predicate: str
    arguments: List[str]
    confidence: float
    source: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'fact_id': self.fact_id,
            'predicate': self.predicate,
            'arguments': self.arguments,
            'confidence': self.confidence,
            'source': self.source,
            'timestamp': self.timestamp
        }


@dataclass
class ReasoningStep:
    """A single reasoning step."""
    step_id: str
    rule_applied: str
    input_facts: List[str]
    output_facts: List[str]
    confidence: float
    reasoning_chain: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'step_id': self.step_id,
            'rule_applied': self.rule_applied,
            'input_facts': self.input_facts,
            'output_facts': self.output_facts,
            'confidence': self.confidence,
            'reasoning_chain': self.reasoning_chain
        }


class EILRulepackV2:
    """EIL Rulepack v2 with advanced reasoning rules."""
    
    def __init__(self):
        """Initialize the EIL rulepack v2."""
        self.rules = self._initialize_rules()
        self.fact_counter = 0
        self.step_counter = 0
    
    def _initialize_rules(self) -> List[EILRule]:
        """Initialize the EIL reasoning rules."""
        rules = []
        
        # 1. Negation scope over locatives
        rules.append(EILRule(
            rule_id="neg_scope_locative_001",
            rule_type=RuleType.NEGATION_SCOPE,
            name="Negation Scope Over Locatives",
            description="NOT(ON(x,y)) ⇒ NOT(AT_LOC(x,on(y)))",
            antecedent="NOT(ON(x,y))",
            consequent="NOT(AT_LOC(x,on(y)))",
            confidence=0.9,
            conditions=["x is entity", "y is location"],
            examples=[
                {
                    "input": "NOT(ON(cat, mat))",
                    "output": "NOT(AT_LOC(cat, on(mat)))",
                    "explanation": "Negation scopes over the locative relationship"
                }
            ]
        ))
        
        # 2. Desire/modality chain
        rules.append(EILRule(
            rule_id="desire_modality_chain_001",
            rule_type=RuleType.DESIRE_MODALITY_CHAIN,
            name="Desire Modality Chain",
            description="WANT(x,a) ∧ CAN(x,a) ⇒ TRY(x,a)",
            antecedent="WANT(x,a) ∧ CAN(x,a)",
            consequent="TRY(x,a)",
            confidence=0.8,
            conditions=["x is agent", "a is action"],
            examples=[
                {
                    "input": "WANT(person, help) ∧ CAN(person, help)",
                    "output": "TRY(person, help)",
                    "explanation": "Desire plus ability implies attempt"
                }
            ]
        ))
        
        # 3. Aspect/mood molecules
        rules.append(EILRule(
            rule_id="aspect_mood_molecule_001",
            rule_type=RuleType.ASPECT_MOOD_MOLECULE,
            name="Almost Do Molecule",
            description="ALMOST_DO(x,a) ⇒ WANT(x,a) ∧ NOT(DO(x,a))",
            antecedent="ALMOST_DO(x,a)",
            consequent="WANT(x,a) ∧ NOT(DO(x,a))",
            confidence=0.85,
            conditions=["x is agent", "a is action"],
            examples=[
                {
                    "input": "ALMOST_DO(person, leave)",
                    "output": "WANT(person, leave) ∧ NOT(DO(person, leave))",
                    "explanation": "Almost doing implies desire but not execution"
                }
            ]
        ))
        
        rules.append(EILRule(
            rule_id="aspect_mood_molecule_002",
            rule_type=RuleType.ASPECT_MOOD_MOLECULE,
            name="Recent Past Molecule",
            description="RECENT_PAST(DO(x,a)) ⇒ DO(x,a) ∧ BEFORE(DO(x,a), NOW)",
            antecedent="RECENT_PAST(DO(x,a))",
            consequent="DO(x,a) ∧ BEFORE(DO(x,a), NOW)",
            confidence=0.9,
            conditions=["x is agent", "a is action"],
            examples=[
                {
                    "input": "RECENT_PAST(DO(person, arrive))",
                    "output": "DO(person, arrive) ∧ BEFORE(DO(person, arrive), NOW)",
                    "explanation": "Recent past implies action occurred before now"
                }
            ]
        ))
        
        rules.append(EILRule(
            rule_id="aspect_mood_molecule_003",
            rule_type=RuleType.ASPECT_MOOD_MOLECULE,
            name="Ongoing For Molecule",
            description="ONGOING_FOR(DO(x,a), duration) ⇒ DO(x,a) ∧ DURING(DO(x,a), duration)",
            antecedent="ONGOING_FOR(DO(x,a), duration)",
            consequent="DO(x,a) ∧ DURING(DO(x,a), duration)",
            confidence=0.85,
            conditions=["x is agent", "a is action", "duration is time_period"],
            examples=[
                {
                    "input": "ONGOING_FOR(DO(person, work), hours(2))",
                    "output": "DO(person, work) ∧ DURING(DO(person, work), hours(2))",
                    "explanation": "Ongoing for implies action during time period"
                }
            ]
        ))
        
        # 4. Experiencer mapping
        rules.append(EILRule(
            rule_id="experiencer_mapping_001",
            rule_type=RuleType.EXPERIENCER_MAPPING,
            name="Spanish Gustar Mapping",
            description="gustar(stimulus, experiencer) ⇒ LIKE(experiencer, stimulus)",
            antecedent="gustar(stimulus, experiencer)",
            consequent="LIKE(experiencer, stimulus)",
            confidence=0.95,
            conditions=["stimulus is object", "experiencer is person"],
            examples=[
                {
                    "input": "gustar(music, person)",
                    "output": "LIKE(person, music)",
                    "explanation": "Spanish gustar maps to NSM LIKE with reversed arguments"
                }
            ]
        ))
        
        rules.append(EILRule(
            rule_id="experiencer_mapping_002",
            rule_type=RuleType.EXPERIENCER_MAPPING,
            name="French Plaire Mapping",
            description="plaire(stimulus, experiencer) ⇒ LIKE(experiencer, stimulus)",
            antecedent="plaire(stimulus, experiencer)",
            consequent="LIKE(experiencer, stimulus)",
            confidence=0.95,
            conditions=["stimulus is object", "experiencer is person"],
            examples=[
                {
                    "input": "plaire(movie, person)",
                    "output": "LIKE(person, movie)",
                    "explanation": "French plaire maps to NSM LIKE with reversed arguments"
                }
            ]
        ))
        
        # 5. Quantifier skeletons with monotonicity guards
        rules.append(EILRule(
            rule_id="quantifier_skeleton_001",
            rule_type=RuleType.QUANTIFIER_SKELETON,
            name="Universal Quantifier Skeleton",
            description="ALL(x, P(x)) ∧ Q(x) ⇒ ALL(x, P(x) ∧ Q(x))",
            antecedent="ALL(x, P(x)) ∧ Q(x)",
            consequent="ALL(x, P(x) ∧ Q(x))",
            confidence=0.9,
            conditions=["P is predicate", "Q is predicate", "x is variable"],
            examples=[
                {
                    "input": "ALL(child, CAN(child, play)) ∧ HERE",
                    "output": "ALL(child, CAN(child, play) ∧ HERE)",
                    "explanation": "Universal quantifier distributes over conjunction"
                }
            ]
        ))
        
        rules.append(EILRule(
            rule_id="quantifier_skeleton_002",
            rule_type=RuleType.QUANTIFIER_SKELETON,
            name="Existential Quantifier Skeleton",
            description="SOME(x, P(x)) ∧ Q(x) ⇒ SOME(x, P(x) ∧ Q(x))",
            antecedent="SOME(x, P(x)) ∧ Q(x)",
            consequent="SOME(x, P(x) ∧ Q(x))",
            confidence=0.85,
            conditions=["P is predicate", "Q is predicate", "x is variable"],
            examples=[
                {
                    "input": "SOME(person, WANT(person, help)) ∧ CAN(person, help)",
                    "output": "SOME(person, WANT(person, help) ∧ CAN(person, help))",
                    "explanation": "Existential quantifier distributes over conjunction"
                }
            ]
        ))
        
        # 6. Causation defaults with counterfactual scaffold
        rules.append(EILRule(
            rule_id="causation_default_001",
            rule_type=RuleType.CAUSATION_DEFAULT,
            name="Causation Temporal Default",
            description="CAUSE(e1,e2) ⇒ BEFORE(e1,e2)",
            antecedent="CAUSE(e1,e2)",
            consequent="BEFORE(e1,e2)",
            confidence=0.9,
            conditions=["e1 is event", "e2 is event"],
            examples=[
                {
                    "input": "CAUSE(rain, wet_ground)",
                    "output": "BEFORE(rain, wet_ground)",
                    "explanation": "Causation implies temporal precedence"
                }
            ]
        ))
        
        rules.append(EILRule(
            rule_id="causation_default_002",
            rule_type=RuleType.CAUSATION_DEFAULT,
            name="Counterfactual Causation",
            description="CAUSE(e1,e2) ∧ NOT(e1) ⇒ NOT(e2)",
            antecedent="CAUSE(e1,e2) ∧ NOT(e1)",
            consequent="NOT(e2)",
            confidence=0.8,
            conditions=["e1 is event", "e2 is event"],
            examples=[
                {
                    "input": "CAUSE(rain, wet_ground) ∧ NOT(rain)",
                    "output": "NOT(wet_ground)",
                    "explanation": "Counterfactual: no cause implies no effect"
                }
            ]
        ))
        
        return rules
    
    def apply_rule(self, rule: EILRule, facts: List[EILFact]) -> List[EILFact]:
        """Apply a rule to generate new facts."""
        new_facts = []
        
        # Simple pattern matching for rule application
        if rule.rule_type == RuleType.NEGATION_SCOPE:
            new_facts.extend(self._apply_negation_scope(rule, facts))
        elif rule.rule_type == RuleType.DESIRE_MODALITY_CHAIN:
            new_facts.extend(self._apply_desire_modality_chain(rule, facts))
        elif rule.rule_type == RuleType.ASPECT_MOOD_MOLECULE:
            new_facts.extend(self._apply_aspect_mood_molecule(rule, facts))
        elif rule.rule_type == RuleType.EXPERIENCER_MAPPING:
            new_facts.extend(self._apply_experiencer_mapping(rule, facts))
        elif rule.rule_type == RuleType.QUANTIFIER_SKELETON:
            new_facts.extend(self._apply_quantifier_skeleton(rule, facts))
        elif rule.rule_type == RuleType.CAUSATION_DEFAULT:
            new_facts.extend(self._apply_causation_default(rule, facts))
        
        return new_facts
    
    def _apply_negation_scope(self, rule: EILRule, facts: List[EILFact]) -> List[EILFact]:
        """Apply negation scope over locatives rule."""
        new_facts = []
        
        for fact in facts:
            if fact.predicate == "NOT" and len(fact.arguments) >= 1:
                # Check if argument contains locative
                arg = fact.arguments[0]
                if "ON(" in arg:
                    # Extract entities
                    match = re.search(r'ON\(([^,]+),([^)]+)\)', arg)
                    if match:
                        x, y = match.groups()
                        new_fact = EILFact(
                            fact_id=f"fact_{self.fact_counter}",
                            predicate="NOT",
                            arguments=[f"AT_LOC({x.strip()}, on({y.strip()}))"],
                            confidence=rule.confidence * fact.confidence,
                            source=f"rule_{rule.rule_id}",
                            timestamp=time.time()
                        )
                        new_facts.append(new_fact)
                        self.fact_counter += 1
        
        return new_facts
    
    def _apply_desire_modality_chain(self, rule: EILRule, facts: List[EILFact]) -> List[EILFact]:
        """Apply desire/modality chain rule."""
        new_facts = []
        
        # Find WANT and CAN facts with same agent and action
        want_facts = [f for f in facts if f.predicate == "WANT"]
        can_facts = [f for f in facts if f.predicate == "CAN"]
        
        for want_fact in want_facts:
            for can_fact in can_facts:
                if (len(want_fact.arguments) >= 2 and len(can_fact.arguments) >= 2 and
                    want_fact.arguments[0] == can_fact.arguments[0] and  # same agent
                    want_fact.arguments[1] == can_fact.arguments[1]):    # same action
                    
                    new_fact = EILFact(
                        fact_id=f"fact_{self.fact_counter}",
                        predicate="TRY",
                        arguments=[want_fact.arguments[0], want_fact.arguments[1]],
                        confidence=rule.confidence * min(want_fact.confidence, can_fact.confidence),
                        source=f"rule_{rule.rule_id}",
                        timestamp=time.time()
                    )
                    new_facts.append(new_fact)
                    self.fact_counter += 1
        
        return new_facts
    
    def _apply_aspect_mood_molecule(self, rule: EILRule, facts: List[EILFact]) -> List[EILFact]:
        """Apply aspect/mood molecule rules."""
        new_facts = []
        
        for fact in facts:
            if "ALMOST_DO" in fact.predicate:
                # ALMOST_DO(x,a) ⇒ WANT(x,a) ∧ NOT(DO(x,a))
                if len(fact.arguments) >= 2:
                    x, a = fact.arguments[0], fact.arguments[1]
                    
                    # Create WANT fact
                    want_fact = EILFact(
                        fact_id=f"fact_{self.fact_counter}",
                        predicate="WANT",
                        arguments=[x, a],
                        confidence=rule.confidence * fact.confidence,
                        source=f"rule_{rule.rule_id}",
                        timestamp=time.time()
                    )
                    new_facts.append(want_fact)
                    self.fact_counter += 1
                    
                    # Create NOT DO fact
                    not_do_fact = EILFact(
                        fact_id=f"fact_{self.fact_counter}",
                        predicate="NOT",
                        arguments=[f"DO({x}, {a})"],
                        confidence=rule.confidence * fact.confidence,
                        source=f"rule_{rule.rule_id}",
                        timestamp=time.time()
                    )
                    new_facts.append(not_do_fact)
                    self.fact_counter += 1
        
        return new_facts
    
    def _apply_experiencer_mapping(self, rule: EILRule, facts: List[EILFact]) -> List[EILFact]:
        """Apply experiencer mapping rules."""
        new_facts = []
        
        for fact in facts:
            if fact.predicate in ["gustar", "plaire"] and len(fact.arguments) >= 2:
                stimulus, experiencer = fact.arguments[0], fact.arguments[1]
                
                new_fact = EILFact(
                    fact_id=f"fact_{self.fact_counter}",
                    predicate="LIKE",
                    arguments=[experiencer, stimulus],  # Reversed arguments
                    confidence=rule.confidence * fact.confidence,
                    source=f"rule_{rule.rule_id}",
                    timestamp=time.time()
                )
                new_facts.append(new_fact)
                self.fact_counter += 1
        
        return new_facts
    
    def _apply_quantifier_skeleton(self, rule: EILRule, facts: List[EILFact]) -> List[EILFact]:
        """Apply quantifier skeleton rules."""
        new_facts = []
        
        for fact in facts:
            if fact.predicate in ["ALL", "SOME"] and len(fact.arguments) >= 2:
                quantifier, predicate_expr = fact.arguments[0], fact.arguments[1]
                
                # Look for additional predicates to combine (but avoid self-reference)
                for other_fact in facts:
                    if (other_fact.fact_id != fact.fact_id and 
                        other_fact.predicate not in ["ALL", "SOME"] and  # Avoid combining with other quantifiers
                        len(other_fact.arguments) > 0):
                        
                        # Combine predicates
                        other_pred = f"{other_fact.predicate}({other_fact.arguments[0]})"
                        combined_predicate = f"{predicate_expr} ∧ {other_pred}"
                        
                        new_fact = EILFact(
                            fact_id=f"fact_{self.fact_counter}",
                            predicate=fact.predicate,
                            arguments=[quantifier, combined_predicate],
                            confidence=rule.confidence * fact.confidence * other_fact.confidence,
                            source=f"rule_{rule.rule_id}",
                            timestamp=time.time()
                        )
                        new_facts.append(new_fact)
                        self.fact_counter += 1
                        
                        # Limit combinations to prevent explosion
                        if len(new_facts) >= 5:
                            break
        
        return new_facts
    
    def _apply_causation_default(self, rule: EILRule, facts: List[EILFact]) -> List[EILFact]:
        """Apply causation default rules."""
        new_facts = []
        
        for fact in facts:
            if fact.predicate == "CAUSE" and len(fact.arguments) >= 2:
                e1, e2 = fact.arguments[0], fact.arguments[1]
                
                # CAUSE(e1,e2) ⇒ BEFORE(e1,e2)
                before_fact = EILFact(
                    fact_id=f"fact_{self.fact_counter}",
                    predicate="BEFORE",
                    arguments=[e1, e2],
                    confidence=rule.confidence * fact.confidence,
                    source=f"rule_{rule.rule_id}",
                    timestamp=time.time()
                )
                new_facts.append(before_fact)
                self.fact_counter += 1
        
        return new_facts
    
    def reason_with_facts(self, initial_facts: List[EILFact], max_steps: int = 10) -> Dict[str, Any]:
        """Apply reasoning rules to generate new facts."""
        logger.info(f"Starting reasoning with {len(initial_facts)} initial facts")
        
        all_facts = initial_facts.copy()
        reasoning_steps = []
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            new_facts_in_step = []
            
            # Track facts before this step to avoid infinite loops
            facts_before_step = len(all_facts)
            
            # Apply each rule
            for rule in self.rules:
                new_facts = self.apply_rule(rule, all_facts)
                if new_facts:
                    new_facts_in_step.extend(new_facts)
                    all_facts.extend(new_facts)
                    
                    # Create reasoning step
                    step = ReasoningStep(
                        step_id=f"step_{self.step_counter}",
                        rule_applied=rule.rule_id,
                        input_facts=[f.fact_id for f in all_facts if f not in new_facts],
                        output_facts=[f.fact_id for f in new_facts],
                        confidence=np.mean([f.confidence for f in new_facts]) if new_facts else 0,
                        reasoning_chain=f"Applied {rule.name}: {rule.antecedent} → {rule.consequent}"
                    )
                    reasoning_steps.append(step)
                    self.step_counter += 1
            
            # Stop if no new facts generated in this step
            if len(all_facts) == facts_before_step:
                logger.info(f"No new facts generated in step {step_count}, stopping")
                break
            
            # Safety check: stop if too many facts
            if len(all_facts) > 100:
                logger.warning(f"Too many facts generated ({len(all_facts)}), stopping to prevent explosion")
                break
        
        return {
            'initial_facts': [f.to_dict() for f in initial_facts],
            'derived_facts': [f.to_dict() for f in all_facts if f not in initial_facts],
            'all_facts': [f.to_dict() for f in all_facts],
            'reasoning_steps': [s.to_dict() for s in reasoning_steps],
            'total_steps': step_count,
            'total_facts': len(all_facts),
            'derived_facts_count': len([f for f in all_facts if f not in initial_facts])
        }


class EILRulepackV2System:
    """Comprehensive EIL rulepack v2 system."""
    
    def __init__(self):
        """Initialize the EIL rulepack v2 system."""
        self.rulepack = EILRulepackV2()
    
    def run_rulepack_analysis(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive EIL rulepack v2 analysis."""
        logger.info(f"Running EIL rulepack v2 analysis on {len(test_cases)} test cases")
        
        analysis_results = {
            'test_configuration': {
                'num_test_cases': len(test_cases),
                'num_rules': len(self.rulepack.rules),
                'timestamp': time.time()
            },
            'rules': [rule.to_dict() for rule in self.rulepack.rules],
            'reasoning_results': [],
            'rule_analysis': {},
            'recommendations': []
        }
        
        # Run reasoning on test cases
        for i, test_case in enumerate(test_cases):
            initial_facts = self._create_facts_from_test_case(test_case)
            reasoning_result = self.rulepack.reason_with_facts(initial_facts)
            reasoning_result['test_case'] = test_case
            analysis_results['reasoning_results'].append(reasoning_result)
        
        # Analyze results
        analysis_results['rule_analysis'] = self._analyze_rule_usage(analysis_results['reasoning_results'])
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_rulepack_recommendations(
            analysis_results['rule_analysis']
        )
        
        return analysis_results
    
    def _create_facts_from_test_case(self, test_case: Dict[str, Any]) -> List[EILFact]:
        """Create EIL facts from test case."""
        facts = []
        
        # Parse test case into facts
        if 'facts' in test_case:
            for fact_data in test_case['facts']:
                fact = EILFact(
                    fact_id=fact_data.get('fact_id', f"test_fact_{len(facts)}"),
                    predicate=fact_data['predicate'],
                    arguments=fact_data.get('arguments', []),
                    confidence=fact_data.get('confidence', 0.8),
                    source="test_case",
                    timestamp=time.time()
                )
                facts.append(fact)
        
        return facts
    
    def _analyze_rule_usage(self, reasoning_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze rule usage across reasoning results."""
        analysis = {
            'total_reasoning_steps': 0,
            'rule_usage_count': defaultdict(int),
            'rule_type_usage': defaultdict(int),
            'average_facts_per_step': 0,
            'reasoning_depth': defaultdict(int),
            'confidence_distribution': defaultdict(int)
        }
        
        total_steps = 0
        total_facts = 0
        
        for result in reasoning_results:
            steps = result.get('reasoning_steps', [])
            total_steps += len(steps)
            
            for step in steps:
                rule_id = step.get('rule_applied', 'unknown')
                analysis['rule_usage_count'][rule_id] += 1
                
                # Find rule type
                for rule in self.rulepack.rules:
                    if rule.rule_id == rule_id:
                        analysis['rule_type_usage'][rule.rule_type.value] += 1
                        break
                
                # Count facts
                output_facts = step.get('output_facts', [])
                total_facts += len(output_facts)
                
                # Confidence distribution
                confidence = step.get('confidence', 0)
                if confidence >= 0.9:
                    analysis['confidence_distribution']['high'] += 1
                elif confidence >= 0.7:
                    analysis['confidence_distribution']['medium'] += 1
                else:
                    analysis['confidence_distribution']['low'] += 1
            
            # Reasoning depth
            depth = result.get('total_steps', 0)
            analysis['reasoning_depth'][depth] += 1
        
        analysis['total_reasoning_steps'] = total_steps
        analysis['average_facts_per_step'] = total_facts / total_steps if total_steps > 0 else 0
        
        return analysis
    
    def _generate_rulepack_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations from rulepack analysis."""
        recommendations = []
        
        # Rule usage recommendations
        if analysis['total_reasoning_steps'] > 0:
            recommendations.append(f"Applied {analysis['total_reasoning_steps']} reasoning steps successfully")
        
        # Most used rule
        if analysis['rule_usage_count']:
            most_used_rule = max(analysis['rule_usage_count'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Most active rule: {most_used_rule}")
        
        # Rule type recommendations
        if analysis['rule_type_usage']:
            most_used_type = max(analysis['rule_type_usage'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Most active rule type: {most_used_type}")
        
        # Confidence recommendations
        if analysis['confidence_distribution']['low'] > 0:
            recommendations.append(f"Improve confidence for {analysis['confidence_distribution']['low']} low-confidence steps")
        
        # Reasoning depth recommendations
        if analysis['reasoning_depth']:
            max_depth = max(analysis['reasoning_depth'].keys())
            recommendations.append(f"Maximum reasoning depth achieved: {max_depth} steps")
        
        return recommendations


def main():
    """Main function to demonstrate EIL rulepack v2."""
    logger.info("Starting EIL rulepack v2 demonstration...")
    
    # Initialize system
    system = EILRulepackV2System()
    
    # Test cases
    test_cases = [
        {
            "name": "Negation Scope Test",
            "description": "Test negation scope over locatives",
            "facts": [
                {"predicate": "NOT", "arguments": ["ON(cat, mat)"], "confidence": 0.9}
            ]
        },
        {
            "name": "Desire Modality Chain Test",
            "description": "Test desire/modality chain reasoning",
            "facts": [
                {"predicate": "WANT", "arguments": ["person", "help"], "confidence": 0.8},
                {"predicate": "CAN", "arguments": ["person", "help"], "confidence": 0.9}
            ]
        },
        {
            "name": "Aspect Mood Molecule Test",
            "description": "Test aspect/mood molecule reasoning",
            "facts": [
                {"predicate": "ALMOST_DO", "arguments": ["person", "leave"], "confidence": 0.85}
            ]
        },
        {
            "name": "Experiencer Mapping Test",
            "description": "Test experiencer mapping",
            "facts": [
                {"predicate": "gustar", "arguments": ["music", "person"], "confidence": 0.95}
            ]
        },
        {
            "name": "Quantifier Skeleton Test",
            "description": "Test quantifier skeleton reasoning",
            "facts": [
                {"predicate": "ALL", "arguments": ["child", "CAN(child, play)"], "confidence": 0.9},
                {"predicate": "HERE", "arguments": [], "confidence": 0.8}
            ]
        },
        {
            "name": "Causation Default Test",
            "description": "Test causation default reasoning",
            "facts": [
                {"predicate": "CAUSE", "arguments": ["rain", "wet_ground"], "confidence": 0.9}
            ]
        }
    ]
    
    # Run analysis
    analysis_results = system.run_rulepack_analysis(test_cases)
    
    # Print results
    print("\n" + "="*80)
    print("EIL RULEPACK V2 RESULTS")
    print("="*80)
    
    print(f"Test Configuration:")
    print(f"  Number of Test Cases: {analysis_results['test_configuration']['num_test_cases']}")
    print(f"  Number of Rules: {analysis_results['test_configuration']['num_rules']}")
    
    print(f"\nRule Analysis:")
    analysis = analysis_results['rule_analysis']
    print(f"  Total Reasoning Steps: {analysis['total_reasoning_steps']}")
    print(f"  Average Facts per Step: {analysis['average_facts_per_step']:.2f}")
    
    print(f"\nRule Usage Count:")
    for rule_id, count in analysis['rule_usage_count'].items():
        print(f"  {rule_id}: {count}")
    
    print(f"\nRule Type Usage:")
    for rule_type, count in analysis['rule_type_usage'].items():
        print(f"  {rule_type}: {count}")
    
    print(f"\nConfidence Distribution:")
    for level, count in analysis['confidence_distribution'].items():
        print(f"  {level}: {count}")
    
    print(f"\nReasoning Depth Distribution:")
    for depth, count in analysis['reasoning_depth'].items():
        print(f"  {depth} steps: {count} cases")
    
    print(f"\nSample Reasoning Results:")
    for i, result in enumerate(analysis_results['reasoning_results'][:3]):
        print(f"  {i+1}. {result['test_case']['name']}")
        print(f"     Initial Facts: {len(result['initial_facts'])}")
        print(f"     Derived Facts: {result['derived_facts_count']}")
        print(f"     Reasoning Steps: {len(result['reasoning_steps'])}")
        if result['reasoning_steps']:
            print(f"     First Step: {result['reasoning_steps'][0]['reasoning_chain']}")
        print()
    
    print(f"\nRecommendations:")
    for i, recommendation in enumerate(analysis_results['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    # Save results
    output_path = Path("data/eil_rulepack_v2_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(convert_numpy_types(analysis_results), f, indent=2)
    
    logger.info(f"EIL rulepack v2 results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("EIL rulepack v2 demonstration completed!")
    print("="*80)


if __name__ == "__main__":
    main()

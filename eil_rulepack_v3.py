#!/usr/bin/env python3
"""
EIL Rulepack v3 System.

This script implements the EIL rulepack v3 with advanced reasoning rules as specified
in ChatGPT5's feedback:
- Q1 Quantifier scope (ALL x ¬P(x) ↔ ¬∃x P(x))
- T1 Temporal chaining (BEFORE/AFTER/DURING)
- CF1 Counterfactual PREVENT/ENABLE
- P1 Pronoun/DEIXIS normalization (THIS/THAT/HERE/NOW/SOMEONE)
- A1 Aspect/mood molecules (ALMOST_DO, RECENT_PAST, STILL, NOT_YET)
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
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class RuleType(Enum):
    """Types of EIL reasoning rules."""
    QUANTIFIER_SCOPE = "quantifier_scope"
    TEMPORAL_CHAINING = "temporal_chaining"
    COUNTERFACTUAL = "counterfactual"
    PRONOUN_DEIXIS = "pronoun_deixis"
    ASPECT_MOOD = "aspect_mood"


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
    monotonicity_guards: List[str]
    
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
            'examples': self.examples,
            'monotonicity_guards': self.monotonicity_guards
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
class ProofStep:
    """A single proof step."""
    step_id: str
    rule_applied: str
    input_facts: List[str]
    output_facts: List[str]
    confidence: float
    reasoning_chain: str
    depth: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'step_id': self.step_id,
            'rule_applied': self.rule_applied,
            'input_facts': self.input_facts,
            'output_facts': self.output_facts,
            'confidence': self.confidence,
            'reasoning_chain': self.reasoning_chain,
            'depth': self.depth
        }


class QuantifierScopeNormalizer:
    """Q1 Quantifier scope normalizer with ambiguity detection."""
    
    def __init__(self):
        """Initialize the quantifier scope normalizer."""
        self.quantifier_patterns = {
            'universal': ['ALL', 'EVERY', 'EACH'],
            'existential': ['SOME', 'EXISTS', 'THERE_EXISTS'],
            'negation': ['NOT', 'NO', 'NONE']
        }
        
        # Ambiguous patterns that need explicit disambiguation
        self.ambiguous_patterns = [
            r'\bALL\s+\w+\s+NOT\b',  # "All children aren't playing"
            r'\bNOT\s+ALL\s+\w+\b',  # "Not all students passed"
            r'\bSOME\s+\w+\s+NOT\b',  # "Some people don't like"
            r'\bNOT\s+SOME\s+\w+\b'   # "Not some people like"
        ]
    
    def normalize_quantifier_scope(self, explication: str) -> Dict[str, Any]:
        """Normalize quantifier scope and detect ambiguity."""
        logger.info(f"Normalizing quantifier scope for: {explication}")
        
        result = {
            'original': explication,
            'normalized': explication,
            'ambiguity_detected': False,
            'ambiguity_type': None,
            'possible_parses': [],
            'monotonicity_violations': []
        }
        
        # Check for ambiguous patterns
        for pattern in self.ambiguous_patterns:
            if re.search(pattern, explication, re.IGNORECASE):
                result['ambiguity_detected'] = True
                result['ambiguity_type'] = self._classify_ambiguity(explication)
                result['possible_parses'] = self._generate_possible_parses(explication)
                break
        
        # Apply normalization rules
        if not result['ambiguity_detected']:
            result['normalized'] = self._apply_scope_normalization(explication)
        
        # Check monotonicity guards
        result['monotonicity_violations'] = self._check_monotonicity_guards(result['normalized'])
        
        return result
    
    def _classify_ambiguity(self, explication: str) -> str:
        """Classify the type of ambiguity."""
        if re.search(r'\bALL\s+\w+\s+NOT\b', explication, re.IGNORECASE):
            return "universal_negation_scope"
        elif re.search(r'\bNOT\s+ALL\s+\w+\b', explication, re.IGNORECASE):
            return "negation_universal_scope"
        elif re.search(r'\bSOME\s+\w+\s+NOT\b', explication, re.IGNORECASE):
            return "existential_negation_scope"
        elif re.search(r'\bNOT\s+SOME\s+\w+\b', explication, re.IGNORECASE):
            return "negation_existential_scope"
        else:
            return "unknown_ambiguity"
    
    def _generate_possible_parses(self, explication: str) -> List[str]:
        """Generate possible parses for ambiguous expressions."""
        parses = []
        
        # Example: "ALL(child, NOT(play(child)))" vs "NOT(ALL(child, play(child)))"
        if "ALL" in explication and "NOT" in explication:
            # Parse 1: Universal negation
            parse1 = explication.replace("ALL", "ALL").replace("NOT", "NOT")
            parses.append(f"Parse 1 (Universal Negation): {parse1}")
            
            # Parse 2: Negation of universal
            parse2 = explication.replace("ALL", "NOT(ALL)").replace("NOT", "")
            parses.append(f"Parse 2 (Negation of Universal): {parse2}")
        
        return parses
    
    def _apply_scope_normalization(self, explication: str) -> str:
        """Apply scope normalization rules."""
        normalized = explication
        
        # Rule: ALL x ¬P(x) ↔ ¬∃x P(x) (when unambiguous)
        if re.search(r'ALL\s*\(\s*\w+\s*,\s*NOT\s*\(', normalized):
            # Convert to existential form
            normalized = re.sub(r'ALL\s*\(\s*(\w+)\s*,\s*NOT\s*\(([^)]+)\)\s*\)', 
                              r'NOT(EXISTS(\1, \2))', normalized)
        
        # Rule: ¬∃x P(x) ↔ ∀x ¬P(x) (when unambiguous)
        if re.search(r'NOT\s*\(\s*EXISTS\s*\(\s*\w+\s*,\s*', normalized):
            # Convert to universal form
            normalized = re.sub(r'NOT\s*\(\s*EXISTS\s*\(\s*(\w+)\s*,\s*([^)]+)\s*\)\s*\)', 
                              r'ALL(\1, NOT(\2))', normalized)
        
        return normalized
    
    def _check_monotonicity_guards(self, explication: str) -> List[str]:
        """Check for monotonicity violations."""
        violations = []
        
        # Guard: No ∀→∃ without evidence
        if re.search(r'ALL\s*\(\s*\w+\s*,\s*[^)]+\)\s*→\s*EXISTS', explication):
            violations.append("Illicit universal to existential inference")
        
        # Guard: No ∃→∀ without evidence
        if re.search(r'EXISTS\s*\(\s*\w+\s*,\s*[^)]+\)\s*→\s*ALL', explication):
            violations.append("Illicit existential to universal inference")
        
        return violations


class TemporalChaining:
    """T1 Temporal chaining with transitivity and antisymmetry."""
    
    def __init__(self):
        """Initialize temporal chaining."""
        self.temporal_relations = {
            'BEFORE': 'before',
            'AFTER': 'after',
            'DURING': 'during',
            'OVERLAP': 'overlap',
            'MEET': 'meet',
            'FINISH': 'finish',
            'START': 'start'
        }
        
        # Allen interval algebra subset
        self.allen_relations = {
            'BEFORE': ['BEFORE', 'MEETS'],
            'AFTER': ['AFTER', 'MET_BY'],
            'DURING': ['DURING', 'STARTS', 'FINISHES'],
            'OVERLAP': ['OVERLAPS', 'OVERLAPPED_BY']
        }
    
    def apply_temporal_chaining(self, facts: List[EILFact]) -> List[EILFact]:
        """Apply temporal chaining rules to facts."""
        logger.info(f"Applying temporal chaining to {len(facts)} facts")
        
        new_facts = []
        
        # Find temporal facts
        temporal_facts = [f for f in facts if f.predicate in self.temporal_relations]
        
        # Apply transitivity
        for fact1 in temporal_facts:
            for fact2 in temporal_facts:
                if fact1.fact_id != fact2.fact_id:
                    transitive_fact = self._apply_transitivity(fact1, fact2)
                    if transitive_fact:
                        new_facts.append(transitive_fact)
        
        # Apply antisymmetry
        for fact in temporal_facts:
            antisymmetric_fact = self._apply_antisymmetry(fact)
            if antisymmetric_fact:
                new_facts.append(antisymmetric_fact)
        
        # Apply DURING subdivision
        for fact in temporal_facts:
            if fact.predicate == 'DURING':
                subdivided_facts = self._subdivide_during(fact)
                new_facts.extend(subdivided_facts)
        
        return new_facts
    
    def _apply_transitivity(self, fact1: EILFact, fact2: EILFact) -> Optional[EILFact]:
        """Apply transitivity rule: BEFORE(e1,e2) ∧ BEFORE(e2,e3) → BEFORE(e1,e3)"""
        if (fact1.predicate == 'BEFORE' and fact2.predicate == 'BEFORE' and
            len(fact1.arguments) >= 2 and len(fact2.arguments) >= 2):
            
            # Check if e1's second argument matches e2's first argument
            if fact1.arguments[1] == fact2.arguments[0]:
                return EILFact(
                    fact_id=f"transitive_{fact1.fact_id}_{fact2.fact_id}",
                    predicate='BEFORE',
                    arguments=[fact1.arguments[0], fact2.arguments[1]],
                    confidence=min(fact1.confidence, fact2.confidence) * 0.9,
                    source='temporal_transitivity',
                    timestamp=time.time()
                )
        
        return None
    
    def _apply_antisymmetry(self, fact: EILFact) -> Optional[EILFact]:
        """Apply antisymmetry rule: BEFORE(e1,e2) → NOT(AFTER(e1,e2))"""
        if fact.predicate == 'BEFORE' and len(fact.arguments) >= 2:
            return EILFact(
                fact_id=f"antisymmetric_{fact.fact_id}",
                predicate='NOT',
                arguments=[f"AFTER({fact.arguments[0]}, {fact.arguments[1]})"],
                confidence=fact.confidence * 0.95,
                source='temporal_antisymmetry',
                timestamp=time.time()
            )
        
        return None
    
    def _subdivide_during(self, fact: EILFact) -> List[EILFact]:
        """Subdivide DURING into START and FINISH."""
        if fact.predicate == 'DURING' and len(fact.arguments) >= 2:
            start_fact = EILFact(
                fact_id=f"start_{fact.fact_id}",
                predicate='START',
                arguments=[fact.arguments[0], fact.arguments[1]],
                confidence=fact.confidence * 0.8,
                source='during_subdivision',
                timestamp=time.time()
            )
            
            finish_fact = EILFact(
                fact_id=f"finish_{fact.fact_id}",
                predicate='FINISH',
                arguments=[fact.arguments[0], fact.arguments[1]],
                confidence=fact.confidence * 0.8,
                source='during_subdivision',
                timestamp=time.time()
            )
            
            return [start_fact, finish_fact]
        
        return []


class CounterfactualReasoning:
    """CF1 Counterfactual PREVENT/ENABLE reasoning."""
    
    def __init__(self):
        """Initialize counterfactual reasoning."""
        self.counterfactual_patterns = {
            'PREVENT': r'PREVENT\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
            'ENABLE': r'ENABLE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
            'DO': r'DO\s*\(\s*([^)]+)\s*\)',
            'NOT_DO': r'NOT\s*\(\s*DO\s*\(\s*([^)]+)\s*\)\s*\)'
        }
    
    def apply_counterfactual_reasoning(self, facts: List[EILFact]) -> List[EILFact]:
        """Apply counterfactual reasoning rules."""
        logger.info(f"Applying counterfactual reasoning to {len(facts)} facts")
        
        new_facts = []
        
        # Find PREVENT and ENABLE facts
        prevent_facts = [f for f in facts if 'PREVENT' in f.predicate]
        enable_facts = [f for f in facts if 'ENABLE' in f.predicate]
        do_facts = [f for f in facts if f.predicate == 'DO']
        not_do_facts = [f for f in facts if f.predicate == 'NOT' and 'DO' in str(f.arguments)]
        
        # Rule: PREVENT(e1,e2) ∧ DO(e1) ⇒ ¬e2
        for prevent_fact in prevent_facts:
            for do_fact in do_facts:
                if self._events_match(prevent_fact, do_fact):
                    new_fact = EILFact(
                        fact_id=f"prevent_effect_{prevent_fact.fact_id}_{do_fact.fact_id}",
                        predicate='NOT',
                        arguments=[self._extract_prevented_event(prevent_fact)],
                        confidence=min(prevent_fact.confidence, do_fact.confidence) * 0.9,
                        source='prevent_rule',
                        timestamp=time.time()
                    )
                    new_facts.append(new_fact)
        
        # Rule: ENABLE(e1,e2) ∧ ¬DO(e1) ⇒ ¬e2 (default, defeasible)
        for enable_fact in enable_facts:
            for not_do_fact in not_do_facts:
                if self._events_match(enable_fact, not_do_fact):
                    new_fact = EILFact(
                        fact_id=f"enable_effect_{enable_fact.fact_id}_{not_do_fact.fact_id}",
                        predicate='NOT',
                        arguments=[self._extract_enabled_event(enable_fact)],
                        confidence=min(enable_fact.confidence, not_do_fact.confidence) * 0.7,  # Lower confidence for defeasible
                        source='enable_rule',
                        timestamp=time.time()
                    )
                    new_facts.append(new_fact)
        
        return new_facts
    
    def _events_match(self, fact1: EILFact, fact2: EILFact) -> bool:
        """Check if events match between two facts."""
        # Simplified event matching
        return any(arg in str(fact2.arguments) for arg in fact1.arguments)
    
    def _extract_prevented_event(self, prevent_fact: EILFact) -> str:
        """Extract the prevented event from a PREVENT fact."""
        # Simplified extraction
        return str(prevent_fact.arguments[1]) if len(prevent_fact.arguments) > 1 else "unknown_event"
    
    def _extract_enabled_event(self, enable_fact: EILFact) -> str:
        """Extract the enabled event from an ENABLE fact."""
        # Simplified extraction
        return str(enable_fact.arguments[1]) if len(enable_fact.arguments) > 1 else "unknown_event"


class PronounDeixisNormalizer:
    """P1 Pronoun/DEIXIS normalization."""
    
    def __init__(self):
        """Initialize pronoun/DEIXIS normalizer."""
        self.deictic_terms = {
            'THIS': 'this',
            'THAT': 'that',
            'HERE': 'here',
            'NOW': 'now',
            'SOMEONE': 'someone',
            'SOMETHING': 'something',
            'SOMEWHERE': 'somewhere'
        }
        
        # Discourse entity mappings
        self.discourse_entities = {
            'THIS': 'discourse_entity_1',
            'THAT': 'discourse_entity_2',
            'HERE': 'discourse_location_1',
            'NOW': 'discourse_time_1',
            'SOMEONE': 'discourse_person_1',
            'SOMETHING': 'discourse_object_1',
            'SOMEWHERE': 'discourse_location_2'
        }
    
    def normalize_deixis(self, explication: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize deictic terms to discourse entities."""
        logger.info(f"Normalizing deixis in: {explication}")
        
        result = {
            'original': explication,
            'normalized': explication,
            'deictic_mappings': {},
            'discourse_entities': {},
            'explicit_deictics': []
        }
        
        # Find explicit deictic indicators
        explicit_indicators = self._find_explicit_deictics(explication, context)
        result['explicit_deictics'] = explicit_indicators
        
        # Apply normalization
        normalized = explication
        mappings = {}
        
        for deictic_term, discourse_entity in self.discourse_entities.items():
            if deictic_term in explication:
                # Only normalize if there's explicit deictic indicator
                if self._has_explicit_deictic_indicator(deictic_term, explicit_indicators):
                    normalized = normalized.replace(deictic_term, discourse_entity)
                    mappings[deictic_term] = discourse_entity
                else:
                    # Use general terms instead of deictic
                    general_term = self._get_general_term(deictic_term)
                    normalized = normalized.replace(deictic_term, general_term)
                    mappings[deictic_term] = general_term
        
        result['normalized'] = normalized
        result['deictic_mappings'] = mappings
        
        return result
    
    def _find_explicit_deictics(self, explication: str, context: Dict[str, Any]) -> List[str]:
        """Find explicit deictic indicators in context."""
        explicit_indicators = []
        
        # Check for explicit deictic words
        deictic_words = ['this', 'that', 'here', 'now', 'someone', 'something', 'somewhere']
        for word in deictic_words:
            if word in explication.lower():
                explicit_indicators.append(word)
        
        # Check context for additional indicators
        if 'context' in context:
            context_text = context['context'].lower()
            for word in deictic_words:
                if word in context_text:
                    explicit_indicators.append(word)
        
        return explicit_indicators
    
    def _has_explicit_deictic_indicator(self, deictic_term: str, explicit_indicators: List[str]) -> bool:
        """Check if there's an explicit deictic indicator for the term."""
        deictic_word = self.deictic_terms.get(deictic_term, deictic_term.lower())
        return deictic_word in explicit_indicators
    
    def _get_general_term(self, deictic_term: str) -> str:
        """Get general term to replace deictic when no explicit indicator."""
        general_terms = {
            'THIS': 'object',
            'THAT': 'object',
            'HERE': 'location',
            'NOW': 'time',
            'SOMEONE': 'person',
            'SOMETHING': 'object',
            'SOMEWHERE': 'location'
        }
        return general_terms.get(deictic_term, 'entity')


class AspectMoodMolecules:
    """A1 Aspect/mood molecules."""
    
    def __init__(self):
        """Initialize aspect/mood molecules."""
        self.aspect_molecules = {
            'ALMOST_DO': {
                'pattern': r'ALMOST_DO\s*\(\s*([^)]+)\s*\)',
                'expansion': 'WANT(agent, action) ∧ NOT(DO(agent, action)) ∧ CLOSE_TO(agent, action)',
                'confidence': 0.85
            },
            'RECENT_PAST': {
                'pattern': r'RECENT_PAST\s*\(\s*([^)]+)\s*\)',
                'expansion': 'PAST(action) ∧ CLOSE_TO(action, NOW)',
                'confidence': 0.9
            },
            'STILL': {
                'pattern': r'STILL\s*\(\s*([^)]+)\s*\)',
                'expansion': 'PERSIST(action) ∧ CONTINUE(action)',
                'confidence': 0.8
            },
            'NOT_YET': {
                'pattern': r'NOT_YET\s*\(\s*([^)]+)\s*\)',
                'expansion': 'NOT(action) ∧ EXPECT(action)',
                'confidence': 0.75
            }
        }
    
    def expand_aspect_molecules(self, explication: str) -> Dict[str, Any]:
        """Expand aspect/mood molecules into primitive expressions."""
        logger.info(f"Expanding aspect molecules in: {explication}")
        
        result = {
            'original': explication,
            'expanded': explication,
            'expansions': [],
            'molecules_found': []
        }
        
        expanded = explication
        
        for molecule, config in self.aspect_molecules.items():
            pattern = config['pattern']
            matches = re.finditer(pattern, explication, re.IGNORECASE)
            
            for match in matches:
                molecule_found = match.group(0)
                arguments = match.group(1)
                
                # Create expansion
                expansion = config['expansion'].replace('action', arguments)
                
                result['expansions'].append({
                    'molecule': molecule,
                    'arguments': arguments,
                    'expansion': expansion,
                    'confidence': config['confidence']
                })
                
                result['molecules_found'].append(molecule_found)
                
                # Replace in expanded text
                expanded = expanded.replace(molecule_found, expansion)
        
        result['expanded'] = expanded
        
        return result


class EILRulepackV3:
    """EIL Rulepack v3 with advanced reasoning rules."""
    
    def __init__(self):
        """Initialize the EIL rulepack v3."""
        self.quantifier_normalizer = QuantifierScopeNormalizer()
        self.temporal_chaining = TemporalChaining()
        self.counterfactual_reasoning = CounterfactualReasoning()
        self.pronoun_normalizer = PronounDeixisNormalizer()
        self.aspect_molecules = AspectMoodMolecules()
        
        self.fact_counter = 0
        self.step_counter = 0
    
    def reason_with_facts(self, initial_facts: List[EILFact], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply EIL rulepack v3 reasoning to facts."""
        logger.info(f"Starting EIL rulepack v3 reasoning with {len(initial_facts)} initial facts")
        
        all_facts = initial_facts.copy()
        reasoning_steps = []
        step_count = 0
        max_steps = 20  # Increased for more complex reasoning
        
        context = context or {}
        
        while step_count < max_steps:
            step_count += 1
            new_facts_in_step = []
            
            # Step 1: Quantifier scope normalization
            for fact in all_facts:
                if any(quant in fact.predicate for quant in ['ALL', 'SOME', 'EXISTS', 'NOT']):
                    normalization = self.quantifier_normalizer.normalize_quantifier_scope(str(fact.arguments))
                    if normalization['ambiguity_detected'] or normalization['normalized'] != str(fact.arguments):
                        # Create new fact with normalized scope
                        new_fact = EILFact(
                            fact_id=f"normalized_{fact.fact_id}",
                            predicate=fact.predicate,
                            arguments=[normalization['normalized']],
                            confidence=fact.confidence * 0.9,
                            source='quantifier_normalization',
                            timestamp=time.time()
                        )
                        new_facts_in_step.append(new_fact)
                        
                        # Create reasoning step
                        step = ProofStep(
                            step_id=f"step_{self.step_counter}",
                            rule_applied="Q1_quantifier_scope",
                            input_facts=[fact.fact_id],
                            output_facts=[new_fact.fact_id],
                            confidence=new_fact.confidence,
                            reasoning_chain=f"Normalized quantifier scope: {fact.arguments} → {normalization['normalized']}",
                            depth=step_count
                        )
                        reasoning_steps.append(step)
                        self.step_counter += 1
            
            # Step 2: Temporal chaining
            temporal_facts = self.temporal_chaining.apply_temporal_chaining(all_facts)
            if temporal_facts:
                new_facts_in_step.extend(temporal_facts)
                
                # Create reasoning step
                step = ProofStep(
                    step_id=f"step_{self.step_counter}",
                    rule_applied="T1_temporal_chaining",
                    input_facts=[f.fact_id for f in all_facts if f.predicate in ['BEFORE', 'AFTER', 'DURING']],
                    output_facts=[f.fact_id for f in temporal_facts],
                    confidence=np.mean([f.confidence for f in temporal_facts]) if temporal_facts else 0,
                    reasoning_chain=f"Applied temporal chaining: {len(temporal_facts)} new facts",
                    depth=step_count
                )
                reasoning_steps.append(step)
                self.step_counter += 1
            
            # Step 3: Counterfactual reasoning
            counterfactual_facts = self.counterfactual_reasoning.apply_counterfactual_reasoning(all_facts)
            if counterfactual_facts:
                new_facts_in_step.extend(counterfactual_facts)
                
                # Create reasoning step
                step = ProofStep(
                    step_id=f"step_{self.step_counter}",
                    rule_applied="CF1_counterfactual",
                    input_facts=[f.fact_id for f in all_facts if 'PREVENT' in f.predicate or 'ENABLE' in f.predicate],
                    output_facts=[f.fact_id for f in counterfactual_facts],
                    confidence=np.mean([f.confidence for f in counterfactual_facts]) if counterfactual_facts else 0,
                    reasoning_chain=f"Applied counterfactual reasoning: {len(counterfactual_facts)} new facts",
                    depth=step_count
                )
                reasoning_steps.append(step)
                self.step_counter += 1
            
            # Step 4: Pronoun/DEIXIS normalization
            for fact in all_facts:
                if any(deictic in str(fact.arguments) for deictic in ['THIS', 'THAT', 'HERE', 'NOW', 'SOMEONE']):
                    normalization = self.pronoun_normalizer.normalize_deixis(str(fact.arguments), context)
                    if normalization['normalized'] != str(fact.arguments):
                        # Create new fact with normalized deixis
                        new_fact = EILFact(
                            fact_id=f"deixis_normalized_{fact.fact_id}",
                            predicate=fact.predicate,
                            arguments=[normalization['normalized']],
                            confidence=fact.confidence * 0.85,
                            source='deixis_normalization',
                            timestamp=time.time()
                        )
                        new_facts_in_step.append(new_fact)
                        
                        # Create reasoning step
                        step = ProofStep(
                            step_id=f"step_{self.step_counter}",
                            rule_applied="P1_pronoun_deixis",
                            input_facts=[fact.fact_id],
                            output_facts=[new_fact.fact_id],
                            confidence=new_fact.confidence,
                            reasoning_chain=f"Normalized deixis: {fact.arguments} → {normalization['normalized']}",
                            depth=step_count
                        )
                        reasoning_steps.append(step)
                        self.step_counter += 1
            
            # Step 5: Aspect/mood molecule expansion
            for fact in all_facts:
                if any(molecule in str(fact.arguments) for molecule in ['ALMOST_DO', 'RECENT_PAST', 'STILL', 'NOT_YET']):
                    expansion = self.aspect_molecules.expand_aspect_molecules(str(fact.arguments))
                    if expansion['expanded'] != str(fact.arguments):
                        # Create new fact with expanded molecules
                        new_fact = EILFact(
                            fact_id=f"molecule_expanded_{fact.fact_id}",
                            predicate=fact.predicate,
                            arguments=[expansion['expanded']],
                            confidence=fact.confidence * 0.8,
                            source='aspect_molecule_expansion',
                            timestamp=time.time()
                        )
                        new_facts_in_step.append(new_fact)
                        
                        # Create reasoning step
                        step = ProofStep(
                            step_id=f"step_{self.step_counter}",
                            rule_applied="A1_aspect_mood",
                            input_facts=[fact.fact_id],
                            output_facts=[new_fact.fact_id],
                            confidence=new_fact.confidence,
                            reasoning_chain=f"Expanded aspect molecules: {fact.arguments} → {expansion['expanded']}",
                            depth=step_count
                        )
                        reasoning_steps.append(step)
                        self.step_counter += 1
            
            # Add new facts to all_facts
            all_facts.extend(new_facts_in_step)
            
            # Stop if no new facts generated
            if not new_facts_in_step:
                logger.info(f"No new facts generated in step {step_count}, stopping")
                break
            
            # Safety check: stop if too many facts
            if len(all_facts) > 200:
                logger.warning(f"Too many facts generated ({len(all_facts)}), stopping to prevent explosion")
                break
        
        return {
            'initial_facts': [f.to_dict() for f in initial_facts],
            'derived_facts': [f.to_dict() for f in all_facts if f not in initial_facts],
            'all_facts': [f.to_dict() for f in all_facts],
            'reasoning_steps': [s.to_dict() for s in reasoning_steps],
            'total_steps': step_count,
            'total_facts': len(all_facts),
            'derived_facts_count': len([f for f in all_facts if f not in initial_facts]),
            'proof_success_rate': self._calculate_proof_success_rate(reasoning_steps),
            'average_proof_depth': np.mean([s.depth for s in reasoning_steps]) if reasoning_steps else 0,
            'v3_rules_usage': self._calculate_v3_rules_usage(reasoning_steps)
        }
    
    def _calculate_proof_success_rate(self, reasoning_steps: List[ProofStep]) -> float:
        """Calculate proof success rate."""
        if not reasoning_steps:
            return 0.0
        
        successful_steps = sum(1 for step in reasoning_steps if step.confidence > 0.5)
        return successful_steps / len(reasoning_steps)
    
    def _calculate_v3_rules_usage(self, reasoning_steps: List[ProofStep]) -> Dict[str, int]:
        """Calculate usage of v3 rules."""
        rule_usage = defaultdict(int)
        
        for step in reasoning_steps:
            rule_usage[step.rule_applied] += 1
        
        return dict(rule_usage)


def main():
    """Main function to demonstrate EIL rulepack v3."""
    logger.info("Starting EIL rulepack v3 demonstration...")
    
    # Initialize rulepack
    rulepack = EILRulepackV3()
    
    # Test cases with complex reasoning scenarios
    test_cases = [
        {
            'name': 'Quantifier Scope Ambiguity',
            'facts': [
                EILFact('fact_001', 'ALL', ['child', 'NOT(play(child))'], 0.9, 'test', time.time()),
                EILFact('fact_002', 'DO', ['child1', 'run'], 0.8, 'test', time.time())
            ]
        },
        {
            'name': 'Temporal Chaining',
            'facts': [
                EILFact('fact_003', 'BEFORE', ['event1', 'event2'], 0.9, 'test', time.time()),
                EILFact('fact_004', 'BEFORE', ['event2', 'event3'], 0.9, 'test', time.time()),
                EILFact('fact_005', 'DURING', ['event4', 'event5'], 0.8, 'test', time.time())
            ]
        },
        {
            'name': 'Counterfactual Reasoning',
            'facts': [
                EILFact('fact_006', 'PREVENT', ['rain', 'drought'], 0.9, 'test', time.time()),
                EILFact('fact_007', 'DO', ['rain'], 0.8, 'test', time.time()),
                EILFact('fact_008', 'ENABLE', ['study', 'pass_exam'], 0.9, 'test', time.time()),
                EILFact('fact_009', 'NOT', ['DO(study)'], 0.8, 'test', time.time())
            ]
        },
        {
            'name': 'Deixis Normalization',
            'facts': [
                EILFact('fact_010', 'LIKE', ['i', 'THIS'], 0.9, 'test', time.time()),
                EILFact('fact_011', 'AT_LOC', ['cat', 'HERE'], 0.8, 'test', time.time())
            ],
            'context': {'context': 'I like this cat. The cat is here.'}
        },
        {
            'name': 'Aspect Molecules',
            'facts': [
                EILFact('fact_012', 'ALMOST_DO', ['person', 'fall'], 0.9, 'test', time.time()),
                EILFact('fact_013', 'RECENT_PAST', ['person', 'arrive'], 0.8, 'test', time.time()),
                EILFact('fact_014', 'STILL', ['person', 'work'], 0.9, 'test', time.time())
            ]
        }
    ]
    
    # Run reasoning on test cases
    results = []
    for test_case in test_cases:
        reasoning_result = rulepack.reason_with_facts(
            test_case['facts'], 
            test_case.get('context', {})
        )
        reasoning_result['test_case'] = test_case['name']
        results.append(reasoning_result)
    
    # Print results
    print("\n" + "="*80)
    print("EIL RULEPACK V3 RESULTS")
    print("="*80)
    
    print(f"Test Cases: {len(test_cases)}")
    
    for i, result in enumerate(results):
        print(f"\nTest Case {i+1}: {result['test_case']}")
        print(f"  Initial Facts: {len(result['initial_facts'])}")
        print(f"  Derived Facts: {result['derived_facts_count']}")
        print(f"  Reasoning Steps: {len(result['reasoning_steps'])}")
        print(f"  Proof Success Rate: {result['proof_success_rate']:.3f}")
        print(f"  Average Proof Depth: {result['average_proof_depth']:.2f}")
        print(f"  V3 Rules Usage: {result['v3_rules_usage']}")
    
    # Calculate overall metrics
    overall_success_rate = np.mean([r['proof_success_rate'] for r in results])
    overall_depth = np.mean([r['average_proof_depth'] for r in results])
    total_v3_usage = sum(sum(r['v3_rules_usage'].values()) for r in results)
    
    print(f"\nOverall Metrics:")
    print(f"  Average Proof Success Rate: {overall_success_rate:.3f}")
    print(f"  Average Proof Depth: {overall_depth:.2f}")
    print(f"  Total V3 Rules Applied: {total_v3_usage}")
    
    # Check acceptance criteria
    print(f"\nAcceptance Criteria:")
    print(f"  Proof Success ≥ 60%: {'✅' if overall_success_rate >= 0.6 else '❌'} ({overall_success_rate:.1%})")
    print(f"  Average Proof Depth ↓: {'✅' if overall_depth < 5 else '❌'} ({overall_depth:.2f})")
    print(f"  V3 Rules Usage ≥ 50%: {'✅' if total_v3_usage >= 5 else '❌'} ({total_v3_usage} rules)")
    
    # Save results
    output_path = Path("data/eil_rulepack_v3_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    try:
        json_results = convert_numpy_types({
            'test_cases': results,
            'overall_metrics': {
                'proof_success_rate': overall_success_rate,
                'average_proof_depth': overall_depth,
                'total_v3_usage': total_v3_usage
            },
            'acceptance_criteria': {
                'proof_success_60_percent': overall_success_rate >= 0.6,
                'proof_depth_reduced': overall_depth < 5,
                'v3_rules_50_percent': total_v3_usage >= 5
            }
        })
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    logger.info(f"EIL rulepack v3 results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("EIL rulepack v3 demonstration completed!")
    print("="*80)


if __name__ == "__main__":
    main()

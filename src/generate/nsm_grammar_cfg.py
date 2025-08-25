#!/usr/bin/env python3
"""
NSM Typed CFG Grammar System

This module implements a typed Context-Free Grammar for NSM explications
with legality checking, molecule caps, and safety rails as specified in the plan.
"""

import json
import logging
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class GrammarType(Enum):
    """Grammar types for NSM primitives."""
    EVENT = "EVENT"
    STATE = "STATE"
    THING = "THING"
    PERSON = "PERSON"
    PLACE = "PLACE"
    TIME = "TIME"
    QUANT = "QUANT"
    SAY = "SAY"
    THINK = "THINK"
    WANT = "WANT"
    CAN = "CAN"
    NOT = "NOT"
    BECAUSE = "BECAUSE"
    IF = "IF"
    WHEN = "WHEN"
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    VERY = "VERY"
    MORE = "MORE"
    LIKE = "LIKE"
    GOOD = "GOOD"
    BAD = "BAD"
    TRUE = "TRUE"
    FALSE = "FALSE"

@dataclass
class GrammarConstraint:
    """Grammar constraints for safety rails."""
    max_depth: int = 6
    molecule_ratio_max: float = 0.25
    max_scope_depth: int = 3
    max_clause_length: int = 20

@dataclass
class GrammarState:
    """Current state of grammar parsing."""
    stack: List[str] = field(default_factory=list)
    depth: int = 0
    scope_depth: int = 0
    molecule_count: int = 0
    total_tokens: int = 0
    current_scope: Optional[str] = None

class NSMTypedCFG:
    """Typed Context-Free Grammar for NSM explications."""
    
    def __init__(self, grammar_file: Optional[str] = None):
        """Initialize the NSM typed CFG.
        
        Args:
            grammar_file: Path to grammar JSON file (optional)
        """
        if grammar_file:
            self.load_grammar(grammar_file)
        else:
            self._initialize_default_grammar()
        
        self.constraints = GrammarConstraint()
        self.state = GrammarState()
    
    def _initialize_default_grammar(self):
        """Initialize the default NSM grammar (ptb-0.3)."""
        self.grammar = {
            "version": "ptb-0.3",
            "terminals": [
                "I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING", "BODY",
                "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR",
                "BECAUSE", "IF", "NOT", "SAME", "DIFFERENT", "MAYBE",
                "BEFORE", "AFTER", "WHEN", "CAUSE", "MAKE", "LET",
                "IN", "ON", "UNDER", "NEAR", "FAR", "INSIDE",
                "ALL", "MANY", "SOME", "FEW", "MUCH", "LITTLE",
                "GOOD", "BAD", "BIG", "SMALL", "RIGHT", "WRONG",
                "DO", "HAPPEN", "MOVE", "TOUCH", "LIVE", "DIE",
                "THIS", "THE SAME", "OTHER", "ONE", "TWO", "SOME",
                "VERY", "MORE", "LIKE", "KIND OF",
                "SAY", "WORDS", "TRUE", "FALSE", "WHERE", "WHEN"
            ],
            "types": {
                "I": "PERSON", "YOU": "PERSON", "SOMEONE": "PERSON", "PEOPLE": "PERSON",
                "SOMETHING": "THING", "THING": "THING", "BODY": "THING",
                "THINK": "EVENT", "KNOW": "STATE", "WANT": "EVENT", "FEEL": "EVENT",
                "SEE": "EVENT", "HEAR": "EVENT", "SAY": "EVENT",
                "BECAUSE": "BECAUSE", "IF": "IF", "NOT": "NOT", "WHEN": "WHEN",
                "BEFORE": "BEFORE", "AFTER": "AFTER", "TRUE": "TRUE", "FALSE": "FALSE",
                "DO": "EVENT", "HAPPEN": "EVENT", "MOVE": "EVENT",
                "GOOD": "GOOD", "BAD": "BAD", "BIG": "QUANT", "SMALL": "QUANT",
                "VERY": "VERY", "MORE": "MORE", "LIKE": "LIKE",
                "ALL": "QUANT", "MANY": "QUANT", "SOME": "QUANT", "FEW": "QUANT",
                "IN": "PLACE", "ON": "PLACE", "UNDER": "PLACE", "NEAR": "PLACE",
                "THIS": "THING", "THAT": "THING", "WHERE": "PLACE"
            },
            "productions": {
                "CLAUSE": [
                    ["EVENT"],
                    ["STATE"],
                    ["CLAUSE", "CAUSECOND"],
                    ["POLARITY", "CLAUSE"]
                ],
                "EVENT": [
                    ["DO", "(", "AGENT", ",", "ACTION", ")"],
                    ["SAY", "(", "AGENT", ",", "CLAUSE", ")"],
                    ["THINK", "(", "AGENT", ",", "CLAUSE", ")"],
                    ["WANT", "(", "AGENT", ",", "CLAUSE", ")"],
                    ["SEE", "(", "AGENT", ",", "THING", ")"],
                    ["HEAR", "(", "AGENT", ",", "THING", ")"],
                    ["FEEL", "(", "AGENT", ",", "STATE", ")"],
                    ["MOVE", "(", "AGENT", ",", "PLACE", ")"],
                    ["HAPPEN", "(", "EVENT", ")"]
                ],
                "STATE": [
                    ["BE", "(", "SUBJ", ",", "PRED", ")"],
                    ["TRUE", "(", "PROPOSITION", ")"],
                    ["FALSE", "(", "PROPOSITION", ")"],
                    ["KNOW", "(", "AGENT", ",", "CLAUSE", ")"],
                    ["GOOD", "(", "THING", ")"],
                    ["BAD", "(", "THING", ")"],
                    ["BIG", "(", "THING", ")"],
                    ["SMALL", "(", "THING", ")"],
                    ["SAME", "(", "THING1", ",", "THING2", ")"],
                    ["DIFFERENT", "(", "THING1", ",", "THING2", ")"]
                ],
                "CAUSECOND": [
                    ["BECAUSE", "CLAUSE"],
                    ["IF", "CLAUSE"],
                    ["WHEN", "CLAUSE"],
                    ["BEFORE", "CLAUSE"],
                    ["AFTER", "CLAUSE"]
                ],
                "POLARITY": [
                    ["NOT"],
                    ["MAYBE"]
                ],
                "AGENT": [
                    ["I"], ["YOU"], ["SOMEONE"], ["PEOPLE"]
                ],
                "ACTION": [
                    ["DO"], ["SAY"], ["THINK"], ["WANT"], ["SEE"], ["HEAR"], ["FEEL"], ["MOVE"]
                ],
                "SUBJ": [
                    ["I"], ["YOU"], ["SOMEONE"], ["PEOPLE"], ["SOMETHING"], ["THING"]
                ],
                "PRED": [
                    ["GOOD"], ["BAD"], ["BIG"], ["SMALL"], ["SAME"], ["DIFFERENT"]
                ],
                "THING": [
                    ["THIS"], ["THAT"], ["SOMETHING"], ["THING"], ["BODY"]
                ],
                "PLACE": [
                    ["IN"], ["ON"], ["UNDER"], ["NEAR"], ["FAR"], ["INSIDE"], ["WHERE"]
                ],
                "TIME": [
                    ["WHEN"], ["BEFORE"], ["AFTER"], ["NOW"]
                ],
                "QUANT": [
                    ["ALL"], ["MANY"], ["SOME"], ["FEW"], ["MUCH"], ["LITTLE"], ["ONE"], ["TWO"]
                ],
                "PROPOSITION": [
                    ["CLAUSE"]
                ]
            },
            "constraints": {
                "max_depth": 6,
                "molecule_ratio_max": 0.25,
                "max_scope_depth": 3,
                "max_clause_length": 20
            }
        }
    
    def load_grammar(self, grammar_file: str):
        """Load grammar from JSON file."""
        try:
            with open(grammar_file, 'r', encoding='utf-8') as f:
                self.grammar = json.load(f)
            logger.info(f"Loaded grammar version {self.grammar.get('version', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to load grammar from {grammar_file}: {e}")
            self._initialize_default_grammar()
    
    def save_grammar(self, grammar_file: str):
        """Save grammar to JSON file."""
        try:
            with open(grammar_file, 'w', encoding='utf-8') as f:
                json.dump(self.grammar, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved grammar to {grammar_file}")
        except Exception as e:
            logger.error(f"Failed to save grammar to {grammar_file}: {e}")
    
    def get_allowed_tokens(self, prefix_tokens: List[str]) -> Set[str]:
        """Get allowed next tokens based on current grammar state.
        
        Args:
            prefix_tokens: List of tokens generated so far
            
        Returns:
            Set of allowed next tokens
        """
        allowed = set()
        
        # Reset state
        self.state = GrammarState()
        
        # Parse prefix tokens to determine current state
        for token in prefix_tokens:
            self._update_state(token)
        
        # Get allowed tokens based on current state
        if not prefix_tokens:
            # Start with clause-level productions
            allowed.update(self._get_clause_starters())
        else:
            # Get allowed tokens based on current production
            allowed.update(self._get_next_tokens(prefix_tokens))
        
        # Apply constraints
        allowed = self._apply_constraints(allowed, prefix_tokens)
        
        return allowed
    
    def _update_state(self, token: str):
        """Update grammar state based on token."""
        self.state.total_tokens += 1
        
        # Check for scope changes
        if token == "(":
            self.state.scope_depth += 1
            self.state.depth += 1
        elif token == ")":
            self.state.scope_depth -= 1
            self.state.depth -= 1
        
        # Check for molecules
        if self.state.total_tokens > 1:
            prev_token = self.state.stack[-1] if self.state.stack else ""
            molecule = f"{prev_token} {token}"
            if self._is_molecule(molecule):
                self.state.molecule_count += 1
        
        self.state.stack.append(token)
    
    def _get_clause_starters(self) -> Set[str]:
        """Get tokens that can start a clause."""
        starters = set()
        
        # Add event starters
        starters.update(["DO", "SAY", "THINK", "WANT", "SEE", "HEAR", "FEEL", "MOVE", "HAPPEN"])
        
        # Add state starters
        starters.update(["BE", "TRUE", "FALSE", "KNOW", "GOOD", "BAD", "BIG", "SMALL", "SAME", "DIFFERENT"])
        
        # Add polarity starters
        starters.update(["NOT", "MAYBE"])
        
        return starters
    
    def _get_next_tokens(self, prefix_tokens: List[str]) -> Set[str]:
        """Get next allowed tokens based on current production."""
        allowed = set()
        
        # Simple heuristic: if we're in a production, continue it
        if len(prefix_tokens) >= 2:
            last_two = prefix_tokens[-2:]
            
            # Check for common patterns
            if last_two == ["DO", "("]:
                allowed.update(["I", "YOU", "SOMEONE", "PEOPLE"])
            elif last_two == ["SAY", "("]:
                allowed.update(["I", "YOU", "SOMEONE", "PEOPLE"])
            elif last_two == ["THINK", "("]:
                allowed.update(["I", "YOU", "SOMEONE", "PEOPLE"])
            elif last_two == ["BECAUSE", "("]:
                allowed.update(self._get_clause_starters())
            elif last_two == ["IF", "("]:
                allowed.update(self._get_clause_starters())
            elif last_two == ["WHEN", "("]:
                allowed.update(self._get_clause_starters())
            elif last_two == ["NOT", "("]:
                allowed.update(self._get_clause_starters())
            elif prefix_tokens[-1] == ",":
                # After comma, expect various tokens
                allowed.update(["CLAUSE", "THING", "PLACE", "TIME", "QUANT"])
            elif prefix_tokens[-1] == ")":
                # After closing paren, expect more clauses or end
                allowed.update(["CLAUSE", "CAUSECOND", "POLARITY"])
        
        # If no specific pattern, allow common continuations
        if not allowed:
            allowed.update(self.grammar["terminals"])
        
        return allowed
    
    def _apply_constraints(self, allowed: Set[str], prefix_tokens: List[str]) -> Set[str]:
        """Apply grammar constraints to allowed tokens."""
        constrained = allowed.copy()
        
        # Check depth constraint
        if self.state.depth >= self.constraints.max_depth:
            # Only allow closing tokens
            constrained = constrained.intersection([")", "CLAUSE", "CAUSECOND"])
        
        # Check scope depth constraint
        if self.state.scope_depth >= self.constraints.max_scope_depth:
            # Only allow closing parentheses
            constrained = constrained.intersection([")"])
        
        # Check clause length constraint
        if len(prefix_tokens) >= self.constraints.max_clause_length:
            # Only allow closing tokens
            constrained = constrained.intersection([")", "CLAUSE", "CAUSECOND"])
        
        # Check molecule ratio constraint
        if self.state.total_tokens > 0:
            current_ratio = self.state.molecule_count / self.state.total_tokens
            if current_ratio >= self.constraints.molecule_ratio_max:
                # Avoid creating more molecules
                constrained = constrained - self._get_molecule_starters()
        
        return constrained
    
    def _is_molecule(self, token_pair: str) -> bool:
        """Check if a token pair is a molecule."""
        molecules = [
            "I THINK", "YOU KNOW", "I WANT", "YOU FEEL", "I SEE", "YOU HEAR",
            "BECAUSE IF", "IF NOT", "SAME AS", "DIFFERENT FROM",
            "BEFORE WHEN", "AFTER WHEN", "WHEN THIS",
            "IN THIS", "ON THIS", "NEAR THIS", "FAR FROM",
            "ALL THIS", "MANY THINGS", "SOME PEOPLE", "FEW THINGS",
            "VERY GOOD", "VERY BAD", "BIG THING", "SMALL THING",
            "DO THIS", "HAPPEN HERE", "MOVE THERE",
            "SAY WORDS", "TRUE THING", "FALSE THING", "WHERE THIS", "WHEN THIS"
        ]
        return token_pair in molecules
    
    def _get_molecule_starters(self) -> Set[str]:
        """Get tokens that can start molecules."""
        return {"I", "YOU", "BECAUSE", "IF", "SAME", "DIFFERENT", "BEFORE", "AFTER", "WHEN", "IN", "ON", "NEAR", "FAR", "ALL", "MANY", "SOME", "FEW", "VERY", "DO", "HAPPEN", "MOVE", "SAY", "TRUE", "FALSE", "WHERE"}
    
    def check_legality(self, tokens: List[str]) -> Tuple[bool, float, List[str]]:
        """Check if a sequence of tokens is legal according to the grammar.
        
        Args:
            tokens: List of tokens to check
            
        Returns:
            Tuple of (is_legal, legality_score, violations)
        """
        violations = []
        score = 1.0
        
        # Reset state
        self.state = GrammarState()
        
        # Check each token
        for i, token in enumerate(tokens):
            # Get allowed tokens at this point
            prefix = tokens[:i]
            allowed = self.get_allowed_tokens(prefix)
            
            # Check if token is allowed
            if token not in allowed:
                violations.append(f"Token '{token}' not allowed at position {i}")
                score *= 0.8  # Penalty for illegal token
            
            # Update state
            self._update_state(token)
            
            # Check constraints
            if self.state.depth > self.constraints.max_depth:
                violations.append(f"Depth {self.state.depth} exceeds max {self.constraints.max_depth}")
                score *= 0.9
            
            if self.state.scope_depth > self.constraints.max_scope_depth:
                violations.append(f"Scope depth {self.state.scope_depth} exceeds max {self.constraints.max_scope_depth}")
                score *= 0.9
        
        # Check final molecule ratio
        if self.state.total_tokens > 0:
            molecule_ratio = self.state.molecule_count / self.state.total_tokens
            if molecule_ratio > self.constraints.molecule_ratio_max:
                violations.append(f"Molecule ratio {molecule_ratio:.3f} exceeds max {self.constraints.molecule_ratio_max}")
                score *= 0.9
        
        is_legal = len(violations) == 0
        
        return is_legal, score, violations
    
    def get_grammar_info(self) -> Dict[str, Any]:
        """Get information about the grammar."""
        return {
            "version": self.grammar.get("version", "unknown"),
            "num_terminals": len(self.grammar.get("terminals", [])),
            "num_productions": len(self.grammar.get("productions", {})),
            "constraints": self.constraints.__dict__,
            "types": len(self.grammar.get("types", {}))
        }

def create_grammar_ptb_03() -> NSMTypedCFG:
    """Create the ptb-0.3 grammar as specified in the plan."""
    cfg = NSMTypedCFG()
    
    # Save the grammar to a file
    cfg.save_grammar("data/grammar_ptb_03.json")
    
    return cfg

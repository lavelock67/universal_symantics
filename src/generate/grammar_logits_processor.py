#!/usr/bin/env python3
"""
Grammar Logits Processor for Constrained Decoding

This module implements logit-masking constrained decoding that forces output
to respect the NSM grammar, as specified in the plan.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

from .nsm_grammar_cfg import NSMTypedCFG, GrammarState

logger = logging.getLogger(__name__)

class ConstraintMode(Enum):
    """Constraint modes for generation."""
    HARD = "hard"      # Hard constraints (mask to -inf)
    HYBRID = "hybrid"  # Hard + soft penalties
    OFF = "off"        # No constraints

@dataclass
class GrammarLogitsConfig:
    """Configuration for grammar logits processing."""
    constraint_mode: ConstraintMode = ConstraintMode.HARD
    illegality_penalty: float = 1.0
    molecule_ratio_penalty: float = 0.5
    scope_violation_penalty: float = 0.8
    repetition_penalty: float = 0.1
    duplication_penalty: float = 0.2
    triviality_penalty: float = 0.3
    decoding_profile: str = "hybrid"
    beam_k: int = 4
    max_depth: int = 6
    molecule_ratio_max: float = 0.25

class GrammarState:
    """Enhanced grammar state for logits processing."""
    
    def __init__(self, grammar: NSMTypedCFG):
        """Initialize grammar state.
        
        Args:
            grammar: NSM typed CFG grammar
        """
        self.grammar = grammar
        self.stack: List[str] = []
        self.depth: int = 0
        self.scope_depth: int = 0
        self.molecule_count: int = 0
        self.total_tokens: int = 0
        self.current_scope: Optional[str] = None
        self.violations: List[str] = []
        self.beam_scores: List[float] = []
    
    def allowed_tokens(self, prefix_tokens: List[str]) -> Set[str]:
        """Get allowed next tokens based on current state.
        
        Args:
            prefix_tokens: List of tokens generated so far
            
        Returns:
            Set of allowed next tokens
        """
        return self.grammar.get_allowed_tokens(prefix_tokens)
    
    def update(self, token: str):
        """Update state with new token.
        
        Args:
            token: New token to add
        """
        self.total_tokens += 1
        self.stack.append(token)
        
        # Check for scope changes
        if token == "(":
            self.scope_depth += 1
            self.depth += 1
        elif token == ")":
            self.scope_depth -= 1
            self.depth -= 1
        
        # Check for molecules
        if self.total_tokens > 1:
            prev_token = self.stack[-2] if len(self.stack) > 1 else ""
            molecule = f"{prev_token} {token}"
            if self._is_molecule(molecule):
                self.molecule_count += 1
    
    def _is_molecule(self, token_pair: str) -> bool:
        """Check if a token pair is a molecule."""
        return self.grammar._is_molecule(token_pair)
    
    def get_violations(self) -> List[str]:
        """Get current grammar violations."""
        violations = []
        
        # Check depth constraint
        if self.depth > self.grammar.constraints.max_depth:
            violations.append(f"Depth {self.depth} exceeds max {self.grammar.constraints.max_depth}")
        
        # Check scope depth constraint
        if self.scope_depth > self.grammar.constraints.max_scope_depth:
            violations.append(f"Scope depth {self.scope_depth} exceeds max {self.grammar.constraints.max_scope_depth}")
        
        # Check molecule ratio constraint
        if self.total_tokens > 0:
            molecule_ratio = self.molecule_count / self.total_tokens
            if molecule_ratio > self.grammar.constraints.molecule_ratio_max:
                violations.append(f"Molecule ratio {molecule_ratio:.3f} exceeds max {self.grammar.constraints.molecule_ratio_max}")
        
        return violations
    
    def get_penalty_score(self, config: GrammarLogitsConfig) -> float:
        """Calculate penalty score for current state.
        
        Args:
            config: Logits processing configuration
            
        Returns:
            Penalty score (higher = more penalty)
        """
        penalty = 0.0
        
        # Calculate repetition penalty (3-gram repetition)
        repetition_penalty = self._calculate_repetition_penalty(config.repetition_penalty)
        
        # Calculate duplication penalty (duplicate content)
        duplication_penalty = self._calculate_duplication_penalty(config.duplication_penalty)
        
        # Calculate triviality penalty (no new nodes)
        triviality_penalty = self._calculate_triviality_penalty(config.triviality_penalty)
        
        penalty = repetition_penalty + duplication_penalty + triviality_penalty
        return penalty
    
    def _calculate_repetition_penalty(self, lambda_rep: float) -> float:
        """Calculate repetition penalty based on 3-gram repetition."""
        if len(self.stack) < 3:
            return 0.0
        
        # Count 3-gram repetitions
        trigrams = []
        for i in range(len(self.stack) - 2):
            trigram = tuple(self.stack[i:i+3])
            trigrams.append(trigram)
        
        unique_trigrams = set(trigrams)
        repetition_ratio = 1.0 - (len(unique_trigrams) / len(trigrams)) if trigrams else 0.0
        
        return lambda_rep * repetition_ratio
    
    def _calculate_duplication_penalty(self, lambda_dup: float) -> float:
        """Calculate duplication penalty for duplicate content."""
        if len(self.stack) < 2:
            return 0.0
        
        # Count consecutive duplicates
        consecutive_duplicates = 0
        for i in range(1, len(self.stack)):
            if self.stack[i] == self.stack[i-1]:
                consecutive_duplicates += 1
        
        duplication_ratio = consecutive_duplicates / (len(self.stack) - 1) if len(self.stack) > 1 else 0.0
        return lambda_dup * duplication_ratio
    
    def _calculate_triviality_penalty(self, lambda_triv: float) -> float:
        """Calculate triviality penalty for no new nodes."""
        if len(self.stack) < 2:
            return 0.0
        
        # Check if the graph is identical to input (trivial transformation)
        # This is a simplified check - in practice, you'd compare EIL graphs
        unique_tokens = set(self.stack)
        token_diversity = len(unique_tokens) / len(self.stack) if self.stack else 0.0
        
        # Low diversity suggests trivial content
        triviality_score = 1.0 - token_diversity
        return lambda_triv * triviality_score
        
        # Illegality penalty
        violations = self.get_violations()
        if violations:
            penalty += config.illegality_penalty * len(violations)
        
        # Molecule ratio penalty
        if self.total_tokens > 0:
            molecule_ratio = self.molecule_count / self.total_tokens
            if molecule_ratio > config.molecule_ratio_max:
                penalty += config.molecule_ratio_penalty * (molecule_ratio - config.molecule_ratio_max)
        
        # Scope violation penalty
        if self.scope_depth > self.grammar.constraints.max_scope_depth:
            penalty += config.scope_violation_penalty * (self.scope_depth - self.grammar.constraints.max_scope_depth)
        
        return penalty

class GrammarLogitsProcessor:
    """Grammar-aware logits processor for constrained decoding."""
    
    def __init__(self, grammar: NSMTypedCFG, config: Optional[GrammarLogitsConfig] = None):
        """Initialize the grammar logits processor.
        
        Args:
            grammar: NSM typed CFG grammar
            config: Processing configuration
        """
        self.grammar = grammar
        self.config = config or GrammarLogitsConfig()
        self.states: Dict[int, GrammarState] = {}
    
    def __call__(self, step_logits: torch.Tensor, prefix_tokens: List[str], 
                 beam_id: int = 0) -> torch.Tensor:
        """Process logits with grammar constraints.
        
        Args:
            step_logits: Raw logits from model
            prefix_tokens: Tokens generated so far
            beam_id: Beam search ID
            
        Returns:
            Processed logits with constraints applied
        """
        # Get or create grammar state for this beam
        if beam_id not in self.states:
            self.states[beam_id] = GrammarState(self.grammar)
        
        state = self.states[beam_id]
        
        # Get allowed tokens
        allowed_tokens = state.allowed_tokens(prefix_tokens)
        
        # Create mask
        mask = torch.full_like(step_logits, float('-inf'))
        
        # Apply hard constraints
        if self.config.constraint_mode in [ConstraintMode.HARD, ConstraintMode.HYBRID]:
            # Mask all tokens except allowed ones
            for token in allowed_tokens:
                # Find token index (simplified - in practice you'd have a proper tokenizer)
                token_idx = self._get_token_index(token)
                if token_idx is not None:
                    mask[token_idx] = 0.0
        
        # Apply soft penalties
        if self.config.constraint_mode == ConstraintMode.HYBRID:
            penalty_score = state.get_penalty_score(self.config)
            if penalty_score > 0:
                # Apply penalty to all logits
                step_logits = step_logits - penalty_score
        
        # Combine logits and mask
        processed_logits = step_logits + mask
        
        return processed_logits
    
    def _get_token_index(self, token: str) -> Optional[int]:
        """Get token index in vocabulary (simplified implementation).
        
        Args:
            token: Token to find
            
        Returns:
            Token index or None if not found
        """
        # This is a simplified implementation
        # In practice, you'd use a proper tokenizer
        if token in self.grammar.grammar["terminals"]:
            return self.grammar.grammar["terminals"].index(token)
        return None
    
    def update_state(self, token: str, beam_id: int = 0):
        """Update grammar state with new token.
        
        Args:
            token: New token
            beam_id: Beam search ID
        """
        if beam_id in self.states:
            self.states[beam_id].update(token)
    
    def get_beam_score(self, logprob: float, prefix_tokens: List[str], 
                      beam_id: int = 0) -> float:
        """Calculate beam search score with grammar penalties.
        
        Args:
            logprob: Log probability from model
            prefix_tokens: Tokens generated so far
            beam_id: Beam search ID
            
        Returns:
            Combined score for beam search
        """
        if beam_id not in self.states:
            return logprob
        
        state = self.states[beam_id]
        
        # Base score is log probability
        score = logprob
        
        # Apply grammar penalties
        penalty_score = state.get_penalty_score(self.config)
        score -= penalty_score
        
        return score
    
    def should_drop_beam(self, prefix_tokens: List[str], beam_id: int = 0) -> bool:
        """Check if beam should be dropped due to hard constraint violations.
        
        Args:
            prefix_tokens: Tokens generated so far
            beam_id: Beam search ID
            
        Returns:
            True if beam should be dropped
        """
        if beam_id not in self.states:
            return False
        
        state = self.states[beam_id]
        violations = state.get_violations()
        
        # Drop beam if there are hard constraint violations
        return len(violations) > 0
    
    def reset_beam(self, beam_id: int):
        """Reset grammar state for a beam.
        
        Args:
            beam_id: Beam search ID
        """
        if beam_id in self.states:
            self.states[beam_id] = GrammarState(self.grammar)

class GrammarAwareDecoder:
    """Grammar-aware decoder with logits processing."""
    
    def __init__(self, grammar: NSMTypedCFG, config: Optional[GrammarLogitsConfig] = None):
        """Initialize the grammar-aware decoder.
        
        Args:
            grammar: NSM typed CFG grammar
            config: Decoding configuration
        """
        self.grammar = grammar
        self.config = config or GrammarLogitsConfig()
        self.logits_processor = GrammarLogitsProcessor(grammar, config)
    
    def generate_constrained(self, prompt: str, max_length: int = 20) -> Dict[str, Any]:
        """Generate text with grammar constraints.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            
        Returns:
            Generation result with metadata
        """
        # Extract NSM primes from prompt
        prompt_primes = self._extract_primes_from_text(prompt)
        
        # Initialize generation
        generated_tokens = []
        beam_scores = [0.0]  # Single beam for now
        beam_id = 0
        
        # Reset grammar state
        self.logits_processor.reset_beam(beam_id)
        
        # Generate tokens
        for step in range(max_length):
            # Get allowed tokens
            if beam_id not in self.logits_processor.states:
                self.logits_processor.states[beam_id] = GrammarState(self.grammar)
            allowed_tokens = self.logits_processor.states[beam_id].allowed_tokens(generated_tokens)
            
            if not allowed_tokens:
                break
            
            # Select next token (simplified - in practice you'd use a language model)
            next_token = self._select_next_token(allowed_tokens, prompt_primes)
            
            # Check if beam should be dropped
            if self.logits_processor.should_drop_beam(generated_tokens + [next_token], beam_id):
                break
            
            # Add token
            generated_tokens.append(next_token)
            self.logits_processor.update_state(next_token, beam_id)
            
            # Update beam score
            beam_scores[beam_id] = self.logits_processor.get_beam_score(
                beam_scores[beam_id], generated_tokens, beam_id
            )
        
        # Check legality
        is_legal, legality_score, violations = self.grammar.check_legality(generated_tokens)
        
        return {
            "generated_tokens": generated_tokens,
            "generated_text": " ".join(generated_tokens),
            "is_legal": is_legal,
            "legality_score": legality_score,
            "violations": violations,
            "beam_score": beam_scores[beam_id],
            "constraint_mode": self.config.constraint_mode.value
        }
    
    def _extract_primes_from_text(self, text: str) -> List[str]:
        """Extract NSM primes from text."""
        words = text.upper().split()
        primes = []
        for word in words:
            if word in self.grammar.grammar["terminals"]:
                primes.append(word)
        return primes
    
    def _select_next_token(self, allowed_tokens: Set[str], prompt_primes: List[str]) -> str:
        """Select next token from allowed tokens (simplified implementation)."""
        # Prefer tokens that appear in prompt
        for token in allowed_tokens:
            if token in prompt_primes:
                return token
        
        # Fall back to first allowed token
        return list(allowed_tokens)[0] if allowed_tokens else "THIS"

def create_grammar_logits_processor(grammar_file: Optional[str] = None,
                                  config: Optional[GrammarLogitsConfig] = None) -> GrammarAwareDecoder:
    """Create a grammar-aware decoder with logits processing.
    
    Args:
        grammar_file: Path to grammar file (optional)
        config: Decoding configuration (optional)
        
    Returns:
        Grammar-aware decoder
    """
    # Load or create grammar
    if grammar_file:
        grammar = NSMTypedCFG(grammar_file)
    else:
        grammar = NSMTypedCFG()
    
    # Create decoder
    decoder = GrammarAwareDecoder(grammar, config)
    
    return decoder

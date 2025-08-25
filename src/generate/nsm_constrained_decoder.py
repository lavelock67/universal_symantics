#!/usr/bin/env python3
"""
NSM Constrained Decoder

This module implements primes-first decoding that forces output to use only
NSM primes and allowed molecules, as suggested by ChatGPT5.
"""

import logging
import re
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# All 65 NSM primes
ALL_NSM_PRIMES = {
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
}

# NSM grammar molecules (allowed combinations)
NSM_MOLECULES = {
    # Basic combinations
    "I THINK": "I THINK",
    "YOU KNOW": "YOU KNOW", 
    "I WANT": "I WANT",
    "YOU FEEL": "YOU FEEL",
    "I SEE": "I SEE",
    "YOU HEAR": "YOU HEAR",
    
    # Logical combinations
    "BECAUSE IF": "BECAUSE IF",
    "IF NOT": "IF NOT",
    "SAME AS": "SAME AS",
    "DIFFERENT FROM": "DIFFERENT FROM",
    
    # Temporal combinations
    "BEFORE WHEN": "BEFORE WHEN",
    "AFTER WHEN": "AFTER WHEN",
    "WHEN THIS": "WHEN THIS",
    
    # Spatial combinations
    "IN THIS": "IN THIS",
    "ON THIS": "ON THIS",
    "NEAR THIS": "NEAR THIS",
    "FAR FROM": "FAR FROM",
    
    # Quantifier combinations
    "ALL THIS": "ALL THIS",
    "MANY THINGS": "MANY THINGS",
    "SOME PEOPLE": "SOME PEOPLE",
    "FEW THINGS": "FEW THINGS",
    
    # Evaluator combinations
    "VERY GOOD": "VERY GOOD",
    "VERY BAD": "VERY BAD",
    "BIG THING": "BIG THING",
    "SMALL THING": "SMALL THING",
    
    # Action combinations
    "DO THIS": "DO THIS",
    "HAPPEN HERE": "HAPPEN HERE",
    "MOVE THERE": "MOVE THERE",
    
    # Communication combinations
    "SAY WORDS": "SAY WORDS",
    "TRUE THING": "TRUE THING",
    "FALSE THING": "FALSE THING",
    "WHERE THIS": "WHERE THIS",
    "WHEN THIS": "WHEN THIS"
}

@dataclass
class NSMConstraint:
    """Constraint for NSM generation."""
    allowed_primes: Set[str]
    allowed_molecules: Set[str]
    grammar_rules: List[str]
    max_length: int = 50
    min_length: int = 5

@dataclass
class NSMGenerationResult:
    """Result of NSM constrained generation."""
    text: str
    used_primes: List[str]
    used_molecules: List[str]
    confidence: float
    grammar_violations: List[str]
    nsm_compliance: float

class NSMConstrainedDecoder:
    """Decoder that forces output to use only NSM primes and allowed molecules."""
    
    def __init__(self, allowed_primes: Optional[Set[str]] = None, 
                 allowed_molecules: Optional[Set[str]] = None):
        """Initialize the NSM constrained decoder.
        
        Args:
            allowed_primes: Set of allowed NSM primes (defaults to all 65)
            allowed_molecules: Set of allowed NSM molecules (defaults to predefined set)
        """
        self.allowed_primes = allowed_primes or ALL_NSM_PRIMES
        self.allowed_molecules = allowed_molecules or set(NSM_MOLECULES.keys())
        self.grammar_rules = self._build_grammar_rules()
        
    def _build_grammar_rules(self) -> List[str]:
        """Build NSM grammar rules for validation."""
        return [
            "SENTENCE -> SUBJECT PREDICATE",
            "SENTENCE -> SUBJECT VERB OBJECT",
            "SENTENCE -> SUBJECT VERB",
            "SUBJECT -> I | YOU | SOMEONE | PEOPLE | SOMETHING | THING | BODY",
            "PREDICATE -> THINK | KNOW | WANT | FEEL | SEE | HEAR",
            "VERB -> DO | HAPPEN | MOVE | TOUCH | LIVE | DIE | SAY",
            "OBJECT -> THIS | THAT | SOMETHING | THING | WORDS",
            "MODIFIER -> VERY | MORE | BIG | SMALL | GOOD | BAD",
            "CONNECTOR -> BECAUSE | IF | WHEN | WHERE | NOT",
            "QUANTIFIER -> ALL | MANY | SOME | FEW | MUCH | LITTLE"
        ]
    
    def validate_nsm_compliance(self, text: str) -> Tuple[bool, List[str], float]:
        """Validate if text complies with NSM constraints.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_compliant, violations, compliance_score)
        """
        words = text.upper().split()
        violations = []
        used_primes = []
        used_molecules = []
        
        # Check each word
        for word in words:
            if word in self.allowed_primes:
                used_primes.append(word)
            else:
                violations.append(f"Non-NSM word: {word}")
        
        # Check for molecules
        for i in range(len(words) - 1):
            molecule = f"{words[i]} {words[i+1]}"
            if molecule in self.allowed_molecules:
                used_molecules.append(molecule)
        
        # Calculate compliance score
        total_words = len(words)
        nsm_words = len(used_primes)
        compliance_score = nsm_words / total_words if total_words > 0 else 0.0
        
        is_compliant = len(violations) == 0
        
        return is_compliant, violations, compliance_score
    
    def generate_constrained_text(self, prompt: str, max_length: int = 20) -> NSMGenerationResult:
        """Generate text constrained to NSM primes and molecules.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            
        Returns:
            NSMGenerationResult with generated text and metadata
        """
        # Extract NSM primes from prompt
        prompt_primes = self._extract_primes_from_text(prompt)
        
        # Generate constrained text
        generated_words = []
        used_primes = []
        used_molecules = []
        
        # Start with a subject
        subjects = ["I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING"]
        if prompt_primes:
            # Use primes from prompt if available
            available_subjects = [p for p in prompt_primes if p in subjects]
            if available_subjects:
                generated_words.append(available_subjects[0])
                used_primes.append(available_subjects[0])
            else:
                generated_words.append("I")
                used_primes.append("I")
        else:
            generated_words.append("I")
            used_primes.append("I")
        
        # Add predicate or verb
        predicates = ["THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR"]
        verbs = ["DO", "HAPPEN", "MOVE", "SAY"]
        
        if prompt_primes:
            available_predicates = [p for p in prompt_primes if p in predicates + verbs]
            if available_predicates:
                generated_words.append(available_predicates[0])
                used_primes.append(available_predicates[0])
            else:
                generated_words.append("THINK")
                used_primes.append("THINK")
        else:
            generated_words.append("THINK")
            used_primes.append("THINK")
        
        # Add object or modifier
        objects = ["THIS", "THAT", "SOMETHING", "THING", "WORDS"]
        modifiers = ["VERY", "MORE", "BIG", "SMALL", "GOOD", "BAD"]
        
        if prompt_primes:
            available_objects = [p for p in prompt_primes if p in objects + modifiers]
            if available_objects:
                generated_words.append(available_objects[0])
                used_primes.append(available_objects[0])
            else:
                generated_words.append("THIS")
                used_primes.append("THIS")
        else:
            generated_words.append("THIS")
            used_primes.append("THIS")
        
        # Check for molecules
        for i in range(len(generated_words) - 1):
            molecule = f"{generated_words[i]} {generated_words[i+1]}"
            if molecule in self.allowed_molecules:
                used_molecules.append(molecule)
        
        generated_text = " ".join(generated_words)
        
        # Validate compliance
        is_compliant, violations, compliance_score = self.validate_nsm_compliance(generated_text)
        
        return NSMGenerationResult(
            text=generated_text,
            used_primes=used_primes,
            used_molecules=used_molecules,
            confidence=compliance_score,
            grammar_violations=violations,
            nsm_compliance=compliance_score
        )
    
    def _extract_primes_from_text(self, text: str) -> List[str]:
        """Extract NSM primes from text."""
        words = text.upper().split()
        primes = []
        for word in words:
            if word in self.allowed_primes:
                primes.append(word)
        return primes
    
    def generate_with_grammar_rules(self, prompt: str, grammar_focus: str = "balanced") -> NSMGenerationResult:
        """Generate text with specific grammar focus.
        
        Args:
            prompt: Input prompt
            grammar_focus: Focus area ("balanced", "subject_verb", "logical", "temporal")
            
        Returns:
            NSMGenerationResult with generated text
        """
        if grammar_focus == "subject_verb":
            # Focus on subject-verb-object patterns
            patterns = [
                ["I", "THINK", "THIS"],
                ["YOU", "KNOW", "THAT"],
                ["SOMEONE", "DOES", "SOMETHING"],
                ["PEOPLE", "WANT", "THINGS"]
            ]
        elif grammar_focus == "logical":
            # Focus on logical operators
            patterns = [
                ["IF", "THIS", "THEN", "THAT"],
                ["BECAUSE", "THIS", "THAT", "HAPPENS"],
                ["NOT", "THIS", "BUT", "THAT"]
            ]
        elif grammar_focus == "temporal":
            # Focus on temporal expressions
            patterns = [
                ["BEFORE", "THIS", "THAT", "HAPPENS"],
                ["AFTER", "THIS", "THEN", "THAT"],
                ["WHEN", "THIS", "THEN", "THAT"]
            ]
        else:
            # Balanced approach
            patterns = [
                ["I", "THINK", "THIS"],
                ["YOU", "KNOW", "THAT"],
                ["IF", "THIS", "THEN", "THAT"],
                ["BECAUSE", "THIS", "THAT"]
            ]
        
        # Select pattern based on prompt
        prompt_primes = self._extract_primes_from_text(prompt)
        selected_pattern = patterns[0]  # Default to first pattern
        
        # Try to match pattern with prompt primes
        for pattern in patterns:
            if any(prime in prompt_primes for prime in pattern):
                selected_pattern = pattern
                break
        
        # Generate text from pattern
        generated_words = []
        used_primes = []
        
        for word in selected_pattern:
            if word in self.allowed_primes:
                generated_words.append(word)
                used_primes.append(word)
            else:
                # Replace with closest NSM prime
                if word in ["THEN", "THAT"]:
                    generated_words.append("THIS")
                    used_primes.append("THIS")
                elif word in ["DOES", "HAPPENS"]:
                    generated_words.append("DO")
                    used_primes.append("DO")
                else:
                    generated_words.append("THIS")
                    used_primes.append("THIS")
        
        generated_text = " ".join(generated_words)
        
        # Check for molecules
        used_molecules = []
        for i in range(len(generated_words) - 1):
            molecule = f"{generated_words[i]} {generated_words[i+1]}"
            if molecule in self.allowed_molecules:
                used_molecules.append(molecule)
        
        # Validate compliance
        is_compliant, violations, compliance_score = self.validate_nsm_compliance(generated_text)
        
        return NSMGenerationResult(
            text=generated_text,
            used_primes=used_primes,
            used_molecules=used_molecules,
            confidence=compliance_score,
            grammar_violations=violations,
            nsm_compliance=compliance_score
        )

class NSMProofTrace:
    """Proof trace for NSM generation showing violations and corrections."""
    
    def __init__(self):
        self.steps = []
        self.violations = []
        self.corrections = []
        
    def add_step(self, step: str, details: Dict[str, Any]):
        """Add a step to the proof trace."""
        self.steps.append({
            "step": step,
            "details": details,
            "timestamp": len(self.steps)
        })
    
    def add_violation(self, violation: str, correction: str):
        """Add a violation and its correction."""
        self.violations.append(violation)
        self.corrections.append(correction)
    
    def get_trace(self) -> Dict[str, Any]:
        """Get the complete proof trace."""
        return {
            "steps": self.steps,
            "violations": self.violations,
            "corrections": self.corrections,
            "total_steps": len(self.steps),
            "total_violations": len(self.violations)
        }

def create_nsm_constraint(allowed_primes: Optional[Set[str]] = None,
                         allowed_molecules: Optional[Set[str]] = None,
                         max_length: int = 50) -> NSMConstraint:
    """Create an NSM constraint configuration.
    
    Args:
        allowed_primes: Set of allowed NSM primes
        allowed_molecules: Set of allowed NSM molecules
        max_length: Maximum length of generated text
        
    Returns:
        NSMConstraint configuration
    """
    return NSMConstraint(
        allowed_primes=allowed_primes or ALL_NSM_PRIMES,
        allowed_molecules=allowed_molecules or set(NSM_MOLECULES.keys()),
        grammar_rules=[
            "SENTENCE -> SUBJECT PREDICATE",
            "SENTENCE -> SUBJECT VERB OBJECT",
            "SUBJECT -> I | YOU | SOMEONE | PEOPLE | SOMETHING | THING | BODY",
            "PREDICATE -> THINK | KNOW | WANT | FEEL | SEE | HEAR",
            "VERB -> DO | HAPPEN | MOVE | TOUCH | LIVE | DIE | SAY",
            "OBJECT -> THIS | THAT | SOMETHING | THING | WORDS"
        ],
        max_length=max_length
    )

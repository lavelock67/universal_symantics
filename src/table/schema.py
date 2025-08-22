"""Schema definitions for the Periodic Table of Information Primitives.

This module defines the structure of primitives, their categories, operations,
and composition rules that form the foundation of the periodic table.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


class PrimitiveCategory(Enum):
    """Categories of information primitives."""
    
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    LOGICAL = "logical"
    QUANTITATIVE = "quantitative"
    STRUCTURAL = "structural"
    INFORMATIONAL = "informational"
    COGNITIVE = "cognitive"


@dataclass(frozen=True)
class PrimitiveSignature:
    """Signature defining a primitive's input/output structure."""
    
    arity: int  # Number of arguments
    input_types: List[str] = field(default_factory=list)
    output_type: str = "any"
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Primitive:
    """An information primitive in the periodic table.
    
    Attributes:
        name: Unique identifier for the primitive
        category: Category from PrimitiveCategory enum
        signature: Input/output structure definition
        inverse: Name of inverse primitive if it exists
        neutral: Neutral element for composition
        compose_rules: Rules for composing with other primitives
        description: Human-readable description
        examples: Example usage patterns
    """
    
    name: str
    category: PrimitiveCategory
    signature: PrimitiveSignature
    inverse: Optional[str] = None
    neutral: Optional[str] = None
    compose_rules: List[Dict[str, Any]] = field(default_factory=list)
    # Algebraic properties
    symmetric: bool = False
    transitive: bool = False
    antisymmetric: bool = False
    description: str = ""
    examples: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate primitive after initialization."""
        if not self.name:
            raise ValueError("Primitive name cannot be empty")
        if self.arity < 0:
            raise ValueError("Arity must be non-negative")
    
    @property
    def arity(self) -> int:
        """Get the arity of this primitive."""
        return self.signature.arity
    
    def can_compose_with(self, other: "Primitive") -> bool:
        """Check if this primitive can compose with another.
        
        Args:
            other: The primitive to check composition with
            
        Returns:
            True if composition is possible, False otherwise
        """
        # Check if there's a specific composition rule
        for rule in self.compose_rules:
            if rule.get("with") == other.name:
                return True
        
        # Default: check if output type matches input type
        if self.signature.output_type != "any" and other.signature.input_types:
            return self.signature.output_type in other.signature.input_types
        
        return True


@dataclass
class CompositionRule:
    """Rule for composing two primitives.
    
    Attributes:
        left: Name of left primitive
        right: Name of right primitive
        result: Name of resulting primitive or composition pattern
        conditions: Conditions that must be met for composition
        description: Human-readable description of the rule
    """
    
    left: str
    right: str
    result: Union[str, List[str]]
    conditions: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class PeriodicTable:
    """The complete periodic table of information primitives.
    
    Attributes:
        primitives: Dictionary mapping primitive names to Primitive objects
        composition_rules: List of composition rules
        categories: Set of all categories present
        version: Version of the periodic table
    """
    
    primitives: Dict[str, Primitive] = field(default_factory=dict)
    composition_rules: List[CompositionRule] = field(default_factory=list)
    categories: Set[PrimitiveCategory] = field(default_factory=set)
    version: str = "0.1.0"
    
    def add_primitive(self, primitive: Primitive) -> None:
        """Add a primitive to the table.
        
        Args:
            primitive: The primitive to add
        """
        if primitive.name in self.primitives:
            raise ValueError(f"Primitive {primitive.name} already exists")
        
        self.primitives[primitive.name] = primitive
        self.categories.add(primitive.category)
    
    def get_primitives_by_category(self, category: PrimitiveCategory) -> List[Primitive]:
        """Get all primitives in a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of primitives in the category
        """
        return [p for p in self.primitives.values() if p.category == category]
    
    def get_primitive(self, name: str) -> Optional[Primitive]:
        """Get a primitive by name.
        
        Args:
            name: Name of the primitive
            
        Returns:
            The primitive if found, None otherwise
        """
        return self.primitives.get(name)
    
    def add_composition_rule(self, rule: CompositionRule) -> None:
        """Add a composition rule to the table.
        
        Args:
            rule: The composition rule to add
        """
        self.composition_rules.append(rule)
    
    def find_composition_rule(self, left: str, right: str) -> Optional[CompositionRule]:
        """Find a composition rule for two primitives.
        
        Args:
            left: Name of left primitive
            right: Name of right primitive
            
        Returns:
            The composition rule if found, None otherwise
        """
        for rule in self.composition_rules:
            if rule.left == left and rule.right == right:
                return rule
        return None
    
    def get_inverse_pairs(self) -> List[Tuple[str, str]]:
        """Get all inverse primitive pairs.
        
        Returns:
            List of (primitive, inverse) pairs
        """
        pairs = []
        for primitive in self.primitives.values():
            if primitive.inverse and primitive.inverse in self.primitives:
                # Avoid duplicates
                if not any(p[1] == primitive.name for p in pairs):
                    pairs.append((primitive.name, primitive.inverse))
        return pairs
    
    def validate(self) -> List[str]:
        """Validate the periodic table for consistency.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check that all inverse references exist
        for primitive in self.primitives.values():
            if primitive.inverse and primitive.inverse not in self.primitives:
                errors.append(f"Primitive {primitive.name} references non-existent inverse {primitive.inverse}")
        
        # Check that all neutral references exist
        for primitive in self.primitives.values():
            if primitive.neutral and primitive.neutral not in self.primitives:
                errors.append(f"Primitive {primitive.name} references non-existent neutral {primitive.neutral}")
        
        # Check composition rules reference existing primitives
        for rule in self.composition_rules:
            if rule.left not in self.primitives:
                errors.append(f"Composition rule references non-existent primitive {rule.left}")
            if rule.right not in self.primitives:
                errors.append(f"Composition rule references non-existent primitive {rule.right}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the periodic table to a dictionary representation.
        
        Returns:
            Dictionary representation of the periodic table
        """
        return {
            "version": self.version,
            "categories": [cat.value for cat in self.categories],
            "primitives": [
                {
                    "name": p.name,
                    "kind": p.category.value,
                    "signature": {
                        "arity": p.signature.arity,
                        "input_types": p.signature.input_types,
                        "output_type": p.signature.output_type,
                        "constraints": p.signature.constraints,
                    },
                    "inverse": p.inverse,
                    "neutral": p.neutral,
                    "compose_rules": p.compose_rules,
                    "symmetric": p.symmetric,
                    "transitive": p.transitive,
                    "antisymmetric": p.antisymmetric,
                    "description": p.description,
                    "examples": p.examples,
                }
                for p in self.primitives.values()
            ],
            "algebra": {
                "compose": [
                    [rule.left, rule.right, rule.result]
                    if isinstance(rule.result, str)
                    else [rule.left, rule.right] + rule.result
                    for rule in self.composition_rules
                ],
                "inverse_pairs": self.get_inverse_pairs(),
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeriodicTable":
        """Create a periodic table from a dictionary representation.
        
        Args:
            data: Dictionary representation of the periodic table
            
        Returns:
            PeriodicTable instance
        """
        table = cls(version=data.get("version", "0.1.0"))
        
        # Add primitives
        for p_data in data.get("primitives", []):
            signature = PrimitiveSignature(
                arity=p_data["signature"]["arity"],
                input_types=p_data["signature"].get("input_types", []),
                output_type=p_data["signature"].get("output_type", "any"),
                constraints=p_data["signature"].get("constraints", {}),
            )
            
            primitive = Primitive(
                name=p_data["name"],
                category=PrimitiveCategory(p_data["kind"]),
                signature=signature,
                inverse=p_data.get("inverse"),
                neutral=p_data.get("neutral"),
                compose_rules=p_data.get("compose_rules", []),
                symmetric=p_data.get("symmetric", False),
                transitive=p_data.get("transitive", False),
                antisymmetric=p_data.get("antisymmetric", False),
                description=p_data.get("description", ""),
                examples=p_data.get("examples", []),
            )
            table.add_primitive(primitive)
        
        # Add composition rules
        for rule_data in data.get("algebra", {}).get("compose", []):
            if len(rule_data) >= 3:
                rule = CompositionRule(
                    left=rule_data[0],
                    right=rule_data[1],
                    result=rule_data[2] if len(rule_data) == 3 else rule_data[2:],
                )
                table.add_composition_rule(rule)
        
        return table

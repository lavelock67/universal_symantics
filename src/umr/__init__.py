"""Uniform Meaning Representation (UMR) integration module.

This module provides parsing, generation, and evaluation capabilities for UMR,
a graph-based semantic representation that can serve as an interlingual baseline
for cross-language primitive detection and translation.
"""

from .parser import UMRParser
from .generator import UMRGenerator
from .evaluator import UMREvaluator
from .graph import UMRGraph

__all__ = [
    "UMRParser",
    "UMRGenerator", 
    "UMREvaluator",
    "UMRGraph"
]

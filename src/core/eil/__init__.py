"""
EIL (Executable Interlingua) Core Module

This module implements the irreducible core for universal translation:
1. Semantic extractor → EIL graph
2. EIL graph validator  
3. Decision layer (router)
"""

from .graph import EILGraph, EILNode, EILRelation
from .validator import EILValidator
from .router import EILRouter
from .extractor import EILExtractor
from .realizer import EILRealizer

__all__ = [
    'EILGraph', 'EILNode', 'EILRelation',
    'EILValidator', 'EILRouter', 'EILExtractor', 'EILRealizer'
]


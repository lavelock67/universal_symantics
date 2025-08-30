"""
NLP Components package.

Contains SpaCy pipeline components for the Universal Translator.
"""

from .mwe_normalizer import MWENormalizer, create_mwe_normalizer

__all__ = ["MWENormalizer", "create_mwe_normalizer"]

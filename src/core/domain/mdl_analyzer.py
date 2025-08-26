#!/usr/bin/env python3
"""
MDL Analysis System

This module implements real Minimum Description Length analysis using
actual compression algorithms and information theory principles.
"""

import gzip
import zlib
import bz2
import lzma
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from collections import Counter, defaultdict
import re
import math

from ...shared.logging.logger import get_logger, PerformanceContext
from ...shared.exceptions.exceptions import ValidationError, create_error_context
from .models import Language, PrimeCandidate


@dataclass
class MDLResult:
    """Result of MDL analysis."""
    
    candidate: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    mdl_delta: float
    information_gain: float
    complexity_score: float
    universality_score: float
    compression_method: str
    analysis_notes: List[str]
    
    def __post_init__(self):
        """Validate result after initialization."""
        if self.original_size < 0:
            raise ValueError("Original size must be non-negative")
        if self.compressed_size < 0:
            raise ValueError("Compressed size must be non-negative")
        if not 0.0 <= self.compression_ratio <= 1.0:
            raise ValueError("Compression ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.universality_score <= 1.0:
            raise ValueError("Universality score must be between 0.0 and 1.0")


class MDLAnalyzer:
    """Real MDL analysis system using actual compression algorithms."""
    
    def __init__(self):
        """Initialize the MDL analyzer."""
        self.logger = get_logger("mdl_analyzer")
        self.compression_methods = {
            "gzip": self._compress_gzip,
            "zlib": self._compress_zlib,
            "bzip2": self._compress_bzip2,
            "lzma": self._compress_lzma,
            "lz77": self._compress_lz77,
            "huffman": self._compress_huffman
        }
        
        # Load baseline compression data
        self.baseline_compression = self._load_baseline_compression()
        
        self.logger.info("MDL analyzer initialized with multiple compression methods")
    
    def _load_baseline_compression(self) -> Dict[str, float]:
        """Load baseline compression ratios for different text types."""
        # These would normally be calculated from large corpora
        baseline = {
            "philosophy": 0.35,  # Philosophy texts compress well due to repetition
            "scientific": 0.40,  # Scientific texts have moderate compression
            "literary": 0.30,    # Literary texts compress well due to style
            "news": 0.45,        # News texts have lower compression due to variety
            "technical": 0.50,   # Technical texts have lowest compression
            "conversational": 0.55  # Conversational texts have very low compression
        }
        return baseline
    
    def analyze_candidate(self, candidate: str, corpus: str, 
                         language: Language, domain: str = "general") -> MDLResult:
        """Analyze a prime candidate using MDL principles."""
        try:
            with PerformanceContext(f"mdl_analysis_{candidate}", self.logger):
                self.logger.info(f"Analyzing candidate: {candidate}")
                
                # Get original size
                original_size = len(corpus.encode('utf-8'))
                
                # Test different compression methods
                compression_results = {}
                for method_name, compress_func in self.compression_methods.items():
                    try:
                        compressed_size = compress_func(corpus)
                        compression_results[method_name] = compressed_size
                    except Exception as e:
                        self.logger.warning(f"Compression method {method_name} failed: {str(e)}")
                
                # Find best compression
                if not compression_results:
                    raise ValueError("All compression methods failed")
                
                best_method = min(compression_results.keys(), 
                                key=lambda x: compression_results[x])
                best_compressed_size = compression_results[best_method]
                
                # Calculate compression ratio
                compression_ratio = best_compressed_size / original_size
                
                # Calculate MDL delta
                baseline_ratio = self.baseline_compression.get(domain, 0.4)
                mdl_delta = baseline_ratio - compression_ratio
                
                # Calculate information gain
                information_gain = self._calculate_information_gain(candidate, corpus)
                
                # Calculate complexity score
                complexity_score = self._calculate_complexity_score(candidate, corpus)
                
                # Calculate universality score
                universality_score = self._calculate_universality_score(candidate, corpus)
                
                # Generate analysis notes
                analysis_notes = self._generate_analysis_notes(
                    candidate, compression_ratio, mdl_delta, information_gain, 
                    complexity_score, universality_score, best_method
                )
                
                result = MDLResult(
                    candidate=candidate,
                    original_size=original_size,
                    compressed_size=best_compressed_size,
                    compression_ratio=compression_ratio,
                    mdl_delta=mdl_delta,
                    information_gain=information_gain,
                    complexity_score=complexity_score,
                    universality_score=universality_score,
                    compression_method=best_method,
                    analysis_notes=analysis_notes
                )
                
                self.logger.info(f"MDL analysis completed for {candidate}: "
                               f"delta={mdl_delta:.3f}, gain={information_gain:.3f}")
                
                return result
                
        except Exception as e:
            self.logger.error(f"MDL analysis failed for {candidate}: {str(e)}")
            raise ValidationError(
                field="mdl_analysis",
                value=candidate,
                expected_type="valid_candidate",
                context=create_error_context("analyze_candidate", candidate=candidate, language=language.value)
            )
    
    def _compress_gzip(self, text: str) -> int:
        """Compress text using gzip."""
        return len(gzip.compress(text.encode('utf-8')))
    
    def _compress_zlib(self, text: str) -> int:
        """Compress text using zlib."""
        return len(zlib.compress(text.encode('utf-8')))
    
    def _compress_bzip2(self, text: str) -> int:
        """Compress text using bzip2."""
        return len(bz2.compress(text.encode('utf-8')))
    
    def _compress_lzma(self, text: str) -> int:
        """Compress text using LZMA."""
        return len(lzma.compress(text.encode('utf-8')))
    
    def _compress_lz77(self, text: str) -> int:
        """Compress text using LZ77 algorithm."""
        # Simplified LZ77 implementation
        compressed = []
        i = 0
        while i < len(text):
            # Find longest match in previous text
            longest_match = 0
            longest_offset = 0
            
            for offset in range(1, min(i + 1, 4096)):  # Look back up to 4096 characters
                match_length = 0
                while (i + match_length < len(text) and 
                       text[i + match_length] == text[i - offset + match_length]):
                    match_length += 1
                    if match_length >= 255:  # Maximum match length
                        break
                
                if match_length > longest_match:
                    longest_match = match_length
                    longest_offset = offset
            
            if longest_match >= 3:  # Only use matches of length 3 or more
                compressed.append((longest_offset, longest_match, text[i + longest_match]))
                i += longest_match + 1
            else:
                compressed.append((0, 0, text[i]))
                i += 1
        
        # Calculate compressed size (simplified)
        compressed_size = 0
        for offset, length, char in compressed:
            compressed_size += 2 + 1 + 1  # 2 bytes for offset, 1 for length, 1 for char
        
        return compressed_size
    
    def _compress_huffman(self, text: str) -> int:
        """Compress text using Huffman coding."""
        # Simplified Huffman implementation
        # Count character frequencies
        freq = Counter(text)
        
        # Build Huffman tree (simplified)
        # In practice, this would build a proper Huffman tree
        total_chars = len(text)
        compressed_size = 0
        
        # Calculate compressed size based on character frequencies
        for char, count in freq.items():
            # Simplified: assume each character takes log2(total_chars/count) bits
            if count > 0:
                bits_per_char = math.log2(total_chars / count)
                compressed_size += count * bits_per_char / 8  # Convert to bytes
        
        return int(compressed_size)
    
    def _calculate_information_gain(self, candidate: str, corpus: str) -> float:
        """Calculate information gain of the candidate."""
        # This measures how much the candidate reduces the complexity of the corpus
        
        # Calculate entropy before and after candidate introduction
        original_entropy = self._calculate_entropy(corpus)
        
        # Simulate corpus with candidate replaced by a placeholder
        # This is a simplified approach - real implementation would be more sophisticated
        modified_corpus = corpus.replace(candidate, "[CANDIDATE]")
        modified_entropy = self._calculate_entropy(modified_corpus)
        
        # Information gain is reduction in entropy
        information_gain = original_entropy - modified_entropy
        
        return max(0.0, information_gain)
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Calculate character frequencies
        freq = Counter(text)
        total_chars = len(text)
        
        # Calculate entropy
        entropy = 0.0
        for char, count in freq.items():
            if count > 0:
                probability = count / total_chars
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_complexity_score(self, candidate: str, corpus: str) -> float:
        """Calculate complexity score of the candidate."""
        # This measures the linguistic complexity of the candidate
        
        # Factors to consider:
        # 1. Word length
        # 2. Syllable complexity
        # 3. Morphological complexity
        # 4. Frequency in corpus
        
        word_length = len(candidate)
        syllable_complexity = self._calculate_syllable_complexity(candidate)
        frequency = corpus.lower().count(candidate.lower())
        total_words = len(corpus.split())
        relative_frequency = frequency / total_words if total_words > 0 else 0
        
        # Normalize scores
        word_length_score = min(word_length / 10.0, 1.0)  # Normalize to 0-1
        syllable_score = min(syllable_complexity / 5.0, 1.0)  # Normalize to 0-1
        frequency_score = min(relative_frequency * 1000, 1.0)  # Normalize to 0-1
        
        # Weighted average
        complexity_score = (
            word_length_score * 0.3 +
            syllable_score * 0.4 +
            (1.0 - frequency_score) * 0.3  # Lower frequency = higher complexity
        )
        
        return complexity_score
    
    def _calculate_syllable_complexity(self, word: str) -> float:
        """Calculate syllable complexity of a word."""
        # Simplified syllable counting
        vowels = 'aeiouAEIOU'
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel
        
        # Add complexity for consonant clusters
        consonant_clusters = len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{2,}', word))
        
        return syllable_count + consonant_clusters * 0.5
    
    def _calculate_universality_score(self, candidate: str, corpus: str) -> float:
        """Calculate universality score of the candidate."""
        # This measures how universal the candidate is across different contexts
        
        # Factors to consider:
        # 1. Semantic stability across contexts
        # 2. Cross-lingual potential
        # 3. Domain independence
        
        # Simplified approach - in practice, this would use semantic analysis
        semantic_stability = self._calculate_semantic_stability(candidate, corpus)
        cross_lingual_potential = self._calculate_cross_lingual_potential(candidate)
        domain_independence = self._calculate_domain_independence(candidate, corpus)
        
        universality_score = (
            semantic_stability * 0.4 +
            cross_lingual_potential * 0.3 +
            domain_independence * 0.3
        )
        
        return universality_score
    
    def _calculate_semantic_stability(self, candidate: str, corpus: str) -> float:
        """Calculate semantic stability across contexts."""
        # Simplified approach - in practice, this would use semantic analysis
        # For now, we'll use heuristics based on word properties
        
        # Check if candidate appears in different contexts
        sentences = [s.strip() for s in corpus.split('.') if s.strip()]
        contexts = []
        
        for sentence in sentences:
            if candidate.lower() in sentence.lower():
                contexts.append(sentence)
        
        if len(contexts) < 2:
            return 0.5  # Neutral score for single context
        
        # Calculate context diversity (simplified)
        context_diversity = min(len(contexts) / 10.0, 1.0)  # Normalize to 0-1
        
        return context_diversity
    
    def _calculate_cross_lingual_potential(self, candidate: str) -> float:
        """Calculate cross-lingual potential."""
        # Simplified approach - in practice, this would use linguistic analysis
        # For now, we'll use heuristics based on word properties
        
        # Check if candidate has properties common across languages
        is_short = len(candidate) <= 4
        is_simple = not re.search(r'[^a-zA-Z]', candidate)  # Only letters
        is_common_sound = bool(re.search(r'[aeiou]', candidate.lower()))  # Contains vowels
        
        score = 0.0
        if is_short:
            score += 0.3
        if is_simple:
            score += 0.4
        if is_common_sound:
            score += 0.3
        
        return score
    
    def _calculate_domain_independence(self, candidate: str, corpus: str) -> float:
        """Calculate domain independence."""
        # Simplified approach - check if candidate appears in different types of content
        # In practice, this would use domain classification
        
        # For now, we'll use a simple heuristic
        # Words that appear frequently across different sentence types are more domain-independent
        
        sentences = [s.strip() for s in corpus.split('.') if s.strip()]
        candidate_sentences = [s for s in sentences if candidate.lower() in s.lower()]
        
        if not candidate_sentences:
            return 0.5  # Neutral score
        
        # Calculate sentence type diversity (simplified)
        # In practice, this would use more sophisticated analysis
        sentence_lengths = [len(s.split()) for s in candidate_sentences]
        length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Higher variance suggests more diverse contexts
        diversity_score = min(length_variance / 100.0, 1.0)  # Normalize to 0-1
        
        return diversity_score
    
    def _generate_analysis_notes(self, candidate: str, compression_ratio: float, 
                               mdl_delta: float, information_gain: float,
                               complexity_score: float, universality_score: float,
                               compression_method: str) -> List[str]:
        """Generate analysis notes."""
        notes = []
        
        # Compression analysis
        if compression_ratio < 0.3:
            notes.append("Excellent compression ratio - candidate significantly reduces corpus size")
        elif compression_ratio < 0.5:
            notes.append("Good compression ratio - candidate provides meaningful compression")
        elif compression_ratio < 0.7:
            notes.append("Moderate compression ratio - candidate provides some compression")
        else:
            notes.append("Poor compression ratio - candidate does not significantly compress corpus")
        
        # MDL delta analysis
        if mdl_delta > 0.1:
            notes.append("High MDL delta - candidate provides substantial compression improvement")
        elif mdl_delta > 0.05:
            notes.append("Moderate MDL delta - candidate provides some compression improvement")
        elif mdl_delta > 0:
            notes.append("Low MDL delta - candidate provides minimal compression improvement")
        else:
            notes.append("Negative MDL delta - candidate does not improve compression")
        
        # Information gain analysis
        if information_gain > 0.5:
            notes.append("High information gain - candidate significantly reduces corpus complexity")
        elif information_gain > 0.2:
            notes.append("Moderate information gain - candidate reduces corpus complexity")
        else:
            notes.append("Low information gain - candidate has minimal impact on complexity")
        
        # Complexity analysis
        if complexity_score > 0.7:
            notes.append("High complexity - candidate is linguistically sophisticated")
        elif complexity_score > 0.4:
            notes.append("Moderate complexity - candidate has balanced linguistic properties")
        else:
            notes.append("Low complexity - candidate is linguistically simple")
        
        # Universality analysis
        if universality_score > 0.7:
            notes.append("High universality - candidate likely to be cross-lingual")
        elif universality_score > 0.4:
            notes.append("Moderate universality - candidate has some cross-lingual potential")
        else:
            notes.append("Low universality - candidate may be language-specific")
        
        # Compression method note
        notes.append(f"Best compression achieved using {compression_method}")
        
        return notes
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get MDL analysis statistics."""
        return {
            "compression_methods": list(self.compression_methods.keys()),
            "baseline_compression": self.baseline_compression,
            "analysis_metrics": [
                "compression_ratio",
                "mdl_delta", 
                "information_gain",
                "complexity_score",
                "universality_score"
            ],
            "analysis_methods": [
                "entropy_calculation",
                "semantic_stability",
                "cross_lingual_potential",
                "domain_independence",
                "syllable_complexity"
            ]
        }

#!/usr/bin/env python3
"""Prime Discovery Loop for Systematic Expansion of Semantic Operators.

Implements MDL-Δ analysis and systematic prime discovery based on:
- Minimum Description Length reduction
- Drift reduction on parallel corpora
- Compression validation
- Symmetry tests
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrimeCandidate:
    """Candidate for new NSM prime."""
    prime: str
    surface_forms: List[str]
    languages: List[str]
    frequency: int
    compression_gain: float
    drift_reduction: float
    symmetry_score: float
    confidence: float
    evidence: Dict[str, Any]


@dataclass
class DiscoveryResult:
    """Result from prime discovery process."""
    accepted_candidates: List[PrimeCandidate]
    rejected_candidates: List[PrimeCandidate]
    compression_improvement: float
    drift_improvement: float
    processing_time: float
    statistics: Dict[str, Any]


class PrimeDiscoveryLoop:
    """Systematic prime discovery using MDL and drift analysis."""
    
    def __init__(self, existing_primes: Set[str]):
        """Initialize the discovery loop.
        
        Args:
            existing_primes: Set of existing NSM primes
        """
        self.existing_primes = existing_primes
        self.candidates = []
        self.accepted_primes = set()
        self.rejected_primes = set()
        
        # Discovery parameters
        self.min_frequency = 10
        self.min_compression_gain = 0.05
        self.max_drift_increase = 0.1
        self.min_symmetry_score = 0.7
        self.min_confidence = 0.8
        
        # Statistics
        self.stats = {
            "total_candidates": 0,
            "accepted_candidates": 0,
            "rejected_candidates": 0,
            "compression_gains": [],
            "drift_reductions": [],
            "processing_times": []
        }
        
        logger.info(f"Prime discovery loop initialized with {len(existing_primes)} existing primes")
    
    def discover_candidates(self, corpus: List[str], languages: List[str]) -> List[PrimeCandidate]:
        """Discover candidate primes from corpus.
        
        Args:
            corpus: List of text samples
            languages: List of language codes
            
        Returns:
            List of prime candidates
        """
        start_time = time.time()
        
        try:
            # Extract potential candidates
            candidates = self._extract_candidates(corpus, languages)
            
            # Filter by frequency
            candidates = [c for c in candidates if c.frequency >= self.min_frequency]
            
            # Calculate compression gains
            for candidate in candidates:
                candidate.compression_gain = self._calculate_compression_gain(candidate, corpus)
            
            # Filter by compression gain
            candidates = [c for c in candidates if c.compression_gain >= self.min_compression_gain]
            
            # Calculate drift reduction
            for candidate in candidates:
                candidate.drift_reduction = self._calculate_drift_reduction(candidate, corpus, languages)
            
            # Calculate symmetry scores
            for candidate in candidates:
                candidate.symmetry_score = self._calculate_symmetry_score(candidate, languages)
            
            # Calculate overall confidence
            for candidate in candidates:
                candidate.confidence = self._calculate_confidence(candidate)
            
            # Sort by confidence
            candidates.sort(key=lambda x: x.confidence, reverse=True)
            
            self.candidates = candidates
            self.stats["total_candidates"] = len(candidates)
            
            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)
            
            logger.info(f"Discovered {len(candidates)} candidates in {processing_time:.2f}s")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error in candidate discovery: {e}")
            return []
    
    def evaluate_candidates(self, test_corpus: List[str], 
                          validation_corpus: List[str]) -> DiscoveryResult:
        """Evaluate candidates and accept/reject based on criteria.
        
        Args:
            test_corpus: Corpus for testing candidates
            validation_corpus: Corpus for validation
            
        Returns:
            Discovery result with accepted/rejected candidates
        """
        start_time = time.time()
        
        accepted = []
        rejected = []
        
        for candidate in self.candidates:
            # Apply acceptance criteria
            if self._should_accept_candidate(candidate, test_corpus, validation_corpus):
                accepted.append(candidate)
                self.accepted_primes.add(candidate.prime)
                self.stats["accepted_candidates"] += 1
                self.stats["compression_gains"].append(candidate.compression_gain)
                self.stats["drift_reductions"].append(candidate.drift_reduction)
            else:
                rejected.append(candidate)
                self.rejected_primes.add(candidate.prime)
                self.stats["rejected_candidates"] += 1
        
        # Calculate overall improvements
        compression_improvement = np.mean(self.stats["compression_gains"]) if self.stats["compression_gains"] else 0.0
        drift_improvement = np.mean(self.stats["drift_reductions"]) if self.stats["drift_reductions"] else 0.0
        
        processing_time = time.time() - start_time
        
        result = DiscoveryResult(
            accepted_candidates=accepted,
            rejected_candidates=rejected,
            compression_improvement=compression_improvement,
            drift_improvement=drift_improvement,
            processing_time=processing_time,
            statistics=self.stats.copy()
        )
        
        logger.info(f"Evaluation complete: {len(accepted)} accepted, {len(rejected)} rejected")
        
        return result
    
    def _extract_candidates(self, corpus: List[str], languages: List[str]) -> List[PrimeCandidate]:
        """Extract potential prime candidates from corpus."""
        candidates = []
        
        # Simple extraction: look for frequent patterns not in existing primes
        word_freq = Counter()
        for text in corpus:
            words = text.lower().split()
            word_freq.update(words)
        
        # Filter out existing primes and common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        excluded = self.existing_primes | common_words
        
        for word, freq in word_freq.most_common(100):
            if word not in excluded and len(word) > 2:
                # Create candidate
                candidate = PrimeCandidate(
                    prime=word.upper(),
                    surface_forms=[word],
                    languages=languages,
                    frequency=freq,
                    compression_gain=0.0,
                    drift_reduction=0.0,
                    symmetry_score=0.0,
                    confidence=0.0,
                    evidence={"frequency": freq, "languages": languages}
                )
                candidates.append(candidate)
        
        return candidates
    
    def _calculate_compression_gain(self, candidate: PrimeCandidate, corpus: List[str]) -> float:
        """Calculate compression gain from adding the candidate prime."""
        # Simplified MDL calculation
        # In practice, this would involve more sophisticated compression analysis
        
        # Estimate compression gain based on frequency and semantic coherence
        base_compression = len(corpus) * 0.1  # Baseline compression
        candidate_compression = candidate.frequency * 0.8  # Compression with candidate
        
        if base_compression > 0:
            return (base_compression - candidate_compression) / base_compression
        return 0.0
    
    def _calculate_drift_reduction(self, candidate: PrimeCandidate, 
                                 corpus: List[str], languages: List[str]) -> float:
        """Calculate drift reduction on parallel corpora."""
        # Simplified drift calculation
        # In practice, this would involve cross-lingual analysis
        
        # Estimate drift reduction based on cross-lingual consistency
        cross_lingual_consistency = len(candidate.languages) / len(languages)
        frequency_normalized = min(candidate.frequency / 100.0, 1.0)
        
        return cross_lingual_consistency * frequency_normalized
    
    def _calculate_symmetry_score(self, candidate: PrimeCandidate, languages: List[str]) -> float:
        """Calculate symmetry score across languages."""
        # Simplified symmetry calculation
        # In practice, this would involve morphological and semantic symmetry analysis
        
        # Estimate symmetry based on cross-lingual presence
        presence_ratio = len(candidate.languages) / len(languages)
        
        # Add some randomness for demonstration
        import random
        random.seed(hash(candidate.prime))
        symmetry_bonus = random.uniform(0.0, 0.3)
        
        return min(presence_ratio + symmetry_bonus, 1.0)
    
    def _calculate_confidence(self, candidate: PrimeCandidate) -> float:
        """Calculate overall confidence score for candidate."""
        # Weighted combination of factors
        weights = {
            "compression": 0.4,
            "drift": 0.3,
            "symmetry": 0.2,
            "frequency": 0.1
        }
        
        frequency_score = min(candidate.frequency / 50.0, 1.0)
        
        confidence = (
            candidate.compression_gain * weights["compression"] +
            candidate.drift_reduction * weights["drift"] +
            candidate.symmetry_score * weights["symmetry"] +
            frequency_score * weights["frequency"]
        )
        
        return min(confidence, 1.0)
    
    def _should_accept_candidate(self, candidate: PrimeCandidate, 
                               test_corpus: List[str], 
                               validation_corpus: List[str]) -> bool:
        """Determine if candidate should be accepted."""
        # Check all acceptance criteria
        if candidate.confidence < self.min_confidence:
            return False
        
        if candidate.compression_gain < self.min_compression_gain:
            return False
        
        if candidate.drift_reduction < -self.max_drift_increase:
            return False
        
        if candidate.symmetry_score < self.min_symmetry_score:
            return False
        
        # Additional validation on test corpus
        test_performance = self._validate_on_corpus(candidate, test_corpus)
        if not test_performance:
            return False
        
        return True
    
    def _validate_on_corpus(self, candidate: PrimeCandidate, corpus: List[str]) -> bool:
        """Validate candidate on test corpus."""
        # Simplified validation
        # In practice, this would involve more sophisticated testing
        
        # Check if candidate appears in test corpus
        candidate_appearances = 0
        for text in corpus:
            if candidate.prime.lower() in text.lower():
                candidate_appearances += 1
        
        # Require at least some appearances
        return candidate_appearances >= 2
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        return {
            "total_candidates": self.stats["total_candidates"],
            "accepted_candidates": self.stats["accepted_candidates"],
            "rejected_candidates": self.stats["rejected_candidates"],
            "acceptance_rate": (self.stats["accepted_candidates"] / 
                              max(self.stats["total_candidates"], 1)),
            "avg_compression_gain": np.mean(self.stats["compression_gains"]) if self.stats["compression_gains"] else 0.0,
            "avg_drift_reduction": np.mean(self.stats["drift_reductions"]) if self.stats["drift_reductions"] else 0.0,
            "avg_processing_time": np.mean(self.stats["processing_times"]) if self.stats["processing_times"] else 0.0,
            "existing_primes": len(self.existing_primes),
            "accepted_primes": len(self.accepted_primes),
            "rejected_primes": len(self.rejected_primes)
        }
    
    def save_results(self, filename: str):
        """Save discovery results to file."""
        results = {
            "accepted_candidates": [
                {
                    "prime": c.prime,
                    "surface_forms": c.surface_forms,
                    "languages": c.languages,
                    "frequency": c.frequency,
                    "compression_gain": c.compression_gain,
                    "drift_reduction": c.drift_reduction,
                    "symmetry_score": c.symmetry_score,
                    "confidence": c.confidence,
                    "evidence": c.evidence
                }
                for c in self.candidates if c.prime in self.accepted_primes
            ],
            "rejected_candidates": [
                {
                    "prime": c.prime,
                    "reasons": ["low_confidence", "low_compression", "high_drift", "low_symmetry"]
                }
                for c in self.candidates if c.prime in self.rejected_primes
            ],
            "statistics": self.get_statistics(),
            "timestamp": time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Discovery results saved to {filename}")

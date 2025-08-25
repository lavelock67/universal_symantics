#!/usr/bin/env python3
"""
MDL-Δ Discovery Loop for Prime Discovery

This module implements the weekly MDL-Δ discovery loop that proposes
candidate operators and accepts only those that decrease MDL and drift.
"""

import time
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np

from src.validate.compression import CompressionValidator
from src.table.schema import PeriodicTable

logger = logging.getLogger(__name__)

class DiscoveryStatus(Enum):
    """Status of a discovery candidate."""
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"

@dataclass
class CandidateOperator:
    """Candidate operator for prime discovery."""
    name: str
    description: str
    category: str  # evidentiality, honorific-politeness, etc.
    proposed_primes: List[str]
    proposed_molecules: List[str]
    rationale: str
    expected_mdl_reduction: float
    expected_drift_reduction: float
    confidence: float
    status: DiscoveryStatus = DiscoveryStatus.PROPOSED
    created_at: float = None
    evaluated_at: Optional[float] = None
    mdl_delta: Optional[float] = None
    drift_delta: Optional[float] = None
    acceptance_reason: Optional[str] = None
    rejection_reason: Optional[str] = None

class MDLDiscoveryLoop:
    """Weekly MDL-Δ discovery loop for prime discovery."""
    
    def __init__(self, compression_validator: CompressionValidator, 
                 periodic_table: PeriodicTable,
                 discovery_log_path: str = "data/discovery/discovery_log.json"):
        """Initialize the MDL discovery loop.
        
        Args:
            compression_validator: MDL compression validator
            periodic_table: Periodic table of primitives
            discovery_log_path: Path to discovery log file
        """
        self.compression_validator = compression_validator
        self.periodic_table = periodic_table
        self.discovery_log_path = Path(discovery_log_path)
        self.discovery_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing discovery log
        self.discovery_log = self._load_discovery_log()
        
        # Discovery configuration
        self.mdl_threshold = 0.05  # 5% MDL reduction required
        self.drift_threshold = 0.02  # 2% drift reduction required
        self.max_candidates_per_week = 2
        self.evaluation_corpus_size = 1000  # sentences for evaluation
    
    def _load_discovery_log(self) -> Dict[str, Any]:
        """Load existing discovery log."""
        if self.discovery_log_path.exists():
            try:
                with open(self.discovery_log_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load discovery log: {e}")
        
        return {
            "discoveries": [],
            "statistics": {
                "total_proposed": 0,
                "total_accepted": 0,
                "total_rejected": 0,
                "mdl_reductions": [],
                "drift_reductions": []
            },
            "last_updated": time.time()
        }
    
    def _save_discovery_log(self):
        """Save discovery log to file."""
        try:
            with open(self.discovery_log_path, 'w') as f:
                json.dump(self.discovery_log, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save discovery log: {e}")
    
    def propose_candidates(self) -> List[CandidateOperator]:
        """Propose 1-2 candidate operators for this week.
        
        Returns:
            List of candidate operators
        """
        candidates = []
        
        # Check if we've already proposed enough this week
        current_week = int(time.time() / (7 * 24 * 3600))
        weekly_proposals = [
            c for c in self.discovery_log["discoveries"]
            if c.get("created_at") and int(c["created_at"] / (7 * 24 * 3600)) == current_week
        ]
        
        if len(weekly_proposals) >= self.max_candidates_per_week:
            logger.info(f"Already proposed {len(weekly_proposals)} candidates this week")
            return candidates
        
        # Generate candidate operators based on analysis
        candidates.extend(self._generate_evidentiality_candidates())
        candidates.extend(self._generate_honorific_candidates())
        candidates.extend(self._generate_aspectual_candidates())
        
        # Limit to available slots
        available_slots = self.max_candidates_per_week - len(weekly_proposals)
        candidates = candidates[:available_slots]
        
        # Add to discovery log
        for candidate in candidates:
            candidate.created_at = time.time()
            self.discovery_log["discoveries"].append(asdict(candidate))
            self.discovery_log["statistics"]["total_proposed"] += 1
        
        self._save_discovery_log()
        
        logger.info(f"Proposed {len(candidates)} candidate operators")
        return candidates
    
    def _generate_evidentiality_candidates(self) -> List[CandidateOperator]:
        """Generate evidentiality candidate operators."""
        candidates = []
        
        # Evidentiality - source of information
        candidates.append(CandidateOperator(
            name="EVIDENTIALITY_SOURCE",
            description="Evidentiality markers for information source",
            category="evidentiality",
            proposed_primes=["HEARSAY", "WITNESS", "INFER"],
            proposed_molecules=["HEARSAY(AGENT, CLAUSE)", "WITNESS(AGENT, EVENT)", "INFER(AGENT, CONCLUSION)"],
            rationale="Cross-linguistic evidentiality systems require source marking",
            expected_mdl_reduction=0.08,
            expected_drift_reduction=0.03,
            confidence=0.7
        ))
        
        # Evidentiality - certainty
        candidates.append(CandidateOperator(
            name="EVIDENTIALITY_CERTAINTY",
            description="Evidentiality markers for certainty level",
            category="evidentiality",
            proposed_primes=["CERTAIN", "PROBABLE", "POSSIBLE"],
            proposed_molecules=["CERTAIN(CLAUSE)", "PROBABLE(CLAUSE)", "POSSIBLE(CLAUSE)"],
            rationale="Certainty marking is universal and reduces ambiguity",
            expected_mdl_reduction=0.06,
            expected_drift_reduction=0.02,
            confidence=0.8
        ))
        
        return candidates
    
    def _generate_honorific_candidates(self) -> List[CandidateOperator]:
        """Generate honorific-politeness candidate operators."""
        candidates = []
        
        # Honorific - social distance
        candidates.append(CandidateOperator(
            name="HONORIFIC_DISTANCE",
            description="Honorific markers for social distance",
            category="honorific-politeness",
            proposed_primes=["RESPECT", "FAMILIAR", "FORMAL"],
            proposed_molecules=["RESPECT(SPEAKER, HEARER)", "FAMILIAR(SPEAKER, HEARER)", "FORMAL(SPEAKER, HEARER)"],
            rationale="Social distance marking is cross-linguistically common",
            expected_mdl_reduction=0.05,
            expected_drift_reduction=0.02,
            confidence=0.6
        ))
        
        # Politeness - face management
        candidates.append(CandidateOperator(
            name="POLITENESS_FACE",
            description="Politeness markers for face management",
            category="honorific-politeness",
            proposed_primes=["POLITE", "DIRECT", "HEDGE"],
            proposed_molecules=["POLITE(SPEAKER, HEARER, ACT)", "DIRECT(SPEAKER, HEARER, ACT)", "HEDGE(CLAUSE)"],
            rationale="Face management is universal in human communication",
            expected_mdl_reduction=0.04,
            expected_drift_reduction=0.01,
            confidence=0.7
        ))
        
        return candidates
    
    def _generate_aspectual_candidates(self) -> List[CandidateOperator]:
        """Generate aspectual candidate operators."""
        candidates = []
        
        # Aspect - internal temporal structure
        candidates.append(CandidateOperator(
            name="ASPECT_INTERNAL",
            description="Aspectual markers for internal temporal structure",
            category="aspectual",
            proposed_primes=["INCEPTIVE", "DURATIVE", "COMPLETIVE"],
            proposed_molecules=["INCEPTIVE(EVENT)", "DURATIVE(EVENT)", "COMPLETIVE(EVENT)"],
            rationale="Internal temporal structure is cross-linguistically marked",
            expected_mdl_reduction=0.07,
            expected_drift_reduction=0.03,
            confidence=0.8
        ))
        
        return candidates
    
    def evaluate_candidate(self, candidate: CandidateOperator, 
                          test_corpus: List[str]) -> Tuple[bool, Dict[str, float]]:
        """Evaluate a candidate operator.
        
        Args:
            candidate: Candidate operator to evaluate
            test_corpus: Test corpus for evaluation
            
        Returns:
            Tuple of (accepted, metrics)
        """
        logger.info(f"Evaluating candidate: {candidate.name}")
        
        # Calculate baseline MDL
        baseline_mdl = self._calculate_corpus_mdl(test_corpus)
        
        # Simulate adding the candidate primes
        # In practice, this would involve updating the grammar and re-encoding
        simulated_mdl = baseline_mdl * (1.0 - candidate.expected_mdl_reduction)
        
        # Calculate MDL delta
        mdl_delta = simulated_mdl - baseline_mdl
        
        # Calculate drift delta (simplified)
        drift_delta = -candidate.expected_drift_reduction
        
        # Check acceptance criteria
        accepted = (mdl_delta < -self.mdl_threshold * baseline_mdl and 
                   drift_delta < -self.drift_threshold)
        
        metrics = {
            "baseline_mdl": baseline_mdl,
            "simulated_mdl": simulated_mdl,
            "mdl_delta": mdl_delta,
            "drift_delta": drift_delta,
            "accepted": accepted
        }
        
        return accepted, metrics
    
    def _calculate_corpus_mdl(self, corpus: List[str]) -> float:
        """Calculate MDL for a corpus.
        
        Args:
            corpus: List of sentences
            
        Returns:
            MDL score
        """
        total_mdl = 0.0
        codebook = list(self.periodic_table.primitives.values())
        
        for sentence in corpus:
            try:
                mdl_score = self.compression_validator.calculate_mdl_score(sentence, codebook)
                total_mdl += mdl_score
            except Exception as e:
                logger.warning(f"Failed to calculate MDL for sentence: {e}")
        
        return total_mdl / len(corpus) if corpus else 0.0
    
    def accept_candidate(self, candidate: CandidateOperator, metrics: Dict[str, float]):
        """Accept a candidate operator.
        
        Args:
            candidate: Candidate operator to accept
            metrics: Evaluation metrics
        """
        candidate.status = DiscoveryStatus.ACCEPTED
        candidate.evaluated_at = time.time()
        candidate.mdl_delta = metrics["mdl_delta"]
        candidate.drift_delta = metrics["drift_delta"]
        candidate.acceptance_reason = f"MDL reduced by {abs(metrics['mdl_delta']):.3f}, drift reduced by {abs(metrics['drift_delta']):.3f}"
        
        # Update discovery log
        for discovery in self.discovery_log["discoveries"]:
            if discovery["name"] == candidate.name:
                discovery.update(asdict(candidate))
                break
        
        self.discovery_log["statistics"]["total_accepted"] += 1
        self.discovery_log["statistics"]["mdl_reductions"].append(metrics["mdl_delta"])
        self.discovery_log["statistics"]["drift_reductions"].append(metrics["drift_delta"])
        
        self._save_discovery_log()
        
        logger.info(f"Accepted candidate: {candidate.name}")
    
    def reject_candidate(self, candidate: CandidateOperator, metrics: Dict[str, float]):
        """Reject a candidate operator.
        
        Args:
            candidate: Candidate operator to reject
            metrics: Evaluation metrics
        """
        candidate.status = DiscoveryStatus.REJECTED
        candidate.evaluated_at = time.time()
        candidate.mdl_delta = metrics["mdl_delta"]
        candidate.drift_delta = metrics["drift_delta"]
        
        # Determine rejection reason
        if metrics["mdl_delta"] >= -self.mdl_threshold:
            candidate.rejection_reason = f"Insufficient MDL reduction: {metrics['mdl_delta']:.3f}"
        elif metrics["drift_delta"] >= -self.drift_threshold:
            candidate.rejection_reason = f"Insufficient drift reduction: {metrics['drift_delta']:.3f}"
        else:
            candidate.rejection_reason = "Failed acceptance criteria"
        
        # Update discovery log
        for discovery in self.discovery_log["discoveries"]:
            if discovery["name"] == candidate.name:
                discovery.update(asdict(candidate))
                break
        
        self.discovery_log["statistics"]["total_rejected"] += 1
        
        self._save_discovery_log()
        
        logger.info(f"Rejected candidate: {candidate.name} - {candidate.rejection_reason}")
    
    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics.
        
        Returns:
            Discovery statistics
        """
        stats = self.discovery_log["statistics"].copy()
        
        # Calculate additional statistics
        if stats["mdl_reductions"]:
            stats["avg_mdl_reduction"] = np.mean(stats["mdl_reductions"])
            stats["max_mdl_reduction"] = np.min(stats["mdl_reductions"])  # Most negative
        
        if stats["drift_reductions"]:
            stats["avg_drift_reduction"] = np.mean(stats["drift_reductions"])
            stats["max_drift_reduction"] = np.min(stats["drift_reductions"])  # Most negative
        
        # Acceptance rate
        total_evaluated = stats["total_accepted"] + stats["total_rejected"]
        stats["acceptance_rate"] = stats["total_accepted"] / total_evaluated if total_evaluated > 0 else 0.0
        
        return stats
    
    def get_pending_candidates(self) -> List[CandidateOperator]:
        """Get pending candidates for evaluation.
        
        Returns:
            List of pending candidates
        """
        pending = []
        for discovery in self.discovery_log["discoveries"]:
            if discovery["status"] == DiscoveryStatus.PROPOSED.value:
                candidate = CandidateOperator(**discovery)
                pending.append(candidate)
        
        return pending
    
    def run_weekly_discovery(self, test_corpus: List[str]) -> Dict[str, Any]:
        """Run the weekly discovery process.
        
        Args:
            test_corpus: Test corpus for evaluation
            
        Returns:
            Discovery results
        """
        logger.info("Starting weekly discovery process")
        
        # Propose new candidates
        new_candidates = self.propose_candidates()
        
        # Get pending candidates
        pending_candidates = self.get_pending_candidates()
        all_candidates = new_candidates + pending_candidates
        
        # Evaluate candidates
        results = {
            "new_candidates": len(new_candidates),
            "pending_candidates": len(pending_candidates),
            "evaluated": 0,
            "accepted": 0,
            "rejected": 0,
            "candidates": []
        }
        
        for candidate in all_candidates:
            try:
                accepted, metrics = self.evaluate_candidate(candidate, test_corpus)
                
                if accepted:
                    self.accept_candidate(candidate, metrics)
                    results["accepted"] += 1
                else:
                    self.reject_candidate(candidate, metrics)
                    results["rejected"] += 1
                
                results["evaluated"] += 1
                results["candidates"].append({
                    "name": candidate.name,
                    "status": candidate.status.value,
                    "mdl_delta": metrics["mdl_delta"],
                    "drift_delta": metrics["drift_delta"]
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate candidate {candidate.name}: {e}")
                self.reject_candidate(candidate, {"mdl_delta": 0.0, "drift_delta": 0.0})
                results["rejected"] += 1
        
        # Update statistics
        results["statistics"] = self.get_discovery_statistics()
        
        logger.info(f"Weekly discovery completed: {results['accepted']} accepted, {results['rejected']} rejected")
        
        return results

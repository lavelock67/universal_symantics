"""
EIL Service

Unified service implementing the irreducible core:
1. Semantic extractor → EIL graph
2. EIL graph validator  
3. Decision layer (router)
4. Realizer (optional)
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

from .graph import EILGraph
from .validator import EILValidator, ValidationResult
from .router import EILRouter, RouterResult, RouterDecision
from .extractor import EILExtractor, ExtractionResult
from .realizer import EILRealizer, RealizationResult
from src.core.domain.models import DetectionResult, Language

logger = logging.getLogger(__name__)


@dataclass
class EILProcessingResult:
    """Result of EIL processing pipeline."""
    source_graph: EILGraph
    validation_result: ValidationResult
    router_result: RouterResult
    target_graph: Optional[EILGraph] = None
    realization_result: Optional[RealizationResult] = None
    processing_times: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = {}
        if self.metadata is None:
            self.metadata = {}


class EILService:
    """Unified EIL service implementing the irreducible core."""
    
    def __init__(self):
        """Initialize the EIL service."""
        self.extractor = EILExtractor()
        self.validator = EILValidator()
        self.router = EILRouter()
        self.realizer = EILRealizer()
        
        # Configuration
        self.config = {
            "enable_realization": True,
            "enable_post_check": True,
            "max_processing_time": 30.0,  # seconds
            "confidence_threshold": 0.7
        }
    
    def process_text(self, 
                    text: str, 
                    source_language: str = "en",
                    target_language: Optional[str] = None,
                    detection_result: Optional[DetectionResult] = None) -> EILProcessingResult:
        """Process text through the complete EIL pipeline."""
        
        start_time = time.time()
        processing_times = {}
        
        try:
            # Step 1: Extract EIL graph
            extract_start = time.time()
            if detection_result:
                extraction_result = self.extractor.extract_from_detection(detection_result)
            else:
                # Fallback: create minimal detection result
                detection_result = DetectionResult(
                    primes=[],
                    mwes=[],
                    source_text=text,
                    language=Language(source_language),
                    confidence=0.5
                )
                extraction_result = self.extractor.extract_from_detection(detection_result)
            
            processing_times["extract"] = time.time() - extract_start
            
            # Step 2: Validate EIL graph
            validate_start = time.time()
            validation_result = self.validator.validate_graph(extraction_result.graph)
            processing_times["validate"] = time.time() - validate_start
            
            # Step 3: Route decision
            route_start = time.time()
            router_result = self.router.route(
                extraction_result.graph,
                validation_result,
                target_language or source_language
            )
            processing_times["route"] = time.time() - route_start
            
            # Step 4: Handle router decision
            target_graph = None
            realization_result = None
            
            if router_result.decision == RouterDecision.TRANSLATE:
                # Proceed with translation/realization
                if target_language and target_language != source_language:
                    realize_start = time.time()
                    realization_result = self.realizer.realize(
                        extraction_result.graph, 
                        target_language
                    )
                    processing_times["realize"] = time.time() - realize_start
                    
                    # Post-check if enabled
                    if self.config["enable_post_check"]:
                        post_check_start = time.time()
                        post_check_result = self._post_check(
                            extraction_result.graph,
                            realization_result,
                            source_language,
                            target_language
                        )
                        processing_times["post_check"] = time.time() - post_check_start
                        
                        # Update router result if post-check fails
                        if post_check_result and post_check_result.confidence < self.config["confidence_threshold"]:
                            router_result = self.router.route(
                                extraction_result.graph,
                                validation_result,
                                target_language,
                                round_trip_f1=post_check_result.confidence
                            )
            
            elif router_result.decision == RouterDecision.CLARIFY:
                # Add clarification metadata
                router_result.metadata["clarification_needed"] = True
                router_result.metadata["suggested_actions"] = self._suggest_clarifications(validation_result)
            
            elif router_result.decision == RouterDecision.REGENERATE:
                # Add regeneration metadata
                router_result.metadata["regeneration_needed"] = True
                router_result.metadata["suggested_improvements"] = self._suggest_improvements(extraction_result)
            
            elif router_result.decision == RouterDecision.ABSTAIN:
                # Add abstention metadata
                router_result.metadata["abstention_reason"] = router_result.reasoning
                router_result.metadata["recovery_suggestions"] = self._suggest_recovery(validation_result)
            
            # Calculate total processing time
            processing_times["total"] = time.time() - start_time
            
            return EILProcessingResult(
                source_graph=extraction_result.graph,
                validation_result=validation_result,
                router_result=router_result,
                target_graph=target_graph,
                realization_result=realization_result,
                processing_times=processing_times,
                metadata={
                    "extraction_method": extraction_result.extraction_method,
                    "extraction_confidence": extraction_result.confidence,
                    "source_language": source_language,
                    "target_language": target_language,
                    "config": self.config
                }
            )
            
        except Exception as e:
            logger.error(f"EIL processing failed: {str(e)}")
            processing_times["total"] = time.time() - start_time
            
            # Return error result
            return EILProcessingResult(
                source_graph=EILGraph(),
                validation_result=ValidationResult(
                    is_valid=False,
                    errors=[],
                    warnings=[],
                    info=[],
                    metrics={}
                ),
                router_result=RouterResult(
                    decision=RouterDecision.ABSTAIN,
                    confidence=0.0,
                    reasoning=f"Processing error: {str(e)}",
                    metadata={"error": str(e)}
                ),
                processing_times=processing_times,
                metadata={"error": str(e)}
            )
    
    def _post_check(self, 
                   source_graph: EILGraph, 
                   realization_result: RealizationResult,
                   source_language: str,
                   target_language: str) -> Optional[RealizationResult]:
        """Post-check by re-extracting from realized text and comparing graphs."""
        try:
            # Re-extract from realized text
            re_extraction_result = self.extractor.extract_from_llm(
                realization_result.text, 
                target_language
            )
            
            if re_extraction_result.confidence > 0:
                # Compare graphs (simplified comparison)
                similarity = self._compare_graphs(source_graph, re_extraction_result.graph)
                
                # Create post-check result
                post_check_result = RealizationResult(
                    text=realization_result.text,
                    confidence=similarity,
                    realization_method="post_checked",
                    metadata={
                        "original_confidence": realization_result.confidence,
                        "graph_similarity": similarity,
                        "post_check_method": "graph_comparison"
                    }
                )
                
                return post_check_result
            
            return None
            
        except Exception as e:
            logger.warning(f"Post-check failed: {str(e)}")
            return None
    
    def _compare_graphs(self, graph1: EILGraph, graph2: EILGraph) -> float:
        """Compare two EIL graphs and return similarity score."""
        try:
            # Simple node label comparison
            labels1 = {node.label for node in graph1.nodes.values()}
            labels2 = {node.label for node in graph2.nodes.values()}
            
            if not labels1 and not labels2:
                return 1.0
            elif not labels1 or not labels2:
                return 0.0
            
            intersection = labels1.intersection(labels2)
            union = labels1.union(labels2)
            
            return len(intersection) / len(union)
            
        except Exception:
            return 0.0
    
    def _suggest_clarifications(self, validation_result: ValidationResult) -> List[str]:
        """Suggest clarifications based on validation warnings."""
        suggestions = []
        
        for warning in validation_result.warnings:
            if warning.level.value == "scope":
                suggestions.append("Clarify scope attachment for quantifiers/negation")
            elif warning.level.value == "molecule":
                suggestions.append("Break down complex semantic molecules")
            elif warning.level.value == "depth":
                suggestions.append("Simplify deep semantic structures")
        
        return suggestions
    
    def _suggest_improvements(self, extraction_result: ExtractionResult) -> List[str]:
        """Suggest improvements for regeneration."""
        suggestions = []
        
        if extraction_result.confidence < 0.5:
            suggestions.append("Improve semantic extraction confidence")
        
        if extraction_result.metadata.get("prime_nodes", 0) < 2:
            suggestions.append("Extract more semantic primitives")
        
        if extraction_result.metadata.get("relations", 0) < 1:
            suggestions.append("Establish semantic relations between concepts")
        
        return suggestions
    
    def _suggest_recovery(self, validation_result: ValidationResult) -> List[str]:
        """Suggest recovery strategies for abstention."""
        suggestions = []
        
        error_count = len(validation_result.errors)
        warning_count = len(validation_result.warnings)
        
        if error_count > 5:
            suggestions.append("Fix critical validation errors")
        elif warning_count > 10:
            suggestions.append("Address validation warnings")
        else:
            suggestions.append("Review semantic structure")
        
        return suggestions
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update service configuration."""
        self.config.update(new_config)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "config": self.config,
            "validator_config": {
                "max_depth": self.validator.max_depth,
                "max_molecule_size": self.validator.max_molecule_size
            },
            "router_thresholds": self.router.language_thresholds
        }


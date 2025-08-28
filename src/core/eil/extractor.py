"""
EIL Extractor

Converts NSM detection results to EIL graphs.
Supports both UD-based and LLM-based extraction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

from .graph import EILGraph, EILNode, EILRelation, EILNodeType, EILRelationType
from src.core.domain.models import DetectionResult, NSMPrime, MWE, Language

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of EIL extraction."""
    graph: EILGraph
    confidence: float
    extraction_method: str
    metadata: Dict[str, Any]


class EILExtractor:
    """Extracts EIL graphs from NSM detection results."""
    
    def __init__(self):
        """Initialize the extractor."""
        # NSM prime to EIL node type mapping
        self.prime_to_node_type = {
            # Substantives
            "I": EILNodeType.ENTITY,
            "YOU": EILNodeType.ENTITY,
            "SOMEONE": EILNodeType.ENTITY,
            "PEOPLE": EILNodeType.ENTITY,
            "SOMETHING": EILNodeType.ENTITY,
            "THING": EILNodeType.ENTITY,
            "BODY": EILNodeType.ENTITY,
            
            # Mental predicates
            "THINK": EILNodeType.EVENT,
            "KNOW": EILNodeType.EVENT,
            "WANT": EILNodeType.EVENT,
            "FEEL": EILNodeType.EVENT,
            "SEE": EILNodeType.EVENT,
            "HEAR": EILNodeType.EVENT,
            "SAY": EILNodeType.EVENT,
            
            # Logical operators
            "NOT": EILNodeType.NEGATION,
            "BECAUSE": EILNodeType.LOGICAL,
            "IF": EILNodeType.LOGICAL,
            "MAYBE": EILNodeType.MODAL,
            
            # Temporal
            "WHEN": EILNodeType.TEMPORAL,
            "NOW": EILNodeType.TEMPORAL,
            "BEFORE": EILNodeType.TEMPORAL,
            "AFTER": EILNodeType.TEMPORAL,
            "A_LONG_TIME": EILNodeType.TEMPORAL,
            "A_SHORT_TIME": EILNodeType.TEMPORAL,
            "FOR_SOME_TIME": EILNodeType.TEMPORAL,
            
            # Spatial
            "WHERE": EILNodeType.SPATIAL,
            "HERE": EILNodeType.SPATIAL,
            "ABOVE": EILNodeType.SPATIAL,
            "BELOW": EILNodeType.SPATIAL,
            "NEAR": EILNodeType.SPATIAL,
            "FAR": EILNodeType.SPATIAL,
            "INSIDE": EILNodeType.SPATIAL,
            "BE_SOMEWHERE": EILNodeType.SPATIAL,
            "THERE_IS": EILNodeType.SPATIAL,
            
            # Quantifiers
            "ALL": EILNodeType.QUANTIFIER,
            "SOME": EILNodeType.QUANTIFIER,
            "MANY": EILNodeType.QUANTIFIER,
            "FEW": EILNodeType.QUANTIFIER,
            "ONE": EILNodeType.QUANTIFIER,
            "TWO": EILNodeType.QUANTIFIER,
            "THE_SAME": EILNodeType.QUANTIFIER,
            
            # Evaluators
            "GOOD": EILNodeType.PROPERTY,
            "BAD": EILNodeType.PROPERTY,
            "BIG": EILNodeType.PROPERTY,
            "SMALL": EILNodeType.PROPERTY,
            
            # Actions
            "DO": EILNodeType.EVENT,
            "HAPPEN": EILNodeType.EVENT,
            "MOVE": EILNodeType.EVENT,
            "TOUCH": EILNodeType.EVENT,
            "LIVE": EILNodeType.EVENT,
            "DIE": EILNodeType.EVENT,
            
            # Descriptors
            "THIS": EILNodeType.PROPERTY,
            "OTHER": EILNodeType.PROPERTY,
            "SAME": EILNodeType.PROPERTY,
            "DIFFERENT": EILNodeType.PROPERTY,
            
            # Intensifiers
            "VERY": EILNodeType.PROPERTY,
            "MORE": EILNodeType.PROPERTY,
            "LIKE": EILNodeType.PROPERTY,
            
            # Speech
            "WORDS": EILNodeType.PROPERTY,
            
            # Existence
            "BE_SOMEONE": EILNodeType.ENTITY,
        }
        
        # NSM prime to EIL relation type mapping
        self.prime_to_relation_type = {
            "NOT": EILRelationType.NOT,
            "BECAUSE": EILRelationType.CAUSE,
            "IF": EILRelationType.IF_THEN,
            "BEFORE": EILRelationType.BEFORE,
            "AFTER": EILRelationType.AFTER,
            "ALL": EILRelationType.ALL,
            "SOME": EILRelationType.SOME,
            "MUST": EILRelationType.MUST,
            "CAN": EILRelationType.CAN,
            "SHOULD": EILRelationType.SHOULD,
        }
    
    def extract_from_detection(self, detection_result: DetectionResult) -> ExtractionResult:
        """Extract EIL graph from NSM detection result."""
        try:
            graph = EILGraph()
            
            # Extract from NSM primes
            prime_nodes = self._extract_prime_nodes(detection_result.primes, graph)
            
            # Extract from MWEs
            mwe_nodes = self._extract_mwe_nodes(detection_result.mwes, graph)
            
            # Create relations
            relations = self._create_relations(prime_nodes, mwe_nodes, graph)
            
            # Add metadata
            graph.metadata = {
                "source_text": detection_result.source_text,
                "language": detection_result.language,
                "extraction_method": "nsm_detection",
                "prime_count": len(detection_result.primes),
                "mwe_count": len(detection_result.mwes)
            }
            
            # Calculate confidence
            confidence = self._calculate_extraction_confidence(detection_result)
            
            return ExtractionResult(
                graph=graph,
                confidence=confidence,
                extraction_method="nsm_detection",
                metadata={
                    "prime_nodes": len(prime_nodes),
                    "mwe_nodes": len(mwe_nodes),
                    "relations": len(relations)
                }
            )
            
        except Exception as e:
            logger.error(f"EIL extraction failed: {str(e)}")
            # Return empty graph on error
            return ExtractionResult(
                graph=EILGraph(),
                confidence=0.0,
                extraction_method="nsm_detection",
                metadata={"error": str(e)}
            )
    
    def _extract_prime_nodes(self, primes: List[NSMPrime], graph: EILGraph) -> Dict[str, str]:
        """Extract nodes from NSM primes."""
        prime_nodes = {}
        
        for prime in primes:
            # Determine node type
            node_type = self.prime_to_node_type.get(prime.text, EILNodeType.ENTITY)
            
            # Create node
            node = EILNode(
                label=prime.text,
                node_type=node_type,
                confidence=prime.confidence,
                source="NSM",
                properties={
                    "prime_type": prime.type.value,
                    "language": prime.language.value,
                    "universality_score": prime.universality_score
                }
            )
            
            # Add to graph
            node_id = graph.add_node(node)
            prime_nodes[prime.text] = node_id
        
        return prime_nodes
    
    def _extract_mwe_nodes(self, mwes: List[MWE], graph: EILGraph) -> Dict[str, str]:
        """Extract nodes from MWEs."""
        mwe_nodes = {}
        
        for mwe in mwes:
            # Create MWE node
            node = EILNode(
                label=mwe.text,
                node_type=EILNodeType.ENTITY,  # MWEs are typically entities
                confidence=mwe.confidence,
                source="MWE",
                properties={
                    "mwe_type": mwe.type.value,
                    "primes": mwe.primes,
                    "start": mwe.start,
                    "end": mwe.end
                }
            )
            
            # Add to graph
            node_id = graph.add_node(node)
            mwe_nodes[mwe.text] = node_id
        
        return mwe_nodes
    
    def _create_relations(self, prime_nodes: Dict[str, str], mwe_nodes: Dict[str, str], graph: EILGraph) -> List[str]:
        """Create relations between nodes."""
        relations = []
        
        # Create relations between primes and MWEs
        for mwe_text, mwe_node_id in mwe_nodes.items():
            mwe_node = graph.get_node(mwe_node_id)
            if mwe_node and "primes" in mwe_node.properties:
                for prime_text in mwe_node.properties["primes"]:
                    if prime_text in prime_nodes:
                        # Create relation from MWE to prime
                        relation = EILRelation(
                            source_id=mwe_node_id,
                            target_id=prime_nodes[prime_text],
                            relation_type=EILRelationType.AND,  # MWE contains prime
                            confidence=mwe_node.confidence,
                            properties={"relation_type": "mwe_contains_prime"}
                        )
                        relation_id = graph.add_relation(relation)
                        relations.append(relation_id)
        
        # Create logical relations between primes
        logical_primes = ["NOT", "BECAUSE", "IF", "BEFORE", "AFTER", "ALL", "SOME"]
        for prime_text in logical_primes:
            if prime_text in prime_nodes:
                # Find related primes to connect
                for other_prime_text, other_node_id in prime_nodes.items():
                    if other_prime_text != prime_text:
                        relation_type = self.prime_to_relation_type.get(prime_text, EILRelationType.AND)
                        
                        relation = EILRelation(
                            source_id=prime_nodes[prime_text],
                            target_id=other_node_id,
                            relation_type=relation_type,
                            confidence=0.8,  # Default confidence for logical relations
                            properties={"relation_type": "logical_connection"}
                        )
                        relation_id = graph.add_relation(relation)
                        relations.append(relation_id)
        
        return relations
    
    def _calculate_extraction_confidence(self, detection_result: DetectionResult) -> float:
        """Calculate confidence for the extraction."""
        if not detection_result.primes:
            return 0.0
        
        # Average confidence of primes
        prime_confidence = sum(prime.confidence for prime in detection_result.primes) / len(detection_result.primes)
        
        # MWE confidence if available
        mwe_confidence = 0.0
        if detection_result.mwes:
            mwe_confidence = sum(mwe.confidence for mwe in detection_result.mwes) / len(detection_result.mwes)
        
        # Weighted average
        if detection_result.mwes:
            return (prime_confidence * 0.7) + (mwe_confidence * 0.3)
        else:
            return prime_confidence
    
    def extract_from_llm(self, text: str, language: str = "en") -> ExtractionResult:
        """Extract EIL graph from LLM explication (placeholder for future implementation)."""
        # This would integrate with a language model for direct EIL extraction
        # For now, return empty result
        logger.warning("LLM extraction not yet implemented")
        
        return ExtractionResult(
            graph=EILGraph(),
            confidence=0.0,
            extraction_method="llm_explication",
            metadata={"error": "LLM extraction not implemented"}
        )
    
    def extract_from_ud(self, ud_parse: Dict[str, Any], language: str = "en") -> ExtractionResult:
        """Extract EIL graph from UD parse (placeholder for future implementation)."""
        # This would integrate with Universal Dependencies parsing
        # For now, return empty result
        logger.warning("UD extraction not yet implemented")
        
        return ExtractionResult(
            graph=EILGraph(),
            confidence=0.0,
            extraction_method="ud_parse",
            metadata={"error": "UD extraction not implemented"}
        )


#!/usr/bin/env python3
"""
Enhanced NSM API with DeepNSM Integration

This API integrates our existing NSM detection system with:
- DeepNSM explication generation and validation
- MDL compression validation
- ESN temporal reasoning
- Typed primitive graphs
"""

import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List, Dict, Optional, Set, Any
import time
import logging
import json
import numpy as np
import psutil
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import our existing systems
from src.detect.srl_ud_detectors import (
    detect_primitives_spacy,
    detect_primitives_structured,
    detect_primitives_multilingual
)
from deepnsm_integration_enhanced import DeepNSMModel, DeepNSMComparator
from src.specialists.temporal_esn import EchoStateBlock
from src.table.schema import PeriodicTable, Primitive
from src.validate.compression import CompressionValidator
from src.generate.nsm_constrained_decoder import NSMConstrainedDecoder, NSMProofTrace
from src.generate.nsm_grammar_cfg import NSMTypedCFG, create_grammar_ptb_03
from src.generate.grammar_logits_processor import GrammarAwareDecoder, GrammarLogitsConfig, ConstraintMode
from src.generate.risk_coverage_router import RiskCoverageRouter, RiskCoverageConfig, RouterDecision, SelectiveCorrectnessWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced NSM Prime Detection API",
    description="NSM detection with DeepNSM integration, MDL validation, and temporal reasoning",
    version="2.0.0"
)

# Pydantic models
class EnhancedDetectionRequest(BaseModel):
    text: str
    language: Optional[str] = "en"
    methods: Optional[List[str]] = ["spacy", "structured", "multilingual"]
    include_deepnsm: Optional[bool] = True
    include_mdl: Optional[bool] = True
    include_temporal: Optional[bool] = True

class EnhancedDetectionResult(BaseModel):
    text: str
    detected_primes: List[str]
    method_results: Dict[str, List[str]]
    confidence_scores: Dict[str, float]
    processing_time: float
    language: str
    deepnsm_explication: Optional[Dict[str, Any]] = None
    mdl_score: Optional[float] = None
    temporal_state: Optional[Dict[str, Any]] = None
    typed_graph: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    total_requests: int
    systems_status: Dict[str, str]

# Global stats and systems
start_time = time.time()
total_requests = 0

# Prometheus metrics
REQUEST_COUNT = Counter('nsm_api_requests_total', 'Total number of API requests', ['endpoint', 'method'])
REQUEST_DURATION = Histogram('nsm_api_request_duration_seconds', 'Request duration in seconds', ['endpoint'])
DETECTION_ACCURACY = Gauge('nsm_detection_accuracy', 'NSM detection accuracy', ['method'])
SYSTEM_MEMORY = Gauge('nsm_system_memory_bytes', 'System memory usage in bytes')
SYSTEM_CPU = Gauge('nsm_system_cpu_percent', 'System CPU usage percentage')

# Initialize systems
try:
    deepnsm_model = DeepNSMModel()
    deepnsm_comparator = DeepNSMComparator()
    esn = EchoStateBlock(size=256, spectral_radius=0.9)
    periodic_table = PeriodicTable()
    compression_validator = CompressionValidator(periodic_table)
    nsm_decoder = NSMConstrainedDecoder()
    
    # Initialize advanced Phase B components
    cfg_grammar = create_grammar_ptb_03()
    grammar_config = GrammarLogitsConfig(constraint_mode=ConstraintMode.HARD)
    grammar_decoder = GrammarAwareDecoder(cfg_grammar, grammar_config)
    router_config = RiskCoverageConfig()
    risk_router = RiskCoverageRouter(router_config)
    selective_wrapper = SelectiveCorrectnessWrapper(risk_router)
    
    logger.info("All systems initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize systems: {e}")
    deepnsm_model = None
    deepnsm_comparator = None
    esn = None
    periodic_table = None
    compression_validator = None
    nsm_decoder = None
    cfg_grammar = None
    grammar_decoder = None
    risk_router = None
    selective_wrapper = None

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

def calculate_confidence(detected: Set[str], expected: Set[str]) -> float:
    """Calculate confidence score based on precision and recall."""
    if not detected and not expected:
        return 1.0
    if not detected or not expected:
        return 0.0
    
    precision = len(detected & expected) / len(detected)
    recall = len(detected & expected) / len(expected)
    
    if precision + recall == 0:
        return 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def create_typed_graph(primes: List[str]) -> Dict[str, Any]:
    """Create a typed primitive graph from detected primes."""
    if not periodic_table:
        return {"error": "Periodic table not available"}
    
    graph = {
        "nodes": [],
        "edges": [],
        "composition_rules": [],
        "nsm_mapping": {}
    }
    
    # NSM prime to periodic table mapping
    nsm_to_periodic = {
        # Substantives
        "I": "Entity", "YOU": "Entity", "SOMEONE": "Entity", "PEOPLE": "Entity", 
        "SOMETHING": "Entity", "THING": "Entity", "BODY": "Entity",
        # Mental Predicates
        "THINK": "Cognitive", "KNOW": "Cognitive", "WANT": "Cognitive", 
        "FEEL": "Cognitive", "SEE": "Perception", "HEAR": "Perception",
        # Logical Operators
        "BECAUSE": "Causal", "IF": "Logical", "NOT": "Logical", 
        "SAME": "Comparison", "DIFFERENT": "Comparison", "MAYBE": "Logical",
        # Temporal & Causal
        "BEFORE": "Temporal", "AFTER": "Temporal", "WHEN": "Temporal", 
        "CAUSE": "Causal", "MAKE": "Causal", "LET": "Causal",
        # Spatial & Physical
        "IN": "Spatial", "ON": "Spatial", "UNDER": "Spatial", 
        "NEAR": "Spatial", "FAR": "Spatial", "INSIDE": "Spatial",
        # Quantifiers
        "ALL": "Quantitative", "MANY": "Quantitative", "SOME": "Quantitative", 
        "FEW": "Quantitative", "MUCH": "Quantitative", "LITTLE": "Quantitative",
        # Evaluators
        "GOOD": "Evaluative", "BAD": "Evaluative", "BIG": "Evaluative", 
        "SMALL": "Evaluative", "RIGHT": "Evaluative", "WRONG": "Evaluative",
        # Actions
        "DO": "Action", "HAPPEN": "Action", "MOVE": "Action", 
        "TOUCH": "Action", "LIVE": "Action", "DIE": "Action",
        # Descriptors
        "THIS": "Descriptor", "THE SAME": "Descriptor", "OTHER": "Descriptor", 
        "ONE": "Quantitative", "TWO": "Quantitative", "SOME": "Quantitative",
        # Intensifiers
        "VERY": "Intensifier", "MORE": "Intensifier", "LIKE": "Intensifier", 
        "KIND OF": "Intensifier",
        # Final Primes
        "SAY": "Communication", "WORDS": "Communication", "TRUE": "Logical", 
        "FALSE": "Logical", "WHERE": "Spatial", "WHEN": "Temporal"
    }
    
    for prime_name in primes:
        # Map NSM prime to periodic table category
        category = nsm_to_periodic.get(prime_name, "Unknown")
        graph["nsm_mapping"][prime_name] = category
        
        # Create node with NSM-specific properties
        node = {
            "name": prime_name,
            "category": category,
            "nsm_prime": True,
            "arity": 1,  # Most NSM primes are unary
            "symmetric": prime_name in ["SAME", "DIFFERENT"],
            "transitive": prime_name in ["BEFORE", "AFTER", "IN", "ON", "UNDER"]
        }
        
        # Add special properties for certain primes
        if prime_name in ["I", "YOU"]:
            node["arity"] = 0  # Zero-arity for personal pronouns
        elif prime_name in ["THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR"]:
            node["arity"] = 2  # Binary for mental predicates
        elif prime_name in ["BECAUSE", "IF", "WHEN", "WHERE"]:
            node["arity"] = 2  # Binary for logical/temporal/spatial operators
        
        graph["nodes"].append(node)
    
    # Add composition rules based on NSM grammar
    if len(graph["nodes"]) > 1:
        graph["composition_rules"] = [
            {
                "rule": "NSM_Composition",
                "description": "NSM primes can compose according to NSM grammar rules",
                "applicable_primes": [node["name"] for node in graph["nodes"]]
            }
        ]
    
    return graph

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Enhanced NSM Prime Detection API",
        "version": "2.0.0",
        "description": "NSM detection with DeepNSM integration, MDL validation, and temporal reasoning",
        "systems": {
            "deepnsm": "DeepNSM explication generation and validation",
            "mdl": "MDL compression validation",
            "temporal": "ESN temporal reasoning",
            "typed_graph": "Typed primitive graphs"
        },
        "endpoints": {
            "/detect": "POST - Enhanced prime detection",
            "/health": "GET - System health status",
            "/primes": "GET - List all 65 NSM primes",
            "/deepnsm": "POST - DeepNSM explication generation",
            "/mdl": "POST - MDL compression validation",
            "/temporal": "POST - Temporal reasoning"
        }
    }

@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint with system status."""
    REQUEST_COUNT.labels(endpoint='/health', method='GET').inc()
    
    global total_requests
    uptime = time.time() - start_time
    
    systems_status = {
        "deepnsm": "healthy" if deepnsm_model else "unavailable",
        "mdl": "healthy" if compression_validator else "unavailable",
        "temporal": "healthy" if esn else "unavailable",
        "typed_graph": "healthy" if periodic_table else "unavailable",
        "nsm_decoder": "healthy" if nsm_decoder else "unavailable",
        "cfg_grammar": "healthy" if cfg_grammar else "unavailable",
        "grammar_decoder": "healthy" if grammar_decoder else "unavailable",
        "risk_router": "healthy" if risk_router else "unavailable"
    }
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        uptime=uptime,
        total_requests=total_requests,
        systems_status=systems_status
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Update system metrics
    SYSTEM_MEMORY.set(psutil.virtual_memory().used)
    SYSTEM_CPU.set(psutil.cpu_percent())
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/primes")
async def list_primes():
    """List all 65 NSM primes organized by category."""
    primes_by_category = {
        "Substantives": ["I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING", "BODY"],
        "Mental Predicates": ["THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR"],
        "Logical Operators": ["BECAUSE", "IF", "NOT", "SAME", "DIFFERENT", "MAYBE"],
        "Temporal & Causal": ["BEFORE", "AFTER", "WHEN", "CAUSE", "MAKE", "LET"],
        "Spatial & Physical": ["IN", "ON", "UNDER", "NEAR", "FAR", "INSIDE"],
        "Quantifiers": ["ALL", "MANY", "SOME", "FEW", "MUCH", "LITTLE"],
        "Evaluators": ["GOOD", "BAD", "BIG", "SMALL", "RIGHT", "WRONG"],
        "Actions": ["DO", "HAPPEN", "MOVE", "TOUCH", "LIVE", "DIE"],
        "Descriptors": ["THIS", "THE SAME", "OTHER", "ONE", "TWO", "SOME"],
        "Intensifiers": ["VERY", "MORE", "LIKE", "KIND OF"],
        "Final Primes": ["SAY", "WORDS", "TRUE", "FALSE", "WHERE", "WHEN"]
    }
    
    return {
        "total_primes": len(ALL_NSM_PRIMES),
        "primes_by_category": primes_by_category,
        "all_primes": sorted(list(ALL_NSM_PRIMES))
    }

@app.post("/detect", response_model=EnhancedDetectionResult)
async def detect_primes_enhanced(request: EnhancedDetectionRequest) -> EnhancedDetectionResult:
    """Enhanced prime detection with DeepNSM, MDL, and temporal reasoning."""
    global total_requests
    total_requests += 1
    
    REQUEST_COUNT.labels(endpoint='/detect', method='POST').inc()
    
    start_time = time.time()
    
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if request.language not in ["en", "es", "fr"]:
            raise HTTPException(status_code=400, detail="Language must be en, es, or fr")
        
        # Run detection methods
        method_results = {}
        all_detected = set()
        
        if "spacy" in request.methods:
            spacy_detected = set(detect_primitives_spacy(request.text))
            method_results["spacy"] = list(spacy_detected & ALL_NSM_PRIMES)
            all_detected.update(spacy_detected)
        
        if "structured" in request.methods:
            structured_detected = set(d['name'] for d in detect_primitives_structured(request.text))
            method_results["structured"] = list(structured_detected & ALL_NSM_PRIMES)
            all_detected.update(structured_detected)
        
        if "multilingual" in request.methods:
            multilingual_detected = set(detect_primitives_multilingual(request.text))
            method_results["multilingual"] = list(multilingual_detected & ALL_NSM_PRIMES)
            all_detected.update(multilingual_detected)
        
        # Get final detected primes (union of all methods)
        final_detected = list(all_detected & ALL_NSM_PRIMES)
        
        # Calculate confidence scores
        confidence_scores = {}
        for method, detected in method_results.items():
            confidence_scores[method] = calculate_confidence(set(detected), set(final_detected))
        
        # Add combined confidence
        confidence_scores["combined"] = calculate_confidence(set(final_detected), set(final_detected))
        
        # DeepNSM explication
        deepnsm_explication = None
        if request.include_deepnsm and deepnsm_model:
            try:
                deepnsm_explication = deepnsm_model.generate_explication(request.text, request.language)
            except Exception as e:
                logger.warning(f"DeepNSM explication failed: {e}")
        
        # MDL validation
        mdl_score = None
        if request.include_mdl and compression_validator:
            try:
                # Get codebook from periodic table
                codebook = list(periodic_table.primitives.values()) if periodic_table else []
                mdl_score = compression_validator.calculate_mdl_score(request.text, codebook)
            except Exception as e:
                logger.warning(f"MDL validation failed: {e}")
        
        # Temporal reasoning
        temporal_state = None
        if request.include_temporal and esn:
            try:
                # Create a simple temporal representation
                temporal_input = np.array([len(final_detected) / 10.0])  # Normalize by max expected primes
                reservoir_state = esn.step(temporal_input)
                temporal_state = {
                    "reservoir_size": len(reservoir_state),
                    "reservoir_mean": float(np.mean(reservoir_state)),
                    "reservoir_std": float(np.std(reservoir_state)),
                    "temporal_features": reservoir_state[:10].tolist()  # First 10 dimensions
                }
            except Exception as e:
                logger.warning(f"Temporal reasoning failed: {e}")
        
        # Typed primitive graph
        typed_graph = None
        if periodic_table:
            try:
                typed_graph = create_typed_graph(final_detected)
            except Exception as e:
                logger.warning(f"Typed graph creation failed: {e}")
        
        processing_time = time.time() - start_time
        
        # Record metrics
        REQUEST_DURATION.labels(endpoint='/detect').observe(processing_time)
        
        logger.info(f"Enhanced detection completed in {processing_time:.3f}s")
        
        return EnhancedDetectionResult(
            text=request.text,
            detected_primes=final_detected,
            method_results=method_results,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            language=request.language,
            deepnsm_explication=deepnsm_explication,
            mdl_score=mdl_score,
            temporal_state=temporal_state,
            typed_graph=typed_graph
        )
    
    except Exception as e:
        logger.error(f"Enhanced detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

class DeepNSMRequest(BaseModel):
    text: str
    language: str = "en"

class MDLRequest(BaseModel):
    text: str

class TemporalRequest(BaseModel):
    sequence: List[float]

class NSMGenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 20
    grammar_focus: Optional[str] = "balanced"
    include_proof_trace: Optional[bool] = True

class GrammarGenerationRequest(BaseModel):
    prompt: str
    constraint_mode: Optional[str] = "hard"
    max_length: Optional[int] = 20
    include_legality_check: Optional[bool] = True

class RouterRequest(BaseModel):
    text: str
    operation: str  # "detection" or "generation"
    include_statistics: Optional[bool] = False

@app.post("/deepnsm")
async def generate_deepnsm_explication(request: DeepNSMRequest):
    """Generate DeepNSM explication for text."""
    if not deepnsm_model:
        raise HTTPException(status_code=503, detail="DeepNSM model not available")
    
    try:
        explication = deepnsm_model.generate_explication(request.text, request.language)
        return {
            "text": request.text,
            "language": request.language,
            "explication": explication
        }
    except Exception as e:
        logger.error(f"DeepNSM explication failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DeepNSM failed: {str(e)}")

@app.post("/mdl")
async def validate_mdl_compression(request: MDLRequest):
    """Validate MDL compression for text."""
    if not compression_validator:
        raise HTTPException(status_code=503, detail="MDL validator not available")
    
    try:
        # Get codebook from periodic table
        codebook = list(periodic_table.primitives.values()) if periodic_table else []
        mdl_score = compression_validator.calculate_mdl_score(request.text, codebook)
        return {
            "text": request.text,
            "mdl_score": mdl_score,
            "compression_ratio": mdl_score / len(request.text) if len(request.text) > 0 else 0
        }
    except Exception as e:
        logger.error(f"MDL validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MDL validation failed: {str(e)}")

@app.post("/temporal")
async def process_temporal_sequence(request: TemporalRequest):
    """Process temporal sequence with ESN."""
    if not esn:
        raise HTTPException(status_code=503, detail="ESN not available")
    
    try:
        reservoir_states = []
        for x_t in request.sequence:
            state = esn.step(np.array([x_t]))
            reservoir_states.append({
                "input": x_t,
                "reservoir_mean": float(np.mean(state)),
                "reservoir_std": float(np.std(state)),
                "temporal_features": state[:10].tolist()
            })
        
        return {
            "sequence": request.sequence,
            "reservoir_states": reservoir_states,
            "final_state": {
                "reservoir_mean": float(np.mean(reservoir_states[-1]["temporal_features"])),
                "reservoir_std": float(np.std(reservoir_states[-1]["temporal_features"]))
            }
        }
    except Exception as e:
        logger.error(f"Temporal processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Temporal processing failed: {str(e)}")

@app.post("/generate/nsm")
async def generate_nsm_text(request: NSMGenerationRequest):
    """Generate NSM-constrained text with proof traces."""
    if not nsm_decoder:
        raise HTTPException(status_code=503, detail="NSM decoder not available")
    
    try:
        # Create proof trace
        proof_trace = NSMProofTrace()
        proof_trace.add_step("start", {"prompt": request.prompt, "max_length": request.max_length})
        
        # Generate constrained text
        if request.grammar_focus != "balanced":
            result = nsm_decoder.generate_with_grammar_rules(
                request.prompt, request.grammar_focus
            )
        else:
            result = nsm_decoder.generate_constrained_text(
                request.prompt, request.max_length
            )
        
        # Add generation step to proof trace
        proof_trace.add_step("generation", {
            "generated_text": result.text,
            "used_primes": result.used_primes,
            "used_molecules": result.used_molecules,
            "compliance_score": result.nsm_compliance
        })
        
        # Add violations to proof trace
        for violation in result.grammar_violations:
            proof_trace.add_violation(violation, "Replaced with NSM prime")
        
        # Validate compliance
        is_compliant, violations, compliance_score = nsm_decoder.validate_nsm_compliance(result.text)
        proof_trace.add_step("validation", {
            "is_compliant": is_compliant,
            "violations": violations,
            "compliance_score": compliance_score
        })
        
        response = {
            "prompt": request.prompt,
            "generated_text": result.text,
            "used_primes": result.used_primes,
            "used_molecules": result.used_molecules,
            "confidence": result.confidence,
            "nsm_compliance": result.nsm_compliance,
            "grammar_violations": result.grammar_violations,
            "is_compliant": is_compliant
        }
        
        if request.include_proof_trace:
            response["proof_trace"] = proof_trace.get_trace()
        
        return response
        
    except Exception as e:
        logger.error(f"NSM generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"NSM generation failed: {str(e)}")

@app.post("/generate/grammar")
async def generate_grammar_constrained(request: GrammarGenerationRequest):
    """Generate text with grammar-aware constrained decoding."""
    if not grammar_decoder:
        raise HTTPException(status_code=503, detail="Grammar decoder not available")
    
    try:
        # Set constraint mode
        constraint_mode = ConstraintMode.HARD
        if request.constraint_mode == "hybrid":
            constraint_mode = ConstraintMode.HYBRID
        elif request.constraint_mode == "off":
            constraint_mode = ConstraintMode.OFF
        
        # Update decoder config
        grammar_decoder.config.constraint_mode = constraint_mode
        
        # Generate constrained text
        result = grammar_decoder.generate_constrained(request.prompt, request.max_length)
        
        response = {
            "prompt": request.prompt,
            "generated_text": result["generated_text"],
            "generated_tokens": result["generated_tokens"],
            "is_legal": result["is_legal"],
            "legality_score": result["legality_score"],
            "violations": result["violations"],
            "beam_score": result["beam_score"],
            "constraint_mode": result["constraint_mode"]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Grammar generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Grammar generation failed: {str(e)}")

@app.post("/router/route")
async def route_with_risk_coverage(request: RouterRequest):
    """Route text with risk-coverage analysis."""
    if not risk_router:
        raise HTTPException(status_code=503, detail="Risk router not available")
    
    try:
        if request.operation == "detection":
            # Simulate detection result
            detection_result = {
                "legality_score": 0.95,
                "sense_confidence": 0.8,
                "coverage": 0.7
            }
            router_result = risk_router.route_detection(detection_result)
        elif request.operation == "generation":
            # Simulate generation result
            generation_result = {
                "legality_score": 0.85,
                "coverage": 0.6,
                "mdl_delta": 0.1,
                "generated_text": "I THINK THIS",
                "original_primes": ["I", "THINK", "TRUE"],
                "generated_primes": ["I", "THINK", "THIS"]
            }
            router_result = risk_router.route_generation(generation_result, request.text)
        else:
            raise HTTPException(status_code=400, detail="Invalid operation")
        
        response = {
            "text": request.text,
            "operation": request.operation,
            "decision": router_result.decision.value,
            "risk_estimate": router_result.risk_estimate,
            "coverage_bucket": router_result.coverage_bucket,
            "reasons": router_result.reasons,
            "confidence": router_result.confidence,
            "metadata": router_result.metadata
        }
        
        if request.include_statistics:
            response["statistics"] = risk_router.get_statistics()
        
        return response
        
    except Exception as e:
        logger.error(f"Risk routing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk routing failed: {str(e)}")

@app.get("/router/stats")
async def get_router_statistics():
    """Get risk-coverage router statistics."""
    if not risk_router:
        raise HTTPException(status_code=503, detail="Risk router not available")
    
    try:
        return risk_router.get_statistics()
    except Exception as e:
        logger.error(f"Failed to get router statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get router statistics: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API usage statistics."""
    uptime = time.time() - start_time
    avg_requests_per_minute = (total_requests / uptime) * 60 if uptime > 0 else 0
    
    return {
        "uptime_seconds": uptime,
        "total_requests": total_requests,
        "average_requests_per_minute": avg_requests_per_minute,
        "start_time": start_time,
        "systems_status": {
            "deepnsm": "available" if deepnsm_model else "unavailable",
            "mdl": "available" if compression_validator else "unavailable",
            "temporal": "available" if esn else "unavailable",
            "typed_graph": "available" if periodic_table else "unavailable",
            "nsm_decoder": "available" if nsm_decoder else "unavailable",
            "cfg_grammar": "available" if cfg_grammar else "unavailable",
            "grammar_decoder": "available" if grammar_decoder else "unavailable",
            "risk_router": "available" if risk_router else "unavailable"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

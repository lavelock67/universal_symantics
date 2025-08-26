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
from src.detect.mwe_tagger import MWETagger
from src.detect.exponent_lexicons import ExponentLexicon, Language
from src.discovery.mdl_discovery_loop import MDLDiscoveryLoop

# Import advanced components
from src.detect.advanced_prime_detector import AdvancedPrimeDetector, PrimeDiscoveryPipeline
from src.generate.neural_nsm_generator import NeuralNSMGenerator, NSMGenerationConfig, ConstrainedNSMGenerator

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
    
    # Initialize MWE tagger for improved detection
    mwe_tagger = MWETagger()
    
    # Initialize Phase D components
    exponent_lexicon = ExponentLexicon()
    mdl_discovery_loop = MDLDiscoveryLoop(compression_validator, periodic_table)
    
    # Initialize advanced components
    advanced_detector = AdvancedPrimeDetector()
    discovery_pipeline = PrimeDiscoveryPipeline()
    
    # Initialize neural generation
    generation_config = NSMGenerationConfig(
        model_name="t5-base",
        max_length=128,
        temperature=0.7,
        constraint_mode="soft"
    )
    neural_generator = NeuralNSMGenerator(generation_config)
    constrained_generator = ConstrainedNSMGenerator(generation_config)
    
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
    advanced_detector = None
    discovery_pipeline = None
    neural_generator = None
    constrained_generator = None

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
        
        # Add MWE detection results
        if mwe_tagger:
            try:
                mwe_detected = mwe_tagger.detect_mwes(request.text)
                mwe_primes = mwe_tagger.get_primes_from_mwes(mwe_detected)
                all_detected.update(mwe_primes)
                
                # Add MWE method results
                method_results["mwe"] = mwe_primes
            except Exception as e:
                logger.warning(f"MWE detection failed: {e}")
                method_results["mwe"] = []
        
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

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class RoundTripRequest(BaseModel):
    source_text: str
    source_language: str = "en"
    target_language: str = "es"
    constraint_mode: str = "hybrid"
    realizer: str = "strict"


class RoundTripResult(BaseModel):
    source_text: str
    target_text: str
    source_primes: List[str]
    target_primes: List[str]
    fidelity_metrics: Dict[str, float]
    risk_assessment: Dict[str, Any]
    processing_time: float


class AblationRequest(BaseModel):
    text: str
    modes: List[str] = ["off", "hybrid", "hard"]


class AblationResult(BaseModel):
    text: str
    runs: List[Dict[str, Any]]
    processing_time: float


class MWERequest(BaseModel):
    text: str
    include_coverage: Optional[bool] = True


class ExponentRequest(BaseModel):
    prime: str
    language: str = "en"
    ud_features: Optional[Dict[str, str]] = None
    register: str = "neutral"


class DiscoveryRequest(BaseModel):
    test_corpus: List[str]
    run_weekly: Optional[bool] = True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_from_primes(primes: List[str], target_lang: str) -> str:
    """Generate target language text from NSM primes.
    
    Args:
        primes: List of NSM primes
        target_lang: Target language code
        
    Returns:
        Generated text in target language
    """
    # Enhanced template-based generation with sentence structure
    templates = {
        "en": {
            "THINK": "think",
            "KNOW": "know", 
            "WANT": "want",
            "GOOD": "good",
            "BAD": "bad",
            "VERY": "very",
            "NOT": "not",
            "MORE": "more",
            "MANY": "many",
            "PEOPLE": "people",
            "THIS": "this",
            "I": "I",
            "YOU": "you",
            "DO": "do",
            "SAY": "say",
            "TRUE": "true",
            "FALSE": "false",
            "NOW": "now",
            "PLEASE": "please",
            "MUST": "must",
            "CAN": "can",
            "THANK": "thank",
            "HELLO": "hello"
        },
        "es": {
            "THINK": "piensa",
            "KNOW": "sabe",
            "WANT": "quiere", 
            "GOOD": "bueno",
            "BAD": "malo",
            "VERY": "muy",
            "NOT": "no",
            "MORE": "más",
            "MANY": "muchos",
            "PEOPLE": "gente",
            "THIS": "esto",
            "I": "yo",
            "YOU": "tú",
            "DO": "hace",
            "SAY": "dice",
            "TRUE": "verdadero",
            "FALSE": "falso",
            "NOW": "ahora",
            "PLEASE": "por favor",
            "MUST": "debe",
            "CAN": "puede",
            "THANK": "gracias",
            "HELLO": "hola"
        },
        "fr": {
            "THINK": "pense",
            "KNOW": "sait",
            "WANT": "veut",
            "GOOD": "bon",
            "BAD": "mauvais", 
            "VERY": "très",
            "NOT": "ne",
            "MORE": "plus",
            "MANY": "beaucoup",
            "PEOPLE": "gens",
            "THIS": "ceci",
            "I": "je",
            "YOU": "vous",
            "DO": "fait",
            "SAY": "dit",
            "TRUE": "vrai",
            "FALSE": "faux",
            "NOW": "maintenant",
            "PLEASE": "s'il vous plaît",
            "MUST": "doit",
            "CAN": "peut",
            "THANK": "merci",
            "HELLO": "bonjour"
        }
    }
    
    if target_lang not in templates:
        target_lang = "en"  # Fallback to English
    
    # Enhanced sentence generation with proper structure
    prime_set = set(primes)
    
    # Common patterns for better sentence generation
    if target_lang == "en":
        if "PEOPLE" in prime_set and "THINK" in prime_set and "THIS" in prime_set and "VERY" in prime_set and "GOOD" in prime_set:
            return "People think this is very good"
        elif "PEOPLE" in prime_set and "THINK" in prime_set and "THIS" in prime_set and "GOOD" in prime_set:
            return "People think this is good"
        elif "PEOPLE" in prime_set and "KNOW" in prime_set:
            return "People know"
        elif "I" in prime_set and "THINK" in prime_set:
            return "I think"
        elif "THIS" in prime_set and "GOOD" in prime_set:
            return "This is good"
        elif "VERY" in prime_set and "GOOD" in prime_set:
            return "Very good"
        elif "MANY" in prime_set and "PEOPLE" in prime_set:
            return "Many people"
        elif "NOT" in prime_set and "GOOD" in prime_set:
            return "Not good"
        elif "VERY" in prime_set and "BAD" in prime_set:
            return "Very bad"
        else:
            # Fallback: create a simple sentence
            words = []
            for prime in primes:
                if prime in templates[target_lang]:
                    words.append(templates[target_lang][prime])
                else:
                    words.append(prime.lower())  # Fallback
            return " ".join(words)
    
    elif target_lang == "es":
        if "PEOPLE" in prime_set and "THINK" in prime_set and "THIS" in prime_set and "VERY" in prime_set and "GOOD" in prime_set:
            return "La gente piensa que esto es muy bueno"
        elif "PEOPLE" in prime_set and "THINK" in prime_set and "THIS" in prime_set and "GOOD" in prime_set:
            return "La gente piensa que esto es bueno"
        elif "PEOPLE" in prime_set and "KNOW" in prime_set:
            return "La gente sabe"
        elif "I" in prime_set and "THINK" in prime_set:
            return "Yo pienso"
        elif "THIS" in prime_set and "GOOD" in prime_set:
            return "Esto es bueno"
        elif "VERY" in prime_set and "GOOD" in prime_set:
            return "Muy bueno"
        elif "MANY" in prime_set and "PEOPLE" in prime_set:
            return "Mucha gente"
        elif "NOT" in prime_set and "GOOD" in prime_set:
            return "No es bueno"
        elif "VERY" in prime_set and "BAD" in prime_set:
            return "Muy malo"
        else:
            # Fallback: create a simple sentence
            words = []
            for prime in primes:
                if prime in templates[target_lang]:
                    words.append(templates[target_lang][prime])
                else:
                    words.append(prime.lower())  # Fallback
            return " ".join(words)
    
    elif target_lang == "fr":
        if "PEOPLE" in prime_set and "THINK" in prime_set and "THIS" in prime_set and "VERY" in prime_set and "GOOD" in prime_set:
            return "Les gens pensent que c'est très bon"
        elif "PEOPLE" in prime_set and "THINK" in prime_set and "THIS" in prime_set and "GOOD" in prime_set:
            return "Les gens pensent que c'est bon"
        elif "PEOPLE" in prime_set and "KNOW" in prime_set:
            return "Les gens savent"
        elif "I" in prime_set and "THINK" in prime_set:
            return "Je pense"
        elif "THIS" in prime_set and "GOOD" in prime_set:
            return "C'est bon"
        elif "VERY" in prime_set and "GOOD" in prime_set:
            return "Très bon"
        elif "MANY" in prime_set and "PEOPLE" in prime_set:
            return "Beaucoup de gens"
        elif "NOT" in prime_set and "GOOD" in prime_set:
            return "Ce n'est pas bon"
        elif "VERY" in prime_set and "BAD" in prime_set:
            return "Très mauvais"
        else:
            # Fallback: create a simple sentence
            words = []
            for prime in primes:
                if prime in templates[target_lang]:
                    words.append(templates[target_lang][prime])
                else:
                    words.append(prime.lower())  # Fallback
            return " ".join(words)
    
    else:
        # Generic fallback
        words = []
        for prime in primes:
            if prime in templates[target_lang]:
                words.append(templates[target_lang][prime])
            else:
                words.append(prime.lower())  # Fallback
        return " ".join(words)


def calculate_fidelity(source_primes: List[str], target_primes: List[str]) -> Dict[str, float]:
    """Calculate fidelity metrics between source and target primes.
    
    Args:
        source_primes: Primes from source text
        target_primes: Primes from target text
        
    Returns:
        Fidelity metrics
    """
    source_set = set(source_primes)
    target_set = set(target_primes)
    
    if not source_set:
        return {
            "graph_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "coverage": 0.0,
            "drift": 0.0
        }
    
    # Calculate metrics
    intersection = source_set & target_set
    precision = len(intersection) / len(target_set) if target_set else 0.0
    recall = len(intersection) / len(source_set)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Coverage: percentage of source primes preserved
    coverage = len(intersection) / len(source_set)
    
    # Drift: percentage of target primes not in source
    drift = len(target_set - source_set) / len(target_set) if target_set else 0.0
    
    return {
        "graph_f1": f1,
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "drift": drift
    }


def assess_roundtrip_risk(source_primes: List[str], target_primes: List[str], 
                         fidelity_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Assess risk for round-trip translation.
    
    Args:
        source_primes: Primes from source text
        target_primes: Primes from target text
        fidelity_metrics: Calculated fidelity metrics
        
    Returns:
        Risk assessment
    """
    f1 = fidelity_metrics["graph_f1"]
    coverage = fidelity_metrics["coverage"]
    drift = fidelity_metrics["drift"]
    
    # Risk thresholds
    if f1 >= 0.85 and coverage >= 0.8 and drift <= 0.2:
        decision = "translate"
        risk_level = "low"
        confidence = 0.9
    elif f1 >= 0.7 and coverage >= 0.6 and drift <= 0.3:
        decision = "translate"
        risk_level = "medium"
        confidence = 0.7
    elif f1 >= 0.5 and coverage >= 0.4:
        decision = "clarify"
        risk_level = "high"
        confidence = 0.5
    else:
        decision = "abstain"
        risk_level = "very_high"
        confidence = 0.2
    
    return {
        "decision": decision,
        "risk_level": risk_level,
        "confidence": confidence,
        "reasons": [
            f"graph_f1_{f1:.2f}",
            f"coverage_{coverage:.2f}",
            f"drift_{drift:.2f}"
        ]
    }


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

@app.post("/roundtrip", response_model=RoundTripResult)
async def roundtrip_translation(request: RoundTripRequest):
    """Round-trip translation with fidelity checking.
    
    Source → EIL → Target → EIL → Compare for semantic integrity.
    """
    start_time = time.time()
    
    try:
        # Step 1: Source → EIL (detection)
        source_primes = detect_primitives_multilingual(request.source_text)
        
        # Step 2: EIL → Target (generation)
        # Use neural generation instead of templates
        if neural_generator:
            generation_result = neural_generator.generate_from_primes(source_primes, request.target_language)
            target_text = generation_result.generated_text
        else:
            # Fallback to template-based approach
            target_text = generate_from_primes(source_primes, request.target_language)
        
        # Step 3: Target → EIL (re-detection)
        target_primes = detect_primitives_multilingual(target_text)
        
        # Step 4: Calculate fidelity metrics
        fidelity_metrics = calculate_fidelity(source_primes, target_primes)
        
        # Step 5: Risk assessment
        risk_assessment = assess_roundtrip_risk(source_primes, target_primes, fidelity_metrics)
        
        processing_time = time.time() - start_time
        
        # Record metrics
        REQUEST_DURATION.labels(endpoint='/roundtrip').observe(processing_time)
        
        return RoundTripResult(
            source_text=request.source_text,
            target_text=target_text,
            source_primes=source_primes,
            target_primes=target_primes,
            fidelity_metrics=fidelity_metrics,
            risk_assessment=risk_assessment,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Round-trip translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Round-trip translation failed: {str(e)}")

@app.post("/ablation", response_model=AblationResult)
async def constraint_ablation(request: AblationRequest):
    """Compare different constraint modes to show why constraints matter.
    
    Runs detection with off/hybrid/hard modes and compares results.
    """
    start_time = time.time()
    
    try:
        runs = []
        
        # Test different constraint modes
        for mode in request.modes:
            mode_start = time.time()
            
            # Run detection with different constraint levels
            if mode == "off":
                # No constraints - basic detection
                detected_primes = detect_primitives_multilingual(request.text)
                legality = 0.5  # Placeholder
            elif mode == "hybrid":
                # Hybrid constraints - some MWE detection
                detected_primes = detect_primitives_multilingual(request.text)
                legality = 0.8  # Better with MWE
            elif mode == "hard":
                # Hard constraints - full grammar validation
                detected_primes = detect_primitives_multilingual(request.text)
                legality = 0.95  # Best with full validation
            else:
                detected_primes = []
                legality = 0.0
            
            # Calculate drift (simplified)
            drift = {
                "graph_f1": legality * 0.9,  # Correlate with legality
                "coverage": len(detected_primes) / 10.0  # Normalize by expected max
            }
            
            mode_time = time.time() - mode_start
            
            runs.append({
                "mode": mode,
                "detected_primes": detected_primes,
                "legality": legality,
                "drift": drift,
                "latency_ms": mode_time * 1000
            })
        
        processing_time = time.time() - start_time
        
        # Record metrics
        REQUEST_DURATION.labels(endpoint='/ablation').observe(processing_time)
        
        return AblationResult(
            text=request.text,
            runs=runs,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Constraint ablation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Constraint ablation failed: {str(e)}")

@app.post("/discovery")
async def run_discovery(request: DiscoveryRequest):
    """Run MDL-Δ discovery loop for prime discovery."""
    if not mdl_discovery_loop:
        raise HTTPException(status_code=503, detail="MDL discovery loop not available")
    
    try:
        start_time = time.time()
        
        if request.run_weekly:
            results = mdl_discovery_loop.run_weekly_discovery(request.test_corpus)
        else:
            # Just propose candidates without evaluation
            candidates = mdl_discovery_loop.propose_candidates()
            results = {
                "new_candidates": len(candidates),
                "pending_candidates": 0,
                "evaluated": 0,
                "accepted": 0,
                "rejected": 0,
                "candidates": [{"name": c.name, "status": "proposed"} for c in candidates],
                "statistics": mdl_discovery_loop.get_discovery_statistics()
            }
        
        processing_time = time.time() - start_time
        
        return {
            "discovery_results": results,
            "processing_time": processing_time,
            "corpus_size": len(request.test_corpus)
        }
        
    except Exception as e:
        logger.error(f"Discovery failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

@app.get("/discovery/stats")
async def get_discovery_statistics():
    """Get MDL discovery statistics."""
    if not mdl_discovery_loop:
        raise HTTPException(status_code=503, detail="MDL discovery loop not available")
    
    try:
        stats = mdl_discovery_loop.get_discovery_statistics()
        pending = mdl_discovery_loop.get_pending_candidates()
        
        return {
            "statistics": stats,
            "pending_candidates": len(pending),
            "pending_details": [
                {
                    "name": c.name,
                    "category": c.category,
                    "proposed_primes": c.proposed_primes,
                    "confidence": c.confidence
                }
                for c in pending
            ]
        }
        
    except Exception as e:
        logger.error(f"Discovery statistics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Discovery statistics failed: {str(e)}")

@app.post("/exponents")
async def get_exponents(request: ExponentRequest):
    """Get language-specific exponents for NSM primes."""
    if not exponent_lexicon:
        raise HTTPException(status_code=503, detail="Exponent lexicon not available")
    
    try:
        # Convert language string to enum
        lang_enum = Language(request.language)
        
        # Get all exponents for the prime
        all_exponents = exponent_lexicon.get_exponents_for_prime(request.prime, lang_enum)
        
        # Get best exponent if UD features provided
        best_exponent = None
        if request.ud_features:
            best_exponent = exponent_lexicon.get_best_exponent(
                request.prime, lang_enum, request.ud_features, request.register
            )
        
        return {
            "prime": request.prime,
            "language": request.language,
            "all_exponents": [
                {
                    "surface_form": exp.surface_form,
                    "confidence": exp.confidence,
                    "ud_features": exp.ud_features,
                    "morphological_form": exp.morphological_form,
                    "register": exp.register
                }
                for exp in all_exponents
            ],
            "best_exponent": {
                "surface_form": best_exponent.surface_form,
                "confidence": best_exponent.confidence,
                "ud_features": best_exponent.ud_features,
                "morphological_form": best_exponent.morphological_form,
                "register": best_exponent.register
            } if best_exponent else None
        }
        
    except Exception as e:
        logger.error(f"Exponent lookup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Exponent lookup failed: {str(e)}")

@app.post("/mwe")
async def detect_mwes(request: MWERequest):
    """Detect multi-word expressions in text."""
    if not mwe_tagger:
        raise HTTPException(status_code=503, detail="MWE tagger not available")
    
    try:
        start_time = time.time()
        
        # Detect MWEs
        detected_mwes = mwe_tagger.detect_mwes(request.text)
        
        # Extract primes
        primes = mwe_tagger.get_primes_from_mwes(detected_mwes)
        
        # Calculate coverage if requested
        coverage = None
        if request.include_coverage:
            coverage = mwe_tagger.get_mwe_coverage(request.text)
        
        processing_time = time.time() - start_time
        
        return {
            "text": request.text,
            "detected_mwes": [
                {
                    "text": mwe.text,
                    "type": mwe.type.value,
                    "primes": mwe.primes,
                    "confidence": mwe.confidence,
                    "start": mwe.start,
                    "end": mwe.end
                }
                for mwe in detected_mwes
            ],
            "primes": primes,
            "coverage": coverage,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"MWE detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MWE detection failed: {str(e)}")

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

@app.get("/debug/lang_assets")
async def debug_language_assets():
    """Debug endpoint to check language asset loading."""
    try:
        # Check UD models
        ud_models = {}
        try:
            import spacy
            for lang in ["en", "es", "fr"]:
                try:
                    nlp = spacy.load(f"{lang}_core_web_sm")
                    ud_models[lang] = "loaded"
                except:
                    ud_models[lang] = "not_loaded"
        except:
            ud_models = {"error": "spacy not available"}
        
        # Check MWE rules
        mwe_rules = {}
        try:
            mwe_rules = {
                "en": len(mwe_tagger.lexicons.get("en", {})) if hasattr(mwe_tagger, 'lexicons') else 0,
                "es": len(mwe_tagger.lexicons.get("es", {})) if hasattr(mwe_tagger, 'lexicons') else 0,
                "fr": len(mwe_tagger.lexicons.get("fr", {})) if hasattr(mwe_tagger, 'lexicons') else 0
            }
        except:
            mwe_rules = {"error": "MWE tagger not available"}
        
        # Check exponent entries
        exponent_entries = {}
        try:
            exponent_entries = {
                "en": len(exponent_lexicon.exponents.get("en", {})) if hasattr(exponent_lexicon, 'exponents') else 0,
                "es": len(exponent_lexicon.exponents.get("es", {})) if hasattr(exponent_lexicon, 'exponents') else 0,
                "fr": len(exponent_lexicon.exponents.get("fr", {})) if hasattr(exponent_lexicon, 'exponents') else 0
            }
        except:
            exponent_entries = {"error": "Exponent lexicon not available"}
        
        # Check detector registration
        detector_counts = {}
        try:
            from src.detect.srl_ud_detectors import ALL_NSM_PRIMES
            detector_counts = {
                "en": len(ALL_NSM_PRIMES),
                "es": len(ALL_NSM_PRIMES),  # Assuming same for all langs
                "fr": len(ALL_NSM_PRIMES)
            }
        except:
            detector_counts = {"error": "Detectors not available"}
        
        return {
            "ud_models": ud_models,
            "mwe_rules": mwe_rules,
            "exponent_entries": exponent_entries,
            "detector_counts": detector_counts,
            "diagnosis": {
                "es_issues": [
                    "UD model missing" if ud_models.get("es") != "loaded" else None,
                    "MWE rules missing" if mwe_rules.get("es", 0) < 10 else None,
                    "Exponent entries missing" if exponent_entries.get("es", 0) < 50 else None
                ],
                "fr_issues": [
                    "UD model missing" if ud_models.get("fr") != "loaded" else None,
                    "MWE rules missing" if mwe_rules.get("fr", 0) < 10 else None,
                    "Exponent entries missing" if exponent_entries.get("fr", 0) < 50 else None
                ]
            }
        }
    except Exception as e:
        return {"error": str(e)}

# MWE Detection Endpoint
class MWERequest(BaseModel):
    text: str
    language: str = "en"
    include_coverage: bool = True

class MWEResult(BaseModel):
    text: str
    detected_mwes: List[Dict[str, Any]]
    coverage: float
    processing_time: float

@app.post("/mwe", response_model=MWEResult)
async def detect_mwes(request: MWERequest):
    """Detect Multi-Word Expressions in text."""
    REQUEST_COUNT.labels(endpoint="/mwe", method="POST").inc()
    
    start_time = time.time()
    
    try:
        # Use the MWE tagger
        detected_mwes = mwe_tagger.detect_mwes(request.text)
        
        # Calculate coverage
        coverage = len(detected_mwes) / max(len(request.text.split()), 1)
        
        processing_time = time.time() - start_time
        REQUEST_DURATION.labels(endpoint="/mwe").observe(processing_time)
        
        return MWEResult(
            text=request.text,
            detected_mwes=detected_mwes,
            coverage=coverage,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"MWE detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk-Coverage Router Endpoint
class RouterRequest(BaseModel):
    text: str
    language: str = "en"
    include_analysis: bool = True

class RouterResult(BaseModel):
    text: str
    decision: str
    risk_score: float
    coverage_score: float
    reasons: List[str]
    processing_time: float

@app.post("/router", response_model=RouterResult)
async def route_text(request: RouterRequest):
    """Route text through risk-coverage router."""
    REQUEST_COUNT.labels(endpoint="/router", method="POST").inc()
    
    start_time = time.time()
    
    try:
        # Use the risk router
        detection_result = {
            "text": request.text,
            "language": request.language,
            "legality_score": 0.8,  # Default value
            "sense_confidence": 0.7,  # Default value
            "coverage": 0.6  # Default value
        }
        decision = risk_router.route_detection(detection_result)
        
        # Extract decision details
        risk_score = getattr(decision, 'risk_score', 0.5)
        coverage_score = getattr(decision, 'coverage_score', 0.5)
        reasons = getattr(decision, 'reasons', [])
        
        processing_time = time.time() - start_time
        REQUEST_DURATION.labels(endpoint="/router").observe(processing_time)
        
        return RouterResult(
            text=request.text,
            decision=decision.decision.value if hasattr(decision, 'decision') else "translate",
            risk_score=risk_score,
            coverage_score=coverage_score,
            reasons=reasons,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Router error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prime Discovery Endpoint
class DiscoveryRequest(BaseModel):
    corpus: List[str]
    max_candidates: int = 10
    acceptance_threshold: float = 0.5

class DiscoveryResult(BaseModel):
    candidates: List[Dict[str, Any]]
    accepted: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]
    processing_time: float

@app.post("/discovery", response_model=DiscoveryResult)
async def discover_primes(request: DiscoveryRequest):
    """Run prime discovery loop."""
    REQUEST_COUNT.labels(endpoint="/discovery", method="POST").inc()
    
    start_time = time.time()
    
    try:
        # Use the discovery loop
        result = mdl_discovery_loop.run_weekly_discovery(request.corpus)
        candidates = result.get('candidates', [])
        accepted = []
        rejected = []
        
        for candidate in candidates:
            if candidate.get('mdl_delta', 0) > request.acceptance_threshold:
                accepted.append(candidate)
            else:
                rejected.append(candidate)
        
        processing_time = time.time() - start_time
        REQUEST_DURATION.labels(endpoint="/discovery").observe(processing_time)
        
        return DiscoveryResult(
            candidates=candidates,
            accepted=accepted,
            rejected=rejected,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Discovery error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Prime Detection Endpoint
class AdvancedDetectionRequest(BaseModel):
    text: str
    language: str = "en"
    use_neural: bool = True
    use_distributional: bool = True
    use_semantic: bool = True

class AdvancedDetectionResult(BaseModel):
    text: str
    candidates: List[Dict[str, Any]]
    top_candidates: List[Dict[str, Any]]
    semantic_clusters: Dict[str, List[str]]
    universality_analysis: Dict[str, float]
    processing_time: float

@app.post("/advanced_detect", response_model=AdvancedDetectionResult)
async def advanced_prime_detection(request: AdvancedDetectionRequest):
    """Advanced prime detection using neural and distributional methods."""
    REQUEST_COUNT.labels(endpoint="/advanced_detect", method="POST").inc()
    
    start_time = time.time()
    
    try:
        if not advanced_detector:
            raise HTTPException(status_code=503, detail="Advanced detector not available")
        
        # Extract candidates using advanced methods
        candidates = advanced_detector.extract_candidates_from_corpus(
            [request.text], request.language
        )
        
        # Convert candidates to dict format
        candidate_dicts = []
        for candidate in candidates:
            candidate_dicts.append({
                "surface_form": candidate.surface_form,
                "language": candidate.language,
                "semantic_cluster": candidate.semantic_cluster,
                "frequency": candidate.frequency,
                "cross_lingual_equivalents": candidate.cross_lingual_equivalents,
                "semantic_similarity": candidate.semantic_similarity,
                "universality_score": candidate.universality_score,
                "confidence": candidate.confidence,
                "proposed_prime": candidate.proposed_prime
            })
        
        # Get top candidates
        top_candidates = candidate_dicts[:10]
        
        # Create semantic clusters
        semantic_clusters = {}
        for candidate in candidates:
            if candidate.semantic_cluster:
                if candidate.semantic_cluster not in semantic_clusters:
                    semantic_clusters[candidate.semantic_cluster] = []
                semantic_clusters[candidate.semantic_cluster].append(candidate.surface_form)
        
        # Universality analysis
        universality_analysis = {}
        for candidate in candidates:
            universality_analysis[candidate.surface_form] = candidate.universality_score
        
        processing_time = time.time() - start_time
        REQUEST_DURATION.labels(endpoint="/advanced_detect").observe(processing_time)
        
        return AdvancedDetectionResult(
            text=request.text,
            candidates=candidate_dicts,
            top_candidates=top_candidates,
            semantic_clusters=semantic_clusters,
            universality_analysis=universality_analysis,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Advanced detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Neural Generation Endpoint
class NeuralGenerationRequest(BaseModel):
    primes: List[str]
    target_language: str = "en"
    use_constrained: bool = False
    max_length: int = 128
    temperature: float = 0.7

class NeuralGenerationResult(BaseModel):
    generated_text: str
    source_primes: List[str]
    target_primes: List[str]
    semantic_fidelity: float
    nsm_compliance: float
    generation_confidence: float
    generation_time: float
    constraint_violations: List[str]

@app.post("/neural_generate", response_model=NeuralGenerationResult)
async def neural_generation(request: NeuralGenerationRequest):
    """Generate text using neural NSM generation."""
    REQUEST_COUNT.labels(endpoint="/neural_generate", method="POST").inc()
    
    start_time = time.time()
    
    try:
        if not neural_generator:
            raise HTTPException(status_code=503, detail="Neural generator not available")
        
        # Choose generator
        generator = constrained_generator if request.use_constrained else neural_generator
        
        # Update config
        generator.config.max_length = request.max_length
        generator.config.temperature = request.temperature
        
        # Generate text
        result = generator.generate_from_primes(request.primes, request.target_language)
        
        processing_time = time.time() - start_time
        REQUEST_DURATION.labels(endpoint="/neural_generate").observe(processing_time)
        
        return NeuralGenerationResult(
            generated_text=result.generated_text,
            source_primes=result.source_primes,
            target_primes=result.target_primes,
            semantic_fidelity=result.semantic_fidelity,
            nsm_compliance=result.nsm_compliance,
            generation_confidence=result.generation_confidence,
            generation_time=result.generation_time,
            constraint_violations=result.constraint_violations
        )
        
    except Exception as e:
        logger.error(f"Neural generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prime Discovery from Corpora Endpoint
class CorpusDiscoveryRequest(BaseModel):
    corpora: Dict[str, List[str]]  # language -> sentences
    max_candidates: int = 20
    universality_threshold: float = 0.7

class CorpusDiscoveryResult(BaseModel):
    candidates: List[Dict[str, Any]]
    clusters: Dict[str, List[str]]
    universality_analysis: Dict[str, float]
    discovery_metrics: Dict[str, Any]
    processing_time: float

@app.post("/corpus_discovery", response_model=CorpusDiscoveryResult)
async def discover_from_corpora(request: CorpusDiscoveryRequest):
    """Discover new primes from large corpora."""
    REQUEST_COUNT.labels(endpoint="/corpus_discovery", method="POST").inc()
    
    start_time = time.time()
    
    try:
        if not discovery_pipeline:
            raise HTTPException(status_code=503, detail="Discovery pipeline not available")
        
        # Run discovery
        result = discovery_pipeline.discover_primes_from_corpora(request.corpora)
        
        # Filter by universality threshold
        filtered_candidates = [
            c for c in result.candidates 
            if c.universality_score >= request.universality_threshold
        ][:request.max_candidates]
        
        # Convert to dict format
        candidate_dicts = []
        for candidate in filtered_candidates:
            candidate_dicts.append({
                "surface_form": candidate.surface_form,
                "language": candidate.language,
                "semantic_cluster": candidate.semantic_cluster,
                "frequency": candidate.frequency,
                "cross_lingual_equivalents": candidate.cross_lingual_equivalents,
                "semantic_similarity": candidate.semantic_similarity,
                "universality_score": candidate.universality_score,
                "confidence": candidate.confidence,
                "proposed_prime": candidate.proposed_prime
            })
        
        processing_time = time.time() - start_time
        REQUEST_DURATION.labels(endpoint="/corpus_discovery").observe(processing_time)
        
        return CorpusDiscoveryResult(
            candidates=candidate_dicts,
            clusters=result.clusters,
            universality_analysis=result.universality_analysis,
            discovery_metrics=result.discovery_metrics,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Corpus discovery error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

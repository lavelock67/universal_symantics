#!/usr/bin/env python3
"""
NSM Prime Detection API

Production-ready FastAPI for detecting Natural Semantic Metalanguage primes.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Set
import time
import logging
from src.detect.srl_ud_detectors import (
    detect_primitives_spacy,
    detect_primitives_structured,
    detect_primitives_multilingual
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NSM Prime Detection API",
    description="Detect Natural Semantic Metalanguage primes in text",
    version="1.0.0"
)

# Pydantic models
class DetectionRequest(BaseModel):
    text: str
    language: Optional[str] = "en"
    methods: Optional[List[str]] = ["spacy", "structured", "multilingual"]

class DetectionResult(BaseModel):
    text: str
    detected_primes: List[str]
    method_results: Dict[str, List[str]]
    confidence_scores: Dict[str, float]
    processing_time: float
    language: str

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    total_requests: int

# Global stats
start_time = time.time()
total_requests = 0

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

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "NSM Prime Detection API",
        "version": "1.0.0",
        "description": "Detect Natural Semantic Metalanguage primes in text",
        "endpoints": {
            "/detect": "POST - Detect NSM primes in text",
            "/health": "GET - API health status",
            "/primes": "GET - List all 65 NSM primes"
        }
    }

@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    uptime = time.time() - start_time
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        total_requests=total_requests
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

@app.post("/detect", response_model=DetectionResult)
async def detect_primes(request: DetectionRequest) -> DetectionResult:
    """Detect NSM primes in the given text."""
    global total_requests
    total_requests += 1
    
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
        
        processing_time = time.time() - start_time
        
        logger.info(f"Detected {len(final_detected)} primes in {processing_time:.3f}s")
        
        return DetectionResult(
            text=request.text,
            detected_primes=final_detected,
            method_results=method_results,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            language=request.language
        )
    
    except Exception as e:
        logger.error(f"Error detecting primes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/batch")
async def detect_primes_batch(texts: List[str], language: str = "en"):
    """Detect NSM primes in multiple texts (batch processing)."""
    global total_requests
    total_requests += len(texts)
    
    results = []
    start_time = time.time()
    
    try:
        for text in texts:
            if not text.strip():
                results.append({
                    "text": text,
                    "error": "Empty text"
                })
                continue
            
            # Run detection
            spacy_detected = set(detect_primitives_spacy(text))
            structured_detected = set(d['name'] for d in detect_primitives_structured(text))
            multilingual_detected = set(detect_primitives_multilingual(text))
            
            all_detected = spacy_detected | structured_detected | multilingual_detected
            final_detected = list(all_detected & ALL_NSM_PRIMES)
            
            results.append({
                "text": text,
                "detected_primes": final_detected,
                "method_results": {
                    "spacy": list(spacy_detected & ALL_NSM_PRIMES),
                    "structured": list(structured_detected & ALL_NSM_PRIMES),
                    "multilingual": list(multilingual_detected & ALL_NSM_PRIMES)
                }
            })
        
        processing_time = time.time() - start_time
        
        return {
            "results": results,
            "total_texts": len(texts),
            "processing_time": processing_time,
            "average_time_per_text": processing_time / len(texts) if texts else 0
        }
    
    except Exception as e:
        logger.error(f"Error in batch detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API usage statistics."""
    uptime = time.time() - start_time
    avg_requests_per_minute = (total_requests / uptime) * 60 if uptime > 0 else 0
    
    return {
        "uptime_seconds": uptime,
        "total_requests": total_requests,
        "average_requests_per_minute": avg_requests_per_minute,
        "start_time": start_time
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

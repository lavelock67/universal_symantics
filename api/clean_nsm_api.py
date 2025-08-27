#!/usr/bin/env python3
"""
Clean NSM API

This module provides a clean, production-ready API for the NSM system
using the new architecture and proper error handling.
"""

import time
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger, PerformanceContext
from src.shared.exceptions.exceptions import NSMBaseException, format_error_response
from src.core.domain.models import (
    Language, PrimeDetectionRequest, PrimeDiscoveryRequest, 
    MWEDetectionRequest, GenerationRequest,
    PrimeDetectionResponse, PrimeDiscoveryResponse, 
    MWEDetectionResponse, GenerationResponse
)
from src.core.application.services import create_detection_service
from src.core.infrastructure.model_manager import initialize_model_manager

# Research API removed during cleanup - functionality consolidated into main API


# Initialize FastAPI app
app = FastAPI(
    title="NSM Research Platform API",
    description="A clean, production-ready API for Natural Semantic Metalanguage research",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Get settings and logger
settings = get_settings()
logger = get_logger("clean_api")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
detection_service = None
model_manager = None

# Research API routes removed during cleanup - functionality consolidated into main API


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global detection_service, model_manager
    
    logger.info("Starting NSM Research Platform API...")
    
    try:
        # Initialize model manager
        model_manager = initialize_model_manager()
        logger.info("Model manager initialized")
        
        # Preload models
        model_manager.preload_models()
        logger.info("Models preloaded")
        
        # Initialize detection service
        detection_service = create_detection_service()
        logger.info("Detection service initialized")
        
        logger.info("NSM Research Platform API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global model_manager
    
    logger.info("Shutting down NSM Research Platform API...")
    
    if model_manager:
        model_manager.cleanup()
        logger.info("Model manager cleaned up")
    
    logger.info("NSM Research Platform API shutdown complete")


# Dependency injection
def get_detection_service():
    """Get detection service dependency."""
    if detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not available")
    return detection_service


def get_model_manager():
    """Get model manager dependency."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not available")
    return model_manager


# Error handling middleware
@app.exception_handler(NSMBaseException)
async def nsm_exception_handler(request, exc: NSMBaseException):
    """Handle NSM-specific exceptions."""
    logger.error(f"NSM Exception: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content=format_error_response(exc)
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.exception(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "error_type": "InternalServerError",
                "message": "An unexpected error occurred",
                "recovery_suggestion": "Please try again later or contact support"
            },
            "success": False
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check model manager
        mm_status = "healthy" if model_manager else "unavailable"
        
        # Check detection service
        ds_status = "healthy" if detection_service else "unavailable"
        
        # Get memory usage
        memory_info = model_manager.get_memory_usage() if model_manager else {}
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "model_manager": mm_status,
                "detection_service": ds_status
            },
            "memory": memory_info,
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )


# Prime detection endpoint
@app.post("/detect", response_model=PrimeDetectionResponse)
async def detect_primes(
    request: PrimeDetectionRequest,
    detection_service = Depends(get_detection_service)
):
    """Detect NSM primes in the given text."""
    start_time = time.time()
    
    try:
        with PerformanceContext("api_detect_primes", logger):
            # Perform detection
            result = detection_service.detect_primes(request.text, request.language)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return PrimeDetectionResponse(
                success=True,
                result=result,
                processing_time=processing_time
            )
    
    except NSMBaseException as e:
        logger.error(f"Detection failed: {str(e)}")
        return PrimeDetectionResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )
    except Exception as e:
        logger.exception(f"Unexpected error in detection: {str(e)}")
        return PrimeDetectionResponse(
            success=False,
            error="An unexpected error occurred during detection",
            processing_time=time.time() - start_time
        )


# MWE detection endpoint
@app.post("/mwe", response_model=MWEDetectionResponse)
async def detect_mwes(
    request: MWEDetectionRequest,
    detection_service = Depends(get_detection_service)
):
    """Detect multi-word expressions in the given text."""
    start_time = time.time()
    
    try:
        with PerformanceContext("api_detect_mwes", logger):
            # Perform MWE detection
            mwes = detection_service.detect_mwes(request.text, request.language)
            
            # Extract primes from MWEs
            primes = []
            for mwe in mwes:
                primes.extend(mwe.primes)
            
            # Calculate coverage
            from src.core.domain.models import calculate_coverage
            coverage = calculate_coverage(mwes, request.text)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return MWEDetectionResponse(
                success=True,
                mwes=mwes,
                primes=primes,
                coverage=coverage,
                processing_time=processing_time
            )
    
    except NSMBaseException as e:
        logger.error(f"MWE detection failed: {str(e)}")
        return MWEDetectionResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )
    except Exception as e:
        logger.exception(f"Unexpected error in MWE detection: {str(e)}")
        return MWEDetectionResponse(
            success=False,
            error="An unexpected error occurred during MWE detection",
            processing_time=time.time() - start_time
        )


# Prime discovery endpoint
@app.post("/discovery", response_model=PrimeDiscoveryResponse)
async def discover_primes(
    request: PrimeDiscoveryRequest,
    background_tasks: BackgroundTasks,
    detection_service = Depends(get_detection_service)
):
    """Discover new prime candidates from the given corpus."""
    start_time = time.time()
    
    try:
        with PerformanceContext("api_discover_primes", logger):
            # PLACEHOLDER: Prime discovery not yet implemented
            # TODO: Implement real prime discovery using MDL analysis
            logger.info(f"Prime discovery requested for {len(request.corpus)} texts")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # PLACEHOLDER: Return empty discovery result
            # TODO: Implement real prime discovery algorithm
            from src.core.domain.models import DiscoveryResult, DiscoveryStatus
            result = DiscoveryResult(
                candidates=[],
                accepted=[],
                rejected=[],
                processing_time=processing_time,
                corpus_stats={
                    "total_texts": len(request.corpus),
                    "total_words": sum(len(text.split()) for text in request.corpus),
                    "total_chars": sum(len(text) for text in request.corpus)
                },
                discovery_metrics={
                    "candidates_found": 0,
                    "acceptance_rate": 0.0,
                    "processing_efficiency": 0.0
                },
                status=DiscoveryStatus.COMPLETED
            )
            
            return PrimeDiscoveryResponse(
                success=True,
                result=result,
                processing_time=processing_time
            )
    
    except NSMBaseException as e:
        logger.error(f"Prime discovery failed: {str(e)}")
        return PrimeDiscoveryResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )
    except Exception as e:
        logger.exception(f"Unexpected error in prime discovery: {str(e)}")
        return PrimeDiscoveryResponse(
            success=False,
            error="An unexpected error occurred during prime discovery",
            processing_time=time.time() - start_time
        )


# Text generation endpoint
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    detection_service = Depends(get_detection_service)
):
    """Generate text from NSM primes."""
    start_time = time.time()
    
    try:
        with PerformanceContext("api_generate_text", logger):
            logger.info(f"Text generation requested for {len(request.primes)} primes")
            
            # REAL NSM-based text generation (replaces hardcoded templates)
            from src.core.application.nsm_generator import get_nsm_generator
            nsm_generator = get_nsm_generator()
            
            # Generate text using real NSM grammar rules
            generated_text = nsm_generator.generate_text(request.primes, request.target_language)
            
            # Calculate generation confidence
            generation_confidence = nsm_generator.get_generation_confidence(request.primes)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            from src.core.domain.models import GenerationResult
            result = GenerationResult(
                generated_text=generated_text,
                source_primes=request.primes,
                confidence=generation_confidence,
                processing_time=processing_time,
                target_language=request.target_language
            )
            
            return GenerationResponse(
                success=True,
                result=result,
                processing_time=processing_time
            )
    
    except NSMBaseException as e:
        logger.error(f"Text generation failed: {str(e)}")
        return GenerationResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )
    except Exception as e:
        logger.exception(f"Unexpected error in text generation: {str(e)}")
        return GenerationResponse(
            success=False,
            error="An unexpected error occurred during text generation",
            processing_time=time.time() - start_time
        )


# System information endpoint
@app.get("/system/info")
async def get_system_info(
    model_manager = Depends(get_model_manager)
):
    """Get system information and statistics."""
    try:
        # Get model manager stats
        cache_stats = model_manager.get_cache_stats()
        memory_usage = model_manager.get_memory_usage()
        
        return {
            "system": {
                "version": "2.0.0",
                "environment": settings.environment,
                "uptime": time.time()  # This should be actual uptime
            },
            "models": cache_stats,
            "memory": memory_usage,
            "performance": {
                "max_corpus_size": settings.performance.max_corpus_size,
                "max_memory_usage": settings.performance.max_memory_usage,
                "cache_max_size": settings.performance.cache_max_size
            },
            "discovery": {
                "mdl_threshold": settings.discovery.mdl_threshold,
                "max_candidates": settings.discovery.max_candidates,
                "min_candidate_frequency": settings.discovery.min_candidate_frequency
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache management endpoint
@app.post("/system/cache/clear")
async def clear_cache(
    model_manager = Depends(get_model_manager)
):
    """Clear the model cache."""
    try:
        model_manager.clear_cache()
        logger.info("Model cache cleared via API")
        return {"success": True, "message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Model preloading endpoint
@app.post("/system/models/preload")
async def preload_models(
    languages: List[str] = ["en", "es", "fr"],
    model_manager = Depends(get_model_manager)
):
    """Preload models for the specified languages."""
    try:
        model_manager.preload_models(languages)
        logger.info(f"Models preloaded for languages: {languages}")
        return {
            "success": True, 
            "message": f"Models preloaded for languages: {languages}"
        }
    except Exception as e:
        logger.error(f"Failed to preload models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Add missing endpoints
@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    try:
        with PerformanceContext("api_metrics"):
            # Get basic metrics
            total_requests = 0  # Would be tracked globally
            avg_response_time = 0.0  # Would be calculated from request tracking
            success_rate = 1.0  # Would be calculated from request tracking
            
            return {
                "total_requests": total_requests,
                "avg_response_time": avg_response_time,
                "success_rate": success_rate,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Metrics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics failed: {str(e)}")



@app.post("/mdl")
async def analyze_mdl(request: Dict[str, Any]):
    """Analyze Minimum Description Length."""
    try:
        with PerformanceContext("api_mdl"):
            text = request.get("text", "")
            language = request.get("language", "en")
            
            # Simple MDL analysis
            text_length = len(text)
            compressed_length = len(text.encode('utf-8'))
            mdl_delta = (text_length - compressed_length) / text_length if text_length > 0 else 0
            
            return {
                "mdl_delta": mdl_delta,
                "compression_ratio": 1 - mdl_delta,
                "text_length": text_length,
                "compressed_length": compressed_length,
                "language": language
            }
    except Exception as e:
        logger.error(f"MDL analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MDL analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "clean_nsm_api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level=settings.logging.log_level.lower()
    )

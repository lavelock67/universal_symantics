"""
Production API for Universal Translator
FastAPI application with comprehensive endpoints, health checks, and monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.core.translation.production_pipeline_orchestrator import (
    ProductionPipelineOrchestrator,
    ProductionTranslationRequest,
    ProductionTranslationResult,
    PipelineMode,
    QualityLevel
)
from src.core.application.health_panel import get_health_panel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Universal Translator API",
    description="Production-ready universal translation system with NSM primes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline orchestrator
pipeline_orchestrator = ProductionPipelineOrchestrator()

# Pydantic models for API
class TranslationRequest(BaseModel):
    source_text: str = Field(..., description="Source text to translate")
    source_language: str = Field(..., description="Source language code (e.g., 'en', 'es', 'fr')")
    target_language: str = Field(..., description="Target language code")
    mode: str = Field(default="hybrid", description="Translation mode: standard, neural, hybrid, research")
    quality_level: str = Field(default="standard", description="Quality level: basic, standard, high, research")
    glossary_terms: Optional[Dict[str, str]] = Field(default=None, description="Domain-specific glossary terms")
    cultural_context: Optional[Dict[str, Any]] = Field(default=None, description="Cultural context for adaptation")
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")
    enable_observability: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_guarantees: bool = Field(default=True, description="Enable quality guarantees")

class TranslationResponse(BaseModel):
    translated_text: str
    source_text: str
    source_language: str
    target_language: str
    mode: str
    quality_level: str
    confidence_score: float
    success: bool
    error_message: Optional[str] = None
    metrics: Dict[str, Any]
    detected_primes: List[str]
    cultural_adaptations: List[str]
    glossary_preserved: List[str]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    uptime: float
    request_count: int
    error_count: int
    error_rate: float
    components: Dict[str, str]

class BatchTranslationRequest(BaseModel):
    translations: List[TranslationRequest]
    parallel_processing: bool = Field(default=True, description="Process translations in parallel")

class BatchTranslationResponse(BaseModel):
    results: List[TranslationResponse]
    total_processing_time: float
    success_count: int
    error_count: int

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Universal Translator API")
    # Initialize any additional resources here

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Universal Translator API")
    # Cleanup resources here

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Get system health status"""
    start_time = time.time()
    health_status = pipeline_orchestrator.get_health_status()
    
    return HealthResponse(
        status=health_status["status"],
        timestamp=start_time,
        uptime=start_time,  # Simplified uptime calculation
        request_count=health_status["request_count"],
        error_count=health_status["error_count"],
        error_rate=health_status["error_rate"],
        components=health_status["components"]
    )


@app.get("/health/panel")
async def health_panel():
    """Comprehensive health panel with pipeline integrity metrics."""
    try:
        health_panel = get_health_panel()
        return health_panel.get_health_status()
    except Exception as e:
        logger.error(f"Health panel failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health panel failed: {e}")


@app.get("/health/performance")
async def performance_metrics():
    """Detailed performance metrics with histograms."""
    try:
        health_panel = get_health_panel()
        return health_panel.get_performance_metrics()
    except Exception as e:
        logger.error(f"Performance metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance metrics failed: {e}")


@app.get("/health/integrity")
async def pipeline_integrity():
    """Pipeline integrity report."""
    try:
        health_panel = get_health_panel()
        return health_panel.get_pipeline_integrity_report()
    except Exception as e:
        logger.error(f"Pipeline integrity report failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline integrity report failed: {e}")

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Main translation endpoint
@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate text using the universal translator pipeline"""
    start_time = time.time()
    
    try:
        # Convert API request to pipeline request
        pipeline_request = ProductionTranslationRequest(
            source_text=request.source_text,
            source_language=request.source_language,
            target_language=request.target_language,
            mode=PipelineMode(request.mode),
            quality_level=QualityLevel(request.quality_level),
            glossary_terms=request.glossary_terms,
            cultural_context=request.cultural_context,
            timeout_seconds=request.timeout_seconds,
            enable_observability=request.enable_observability,
            enable_guarantees=request.enable_guarantees
        )
        
        # Execute translation
        result = await pipeline_orchestrator.translate(pipeline_request)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Convert to API response
        response = TranslationResponse(
            translated_text=result.translated_text,
            source_text=result.source_text,
            source_language=result.source_language,
            target_language=result.target_language,
            mode=result.mode.value,
            quality_level=result.quality_level.value,
            confidence_score=result.confidence_score,
            success=result.success,
            error_message=result.error_message,
            metrics={
                "total_duration": result.metrics.total_duration,
                "detection_duration": result.metrics.detection_duration,
                "decomposition_duration": result.metrics.decomposition_duration,
                "generation_duration": result.metrics.generation_duration,
                "adaptation_duration": result.metrics.adaptation_duration,
                "prime_count": result.metrics.prime_count,
                "graph_f1_score": result.metrics.graph_f1_score,
                "cultural_adaptations": result.metrics.cultural_adaptations
            },
            detected_primes=result.detected_primes,
            cultural_adaptations=result.cultural_adaptations,
            glossary_preserved=result.glossary_preserved,
            processing_time=processing_time
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message)
        
        return response
        
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# Batch translation endpoint
@app.post("/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(request: BatchTranslationRequest):
    """Translate multiple texts in batch"""
    start_time = time.time()
    results = []
    
    try:
        if request.parallel_processing:
            # Process translations in parallel
            tasks = []
            for translation_request in request.translations:
                pipeline_request = ProductionTranslationRequest(
                    source_text=translation_request.source_text,
                    source_language=translation_request.source_language,
                    target_language=translation_request.target_language,
                    mode=PipelineMode(translation_request.mode),
                    quality_level=QualityLevel(translation_request.quality_level),
                    glossary_terms=translation_request.glossary_terms,
                    cultural_context=translation_request.cultural_context,
                    timeout_seconds=translation_request.timeout_seconds,
                    enable_observability=translation_request.enable_observability,
                    enable_guarantees=translation_request.enable_guarantees
                )
                tasks.append(pipeline_orchestrator.translate(pipeline_request))
            
            # Execute all translations concurrently
            pipeline_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert results
            for i, result in enumerate(pipeline_results):
                if isinstance(result, Exception):
                    # Handle exception
                    response = TranslationResponse(
                        translated_text="",
                        source_text=request.translations[i].source_text,
                        source_language=request.translations[i].source_language,
                        target_language=request.translations[i].target_language,
                        mode=request.translations[i].mode,
                        quality_level=request.translations[i].quality_level,
                        confidence_score=0.0,
                        success=False,
                        error_message=str(result),
                        metrics={},
                        detected_primes=[],
                        cultural_adaptations=[],
                        glossary_preserved=[],
                        processing_time=0.0
                    )
                else:
                    # Convert successful result
                    response = TranslationResponse(
                        translated_text=result.translated_text,
                        source_text=result.source_text,
                        source_language=result.source_language,
                        target_language=result.target_language,
                        mode=result.mode.value,
                        quality_level=result.quality_level.value,
                        confidence_score=result.confidence_score,
                        success=result.success,
                        error_message=result.error_message,
                        metrics={
                            "total_duration": result.metrics.total_duration,
                            "detection_duration": result.metrics.detection_duration,
                            "decomposition_duration": result.metrics.decomposition_duration,
                            "generation_duration": result.metrics.generation_duration,
                            "adaptation_duration": result.metrics.adaptation_duration,
                            "prime_count": result.metrics.prime_count,
                            "graph_f1_score": result.metrics.graph_f1_score,
                            "cultural_adaptations": result.metrics.cultural_adaptations
                        },
                        detected_primes=result.detected_primes,
                        cultural_adaptations=result.cultural_adaptations,
                        glossary_preserved=result.glossary_preserved,
                        processing_time=result.metrics.total_duration
                    )
                results.append(response)
        else:
            # Process translations sequentially
            for translation_request in request.translations:
                pipeline_request = ProductionTranslationRequest(
                    source_text=translation_request.source_text,
                    source_language=translation_request.source_language,
                    target_language=translation_request.target_language,
                    mode=PipelineMode(translation_request.mode),
                    quality_level=QualityLevel(translation_request.quality_level),
                    glossary_terms=translation_request.glossary_terms,
                    cultural_context=translation_request.cultural_context,
                    timeout_seconds=translation_request.timeout_seconds,
                    enable_observability=translation_request.enable_observability,
                    enable_guarantees=translation_request.enable_guarantees
                )
                
                result = await pipeline_orchestrator.translate(pipeline_request)
                
                response = TranslationResponse(
                    translated_text=result.translated_text,
                    source_text=result.source_text,
                    source_language=result.source_language,
                    target_language=result.target_language,
                    mode=result.mode.value,
                    quality_level=result.quality_level.value,
                    confidence_score=result.confidence_score,
                    success=result.success,
                    error_message=result.error_message,
                    metrics={
                        "total_duration": result.metrics.total_duration,
                        "detection_duration": result.metrics.detection_duration,
                        "decomposition_duration": result.metrics.decomposition_duration,
                        "generation_duration": result.metrics.generation_duration,
                        "adaptation_duration": result.metrics.adaptation_duration,
                        "prime_count": result.metrics.prime_count,
                        "graph_f1_score": result.metrics.graph_f1_score,
                        "cultural_adaptations": result.metrics.cultural_adaptations
                    },
                    detected_primes=result.detected_primes,
                    cultural_adaptations=result.cultural_adaptations,
                    glossary_preserved=result.glossary_preserved,
                    processing_time=result.metrics.total_duration
                )
                results.append(response)
        
        # Calculate batch statistics
        total_processing_time = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        error_count = len(results) - success_count
        
        return BatchTranslationResponse(
            results=results,
            total_processing_time=total_processing_time,
            success_count=success_count,
            error_count=error_count
        )
        
    except Exception as e:
        logger.error(f"Batch translation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(e)}")

# Performance metrics endpoint
@app.get("/metrics/performance")
async def get_performance_metrics():
    """Get detailed performance metrics"""
    return pipeline_orchestrator.get_performance_metrics()

# Supported languages endpoint
@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"}
        ],
        "translation_modes": [
            {"mode": "standard", "description": "Prime-based translation"},
            {"mode": "neural", "description": "Neural translation with guarantees"},
            {"mode": "hybrid", "description": "Combined approach with fallback"},
            {"mode": "research", "description": "Full pipeline with analysis"}
        ],
        "quality_levels": [
            {"level": "basic", "description": "Fast translation with minimal guarantees"},
            {"level": "standard", "description": "Balanced quality and performance"},
            {"level": "high", "description": "High quality with comprehensive analysis"},
            {"level": "research", "description": "Maximum quality for research use"}
        ]
    }

# Debug endpoint for language assets
@app.get("/debug/lang_assets")
async def get_language_assets():
    """Get information about loaded language assets (UD, SRL, MWE models)"""
    try:
        # Get orchestrator instance
        orchestrator = pipeline_orchestrator
        
        # Collect asset information
        assets = {}
        
        # Check detection service assets
        if hasattr(orchestrator.detection_service, 'missing_prime_detector'):
            detector = orchestrator.detection_service.missing_prime_detector
            if detector:
                assets['missing_prime_detector'] = {
                    'status': 'loaded',
                    'models': list(detector.models.keys()) if hasattr(detector, 'models') else []
                }
        
        # Check UD+SRL engine assets
        if hasattr(orchestrator, 'ud_srl_engine'):
            ud_srl = orchestrator.ud_srl_engine
            assets['ud_srl_engine'] = {
                'status': 'loaded',
                'models': getattr(ud_srl, 'models', {}),
                'patterns': {
                    'srl_patterns': len(getattr(ud_srl, 'srl_patterns', {})),
                    'ud_role_patterns': len(getattr(ud_srl, 'ud_role_patterns', {}))
                }
            }
        
        # Check neural generator assets
        if hasattr(orchestrator, 'neural_generator'):
            neural = orchestrator.neural_generator
            assets['neural_generator'] = {
                'status': 'loaded',
                'model_type': getattr(neural, 'model_type', 'unknown'),
                'languages': getattr(neural, 'supported_languages', [])
            }
        
        # Check cultural adapter assets
        if hasattr(orchestrator, 'cultural_adapter'):
            adapter = orchestrator.cultural_adapter
            assets['cultural_adapter'] = {
                'status': 'loaded',
                'contexts': len(getattr(adapter, 'cultural_contexts', {})),
                'expressions': len(getattr(adapter, 'idiomatic_expressions', {}))
            }
        
        # Check entity extractor assets
        if hasattr(orchestrator, 'entity_extractor'):
            extractor = orchestrator.entity_extractor
            assets['entity_extractor'] = {
                'status': 'loaded',
                'ner_model': getattr(extractor, 'ner_model', 'unknown')
            }
        
        # Check knowledge graph integrator
        if hasattr(orchestrator, 'kg_integrator'):
            kg = orchestrator.kg_integrator
            assets['knowledge_graph'] = {
                'status': 'loaded',
                'endpoint': getattr(kg, 'wikidata_endpoint', 'unknown')
            }
        
        # Supported languages
        supported_languages = [
            {"code": "en", "name": "English", "status": "supported"},
            {"code": "es", "name": "Spanish", "status": "supported"},
            {"code": "fr", "name": "French", "status": "supported"},
            {"code": "de", "name": "German", "status": "supported"},
            {"code": "it", "name": "Italian", "status": "supported"},
            {"code": "pt", "name": "Portuguese", "status": "supported"},
            {"code": "ru", "name": "Russian", "status": "supported"},
            {"code": "zh", "name": "Chinese", "status": "supported"},
            {"code": "ja", "name": "Japanese", "status": "supported"},
            {"code": "ko", "name": "Korean", "status": "supported"}
        ]
        
        return {
            "timestamp": time.time(),
            "assets": assets,
            "supported_languages": supported_languages,
            "total_assets": len(assets),
            "status": "operational"
        }
        
    except Exception as e:
        logger.error(f"Failed to get language assets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get language assets: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Universal Translator API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "translate": "/translate",
            "batch_translate": "/translate/batch",
            "health": "/health",
            "metrics": "/metrics",
            "performance": "/metrics/performance",
            "languages": "/languages",
            "documentation": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

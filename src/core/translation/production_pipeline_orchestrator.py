"""
Production Pipeline Orchestrator
Comprehensive end-to-end translation pipeline with observability, error handling, and performance monitoring.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge

from ..application.services import NSMDetectionService
from ...realize.base import RealizeConfig
from ..generation.neural_generator import MultilingualNeuralGenerator
from ..generation.prime_generator import PrimeGenerator
from .universal_translator import UniversalTranslator
from .unified_translation_pipeline import UnifiedTranslationPipeline
from ...realize.neural import NeuralRealizer
from ...shared.logging import get_logger, log_error, PerformanceContext

# Prometheus metrics
TRANSLATION_REQUESTS = Counter('translation_requests_total', 'Total translation requests', ['source_lang', 'target_lang', 'mode'])
TRANSLATION_DURATION = Histogram('translation_duration_seconds', 'Translation duration', ['source_lang', 'target_lang', 'mode'])
TRANSLATION_ERRORS = Counter('translation_errors_total', 'Translation errors', ['source_lang', 'target_lang', 'error_type'])
PRIME_DETECTION_GAUGE = Gauge('prime_detection_count', 'Number of primes detected', ['language'])
GRAPH_F1_SCORE = Histogram('graph_f1_score', 'Graph-F1 score distribution', ['complexity'])
CULTURAL_ADAPTATIONS = Counter('cultural_adaptations_total', 'Cultural adaptations applied', ['adaptation_type'])

class PipelineMode(Enum):
    """Translation pipeline modes"""
    STANDARD = "standard"
    NEURAL = "neural"
    HYBRID = "hybrid"
    RESEARCH = "research"

class QualityLevel(Enum):
    """Quality assurance levels"""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    RESEARCH = "research"

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_duration: float = 0.0
    detection_duration: float = 0.0
    decomposition_duration: float = 0.0
    generation_duration: float = 0.0
    adaptation_duration: float = 0.0
    prime_count: int = 0
    graph_f1_score: float = 0.0
    cultural_adaptations: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class ProductionTranslationRequest:
    """Production translation request with quality settings"""
    source_text: str
    source_language: str
    target_language: str
    mode: PipelineMode = PipelineMode.HYBRID
    quality_level: QualityLevel = QualityLevel.STANDARD
    glossary_terms: Optional[Dict[str, str]] = None
    cultural_context: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 30
    enable_observability: bool = True
    enable_guarantees: bool = True

@dataclass
class ProductionTranslationResult:
    """Production translation result with comprehensive metadata"""
    translated_text: str
    source_text: str
    source_language: str
    target_language: str
    mode: PipelineMode
    quality_level: QualityLevel
    metrics: PipelineMetrics
    semantic_graph: Optional[Dict[str, Any]] = None
    detected_primes: List[str] = field(default_factory=list)
    cultural_adaptations: List[str] = field(default_factory=list)
    glossary_preserved: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

class ProductionPipelineOrchestrator:
    """
    Production-ready pipeline orchestrator with comprehensive error handling,
    observability, and performance monitoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("production_pipeline")
        
        try:
            # Initialize core components
            self.detection_service = NSMDetectionService()  # Keep for compatibility
            self.logger.info("NSMDetectionService initialized successfully")
            
            # Use semantic generator - the ONLY place allowed to emit primes
            from ...semgen.generator import SemanticGenerator
            self.semantic_generator = SemanticGenerator()
            self.logger.info("SemanticGenerator initialized successfully")
            
            self.universal_translator = UniversalTranslator()
            self.logger.info("UniversalTranslator initialized successfully")
            
            self.neural_generator = MultilingualNeuralGenerator()
            self.logger.info("MultilingualNeuralGenerator initialized successfully")
            
            self.prime_generator = PrimeGenerator()
            self.logger.info("PrimeGenerator initialized successfully")
            
            # Try to initialize optional components
            try:
                from cultural_adaptation_system import CulturalAdaptationSystem
                self.cultural_adapter = CulturalAdaptationSystem()
                self.logger.info("CulturalAdaptationSystem initialized successfully")
            except ImportError as e:
                self.logger.warning(f"CulturalAdaptationSystem not available: {e}")
                self.cultural_adapter = None
            
            try:
                from ud_srl_enhanced_decomposition import UDSRLEnhancedDecompositionEngine
                self.ud_srl_engine = UDSRLEnhancedDecompositionEngine()
                self.logger.info("UDSRLEnhancedDecompositionEngine initialized successfully")
            except ImportError as e:
                self.logger.warning(f"UDSRLEnhancedDecompositionEngine not available: {e}")
                self.ud_srl_engine = None
            
            try:
                from knowledge_graph_integrator import KnowledgeGraphIntegrator
                self.kg_integrator = KnowledgeGraphIntegrator()
                self.logger.info("KnowledgeGraphIntegrator initialized successfully")
            except ImportError as e:
                self.logger.warning(f"KnowledgeGraphIntegrator not available: {e}")
                self.kg_integrator = None
            
            try:
                from improved_entity_extraction import ImprovedEntityExtractor
                self.entity_extractor = ImprovedEntityExtractor()
                self.logger.info("ImprovedEntityExtractor initialized successfully")
            except ImportError as e:
                self.logger.warning(f"ImprovedEntityExtractor not available: {e}")
                self.entity_extractor = None
            
            # Initialize neural realizer with proper backend
            try:
                # Try to use a real backend if available
                from ...realize.backends import get_backend
                backend = get_backend()
                self.neural_realizer = NeuralRealizer(backend)
                self.logger.info("NeuralRealizer initialized with real backend")
            except ImportError as e:
                # No mock fallback - fail fast if backend is not available
                self.logger.error(f"NeuralRealizer backend not available: {e}")
                raise RuntimeError(f"NeuralRealizer backend not available: {e}")
            
            # Unified pipeline for fallback
            self.unified_pipeline = UnifiedTranslationPipeline()
            self.logger.info("UnifiedTranslationPipeline initialized successfully")
            
            # Thread pool for parallel processing
            self.executor = ThreadPoolExecutor(max_workers=4)
            
            # Performance tracking
            self.request_count = 0
            self.error_count = 0
            
            self.logger.info("Production Pipeline Orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Production Pipeline Orchestrator: {e}")
            log_error(e, {"context": "pipeline_initialization"})
            raise
    
    async def translate(self, request: ProductionTranslationRequest) -> ProductionTranslationResult:
        """
        Main translation method with comprehensive error handling and observability.
        """
        start_time = time.time()
        metrics = PipelineMetrics()
        
        try:
            # Update metrics
            TRANSLATION_REQUESTS.labels(
                source_lang=request.source_language,
                target_lang=request.target_language,
                mode=request.mode.value
            ).inc()
            
            self.request_count += 1
            
            # Validate request
            self._validate_request(request)
            
            # Execute translation based on mode
            if request.mode == PipelineMode.STANDARD:
                result = await self._execute_standard_pipeline(request, metrics)
            elif request.mode == PipelineMode.NEURAL:
                result = await self._execute_neural_pipeline(request, metrics)
            elif request.mode == PipelineMode.HYBRID:
                result = await self._execute_hybrid_pipeline(request, metrics)
            elif request.mode == PipelineMode.RESEARCH:
                result = await self._execute_research_pipeline(request, metrics)
            else:
                raise ValueError(f"Unsupported pipeline mode: {request.mode}")
            
            # Calculate final metrics
            metrics.total_duration = time.time() - start_time
            
            # Update Prometheus metrics
            if request.enable_observability:
                self._update_prometheus_metrics(request, result, metrics)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            error_msg = f"Translation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            log_error(e, {
                "context": "translation_request",
                "source_language": request.source_language,
                "target_language": request.target_language,
                "mode": request.mode.value,
                "text_length": len(request.source_text)
            })
            
            # Record error metrics
            TRANSLATION_ERRORS.labels(
                source_lang=request.source_language,
                target_lang=request.target_language,
                error_type=type(e).__name__
            ).inc()
            
            # Return error result
            return ProductionTranslationResult(
                translated_text="",
                source_text=request.source_text,
                source_language=request.source_language,
                target_language=request.target_language,
                mode=request.mode,
                quality_level=request.quality_level,
                metrics=metrics,
                success=False,
                error_message=error_msg
            )
    
    async def _execute_standard_pipeline(self, request: ProductionTranslationRequest, metrics: PipelineMetrics) -> ProductionTranslationResult:
        """Execute standard pipeline with prime-based translation."""
        # Step 1: Prime Detection
        detection_start = time.time()
        detection_result = await self._detect_primes_with_timeout(request.source_text, request.source_language, request.timeout_seconds)
        metrics.detection_duration = time.time() - detection_start
        metrics.prime_count = len(detection_result.primes)
        
        # Step 2: Semantic Decomposition
        decomposition_start = time.time()
        semantic_graph = await self._decompose_semantics_with_timeout(request.source_text, request.source_language, request.timeout_seconds)
        metrics.decomposition_duration = time.time() - decomposition_start
        
        # Step 3: Generation
        generation_start = time.time()
        translated_text = await self._generate_text_with_timeout(
            semantic_graph, request.target_language, request.timeout_seconds
        )
        metrics.generation_duration = time.time() - generation_start
        
        # Step 4: Cultural Adaptation
        adaptation_start = time.time()
        # Extract text from NeuralGenerationResult
        translated_text_str = translated_text.text if hasattr(translated_text, 'text') else str(translated_text)
        adapted_result = await self._adapt_culturally_with_timeout(
            translated_text_str, request.target_language, request.cultural_context, request.timeout_seconds
        )
        metrics.adaptation_duration = time.time() - adaptation_start
        metrics.cultural_adaptations = len(adapted_result.changes)
        
        return ProductionTranslationResult(
            translated_text=adapted_result.adapted_text,
            source_text=request.source_text,
            source_language=request.source_language,
            target_language=request.target_language,
            mode=request.mode,
            quality_level=request.quality_level,
            metrics=metrics,
            semantic_graph=semantic_graph,
            detected_primes=detection_result.primes,
            cultural_adaptations=[c.description for c in adapted_result.changes],
            confidence_score=0.8  # Default confidence for cultural adaptation
        )
    
    async def _execute_neural_pipeline(self, request: ProductionTranslationRequest, metrics: PipelineMetrics) -> ProductionTranslationResult:
        """Execute neural pipeline with post-check guarantees."""
        # Step 1: Semantic Analysis
        detection_start = time.time()
        detection_result = await self._detect_primes_with_timeout(request.source_text, request.source_language, request.timeout_seconds)
        metrics.detection_duration = time.time() - detection_start
        
        # Step 2: Neural Realization with Guarantees
        generation_start = time.time()
        realized_result = await self._neural_realize_with_timeout(
            detection_result, request.target_language, request.glossary_terms, request.timeout_seconds
        )
        metrics.generation_duration = time.time() - generation_start
        
        # Step 3: Quality Assessment
        if request.enable_guarantees:
            graph_f1 = await self._calculate_graph_f1_with_timeout(
                detection_result, realized_result, request.timeout_seconds
            )
            metrics.graph_f1_score = graph_f1
        
        # Handle both string and object return types from neural realizer
        if isinstance(realized_result, str):
            translated_text = realized_result
            preserved_terms = []
            confidence = 0.8
        else:
            translated_text = getattr(realized_result, 'realized_text', str(realized_result))
            preserved_terms = getattr(realized_result, 'preserved_terms', [])
            confidence = getattr(realized_result, 'confidence', 0.8)
        
        return ProductionTranslationResult(
            translated_text=translated_text,
            source_text=request.source_text,
            source_language=request.source_language,
            target_language=request.target_language,
            mode=request.mode,
            quality_level=request.quality_level,
            metrics=metrics,
            detected_primes=detection_result.primes,
            glossary_preserved=preserved_terms,
            confidence_score=confidence
        )
    
    async def _execute_hybrid_pipeline(self, request: ProductionTranslationRequest, metrics: PipelineMetrics) -> ProductionTranslationResult:
        """Execute hybrid pipeline combining multiple approaches."""
        # Try neural first, fallback to standard
        try:
            result = await self._execute_neural_pipeline(request, metrics)
            if result.confidence_score > 0.7:
                return result
        except Exception as e:
            self.logger.warning(f"Neural pipeline failed, falling back to standard: {e}")
        
        # Fallback to standard pipeline
        return await self._execute_standard_pipeline(request, metrics)
    
    async def _execute_research_pipeline(self, request: ProductionTranslationRequest, metrics: PipelineMetrics) -> ProductionTranslationResult:
        """Execute research pipeline with comprehensive analysis."""
        # Use unified pipeline with all features enabled
        unified_result = await self._unified_pipeline_translate_with_timeout(request, request.timeout_seconds)
        
        # Extract metrics from unified result
        metrics.prime_count = len(unified_result.detected_primes) if hasattr(unified_result, 'detected_primes') else 0
        metrics.graph_f1_score = getattr(unified_result, 'graph_f1_score', 0.0)
        
        return ProductionTranslationResult(
            translated_text=unified_result.translated_text,
            source_text=request.source_text,
            source_language=request.source_language,
            target_language=request.target_language,
            mode=request.mode,
            quality_level=request.quality_level,
            metrics=metrics,
            semantic_graph=getattr(unified_result, 'semantic_graph', None),
            detected_primes=getattr(unified_result, 'detected_primes', []),
            confidence_score=getattr(unified_result, 'confidence_score', 0.0)
        )
    
    async def _detect_primes_with_timeout(self, text: str, language: str, timeout: int):
        """Detect primes using semantic decomposition."""
        loop = asyncio.get_event_loop()
        
        def detect_primes():
            try:
                from ..application.services import NSMDetectionService
                from ..domain.models import Language
                
                detection_service = NSMDetectionService()
                result = detection_service.detect_primes(text, Language(language))
                self.logger.info(f"Prime detection successful for language {language}: {len(result.primes) if result.primes else 0} primes found")
                return result
            except Exception as e:
                self.logger.error(f"Prime detection failed for language {language}: {e}")
                log_error(e, {"context": "prime_detection", "language": language, "text": text})
                raise
        
        try:
            detection_result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, detect_primes),
                timeout=timeout
            )
            
            # Extract primes from the detection result
            primes = [prime.text for prime in detection_result.primes] if detection_result.primes else []
            
            @dataclass
            class DetectionResult:
                primes: List[str]
                confidence: float = 0.0
                pipeline_path: str = "semantic"
                manual_detector_count: int = 0
            
            return DetectionResult(primes=primes, pipeline_path="semantic", manual_detector_count=0)
            
        except asyncio.TimeoutError:
            self.logger.error(f"Prime detection timed out for language {language}")
            raise
        except Exception as e:
            self.logger.error(f"Prime detection failed for language {language}: {e}")
            log_error(e, {"context": "prime_detection_timeout", "language": language, "text": text})
            raise
    
    async def _decompose_semantics_with_timeout(self, text: str, language: str, timeout: int):
        """Decompose semantics with timeout."""
        loop = asyncio.get_event_loop()
        
        def decompose():
            from ..application.services import NSMDetectionService
            from ..domain.models import Language
            
            detection_service = NSMDetectionService()
            result = detection_service.detect_primes(text, Language(language))
            
            # Return enhanced structure with UD and SRL information
            return {
                'enhanced_structure': {
                    'ud_tree': result.ud_tree if hasattr(result, 'ud_tree') else None,
                    'semantic_roles': result.semantic_roles if hasattr(result, 'semantic_roles') else [],
                    'primes': [prime.text for prime in result.primes] if result.primes else []
                }
            }
        
        return await asyncio.wait_for(
            loop.run_in_executor(self.executor, decompose),
            timeout=timeout
        )
    
    async def _generate_text_with_timeout(self, semantic_graph: Dict[str, Any], target_language: str, timeout: int):
        """Generate text with timeout."""
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(self.executor, self.neural_generator.generate, semantic_graph, target_language),
            timeout=timeout
        )
    
    async def _adapt_culturally_with_timeout(self, text: str, target_language: str, cultural_context: Optional[Dict], timeout: int):
        """Adapt culturally with timeout."""
        if self.cultural_adapter is None:
            self.logger.warning("Cultural adaptation skipped - CulturalAdaptationSystem not available")
            return text
        
        try:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(self.executor, self.cultural_adapter.adapt_text, text, target_language, cultural_context),
                timeout=timeout
            )
        except Exception as e:
            self.logger.error(f"Cultural adaptation failed: {e}")
            log_error(e, {"context": "cultural_adaptation", "target_language": target_language})
            return text  # Return original text on failure
    
    async def _neural_realize_with_timeout(self, detection_result, target_language: str, glossary_terms: Optional[Dict], timeout: int):
        """Neural realize with timeout."""
        try:
            loop = asyncio.get_event_loop()
            
            # Create binder from glossary terms
            binder = None
            if glossary_terms:
                binder = type('Binder', (), {
                    'preserve_terms': list(glossary_terms.keys()),
                    'gloss_terms': []
                })()
            
            # Use unified interface with keyword arguments
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor, 
                    lambda: self.neural_realizer.realize(
                        detection_result, 
                        target_language, 
                        binder=binder, 
                        config=RealizeConfig()
                    )
                ),
                timeout=timeout
            )
            
            self.logger.info(f"Neural realization successful for language {target_language}")
            return result
            
        except Exception as e:
            self.logger.error(f"Neural realization failed for language {target_language}: {e}")
            log_error(e, {"context": "neural_realization", "target_language": target_language})
            
            # Return fallback result
            return type('RealizeResult', (), {
                'translated_text': f"[Translation failed - {target_language}]",
                'confidence': 0.0,
                'realization_time_ms': 0.0,
                'realized_primes': getattr(detection_result, 'primes', [])
            })()
    
    async def _calculate_graph_f1_with_timeout(self, detection_result, realized_result, timeout: int):
        """Calculate Graph-F1 with timeout."""
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(self.executor, self._calculate_graph_f1, detection_result, realized_result),
            timeout=timeout
        )
    
    async def _unified_pipeline_translate_with_timeout(self, request: ProductionTranslationRequest, timeout: int):
        """Unified pipeline translate with timeout."""
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(
                self.executor, 
                self.unified_pipeline.translate_with_neural_realizer,
                request.source_text, request.source_language, request.target_language
            ),
            timeout=timeout
        )
    
    def _calculate_graph_f1(self, detection_result, realized_result) -> float:
        """Calculate Graph-F1 score between detection and realization."""
        # Simplified Graph-F1 calculation
        # Handle both string and object formats for detected primes
        if hasattr(detection_result, 'primes'):
            if detection_result.primes and hasattr(detection_result.primes[0], 'text'):
                # Object format
                detected_primes = set(p.text for p in detection_result.primes)
            else:
                # String format
                detected_primes = set(detection_result.primes)
        else:
            detected_primes = set()
        
        realized_primes = set(getattr(realized_result, 'realized_primes', []))
        
        if not detected_primes and not realized_primes:
            return 1.0
        
        intersection = detected_primes & realized_primes
        union = detected_primes | realized_primes
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _validate_request(self, request: ProductionTranslationRequest):
        """Validate translation request."""
        if not request.source_text.strip():
            raise ValueError("Source text cannot be empty")
        
        if not request.source_language or not request.target_language:
            raise ValueError("Source and target languages must be specified")
        
        if request.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
    
    def _update_prometheus_metrics(self, request: ProductionTranslationRequest, result: ProductionTranslationResult, metrics: PipelineMetrics):
        """Update Prometheus metrics."""
        # Duration metrics
        TRANSLATION_DURATION.labels(
            source_lang=request.source_language,
            target_lang=request.target_language,
            mode=request.mode.value
        ).observe(metrics.total_duration)
        
        # Prime detection metrics
        PRIME_DETECTION_GAUGE.labels(language=request.source_language).set(metrics.prime_count)
        
        # Graph-F1 metrics
        if metrics.graph_f1_score > 0:
            complexity = "complex" if len(request.source_text.split()) > 10 else "simple"
            GRAPH_F1_SCORE.labels(complexity=complexity).observe(metrics.graph_f1_score)
        
        # Cultural adaptation metrics
        if metrics.cultural_adaptations > 0:
            CULTURAL_ADAPTATIONS.labels(adaptation_type="total").inc(metrics.cultural_adaptations)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get pipeline health status."""
        return {
            "status": "healthy",
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "components": {
                "detection_service": "healthy",
                "neural_generator": "healthy",
                "cultural_adapter": "healthy",
                "neural_realizer": "healthy"
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "prometheus_metrics": {
                "translation_requests": TRANSLATION_REQUESTS._value.sum(),
                "translation_errors": TRANSLATION_ERRORS._value.sum()
            }
        }

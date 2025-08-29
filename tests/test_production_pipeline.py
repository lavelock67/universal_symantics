"""
Production Pipeline Test Suite
Comprehensive testing for the production-ready universal translator pipeline.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.core.translation.production_pipeline_orchestrator import (
    ProductionPipelineOrchestrator,
    ProductionTranslationRequest,
    PipelineMode,
    QualityLevel,
    PipelineMetrics
)
from src.core.application.services import DetectionResult, NSMPrime
from cultural_adaptation_system import AdaptationResult, AdaptationChange
from neural_realizer_with_guarantees import RealizationResult

class TestProductionPipelineOrchestrator:
    """Test suite for ProductionPipelineOrchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a test orchestrator instance"""
        return ProductionPipelineOrchestrator()
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample translation request"""
        return ProductionTranslationRequest(
            source_text="The boy kicked the ball.",
            source_language="en",
            target_language="es",
            mode=PipelineMode.HYBRID,
            quality_level=QualityLevel.STANDARD,
            timeout_seconds=10
        )
    
    @pytest.fixture
    def mock_detection_result(self):
        """Create a mock detection result"""
        primes = [
            NSMPrime(text="SOMEONE", type="substantive", language="en", confidence=0.9, frequency=1),
            NSMPrime(text="DO", type="action", language="en", confidence=0.8, frequency=1),
            NSMPrime(text="TOUCH", type="action", language="en", confidence=0.85, frequency=1),
            NSMPrime(text="THING", type="substantive", language="en", confidence=0.9, frequency=1)
        ]
        return DetectionResult(primes=primes, text="The boy kicked the ball.", language="en")
    
    @pytest.fixture
    def mock_adaptation_result(self):
        """Create a mock adaptation result"""
        changes = [
            AdaptationChange(
                original="The boy kicked the ball.",
                adapted="El niño pateó la pelota.",
                change_type="translation",
                description="Translated to Spanish",
                confidence=0.9
            )
        ]
        return AdaptationResult(
            adapted_text="El niño pateó la pelota.",
            changes=changes,
            confidence=0.9
        )
    
    @pytest.fixture
    def mock_realization_result(self):
        """Create a mock realization result"""
        return RealizationResult(
            realized_text="El niño pateó la pelota.",
            realized_primes=["SOMEONE", "DO", "TOUCH", "THING"],
            preserved_terms=[],
            confidence=0.9,
            graph_f1_score=0.85
        )
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator is not None
        assert orchestrator.request_count == 0
        assert orchestrator.error_count == 0
        assert orchestrator.executor is not None
    
    @pytest.mark.asyncio
    async def test_validate_request_valid(self, orchestrator, sample_request):
        """Test request validation with valid request"""
        # Should not raise any exception
        orchestrator._validate_request(sample_request)
    
    @pytest.mark.asyncio
    async def test_validate_request_empty_text(self, orchestrator):
        """Test request validation with empty text"""
        request = ProductionTranslationRequest(
            source_text="",
            source_language="en",
            target_language="es"
        )
        with pytest.raises(ValueError, match="Source text cannot be empty"):
            orchestrator._validate_request(request)
    
    @pytest.mark.asyncio
    async def test_validate_request_missing_languages(self, orchestrator):
        """Test request validation with missing languages"""
        request = ProductionTranslationRequest(
            source_text="Hello",
            source_language="",
            target_language="es"
        )
        with pytest.raises(ValueError, match="Source and target languages must be specified"):
            orchestrator._validate_request(request)
    
    @pytest.mark.asyncio
    async def test_validate_request_invalid_timeout(self, orchestrator):
        """Test request validation with invalid timeout"""
        request = ProductionTranslationRequest(
            source_text="Hello",
            source_language="en",
            target_language="es",
            timeout_seconds=0
        )
        with pytest.raises(ValueError, match="Timeout must be positive"):
            orchestrator._validate_request(request)
    
    @pytest.mark.asyncio
    async def test_calculate_graph_f1_perfect_match(self, orchestrator, mock_detection_result):
        """Test Graph-F1 calculation with perfect match"""
        mock_realization_result = Mock()
        mock_realization_result.realized_primes = ["SOMEONE", "DO", "TOUCH", "THING"]
        
        f1_score = orchestrator._calculate_graph_f1(mock_detection_result, mock_realization_result)
        assert f1_score == 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_graph_f1_partial_match(self, orchestrator, mock_detection_result):
        """Test Graph-F1 calculation with partial match"""
        mock_realization_result = Mock()
        mock_realization_result.realized_primes = ["SOMEONE", "DO", "TOUCH"]  # Missing THING
        
        f1_score = orchestrator._calculate_graph_f1(mock_detection_result, mock_realization_result)
        assert f1_score == 0.75  # 3/4 = 0.75
    
    @pytest.mark.asyncio
    async def test_calculate_graph_f1_no_match(self, orchestrator, mock_detection_result):
        """Test Graph-F1 calculation with no match"""
        mock_realization_result = Mock()
        mock_realization_result.realized_primes = ["HELLO", "WORLD"]
        
        f1_score = orchestrator._calculate_graph_f1(mock_detection_result, mock_realization_result)
        assert f1_score == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_graph_f1_empty_results(self, orchestrator):
        """Test Graph-F1 calculation with empty results"""
        empty_detection = DetectionResult(primes=[], text="", language="en")
        mock_realization_result = Mock()
        mock_realization_result.realized_primes = []
        
        f1_score = orchestrator._calculate_graph_f1(empty_detection, mock_realization_result)
        assert f1_score == 1.0  # Perfect match when both are empty
    
    @pytest.mark.asyncio
    async def test_standard_pipeline_execution(self, orchestrator, sample_request, mock_detection_result, mock_adaptation_result):
        """Test standard pipeline execution"""
        metrics = PipelineMetrics()
        
        # Mock the component methods
        with patch.object(orchestrator, '_detect_primes_with_timeout', return_value=mock_detection_result), \
             patch.object(orchestrator, '_decompose_semantics_with_timeout', return_value={"type": "sentence"}), \
             patch.object(orchestrator, '_generate_text_with_timeout', return_value="El niño pateó la pelota."), \
             patch.object(orchestrator, '_adapt_culturally_with_timeout', return_value=mock_adaptation_result):
            
            result = await orchestrator._execute_standard_pipeline(sample_request, metrics)
            
            assert result.translated_text == "El niño pateó la pelota."
            assert result.source_text == sample_request.source_text
            assert result.source_language == sample_request.source_language
            assert result.target_language == sample_request.target_language
            assert result.mode == sample_request.mode
            assert result.quality_level == sample_request.quality_level
            assert result.success is True
            assert result.confidence_score == 0.9
            assert len(result.detected_primes) == 4
            assert len(result.cultural_adaptations) == 1
    
    @pytest.mark.asyncio
    async def test_neural_pipeline_execution(self, orchestrator, sample_request, mock_detection_result, mock_realization_result):
        """Test neural pipeline execution"""
        metrics = PipelineMetrics()
        
        # Mock the component methods
        with patch.object(orchestrator, '_detect_primes_with_timeout', return_value=mock_detection_result), \
             patch.object(orchestrator, '_neural_realize_with_timeout', return_value=mock_realization_result), \
             patch.object(orchestrator, '_calculate_graph_f1_with_timeout', return_value=0.85):
            
            result = await orchestrator._execute_neural_pipeline(sample_request, metrics)
            
            assert result.translated_text == "El niño pateó la pelota."
            assert result.success is True
            assert result.confidence_score == 0.9
            assert len(result.detected_primes) == 4
            assert len(result.glossary_preserved) == 0
            assert metrics.graph_f1_score == 0.85
    
    @pytest.mark.asyncio
    async def test_hybrid_pipeline_execution(self, orchestrator, sample_request, mock_detection_result, mock_realization_result):
        """Test hybrid pipeline execution"""
        metrics = PipelineMetrics()
        
        # Mock neural pipeline to succeed with high confidence
        mock_realization_result.confidence = 0.8
        
        with patch.object(orchestrator, '_detect_primes_with_timeout', return_value=mock_detection_result), \
             patch.object(orchestrator, '_neural_realize_with_timeout', return_value=mock_realization_result), \
             patch.object(orchestrator, '_calculate_graph_f1_with_timeout', return_value=0.85):
            
            result = await orchestrator._execute_hybrid_pipeline(sample_request, metrics)
            
            assert result.translated_text == "El niño pateó la pelota."
            assert result.success is True
            assert result.confidence_score == 0.8
    
    @pytest.mark.asyncio
    async def test_hybrid_pipeline_fallback(self, orchestrator, sample_request, mock_detection_result, mock_adaptation_result):
        """Test hybrid pipeline fallback to standard"""
        metrics = PipelineMetrics()
        
        # Mock neural pipeline to fail
        with patch.object(orchestrator, '_detect_primes_with_timeout', return_value=mock_detection_result), \
             patch.object(orchestrator, '_neural_realize_with_timeout', side_effect=Exception("Neural failed")), \
             patch.object(orchestrator, '_decompose_semantics_with_timeout', return_value={"type": "sentence"}), \
             patch.object(orchestrator, '_generate_text_with_timeout', return_value="El niño pateó la pelota."), \
             patch.object(orchestrator, '_adapt_culturally_with_timeout', return_value=mock_adaptation_result):
            
            result = await orchestrator._execute_hybrid_pipeline(sample_request, metrics)
            
            assert result.translated_text == "El niño pateó la pelota."
            assert result.success is True
    
    @pytest.mark.asyncio
    async def test_research_pipeline_execution(self, orchestrator, sample_request):
        """Test research pipeline execution"""
        metrics = PipelineMetrics()
        
        # Mock unified pipeline result
        mock_unified_result = Mock()
        mock_unified_result.translated_text = "El niño pateó la pelota."
        mock_unified_result.detected_primes = ["SOMEONE", "DO", "TOUCH", "THING"]
        mock_unified_result.confidence_score = 0.9
        mock_unified_result.graph_f1_score = 0.85
        
        with patch.object(orchestrator, '_unified_pipeline_translate_with_timeout', return_value=mock_unified_result):
            
            result = await orchestrator._execute_research_pipeline(sample_request, metrics)
            
            assert result.translated_text == "El niño pateó la pelota."
            assert result.success is True
            assert result.confidence_score == 0.9
            assert len(result.detected_primes) == 4
            assert metrics.graph_f1_score == 0.85
    
    @pytest.mark.asyncio
    async def test_translate_success(self, orchestrator, sample_request, mock_detection_result, mock_adaptation_result):
        """Test successful translation"""
        with patch.object(orchestrator, '_detect_primes_with_timeout', return_value=mock_detection_result), \
             patch.object(orchestrator, '_decompose_semantics_with_timeout', return_value={"type": "sentence"}), \
             patch.object(orchestrator, '_generate_text_with_timeout', return_value="El niño pateó la pelota."), \
             patch.object(orchestrator, '_adapt_culturally_with_timeout', return_value=mock_adaptation_result):
            
            result = await orchestrator.translate(sample_request)
            
            assert result.success is True
            assert result.translated_text == "El niño pateó la pelota."
            assert result.metrics.total_duration > 0
            assert orchestrator.request_count == 1
            assert orchestrator.error_count == 0
    
    @pytest.mark.asyncio
    async def test_translate_failure(self, orchestrator, sample_request):
        """Test translation failure handling"""
        with patch.object(orchestrator, '_detect_primes_with_timeout', side_effect=Exception("Detection failed")):
            
            result = await orchestrator.translate(sample_request)
            
            assert result.success is False
            assert result.translated_text == ""
            assert "Detection failed" in result.error_message
            assert orchestrator.request_count == 1
            assert orchestrator.error_count == 1
    
    @pytest.mark.asyncio
    async def test_translate_timeout(self, orchestrator, sample_request):
        """Test translation timeout handling"""
        async def slow_detection(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate slow operation
            return Mock()
        
        with patch.object(orchestrator, '_detect_primes_with_timeout', side_effect=slow_detection):
            
            result = await orchestrator.translate(sample_request)
            
            assert result.success is False
            assert "timeout" in result.error_message.lower() or "timeout" in str(result.error_message)
    
    @pytest.mark.asyncio
    async def test_health_status(self, orchestrator):
        """Test health status reporting"""
        # Simulate some requests
        orchestrator.request_count = 10
        orchestrator.error_count = 2
        
        health = orchestrator.get_health_status()
        
        assert health["status"] == "healthy"
        assert health["request_count"] == 10
        assert health["error_count"] == 2
        assert health["error_rate"] == 0.2
        assert "components" in health
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, orchestrator):
        """Test performance metrics reporting"""
        # Simulate some requests
        orchestrator.request_count = 15
        orchestrator.error_count = 3
        
        metrics = orchestrator.get_performance_metrics()
        
        assert metrics["total_requests"] == 15
        assert metrics["total_errors"] == 3
        assert metrics["error_rate"] == 0.2
        assert "prometheus_metrics" in metrics
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics_update(self, orchestrator, sample_request, mock_detection_result, mock_adaptation_result):
        """Test Prometheus metrics update"""
        with patch.object(orchestrator, '_detect_primes_with_timeout', return_value=mock_detection_result), \
             patch.object(orchestrator, '_decompose_semantics_with_timeout', return_value={"type": "sentence"}), \
             patch.object(orchestrator, '_generate_text_with_timeout', return_value="El niño pateó la pelota."), \
             patch.object(orchestrator, '_adapt_culturally_with_timeout', return_value=mock_adaptation_result):
            
            result = await orchestrator.translate(sample_request)
            
            # The metrics should be updated (we can't easily test the actual Prometheus values,
            # but we can ensure the method doesn't raise exceptions)
            assert result.success is True

class TestProductionAPI:
    """Test suite for Production API endpoints"""
    
    @pytest.fixture
    def api_client(self):
        """Create a test API client"""
        from fastapi.testclient import TestClient
        from api.production_api import app
        return TestClient(app)
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint"""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Universal Translator API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
    
    def test_health_endpoint(self, api_client):
        """Test health endpoint"""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "request_count" in data
        assert "error_count" in data
    
    def test_metrics_endpoint(self, api_client):
        """Test metrics endpoint"""
        response = api_client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    
    def test_languages_endpoint(self, api_client):
        """Test languages endpoint"""
        response = api_client.get("/languages")
        assert response.status_code == 200
        data = response.json()
        assert "supported_languages" in data
        assert "translation_modes" in data
        assert "quality_levels" in data
        assert len(data["supported_languages"]) > 0
    
    def test_translate_endpoint_valid_request(self, api_client):
        """Test translate endpoint with valid request"""
        request_data = {
            "source_text": "Hello world",
            "source_language": "en",
            "target_language": "es",
            "mode": "hybrid",
            "quality_level": "standard"
        }
        
        # Mock the pipeline orchestrator
        with patch('api.production_api.pipeline_orchestrator') as mock_orchestrator:
            mock_result = Mock()
            mock_result.translated_text = "Hola mundo"
            mock_result.success = True
            mock_result.confidence_score = 0.9
            mock_result.metrics = Mock()
            mock_result.metrics.total_duration = 1.0
            mock_result.detected_primes = ["HELLO", "WORLD"]
            mock_result.cultural_adaptations = []
            mock_result.glossary_preserved = []
            
            mock_orchestrator.translate.return_value = mock_result
            
            response = api_client.post("/translate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["translated_text"] == "Hola mundo"
            assert data["success"] is True
            assert data["confidence_score"] == 0.9
    
    def test_translate_endpoint_invalid_request(self, api_client):
        """Test translate endpoint with invalid request"""
        request_data = {
            "source_text": "",  # Empty text
            "source_language": "en",
            "target_language": "es"
        }
        
        response = api_client.post("/translate", json=request_data)
        assert response.status_code == 500  # Should fail validation
    
    def test_batch_translate_endpoint(self, api_client):
        """Test batch translate endpoint"""
        request_data = {
            "translations": [
                {
                    "source_text": "Hello",
                    "source_language": "en",
                    "target_language": "es"
                },
                {
                    "source_text": "Goodbye",
                    "source_language": "en",
                    "target_language": "es"
                }
            ],
            "parallel_processing": True
        }
        
        # Mock the pipeline orchestrator
        with patch('api.production_api.pipeline_orchestrator') as mock_orchestrator:
            mock_result = Mock()
            mock_result.translated_text = "Translated"
            mock_result.success = True
            mock_result.confidence_score = 0.9
            mock_result.metrics = Mock()
            mock_result.metrics.total_duration = 1.0
            mock_result.detected_primes = []
            mock_result.cultural_adaptations = []
            mock_result.glossary_preserved = []
            
            mock_orchestrator.translate.return_value = mock_result
            
            response = api_client.post("/translate/batch", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 2
            assert data["success_count"] == 2
            assert data["error_count"] == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

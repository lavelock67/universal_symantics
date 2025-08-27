#!/usr/bin/env python3
"""
Model Management System

This module provides a sophisticated model management system with caching,
lazy loading, performance monitoring, and proper resource management.
"""

import os
import time
import threading
from typing import Dict, Any, Optional, Callable
from functools import lru_cache
from pathlib import Path
import psutil
import torch
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration

from ...shared.config.settings import get_settings
from ...shared.logging.logger import get_logger, PerformanceContext
from ...shared.exceptions.exceptions import ModelLoadingError, MemoryError, create_error_context


class ModelCache:
    """Advanced LRU cache for model instances with TTL and memory management."""
    
    def __init__(self, max_size: int = 10, ttl_seconds: int = 3600):
        """Initialize model cache."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._access_times: Dict[str, float] = {}
        self._creation_times: Dict[str, float] = {}
        self._memory_usage: Dict[str, float] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get model from cache with TTL check."""
        with self._lock:
            if key in self._cache:
                # Check TTL
                if time.time() - self._creation_times[key] > self.ttl_seconds:
                    self._remove(key)
                    return None
                
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    def put(self, key: str, model: Any, memory_usage_mb: float = 0.0) -> None:
        """Put model in cache with memory tracking."""
        with self._lock:
            # Remove least recently used if cache is full
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = model
            self._access_times[key] = time.time()
            self._creation_times[key] = time.time()
            self._memory_usage[key] = memory_usage_mb
    
    def _evict_lru(self) -> None:
        """Evict least recently used model."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove(lru_key)
    
    def _remove(self, key: str) -> None:
        """Remove a specific key from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]
        if key in self._creation_times:
            del self._creation_times[key]
        if key in self._memory_usage:
            del self._memory_usage[key]
    
    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._creation_times.clear()
            self._memory_usage.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_memory_usage(self) -> float:
        """Get total memory usage of cached models in MB."""
        with self._lock:
            return sum(self._memory_usage.values())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_mb": sum(self._memory_usage.values()),
                "ttl_seconds": self.ttl_seconds,
                "keys": list(self._cache.keys())
            }


class ModelManager:
    """Advanced model manager with caching, monitoring, and performance optimization."""
    
    def __init__(self):
        """Initialize model manager."""
        self.settings = get_settings()
        self.logger = get_logger("model_manager")
        
        # Initialize cache with TTL
        self.cache = ModelCache(
            max_size=self.settings.performance.cache_max_size,
            ttl_seconds=self.settings.performance.cache_ttl_seconds
        )
        
        self._models: Dict[str, Any] = {}
        self._loading_locks: Dict[str, threading.Lock] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self._load_times: Dict[str, float] = {}
        self._memory_usage: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        
        # Resource monitoring
        self._last_memory_check = time.time()
        self._memory_check_interval = 300  # 5 minutes
        
        # Initialize cache directory
        self._init_cache_directory()
        
        # Start resource monitoring if enabled
        if self.settings.performance.enable_resource_monitoring:
            self._start_resource_monitoring()
    
    def _start_resource_monitoring(self) -> None:
        """Start background resource monitoring."""
        def monitor_resources():
            while True:
                try:
                    self._check_memory_usage()
                    time.sleep(self.settings.performance.metrics_interval_seconds)
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {str(e)}")
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def _check_memory_usage(self) -> None:
        """Check memory usage and trigger cleanup if needed."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.settings.performance.max_memory_usage_mb:
            self.logger.warning(f"Memory usage high: {memory_mb:.1f}MB, triggering cleanup")
            self._cleanup_memory()
    
    def _cleanup_memory(self) -> None:
        """Clean up memory by clearing cache and triggering garbage collection."""
        import gc
        
        # Clear cache if memory usage is high
        cache_memory = self.cache.get_memory_usage()
        if cache_memory > self.settings.performance.max_memory_usage_mb * 0.5:
            self.logger.info(f"Clearing cache to free {cache_memory:.1f}MB")
            self.cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("Memory cleanup completed")
    
    def _init_cache_directory(self) -> None:
        """Initialize model cache directory."""
        cache_dir = Path(self.settings.model.model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Model cache directory initialized: {cache_dir}")
    
    @lru_cache(maxsize=10)
    def get_spacy_model(self, language: str) -> spacy.language.Language:
        """Get SpaCy model with caching."""
        model_key = f"spacy_{language}"
        
        # Check cache first
        cached_model = self.cache.get(model_key)
        if cached_model:
            self.logger.debug(f"Using cached SpaCy model: {language}")
            return cached_model
        
        # Load model if not in cache
        return self._load_spacy_model(language, model_key)
    
    def _load_spacy_model(self, language: str, model_key: str) -> spacy.language.Language:
        """Load SpaCy model with performance monitoring."""
        with PerformanceContext(f"load_spacy_model_{language}", self.logger):
            try:
                # Get model name from settings
                model_name = getattr(self.settings.model, f"spacy_model_{language}")
                
                # Check if model exists
                if not spacy.util.is_package(model_name):
                    self.logger.warning(f"SpaCy model {model_name} not found, trying to download...")
                    try:
                        spacy.cli.download(model_name)
                    except Exception as e:
                        raise ModelLoadingError(
                            model_name=model_name,
                            error_details=f"Failed to download model: {str(e)}",
                            context=create_error_context("load_spacy_model", language=language)
                        )
                
                # Load model
                model = spacy.load(model_name)
                
                # Note: Dependency matcher is not available in standard SpaCy models
                # We'll use lexical patterns instead
                
                # Cache the model
                self.cache.put(model_key, model)
                
                # Record metadata
                self._model_metadata[model_key] = {
                    "type": "spacy",
                    "language": language,
                    "model_name": model_name,
                    "loaded_at": time.time()
                }
                
                self.logger.info(f"Model Loaded - {model_name}")
                return model
                
            except Exception as e:
                # Handle the case where model_name might not be defined
                model_name = getattr(self.settings.model, f"spacy_model_{language}", f"spacy_model_{language}")
                raise ModelLoadingError(
                    model_name=model_name,
                    error_details=str(e),
                    context=create_error_context("load_spacy_model", language=language)
                )
    
    def get_sbert_model(self) -> SentenceTransformer:
        """Get SBERT model with caching."""
        model_key = "sbert"
        
        # Check cache first
        cached_model = self.cache.get(model_key)
        if cached_model:
            self.logger.debug("Using cached SBERT model")
            return cached_model
        
        # Load model if not in cache
        return self._load_sbert_model(model_key)
    
    def _load_sbert_model(self, model_key: str) -> SentenceTransformer:
        """Load SBERT model with performance monitoring."""
        with PerformanceContext("load_sbert_model", self.logger):
            try:
                model_name = self.settings.model.sbert_model
                
                # Check memory before loading
                self._check_memory_before_loading("sbert", model_name)
                
                # Load model
                model = SentenceTransformer(model_name)
                
                # Cache the model
                self.cache.put(model_key, model)
                
                # Record metadata
                self._model_metadata[model_key] = {
                    "type": "sbert",
                    "model_name": model_name,
                    "loaded_at": time.time()
                }
                
                self.logger.model_loading(model_name, time.time())
                return model
                
            except Exception as e:
                raise ModelLoadingError(
                    model_name=model_name,
                    error_details=str(e),
                    context=create_error_context("load_sbert_model")
                )
    
    def get_generation_model(self) -> tuple[T5Tokenizer, T5ForConditionalGeneration]:
        """Get text generation model with caching."""
        model_key = "generation"
        
        # Check cache first
        cached_model = self.cache.get(model_key)
        if cached_model:
            self.logger.debug("Using cached generation model")
            return cached_model
        
        # Load model if not in cache
        return self._load_generation_model(model_key)
    
    def _load_generation_model(self, model_key: str) -> tuple[T5Tokenizer, T5ForConditionalGeneration]:
        """Load generation model with performance monitoring."""
        with PerformanceContext("load_generation_model", self.logger):
            try:
                model_name = self.settings.model.generation_model
                
                # Check memory before loading
                self._check_memory_before_loading("generation", model_name)
                
                # Load tokenizer and model
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
                
                # Move to device if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                # Cache the model
                self.cache.put(model_key, (tokenizer, model))
                
                # Record metadata
                self._model_metadata[model_key] = {
                    "type": "generation",
                    "model_name": model_name,
                    "device": str(device),
                    "loaded_at": time.time()
                }
                
                self.logger.model_loading(model_name, time.time())
                return tokenizer, model
                
            except Exception as e:
                raise ModelLoadingError(
                    model_name=model_name,
                    error_details=str(e),
                    context=create_error_context("load_generation_model")
                )
    
    def _check_memory_before_loading(self, model_type: str, model_name: str) -> None:
        """Check available memory before loading large models."""
        try:
            # Get current memory usage
            process = psutil.Process()
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Estimate model memory requirements (rough estimates)
            model_memory_estimates = {
                "sbert": 500,  # ~500MB for SBERT models
                "generation": 1000,  # ~1GB for T5 models
                "spacy": 100,  # ~100MB for SpaCy models
            }
            
            estimated_memory_mb = model_memory_estimates.get(model_type, 500)
            total_required_mb = current_memory_mb + estimated_memory_mb
            
            if total_required_mb > self.settings.performance.max_memory_usage_mb:
                raise MemoryError(
                    current_memory_mb=current_memory_mb,
                    limit_mb=self.settings.performance.max_memory_usage_mb,
                    context=create_error_context("check_memory", model_type=model_type, model_name=model_name)
                )
            
            self.logger.debug(f"Memory check passed: {current_memory_mb:.1f}MB + {estimated_memory_mb}MB = {total_required_mb:.1f}MB")
            
        except Exception as e:
            self.logger.warning(f"Memory check failed: {str(e)}")
            # Continue loading anyway, but log the warning
    
    def get_model_metadata(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model."""
        return self._model_metadata.get(model_key)
    
    def get_all_model_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all loaded models."""
        return self._model_metadata.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": self.cache.size(),
            "max_cache_size": self.cache.max_size,
            "model_metadata": self._model_metadata,
            "load_times": self._load_times.copy(),
            "memory_usage": self._memory_usage.copy()
        }
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        self.cache.clear()
        self._model_metadata.clear()
        self._load_times.clear()
        self._memory_usage.clear()
        self.logger.info("Model cache cleared")
    
    def preload_models(self, languages: list[str] = None) -> None:
        """Preload commonly used models."""
        if languages is None:
            languages = ["en", "es", "fr"]
        
        self.logger.info(f"Preloading models for languages: {languages}")
        
        # Preload SpaCy models
        for lang in languages:
            try:
                self.get_spacy_model(lang)
            except Exception as e:
                self.logger.warning(f"Failed to preload SpaCy model for {lang}: {str(e)}")
        
        # Preload SBERT model
        try:
            self.get_sbert_model()
        except Exception as e:
            self.logger.warning(f"Failed to preload SBERT model: {str(e)}")
        
        # Preload generation model
        try:
            self.get_generation_model()
        except Exception as e:
            self.logger.warning(f"Failed to preload generation model: {str(e)}")
        
        self.logger.info("Model preloading completed")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage for all models."""
        try:
            process = psutil.Process()
            total_memory_mb = process.memory_info().rss / 1024 / 1024
            
            return {
                "total_memory_mb": total_memory_mb,
                "cache_memory_mb": self.cache.get_memory_usage(),
                "cache_size": self.cache.size(),
                "available_memory_mb": self.settings.performance.max_memory_usage_mb - total_memory_mb,
                "memory_limit_mb": self.settings.performance.max_memory_usage_mb
            }
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {str(e)}")
            return {"error": str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        try:
            process = psutil.Process()
            
            return {
                "memory": self.get_memory_usage(),
                "cache": self.cache.get_cache_stats(),
                "cpu_percent": process.cpu_percent(),
                "load_times": self._load_times.copy(),
                "access_counts": self._access_counts.copy(),
                "model_metadata": self._model_metadata.copy()
            }
        except Exception as e:
            self.logger.warning(f"Failed to get performance stats: {str(e)}")
            return {"error": str(e)}
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get system resource usage."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "process_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
            }
        except Exception as e:
            self.logger.warning(f"Failed to get resource usage: {str(e)}")
            return {"error": str(e)}
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_cache()
        self.logger.info("Model manager cleanup completed")


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def initialize_model_manager() -> ModelManager:
    """Initialize the model manager."""
    global _model_manager
    if _model_manager is not None:
        _model_manager.cleanup()
    
    _model_manager = ModelManager()
    return _model_manager


def cleanup_model_manager() -> None:
    """Clean up the model manager."""
    global _model_manager
    if _model_manager is not None:
        _model_manager.cleanup()
        _model_manager = None

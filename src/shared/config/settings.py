#!/usr/bin/env python3
"""
Centralized Configuration Management

This module provides a clean, type-safe configuration system using Pydantic BaseSettings.
All configuration is centralized here and can be overridden via environment variables.
"""

import os
from typing import Dict, List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path


class ModelSettings(BaseSettings):
    """Model-related configuration settings."""
    
    # SpaCy model configurations
    spacy_model_en: str = Field(default="en_core_web_sm", description="English SpaCy model")
    spacy_model_es: str = Field(default="es_core_news_sm", description="Spanish SpaCy model")
    spacy_model_fr: str = Field(default="fr_core_news_sm", description="French SpaCy model")
    
    # Transformer model configurations
    sbert_model: str = Field(default="sentence-transformers/all-mpnet-base-v2", description="SBERT model for semantic similarity")
    generation_model: str = Field(default="t5-base", description="Text generation model")
    
    # Model loading settings
    model_cache_dir: str = Field(default="./models", description="Directory for caching models")
    lazy_load_models: bool = Field(default=True, description="Load models on demand")
    
    class Config:
        env_prefix = "MODEL_"


class PerformanceSettings(BaseSettings):
    """Performance and resource management settings."""
    
    # Corpus processing limits
    max_corpus_size: int = Field(default=1000000, description="Maximum corpus size in characters")
    max_corpus_words: int = Field(default=200000, description="Maximum corpus size in words")
    
    # Caching settings
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache entries")
    
    # Processing settings
    batch_size: int = Field(default=100, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Maximum worker threads")
    
    # Memory management
    max_memory_usage: int = Field(default=8192, description="Maximum memory usage in MB")
    
    class Config:
        env_prefix = "PERF_"


class DiscoverySettings(BaseSettings):
    """Prime discovery configuration settings."""
    
    # MDL analysis settings
    mdl_threshold: float = Field(default=0.6, description="MDL acceptance threshold")
    min_candidate_frequency: int = Field(default=5, description="Minimum candidate frequency")
    max_candidates: int = Field(default=50, description="Maximum candidates to generate")
    
    # Semantic clustering settings
    clustering_eps: float = Field(default=0.3, description="DBSCAN epsilon parameter")
    clustering_min_samples: int = Field(default=2, description="DBSCAN min_samples parameter")
    
    # Cross-lingual validation settings
    min_universality_score: float = Field(default=0.7, description="Minimum universality score")
    required_languages: List[str] = Field(default=["en", "es", "fr"], description="Required languages for validation")
    
    class Config:
        env_prefix = "DISCOVERY_"


class APISettings(BaseSettings):
    """API and web service settings."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8001, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    
    class Config:
        env_prefix = "API_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    # Log levels
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    
    # Log destinations
    log_file: Optional[str] = Field(default=None, description="Log file path")
    log_to_console: bool = Field(default=True, description="Log to console")
    
    # Structured logging
    use_json_logging: bool = Field(default=False, description="Use JSON logging format")
    
    class Config:
        env_prefix = "LOG_"


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # Database connection
    database_url: str = Field(default="sqlite:///./nsm_research.db", description="Database connection URL")
    database_pool_size: int = Field(default=10, description="Database connection pool size")
    
    # Migration settings
    auto_migrate: bool = Field(default=True, description="Auto-run database migrations")
    
    class Config:
        env_prefix = "DB_"


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections."""
    
    # Environment
    environment: str = Field(default="development", description="Environment (development/production)")
    
    # Sub-settings
    model: ModelSettings = ModelSettings()
    performance: PerformanceSettings = PerformanceSettings()
    discovery: DiscoverySettings = DiscoverySettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    database: DatabaseSettings = DatabaseSettings()
    
    # Validation
    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['development', 'production', 'testing']:
            raise ValueError('Environment must be development, production, or testing')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global settings
    settings = Settings()
    return settings


# Environment-specific settings
def is_development() -> bool:
    """Check if running in development mode."""
    return settings.environment == "development"


def is_production() -> bool:
    """Check if running in production mode."""
    return settings.environment == "production"


def is_testing() -> bool:
    """Check if running in testing mode."""
    return settings.environment == "testing"

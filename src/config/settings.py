#!/usr/bin/env python3
"""Configuration management using Pydantic BaseSettings."""

import os
from typing import Dict, Any
from pydantic import BaseSettings, Field
from functools import lru_cache


class APISettings(BaseSettings):
    """API configuration settings."""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8001, env="API_PORT")
    workers: int = Field(default=4, env="API_WORKERS")
    reload: bool = Field(default=True, env="API_RELOAD")


class ModelSettings(BaseSettings):
    """Model path settings."""
    en_core_web_sm: str = Field(default="en_core_web_sm", env="EN_CORE_WEB_SM_PATH")
    es_core_news_sm: str = Field(default="es_core_news_sm", env="ES_CORE_NEWS_SM_PATH")
    fr_core_news_sm: str = Field(default="fr_core_news_sm", env="FR_CORE_NEWS_SM_PATH")


class RouterThresholds(BaseSettings):
    """Router thresholds per language."""
    
    # English thresholds
    en_legality: float = Field(default=0.9, env="EN_LEGALITY_THRESHOLD")
    en_drift: float = Field(default=0.15, env="EN_DRIFT_THRESHOLD")
    en_confidence: float = Field(default=0.7, env="EN_CONFIDENCE_THRESHOLD")
    
    # Spanish thresholds
    es_legality: float = Field(default=0.85, env="ES_LEGALITY_THRESHOLD")
    es_drift: float = Field(default=0.2, env="ES_DRIFT_THRESHOLD")
    es_confidence: float = Field(default=0.65, env="ES_CONFIDENCE_THRESHOLD")
    
    # French thresholds
    fr_legality: float = Field(default=0.85, env="FR_LEGALITY_THRESHOLD")
    fr_drift: float = Field(default=0.2, env="FR_DRIFT_THRESHOLD")
    fr_confidence: float = Field(default=0.65, env="FR_CONFIDENCE_THRESHOLD")
    
    def get_thresholds(self, lang: str) -> Dict[str, float]:
        """Get thresholds for a specific language."""
        prefix = lang.lower()
        return {
            "legality": getattr(self, f"{prefix}_legality"),
            "drift": getattr(self, f"{prefix}_drift"),
            "confidence": getattr(self, f"{prefix}_confidence")
        }


class SafetyWeights(BaseSettings):
    """Safety-critical feature weights."""
    negation_scope: float = Field(default=0.4, env="NEGATION_SCOPE_WEIGHT")
    quantifier_scope: float = Field(default=0.3, env="QUANTIFIER_SCOPE_WEIGHT")
    sense_confidence: float = Field(default=0.2, env="SENSE_CONFIDENCE_WEIGHT")
    mwe_coverage: float = Field(default=0.1, env="MWE_COVERAGE_WEIGHT")


class DiscoverySettings(BaseSettings):
    """Prime discovery settings."""
    min_frequency: int = Field(default=10, env="MIN_FREQUENCY")
    min_compression_gain: float = Field(default=0.05, env="MIN_COMPRESSION_GAIN")
    max_drift_increase: float = Field(default=0.1, env="MAX_DRIFT_INCREASE")
    min_symmetry_score: float = Field(default=0.7, env="MIN_SYMMETRY_SCORE")
    min_confidence: float = Field(default=0.8, env="MIN_CONFIDENCE")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")


class DevelopmentSettings(BaseSettings):
    """Development settings."""
    debug: bool = Field(default=False, env="DEBUG")
    test_mode: bool = Field(default=False, env="TEST_MODE")


class Settings(BaseSettings):
    """Main settings class combining all configuration."""
    api: APISettings = APISettings()
    models: ModelSettings = ModelSettings()
    router: RouterThresholds = RouterThresholds()
    safety: SafetyWeights = SafetyWeights()
    discovery: DiscoverySettings = DiscoverySettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    dev: DevelopmentSettings = DevelopmentSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def load_dotenv_simple(filename: str = ".env") -> None:
    """Simple .env loader without external dependencies."""
    if not os.path.exists(filename):
        return
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ.setdefault(key, value)

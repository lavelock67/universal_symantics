#!/usr/bin/env python3
"""
Exception handling for the NSM system.
"""

from .exceptions import (
    NSMBaseException, ConfigurationError, ModelLoadingError, CorpusProcessingError,
    DiscoveryError, ValidationError, PerformanceError, MemoryError, APIError,
    DatabaseError, CacheError, RateLimitError, AuthenticationError, ResourceNotFoundError,
    TimeoutError, handle_exception, create_error_context, format_error_response
)

__all__ = [
    "NSMBaseException", "ConfigurationError", "ModelLoadingError", "CorpusProcessingError",
    "DiscoveryError", "ValidationError", "PerformanceError", "MemoryError", "APIError",
    "DatabaseError", "CacheError", "RateLimitError", "AuthenticationError", "ResourceNotFoundError",
    "TimeoutError", "handle_exception", "create_error_context", "format_error_response"
]

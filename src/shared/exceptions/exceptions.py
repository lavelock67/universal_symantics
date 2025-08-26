#!/usr/bin/env python3
"""
Custom Exception Classes

This module defines custom exceptions for the NSM system with proper error handling,
context information, and recovery suggestions.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    parameters: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


class NSMBaseException(Exception):
    """Base exception class for all NSM system exceptions."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, 
                 recovery_suggestion: Optional[str] = None):
        """Initialize the exception."""
        super().__init__(message)
        self.message = message
        self.context = context
        self.recovery_suggestion = recovery_suggestion
    
    def __str__(self) -> str:
        """String representation of the exception."""
        result = f"{self.__class__.__name__}: {self.message}"
        if self.recovery_suggestion:
            result += f"\nRecovery suggestion: {self.recovery_suggestion}"
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context.__dict__ if self.context else None,
            "recovery_suggestion": self.recovery_suggestion
        }


class ConfigurationError(NSMBaseException):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 context: Optional[ErrorContext] = None):
        """Initialize configuration error."""
        recovery = f"Check configuration for key: {config_key}" if config_key else "Review configuration settings"
        super().__init__(message, context, recovery)
        self.config_key = config_key


class ModelLoadingError(NSMBaseException):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, error_details: str, 
                 context: Optional[ErrorContext] = None):
        """Initialize model loading error."""
        message = f"Failed to load model '{model_name}': {error_details}"
        recovery = f"Ensure model '{model_name}' is properly installed and accessible"
        super().__init__(message, context, recovery)
        self.model_name = model_name
        self.error_details = error_details


class CorpusProcessingError(NSMBaseException):
    """Raised when corpus processing fails."""
    
    def __init__(self, operation: str, corpus_size: int, error_details: str,
                 context: Optional[ErrorContext] = None):
        """Initialize corpus processing error."""
        message = f"Corpus processing failed during '{operation}': {error_details}"
        recovery = f"Check corpus format and size (current: {corpus_size} characters)"
        super().__init__(message, context, recovery)
        self.operation = operation
        self.corpus_size = corpus_size
        self.error_details = error_details


class DiscoveryError(NSMBaseException):
    """Raised when prime discovery fails."""
    
    def __init__(self, stage: str, error_details: str, candidates_found: int = 0,
                 context: Optional[ErrorContext] = None):
        """Initialize discovery error."""
        message = f"Prime discovery failed at stage '{stage}': {error_details}"
        recovery = f"Review discovery parameters and corpus quality (candidates found: {candidates_found})"
        super().__init__(message, context, recovery)
        self.stage = stage
        self.candidates_found = candidates_found
        self.error_details = error_details


class ValidationError(NSMBaseException):
    """Raised when validation fails."""
    
    def __init__(self, field: str, value: Any, expected_type: str,
                 context: Optional[ErrorContext] = None):
        """Initialize validation error."""
        message = f"Validation failed for field '{field}': got {type(value).__name__}, expected {expected_type}"
        recovery = f"Ensure field '{field}' has the correct type and format"
        super().__init__(message, context, recovery)
        self.field = field
        self.value = value
        self.expected_type = expected_type


class PerformanceError(NSMBaseException):
    """Raised when performance limits are exceeded."""
    
    def __init__(self, metric: str, current_value: float, limit: float,
                 context: Optional[ErrorContext] = None):
        """Initialize performance error."""
        message = f"Performance limit exceeded: {metric} = {current_value} (limit: {limit})"
        recovery = f"Consider reducing corpus size or optimizing processing parameters"
        super().__init__(message, context, recovery)
        self.metric = metric
        self.current_value = current_value
        self.limit = limit


class MemoryError(NSMBaseException):
    """Raised when memory usage is too high."""
    
    def __init__(self, current_memory_mb: float, limit_mb: float,
                 context: Optional[ErrorContext] = None):
        """Initialize memory error."""
        message = f"Memory usage exceeded: {current_memory_mb:.1f}MB (limit: {limit_mb}MB)"
        recovery = "Consider processing smaller batches or upgrading system memory"
        super().__init__(message, context, recovery)
        self.current_memory_mb = current_memory_mb
        self.limit_mb = limit_mb


class APIError(NSMBaseException):
    """Raised when API operations fail."""
    
    def __init__(self, endpoint: str, status_code: int, response_text: str,
                 context: Optional[ErrorContext] = None):
        """Initialize API error."""
        message = f"API call to '{endpoint}' failed with status {status_code}: {response_text}"
        recovery = "Check API endpoint availability and request format"
        super().__init__(message, context, recovery)
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_text = response_text


class DatabaseError(NSMBaseException):
    """Raised when database operations fail."""
    
    def __init__(self, operation: str, table: str, error_details: str,
                 context: Optional[ErrorContext] = None):
        """Initialize database error."""
        message = f"Database operation '{operation}' on table '{table}' failed: {error_details}"
        recovery = "Check database connection and table schema"
        super().__init__(message, context, recovery)
        self.operation = operation
        self.table = table
        self.error_details = error_details


class CacheError(NSMBaseException):
    """Raised when caching operations fail."""
    
    def __init__(self, operation: str, key: str, error_details: str,
                 context: Optional[ErrorContext] = None):
        """Initialize cache error."""
        message = f"Cache operation '{operation}' for key '{key}' failed: {error_details}"
        recovery = "Check cache configuration and storage availability"
        super().__init__(message, context, recovery)
        self.operation = operation
        self.key = key
        self.error_details = error_details


class RateLimitError(NSMBaseException):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, limit: int, window_seconds: int, retry_after: int,
                 context: Optional[ErrorContext] = None):
        """Initialize rate limit error."""
        message = f"Rate limit exceeded: {limit} requests per {window_seconds} seconds"
        recovery = f"Wait {retry_after} seconds before retrying"
        super().__init__(message, context, recovery)
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after


class AuthenticationError(NSMBaseException):
    """Raised when authentication fails."""
    
    def __init__(self, auth_type: str, error_details: str,
                 context: Optional[ErrorContext] = None):
        """Initialize authentication error."""
        message = f"Authentication failed ({auth_type}): {error_details}"
        recovery = "Check credentials and authentication configuration"
        super().__init__(message, context, recovery)
        self.auth_type = auth_type
        self.error_details = error_details


class ResourceNotFoundError(NSMBaseException):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: str,
                 context: Optional[ErrorContext] = None):
        """Initialize resource not found error."""
        message = f"{resource_type} with ID '{resource_id}' not found"
        recovery = f"Verify the {resource_type} ID exists and is accessible"
        super().__init__(message, context, recovery)
        self.resource_type = resource_type
        self.resource_id = resource_id


class TimeoutError(NSMBaseException):
    """Raised when operations timeout."""
    
    def __init__(self, operation: str, timeout_seconds: float,
                 context: Optional[ErrorContext] = None):
        """Initialize timeout error."""
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        recovery = "Consider reducing corpus size or increasing timeout limits"
        super().__init__(message, context, recovery)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


# Error handling utilities
def handle_exception(func):
    """Decorator for automatic exception handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NSMBaseException:
            # Re-raise NSM exceptions as-is
            raise
        except Exception as e:
            # Convert other exceptions to NSM exceptions
            context = ErrorContext(
                operation=func.__name__,
                parameters={"args": str(args), "kwargs": str(kwargs)}
            )
            raise NSMBaseException(
                f"Unexpected error in {func.__name__}: {str(e)}",
                context=context,
                recovery_suggestion="Check system logs for detailed error information"
            ) from e
    return wrapper


def create_error_context(operation: str, **kwargs) -> ErrorContext:
    """Create error context from operation and parameters."""
    return ErrorContext(
        operation=operation,
        parameters=kwargs
    )


def format_error_response(error: NSMBaseException) -> Dict[str, Any]:
    """Format error for API response."""
    return {
        "error": error.to_dict(),
        "success": False,
        "timestamp": error.context.timestamp if error.context else None
    }

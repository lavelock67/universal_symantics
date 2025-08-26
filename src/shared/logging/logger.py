#!/usr/bin/env python3
"""
Comprehensive Logging System

This module provides a clean, structured logging system with proper error handling,
performance monitoring, and configurable output formats.
"""

import logging
import logging.handlers
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from ..config.settings import get_settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", 
                          "msecs", "relativeCreated", "thread", "threadName", 
                          "processName", "process", "getMessage", "exc_info", 
                          "exc_text", "stack_info"]:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class PerformanceFormatter(logging.Formatter):
    """Custom formatter for performance logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format performance log record."""
        if hasattr(record, 'duration'):
            return f"{record.levelname} - {record.name} - {record.getMessage()} - Duration: {record.duration:.3f}s"
        return super().format(record)


class NSMLogger:
    """Main logger class for the NSM system."""
    
    def __init__(self, name: str = "nsm_system"):
        """Initialize the logger."""
        self.name = name
        self.settings = get_settings()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with proper configuration."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.settings.logging.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.settings.logging.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if self.settings.logging.use_json_logging:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(logging.Formatter(self.settings.logging.log_format))
            logger.addHandler(console_handler)
        
        # File handler
        if self.settings.logging.log_file:
            log_path = Path(self.settings.logging.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            
            if self.settings.logging.use_json_logging:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(self.settings.logging.log_format))
            
            logger.addHandler(file_handler)
        
        return logger
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)
    
    def performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.logger.info(f"Performance - {operation}", 
                        extra={"duration": duration, "operation": operation, **kwargs})
    
    def discovery_event(self, event_type: str, corpus_size: int, candidates_found: int, **kwargs):
        """Log discovery-specific events."""
        self.logger.info(f"Discovery Event - {event_type}", 
                        extra={
                            "event_type": event_type,
                            "corpus_size": corpus_size,
                            "candidates_found": candidates_found,
                            **kwargs
                        })
    
    def model_loading(self, model_name: str, duration: float, **kwargs):
        """Log model loading events."""
        self.logger.info(f"Model Loaded - {model_name}", 
                        extra={"model_name": model_name, "duration": duration, **kwargs})


class PerformanceLogger:
    """Specialized logger for performance monitoring."""
    
    def __init__(self, name: str = "performance"):
        """Initialize performance logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Use performance formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(PerformanceFormatter())
        self.logger.addHandler(handler)
    
    def log_operation(self, operation: str, duration: float, **kwargs):
        """Log operation performance."""
        self.logger.info(f"Operation: {operation}", 
                        extra={"duration": duration, "operation": operation, **kwargs})
    
    def log_memory_usage(self, memory_mb: float, **kwargs):
        """Log memory usage."""
        self.logger.info(f"Memory Usage: {memory_mb:.2f}MB", 
                        extra={"memory_mb": memory_mb, **kwargs})


class ErrorLogger:
    """Specialized logger for error tracking."""
    
    def __init__(self, name: str = "errors"):
        """Initialize error logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.ERROR)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            "logs/errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context."""
        self.logger.error(
            f"Error: {str(error)}",
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
                "traceback": traceback.format_exc()
            },
            exc_info=True
        )


# Global logger instances
main_logger = NSMLogger("nsm_system")
performance_logger = PerformanceLogger()
error_logger = ErrorLogger()


def get_logger(name: str = None) -> NSMLogger:
    """Get a logger instance."""
    if name:
        return NSMLogger(name)
    return main_logger


def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    performance_logger.log_operation(operation, duration, **kwargs)


def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log error with context."""
    error_logger.log_error(error, context)


# Context manager for performance logging
class PerformanceContext:
    """Context manager for automatic performance logging."""
    
    def __init__(self, operation: str, logger: NSMLogger = None):
        """Initialize performance context."""
        self.operation = operation
        self.logger = logger or main_logger
        self.start_time = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = datetime.now()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.performance(self.operation, duration)
            
            if exc_type:
                self.logger.exception(f"Operation failed: {self.operation}")
                return False
            else:
                self.logger.debug(f"Completed operation: {self.operation}")
        
        return True

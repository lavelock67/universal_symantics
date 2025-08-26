#!/usr/bin/env python3
"""
Infrastructure components for the NSM system.
"""

from .model_manager import get_model_manager, initialize_model_manager, cleanup_model_manager

__all__ = ["get_model_manager", "initialize_model_manager", "cleanup_model_manager"]

#!/usr/bin/env python3
"""
Debug Detection Error

This script tests the detection service directly to find the error.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from core.application.services import NSMDetectionService
    from core.domain.models import Language
    
    print("✅ Imports successful")
    
    # Create detection service
    detection_service = NSMDetectionService()
    print("✅ Detection service created")
    
    # Test simple detection
    result = detection_service.detect_primes("I think this is good", Language.ENGLISH)
    print("✅ Detection successful")
    print(f"Result: {result}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()


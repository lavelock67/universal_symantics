#!/usr/bin/env python3
"""
Debug the API startup process
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_startup():
    """Test the API startup process"""
    try:
        print("🔍 Testing API startup process...")
        
        from api.clean_nsm_api import startup_event
        
        print("📝 Testing startup event...")
        import asyncio
        
        # Run the startup event
        asyncio.run(startup_event())
        print("✅ Startup event completed successfully")
        
        # Check if the global variables are set
        from api.clean_nsm_api import detection_service, model_manager, eil_service
        
        print(f"📝 Global variables after startup:")
        print(f"  - detection_service: {detection_service is not None}")
        print(f"  - model_manager: {model_manager is not None}")
        print(f"  - eil_service: {eil_service is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_startup()
    if success:
        print("\n🎉 Startup test completed!")
    else:
        print("\n💥 Startup test failed!")

#!/usr/bin/env python3
"""
Test EIL Imports

Simple test to check if EIL module imports are working.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test EIL module imports."""
    print("Testing EIL imports...")
    
    try:
        from src.core.eil.graph import EILGraph, EILNode, EILRelation
        print("  ✅ EIL graph imports successful")
    except Exception as e:
        print(f"  ❌ EIL graph import failed: {e}")
        return False
    
    try:
        from src.core.eil.validator import EILValidator, ValidationResult
        print("  ✅ EIL validator imports successful")
    except Exception as e:
        print(f"  ❌ EIL validator import failed: {e}")
        return False
    
    try:
        from src.core.eil.router import EILRouter, RouterResult, RouterDecision
        print("  ✅ EIL router imports successful")
    except Exception as e:
        print(f"  ❌ EIL router import failed: {e}")
        return False
    
    try:
        from src.core.eil.extractor import EILExtractor, ExtractionResult
        print("  ✅ EIL extractor imports successful")
    except Exception as e:
        print(f"  ❌ EIL extractor import failed: {e}")
        return False
    
    try:
        from src.core.eil.realizer import EILRealizer, RealizationResult
        print("  ✅ EIL realizer imports successful")
    except Exception as e:
        print(f"  ❌ EIL realizer import failed: {e}")
        return False
    
    try:
        from src.core.eil.service import EILService, EILProcessingResult
        print("  ✅ EIL service imports successful")
    except Exception as e:
        print(f"  ❌ EIL service import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic EIL functionality."""
    print("\nTesting basic EIL functionality...")
    
    try:
        from src.core.eil.graph import EILGraph, EILNode, EILNodeType
        from src.core.eil.validator import EILValidator
        from src.core.eil.router import EILRouter
        from src.core.eil.extractor import EILExtractor
        from src.core.eil.realizer import EILRealizer
        from src.core.eil.service import EILService
        
        # Create a simple graph
        graph = EILGraph()
        node = EILNode(label="TEST", node_type=EILNodeType.ENTITY)
        graph.add_node(node)
        
        print(f"  ✅ Created graph with {len(graph.nodes)} nodes")
        
        # Test validator
        validator = EILValidator()
        validation_result = validator.validate_graph(graph)
        print(f"  ✅ Validation successful: {validation_result.is_valid}")
        
        # Test router
        router = EILRouter()
        router_result = router.route(graph, validation_result)
        print(f"  ✅ Routing successful: {router_result.decision.value}")
        
        # Test extractor
        extractor = EILExtractor()
        print(f"  ✅ Extractor initialized")
        
        # Test realizer
        realizer = EILRealizer()
        print(f"  ✅ Realizer initialized")
        
        # Test service
        service = EILService()
        print(f"  ✅ Service initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 50)
    print("EIL IMPORT TEST")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed")
        return
    
    print("\n" + "=" * 50)
    print("✅ ALL EIL TESTS PASSED")
    print("=" * 50)

if __name__ == "__main__":
    main()


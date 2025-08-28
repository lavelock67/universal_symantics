#!/usr/bin/env python3
"""
Debug EIL Extraction

Test EIL extraction directly to identify issues.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_eil_extraction():
    """Test EIL extraction directly."""
    print("Testing EIL extraction directly...")
    
    try:
        from src.core.application.services import create_detection_service
        from src.core.domain.models import Language
        from src.core.eil.extractor import EILExtractor
        
        # Create detection service
        detection_service = create_detection_service()
        print("  ✅ Detection service created")
        
        # Create EIL extractor
        extractor = EILExtractor()
        print("  ✅ EIL extractor created")
        
        # Test detection
        text = "I think this is good"
        language = Language.ENGLISH
        
        print(f"  Testing detection for: '{text}'")
        detection_result = detection_service.detect_primes(text, language)
        
        print(f"  ✅ Detection successful")
        print(f"  Primes found: {len(detection_result.primes)}")
        for prime in detection_result.primes:
            print(f"    - {prime.text} ({prime.type.value}) confidence: {prime.confidence}")
        
        print(f"  MWEs found: {len(detection_result.mwes)}")
        for mwe in detection_result.mwes:
            print(f"    - {mwe.text} ({mwe.type.value}) confidence: {mwe.confidence}")
        
        # Test EIL extraction
        print(f"\n  Testing EIL extraction...")
        extraction_result = extractor.extract_from_detection(detection_result)
        
        print(f"  ✅ EIL extraction successful")
        print(f"  Extraction confidence: {extraction_result.confidence}")
        print(f"  Graph nodes: {len(extraction_result.graph.nodes)}")
        print(f"  Graph relations: {len(extraction_result.graph.relations)}")
        print(f"  Metadata: {extraction_result.metadata}")
        
        # Show graph details
        if extraction_result.graph.nodes:
            print(f"  Graph nodes:")
            for node_id, node in extraction_result.graph.nodes.items():
                print(f"    - {node_id}: {node.label} ({node.node_type.value})")
        
        if extraction_result.graph.relations:
            print(f"  Graph relations:")
            for rel_id, relation in extraction_result.graph.relations.items():
                print(f"    - {rel_id}: {relation.source_id} -> {relation.target_id} ({relation.relation_type.value})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ EIL extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_eil_service():
    """Test EIL service directly."""
    print("\nTesting EIL service directly...")
    
    try:
        from src.core.application.services import create_detection_service
        from src.core.domain.models import Language
        from src.core.eil.service import EILService
        
        # Create services
        detection_service = create_detection_service()
        eil_service = EILService()
        print("  ✅ Services created")
        
        # Test processing
        text = "I think this is good"
        language = Language.ENGLISH
        
        print(f"  Testing EIL processing for: '{text}'")
        
        # Get detection result
        detection_result = detection_service.detect_primes(text, language)
        print(f"  Detection result: {len(detection_result.primes)} primes")
        
        # Process through EIL
        eil_result = eil_service.process_text(
            text=text,
            source_language="en",
            target_language="en",
            detection_result=detection_result
        )
        
        print(f"  ✅ EIL processing successful")
        print(f"  Router decision: {eil_result.router_result.decision.value}")
        print(f"  Router confidence: {eil_result.router_result.confidence}")
        print(f"  Router reasoning: {eil_result.router_result.reasoning}")
        print(f"  Source graph nodes: {len(eil_result.source_graph.nodes)}")
        print(f"  Source graph relations: {len(eil_result.source_graph.relations)}")
        print(f"  Validation valid: {eil_result.validation_result.is_valid}")
        print(f"  Processing times: {eil_result.processing_times}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ EIL service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("=" * 60)
    print("EIL EXTRACTION DEBUG")
    print("=" * 60)
    
    # Test EIL extraction
    if not test_eil_extraction():
        print("\n❌ EIL extraction test failed")
        return
    
    # Test EIL service
    if not test_eil_service():
        print("\n❌ EIL service test failed")
        return
    
    print("\n" + "=" * 60)
    print("✅ ALL EIL DEBUG TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()


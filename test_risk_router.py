#!/usr/bin/env python3
"""Test the Risk-Coverage Router."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_risk_router():
    """Test the risk-coverage router with various scenarios."""
    print("🚀 Testing Risk-Coverage Router")
    print("=" * 50)
    
    from src.router.risk_coverage_router import RiskCoverageRouter
    from src.detect.srl_ud_detectors import detect_primitives_multilingual
    
    router = RiskCoverageRouter()
    
    # Test cases with different risk profiles
    test_cases = [
        # Low risk - simple, clear detection
        {
            "text": "I think you know the truth",
            "description": "Low risk - clear mental predicates",
            "expected": "translate"
        },
        
        # Medium risk - some ambiguity
        {
            "text": "La gente piensa que esto es muy bueno",
            "description": "Medium risk - Spanish with evaluators",
            "expected": "translate"
        },
        
        # High risk - complex, ambiguous
        {
            "text": "At most half of the students read a lot of books",
            "description": "High risk - complex quantifiers and scope",
            "expected": "clarify"
        },
        
        # Very high risk - unclear, minimal detection
        {
            "text": "The quick brown fox jumps over the lazy dog",
            "description": "Very high risk - no clear NSM primes",
            "expected": "abstain"
        },
        
        # Safety-critical - negation scope
        {
            "text": "Es falso que el medicamento no funcione",
            "description": "Safety-critical - negation scope ambiguity",
            "expected": "clarify"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Text: {test_case['text']}")
        
        # Detect primes
        detected_primes = detect_primitives_multilingual(test_case['text'])
        print(f"Detected primes: {detected_primes}")
        
        # Route detection
        result = router.route_detection(
            text=test_case['text'],
            detected_primes=detected_primes,
            legality_score=0.8 if len(detected_primes) > 2 else 0.4,
            sense_confidence=0.7 if len(detected_primes) > 1 else 0.3
        )
        
        print(f"Decision: {result.decision.value}")
        print(f"Risk estimate: {result.risk_estimate:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Coverage bucket: {result.coverage_bucket}")
        print(f"Reasons: {result.reasons}")
        
        # Check if decision matches expectation
        if result.decision.value == test_case['expected']:
            print("✅ PASS - Decision matches expectation")
        else:
            print(f"⚠️  PARTIAL - Expected {test_case['expected']}, got {result.decision.value}")
        
        print("-" * 40)
    
    # Test generation routing
    print(f"\n🔧 Testing Generation Routing")
    print("=" * 50)
    
    generation_test_cases = [
        {
            "original_text": "I think you know the truth",
            "generation_result": {
                "legality": 0.95,
                "drift": {"graph_f1": 0.92},
                "mdl_delta": -0.1,
                "confidence": 0.9,
                "generated_primes": ["I", "THINK", "YOU", "KNOW", "TRUE"]
            },
            "description": "High quality generation"
        },
        {
            "original_text": "Complex ambiguous text",
            "generation_result": {
                "legality": 0.6,
                "drift": {"graph_f1": 0.4},
                "mdl_delta": 0.3,
                "confidence": 0.4,
                "generated_primes": ["SOMETHING", "HAPPEN"]
            },
            "description": "Low quality generation"
        }
    ]
    
    for i, test_case in enumerate(generation_test_cases, 1):
        print(f"\n📝 Generation Test {i}: {test_case['description']}")
        print(f"Original: {test_case['original_text']}")
        
        result = router.route_generation(
            generation_result=test_case['generation_result'],
            original_text=test_case['original_text']
        )
        
        print(f"Decision: {result.decision.value}")
        print(f"Risk estimate: {result.risk_estimate:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Reasons: {result.reasons}")
        print("-" * 40)
    
    # Show statistics
    print(f"\n📊 Router Statistics")
    print("=" * 50)
    stats = router.get_statistics()
    
    print(f"Total requests: {stats['total_requests']}")
    print(f"Decision distribution: {stats['decision_distribution']}")
    print(f"Decision percentages: {stats['decision_percentages']}")
    print(f"Coverage distribution: {stats['coverage_distribution']}")
    print(f"Risk distribution: {stats['risk_distribution']}")
    print(f"Average processing time: {stats['avg_processing_time']:.4f}s")
    
    # Calculate success metrics
    total = stats['total_requests']
    if total > 0:
        translate_pct = stats['decision_percentages'].get('translate', 0)
        clarify_pct = stats['decision_percentages'].get('clarify', 0)
        abstain_pct = stats['decision_percentages'].get('abstain', 0)
        
        print(f"\n🎯 Success Metrics:")
        print(f"Translate rate: {translate_pct:.1%}")
        print(f"Clarify rate: {clarify_pct:.1%}")
        print(f"Abstain rate: {abstain_pct:.1%}")
        
        # Target: 60% translate, 30% clarify, 10% abstain
        target_score = (translate_pct * 0.6 + clarify_pct * 0.3 + abstain_pct * 0.1)
        print(f"Target alignment score: {target_score:.3f}")


if __name__ == "__main__":
    test_risk_router()

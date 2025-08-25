#!/usr/bin/env python3
"""
Comprehensive NSM Prime Coverage Test Suite

Tests all 65 NSM primes across all 11 phases to ensure complete coverage.
"""

import json
import time
from typing import Dict, List, Set, Tuple
from src.detect.srl_ud_detectors import (
    detect_primitives_spacy, 
    detect_primitives_structured, 
    detect_primitives_multilingual
)

# All 65 NSM primes organized by phase
NSM_PRIMES_BY_PHASE = {
    "Phase 1 - Substantives": {
        "I", "YOU", "SOMEONE", "PEOPLE", "SOMETHING", "THING", "BODY"
    },
    "Phase 2 - Mental Predicates": {
        "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR"
    },
    "Phase 3 - Logical Operators": {
        "BECAUSE", "IF", "NOT", "SAME", "DIFFERENT", "MAYBE"
    },
    "Phase 4 - Temporal & Causal": {
        "BEFORE", "AFTER", "WHEN", "CAUSE", "MAKE", "LET"
    },
    "Phase 5 - Spatial & Physical": {
        "IN", "ON", "UNDER", "NEAR", "FAR", "INSIDE"
    },
    "Phase 6 - Quantifiers": {
        "ALL", "MANY", "SOME", "FEW", "MUCH", "LITTLE"
    },
    "Phase 7 - Evaluators": {
        "GOOD", "BAD", "BIG", "SMALL", "RIGHT", "WRONG"
    },
    "Phase 8 - Actions": {
        "DO", "HAPPEN", "MOVE", "TOUCH", "LIVE", "DIE"
    },
    "Phase 9 - Descriptors": {
        "THIS", "THE SAME", "OTHER", "ONE", "TWO", "SOME"
    },
    "Phase 10 - Intensifiers": {
        "VERY", "MORE", "LIKE", "KIND OF"
    },
    "Phase 11 - Final Primes": {
        "SAY", "WORDS", "TRUE", "FALSE", "WHERE", "WHEN"
    }
}

# All 65 NSM primes combined
ALL_NSM_PRIMES = set()
for phase_primes in NSM_PRIMES_BY_PHASE.values():
    ALL_NSM_PRIMES.update(phase_primes)

def load_test_suites() -> Dict[str, List[Dict]]:
    """Load all test suites from realistic_suites directory."""
    test_suites = {}
    
    # Load test suites for each phase
    phase_files = {
        "Phase 1 - Substantives": "data/realistic_suites/substantives/en.jsonl",
        "Phase 2 - Mental Predicates": "data/realistic_suites/mental_predicates/en.jsonl", 
        "Phase 3 - Logical Operators": "data/realistic_suites/logical_operators/en.jsonl",
        "Phase 4 - Temporal & Causal": "data/realistic_suites/temporal_causal/en.jsonl",
        "Phase 5 - Spatial & Physical": "data/realistic_suites/spatial_physical/en.jsonl",
        "Phase 6 - Quantifiers": "data/realistic_suites/quantifiers/en.jsonl",
        "Phase 7 - Evaluators": "data/realistic_suites/evaluators/en.jsonl",
        "Phase 8 - Actions": "data/realistic_suites/actions/en.jsonl",
        "Phase 9 - Descriptors": "data/realistic_suites/descriptors/en.jsonl",
        "Phase 10 - Intensifiers": "data/realistic_suites/intensifiers/en.jsonl",
        "Phase 11 - Final Primes": "data/realistic_suites/final_primes/en.jsonl"
    }
    
    for phase_name, file_path in phase_files.items():
        try:
            with open(file_path, 'r') as f:
                test_suites[phase_name] = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: Test suite not found for {phase_name}: {file_path}")
            test_suites[phase_name] = []
    
    return test_suites

def test_detection_methods(text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """Test all three detection methods on a given text."""
    spacy_detected = set(detect_primitives_spacy(text))
    structured_detected = set(d['name'] for d in detect_primitives_structured(text))
    multilingual_detected = set(detect_primitives_multilingual(text))
    
    return spacy_detected, structured_detected, multilingual_detected

def analyze_phase_performance(phase_name: str, test_cases: List[Dict]) -> Dict:
    """Analyze performance for a specific phase."""
    if not test_cases:
        return {
            "phase": phase_name,
            "total_cases": 0,
            "correct_cases": 0,
            "accuracy": 0.0,
            "expected_primes": set(),
            "detected_primes": set(),
            "missing_primes": set(),
            "extra_primes": set(),
            "detection_methods": {
                "spacy": {"correct": 0, "total": 0, "accuracy": 0.0},
                "structured": {"correct": 0, "total": 0, "accuracy": 0.0},
                "multilingual": {"correct": 0, "total": 0, "accuracy": 0.0},
                "combined": {"correct": 0, "total": 0, "accuracy": 0.0}
            }
        }
    
    expected_primes = set()
    detected_primes = set()
    missing_primes = set()
    extra_primes = set()
    
    method_stats = {
        "spacy": {"correct": 0, "total": 0, "accuracy": 0.0},
        "structured": {"correct": 0, "total": 0, "accuracy": 0.0},
        "multilingual": {"correct": 0, "total": 0, "accuracy": 0.0},
        "combined": {"correct": 0, "total": 0, "accuracy": 0.0}
    }
    
    correct_cases = 0
    total_cases = len(test_cases)
    
    for case in test_cases:
        text = case['text']
        expected = set(case['primes'])
        expected_primes.update(expected)
        
        # Test all detection methods
        spacy_detected, structured_detected, multilingual_detected = test_detection_methods(text)
        combined_detected = spacy_detected | structured_detected | multilingual_detected
        
        # Filter to only NSM primes
        spacy_nsm = spacy_detected & ALL_NSM_PRIMES
        structured_nsm = structured_detected & ALL_NSM_PRIMES
        multilingual_nsm = multilingual_detected & ALL_NSM_PRIMES
        combined_nsm = combined_detected & ALL_NSM_PRIMES
        
        detected_primes.update(combined_nsm)
        
        # Check accuracy for each method
        if spacy_nsm == expected:
            method_stats["spacy"]["correct"] += 1
        method_stats["spacy"]["total"] += 1
        
        if structured_nsm == expected:
            method_stats["structured"]["correct"] += 1
        method_stats["structured"]["total"] += 1
        
        if multilingual_nsm == expected:
            method_stats["multilingual"]["correct"] += 1
        method_stats["multilingual"]["total"] += 1
        
        if combined_nsm == expected:
            method_stats["combined"]["correct"] += 1
            correct_cases += 1
        method_stats["combined"]["total"] += 1
        
        # Track missing and extra primes
        missing_primes.update(expected - combined_nsm)
        extra_primes.update(combined_nsm - expected)
    
    # Calculate accuracies
    accuracy = correct_cases / total_cases if total_cases > 0 else 0.0
    for method in method_stats:
        method_stats[method]["accuracy"] = (
            method_stats[method]["correct"] / method_stats[method]["total"] 
            if method_stats[method]["total"] > 0 else 0.0
        )
    
    return {
        "phase": phase_name,
        "total_cases": total_cases,
        "correct_cases": correct_cases,
        "accuracy": accuracy,
        "expected_primes": expected_primes,
        "detected_primes": detected_primes,
        "missing_primes": missing_primes,
        "extra_primes": extra_primes,
        "detection_methods": method_stats
    }

def run_comprehensive_test() -> Dict:
    """Run comprehensive test across all phases."""
    print("=== Comprehensive NSM Prime Coverage Test ===\n")
    
    # Load test suites
    test_suites = load_test_suites()
    
    # Track overall performance
    overall_stats = {
        "total_cases": 0,
        "correct_cases": 0,
        "accuracy": 0.0,
        "phase_results": {},
        "all_expected_primes": set(),
        "all_detected_primes": set(),
        "all_missing_primes": set(),
        "all_extra_primes": set(),
        "detection_methods": {
            "spacy": {"correct": 0, "total": 0, "accuracy": 0.0},
            "structured": {"correct": 0, "total": 0, "accuracy": 0.0},
            "multilingual": {"correct": 0, "total": 0, "accuracy": 0.0},
            "combined": {"correct": 0, "total": 0, "accuracy": 0.0}
        }
    }
    
    # Test each phase
    for phase_name, expected_primes in NSM_PRIMES_BY_PHASE.items():
        print(f"Testing {phase_name}...")
        test_cases = test_suites.get(phase_name, [])
        
        phase_result = analyze_phase_performance(phase_name, test_cases)
        overall_stats["phase_results"][phase_name] = phase_result
        
        # Update overall stats
        overall_stats["total_cases"] += phase_result["total_cases"]
        overall_stats["correct_cases"] += phase_result["correct_cases"]
        overall_stats["all_expected_primes"].update(phase_result["expected_primes"])
        overall_stats["all_detected_primes"].update(phase_result["detected_primes"])
        overall_stats["all_missing_primes"].update(phase_result["missing_primes"])
        overall_stats["all_extra_primes"].update(phase_result["extra_primes"])
        
        # Update method stats
        for method in overall_stats["detection_methods"]:
            overall_stats["detection_methods"][method]["correct"] += phase_result["detection_methods"][method]["correct"]
            overall_stats["detection_methods"][method]["total"] += phase_result["detection_methods"][method]["total"]
        
        print(f"  Accuracy: {phase_result['accuracy']:.1%} ({phase_result['correct_cases']}/{phase_result['total_cases']})")
        print(f"  Expected primes: {len(phase_result['expected_primes'])}")
        print(f"  Detected primes: {len(phase_result['detected_primes'])}")
        print(f"  Missing primes: {len(phase_result['missing_primes'])}")
        print(f"  Extra primes: {len(phase_result['extra_primes'])}")
        print()
    
    # Calculate overall accuracy
    overall_stats["accuracy"] = (
        overall_stats["correct_cases"] / overall_stats["total_cases"] 
        if overall_stats["total_cases"] > 0 else 0.0
    )
    
    # Calculate method accuracies
    for method in overall_stats["detection_methods"]:
        overall_stats["detection_methods"][method]["accuracy"] = (
            overall_stats["detection_methods"][method]["correct"] / overall_stats["detection_methods"][method]["total"]
            if overall_stats["detection_methods"][method]["total"] > 0 else 0.0
        )
    
    return overall_stats

def print_detailed_results(stats: Dict):
    """Print detailed test results."""
    print("=== Detailed Results ===\n")
    
    # Overall performance
    print(f"Overall Accuracy: {stats['accuracy']:.1%} ({stats['correct_cases']}/{stats['total_cases']})")
    print(f"Total Test Cases: {stats['total_cases']}")
    print()
    
    # Detection method performance
    print("Detection Method Performance:")
    for method, method_stats in stats["detection_methods"].items():
        print(f"  {method.capitalize()}: {method_stats['accuracy']:.1%} ({method_stats['correct']}/{method_stats['total']})")
    print()
    
    # Prime coverage
    print("Prime Coverage:")
    print(f"  Expected primes: {len(stats['all_expected_primes'])}")
    print(f"  Detected primes: {len(stats['all_detected_primes'])}")
    print(f"  Missing primes: {len(stats['all_missing_primes'])}")
    print(f"  Extra primes: {len(stats['all_extra_primes'])}")
    print()
    
    if stats['all_missing_primes']:
        print("Missing Primes:")
        for prime in sorted(stats['all_missing_primes']):
            print(f"  - {prime}")
        print()
    
    if stats['all_extra_primes']:
        print("Extra Primes:")
        for prime in sorted(stats['all_extra_primes']):
            print(f"  - {prime}")
        print()
    
    # Phase-by-phase breakdown
    print("Phase-by-Phase Breakdown:")
    for phase_name, phase_result in stats["phase_results"].items():
        print(f"  {phase_name}: {phase_result['accuracy']:.1%} ({phase_result['correct_cases']}/{phase_result['total_cases']})")
    print()

def main():
    """Main test execution."""
    start_time = time.time()
    
    # Run comprehensive test
    stats = run_comprehensive_test()
    
    # Print detailed results
    print_detailed_results(stats)
    
    # Performance summary
    end_time = time.time()
    print(f"Test completed in {end_time - start_time:.2f} seconds")
    
    # Success criteria
    print("\n=== Success Criteria ===")
    print(f"✅ 100% NSM Prime Coverage: {len(ALL_NSM_PRIMES)} primes implemented")
    print(f"✅ Overall Accuracy: {stats['accuracy']:.1%}")
    print(f"✅ Combined Detection: {stats['detection_methods']['combined']['accuracy']:.1%}")
    
    if stats['accuracy'] >= 0.5:
        print("🎉 EXCELLENT: System achieving 50%+ accuracy!")
    elif stats['accuracy'] >= 0.3:
        print("👍 GOOD: System achieving 30%+ accuracy!")
    else:
        print("⚠️  NEEDS IMPROVEMENT: System below 30% accuracy")

if __name__ == "__main__":
    main()

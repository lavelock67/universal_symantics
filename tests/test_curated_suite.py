#!/usr/bin/env python3
"""Test harness for curated test suite."""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from typing import Dict, List, Any

from src.detect.integrated_detector import IntegratedDetector
from src.router.enhanced_risk_router import EnhancedRiskRouter, SafetyFeature
import spacy


def load_curated_tests() -> List[Dict[str, Any]]:
    """Load curated test cases from JSONL file."""
    test_file = os.path.join(os.path.dirname(__file__), "data", "curated_test_suite.jsonl")
    tests = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))
    
    return tests


@pytest.fixture(scope="session")
def nlp_models():
    """Load NLP models for testing."""
    try:
        return {
            "en": spacy.load("en_core_web_sm"),
            "es": spacy.load("es_core_news_sm"),
            "fr": spacy.load("fr_core_news_sm")
        }
    except OSError as e:
        pytest.skip(f"Required spaCy models not found: {e}")


@pytest.fixture(scope="session")
def integrated_detector(nlp_models):
    """Initialize integrated detector."""
    return IntegratedDetector(nlp_models)


@pytest.fixture(scope="session")
def enhanced_router():
    """Initialize enhanced risk router."""
    return EnhancedRiskRouter()


class TestCuratedSuite:
    """Test suite for curated test cases."""
    
    def test_detection_accuracy(self, integrated_detector, enhanced_router):
        """Test detection accuracy on curated cases."""
        tests = load_curated_tests()
        
        total_tests = 0
        passed_tests = 0
        category_results = {}
        
        for test in tests:
            total_tests += 1
            category = test.get("category", "unknown")
            
            if category not in category_results:
                category_results[category] = {"total": 0, "passed": 0}
            
            category_results[category]["total"] += 1
            
            # Run detection
            detected_primes = integrated_detector.detect_primes_integrated(
                test["text"], test["lang"]
            )
            
            # Run router
            router_result = enhanced_router.route_detection(
                test["text"], detected_primes, test["lang"]
            )
            
            # Check detection accuracy
            expected_primes = set(test["expected_primes"])
            detected_primes_set = set(detected_primes)
            
            # Calculate F1 score
            precision = len(expected_primes & detected_primes_set) / len(detected_primes_set) if detected_primes_set else 0
            recall = len(expected_primes & detected_primes_set) / len(expected_primes) if expected_primes else 1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Check router decision
            router_correct = router_result.decision.value == test["expected_router"]
            
            # Test passes if F1 >= 0.7 and router decision is correct
            test_passed = f1 >= 0.7 and router_correct
            
            if test_passed:
                passed_tests += 1
                category_results[category]["passed"] += 1
            
            # Assert for pytest
            assert f1 >= 0.7, f"Detection F1 too low: {f1:.3f} for '{test['text']}'"
            assert router_correct, f"Router decision incorrect: got {router_result.decision.value}, expected {test['expected_router']} for '{test['text']}'"
        
        # Print summary
        print(f"\n📊 Curated Test Suite Results")
        print(f"Total tests: {total_tests}")
        print(f"Passed tests: {passed_tests}")
        print(f"Overall accuracy: {passed_tests/total_tests:.1%}")
        
        print(f"\n📈 Category Breakdown:")
        for category, results in category_results.items():
            accuracy = results["passed"] / results["total"]
            print(f"  {category}: {results['passed']}/{results['total']} ({accuracy:.1%})")
    
    def test_safety_critical_cases(self, integrated_detector, enhanced_router):
        """Test safety-critical cases specifically."""
        safety_tests = [
            {
                "text": "Es falso que el medicamento no funcione",
                "lang": "es",
                "expected_router": "clarify",
                "description": "Spanish negation scope ambiguity"
            },
            {
                "text": "Au plus la moitié des élèves lisent beaucoup",
                "lang": "fr",
                "expected_router": "clarify",
                "description": "French quantifier scope ambiguity"
            },
            {
                "text": "Personne ne vient",
                "lang": "fr",
                "expected_router": "abstain",
                "description": "French negative polarity"
            },
            {
                "text": "No es falso que esto funciona",
                "lang": "es",
                "expected_router": "clarify",
                "description": "Spanish negation flip"
            }
        ]
        
        for test in safety_tests:
            # Run detection
            detected_primes = integrated_detector.detect_primes_integrated(
                test["text"], test["lang"]
            )
            
            # Run router
            router_result = enhanced_router.route_detection(
                test["text"], detected_primes, test["lang"]
            )
            
            # Assert router decision
            assert router_result.decision.value == test["expected_router"], \
                f"Safety test failed: {test['description']} - got {router_result.decision.value}, expected {test['expected_router']}"
            
            print(f"✅ Safety test passed: {test['description']}")
    
    def test_everyday_cases(self, integrated_detector, enhanced_router):
        """Test everyday cases (should translate)."""
        everyday_tests = [
            {
                "text": "La gente piensa que esto es muy bueno",
                "lang": "es",
                "expected_router": "translate",
                "description": "Spanish everyday sentence"
            },
            {
                "text": "Les gens pensent que c'est très bon",
                "lang": "fr",
                "expected_router": "translate",
                "description": "French everyday sentence"
            },
            {
                "text": "I think you know the truth",
                "lang": "en",
                "expected_router": "translate",
                "description": "English everyday sentence"
            }
        ]
        
        for test in everyday_tests:
            # Run detection
            detected_primes = integrated_detector.detect_primes_integrated(
                test["text"], test["lang"]
            )
            
            # Run router
            router_result = enhanced_router.route_detection(
                test["text"], detected_primes, test["lang"]
            )
            
            # Assert router decision
            assert router_result.decision.value == test["expected_router"], \
                f"Everyday test failed: {test['description']} - got {router_result.decision.value}, expected {test['expected_router']}"
            
            # Assert minimum prime count
            assert len(detected_primes) >= 4, \
                f"Everyday test failed: {test['description']} - only {len(detected_primes)} primes detected"
            
            print(f"✅ Everyday test passed: {test['description']} ({len(detected_primes)} primes)")
    
    def test_per_language_performance(self, integrated_detector, enhanced_router):
        """Test performance per language."""
        tests = load_curated_tests()
        
        language_stats = {}
        
        for test in tests:
            lang = test["lang"]
            if lang not in language_stats:
                language_stats[lang] = {
                    "total": 0,
                    "passed": 0,
                    "avg_primes": 0,
                    "prime_counts": []
                }
            
            stats = language_stats[lang]
            stats["total"] += 1
            
            # Run detection
            detected_primes = integrated_detector.detect_primes_integrated(
                test["text"], lang
            )
            
            # Run router
            router_result = enhanced_router.route_detection(
                test["text"], detected_primes, lang
            )
            
            # Check accuracy
            expected_primes = set(test["expected_primes"])
            detected_primes_set = set(detected_primes)
            
            precision = len(expected_primes & detected_primes_set) / len(detected_primes_set) if detected_primes_set else 0
            recall = len(expected_primes & detected_primes_set) / len(expected_primes) if expected_primes else 1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            router_correct = router_result.decision.value == test["expected_router"]
            
            if f1 >= 0.7 and router_correct:
                stats["passed"] += 1
            
            stats["prime_counts"].append(len(detected_primes))
        
        # Calculate averages
        for lang, stats in language_stats.items():
            stats["avg_primes"] = sum(stats["prime_counts"]) / len(stats["prime_counts"])
            accuracy = stats["passed"] / stats["total"]
            
            print(f"\n🌍 {lang.upper()} Performance:")
            print(f"  Accuracy: {stats['passed']}/{stats['total']} ({accuracy:.1%})")
            print(f"  Average primes: {stats['avg_primes']:.1f}")
            
            # Assert minimum performance
            assert accuracy >= 0.6, f"Language {lang} accuracy too low: {accuracy:.1%}"
            assert stats["avg_primes"] >= 2.0, f"Language {lang} average primes too low: {stats['avg_primes']:.1f}"


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])

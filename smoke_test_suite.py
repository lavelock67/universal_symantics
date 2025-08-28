#!/usr/bin/env python3
"""
Smoke Test Suite

A focused test suite to validate core functionality and catch critical issues.
Based on the feedback requirements for a "tiny smoke set you can run today."
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from typing import Dict, List, Any, Tuple
import time

from src.core.application.services import NSMDetectionService
from src.core.translation.unified_translation_pipeline import UnifiedTranslationPipeline
from src.core.domain.models import Language
from cultural_adaptation_system import CulturalAdaptationSystem

class SmokeTestSuite:
    """Focused smoke test suite for core functionality."""
    
    def __init__(self):
        """Initialize the smoke test suite."""
        self.detection_service = NSMDetectionService()
        self.translation_pipeline = UnifiedTranslationPipeline()
        self.cultural_adaptation = CulturalAdaptationSystem()
        
        # Test cases from the feedback
        self.smoke_tests = [
            {
                "name": "Spanish Quantifier Scope",
                "text": "La gente piensa que esto es muy bueno.",
                "language": Language.SPANISH,
                "expected_primes": ["PEOPLE", "THINK", "THIS", "VERY", "GOOD"],
                "description": "Test quantifier scope and basic semantic detection"
            },
            {
                "name": "French Negation and Quantifier",
                "text": "Au plus la moitié des élèves lisent beaucoup.",
                "language": Language.FRENCH,
                "expected_primes": ["NOT", "MORE", "HALF", "PEOPLE", "READ", "MANY"],
                "description": "Test negation scope and quantifier handling"
            },
            {
                "name": "Spanish Negation and Truth",
                "text": "Es falso que el medicamento no funcione.",
                "language": Language.SPANISH,
                "expected_primes": ["FALSE", "NOT", "DO", "HAPPEN"],
                "description": "Test truth values and negation scope"
            },
            {
                "name": "Spanish Spatial Relation",
                "text": "El libro está dentro de la caja.",
                "language": Language.SPANISH,
                "expected_primes": ["INSIDE"],  # After adding missing primes
                "description": "Test spatial relation detection"
            },
            {
                "name": "French Spatial Relation",
                "text": "Le livre est dans la boîte.",
                "language": Language.FRENCH,
                "expected_primes": ["INSIDE"],  # After adding missing primes
                "description": "Test spatial relation detection"
            },
            {
                "name": "Cultural Adapter Invariant Test",
                "text": "Send me the report now.",
                "source_lang": Language.ENGLISH,
                "target_lang": Language.SPANISH,
                "description": "Test cultural adaptation preserves time invariants"
            }
        ]
    
    def run_prime_detection_tests(self) -> Dict[str, Any]:
        """Run prime detection smoke tests."""
        
        print("🧪 PRIME DETECTION SMOKE TESTS")
        print("=" * 50)
        
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for test in self.smoke_tests:
            if "expected_primes" not in test:
                continue
                
            results["total_tests"] += 1
            print(f"\n🔍 Test: {test['name']}")
            print(f"Text: {test['text']}")
            print(f"Language: {test['language'].value}")
            print(f"Expected: {test['expected_primes']}")
            
            try:
                # Detect primes
                detection_result = self.detection_service.detect_primes(
                    test['text'], 
                    test['language']
                )
                
                detected_primes = [p.text for p in detection_result.primes]
                print(f"Detected: {detected_primes}")
                
                # Check if expected primes are detected
                missing_primes = [p for p in test['expected_primes'] if p not in detected_primes]
                unexpected_primes = [p for p in detected_primes if p not in test['expected_primes']]
                
                if not missing_primes and not unexpected_primes:
                    print("✅ PASSED")
                    results["passed"] += 1
                    results["details"].append({
                        "test": test['name'],
                        "status": "PASSED",
                        "detected": detected_primes,
                        "expected": test['expected_primes']
                    })
                else:
                    print("❌ FAILED")
                    if missing_primes:
                        print(f"  Missing: {missing_primes}")
                    if unexpected_primes:
                        print(f"  Unexpected: {unexpected_primes}")
                    results["failed"] += 1
                    results["details"].append({
                        "test": test['name'],
                        "status": "FAILED",
                        "detected": detected_primes,
                        "expected": test['expected_primes'],
                        "missing": missing_primes,
                        "unexpected": unexpected_primes
                    })
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
                results["failed"] += 1
                results["details"].append({
                    "test": test['name'],
                    "status": "ERROR",
                    "error": str(e)
                })
        
        return results
    
    def run_cultural_adaptation_tests(self) -> Dict[str, Any]:
        """Run cultural adaptation smoke tests."""
        
        print("\n🌍 CULTURAL ADAPTATION SMOKE TESTS")
        print("=" * 50)
        
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        # Test cultural adaptation with invariant protection
        test_cases = [
            {
                "name": "Time Invariant Preservation",
                "text": "Send me the report now.",
                "target_lang": Language.SPANISH,
                "cultural_context": "es_ES",
                "expected_invariants": ["now"],
                "description": "Time expression should be preserved"
            },
            {
                "name": "Number Invariant Preservation",
                "text": "I need 5 books by tomorrow.",
                "target_lang": Language.FRENCH,
                "cultural_context": "fr_FR",
                "expected_invariants": ["5", "tomorrow"],
                "description": "Numbers and time should be preserved"
            },
            {
                "name": "Truth Value Invariant",
                "text": "This is true and that is false.",
                "target_lang": Language.GERMAN,
                "cultural_context": "de_DE",
                "expected_invariants": ["true", "false"],
                "description": "Truth values should be preserved"
            }
        ]
        
        for test in test_cases:
            results["total_tests"] += 1
            print(f"\n🔍 Test: {test['name']}")
            print(f"Text: {test['text']}")
            print(f"Target: {test['target_lang'].value}")
            print(f"Expected invariants: {test['expected_invariants']}")
            
            try:
                # Apply cultural adaptation
                adaptation_result = self.cultural_adaptation.adapt_text(
                    test['text'],
                    test['target_lang'],
                    test['cultural_context']
                )
                
                print(f"Adapted: {adaptation_result.adapted_text}")
                print(f"Invariants checked: {adaptation_result.invariants_checked}")
                print(f"Invariants violated: {adaptation_result.invariants_violated}")
                
                if adaptation_result.invariants_violated:
                    print(f"Violations: {adaptation_result.violation_details}")
                
                # Check if invariants are preserved
                invariant_preserved = True
                for invariant in test['expected_invariants']:
                    if invariant.lower() not in adaptation_result.adapted_text.lower():
                        invariant_preserved = False
                        break
                
                if invariant_preserved and not adaptation_result.invariants_violated:
                    print("✅ PASSED")
                    results["passed"] += 1
                    results["details"].append({
                        "test": test['name'],
                        "status": "PASSED",
                        "adapted_text": adaptation_result.adapted_text,
                        "invariants_preserved": True
                    })
                else:
                    print("❌ FAILED")
                    results["failed"] += 1
                    results["details"].append({
                        "test": test['name'],
                        "status": "FAILED",
                        "adapted_text": adaptation_result.adapted_text,
                        "invariants_preserved": False,
                        "violations": adaptation_result.violation_details
                    })
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
                results["failed"] += 1
                results["details"].append({
                    "test": test['name'],
                    "status": "ERROR",
                    "error": str(e)
                })
        
        return results
    
    def run_translation_pipeline_tests(self) -> Dict[str, Any]:
        """Run translation pipeline smoke tests."""
        
        print("\n🌐 TRANSLATION PIPELINE SMOKE TESTS")
        print("=" * 50)
        
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        # Test translation pipeline
        test_cases = [
            {
                "name": "English to Spanish Translation",
                "source_text": "The boy kicked the ball.",
                "source_lang": Language.ENGLISH,
                "target_lang": Language.SPANISH,
                "description": "Basic translation test"
            },
            {
                "name": "Spanish to French Translation",
                "source_text": "La gente piensa que esto es muy bueno.",
                "source_lang": Language.SPANISH,
                "target_lang": Language.FRENCH,
                "description": "Cross-language translation test"
            }
        ]
        
        for test in test_cases:
            results["total_tests"] += 1
            print(f"\n🔍 Test: {test['name']}")
            print(f"Source: {test['source_text']}")
            print(f"Translation: {test['source_lang'].value} → {test['target_lang'].value}")
            
            try:
                # Attempt translation
                translated_text = self.translation_pipeline.translate_simple(
                    test['source_text'],
                    test['source_lang'],
                    test['target_lang']
                )
                
                print(f"Result: {translated_text}")
                
                # Basic validation
                if translated_text and not translated_text.startswith("[Translation Error"):
                    print("✅ PASSED")
                    results["passed"] += 1
                    results["details"].append({
                        "test": test['name'],
                        "status": "PASSED",
                        "translated_text": translated_text
                    })
                else:
                    print("❌ FAILED")
                    results["failed"] += 1
                    results["details"].append({
                        "test": test['name'],
                        "status": "FAILED",
                        "translated_text": translated_text
                    })
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
                results["failed"] += 1
                results["details"].append({
                    "test": test['name'],
                    "status": "ERROR",
                    "error": str(e)
                })
        
        return results
    
    def run_comprehensive_smoke_test(self) -> Dict[str, Any]:
        """Run all smoke tests and provide summary."""
        
        print("🚀 COMPREHENSIVE SMOKE TEST SUITE")
        print("=" * 60)
        print("Running focused tests to validate core functionality...")
        print()
        
        start_time = time.time()
        
        # Run all test suites
        prime_results = self.run_prime_detection_tests()
        cultural_results = self.run_cultural_adaptation_tests()
        translation_results = self.run_translation_pipeline_tests()
        
        end_time = time.time()
        
        # Compile comprehensive results
        total_tests = (prime_results["total_tests"] + 
                      cultural_results["total_tests"] + 
                      translation_results["total_tests"])
        
        total_passed = (prime_results["passed"] + 
                       cultural_results["passed"] + 
                       translation_results["passed"])
        
        total_failed = (prime_results["failed"] + 
                       cultural_results["failed"] + 
                       translation_results["failed"])
        
        # Print summary
        print("\n📊 SMOKE TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        print(f"Execution Time: {end_time - start_time:.2f}s")
        
        print("\n📋 DETAILED RESULTS")
        print("-" * 30)
        
        print(f"\nPrime Detection:")
        print(f"  Tests: {prime_results['total_tests']}, Passed: {prime_results['passed']}, Failed: {prime_results['failed']}")
        
        print(f"\nCultural Adaptation:")
        print(f"  Tests: {cultural_results['total_tests']}, Passed: {cultural_results['passed']}, Failed: {cultural_results['failed']}")
        
        print(f"\nTranslation Pipeline:")
        print(f"  Tests: {translation_results['total_tests']}, Passed: {translation_results['passed']}, Failed: {translation_results['failed']}")
        
        # Identify critical issues
        critical_issues = []
        
        if prime_results["failed"] > 0:
            critical_issues.append("Prime detection failures - core functionality affected")
        
        if cultural_results["failed"] > 0:
            critical_issues.append("Cultural adaptation failures - invariant violations")
        
        if translation_results["failed"] > 0:
            critical_issues.append("Translation pipeline failures - end-to-end broken")
        
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES IDENTIFIED:")
            for issue in critical_issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ No critical issues identified!")
        
        return {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "success_rate": (total_passed/total_tests*100) if total_tests > 0 else 0,
            "execution_time": end_time - start_time,
            "critical_issues": critical_issues,
            "prime_detection": prime_results,
            "cultural_adaptation": cultural_results,
            "translation_pipeline": translation_results
        }

def main():
    """Run the comprehensive smoke test suite."""
    
    print("🧪 SMOKE TEST SUITE - VALIDATION")
    print("=" * 60)
    print("This suite validates core functionality based on feedback requirements.")
    print()
    
    # Create and run smoke test suite
    smoke_suite = SmokeTestSuite()
    results = smoke_suite.run_comprehensive_smoke_test()
    
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 30)
    
    if results["success_rate"] >= 90:
        print("✅ System is ready for production deployment")
        print("✅ Core functionality is working correctly")
        print("✅ Proceed with confidence")
    elif results["success_rate"] >= 70:
        print("⚠️ System has some issues but is mostly functional")
        print("⚠️ Address critical issues before deployment")
        print("⚠️ Consider additional testing")
    else:
        print("❌ System has significant issues")
        print("❌ Address critical issues immediately")
        print("❌ Do not deploy until issues are resolved")
    
    print(f"\n📈 NEXT STEPS")
    print("-" * 20)
    
    if results["critical_issues"]:
        print("1. 🔧 Fix critical issues identified above")
        print("2. 🧪 Re-run smoke tests")
        print("3. ✅ Verify all tests pass")
        print("4. 🚀 Proceed with deployment")
    else:
        print("1. ✅ All critical functionality validated")
        print("2. 🚀 Ready for production deployment")
        print("3. 📊 Monitor performance in production")
        print("4. 🔄 Regular smoke test runs recommended")

if __name__ == "__main__":
    main()

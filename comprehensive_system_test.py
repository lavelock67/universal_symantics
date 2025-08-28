#!/usr/bin/env python3
"""
Comprehensive NSM System Test Suite

Tests the current NSM system for:
1. Functionality - Does it work as advertised?
2. Accuracy - How well does it perform?
3. Usefulness - Is it actually helpful for real tasks?
4. Robustness - How does it handle edge cases?
5. Consistency - Are results reliable?
"""

import requests
import json
import time
import statistics
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestResult:
    """Represents a test result."""
    test_name: str
    success: bool
    score: float  # 0-1 scale
    details: str
    execution_time: float
    raw_response: Any = None

class NSMSystemTester:
    """Comprehensive tester for the NSM system."""
    
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url
        self.results = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        print("🧪 Starting Comprehensive NSM System Tests...")
        print("=" * 60)
        
        # Test categories
        test_categories = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Prime Detection Accuracy", self.test_prime_detection_accuracy),
            ("MWE Detection Accuracy", self.test_mwe_detection_accuracy),
            ("Text Generation Quality", self.test_text_generation_quality),
            ("Cross-Lingual Capabilities", self.test_cross_lingual_capabilities),
            ("System Performance", self.test_system_performance),
            ("Error Handling", self.test_error_handling),
            ("Real-World Usefulness", self.test_real_world_usefulness),
            ("Consistency & Reliability", self.test_consistency_reliability),
        ]
        
        category_results = {}
        
        for category_name, test_func in test_categories:
            print(f"\n📋 {category_name}")
            print("-" * 40)
            category_results[category_name] = test_func()
            
        # Generate comprehensive report
        return self.generate_report(category_results)
    
    def test_basic_functionality(self) -> List[TestResult]:
        """Test basic API functionality."""
        results = []
        
        # Test 1: Health endpoint
        start_time = time.time()
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            success = response.status_code == 200
            score = 1.0 if success else 0.0
            details = f"Health check: {response.status_code}"
            results.append(TestResult(
                "Health Check", success, score, details, 
                time.time() - start_time, response.json()
            ))
        except Exception as e:
            results.append(TestResult(
                "Health Check", False, 0.0, f"Error: {str(e)}", 
                time.time() - start_time
            ))
        
        # Test 2: Prime detection basic
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_base_url}/detect",
                json={"text": "People think this is good", "language": "en"},
                timeout=10
            )
            success = response.status_code == 200
            data = response.json() if success else {}
            score = 1.0 if success and data.get("success") else 0.0
            details = f"Prime detection: {len(data.get('result', {}).get('primes', []))} primes found"
            results.append(TestResult(
                "Prime Detection Basic", success, score, details,
                time.time() - start_time, data
            ))
        except Exception as e:
            results.append(TestResult(
                "Prime Detection Basic", False, 0.0, f"Error: {str(e)}",
                time.time() - start_time
            ))
        
        # Test 3: MWE detection basic
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_base_url}/mwe",
                json={"text": "At least half of the students", "language": "en"},
                timeout=10
            )
            success = response.status_code == 200
            data = response.json() if success else {}
            score = 1.0 if success and data.get("success") else 0.0
            details = f"MWE detection: {len(data.get('mwes', []))} MWEs found"
            results.append(TestResult(
                "MWE Detection Basic", success, score, details,
                time.time() - start_time, data
            ))
        except Exception as e:
            results.append(TestResult(
                "MWE Detection Basic", False, 0.0, f"Error: {str(e)}",
                time.time() - start_time
            ))
        
        # Test 4: Text generation basic
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_base_url}/generate",
                json={"primes": ["GOOD"], "target_language": "en"},
                timeout=10
            )
            success = response.status_code == 200
            data = response.json() if success else {}
            score = 1.0 if success and data.get("success") else 0.0
            details = f"Text generation: '{data.get('result', {}).get('generated_text', 'N/A')}'"
            results.append(TestResult(
                "Text Generation Basic", success, score, details,
                time.time() - start_time, data
            ))
        except Exception as e:
            results.append(TestResult(
                "Text Generation Basic", False, 0.0, f"Error: {str(e)}",
                time.time() - start_time
            ))
        
        return results
    
    def test_prime_detection_accuracy(self) -> List[TestResult]:
        """Test prime detection accuracy with known examples."""
        results = []
        
        test_cases = [
            {
                "text": "People think this is very good",
                "expected_primes": ["PEOPLE", "THINK", "VERY", "GOOD"],
                "description": "Basic evaluative statement"
            },
            {
                "text": "I know that you want to read many books",
                "expected_primes": ["I", "KNOW", "YOU", "WANT", "READ", "MANY"],
                "description": "Complex mental predicate"
            },
            {
                "text": "This is not bad at all",
                "expected_primes": ["THIS", "NOT", "BAD"],
                "description": "Negation with evaluator"
            },
            {
                "text": "Some people do this thing",
                "expected_primes": ["SOME", "PEOPLE", "DO", "THIS"],
                "description": "Quantification with action"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.api_base_url}/detect",
                    json={"text": test_case["text"], "language": "en"},
                    timeout=10
                )
                success = response.status_code == 200
                data = response.json() if success else {}
                
                if success and data.get("success"):
                    detected_primes = [p["text"].upper() for p in data.get("result", {}).get("primes", [])]
                    expected = test_case["expected_primes"]
                    
                    # Calculate accuracy
                    correct = sum(1 for prime in expected if prime in detected_primes)
                    precision = correct / len(detected_primes) if detected_primes else 0
                    recall = correct / len(expected) if expected else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    score = f1
                    details = f"{test_case['description']}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}"
                else:
                    score = 0.0
                    details = f"API call failed: {data.get('error', 'Unknown error')}"
                
                results.append(TestResult(
                    f"Prime Detection {i+1}", success, score, details,
                    time.time() - start_time, data
                ))
                
            except Exception as e:
                results.append(TestResult(
                    f"Prime Detection {i+1}", False, 0.0, f"Error: {str(e)}",
                    time.time() - start_time
                ))
        
        return results
    
    def test_mwe_detection_accuracy(self) -> List[TestResult]:
        """Test MWE detection accuracy."""
        results = []
        
        test_cases = [
            {
                "text": "At least half of the students read a lot of books",
                "expected_mwes": ["at least", "half of", "a lot of"],
                "description": "Quantifier MWEs"
            },
            {
                "text": "The people think this is very good",
                "expected_mwes": ["very good"],
                "description": "Intensifier MWE"
            },
            {
                "text": "I do not want to read many books",
                "expected_mwes": ["do not"],
                "description": "Negation MWE"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.api_base_url}/mwe",
                    json={"text": test_case["text"], "language": "en"},
                    timeout=10
                )
                success = response.status_code == 200
                data = response.json() if success else {}
                
                if success and data.get("success"):
                    detected_mwes = [mwe["text"] for mwe in data.get("mwes", [])]
                    expected = test_case["expected_mwes"]
                    
                    # Calculate accuracy
                    correct = sum(1 for mwe in expected if mwe in detected_mwes)
                    precision = correct / len(detected_mwes) if detected_mwes else 0
                    recall = correct / len(expected) if expected else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    score = f1
                    details = f"{test_case['description']}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}"
                else:
                    score = 0.0
                    details = f"API call failed: {data.get('error', 'Unknown error')}"
                
                results.append(TestResult(
                    f"MWE Detection {i+1}", success, score, details,
                    time.time() - start_time, data
                ))
                
            except Exception as e:
                results.append(TestResult(
                    f"MWE Detection {i+1}", False, 0.0, f"Error: {str(e)}",
                    time.time() - start_time
                ))
        
        return results
    
    def test_text_generation_quality(self) -> List[TestResult]:
        """Test text generation quality."""
        results = []
        
        test_cases = [
            {
                "primes": ["GOOD"],
                "expected_patterns": ["good", "this is good"],
                "description": "Single evaluator"
            },
            {
                "primes": ["VERY", "GOOD"],
                "expected_patterns": ["very good"],
                "description": "Intensifier + evaluator"
            },
            {
                "primes": ["PEOPLE", "THINK", "GOOD"],
                "expected_patterns": ["people", "think", "good"],
                "description": "Complex statement"
            },
            {
                "primes": ["NOT", "GOOD"],
                "expected_patterns": ["not", "good"],
                "description": "Negation"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.api_base_url}/generate",
                    json={"primes": test_case["primes"], "target_language": "en"},
                    timeout=10
                )
                success = response.status_code == 200
                data = response.json() if success else {}
                
                if success and data.get("success"):
                    generated_text = data.get("result", {}).get("generated_text", "").lower()
                    expected_patterns = test_case["expected_patterns"]
                    
                    # Check if expected patterns are present
                    pattern_matches = sum(1 for pattern in expected_patterns if pattern in generated_text)
                    score = pattern_matches / len(expected_patterns) if expected_patterns else 0
                    
                    details = f"{test_case['description']}: '{generated_text}' (score: {score:.2f})"
                else:
                    score = 0.0
                    details = f"API call failed: {data.get('error', 'Unknown error')}"
                
                results.append(TestResult(
                    f"Text Generation {i+1}", success, score, details,
                    time.time() - start_time, data
                ))
                
            except Exception as e:
                results.append(TestResult(
                    f"Text Generation {i+1}", False, 0.0, f"Error: {str(e)}",
                    time.time() - start_time
                ))
        
        return results
    
    def test_cross_lingual_capabilities(self) -> List[TestResult]:
        """Test cross-lingual generation capabilities."""
        results = []
        
        test_cases = [
            {
                "primes": ["VERY", "GOOD"],
                "languages": ["en", "es", "fr"],
                "expected_patterns": {
                    "en": ["very good"],
                    "es": ["muy bueno"],
                    "fr": ["très bon"]
                },
                "description": "Cross-lingual intensifier"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            for lang in test_case["languages"]:
                start_time = time.time()
                try:
                    response = requests.post(
                        f"{self.api_base_url}/generate",
                        json={"primes": test_case["primes"], "target_language": lang},
                        timeout=10
                    )
                    success = response.status_code == 200
                    data = response.json() if success else {}
                    
                    if success and data.get("success"):
                        generated_text = data.get("result", {}).get("generated_text", "").lower()
                        expected_patterns = test_case["expected_patterns"][lang]
                        
                        # Check if expected patterns are present
                        pattern_matches = sum(1 for pattern in expected_patterns if pattern in generated_text)
                        score = pattern_matches / len(expected_patterns) if expected_patterns else 0
                        
                        details = f"{test_case['description']} ({lang}): '{generated_text}' (score: {score:.2f})"
                    else:
                        score = 0.0
                        details = f"API call failed: {data.get('error', 'Unknown error')}"
                    
                    results.append(TestResult(
                        f"Cross-Lingual {lang} {i+1}", success, score, details,
                        time.time() - start_time, data
                    ))
                    
                except Exception as e:
                    results.append(TestResult(
                        f"Cross-Lingual {lang} {i+1}", False, 0.0, f"Error: {str(e)}",
                        time.time() - start_time
                    ))
        
        return results
    
    def test_system_performance(self) -> List[TestResult]:
        """Test system performance and response times."""
        results = []
        
        # Test response times for different operations
        operations = [
            ("Health Check", "GET", "/health", None),
            ("Prime Detection", "POST", "/detect", {"text": "People think this is good", "language": "en"}),
            ("MWE Detection", "POST", "/mwe", {"text": "At least half of the students", "language": "en"}),
            ("Text Generation", "POST", "/generate", {"primes": ["GOOD"], "target_language": "en"})
        ]
        
        for op_name, method, endpoint, payload in operations:
            times = []
            successes = 0
            
            # Run 5 times to get average
            for _ in range(5):
                start_time = time.time()
                try:
                    if method == "GET":
                        response = requests.get(f"{self.api_base_url}{endpoint}", timeout=10)
                    else:
                        response = requests.post(f"{self.api_base_url}{endpoint}", json=payload, timeout=10)
                    
                    if response.status_code == 200:
                        successes += 1
                    times.append(time.time() - start_time)
                    
                except Exception:
                    times.append(10.0)  # Timeout value
            
            avg_time = statistics.mean(times)
            success_rate = successes / 5
            score = success_rate * (1.0 - min(avg_time / 5.0, 1.0))  # Penalize slow responses
            
            details = f"Avg time: {avg_time:.3f}s, Success rate: {success_rate:.2f}"
            results.append(TestResult(
                f"Performance {op_name}", success_rate > 0.8, score, details,
                avg_time
            ))
        
        return results
    
    def test_error_handling(self) -> List[TestResult]:
        """Test error handling and edge cases."""
        results = []
        
        error_cases = [
            {
                "endpoint": "/detect",
                "payload": {"text": "", "language": "en"},
                "description": "Empty text"
            },
            {
                "endpoint": "/detect",
                "payload": {"text": "A" * 10000, "language": "en"},  # Very long text
                "description": "Very long text"
            },
            {
                "endpoint": "/generate",
                "payload": {"primes": [], "target_language": "en"},
                "description": "Empty primes list"
            },
            {
                "endpoint": "/generate",
                "payload": {"primes": ["INVALID_PRIME"], "target_language": "en"},
                "description": "Invalid prime"
            }
        ]
        
        for i, test_case in enumerate(error_cases):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.api_base_url}{test_case['endpoint']}",
                    json=test_case["payload"],
                    timeout=10
                )
                
                # Should handle gracefully (not crash)
                success = response.status_code in [200, 400, 422]  # Acceptable responses
                data = response.json() if response.status_code == 200 else {}
                
                if success:
                    score = 1.0 if response.status_code == 200 and data.get("success") else 0.5
                    details = f"{test_case['description']}: Status {response.status_code}"
                else:
                    score = 0.0
                    details = f"{test_case['description']}: Unexpected status {response.status_code}"
                
                results.append(TestResult(
                    f"Error Handling {i+1}", success, score, details,
                    time.time() - start_time, data
                ))
                
            except Exception as e:
                results.append(TestResult(
                    f"Error Handling {i+1}", False, 0.0, f"Error: {str(e)}",
                    time.time() - start_time
                ))
        
        return results
    
    def test_real_world_usefulness(self) -> List[TestResult]:
        """Test real-world usefulness with practical examples."""
        results = []
        
        real_world_cases = [
            {
                "text": "The customer thinks this product is very good",
                "expected_insight": "Positive sentiment with intensifier",
                "description": "Customer feedback analysis"
            },
            {
                "text": "Many people want to read this book",
                "expected_insight": "Quantified interest",
                "description": "Interest quantification"
            },
            {
                "text": "This is not what I expected",
                "expected_insight": "Negative evaluation",
                "description": "Expectation mismatch"
            }
        ]
        
        for i, test_case in enumerate(real_world_cases):
            start_time = time.time()
            try:
                # Test prime detection
                response = requests.post(
                    f"{self.api_base_url}/detect",
                    json={"text": test_case["text"], "language": "en"},
                    timeout=10
                )
                success = response.status_code == 200
                data = response.json() if success else {}
                
                if success and data.get("success"):
                    primes = [p["text"] for p in data.get("result", {}).get("primes", [])]
                    
                    # Check if we can extract meaningful insights
                    has_evaluator = any(p.lower() in ["good", "bad", "not"] for p in primes)
                    has_quantifier = any(p.lower() in ["many", "some", "all"] for p in primes)
                    has_mental = any(p.lower() in ["think", "want", "know"] for p in primes)
                    
                    insight_score = (has_evaluator + has_quantifier + has_mental) / 3
                    score = insight_score
                    details = f"{test_case['description']}: Primes={primes}, Insight score={insight_score:.2f}"
                else:
                    score = 0.0
                    details = f"API call failed: {data.get('error', 'Unknown error')}"
                
                results.append(TestResult(
                    f"Real-World Usefulness {i+1}", success, score, details,
                    time.time() - start_time, data
                ))
                
            except Exception as e:
                results.append(TestResult(
                    f"Real-World Usefulness {i+1}", False, 0.0, f"Error: {str(e)}",
                    time.time() - start_time
                ))
        
        return results
    
    def test_consistency_reliability(self) -> List[TestResult]:
        """Test consistency and reliability of results."""
        results = []
        
        # Test consistency by running same input multiple times
        test_inputs = [
            ("Prime Detection", "/detect", {"text": "People think this is good", "language": "en"}),
            ("MWE Detection", "/mwe", {"text": "At least half of the students", "language": "en"}),
            ("Text Generation", "/generate", {"primes": ["GOOD"], "target_language": "en"})
        ]
        
        for op_name, endpoint, payload in test_inputs:
            start_time = time.time()
            responses = []
            
            # Run 3 times
            for _ in range(3):
                try:
                    response = requests.post(
                        f"{self.api_base_url}{endpoint}",
                        json=payload,
                        timeout=10
                    )
                    if response.status_code == 200:
                        responses.append(response.json())
                except Exception:
                    pass
            
            if len(responses) == 3:
                # Check if all responses are identical
                first_response = str(responses[0])
                consistency = all(str(r) == first_response for r in responses)
                score = 1.0 if consistency else 0.5
                details = f"Consistency: {consistency}, All 3 calls successful"
            else:
                score = 0.0
                details = f"Reliability: Only {len(responses)}/3 calls successful"
            
            results.append(TestResult(
                f"Consistency {op_name}", len(responses) >= 2, score, details,
                time.time() - start_time, responses
            ))
        
        return results
    
    def generate_report(self, category_results: Dict[str, List[TestResult]]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        total_score = 0.0
        category_scores = {}
        
        for category_name, results in category_results.items():
            print(f"\n📋 {category_name}")
            print("-" * 40)
            
            category_total = len(results)
            category_passed = sum(1 for r in results if r.success)
            category_score = statistics.mean([r.score for r in results]) if results else 0.0
            
            total_tests += category_total
            passed_tests += category_passed
            total_score += category_score * category_total
            
            category_scores[category_name] = {
                "total": category_total,
                "passed": category_passed,
                "score": category_score,
                "pass_rate": category_passed / category_total if category_total > 0 else 0
            }
            
            print(f"Tests: {category_passed}/{category_total} passed")
            print(f"Average Score: {category_score:.3f}")
            print(f"Pass Rate: {category_passed/category_total*100:.1f}%")
            
            # Show individual test results
            for result in results:
                status = "✅" if result.success else "❌"
                print(f"  {status} {result.test_name}: {result.score:.3f} ({result.details})")
        
        overall_score = total_score / total_tests if total_tests > 0 else 0.0
        overall_pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        print("\n" + "=" * 60)
        print("🎯 OVERALL ASSESSMENT")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed Tests: {passed_tests}")
        print(f"Overall Pass Rate: {overall_pass_rate*100:.1f}%")
        print(f"Overall Score: {overall_score:.3f}")
        
        # System assessment
        if overall_score >= 0.8:
            assessment = "EXCELLENT - System is working very well"
        elif overall_score >= 0.6:
            assessment = "GOOD - System is functional with some areas for improvement"
        elif overall_score >= 0.4:
            assessment = "FAIR - System needs significant work"
        else:
            assessment = "POOR - System needs major improvements"
        
        print(f"Assessment: {assessment}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 40)
        
        for category_name, scores in category_scores.items():
            if scores["score"] < 0.6:
                print(f"⚠️  {category_name}: Needs improvement (score: {scores['score']:.3f})")
            elif scores["score"] < 0.8:
                print(f"🔧 {category_name}: Could be enhanced (score: {scores['score']:.3f})")
            else:
                print(f"✅ {category_name}: Working well (score: {scores['score']:.3f})")
        
        # Convert TestResult objects to dictionaries for JSON serialization
        serializable_results = {}
        for category_name, results in category_results.items():
            serializable_results[category_name] = [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "score": r.score,
                    "details": r.details,
                    "execution_time": r.execution_time,
                    "raw_response": r.raw_response
                }
                for r in results
            ]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall_score,
            "overall_pass_rate": overall_pass_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "assessment": assessment,
            "category_scores": category_scores,
            "detailed_results": serializable_results
        }

def main():
    """Run comprehensive system tests."""
    tester = NSMSystemTester()
    report = tester.run_all_tests()
    
    # Save report to file
    with open(f"nsm_system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: nsm_system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

if __name__ == "__main__":
    main()

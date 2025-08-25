#!/usr/bin/env python3
"""
Real-World Demo Server for NSM Universal Translator

This module provides comprehensive demonstrations of our complete
universal translator stack with start-to-finish examples.
"""

import requests
import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class DemoResult:
    """Result of a demo test."""
    name: str
    success: bool
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    processing_time: float
    notes: str = ""

class NSMDemoServer:
    """Demo server for NSM Universal Translator."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """Initialize demo server.
        
        Args:
            base_url: Base URL for the NSM API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_roundtrip_translation(self) -> DemoResult:
        """Test round-trip, constraint-aware translation."""
        print("🔄 Testing Round-trip Translation...")
        
        input_data = {
            "source_text": "If it rains tomorrow, we will cancel the picnic.",
            "src_lang": "en",
            "tgt_lang": "es",
            "constraint_mode": "hybrid",
            "realizer": "fluent"
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/roundtrip",
                json=input_data,
                headers={"Content-Type": "application/json"}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                output_data = response.json()
                
                # Analyze results
                legality = output_data.get("legality", 0.0)
                molecule_ratio = output_data.get("molecule_ratio", 0.0)
                graph_f1 = output_data.get("drift", {}).get("graph_f1", 0.0)
                router_decision = output_data.get("router", {}).get("decision", "unknown")
                
                success = (legality >= 0.9 and molecule_ratio < 0.3 and 
                          graph_f1 >= 0.8 and router_decision == "translate")
                
                notes = f"Legality: {legality:.3f}, Molecule Ratio: {molecule_ratio:.3f}, Graph F1: {graph_f1:.3f}, Decision: {router_decision}"
                
                return DemoResult(
                    name="Round-trip Translation",
                    success=success,
                    input_data=input_data,
                    output_data=output_data,
                    processing_time=processing_time,
                    notes=notes
                )
            else:
                return DemoResult(
                    name="Round-trip Translation",
                    success=False,
                    input_data=input_data,
                    output_data={"error": f"HTTP {response.status_code}"},
                    processing_time=time.time() - start_time,
                    notes=f"Failed with status {response.status_code}"
                )
                
        except Exception as e:
            return DemoResult(
                name="Round-trip Translation",
                success=False,
                input_data=input_data,
                output_data={"error": str(e)},
                processing_time=time.time() - start_time,
                notes=f"Exception: {str(e)}"
            )
    
    def test_quantifier_mwe_detection(self) -> DemoResult:
        """Test quantifiers via MWE layer."""
        print("🔢 Testing Quantifier MWE Detection...")
        
        input_data = {
            "text": "At most half of the students read a lot of books",
            "include_coverage": True
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/mwe",
                json=input_data,
                headers={"Content-Type": "application/json"}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                output_data = response.json()
                
                # Analyze results
                detected_mwes = output_data.get("detected_mwes", [])
                primes = output_data.get("primes", [])
                coverage = output_data.get("coverage", {})
                
                # Check for expected MWEs
                expected_mwes = ["at most", "a lot of"]
                found_mwes = [mwe["text"] for mwe in detected_mwes]
                mwe_success = all(mwe in found_mwes for mwe in expected_mwes)
                
                # Check for expected primes
                expected_primes = ["NOT", "MORE", "MANY"]
                prime_success = all(prime in primes for prime in expected_primes)
                
                success = mwe_success and prime_success and len(detected_mwes) >= 2
                
                notes = f"Found {len(detected_mwes)} MWEs: {found_mwes}, Primes: {primes}, Coverage: {coverage.get('total', 0):.2f}"
                
                return DemoResult(
                    name="Quantifier MWE Detection",
                    success=success,
                    input_data=input_data,
                    output_data=output_data,
                    processing_time=processing_time,
                    notes=notes
                )
            else:
                return DemoResult(
                    name="Quantifier MWE Detection",
                    success=False,
                    input_data=input_data,
                    output_data={"error": f"HTTP {response.status_code}"},
                    processing_time=time.time() - start_time,
                    notes=f"Failed with status {response.status_code}"
                )
                
        except Exception as e:
            return DemoResult(
                name="Quantifier MWE Detection",
                success=False,
                input_data=input_data,
                output_data={"error": str(e)},
                processing_time=time.time() - start_time,
                notes=f"Exception: {str(e)}"
            )
    
    def test_risk_coverage_router(self) -> DemoResult:
        """Test risk-coverage router (selective correctness)."""
        print("🎯 Testing Risk-Coverage Router...")
        
        input_data = {
            "text": "It is true that he might not come.",
            "lang": "en"
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/router/route",
                json=input_data,
                headers={"Content-Type": "application/json"}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                output_data = response.json()
                
                # Analyze results
                decision = output_data.get("decision", "unknown")
                risk_estimate = output_data.get("risk_estimate", 1.0)
                reasons = output_data.get("reasons", [])
                
                # Expect clarify decision due to ambiguous scope
                success = (decision == "clarify" and 
                          risk_estimate > 0.1 and 
                          len(reasons) >= 1)
                
                notes = f"Decision: {decision}, Risk: {risk_estimate:.3f}, Reasons: {reasons}"
                
                return DemoResult(
                    name="Risk-Coverage Router",
                    success=success,
                    input_data=input_data,
                    output_data=output_data,
                    processing_time=processing_time,
                    notes=notes
                )
            else:
                return DemoResult(
                    name="Risk-Coverage Router",
                    success=False,
                    input_data=input_data,
                    output_data={"error": f"HTTP {response.status_code}"},
                    processing_time=time.time() - start_time,
                    notes=f"Failed with status {response.status_code}"
                )
                
        except Exception as e:
            return DemoResult(
                name="Risk-Coverage Router",
                success=False,
                input_data=input_data,
                output_data={"error": str(e)},
                processing_time=time.time() - start_time,
                notes=f"Exception: {str(e)}"
            )
    
    def test_constraint_ablation(self) -> DemoResult:
        """Test constraint ablation (why the grammar matters)."""
        print("📊 Testing Constraint Ablation...")
        
        input_data = {
            "text": "I think you know the truth",
            "lang": "en",
            "modes": ["off", "hybrid", "hard"]
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/ablation",
                json=input_data,
                headers={"Content-Type": "application/json"}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                output_data = response.json()
                
                # Analyze results
                runs = output_data.get("runs", [])
                
                if len(runs) == 3:
                    # Check that legality and graph-F1 improve from off→hybrid→hard
                    off_run = runs[0]
                    hybrid_run = runs[1]
                    hard_run = runs[2]
                    
                    legality_improvement = (off_run["legality"] < hybrid_run["legality"] < hard_run["legality"])
                    f1_improvement = (off_run["drift"]["graph_f1"] < hybrid_run["drift"]["graph_f1"] < hard_run["drift"]["graph_f1"])
                    
                    success = legality_improvement and f1_improvement
                    
                    notes = f"Off: legality={off_run['legality']:.3f}, F1={off_run['drift']['graph_f1']:.3f} | " \
                           f"Hybrid: legality={hybrid_run['legality']:.3f}, F1={hybrid_run['drift']['graph_f1']:.3f} | " \
                           f"Hard: legality={hard_run['legality']:.3f}, F1={hard_run['drift']['graph_f1']:.3f}"
                    
                    return DemoResult(
                        name="Constraint Ablation",
                        success=success,
                        input_data=input_data,
                        output_data=output_data,
                        processing_time=processing_time,
                        notes=notes
                    )
                else:
                    return DemoResult(
                        name="Constraint Ablation",
                        success=False,
                        input_data=input_data,
                        output_data=output_data,
                        processing_time=processing_time,
                        notes=f"Expected 3 runs, got {len(runs)}"
                    )
            else:
                return DemoResult(
                    name="Constraint Ablation",
                    success=False,
                    input_data=input_data,
                    output_data={"error": f"HTTP {response.status_code}"},
                    processing_time=time.time() - start_time,
                    notes=f"Failed with status {response.status_code}"
                )
                
        except Exception as e:
            return DemoResult(
                name="Constraint Ablation",
                success=False,
                input_data=input_data,
                output_data={"error": str(e)},
                processing_time=time.time() - start_time,
                notes=f"Exception: {str(e)}"
            )
    
    def test_cross_language_exponents(self) -> DemoResult:
        """Test cross-language exponent lexicons."""
        print("🌍 Testing Cross-Language Exponents...")
        
        test_cases = [
            {"prime": "THINK", "language": "en", "ud_features": {"VerbForm": "Fin", "Person": "3", "Number": "Sing"}},
            {"prime": "PEOPLE", "language": "es", "ud_features": {"Number": "Plur"}},
            {"prime": "GOOD", "language": "fr", "ud_features": {"Degree": "Pos"}}
        ]
        
        results = []
        start_time = time.time()
        
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/exponents",
                    json=test_case,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    best_exponent = result.get("best_exponent")
                    if best_exponent:
                        results.append({
                            "prime": test_case["prime"],
                            "language": test_case["language"],
                            "exponent": best_exponent["surface_form"],
                            "confidence": best_exponent["confidence"]
                        })
                
            except Exception as e:
                print(f"Failed to test {test_case}: {e}")
        
        processing_time = time.time() - start_time
        
        success = len(results) == 3
        
        notes = " | ".join([f"{r['language']} {r['prime']}: '{r['exponent']}' ({r['confidence']:.2f})" for r in results])
        
        return DemoResult(
            name="Cross-Language Exponents",
            success=success,
            input_data={"test_cases": test_cases},
            output_data={"results": results},
            processing_time=processing_time,
            notes=notes
        )
    
    def test_prime_discovery(self) -> DemoResult:
        """Test MDL-Δ prime discovery loop."""
        print("🔬 Testing Prime Discovery...")
        
        test_corpus = [
            "I think you know the truth about this",
            "The people want to feel good about this",
            "All students read many books",
            "Most people think this is very good",
            "Some students feel bad about the test"
        ]
        
        input_data = {
            "test_corpus": test_corpus,
            "run_weekly": True
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/discovery",
                json=input_data,
                headers={"Content-Type": "application/json"}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                output_data = response.json()
                discovery_results = output_data.get("discovery_results", {})
                
                new_candidates = discovery_results.get("new_candidates", 0)
                accepted = discovery_results.get("accepted", 0)
                rejected = discovery_results.get("rejected", 0)
                
                success = new_candidates > 0 and (accepted > 0 or rejected > 0)
                
                notes = f"New candidates: {new_candidates}, Accepted: {accepted}, Rejected: {rejected}"
                
                return DemoResult(
                    name="Prime Discovery",
                    success=success,
                    input_data=input_data,
                    output_data=output_data,
                    processing_time=processing_time,
                    notes=notes
                )
            else:
                return DemoResult(
                    name="Prime Discovery",
                    success=False,
                    input_data=input_data,
                    output_data={"error": f"HTTP {response.status_code}"},
                    processing_time=time.time() - start_time,
                    notes=f"Failed with status {response.status_code}"
                )
                
        except Exception as e:
            return DemoResult(
                name="Prime Discovery",
                success=False,
                input_data=input_data,
                output_data={"error": str(e)},
                processing_time=time.time() - start_time,
                notes=f"Exception: {str(e)}"
            )
    
    def test_enhanced_detection(self) -> DemoResult:
        """Test enhanced detection with all components."""
        print("🔍 Testing Enhanced Detection...")
        
        test_cases = [
            {
                "text": "At most half of the students read a lot of books",
                "language": "en",
                "description": "English quantifiers with MWE"
            },
            {
                "text": "La gente piensa que esto es muy bueno",
                "language": "es", 
                "description": "Spanish mental predicates"
            },
            {
                "text": "Les gens pensent que c'est très bon",
                "language": "fr",
                "description": "French evaluators"
            }
        ]
        
        results = []
        start_time = time.time()
        
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/detect",
                    json={
                        "text": test_case["text"],
                        "language": test_case["language"],
                        "methods": ["spacy", "structured", "multilingual", "mwe"],
                        "include_deepnsm": True,
                        "include_mdl": True,
                        "include_temporal": True
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "language": test_case["language"],
                        "description": test_case["description"],
                        "detected_primes": result.get("detected_primes", []),
                        "processing_time": result.get("processing_time", 0),
                        "methods": list(result.get("method_results", {}).keys())
                    })
                
            except Exception as e:
                print(f"Failed to test {test_case}: {e}")
        
        processing_time = time.time() - start_time
        
        success = len(results) == 3
        
        notes = " | ".join([f"{r['language']}: {len(r['detected_primes'])} primes ({r['processing_time']:.3f}s)" for r in results])
        
        return DemoResult(
            name="Enhanced Detection",
            success=success,
            input_data={"test_cases": test_cases},
            output_data={"results": results},
            processing_time=processing_time,
            notes=notes
        )
    
    def run_all_demos(self) -> List[DemoResult]:
        """Run all demo tests."""
        print("🚀 Starting Comprehensive NSM Universal Translator Demo")
        print("=" * 60)
        
        demos = [
            self.test_roundtrip_translation,
            self.test_quantifier_mwe_detection,
            self.test_risk_coverage_router,
            self.test_constraint_ablation,
            self.test_cross_language_exponents,
            self.test_prime_discovery,
            self.test_enhanced_detection
        ]
        
        results = []
        
        for demo in demos:
            result = demo()
            results.append(result)
            
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} {result.name}")
            print(f"   Time: {result.processing_time:.3f}s")
            print(f"   Notes: {result.notes}")
            print()
        
        # Summary
        passed = sum(1 for r in results if r.success)
        total = len(results)
        total_time = sum(r.processing_time for r in results)
        
        print("=" * 60)
        print(f"🎯 Demo Summary: {passed}/{total} tests passed")
        print(f"⏱️  Total time: {total_time:.3f}s")
        print(f"📊 Success rate: {passed/total*100:.1f}%")
        
        return results

def main():
    """Run the demo server."""
    demo_server = NSMDemoServer()
    results = demo_server.run_all_demos()
    
    # Save results to file
    with open("demo_results.json", "w") as f:
        json.dump([{
            "name": r.name,
            "success": r.success,
            "processing_time": r.processing_time,
            "notes": r.notes,
            "input_data": r.input_data,
            "output_data": r.output_data
        } for r in results], f, indent=2)
    
    print(f"\n📄 Results saved to demo_results.json")

if __name__ == "__main__":
    main()

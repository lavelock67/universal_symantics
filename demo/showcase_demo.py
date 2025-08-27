#!/usr/bin/env python3
"""
NSM Research Platform - Comprehensive Showcase Demo
===================================================

This demo showcases the unique capabilities of our NSM research platform,
demonstrating how it goes beyond simple translation to provide deep semantic
understanding through Natural Semantic Metalanguage analysis.

Features demonstrated:
1. Semantic Prime Discovery
2. Cross-lingual Semantic Alignment  
3. Multi-Word Expression Detection
4. NSM-based Text Generation
5. Semantic Compression Analysis
6. Real-time Linguistic Analysis

Author: NSM Research Team
Date: August 2025
"""

import requests
import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class DemoExample:
    """Represents a demo example with expected outcomes."""
    text: str
    language: str
    expected_primes: List[str]
    expected_mwes: List[str]
    description: str
    category: str


class NSMShowcaseDemo:
    """Comprehensive showcase demo for NSM capabilities."""
    
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url
        self.examples = self._load_demo_examples()
        
    def _load_demo_examples(self) -> List[DemoExample]:
        """Load curated examples that showcase NSM capabilities."""
        return [
            # Category 1: Cross-lingual Semantic Universals
            DemoExample(
                text="People think this is very good",
                language="en",
                expected_primes=["PEOPLE", "THINK", "THIS", "VERY", "GOOD"],
                expected_mwes=[],
                description="Basic semantic universals across languages",
                category="semantic_universals"
            ),
            DemoExample(
                text="La gente piensa que esto es muy bueno",
                language="es", 
                expected_primes=["PEOPLE", "THINK", "THIS", "VERY", "GOOD"],
                expected_mwes=[],
                description="Same semantic content in Spanish",
                category="semantic_universals"
            ),
            
            # Category 2: Complex MWE Detection
            DemoExample(
                text="At least half of the students read a lot of books",
                language="en",
                expected_primes=["PEOPLE", "READ", "MANY"],
                expected_mwes=["at least", "half of", "a lot of"],
                description="Complex quantifier expressions",
                category="mwe_detection"
            ),
            
            # Category 3: Semantic Composition
            DemoExample(
                text="I don't want more people to feel bad about this",
                language="en",
                expected_primes=["I", "NOT", "WANT", "MORE", "PEOPLE", "FEEL", "BAD", "THIS"],
                expected_mwes=["don't", "feel bad", "about this"],
                description="Negation and emotional predicates",
                category="semantic_composition"
            ),
            
            # Category 4: Cognitive Predicates
            DemoExample(
                text="Someone knows that many people think the same thing",
                language="en",
                expected_primes=["SOMEONE", "KNOW", "MANY", "PEOPLE", "THINK", "SAME", "THING"],
                expected_mwes=["the same"],
                description="Episodic and mental predicates",
                category="cognitive_predicates"
            ),
            
            # Category 5: Cross-lingual Complexity
            DemoExample(
                text="Beaucoup de gens pensent que c'est très bien",
                language="fr",
                expected_primes=["MANY", "PEOPLE", "THINK", "THIS", "VERY", "GOOD"],
                expected_mwes=["beaucoup de", "c'est"],
                description="French semantic decomposition",
                category="cross_lingual"
            )
        ]
    
    def run_comprehensive_showcase(self):
        """Run the complete showcase demonstration."""
        print("🌟" * 60)
        print("🌟" + " " * 18 + "NSM RESEARCH PLATFORM SHOWCASE" + " " * 18 + "🌟")
        print("🌟" * 60)
        print()
        print("🔬 Demonstrating breakthrough capabilities in semantic analysis")
        print("📊 Using Natural Semantic Metalanguage for deep language understanding")
        print("🌐 Cross-lingual semantic universals discovery")
        print()
        
        # Check API availability
        if not self._check_api_health():
            print("❌ API not available. Please start the API server first.")
            return
            
        print("✅ API server is healthy and ready")
        print()
        
        # Run showcase by category
        categories = {
            "semantic_universals": "🌐 Cross-lingual Semantic Universals",
            "mwe_detection": "🔍 Multi-Word Expression Detection", 
            "semantic_composition": "🧩 Complex Semantic Composition",
            "cognitive_predicates": "🧠 Cognitive & Mental Predicates",
            "cross_lingual": "🌍 Cross-lingual Analysis"
        }
        
        for category, title in categories.items():
            self._run_category_showcase(category, title)
            
        # Summary and implications
        self._show_research_implications()
        
    def _check_api_health(self) -> bool:
        """Check if the API is responding."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def _run_category_showcase(self, category: str, title: str):
        """Run showcase for a specific category."""
        print("=" * 80)
        print(f"{title}")
        print("=" * 80)
        print()
        
        examples = [ex for ex in self.examples if ex.category == category]
        
        for i, example in enumerate(examples, 1):
            print(f"📝 Example {i}: {example.description}")
            print(f"   Input: \"{example.text}\" [{example.language}]")
            print()
            
            # Detect primes
            primes_result = self._detect_primes(example.text, example.language)
            self._display_primes_analysis(primes_result, example.expected_primes)
            
            # Detect MWEs  
            mwes_result = self._detect_mwes(example.text, example.language)
            self._display_mwes_analysis(mwes_result, example.expected_mwes)
            
            # Generate semantic explication
            if primes_result.get("detected_primes"):
                self._generate_semantic_explication(primes_result["detected_primes"], example.language)
                
            print()
            time.sleep(1)  # Pause for readability
            
    def _detect_primes(self, text: str, language: str) -> Dict[str, Any]:
        """Detect NSM primes in text."""
        try:
            response = requests.post(
                f"{self.api_base_url}/detect",
                json={
                    "text": text,
                    "language": language,
                    "methods": ["spacy", "structured", "multilingual", "mwe"],
                    "include_deepnsm": True,
                    "include_mdl": True
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "primes" in result["result"]:
                    # New API format
                    primes_data = result["result"]["primes"]
                    detected_primes = [prime["text"].upper() for prime in primes_data]
                    confidence = result["result"].get("confidence", 0.0)
                    processing_time = result["result"].get("processing_time", 0.0)
                else:
                    # Old API format
                    detected_primes = result.get("detected_primes", [])
                    confidence = result.get("confidence", 0.0)
                    processing_time = result.get("processing_time", 0.0)
                    
                return {
                    "detected_primes": detected_primes,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "success": True
                }
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def _detect_mwes(self, text: str, language: str) -> Dict[str, Any]:
        """Detect multi-word expressions."""
        try:
            response = requests.post(
                f"{self.api_base_url}/mwe",
                json={
                    "text": text,
                    "language": language
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "mwes" in result["result"]:
                    mwes = result["result"]["mwes"]
                elif "detected_mwes" in result:
                    mwes = result.get("detected_mwes", [])
                else:
                    mwes = result.get("mwes", [])
                    
                return {
                    "detected_mwes": [mwe["text"] if isinstance(mwe, dict) else mwe for mwe in mwes],
                    "success": True
                }
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def _generate_semantic_explication(self, primes: List[str], language: str):
        """Generate semantic explication from primes.""" 
        try:
            # Convert language to enum format
            lang_mapping = {"en": "en", "es": "es", "fr": "fr"}
            target_lang = lang_mapping.get(language, "en")
            
            response = requests.post(
                f"{self.api_base_url}/generate",
                json={
                    "primes": primes,
                    "target_language": target_lang
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    generated_text = result["result"]["generated_text"]
                    confidence = result["result"].get("confidence", 0.0)
                    print(f"   🎯 Generated Explication: \"{generated_text}\"")
                    print(f"   📊 Generation Confidence: {confidence:.2f}")
                else:
                    print(f"   🎯 Generated: {result.get('generated_text', 'No text generated')}")
            else:
                print(f"   ⚠️  Generation failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️  Generation error: {str(e)}")
            
    def _display_primes_analysis(self, result: Dict[str, Any], expected: List[str]):
        """Display primes analysis with comparison."""
        if result.get("success"):
            detected = result["detected_primes"]
            confidence = result.get("confidence", 0.0)
            processing_time = result.get("processing_time", 0.0)
            
            print(f"   🔍 Detected Primes: {detected}")
            print(f"   📈 Expected Primes: {expected}")
            
            # Calculate accuracy
            detected_set = set(detected)
            expected_set = set(expected)
            
            correct = len(detected_set.intersection(expected_set))
            total_expected = len(expected_set)
            precision = correct / len(detected_set) if detected_set else 0
            recall = correct / total_expected if total_expected else 0
            
            print(f"   📊 Accuracy: {correct}/{total_expected} ({recall:.1%} recall, {precision:.1%} precision)")
            print(f"   ⚡ Processing Time: {processing_time:.3f}s")
            print(f"   🎯 Confidence: {confidence:.2f}")
            
            # Highlight missing/extra primes
            missing = expected_set - detected_set
            extra = detected_set - expected_set
            
            if missing:
                print(f"   ⚠️  Missing: {list(missing)}")
            if extra:
                print(f"   ➕ Extra: {list(extra)}")
                
        else:
            print(f"   ❌ Detection failed: {result.get('error', 'Unknown error')}")
            
    def _display_mwes_analysis(self, result: Dict[str, Any], expected: List[str]):
        """Display MWE analysis with comparison."""
        if result.get("success"):
            detected = result["detected_mwes"]
            
            print(f"   🔍 Detected MWEs: {detected}")
            print(f"   📈 Expected MWEs: {expected}")
            
            # Calculate accuracy
            detected_set = set(detected)
            expected_set = set(expected)
            
            correct = len(detected_set.intersection(expected_set))
            total_expected = len(expected_set)
            precision = correct / len(detected_set) if detected_set else 0
            recall = correct / total_expected if total_expected else 0
            
            if total_expected > 0:
                print(f"   📊 MWE Accuracy: {correct}/{total_expected} ({recall:.1%} recall, {precision:.1%} precision)")
            else:
                print(f"   📊 MWE Detection: {len(detected)} expressions found")
                
        else:
            print(f"   ❌ MWE detection failed: {result.get('error', 'Unknown error')}")
            
    def _show_research_implications(self):
        """Show the research and practical implications."""
        print("=" * 80)
        print("🎓 RESEARCH IMPLICATIONS & BREAKTHROUGH CAPABILITIES")
        print("=" * 80)
        print()
        
        implications = [
            "🌐 Cross-lingual Semantic Universals",
            "   • Demonstrates that core meanings transcend language boundaries",
            "   • NSM primes provide universal building blocks for all human languages",
            "   • Enables true semantic comparison across language families",
            "",
            "🔬 Advanced Linguistic Analysis",
            "   • Goes beyond surface syntax to deep semantic structure", 
            "   • Identifies multi-word expressions that express single concepts",
            "   • Maps complex expressions to simple, universal components",
            "",
            "🧠 Cognitive Modeling",
            "   • Models how humans actually think about meaning",
            "   • Captures mental predicates (THINK, KNOW, WANT, FEEL)",
            "   • Represents emotional and evaluative content",
            "",
            "🚀 Practical Applications",
            "   • Machine translation with semantic preservation",
            "   • Cross-cultural communication tools",
            "   • Language learning through universal concepts",
            "   • Automated content analysis and generation",
            "",
            "📊 Technical Achievements",
            "   • Real-time semantic decomposition",
            "   • Multi-language support (English, Spanish, French)",
            "   • High-accuracy prime and MWE detection",
            "   • Scalable architecture for large corpora"
        ]
        
        for line in implications:
            print(line)
            time.sleep(0.1)
            
        print()
        print("🎯 This platform represents a major breakthrough in computational linguistics,")
        print("   bridging the gap between human semantic intuition and machine processing.")
        print()


def main():
    """Run the comprehensive NSM showcase demo."""
    demo = NSMShowcaseDemo()
    demo.run_comprehensive_showcase()


if __name__ == "__main__":
    main()

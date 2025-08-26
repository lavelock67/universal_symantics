#!/usr/bin/env python3
"""
Test script for Advanced Prime Detection System

This script tests the new advanced prime detection capabilities:
- Neural semantic similarity
- Distributional semantics
- Cross-lingual validation
- Prime candidate discovery
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from detect.advanced_prime_detector import AdvancedPrimeDetector, PrimeDiscoveryPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_advanced_prime_detection():
    """Test the advanced prime detection system."""
    print("🧪 Testing Advanced Prime Detection System")
    print("=" * 50)
    
    try:
        # Initialize detector
        print("📥 Initializing Advanced Prime Detector...")
        detector = AdvancedPrimeDetector()
        print("✅ Detector initialized successfully")
        
        # Test corpus
        test_corpus = [
            "La gente piensa que esto es muy bueno",
            "People think this is very good",
            "Les gens pensent que c'est très bon",
            "I want to know what you think about this",
            "Quiero saber qué piensas sobre esto",
            "Je veux savoir ce que tu penses de cela",
            "Some people believe that all things are connected",
            "Algunas personas creen que todas las cosas están conectadas",
            "Certaines personnes croient que toutes les choses sont connectées",
            "The truth is that we must understand each other",
            "La verdad es que debemos entendernos mutuamente",
            "La vérité est que nous devons nous comprendre mutuellement"
        ]
        
        print(f"\n📊 Testing with {len(test_corpus)} multilingual sentences...")
        
        # Test candidate extraction
        print("\n🔍 Extracting prime candidates...")
        candidates = detector.extract_candidates_from_corpus(test_corpus, language="en")
        
        print(f"✅ Found {len(candidates)} candidates")
        
        # Display top candidates
        print("\n🏆 Top 10 Prime Candidates:")
        print("-" * 40)
        for i, candidate in enumerate(candidates[:10]):
            print(f"{i+1:2d}. {candidate.surface_form:15s} | "
                  f"Confidence: {candidate.confidence:.3f} | "
                  f"Universality: {candidate.universality_score:.3f} | "
                  f"Cluster: {candidate.semantic_cluster}")
        
        # Test cross-lingual discovery
        print("\n🌍 Testing Cross-Lingual Prime Discovery...")
        
        # Create multilingual corpora
        corpora = {
            "en": [
                "People think this is very good",
                "I want to know what you think",
                "Some people believe that all things are connected",
                "The truth is that we must understand each other"
            ],
            "es": [
                "La gente piensa que esto es muy bueno",
                "Quiero saber qué piensas sobre esto",
                "Algunas personas creen que todas las cosas están conectadas",
                "La verdad es que debemos entendernos mutuamente"
            ],
            "fr": [
                "Les gens pensent que c'est très bon",
                "Je veux savoir ce que tu penses de cela",
                "Certaines personnes croient que toutes les choses sont connectées",
                "La vérité est que nous devons nous comprendre mutuellement"
            ]
        }
        
        # Initialize discovery pipeline
        discovery_pipeline = PrimeDiscoveryPipeline()
        
        # Discover primes
        print("🔬 Running prime discovery pipeline...")
        discovery_result = discovery_pipeline.discover_primes_from_corpora(corpora)
        
        print(f"✅ Discovery completed!")
        print(f"   - Total candidates: {discovery_result.discovery_metrics['total_candidates']}")
        print(f"   - High confidence: {discovery_result.discovery_metrics['high_confidence_candidates']}")
        print(f"   - Universal candidates: {discovery_result.discovery_metrics['universal_candidates']}")
        print(f"   - Semantic clusters: {discovery_result.discovery_metrics['semantic_clusters']}")
        print(f"   - Avg confidence: {discovery_result.discovery_metrics['average_confidence']:.3f}")
        print(f"   - Avg universality: {discovery_result.discovery_metrics['average_universality']:.3f}")
        
        # Display top universal candidates
        print("\n🌟 Top Universal Prime Candidates:")
        print("-" * 50)
        universal_candidates = [c for c in discovery_result.candidates if c.universality_score > 0.7]
        for i, candidate in enumerate(universal_candidates[:10]):
            print(f"{i+1:2d}. {candidate.surface_form:15s} | "
                  f"Language: {candidate.language:2s} | "
                  f"Universality: {candidate.universality_score:.3f} | "
                  f"Confidence: {candidate.confidence:.3f}")
        
        # Display semantic clusters
        print("\n📊 Semantic Clusters:")
        print("-" * 30)
        for cluster_name, cluster_words in discovery_result.clusters.items():
            if len(cluster_words) > 1:  # Only show clusters with multiple words
                print(f"🔸 {cluster_name}: {', '.join(cluster_words[:5])}")
                if len(cluster_words) > 5:
                    print(f"   ... and {len(cluster_words) - 5} more")
        
        print("\n🎉 Advanced Prime Detection Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

def test_neural_generation():
    """Test the neural NSM generation system."""
    print("\n🧪 Testing Neural NSM Generation System")
    print("=" * 50)
    
    try:
        from generate.neural_nsm_generator import NeuralNSMGenerator, NSMGenerationConfig
        
        # Initialize generator
        print("📥 Initializing Neural NSM Generator...")
        config = NSMGenerationConfig(
            model_name="t5-base",
            max_length=64,
            temperature=0.7,
            constraint_mode="soft"
        )
        generator = NeuralNSMGenerator(config)
        print("✅ Generator initialized successfully")
        
        # Test cases
        test_cases = [
            (["PEOPLE", "THINK", "THIS", "VERY", "GOOD"], "en"),
            (["I", "WANT", "KNOW", "WHAT", "YOU", "THINK"], "en"),
            (["SOME", "PEOPLE", "BELIEVE", "ALL", "THINGS", "CONNECTED"], "en"),
        ]
        
        print(f"\n🎯 Testing generation with {len(test_cases)} test cases...")
        
        for i, (primes, language) in enumerate(test_cases):
            print(f"\n📝 Test Case {i+1}: {primes} -> {language}")
            print("-" * 40)
            
            # Generate text
            result = generator.generate_from_primes(primes, language)
            
            print(f"Input primes: {primes}")
            print(f"Generated text: {result.generated_text}")
            print(f"Target primes: {result.target_primes}")
            print(f"Semantic fidelity: {result.semantic_fidelity:.3f}")
            print(f"NSM compliance: {result.nsm_compliance:.3f}")
            print(f"Generation confidence: {result.generation_confidence:.3f}")
            print(f"Generation time: {result.generation_time:.3f}s")
            
            if result.constraint_violations:
                print(f"⚠️  Constraint violations: {result.constraint_violations}")
        
        print("\n🎉 Neural Generation Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Neural generation test failed: {e}")
        logger.error(f"Neural generation test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    print("🚀 Starting Advanced NSM System Tests")
    print("=" * 60)
    
    # Test prime detection
    detection_success = test_advanced_prime_detection()
    
    # Test neural generation
    generation_success = test_neural_generation()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    print(f"Advanced Prime Detection: {'✅ PASS' if detection_success else '❌ FAIL'}")
    print(f"Neural NSM Generation: {'✅ PASS' if generation_success else '❌ FAIL'}")
    
    if detection_success and generation_success:
        print("\n🎉 All tests passed! The advanced NSM system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")

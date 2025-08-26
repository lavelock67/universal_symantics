#!/usr/bin/env python3
"""Test the Prime Discovery Loop."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prime_discovery():
    """Test the prime discovery loop with sample corpora."""
    print("🚀 Testing Prime Discovery Loop")
    print("=" * 50)
    
    from src.discovery.prime_discovery import PrimeDiscoveryLoop
    from src.detect.srl_ud_detectors import ALL_NSM_PRIMES
    
    # Initialize with existing primes
    existing_primes = set(ALL_NSM_PRIMES)
    discovery = PrimeDiscoveryLoop(existing_primes)
    
    # Sample corpora for testing
    test_corpus = [
        "I think you know the truth about this situation",
        "La gente piensa que esto es muy bueno para todos",
        "Les gens pensent que c'est très bon pour tout le monde",
        "At most half of the students read a lot of books",
        "Au plus la moitié des élèves lisent beaucoup de livres",
        "A lo sumo la mitad de los estudiantes leen muchos libros",
        "Please send me the report now",
        "Por favor envíame el reporte ahora",
        "S'il vous plaît envoyez-moi le rapport maintenant",
        "The weather is very cold today",
        "El clima está muy frío hoy",
        "Le temps est très froid aujourd'hui",
        "We need to understand this problem completely",
        "Necesitamos entender este problema completamente",
        "Nous devons comprendre ce problème complètement"
    ]
    
    validation_corpus = [
        "This is a test of the discovery system",
        "Esto es una prueba del sistema de descubrimiento",
        "Ceci est un test du système de découverte",
        "The system works very well",
        "El sistema funciona muy bien",
        "Le système fonctionne très bien"
    ]
    
    languages = ["en", "es", "fr"]
    
    print(f"📊 Discovery Parameters:")
    print(f"  Existing primes: {len(existing_primes)}")
    print(f"  Test corpus size: {len(test_corpus)}")
    print(f"  Validation corpus size: {len(validation_corpus)}")
    print(f"  Languages: {languages}")
    print()
    
    # Step 1: Discover candidates
    print("🔍 Step 1: Discovering Candidates")
    print("-" * 30)
    
    candidates = discovery.discover_candidates(test_corpus, languages)
    
    print(f"Discovered {len(candidates)} candidates")
    
    # Show top candidates
    print("\nTop 10 candidates:")
    for i, candidate in enumerate(candidates[:10], 1):
        print(f"  {i}. {candidate.prime} (freq: {candidate.frequency}, "
              f"compression: {candidate.compression_gain:.3f}, "
              f"drift: {candidate.drift_reduction:.3f}, "
              f"symmetry: {candidate.symmetry_score:.3f}, "
              f"confidence: {candidate.confidence:.3f})")
    
    # Step 2: Evaluate candidates
    print(f"\n🔍 Step 2: Evaluating Candidates")
    print("-" * 30)
    
    result = discovery.evaluate_candidates(test_corpus, validation_corpus)
    
    print(f"Evaluation complete:")
    print(f"  Accepted: {len(result.accepted_candidates)}")
    print(f"  Rejected: {len(result.rejected_candidates)}")
    print(f"  Compression improvement: {result.compression_improvement:.3f}")
    print(f"  Drift improvement: {result.drift_improvement:.3f}")
    print(f"  Processing time: {result.processing_time:.3f}s")
    
    # Show accepted candidates
    if result.accepted_candidates:
        print(f"\n✅ Accepted Candidates:")
        for i, candidate in enumerate(result.accepted_candidates, 1):
            print(f"  {i}. {candidate.prime}")
            print(f"     Surface forms: {candidate.surface_forms}")
            print(f"     Languages: {candidate.languages}")
            print(f"     Frequency: {candidate.frequency}")
            print(f"     Compression gain: {candidate.compression_gain:.3f}")
            print(f"     Drift reduction: {candidate.drift_reduction:.3f}")
            print(f"     Symmetry score: {candidate.symmetry_score:.3f}")
            print(f"     Confidence: {candidate.confidence:.3f}")
            print()
    
    # Show rejected candidates
    if result.rejected_candidates:
        print(f"\n❌ Rejected Candidates (top 5):")
        for i, candidate in enumerate(result.rejected_candidates[:5], 1):
            print(f"  {i}. {candidate.prime} (confidence: {candidate.confidence:.3f})")
    
    # Step 3: Show statistics
    print(f"\n📊 Discovery Statistics")
    print("-" * 30)
    
    stats = discovery.get_statistics()
    
    print(f"Total candidates: {stats['total_candidates']}")
    print(f"Accepted candidates: {stats['accepted_candidates']}")
    print(f"Rejected candidates: {stats['rejected_candidates']}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
    print(f"Average compression gain: {stats['avg_compression_gain']:.3f}")
    print(f"Average drift reduction: {stats['avg_drift_reduction']:.3f}")
    print(f"Average processing time: {stats['avg_processing_time']:.3f}s")
    print(f"Existing primes: {stats['existing_primes']}")
    print(f"Accepted primes: {stats['accepted_primes']}")
    print(f"Rejected primes: {stats['rejected_primes']}")
    
    # Step 4: Save results
    print(f"\n💾 Step 4: Saving Results")
    print("-" * 30)
    
    discovery.save_results("discovery_results.json")
    print("Results saved to discovery_results.json")
    
    # Step 5: Weekly discovery loop simulation
    print(f"\n🔄 Step 5: Weekly Discovery Loop Simulation")
    print("-" * 30)
    
    # Simulate weekly runs
    weekly_results = []
    for week in range(1, 5):
        print(f"\nWeek {week}:")
        
        # Add some new text to corpus
        new_texts = [
            f"Week {week} additional text for discovery",
            f"Texto adicional de la semana {week} para descubrimiento",
            f"Texte supplémentaire de la semaine {week} pour la découverte"
        ]
        
        extended_corpus = test_corpus + new_texts
        
        # Run discovery
        candidates = discovery.discover_candidates(extended_corpus, languages)
        result = discovery.evaluate_candidates(extended_corpus, validation_corpus)
        
        weekly_results.append({
            "week": week,
            "candidates": len(candidates),
            "accepted": len(result.accepted_candidates),
            "rejected": len(result.rejected_candidates),
            "compression_improvement": result.compression_improvement,
            "drift_improvement": result.drift_improvement
        })
        
        print(f"  Candidates: {len(candidates)}")
        print(f"  Accepted: {len(result.accepted_candidates)}")
        print(f"  Rejected: {len(result.rejected_candidates)}")
        print(f"  Compression: {result.compression_improvement:.3f}")
        print(f"  Drift: {result.drift_improvement:.3f}")
    
    # Show weekly trends
    print(f"\n📈 Weekly Trends:")
    for result in weekly_results:
        print(f"  Week {result['week']}: {result['accepted']} accepted, "
              f"compression +{result['compression_improvement']:.3f}, "
              f"drift +{result['drift_improvement']:.3f}")
    
    print(f"\n🎯 Discovery Loop Complete!")
    print(f"Total new primes discovered: {stats['accepted_primes']}")
    print(f"Total candidates evaluated: {stats['total_candidates']}")
    print(f"Success rate: {stats['acceptance_rate']:.1%}")


if __name__ == "__main__":
    test_prime_discovery()

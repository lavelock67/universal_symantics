#!/usr/bin/env python3
"""Simple demo of the Periodic Table of Information Primitives system."""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.table import (
    Primitive, 
    PrimitiveCategory, 
    PrimitiveSignature, 
    PeriodicTable,
    PrimitiveAlgebra
)
from src.specialists import TemporalESNSpecialist, CentralHub
from src.validate import CompressionValidator


def demo_basic_functionality():
    """Demonstrate basic functionality of the system."""
    print("=" * 60)
    print("PERIODIC TABLE OF INFORMATION PRIMITIVES - DEMO")
    print("=" * 60)
    
    # 1. Create a periodic table
    print("\n1. Creating Periodic Table...")
    table = PeriodicTable()
    
    # Add some basic primitives
    primitives = [
        Primitive("IsA", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=2),
                 description="Is-a relationship", examples=["cat is animal"]),
        Primitive("PartOf", PrimitiveCategory.STRUCTURAL, PrimitiveSignature(arity=2),
                 description="Part-whole relationship", examples=["wheel part of car"]),
        Primitive("Before", PrimitiveCategory.TEMPORAL, PrimitiveSignature(arity=2),
                 description="Temporal precedence", examples=["morning before afternoon"]),
        Primitive("Causes", PrimitiveCategory.CAUSAL, PrimitiveSignature(arity=2),
                 description="Causal relationship", examples=["heat causes expansion"]),
        Primitive("Above", PrimitiveCategory.SPATIAL, PrimitiveSignature(arity=2),
                 description="Spatial relationship", examples=["cloud above ground"]),
    ]
    
    for primitive in primitives:
        table.add_primitive(primitive)
    
    print(f"   Created table with {len(table.primitives)} primitives")
    print(f"   Categories: {[cat.value for cat in table.categories]}")
    
    # 2. Demonstrate algebra operations
    print("\n2. Testing Algebra Operations...")
    algebra = PrimitiveAlgebra(table)
    
    # Test factorization
    text = "The cat is on the mat"
    factors = algebra.factor(text)
    print(f"   Text: '{text}'")
    print(f"   Factors: {[f.name for f in factors]}")
    
    # Test difference operator
    prior = "The cat is on the mat"
    new = "The dog is in the house"
    differences = algebra.delta(prior, new)
    print(f"   Prior: '{prior}'")
    print(f"   New: '{new}'")
    print(f"   Differences: {[d.name for d in differences]}")
    
    # 3. Test temporal ESN specialist
    print("\n3. Testing Temporal ESN Specialist...")
    temporal_specialist = TemporalESNSpecialist(reservoir_size=128, n_esns=2)
    
    # Process a sequence
    sequence = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = temporal_specialist.process_sequence(sequence)
    print(f"   Sequence: {sequence}")
    print(f"   Sequence length: {result['sequence_length']}")
    print(f"   Memory buffer size: {result['memory_buffer_size']}")
    
    # Test temporal coherence
    coherence = temporal_specialist.compute_temporal_coherence(sequence)
    print(f"   Temporal coherence: {coherence:.3f}")
    
    # 4. Test central integration hub
    print("\n4. Testing Central Integration Hub...")
    central_hub = CentralHub(table, temporal_specialist)
    
    # Create multi-modal signals (realistic, less correlated)
    signals = {
        "vision": np.array([0.85, 0.12, 0.73, 0.28, 0.91]),      # Visual features
        "audio": np.array([0.42, 0.67, 0.19, 0.83, 0.35]),       # Audio features  
        "text": np.array([0.58, 0.31, 0.76, 0.44, 0.62]),        # Text features
    }
    
    print("   WARNING: This demo uses placeholder implementations that may produce")
    print("   misleading results. Real performance requires:")
    print("   - Learned primitive embeddings")
    print("   - Proper signal-to-primitive mapping")
    print("   - Domain-specific factorization algorithms")
    print()
    
    # Integrate signals
    result = central_hub.integrate(signals)
    print(f"   Integration score: {result['integration_score']:.3f}")
    print(f"   Phi score: {result['phi_score']:.3f}")
    print(f"   Attention weights: {result['attention_weights']}")
    print(f"   Conflicts detected: {len(result['conflicts'])}")
    
    # 5. Test compression validation
    print("\n5. Testing Compression Validation...")
    validator = CompressionValidator(table)
    
    print("   WARNING: Compression results are theatrical due to placeholder factorization.")
    print("   Real compression requires learned primitive embeddings and proper algorithms.")
    print()
    
    # Test compression on different domains
    domains = ["text", "vision", "logic"]
    results = validator.test_multiple_domains(domains, n_samples=20)
    
    print("   Compression Results (PLACEHOLDER - NOT REAL):")
    for domain, result in results.items():
        if "error" not in result:
            ratio = result.get("avg_compression_ratio", 0.0)
            print(f"     {domain}: {ratio:.3f}x compression (placeholder)")
        else:
            print(f"     {domain}: Error - {result['error']}")
    
    # 6. Show system metrics
    print("\n6. System Metrics...")
    metrics = central_hub.get_integration_metrics()
    print(f"   Current phi: {metrics['current_phi']:.3f}")
    print(f"   Current integration: {metrics['current_integration']:.3f}")
    print(f"   Integration history length: {len(metrics['integration_history'])}")
    print(f"   Phi history length: {len(metrics['phi_history'])}")
    print(f"   Cross-modal connections: {len(metrics['cross_modal_connections'])}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)


def demo_mining_pipeline():
    """Demonstrate the mining pipeline (without actual model loading)."""
    print("\n" + "=" * 60)
    print("MINING PIPELINE DEMO")
    print("=" * 60)
    
    print("\nThis would demonstrate:")
    print("1. Embedding intersection across multiple models")
    print("2. ConceptNet multilingual relation mining")
    print("3. Vision filter clustering")
    print("4. Logic/math pattern extraction")
    print("5. Primitive discovery and validation")
    
    print("\nNote: Full mining requires downloading models and data.")
    print("See README.md for instructions on running the complete pipeline.")


if __name__ == "__main__":
    try:
        demo_basic_functionality()
        demo_mining_pipeline()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python -m src.mining.embedding_miner' to mine primitives from embeddings")
        print("2. Run 'python -m src.mining.conceptnet_miner' to mine from ConceptNet")
        print("3. Run 'python -m src.ui.demo_api' to start the web API")
        print("4. Run 'python -m src.validate.compression' to validate compression")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

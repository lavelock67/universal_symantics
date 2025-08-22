#!/usr/bin/env python3
"""
Create test data for primitive detection system evaluation.

Generates parallel test data and gold labels for comprehensive evaluation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

# Test data for evaluation
TEST_SENTENCES = [
    {
        "en": "The cat is on the mat",
        "es": "El gato está en la alfombra",
        "fr": "Le chat est sur le tapis",
        "primitives": ["AtLocation", "Entity"]
    },
    {
        "en": "This is similar to that",
        "es": "Esto es similar a eso",
        "fr": "Ceci est similaire à cela",
        "primitives": ["SimilarTo", "Entity"]
    },
    {
        "en": "The tool is used for cutting",
        "es": "La herramienta se usa para cortar",
        "fr": "L'outil est utilisé pour couper",
        "primitives": ["UsedFor", "Entity", "Action"]
    },
    {
        "en": "The red car is fast",
        "es": "El coche rojo es rápido",
        "fr": "La voiture rouge est rapide",
        "primitives": ["HasProperty", "Entity"]
    },
    {
        "en": "This is different from that",
        "es": "Esto es diferente de eso",
        "fr": "Ceci est différent de cela",
        "primitives": ["DifferentFrom", "Entity"]
    },
    {
        "en": "I see you",
        "es": "Yo te veo",
        "fr": "Je te vois",
        "primitives": ["I", "you", "see"]
    },
    {
        "en": "This is good",
        "es": "Esto es bueno",
        "fr": "Ceci est bon",
        "primitives": ["this", "is", "good"]
    },
    {
        "en": "I want this",
        "es": "Yo quiero esto",
        "fr": "Je veux ceci",
        "primitives": ["I", "want", "this"]
    },
    {
        "en": "The dog runs quickly",
        "es": "El perro corre rápidamente",
        "fr": "Le chien court rapidement",
        "primitives": ["Entity", "Action", "Property"]
    },
    {
        "en": "The book is in the bag",
        "es": "El libro está en la bolsa",
        "fr": "Le livre est dans le sac",
        "primitives": ["AtLocation", "Entity"]
    }
]

def create_parallel_test_data():
    """Create parallel test data file."""
    data = {
        "metadata": {
            "description": "Parallel test data for primitive detection evaluation",
            "languages": ["en", "es", "fr"],
            "num_sentences": len(TEST_SENTENCES),
            "created": "2024-08-21"
        },
        "data": TEST_SENTENCES
    }
    
    output_path = Path("data/parallel_test_data.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Parallel test data created: {output_path}")
    return output_path

def create_gold_labels():
    """Create gold labels file."""
    gold_data = {
        "metadata": {
            "description": "Gold labels for primitive detection evaluation",
            "num_sentences": len(TEST_SENTENCES),
            "created": "2024-08-21"
        },
        "labels": [item["primitives"] for item in TEST_SENTENCES]
    }
    
    output_path = Path("data/parallel_gold.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gold_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Gold labels created: {output_path}")
    return output_path

def main():
    """Create test data files."""
    print("Creating test data for primitive detection evaluation...")
    
    # Create parallel test data
    parallel_path = create_parallel_test_data()
    
    # Create gold labels
    gold_path = create_gold_labels()
    
    print(f"\n📊 Test Data Summary:")
    print(f"   Parallel sentences: {len(TEST_SENTENCES)}")
    print(f"   Languages: EN, ES, FR")
    print(f"   Primitive types: {len(set(prim for item in TEST_SENTENCES for prim in item['primitives']))}")
    
    print(f"\n✅ Test data creation complete!")
    print(f"   Files created: {parallel_path}, {gold_path}")

if __name__ == "__main__":
    main()

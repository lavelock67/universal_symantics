#!/usr/bin/env python3
"""
Create Parallel Test Data for Honest Cross-Language Evaluation

This script creates proper parallel corpora for testing primitive universality
across languages, ensuring we test on the same content in different languages.
"""

import json
from pathlib import Path

def create_parallel_test_data():
    """Create parallel test data for honest evaluation."""
    
    # Simple parallel sentences that can be used to test primitive detection
    # These are basic, common sentences that should contain universal primitives
    parallel_data = {
        "en": [
            "The cat is on the table",
            "Water is essential for life",
            "The book is part of the library",
            "The sun causes heat",
            "The car has four wheels",
            "A knife is used for cutting",
            "The bird is similar to a plane",
            "The tree is different from a flower",
            "The student does not understand",
            "The problem exists in the system"
        ],
        "es": [
            "El gato está en la mesa",
            "El agua es esencial para la vida",
            "El libro es parte de la biblioteca",
            "El sol causa calor",
            "El coche tiene cuatro ruedas",
            "Un cuchillo se usa para cortar",
            "El pájaro es similar a un avión",
            "El árbol es diferente de una flor",
            "El estudiante no entiende",
            "El problema existe en el sistema"
        ],
        "fr": [
            "Le chat est sur la table",
            "L'eau est essentielle pour la vie",
            "Le livre fait partie de la bibliothèque",
            "Le soleil cause la chaleur",
            "La voiture a quatre roues",
            "Un couteau est utilisé pour couper",
            "L'oiseau est similaire à un avion",
            "L'arbre est différent d'une fleur",
            "L'étudiant ne comprend pas",
            "Le problème existe dans le système"
        ]
    }
    
    # Save parallel data
    output_file = Path("data/parallel_test_data.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parallel_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Created parallel test data with {len(parallel_data['en'])} sentence pairs")
    print(f"  Saved to: {output_file}")
    
    # Also create individual language files for compatibility
    for lang, sentences in parallel_data.items():
        lang_file = Path(f"data/parallel_{lang}_test.txt")
        with open(lang_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        print(f"  Created: {lang_file}")

if __name__ == "__main__":
    create_parallel_test_data()

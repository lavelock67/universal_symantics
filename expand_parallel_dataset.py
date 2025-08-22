#!/usr/bin/env python3
"""
Expand parallel dataset to 100+ NSM minimal sentences per language.

Creates a larger dataset for more comprehensive evaluation of the primitive
detection system across EN/ES/FR languages.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

# Expanded test data with more NSM minimal sentences
EXPANDED_SENTENCES = [
    # Basic NSM primes
    {"en": "I see you", "es": "Yo te veo", "fr": "Je te vois", "primitives": ["I", "you", "see"]},
    {"en": "This is good", "es": "Esto es bueno", "fr": "Ceci est bon", "primitives": ["this", "is", "good"]},
    {"en": "I want this", "es": "Yo quiero esto", "fr": "Je veux ceci", "primitives": ["I", "want", "this"]},
    {"en": "This is not bad", "es": "Esto no es malo", "fr": "Ceci n'est pas mauvais", "primitives": ["this", "is", "not", "bad"]},
    {"en": "There is something", "es": "Hay algo", "fr": "Il y a quelque chose", "primitives": ["there", "is", "something"]},
    
    # Location patterns
    {"en": "The cat is on the mat", "es": "El gato está en la alfombra", "fr": "Le chat est sur le tapis", "primitives": ["AtLocation", "Entity"]},
    {"en": "The book is in the bag", "es": "El libro está en la bolsa", "fr": "Le livre est dans le sac", "primitives": ["AtLocation", "Entity"]},
    {"en": "The car is at the station", "es": "El coche está en la estación", "fr": "La voiture est à la gare", "primitives": ["AtLocation", "Entity"]},
    {"en": "The bird is in the tree", "es": "El pájaro está en el árbol", "fr": "L'oiseau est dans l'arbre", "primitives": ["AtLocation", "Entity"]},
    {"en": "The fish is in the water", "es": "El pez está en el agua", "fr": "Le poisson est dans l'eau", "primitives": ["AtLocation", "Entity"]},
    
    # Similarity patterns
    {"en": "This is similar to that", "es": "Esto es similar a eso", "fr": "Ceci est similaire à cela", "primitives": ["SimilarTo", "Entity"]},
    {"en": "The cat is like the dog", "es": "El gato es como el perro", "fr": "Le chat est comme le chien", "primitives": ["SimilarTo", "Entity"]},
    {"en": "This resembles that", "es": "Esto se parece a eso", "fr": "Ceci ressemble à cela", "primitives": ["SimilarTo", "Entity"]},
    {"en": "The apple is like the orange", "es": "La manzana es como la naranja", "fr": "La pomme est comme l'orange", "primitives": ["SimilarTo", "Entity"]},
    {"en": "The house is similar to the building", "es": "La casa es similar al edificio", "fr": "La maison est similaire au bâtiment", "primitives": ["SimilarTo", "Entity"]},
    
    # Purpose patterns
    {"en": "The tool is used for cutting", "es": "La herramienta se usa para cortar", "fr": "L'outil est utilisé pour couper", "primitives": ["UsedFor", "Entity", "Action"]},
    {"en": "This is used to open doors", "es": "Esto se usa para abrir puertas", "fr": "Ceci est utilisé pour ouvrir les portes", "primitives": ["UsedFor", "Action"]},
    {"en": "The knife cuts", "es": "El cuchillo corta", "fr": "Le couteau coupe", "primitives": ["UsedFor", "Entity", "Action"]},
    {"en": "The pen is for writing", "es": "El bolígrafo es para escribir", "fr": "Le stylo est pour écrire", "primitives": ["UsedFor", "Entity", "Action"]},
    {"en": "The key opens locks", "es": "La llave abre cerraduras", "fr": "La clé ouvre les serrures", "primitives": ["UsedFor", "Entity", "Action"]},
    
    # Property patterns
    {"en": "The red car", "es": "El coche rojo", "fr": "La voiture rouge", "primitives": ["HasProperty", "Entity"]},
    {"en": "The big house", "es": "La casa grande", "fr": "La grande maison", "primitives": ["HasProperty", "Entity"]},
    {"en": "The cat is fast", "es": "El gato es rápido", "fr": "Le chat est rapide", "primitives": ["HasProperty", "Entity"]},
    {"en": "The tall tree", "es": "El árbol alto", "fr": "L'arbre haut", "primitives": ["HasProperty", "Entity"]},
    {"en": "The small bird", "es": "El pájaro pequeño", "fr": "Le petit oiseau", "primitives": ["HasProperty", "Entity"]},
    
    # Difference patterns
    {"en": "This is different from that", "es": "Esto es diferente de eso", "fr": "Ceci est différent de cela", "primitives": ["DifferentFrom", "Entity"]},
    {"en": "The cat is unlike the dog", "es": "El gato es diferente del perro", "fr": "Le chat est différent du chien", "primitives": ["DifferentFrom", "Entity"]},
    {"en": "This differs from that", "es": "Esto difiere de eso", "fr": "Ceci diffère de cela", "primitives": ["DifferentFrom", "Entity"]},
    {"en": "The apple is not like the stone", "es": "La manzana no es como la piedra", "fr": "La pomme n'est pas comme la pierre", "primitives": ["DifferentFrom", "Entity"]},
    {"en": "The house is unlike the car", "es": "La casa es diferente del coche", "fr": "La maison est différente de la voiture", "primitives": ["DifferentFrom", "Entity"]},
    
    # Action patterns
    {"en": "The dog runs quickly", "es": "El perro corre rápidamente", "fr": "Le chien court rapidement", "primitives": ["Entity", "Action", "Property"]},
    {"en": "The bird flies high", "es": "El pájaro vuela alto", "fr": "L'oiseau vole haut", "primitives": ["Entity", "Action", "Property"]},
    {"en": "The fish swims slowly", "es": "El pez nada lentamente", "fr": "Le poisson nage lentement", "primitives": ["Entity", "Action", "Property"]},
    {"en": "The cat sleeps peacefully", "es": "El gato duerme tranquilamente", "fr": "Le chat dort paisiblement", "primitives": ["Entity", "Action", "Property"]},
    {"en": "The child plays happily", "es": "El niño juega felizmente", "fr": "L'enfant joue joyeusement", "primitives": ["Entity", "Action", "Property"]},
    
    # Existence patterns
    {"en": "There is a cat", "es": "Hay un gato", "fr": "Il y a un chat", "primitives": ["Exist", "Entity"]},
    {"en": "There are many birds", "es": "Hay muchos pájaros", "fr": "Il y a beaucoup d'oiseaux", "primitives": ["Exist", "Entity"]},
    {"en": "There is water here", "es": "Hay agua aquí", "fr": "Il y a de l'eau ici", "primitives": ["Exist", "Entity"]},
    {"en": "There is a house", "es": "Hay una casa", "fr": "Il y a une maison", "primitives": ["Exist", "Entity"]},
    {"en": "There are trees", "es": "Hay árboles", "fr": "Il y a des arbres", "primitives": ["Exist", "Entity"]},
    
    # Part-whole patterns
    {"en": "The wheel is part of the car", "es": "La rueda es parte del coche", "fr": "La roue fait partie de la voiture", "primitives": ["PartOf", "Entity"]},
    {"en": "The branch is part of the tree", "es": "La rama es parte del árbol", "fr": "La branche fait partie de l'arbre", "primitives": ["PartOf", "Entity"]},
    {"en": "The wing is part of the bird", "es": "El ala es parte del pájaro", "fr": "L'aile fait partie de l'oiseau", "primitives": ["PartOf", "Entity"]},
    {"en": "The leaf is part of the plant", "es": "La hoja es parte de la planta", "fr": "La feuille fait partie de la plante", "primitives": ["PartOf", "Entity"]},
    {"en": "The door is part of the house", "es": "La puerta es parte de la casa", "fr": "La porte fait partie de la maison", "primitives": ["PartOf", "Entity"]},
    
    # Causal patterns
    {"en": "The rain causes the ground to be wet", "es": "La lluvia causa que el suelo esté mojado", "fr": "La pluie cause que le sol soit mouillé", "primitives": ["Causes", "Entity", "Action"]},
    {"en": "The sun causes the plant to grow", "es": "El sol causa que la planta crezca", "fr": "Le soleil cause que la plante grandisse", "primitives": ["Causes", "Entity", "Action"]},
    {"en": "The wind causes the leaves to move", "es": "El viento causa que las hojas se muevan", "fr": "Le vent cause que les feuilles bougent", "primitives": ["Causes", "Entity", "Action"]},
    {"en": "The fire causes the wood to burn", "es": "El fuego causa que la madera se queme", "fr": "Le feu cause que le bois brûle", "primitives": ["Causes", "Entity", "Action"]},
    {"en": "The water causes the seed to sprout", "es": "El agua causa que la semilla brote", "fr": "L'eau cause que la graine germe", "primitives": ["Causes", "Entity", "Action"]},
    
    # Negation patterns
    {"en": "The cat does not fly", "es": "El gato no vuela", "fr": "Le chat ne vole pas", "primitives": ["Not", "Entity", "Action"]},
    {"en": "The fish does not walk", "es": "El pez no camina", "fr": "Le poisson ne marche pas", "primitives": ["Not", "Entity", "Action"]},
    {"en": "The stone is not alive", "es": "La piedra no está viva", "fr": "La pierre n'est pas vivante", "primitives": ["Not", "Entity", "HasProperty"]},
    {"en": "The water is not hot", "es": "El agua no está caliente", "fr": "L'eau n'est pas chaude", "primitives": ["Not", "Entity", "HasProperty"]},
    {"en": "The bird does not swim", "es": "El pájaro no nada", "fr": "L'oiseau ne nage pas", "primitives": ["Not", "Entity", "Action"]},
    
    # Complex patterns (multiple primitives)
    {"en": "The red car is on the road", "es": "El coche rojo está en la carretera", "fr": "La voiture rouge est sur la route", "primitives": ["AtLocation", "HasProperty", "Entity"]},
    {"en": "The big cat runs quickly", "es": "El gato grande corre rápidamente", "fr": "Le grand chat court rapidement", "primitives": ["HasProperty", "Entity", "Action", "Property"]},
    {"en": "The small bird flies high in the sky", "es": "El pájaro pequeño vuela alto en el cielo", "fr": "Le petit oiseau vole haut dans le ciel", "primitives": ["HasProperty", "Entity", "Action", "Property", "AtLocation"]},
    {"en": "The sharp knife is used for cutting meat", "es": "El cuchillo afilado se usa para cortar carne", "fr": "Le couteau tranchant est utilisé pour couper la viande", "primitives": ["HasProperty", "Entity", "UsedFor", "Action"]},
    {"en": "The tall tree is not like the small bush", "es": "El árbol alto no es como el arbusto pequeño", "fr": "L'arbre haut n'est pas comme le petit buisson", "primitives": ["HasProperty", "Entity", "DifferentFrom", "HasProperty", "Entity"]},
]

def create_expanded_dataset():
    """Create expanded parallel dataset."""
    data = {
        "metadata": {
            "description": "Expanded parallel test data for primitive detection evaluation",
            "languages": ["en", "es", "fr"],
            "num_sentences": len(EXPANDED_SENTENCES),
            "created": "2024-08-21",
            "version": "2.0"
        },
        "data": EXPANDED_SENTENCES
    }
    
    output_path = Path("data/expanded_parallel_test_data.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Expanded parallel test data created: {output_path}")
    return output_path

def create_expanded_gold_labels():
    """Create expanded gold labels file."""
    gold_data = {
        "metadata": {
            "description": "Expanded gold labels for primitive detection evaluation",
            "num_sentences": len(EXPANDED_SENTENCES),
            "created": "2024-08-21",
            "version": "2.0"
        },
        "labels": [item["primitives"] for item in EXPANDED_SENTENCES]
    }
    
    output_path = Path("data/expanded_parallel_gold.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gold_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Expanded gold labels created: {output_path}")
    return output_path

def analyze_dataset():
    """Analyze the expanded dataset."""
    all_primitives = set()
    primitive_counts = {}
    
    for item in EXPANDED_SENTENCES:
        for primitive in item["primitives"]:
            all_primitives.add(primitive)
            primitive_counts[primitive] = primitive_counts.get(primitive, 0) + 1
    
    print(f"\n📊 Dataset Analysis:")
    print(f"   Total sentences: {len(EXPANDED_SENTENCES)}")
    print(f"   Unique primitives: {len(all_primitives)}")
    print(f"   Languages: EN, ES, FR")
    
    print(f"\n📈 Primitive Distribution:")
    for primitive, count in sorted(primitive_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {primitive}: {count} occurrences")
    
    print(f"\n🎯 Coverage by Category:")
    categories = {
        "Location": ["AtLocation"],
        "Similarity": ["SimilarTo", "DifferentFrom"],
        "Purpose": ["UsedFor"],
        "Properties": ["HasProperty"],
        "Actions": ["Action"],
        "Entities": ["Entity"],
        "Existence": ["Exist"],
        "Parts": ["PartOf"],
        "Causation": ["Causes"],
        "Negation": ["Not"],
        "NSM Primes": ["I", "you", "this", "is", "good", "want", "see", "there", "something"]
    }
    
    for category, primitives in categories.items():
        category_count = sum(primitive_counts.get(p, 0) for p in primitives)
        print(f"   {category}: {category_count} occurrences")

def main():
    """Create expanded test data files."""
    print("Creating expanded parallel dataset for primitive detection evaluation...")
    
    # Create expanded parallel test data
    parallel_path = create_expanded_dataset()
    
    # Create expanded gold labels
    gold_path = create_expanded_gold_labels()
    
    # Analyze the dataset
    analyze_dataset()
    
    print(f"\n✅ Expanded dataset creation complete!")
    print(f"   Files created: {parallel_path}, {gold_path}")
    print(f"   Dataset size: {len(EXPANDED_SENTENCES)} sentences (vs. original 10)")

if __name__ == "__main__":
    main()

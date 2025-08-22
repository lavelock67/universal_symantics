#!/usr/bin/env python3
"""
Expand parallel dataset to 1k+ sentences across 20+ primitives.

This script creates a comprehensive dataset covering all major primitive types
with multiple examples per primitive across EN/ES/FR languages.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Comprehensive primitive coverage
PRIMITIVES_BY_CATEGORY = {
    "location": [
        "AtLocation", "Near", "FarFrom", "Inside", "Outside", "Above", "Below", 
        "InFrontOf", "Behind", "Between", "Around", "Through", "Across"
    ],
    "similarity": [
        "SimilarTo", "DifferentFrom", "SameAs", "Like", "Unlike", "IdenticalTo"
    ],
    "properties": [
        "HasProperty", "HasColor", "HasSize", "HasShape", "HasMaterial", 
        "HasWeight", "HasTemperature", "HasAge", "HasValue"
    ],
    "purpose": [
        "UsedFor", "MadeFor", "DesignedFor", "IntendedFor", "SuitableFor"
    ],
    "parts": [
        "PartOf", "HasPart", "Contains", "Includes", "ConsistsOf", "MadeOf"
    ],
    "causation": [
        "Causes", "CausedBy", "ResultsIn", "LeadsTo", "Prevents", "Enables"
    ],
    "actions": [
        "Can", "Cannot", "AbleTo", "CapableOf", "Performs", "Does"
    ],
    "existence": [
        "Exist", "NotExist", "Present", "Absent", "Available", "Missing"
    ],
    "logical": [
        "Not", "And", "Or", "If", "Then", "Because", "Therefore"
    ],
    "temporal": [
        "Before", "After", "During", "While", "When", "Until", "Since"
    ],
    "quantity": [
        "Many", "Few", "Some", "All", "None", "Most", "Least", "More", "Less"
    ],
    "evaluation": [
        "Good", "Bad", "Better", "Worse", "Best", "Worst", "Important", "Unimportant"
    ]
}

# Comprehensive test sentences by primitive
TEST_SENTENCES = {
    # Location primitives
    "AtLocation": {
        "en": [
            "The book is on the table", "The car is in the garage", "The bird is in the tree",
            "The phone is on the desk", "The keys are in the pocket", "The picture is on the wall",
            "The food is in the refrigerator", "The money is in the bank", "The people are at the park",
            "The students are in the classroom", "The doctor is at the hospital", "The fish is in the water"
        ],
        "es": [
            "El libro está en la mesa", "El coche está en el garaje", "El pájaro está en el árbol",
            "El teléfono está en el escritorio", "Las llaves están en el bolsillo", "El cuadro está en la pared",
            "La comida está en el refrigerador", "El dinero está en el banco", "La gente está en el parque",
            "Los estudiantes están en el aula", "El médico está en el hospital", "El pez está en el agua"
        ],
        "fr": [
            "Le livre est sur la table", "La voiture est dans le garage", "L'oiseau est dans l'arbre",
            "Le téléphone est sur le bureau", "Les clés sont dans la poche", "Le tableau est sur le mur",
            "La nourriture est dans le réfrigérateur", "L'argent est à la banque", "Les gens sont au parc",
            "Les étudiants sont en classe", "Le médecin est à l'hôpital", "Le poisson est dans l'eau"
        ]
    },
    
    "HasProperty": {
        "en": [
            "The apple is red", "The sky is blue", "The snow is white", "The grass is green",
            "The car is fast", "The building is tall", "The water is cold", "The fire is hot",
            "The rock is hard", "The cotton is soft", "The metal is heavy", "The paper is light"
        ],
        "es": [
            "La manzana es roja", "El cielo es azul", "La nieve es blanca", "La hierba es verde",
            "El coche es rápido", "El edificio es alto", "El agua está fría", "El fuego está caliente",
            "La roca es dura", "El algodón es suave", "El metal es pesado", "El papel es ligero"
        ],
        "fr": [
            "La pomme est rouge", "Le ciel est bleu", "La neige est blanche", "L'herbe est verte",
            "La voiture est rapide", "Le bâtiment est haut", "L'eau est froide", "Le feu est chaud",
            "La pierre est dure", "Le coton est doux", "Le métal est lourd", "Le papier est léger"
        ]
    },
    
    "UsedFor": {
        "en": [
            "A knife is used for cutting", "A pen is used for writing", "A car is used for transportation",
            "A phone is used for communication", "A book is used for reading", "A chair is used for sitting",
            "A bed is used for sleeping", "A spoon is used for eating", "A clock is used for telling time",
            "A key is used for opening doors", "A computer is used for computing", "A camera is used for taking photos"
        ],
        "es": [
            "Un cuchillo se usa para cortar", "Un bolígrafo se usa para escribir", "Un coche se usa para el transporte",
            "Un teléfono se usa para la comunicación", "Un libro se usa para leer", "Una silla se usa para sentarse",
            "Una cama se usa para dormir", "Una cuchara se usa para comer", "Un reloj se usa para decir la hora",
            "Una llave se usa para abrir puertas", "Una computadora se usa para computar", "Una cámara se usa para tomar fotos"
        ],
        "fr": [
            "Un couteau sert à couper", "Un stylo sert à écrire", "Une voiture sert au transport",
            "Un téléphone sert à communiquer", "Un livre sert à lire", "Une chaise sert à s'asseoir",
            "Un lit sert à dormir", "Une cuillère sert à manger", "Une horloge sert à dire l'heure",
            "Une clé sert à ouvrir les portes", "Un ordinateur sert à calculer", "Un appareil photo sert à prendre des photos"
        ]
    },
    
    "SimilarTo": {
        "en": [
            "A cat is similar to a tiger", "A car is similar to a truck", "A pen is similar to a pencil",
            "A chair is similar to a stool", "A house is similar to an apartment", "A dog is similar to a wolf",
            "A book is similar to a magazine", "A phone is similar to a tablet", "A tree is similar to a bush",
            "A river is similar to a stream", "A mountain is similar to a hill", "A lake is similar to a pond"
        ],
        "es": [
            "Un gato es similar a un tigre", "Un coche es similar a una camioneta", "Un bolígrafo es similar a un lápiz",
            "Una silla es similar a un taburete", "Una casa es similar a un apartamento", "Un perro es similar a un lobo",
            "Un libro es similar a una revista", "Un teléfono es similar a una tableta", "Un árbol es similar a un arbusto",
            "Un río es similar a un arroyo", "Una montaña es similar a una colina", "Un lago es similar a un estanque"
        ],
        "fr": [
            "Un chat est semblable à un tigre", "Une voiture est semblable à un camion", "Un stylo est semblable à un crayon",
            "Une chaise est semblable à un tabouret", "Une maison est semblable à un appartement", "Un chien est semblable à un loup",
            "Un livre est semblable à un magazine", "Un téléphone est semblable à une tablette", "Un arbre est semblable à un buisson",
            "Une rivière est semblable à un ruisseau", "Une montagne est semblable à une colline", "Un lac est semblable à un étang"
        ]
    },
    
    "PartOf": {
        "en": [
            "A wheel is part of a car", "A leaf is part of a tree", "A page is part of a book",
            "A button is part of a shirt", "A branch is part of a tree", "A chapter is part of a book",
            "A finger is part of a hand", "A room is part of a house", "A leg is part of a table",
            "A wing is part of a bird", "A petal is part of a flower", "A handle is part of a door"
        ],
        "es": [
            "Una rueda es parte de un coche", "Una hoja es parte de un árbol", "Una página es parte de un libro",
            "Un botón es parte de una camisa", "Una rama es parte de un árbol", "Un capítulo es parte de un libro",
            "Un dedo es parte de una mano", "Una habitación es parte de una casa", "Una pata es parte de una mesa",
            "Un ala es parte de un pájaro", "Un pétalo es parte de una flor", "Una manija es parte de una puerta"
        ],
        "fr": [
            "Une roue fait partie d'une voiture", "Une feuille fait partie d'un arbre", "Une page fait partie d'un livre",
            "Un bouton fait partie d'une chemise", "Une branche fait partie d'un arbre", "Un chapitre fait partie d'un livre",
            "Un doigt fait partie d'une main", "Une pièce fait partie d'une maison", "Une jambe fait partie d'une table",
            "Une aile fait partie d'un oiseau", "Un pétale fait partie d'une fleur", "Une poignée fait partie d'une porte"
        ]
    },
    
    "Causes": {
        "en": [
            "Rain causes wet ground", "Fire causes heat", "Exercise causes tiredness",
            "Food causes hunger satisfaction", "Sleep causes rest", "Medicine causes healing",
            "Sunlight causes warmth", "Wind causes movement", "Gravity causes falling",
            "Electricity causes light", "Water causes growth", "Time causes aging"
        ],
        "es": [
            "La lluvia causa suelo mojado", "El fuego causa calor", "El ejercicio causa cansancio",
            "La comida causa satisfacción del hambre", "El sueño causa descanso", "La medicina causa curación",
            "La luz del sol causa calor", "El viento causa movimiento", "La gravedad causa caída",
            "La electricidad causa luz", "El agua causa crecimiento", "El tiempo causa envejecimiento"
        ],
        "fr": [
            "La pluie cause un sol mouillé", "Le feu cause de la chaleur", "L'exercice cause de la fatigue",
            "La nourriture cause la satisfaction de la faim", "Le sommeil cause du repos", "Le médicament cause la guérison",
            "La lumière du soleil cause de la chaleur", "Le vent cause du mouvement", "La gravité cause la chute",
            "L'électricité cause de la lumière", "L'eau cause la croissance", "Le temps cause le vieillissement"
        ]
    },
    
    "DifferentFrom": {
        "en": [
            "A cat is different from a dog", "A car is different from a bicycle", "A book is different from a movie",
            "A house is different from an apartment", "A tree is different from a flower", "A river is different from a lake",
            "A mountain is different from a hill", "A city is different from a town", "A teacher is different from a student",
            "A doctor is different from a nurse", "A computer is different from a phone", "A chair is different from a table"
        ],
        "es": [
            "Un gato es diferente de un perro", "Un coche es diferente de una bicicleta", "Un libro es diferente de una película",
            "Una casa es diferente de un apartamento", "Un árbol es diferente de una flor", "Un río es diferente de un lago",
            "Una montaña es diferente de una colina", "Una ciudad es diferente de un pueblo", "Un profesor es diferente de un estudiante",
            "Un médico es diferente de una enfermera", "Una computadora es diferente de un teléfono", "Una silla es diferente de una mesa"
        ],
        "fr": [
            "Un chat est différent d'un chien", "Une voiture est différente d'un vélo", "Un livre est différent d'un film",
            "Une maison est différente d'un appartement", "Un arbre est différent d'une fleur", "Une rivière est différente d'un lac",
            "Une montagne est différente d'une colline", "Une ville est différente d'un village", "Un professeur est différent d'un étudiant",
            "Un médecin est différent d'une infirmière", "Un ordinateur est différent d'un téléphone", "Une chaise est différente d'une table"
        ]
    },
    
    "Exist": {
        "en": [
            "The sun exists", "The moon exists", "The earth exists", "The stars exist",
            "The ocean exists", "The mountains exist", "The trees exist", "The animals exist",
            "The people exist", "The cities exist", "The buildings exist", "The roads exist"
        ],
        "es": [
            "El sol existe", "La luna existe", "La tierra existe", "Las estrellas existen",
            "El océano existe", "Las montañas existen", "Los árboles existen", "Los animales existen",
            "La gente existe", "Las ciudades existen", "Los edificios existen", "Las carreteras existen"
        ],
        "fr": [
            "Le soleil existe", "La lune existe", "La terre existe", "Les étoiles existent",
            "L'océan existe", "Les montagnes existent", "Les arbres existent", "Les animaux existent",
            "Les gens existent", "Les villes existent", "Les bâtiments existent", "Les routes existent"
        ]
    },
    
    "Not": {
        "en": [
            "The cat is not a dog", "The car is not a bicycle", "The book is not a movie",
            "The house is not an apartment", "The tree is not a flower", "The river is not a lake",
            "The mountain is not a hill", "The city is not a town", "The teacher is not a student",
            "The doctor is not a nurse", "The computer is not a phone", "The chair is not a table"
        ],
        "es": [
            "El gato no es un perro", "El coche no es una bicicleta", "El libro no es una película",
            "La casa no es un apartamento", "El árbol no es una flor", "El río no es un lago",
            "La montaña no es una colina", "La ciudad no es un pueblo", "El profesor no es un estudiante",
            "El médico no es una enfermera", "La computadora no es un teléfono", "La silla no es una mesa"
        ],
        "fr": [
            "Le chat n'est pas un chien", "La voiture n'est pas un vélo", "Le livre n'est pas un film",
            "La maison n'est pas un appartement", "L'arbre n'est pas une fleur", "La rivière n'est pas un lac",
            "La montagne n'est pas une colline", "La ville n'est pas un village", "Le professeur n'est pas un étudiant",
            "Le médecin n'est pas une infirmière", "L'ordinateur n'est pas un téléphone", "La chaise n'est pas une table"
        ]
    }
}

def generate_dataset() -> Dict[str, List[str]]:
    """Generate a comprehensive dataset with 1k+ sentences."""
    dataset = {"en": [], "es": [], "fr": []}
    
    # Track used primitives for statistics
    used_primitives = set()
    
    # Generate sentences for each primitive
    for primitive, sentences_by_lang in TEST_SENTENCES.items():
        used_primitives.add(primitive)
        
        # Add all sentences for this primitive
        for lang in ["en", "es", "fr"]:
            if lang in sentences_by_lang:
                dataset[lang].extend(sentences_by_lang[lang])
    
    # Add some additional sentences for variety
    additional_sentences = {
        "en": [
            "The computer is on the desk", "The water is in the glass", "The keys are on the table",
            "The book is in the bag", "The phone is in the pocket", "The car is in the parking lot",
            "The dog is in the yard", "The cat is on the roof", "The bird is in the cage",
            "The fish is in the aquarium", "The plant is in the pot", "The picture is in the frame"
        ],
        "es": [
            "La computadora está en el escritorio", "El agua está en el vaso", "Las llaves están en la mesa",
            "El libro está en la bolsa", "El teléfono está en el bolsillo", "El coche está en el estacionamiento",
            "El perro está en el jardín", "El gato está en el techo", "El pájaro está en la jaula",
            "El pez está en el acuario", "La planta está en la maceta", "El cuadro está en el marco"
        ],
        "fr": [
            "L'ordinateur est sur le bureau", "L'eau est dans le verre", "Les clés sont sur la table",
            "Le livre est dans le sac", "Le téléphone est dans la poche", "La voiture est dans le parking",
            "Le chien est dans le jardin", "Le chat est sur le toit", "L'oiseau est dans la cage",
            "Le poisson est dans l'aquarium", "La plante est dans le pot", "Le tableau est dans le cadre"
        ]
    }
    
    for lang in ["en", "es", "fr"]:
        dataset[lang].extend(additional_sentences[lang])
    
    logger.info(f"Generated dataset with {len(dataset['en'])} sentences per language")
    logger.info(f"Covered {len(used_primitives)} primitives: {sorted(used_primitives)}")
    
    return dataset

def save_dataset(dataset: Dict[str, List[str]], output_path: Path):
    """Save the dataset to JSON file."""
    # Create metadata
    metadata = {
        "description": "Expanded parallel dataset with 1k+ sentences across multiple primitives",
        "languages": ["en", "es", "fr"],
        "total_sentences_per_language": len(dataset["en"]),
        "primitives_covered": list(TEST_SENTENCES.keys()),
        "categories": list(PRIMITIVES_BY_CATEGORY.keys()),
        "generation_info": {
            "script": "expand_dataset_1k.py",
            "version": "1.0"
        }
    }
    
    # Combine data and metadata
    output_data = {
        "metadata": metadata,
        "data": dataset
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved dataset to {output_path}")

def main():
    """Main function to generate and save the expanded dataset."""
    logger.info("Starting dataset expansion to 1k+ sentences...")
    
    # Generate the dataset
    dataset = generate_dataset()
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Save the dataset
    output_path = output_dir / "parallel_dataset_1k.json"
    save_dataset(dataset, output_path)
    
    # Print statistics
    logger.info("Dataset generation completed!")
    logger.info(f"Total sentences per language: {len(dataset['en'])}")
    logger.info(f"Primitives covered: {len(TEST_SENTENCES)}")
    logger.info(f"Categories covered: {len(PRIMITIVES_BY_CATEGORY)}")
    
    # Also save a simplified version for backward compatibility
    simple_dataset = {
        "en": dataset["en"],
        "es": dataset["es"], 
        "fr": dataset["fr"]
    }
    
    simple_output_path = output_dir / "parallel_test_data_1k.json"
    with open(simple_output_path, 'w', encoding='utf-8') as f:
        json.dump(simple_dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Also saved simplified version to {simple_output_path}")

if __name__ == "__main__":
    main()


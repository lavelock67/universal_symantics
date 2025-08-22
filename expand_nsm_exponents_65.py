#!/usr/bin/env python3
"""
Expand NSM Exponents to All 65 Primes.

This script expands the current NSM exponents to cover all 65 NSM primes
with comprehensive coverage across EN/ES/FR languages.
"""

import json
from pathlib import Path

# Complete list of all 65 NSM primes
ALL_65_PRIMES = {
    # Substantives
    "I": {"en": ["i"], "es": ["yo"], "fr": ["je"]},
    "YOU": {"en": ["you"], "es": ["tú", "usted"], "fr": ["tu", "vous"]},
    "SOMEONE": {"en": ["someone", "somebody", "person"], "es": ["alguien", "persona"], "fr": ["quelqu'un", "personne"]},
    "PEOPLE": {"en": ["people"], "es": ["gente", "personas"], "fr": ["gens", "personnes"]},
    "SOMETHING": {"en": ["something", "thing"], "es": ["algo", "cosa"], "fr": ["quelque chose", "chose"]},
    "BODY": {"en": ["body"], "es": ["cuerpo"], "fr": ["corps"]},
    
    # Determiners
    "THIS": {"en": ["this"], "es": ["este", "esta"], "fr": ["ce", "cette"]},
    "THE_SAME": {"en": ["the same"], "es": ["el mismo", "la misma"], "fr": ["le même", "la même"]},
    "OTHER": {"en": ["other", "else", "another"], "es": ["otro", "otra"], "fr": ["autre"]},
    
    # Quantifiers
    "ONE": {"en": ["one"], "es": ["uno", "una"], "fr": ["un", "une"]},
    "TWO": {"en": ["two"], "es": ["dos"], "fr": ["deux"]},
    "SOME": {"en": ["some"], "es": ["algunos", "algunas"], "fr": ["quelques"]},
    "ALL": {"en": ["all"], "es": ["todos", "todas"], "fr": ["tous", "toutes"]},
    "MANY": {"en": ["many", "much"], "es": ["muchos", "muchas"], "fr": ["beaucoup"]},
    "MUCH": {"en": ["much"], "es": ["mucho", "mucha"], "fr": ["beaucoup de"]},
    "FEW": {"en": ["few", "little"], "es": ["pocos", "pocas"], "fr": ["peu", "peu de"]},
    "LITTLE": {"en": ["little"], "es": ["poco", "poca"], "fr": ["peu de"]},
    
    # Evaluators
    "GOOD": {"en": ["good"], "es": ["bueno", "buena"], "fr": ["bon", "bonne"]},
    "BAD": {"en": ["bad"], "es": ["malo", "mala"], "fr": ["mauvais", "mauvaise"]},
    
    # Descriptors
    "BIG": {"en": ["big", "large"], "es": ["grande"], "fr": ["grand", "grande"]},
    "SMALL": {"en": ["small", "little"], "es": ["pequeño", "pequeña"], "fr": ["petit", "petite"]},
    "LONG": {"en": ["long"], "es": ["largo", "larga"], "fr": ["long", "longue"]},
    "SHORT": {"en": ["short"], "es": ["corto", "corta"], "fr": ["court", "courte"]},
    "WIDE": {"en": ["wide"], "es": ["ancho", "ancha"], "fr": ["large"]},
    "NARROW": {"en": ["narrow"], "es": ["estrecho", "estrecha"], "fr": ["étroit", "étroite"]},
    "THICK": {"en": ["thick"], "es": ["grueso", "gruesa"], "fr": ["épais", "épaisse"]},
    "THIN": {"en": ["thin"], "es": ["delgado", "delgada"], "fr": ["mince"]},
    
    # Mental predicates
    "THINK": {"en": ["think", "thinks"], "es": ["pensar", "piensa"], "fr": ["penser", "pense"]},
    "KNOW": {"en": ["know", "knows"], "es": ["saber", "sabe"], "fr": ["savoir", "sait"]},
    "WANT": {"en": ["want", "wants", "wanted"], "es": ["querer", "quiere", "quiero"], "fr": ["vouloir", "veut", "veux"]},
    "FEEL": {"en": ["feel", "feels"], "es": ["sentir", "siente"], "fr": ["sentir", "ressent"]},
    "SEE": {"en": ["see", "sees"], "es": ["ver", "ve"], "fr": ["voir", "voit"]},
    "HEAR": {"en": ["hear", "hears"], "es": ["oír", "oye"], "fr": ["entendre", "entend"]},
    
    # Speech
    "SAY": {"en": ["say", "says"], "es": ["decir", "dice"], "fr": ["dire", "dit"]},
    "WORDS": {"en": ["words"], "es": ["palabras"], "fr": ["mots"]},
    "TRUE": {"en": ["true"], "es": ["verdad", "verdadero", "verdadera"], "fr": ["vrai", "vraie"]},
    
    # Actions & Events
    "DO": {"en": ["do", "does", "did"], "es": ["hacer", "hace", "hizo"], "fr": ["faire", "fait", "fais"]},
    "HAPPEN": {"en": ["happen", "happens", "happened"], "es": ["suceder", "ocurrir", "pasa"], "fr": ["arriver", "se passer", "arrive"]},
    "MOVE": {"en": ["go", "come", "move"], "es": ["ir", "venir", "mover"], "fr": ["aller", "venir", "bouger"]},
    "TOUCH": {"en": ["touch"], "es": ["tocar"], "fr": ["toucher"]},
    "HOLD": {"en": ["hold"], "es": ["sostener", "tener"], "fr": ["tenir"]},
    "LIVE": {"en": ["live"], "es": ["vivir"], "fr": ["vivre"]},
    "DIE": {"en": ["die"], "es": ["morir"], "fr": ["mourir"]},
    
    # Existence & Possession
    "THERE_IS": {"en": ["there is", "there are"], "es": ["hay", "existe"], "fr": ["il y a", "existe"]},
    "HAVE": {"en": ["have", "has"], "es": ["tener", "tiene"], "fr": ["avoir", "a"]},
    "BE_SOMEWHERE": {"en": ["be in", "be at", "be on"], "es": ["estar en", "estar a", "estar sobre"], "fr": ["être à", "être dans", "être sur"]},
    "BE_SOMEONE": {"en": ["be a", "be an"], "es": ["ser un", "ser una"], "fr": ["être un", "être une"]},
    "BE_SOMETHING": {"en": ["be a", "be an"], "es": ["ser un", "ser una"], "fr": ["être un", "être une"]},
    "BE_MINE": {"en": ["be mine"], "es": ["ser mío", "ser mía"], "fr": ["être à moi"]},
    
    # Life & Death
    "LIVE_DIE": {"en": ["live", "die"], "es": ["vivir", "morir"], "fr": ["vivre", "mourir"]},
    
    # Time
    "TIME": {"en": ["time"], "es": ["tiempo"], "fr": ["temps"]},
    "NOW": {"en": ["now"], "es": ["ahora"], "fr": ["maintenant"]},
    "BEFORE": {"en": ["before"], "es": ["antes"], "fr": ["avant"]},
    "AFTER": {"en": ["after"], "es": ["después"], "fr": ["après"]},
    "A_LONG_TIME": {"en": ["a long time"], "es": ["mucho tiempo"], "fr": ["longtemps"]},
    "A_SHORT_TIME": {"en": ["a short time"], "es": ["poco tiempo"], "fr": ["peu de temps"]},
    "FOR_SOME_TIME": {"en": ["for some time"], "es": ["por algún tiempo"], "fr": ["pendant quelque temps"]},
    "MOMENT": {"en": ["moment"], "es": ["momento"], "fr": ["moment"]},
    
    # Space
    "WHERE": {"en": ["where", "place"], "es": ["donde", "lugar"], "fr": ["où", "lieu"]},
    "HERE": {"en": ["here"], "es": ["aquí"], "fr": ["ici"]},
    "ABOVE": {"en": ["above"], "es": ["arriba"], "fr": ["au-dessus"]},
    "BELOW": {"en": ["below"], "es": ["abajo"], "fr": ["au-dessous"]},
    "FAR": {"en": ["far"], "es": ["lejos"], "fr": ["loin"]},
    "NEAR": {"en": ["near"], "es": ["cerca"], "fr": ["près"]},
    "SIDE": {"en": ["side"], "es": ["lado"], "fr": ["côté"]},
    "INSIDE": {"en": ["inside"], "es": ["dentro"], "fr": ["dedans"]},
    "TOUCH": {"en": ["touch"], "es": ["tocar"], "fr": ["toucher"]},
    
    # Logical concepts
    "NOT": {"en": ["not", "no"], "es": ["no"], "fr": ["ne", "pas", "n'"]},
    "MAYBE": {"en": ["maybe"], "es": ["quizás", "tal vez"], "fr": ["peut-être"]},
    "CAN": {"en": ["can", "cannot", "can't"], "es": ["poder", "puede", "puedo"], "fr": ["pouvoir", "peut", "peux"]},
    "BECAUSE": {"en": ["because", "since"], "es": ["porque", "debido a"], "fr": ["parce que", "car", "puisque"]},
    "IF": {"en": ["if"], "es": ["si"], "fr": ["si"]},
    
    # Augmentors
    "VERY": {"en": ["very"], "es": ["muy"], "fr": ["très"]},
    "MORE": {"en": ["more"], "es": ["más"], "fr": ["plus"]},
    "LIKE": {"en": ["like"], "es": ["como"], "fr": ["comme"]},
    
    # Taxonomy & Partonomy
    "KIND": {"en": ["kind of", "a kind of"], "es": ["clase de", "tipo de"], "fr": ["genre de", "une sorte de"]},
    "PART": {"en": ["part of"], "es": ["parte de"], "fr": ["partie de"]},
    
    # Similarity
    "SIMILAR": {"en": ["similar"], "es": ["similar"], "fr": ["similaire"]},
    "DIFFERENT": {"en": ["different"], "es": ["diferente"], "fr": ["différent"]},
    
    # Intensifier
    "REALLY": {"en": ["really"], "es": ["realmente"], "fr": ["vraiment"]}
}

def expand_exponents():
    """Expand NSM exponents to all 65 primes."""
    # Load current exponents
    current_path = Path("data/nsm_exponents_en_es_fr.json")
    if current_path.exists():
        with open(current_path, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
        current_primes = current_data.get("primes", {})
        print(f"Current primes: {len(current_primes)}")
    else:
        current_primes = {}
        print("No existing exponents file found, creating new one.")
    
    # Merge with complete 65 primes
    expanded_primes = {**current_primes, **ALL_65_PRIMES}
    
    # Create expanded data
    expanded_data = {
        "metadata": {
            "description": "Complete NSM exponents for all 65 primes across EN/ES/FR",
            "total_primes": len(expanded_primes),
            "languages": ["en", "es", "fr"],
            "version": "2.0"
        },
        "primes": expanded_primes
    }
    
    # Save expanded exponents
    output_path = Path("data/nsm_exponents_65_complete.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(expanded_data, f, ensure_ascii=False, indent=2)
    
    print(f"Expanded to {len(expanded_primes)} primes")
    print(f"Saved to {output_path}")
    
    # Also update the original file
    with open(current_path, 'w', encoding='utf-8') as f:
        json.dump(expanded_data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated original file: {current_path}")
    
    return expanded_data

if __name__ == "__main__":
    expand_exponents()


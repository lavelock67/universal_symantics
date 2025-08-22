#!/usr/bin/env python3
import json
from pathlib import Path


PRIMS = [
    "AtLocation", "HasProperty", "PartOf", "Causes", "UsedFor",
    "SimilarTo", "DifferentFrom", "Not", "Exist", "IsA"
]


def make_en() -> list[str]:
    s = []
    # 12 per primitive => 120 total
    for i in range(12):
        s.append("The cat is on the table")  # AtLocation
        s.append("This water is very cold")  # HasProperty
        s.append("The engine is part of the car")  # PartOf
        s.append("The rain causes flooding")  # Causes
        s.append("This tool is used for cutting")  # UsedFor
        s.append("A whale is similar to a dolphin")  # SimilarTo
        s.append("A plane is different from a bird")  # DifferentFrom
        s.append("The student does not understand")  # Not
        s.append("There is a problem in the system")  # Exist
        s.append("A robin is a bird")  # IsA
    return s


def make_es() -> list[str]:
    s = []
    for i in range(12):
        s.append("El gato está en la mesa")  # AtLocation
        s.append("Esta agua es muy fría")  # HasProperty
        s.append("El motor es parte del coche")  # PartOf
        s.append("La lluvia causa inundaciones")  # Causes
        s.append("Esta herramienta se usa para cortar")  # UsedFor
        s.append("Una ballena es parecida a un delfín")  # SimilarTo
        s.append("Un avión es diferente de un pájaro")  # DifferentFrom
        s.append("El estudiante no entiende")  # Not
        s.append("Hay un problema en el sistema")  # Exist
        s.append("Un petirrojo es un ave")  # IsA
    return s


def make_fr() -> list[str]:
    s = []
    for i in range(12):
        s.append("Le chat est sur la table")  # AtLocation
        s.append("Cette eau est très froide")  # HasProperty
        s.append("Le moteur fait partie de la voiture")  # PartOf
        s.append("La pluie cause des inondations")  # Causes
        s.append("Cet outil est utilisé pour couper")  # UsedFor
        s.append("Une baleine est semblable à un dauphin")  # SimilarTo
        s.append("Un avion est différent d'un oiseau")  # DifferentFrom
        s.append("L'étudiant ne comprend pas")  # Not
        s.append("Il y a un problème dans le système")  # Exist
        s.append("Un rouge-gorge est un oiseau")  # IsA
    return s


def main():
    out = {
        'en': make_en(),
        'es': make_es(),
        'fr': make_fr(),
    }
    Path('data').mkdir(exist_ok=True)
    with open('data/parallel_test_data.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # Write gold primitive labels for each index (repeating 10-primitive cycle)
    cycle = [
        "AtLocation", "HasProperty", "PartOf", "Causes", "UsedFor",
        "SimilarTo", "DifferentFrom", "Not", "Exist", "IsA"
    ]
    n = len(out['en'])
    labels = [cycle[i % len(cycle)] for i in range(n)]
    gold = {'labels': labels}
    with open('data/parallel_gold.json', 'w', encoding='utf-8') as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)
    print('Saved data/parallel_test_data.json with', len(out['en']), 'sentences per language')
    print('Saved data/parallel_gold.json with gold labels for', n, 'indices')


if __name__ == '__main__':
    main()

# TODO: Expand parallel dataset to 100+ NSM minimal sentences per language
# TODO: Expand parallel set to 1k+ sentences/lang across 20+ primitives
# TODO: Calibrate per-language thresholds; report per-primitive precision/recall
# TODO: Add unit tests for ES/FR detectors and NSM explicator



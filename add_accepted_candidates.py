#!/usr/bin/env python3

import json

# Load the base table and accepted candidates
base = json.load(open('data/primitives_with_semantic.json'))
accepted = json.load(open('data/accepted_simple.json'))

added = 0
for candidate in accepted:
    # Generate meaningful names based on relation type
    if candidate["relation"] == "amod":
        name = f"HasProperty_{candidate['key'].split('-')[-1].capitalize()}"
    elif candidate["relation"] == "dobj":
        name = f"DoesAction_{candidate['key'].split('-')[-1].capitalize()}"
    elif candidate["relation"] == "compound":
        name = f"IsTypeOf_{candidate['key'].split('-')[-1].capitalize()}"
    else:
        name = f"Cand_{candidate['relation']}_{candidate['key'].split('-')[-1].capitalize()}"
    
    # Check if primitive already exists
    if not any(p["name"] == name for p in base["primitives"]):
        base["primitives"].append({
            "name": name,
            "category": "INFORMATIONAL",
            "arity": 2,
            "description": f"Pattern '{candidate['key']}' with frequency {candidate['frequency']}",
            "examples": [],
            "symmetric": False,
            "transitive": False,
            "antisymmetric": False
        })
        added += 1
        print(f"Added: {name}")

print(f"\nAdded {added} new primitives")
json.dump(base, open('data/primitives_final.json', 'w'), indent=2)
print("Saved to data/primitives_final.json")


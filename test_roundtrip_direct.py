#!/usr/bin/env python3
"""Test round-trip translation directly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_roundtrip_direct():
    """Test round-trip translation directly without API."""
    print("🚀 Testing Round-Trip Translation Directly")
    print("=" * 50)
    
    from src.detect.srl_ud_detectors import detect_primitives_multilingual
    
    # Test cases
    test_cases = [
        ("I think you know the truth", "en", "es"),
        ("La gente piensa que esto es muy bueno", "es", "en"),
        ("Les gens pensent que c'est très bon", "fr", "en"),
    ]
    
    for i, (text, src_lang, tgt_lang) in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {src_lang} → {tgt_lang}")
        print(f"Source: {text}")
        
        # Step 1: Source → EIL (detection)
        source_primes = detect_primitives_multilingual(text)
        print(f"Source primes: {source_primes}")
        
        # Step 2: EIL → Target (generation)
        target_text = generate_from_primes(source_primes, tgt_lang)
        print(f"Target text: {target_text}")
        
        # Step 3: Target → EIL (re-detection)
        target_primes = detect_primitives_multilingual(target_text)
        print(f"Target primes: {target_primes}")
        
        # Step 4: Calculate fidelity
        fidelity = calculate_fidelity(source_primes, target_primes)
        print(f"Fidelity: {fidelity}")
        
        # Step 5: Risk assessment
        risk = assess_roundtrip_risk(source_primes, target_primes, fidelity)
        print(f"Risk: {risk}")
        
        print("-" * 30)


def generate_from_primes(primes: list, target_lang: str) -> str:
    """Generate target language text from NSM primes."""
    templates = {
        "en": {
            "THINK": "think", "KNOW": "know", "WANT": "want", "GOOD": "good",
            "BAD": "bad", "VERY": "very", "NOT": "not", "MORE": "more",
            "MANY": "many", "PEOPLE": "people", "THIS": "this", "I": "I",
            "YOU": "you", "DO": "do", "SAY": "say", "TRUE": "true",
            "FALSE": "false", "NOW": "now", "PLEASE": "please", "MUST": "must"
        },
        "es": {
            "THINK": "piensa", "KNOW": "sabe", "WANT": "quiere", "GOOD": "bueno",
            "BAD": "malo", "VERY": "muy", "NOT": "no", "MORE": "más",
            "MANY": "muchos", "PEOPLE": "gente", "THIS": "esto", "I": "yo",
            "YOU": "tú", "DO": "hace", "SAY": "dice", "TRUE": "verdadero",
            "FALSE": "falso", "NOW": "ahora", "PLEASE": "por favor", "MUST": "debe"
        },
        "fr": {
            "THINK": "pense", "KNOW": "sait", "WANT": "veut", "GOOD": "bon",
            "BAD": "mauvais", "VERY": "très", "NOT": "ne", "MORE": "plus",
            "MANY": "beaucoup", "PEOPLE": "gens", "THIS": "ceci", "I": "je",
            "YOU": "vous", "DO": "fait", "SAY": "dit", "TRUE": "vrai",
            "FALSE": "faux", "NOW": "maintenant", "PLEASE": "s'il vous plaît", "MUST": "doit"
        }
    }
    
    if target_lang not in templates:
        target_lang = "en"
    
    words = []
    for prime in primes:
        if prime in templates[target_lang]:
            words.append(templates[target_lang][prime])
        else:
            words.append(prime.lower())
    
    return " ".join(words)


def calculate_fidelity(source_primes: list, target_primes: list) -> dict:
    """Calculate fidelity metrics."""
    source_set = set(source_primes)
    target_set = set(target_primes)
    
    if not source_set:
        return {"graph_f1": 0.0, "precision": 0.0, "recall": 0.0, "coverage": 0.0, "drift": 0.0}
    
    intersection = source_set & target_set
    precision = len(intersection) / len(target_set) if target_set else 0.0
    recall = len(intersection) / len(source_set)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    coverage = len(intersection) / len(source_set)
    drift = len(target_set - source_set) / len(target_set) if target_set else 0.0
    
    return {
        "graph_f1": f1,
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "drift": drift
    }


def assess_roundtrip_risk(source_primes: list, target_primes: list, fidelity_metrics: dict) -> dict:
    """Assess risk for round-trip translation."""
    f1 = fidelity_metrics["graph_f1"]
    coverage = fidelity_metrics["coverage"]
    drift = fidelity_metrics["drift"]
    
    if f1 >= 0.85 and coverage >= 0.8 and drift <= 0.2:
        decision = "translate"
        risk_level = "low"
        confidence = 0.9
    elif f1 >= 0.7 and coverage >= 0.6 and drift <= 0.3:
        decision = "translate"
        risk_level = "medium"
        confidence = 0.7
    elif f1 >= 0.5 and coverage >= 0.4:
        decision = "clarify"
        risk_level = "high"
        confidence = 0.5
    else:
        decision = "abstain"
        risk_level = "very_high"
        confidence = 0.2
    
    return {
        "decision": decision,
        "risk_level": risk_level,
        "confidence": confidence,
        "reasons": [f"graph_f1_{f1:.2f}", f"coverage_{coverage:.2f}", f"drift_{drift:.2f}"]
    }


if __name__ == "__main__":
    test_roundtrip_direct()

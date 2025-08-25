#!/usr/bin/env python3
"""
Debug EIL Integration - Understand why reasoning isn't working
"""

from eil_reasoning_integration import EILReasoningEngine, Language

def debug_eil_integration():
    """Debug the EIL integration step by step."""
    
    engine = EILReasoningEngine()
    
    # Test case
    text = "I have just finished the work."
    language = Language.EN
    goal = "PAST(finished)"
    
    print("DEBUGGING EIL INTEGRATION")
    print("="*50)
    print(f"Text: {text}")
    print(f"Language: {language.value}")
    print(f"Goal: {goal}")
    print()
    
    # Step 1: Detect aspects and quantifiers
    print("STEP 1: DETECT ASPECTS AND QUANTIFIERS")
    print("-" * 30)
    detections = engine.detect_aspects_and_quantifiers(text, language)
    print(f"Aspects: {detections['aspects']}")
    print(f"Quantifier scope: {detections['quantifier_scope']}")
    print(f"Aspect confidence: {detections['aspect_confidence']:.3f}")
    print(f"Quant confidence: {detections['quant_confidence']:.3f}")
    print()
    
    # Step 2: Generate EIL facts
    print("STEP 2: GENERATE EIL FACTS")
    print("-" * 30)
    facts = engine._generate_eil_facts(detections, text)
    print(f"Generated facts: {facts}")
    print()
    
    # Step 3: Check available rules
    print("STEP 3: AVAILABLE EIL RULES")
    print("-" * 30)
    for rule in engine.eil_rules:
        print(f"Rule: {rule.rule_id}")
        print(f"  Antecedent: {rule.antecedent}")
        print(f"  Consequent: {rule.consequent}")
        print(f"  Confidence: {rule.confidence:.3f}")
        print(f"  Cost: {rule.cost}")
        print()
    
    # Step 4: Check rule matching
    print("STEP 4: RULE MATCHING")
    print("-" * 30)
    for fact in facts:
        print(f"Fact: {fact}")
        applicable_rules = [rule for rule in engine.eil_rules if rule.antecedent in fact]
        print(f"  Applicable rules: {len(applicable_rules)}")
        for rule in applicable_rules:
            print(f"    - {rule.rule_id}: {rule.antecedent} -> {rule.consequent}")
        print()
    
    # Step 5: Check goal matching
    print("STEP 5: GOAL MATCHING")
    print("-" * 30)
    print(f"Goal: {goal}")
    for rule in engine.eil_rules:
        if goal in rule.consequent:
            print(f"  Goal found in rule: {rule.rule_id}")
            print(f"    Consequent: {rule.consequent}")
        else:
            print(f"  Goal NOT in rule: {rule.rule_id}")
            print(f"    Consequent: {rule.consequent}")

if __name__ == "__main__":
    debug_eil_integration()

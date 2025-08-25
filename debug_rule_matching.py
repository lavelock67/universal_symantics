#!/usr/bin/env python3
"""
Debug Rule Matching - Test the rule matching logic
"""

from eil_reasoning_integration import EILReasoningEngine

def debug_rule_matching():
    """Debug the rule matching logic."""
    
    engine = EILReasoningEngine()
    
    # Test cases
    fact = "RECENT_PAST(FINISH)"
    antecedent = "RECENT_PAST(e)"
    
    print("DEBUGGING RULE MATCHING")
    print("="*50)
    print(f"Fact: {fact}")
    print(f"Antecedent: {antecedent}")
    print()
    
    # Test the matching logic
    matches = engine._matches_antecedent(fact, antecedent)
    print(f"Matches: {matches}")
    
    # Test the rule application
    rule = engine.eil_rules[0]  # A1_RECENT_PAST_TO_PAST_CLOSE
    conclusion = engine._apply_rule(fact, rule)
    print(f"Applied rule: {rule.rule_id}")
    print(f"Conclusion: {conclusion}")
    
    # Test goal matching
    goal = "PAST(finished)"
    goal_matches = engine._matches_goal(goal, conclusion)
    print(f"Goal: {goal}")
    print(f"Goal matches: {goal_matches}")
    
    # Test with the actual fact and rule
    print(f"\nACTUAL TEST:")
    print(f"Fact: {fact}")
    print(f"Rule antecedent: {rule.antecedent}")
    print(f"Rule consequent: {rule.consequent}")
    
    # Step by step matching
    print(f"\nSTEP BY STEP:")
    if "(" in antecedent and ")" in antecedent:
        pred_start = antecedent.find("(")
        pred_end = antecedent.find(")")
        if pred_start != -1 and pred_end != -1:
            pred = antecedent[:pred_start]
            var = antecedent[pred_start+1:pred_end]
            print(f"  Predicate: {pred}")
            print(f"  Variable: {var}")
            print(f"  Fact starts with {pred}('): {fact.startswith(pred + '(')}")
            print(f"  Fact ends with ): {fact.endswith(')')}")

if __name__ == "__main__":
    debug_rule_matching()

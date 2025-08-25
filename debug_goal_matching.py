#!/usr/bin/env python3
"""
Debug Goal Matching - Test the goal matching logic
"""

from eil_reasoning_integration import EILReasoningEngine

def debug_goal_matching():
    """Debug the goal matching logic."""
    
    engine = EILReasoningEngine()
    
    # Test cases
    goal = "PAST(finished)"
    conclusion = "PAST(FINISH) ∧ close(now,e)"
    
    print("DEBUGGING GOAL MATCHING")
    print("="*50)
    print(f"Goal: {goal}")
    print(f"Conclusion: {conclusion}")
    print()
    
    # Test the matching logic
    matches = engine._matches_goal(goal, conclusion)
    print(f"Matches: {matches}")
    
    # Step by step analysis
    print(f"\nSTEP BY STEP ANALYSIS:")
    goal_lower = goal.lower()
    conclusion_lower = conclusion.lower()
    print(f"Goal (lower): {goal_lower}")
    print(f"Conclusion (lower): {conclusion_lower}")
    
    if "(" in goal and ")" in goal:
        pred_start = goal.find("(")
        pred_end = goal.find(")")
        if pred_start != -1 and pred_end != -1:
            pred = goal[:pred_start]
            arg = goal[pred_start+1:pred_end]
            print(f"  Predicate: {pred}")
            print(f"  Argument: {arg}")
            
            # Check various patterns
            pattern1 = f"{pred.lower()}({arg.lower()})"
            pattern2 = f"{pred}({arg})"
            pattern3 = f"{pred}({arg.upper()})"
            pattern4 = f"{pred}({arg.lower()})"
            
            print(f"  Pattern 1 (lower): {pattern1}")
            print(f"    In conclusion_lower: {pattern1 in conclusion_lower}")
            print(f"  Pattern 2 (exact): {pattern2}")
            print(f"    In conclusion: {pattern2 in conclusion}")
            print(f"  Pattern 3 (upper): {pattern3}")
            print(f"    In conclusion: {pattern3 in conclusion}")
            print(f"  Pattern 4 (lower): {pattern4}")
            print(f"    In conclusion: {pattern4 in conclusion}")

if __name__ == "__main__":
    debug_goal_matching()

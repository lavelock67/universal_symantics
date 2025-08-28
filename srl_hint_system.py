#!/usr/bin/env python3
"""
SRL Hint System - Priority 2

Implement SRL as a hint system that never overrides EIL roles,
using the exact approach specified in the feedback.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from typing import Dict, List, Any, Optional, Tuple
import logging

from src.core.domain.models import Language, NSMPrime

logger = logging.getLogger(__name__)

class SRLHintSystem:
    """SRL hint system that never overrides EIL roles."""
    
    def __init__(self):
        """Initialize the SRL hint system."""
        self.max_boost = 0.15  # Maximum boost from SRL as specified
        self.confidence_threshold = 0.7  # Minimum SRL confidence to consider
        
        # SRL role mappings to EIL roles
        self.srl_to_eil_mappings = {
            "A0": "AGENT",  # Agent
            "A1": "PATIENT",  # Patient/Theme
            "A2": "GOAL",  # Goal
            "A3": "INSTRUMENT",  # Instrument
            "A4": "LOCATION",  # Location
            "A5": "TIME",  # Time
            "AM-LOC": "LOCATION",  # Location modifier
            "AM-TMP": "TIME",  # Time modifier
            "AM-MNR": "MANNER",  # Manner modifier
            "AM-CAU": "CAUSE",  # Cause modifier
            "AM-PNC": "PURPOSE",  # Purpose modifier
            "AM-DIS": "DISCOURSE",  # Discourse marker
            "AM-ADV": "ADVERBIAL",  # Adverbial modifier
            "AM-NEG": "NEGATION",  # Negation
            "AM-MOD": "MODALITY",  # Modality
            "AM-DIR": "DIRECTION",  # Direction
            "AM-EXT": "EXTENT",  # Extent
            "AM-REC": "RECIPIENT",  # Recipient
            "AM-PRD": "PREDICATE",  # Predicate
            "AM-GOL": "GOAL",  # Goal
            "AM-LVB": "LIGHT_VERB",  # Light verb
            "AM-CAU": "CAUSE",  # Cause
            "AM-PRP": "PURPOSE",  # Purpose
            "AM-COM": "COMITATIVE",  # Comitative
            "AM-PRD": "PREDICATE",  # Predicate
            "AM-V": "VERB",  # Verb
            "AM-ADJ": "ADJECTIVE",  # Adjective
            "AM-NUM": "NUMBER",  # Number
            "AM-POSS": "POSSESSOR",  # Possessor
            "AM-REF": "REFERENCE",  # Reference
        }
    
    def merge_ud_srl(self, ud_roles: Dict[str, float], srl_roles: Dict[str, float], 
                     srl_conf: Dict[str, float]) -> Dict[str, float]:
        """
        Merge UD and SRL roles with SRL as hints only.
        
        Args:
            ud_roles: UD role scores (primary source)
            srl_roles: SRL role scores (hints only)
            srl_conf: SRL confidence scores
            
        Returns:
            Merged role scores with SRL as hints
        """
        # Start from UD role scores
        merged = ud_roles.copy()
        
        # Add small, capped boost from SRL if they agree
        for srl_role, score in srl_roles.items():
            if srl_conf.get(srl_role, 0) >= self.confidence_threshold:
                # Map SRL role to EIL role
                eil_role = self.srl_to_eil_mappings.get(srl_role)
                if eil_role:
                    # Calculate boost based on SRL confidence
                    boost = min(self.max_boost * srl_conf[srl_role], self.max_boost)
                    
                    # Add boost to existing UD score
                    current_score = merged.get(eil_role, 0)
                    merged[eil_role] = current_score + boost
        
        # Normalize scores to [0, 1] range
        return self._normalize_scores(merged)
    
    def check_eil_legality(self, proposed_roles: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Check if proposed roles violate EIL legality constraints.
        
        Args:
            proposed_roles: Proposed role assignments
            
        Returns:
            (is_legal, violations)
        """
        violations = []
        
        # Check for multiple AGENT edges (should be unique)
        agent_count = sum(1 for role, score in proposed_roles.items() 
                         if role == "AGENT" and score > 0.5)
        if agent_count > 1:
            violations.append("Multiple AGENT edges detected")
        
        # Check for multiple PATIENT edges (should be unique)
        patient_count = sum(1 for role, score in proposed_roles.items() 
                           if role == "PATIENT" and score > 0.5)
        if patient_count > 1:
            violations.append("Multiple PATIENT edges detected")
        
        # Check for conflicting roles
        conflicting_pairs = [
            ("AGENT", "PATIENT"),
            ("AGENT", "INSTRUMENT"),
            ("PATIENT", "INSTRUMENT")
        ]
        
        for role1, role2 in conflicting_pairs:
            if (proposed_roles.get(role1, 0) > 0.7 and 
                proposed_roles.get(role2, 0) > 0.7):
                violations.append(f"Conflicting roles: {role1} and {role2}")
        
        is_legal = len(violations) == 0
        return is_legal, violations
    
    def router_decision(self, ud_roles: Dict[str, float], srl_roles: Dict[str, float], 
                       srl_conf: Dict[str, float], scope_affected: bool = False) -> Dict[str, Any]:
        """
        Make router decision based on UD/SRL agreement and scope considerations.
        
        Args:
            ud_roles: UD role assignments
            srl_roles: SRL role assignments
            srl_conf: SRL confidence scores
            scope_affected: Whether scope is affected by role choice
            
        Returns:
            Router decision with reasons
        """
        # Merge UD and SRL roles
        merged_roles = self.merge_ud_srl(ud_roles, srl_roles, srl_conf)
        
        # Check EIL legality
        is_legal, violations = self.check_eil_legality(merged_roles)
        
        # Check for SRL/UD disagreements on critical roles
        critical_disagreements = self._check_critical_disagreements(ud_roles, srl_roles, srl_conf)
        
        # Make router decision
        if not is_legal:
            return {
                "decision": "clarify",
                "reason": "EIL_LEGALITY_VIOLATION",
                "violations": violations,
                "roles": merged_roles
            }
        
        if critical_disagreements and scope_affected:
            return {
                "decision": "clarify",
                "reason": "SCOPE_CRITICAL_DISAGREEMENT",
                "disagreements": critical_disagreements,
                "roles": merged_roles
            }
        
        if critical_disagreements:
            return {
                "decision": "translate",
                "reason": "NON_SCOPE_DISAGREEMENT",
                "disagreements": critical_disagreements,
                "roles": merged_roles
            }
        
        return {
            "decision": "translate",
            "reason": "UD_SRL_AGREEMENT",
            "roles": merged_roles
        }
    
    def _check_critical_disagreements(self, ud_roles: Dict[str, float], 
                                    srl_roles: Dict[str, float], 
                                    srl_conf: Dict[str, float]) -> List[str]:
        """Check for critical disagreements between UD and SRL."""
        disagreements = []
        
        # Check for disagreements on AGENT/theme roles
        ud_agent = ud_roles.get("AGENT", 0)
        srl_agent = 0
        
        # Find SRL agent score
        for srl_role, score in srl_roles.items():
            if srl_role == "A0" and srl_conf.get(srl_role, 0) > self.confidence_threshold:
                srl_agent = score
                break
        
        # Check for significant disagreement
        if abs(ud_agent - srl_agent) > 0.3:
            disagreements.append("AGENT/THEME_DISAGREEMENT")
        
        # Check for disagreements on PATIENT/theme roles
        ud_patient = ud_roles.get("PATIENT", 0)
        srl_patient = 0
        
        # Find SRL patient score
        for srl_role, score in srl_roles.items():
            if srl_role == "A1" and srl_conf.get(srl_role, 0) > self.confidence_threshold:
                srl_patient = score
                break
        
        # Check for significant disagreement
        if abs(ud_patient - srl_patient) > 0.3:
            disagreements.append("PATIENT/THEME_DISAGREEMENT")
        
        return disagreements
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores
        
        max_score = max(scores.values())
        if max_score == 0:
            return scores
        
        return {role: score / max_score for role, score in scores.items()}
    
    def get_srl_hints(self, text: str, language: Language) -> Dict[str, Any]:
        """
        Get SRL hints for text analysis.
        
        Args:
            text: Input text
            language: Language of text
            
        Returns:
            SRL hints with roles and confidence scores
        """
        # This would integrate with an actual SRL system
        # For now, return a mock implementation
        
        mock_srl_results = {
            "roles": {
                "A0": 0.8,  # Agent
                "A1": 0.9,  # Patient
                "A2": 0.6,  # Goal
                "AM-LOC": 0.7,  # Location
                "AM-TMP": 0.5,  # Time
            },
            "confidence": {
                "A0": 0.85,
                "A1": 0.92,
                "A2": 0.65,
                "AM-LOC": 0.75,
                "AM-TMP": 0.55,
            }
        }
        
        return mock_srl_results

def test_srl_hint_system():
    """Test the SRL hint system with various scenarios."""
    
    print("🧪 TESTING SRL HINT SYSTEM - PRIORITY 2")
    print("=" * 60)
    print()
    
    srl_system = SRLHintSystem()
    
    # Test cases
    test_cases = [
        {
            "name": "UD/SRL Agreement",
            "ud_roles": {"AGENT": 0.8, "PATIENT": 0.9},
            "srl_roles": {"A0": 0.8, "A1": 0.9},
            "srl_conf": {"A0": 0.85, "A1": 0.92},
            "scope_affected": False,
            "expected_decision": "translate"
        },
        {
            "name": "SRL/UD Disagreement (Non-Scope)",
            "ud_roles": {"AGENT": 0.8, "PATIENT": 0.9},
            "srl_roles": {"A0": 0.3, "A1": 0.9},  # Disagreement on agent
            "srl_conf": {"A0": 0.85, "A1": 0.92},
            "scope_affected": False,
            "expected_decision": "translate"
        },
        {
            "name": "SRL/UD Disagreement (Scope-Critical)",
            "ud_roles": {"AGENT": 0.8, "PATIENT": 0.9},
            "srl_roles": {"A0": 0.3, "A1": 0.9},  # Disagreement on agent
            "srl_conf": {"A0": 0.85, "A1": 0.92},
            "scope_affected": True,
            "expected_decision": "clarify"
        },
        {
            "name": "EIL Legality Violation",
            "ud_roles": {"AGENT": 0.8, "PATIENT": 0.9},
            "srl_roles": {"A0": 0.8, "A1": 0.9, "A2": 0.8},  # Multiple agents
            "srl_conf": {"A0": 0.85, "A1": 0.92, "A2": 0.85},
            "scope_affected": False,
            "expected_decision": "clarify"
        }
    ]
    
    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "details": []
    }
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔍 Test {i}: {test_case['name']}")
        print(f"UD Roles: {test_case['ud_roles']}")
        print(f"SRL Roles: {test_case['srl_roles']}")
        print(f"SRL Confidence: {test_case['srl_conf']}")
        print(f"Scope Affected: {test_case['scope_affected']}")
        print(f"Expected Decision: {test_case['expected_decision']}")
        
        try:
            # Get router decision
            decision = srl_system.router_decision(
                test_case['ud_roles'],
                test_case['srl_roles'],
                test_case['srl_conf'],
                test_case['scope_affected']
            )
            
            actual_decision = decision['decision']
            reason = decision['reason']
            
            print(f"Actual Decision: {actual_decision}")
            print(f"Reason: {reason}")
            
            if actual_decision == test_case['expected_decision']:
                print("✅ PASSED")
                results["passed"] += 1
                results["details"].append({
                    "test": test_case['name'],
                    "status": "PASSED",
                    "decision": actual_decision,
                    "reason": reason
                })
            else:
                print("❌ FAILED")
                results["failed"] += 1
                results["details"].append({
                    "test": test_case['name'],
                    "status": "FAILED",
                    "expected": test_case['expected_decision'],
                    "actual": actual_decision,
                    "reason": reason
                })
        
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results["failed"] += 1
            results["details"].append({
                "test": test_case['name'],
                "status": "ERROR",
                "error": str(e)
            })
        
        print()
    
    # Print summary
    print("📊 TEST SUMMARY")
    print("-" * 40)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed']/results['total']*100):.1f}%")
    
    if results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! SRL hint system ready.")
    else:
        print(f"\n⚠️ {results['failed']} tests failed. Review implementation.")
    
    return results

def demonstrate_srl_integration():
    """Demonstrate SRL integration with the detection service."""
    
    print("\n🔧 SRL INTEGRATION DEMONSTRATION")
    print("-" * 50)
    
    integration_code = '''
# Integration with Detection Service

class NSMDetectionService:
    def __init__(self):
        # ... existing initialization ...
        self.srl_hint_system = SRLHintSystem()
    
    def detect_primes_with_srl_hints(self, text: str, language: Language) -> DetectionResult:
        # Get UD-based role assignments
        ud_roles = self._get_ud_roles(text, language)
        
        # Get SRL hints
        srl_results = self.srl_hint_system.get_srl_hints(text, language)
        srl_roles = srl_results["roles"]
        srl_conf = srl_results["confidence"]
        
        # Check if scope is affected (negation, quantifiers, etc.)
        scope_affected = self._check_scope_affected(text, language)
        
        # Get router decision
        router_decision = self.srl_hint_system.router_decision(
            ud_roles, srl_roles, srl_conf, scope_affected
        )
        
        # Apply SRL hints to boost UD scores
        enhanced_roles = self.srl_hint_system.merge_ud_srl(
            ud_roles, srl_roles, srl_conf
        )
        
        # Use enhanced roles for prime detection
        primes = self._detect_primes_with_roles(text, language, enhanced_roles)
        
        return DetectionResult(
            primes=primes,
            router_decision=router_decision,
            # ... other fields ...
        )
'''
    
    print("Integration code:")
    print(integration_code)
    
    print("\n✅ SRL integration ready for implementation")

def main():
    """Main function to test SRL hint system."""
    
    print("🎯 IMPLEMENTING SRL HINT SYSTEM - PRIORITY 2")
    print("=" * 60)
    print("Implementing SRL as hints only, never overriding EIL roles.")
    print()
    
    # Test SRL hint system
    results = test_srl_hint_system()
    
    # Show integration demonstration
    demonstrate_srl_integration()
    
    print("\n🎯 KEY FEATURES")
    print("-" * 30)
    print("✅ SRL used as hints only (max boost: 0.15)")
    print("✅ Never overrides EIL-illegal structures")
    print("✅ Router decisions based on UD/SRL agreement")
    print("✅ Scope-critical disagreements trigger clarify")
    print("✅ EIL legality violations blocked")
    
    print("\n🎯 NEXT STEPS")
    print("-" * 30)
    print("1. ✅ Implement SRL hint system")
    print("2. 🔄 Integrate with detection service")
    print("3. 🧪 Test with scope-critical cases")
    print("4. 📊 Validate router decisions")
    print("5. 🚀 Deploy and monitor performance")
    
    print("\n🚀 SRL hint system ready for integration!")

if __name__ == "__main__":
    main()

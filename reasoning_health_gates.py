#!/usr/bin/env python3
"""
Reasoning Health Gates - Anti-Theater Code Measures for Proof System
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RuleFamily(Enum):
    """Rule families for coverage tracking."""
    	S1 = "substantive"     # Core substantive rules (NSM primes)
	M2 = "mental"          # Mental predicate rules (NSM primes)
	L3 = "logical"         # Logical operator rules (NSM primes)
	T4 = "temporal"        # Temporal relation rules (NSM primes)
	C4 = "causal"          # Causal relation rules (NSM primes)
	S5 = "spatial"         # Spatial relation rules (NSM primes)
	Q6 = "quantifier"      # Quantifier rules (NSM primes)
	E7 = "evaluator"       # Evaluator rules (NSM primes)
	A8 = "action"          # Action rules (NSM primes)
	D9 = "descriptor"      # Descriptor rules (NSM primes)
	I10 = "intensifier"    # Intensifier rules (NSM primes)
	F11 = "final"          # Final prime rules (NSM primes)
	Q1 = "quantifier"      # Quantifier scope rules
    A1 = "aspect"          # Aspect rules  
    T1 = "temporal"        # Temporal rules
    CF1 = "counterfactual" # Counterfactual rules
    P1 = "deixis"          # Deictic/pragmatic rules


@dataclass
class ProofAnalysis:
    """Analysis of a single proof."""
    proof_id: str
    success: bool
    steps: int
    depth: int
    from_facts_only: bool  # True if goal matched raw facts without derivation
    rules_used: List[str]
    families_used: Set[RuleFamily]
    requires_family: Optional[RuleFamily]  # Family required for this goal type
    is_hard_goal: bool  # True if goal cannot be satisfied by raw facts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'proof_id': self.proof_id,
            'success': self.success,
            'steps': self.steps,
            'depth': self.depth,
            'from_facts_only': self.from_facts_only,
            'rules_used': self.rules_used,
            'families_used': [f.value for f in self.families_used] if self.families_used else [],
            'requires_family': self.requires_family.value if self.requires_family else None,
            'is_hard_goal': self.is_hard_goal
        }


@dataclass
class ReasoningHealthMetrics:
    """Health metrics for reasoning system."""
    total_proofs: int
    successful_proofs: int
    
    # Core anti-theater metrics
    derived_proof_rate: float  # % with ≥1 rule application
    depth_gt_zero_rate: float  # % with depth ≥ 1
    hard_goal_success_rate: float  # % success on derive-only targets
    
    # Family coverage metrics
    family_coverage: Dict[RuleFamily, float]  # % of successes using each family
    required_family_compliance: float  # % where required family was used
    
    # Quality gates
    passes_dpr_gate: bool  # DPR ≥ 70% (dev) / ≥ 60% (CI)
    passes_depth_gate: bool  # Depth>0 ≥ 80%
    passes_family_gates: bool  # Each enabled family ≥ 30%
    passes_hard_goal_gate: bool  # Hard goals ≥ 65%
    passes_required_family_gate: bool  # Required family compliance ≥ 80%
    
    overall_health: str  # HEALTHY, CONCERN, THEATER_CODE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_proofs': self.total_proofs,
            'successful_proofs': self.successful_proofs,
            'derived_proof_rate': self.derived_proof_rate,
            'depth_gt_zero_rate': self.depth_gt_zero_rate,
            'hard_goal_success_rate': self.hard_goal_success_rate,
            'family_coverage': {f.value: rate for f, rate in self.family_coverage.items()},
            'required_family_compliance': self.required_family_compliance,
            'passes_dpr_gate': self.passes_dpr_gate,
            'passes_depth_gate': self.passes_depth_gate,
            'passes_family_gates': self.passes_family_gates,
            'passes_hard_goal_gate': self.passes_hard_goal_gate,
            'passes_required_family_gate': self.passes_required_family_gate,
            'overall_health': self.overall_health
        }


class ReasoningHealthAnalyzer:
    """Analyzer to detect theater code and ensure real reasoning."""
    
    def __init__(self, is_ci: bool = False):
        """Initialize with environment-specific thresholds."""
        self.is_ci = is_ci
        
        # Thresholds (stricter for dev, more lenient for CI)
        self.dpr_threshold = 0.60 if is_ci else 0.70
        self.depth_threshold = 0.80
        self.family_coverage_threshold = 0.30
        self.hard_goal_threshold = 0.65
        self.required_family_threshold = 0.80
        
        		# Enabled rule families
		self.enabled_families = {RuleFamily.S1, RuleFamily.M2, RuleFamily.L3, RuleFamily.T4, RuleFamily.C4, RuleFamily.S5, RuleFamily.Q6, RuleFamily.E7, RuleFamily.A8, RuleFamily.D9, RuleFamily.I10, RuleFamily.F11, RuleFamily.Q1, RuleFamily.A1}
    
    def analyze_proofs(self, proofs: List[ProofAnalysis]) -> ReasoningHealthMetrics:
        """Analyze proofs and compute health metrics."""
        
        if not proofs:
            return self._empty_metrics()
        
        total_proofs = len(proofs)
        successful_proofs = [p for p in proofs if p.success]
        successful_count = len(successful_proofs)
        
        if successful_count == 0:
            return self._empty_metrics()
        
        # Calculate core metrics on hard-goal subset
        hard_proofs = [p for p in proofs if p.is_hard_goal]
        hard_successes = [p for p in hard_proofs if p.success]
        
        # Derived proofs among hard successes
        derived_hard = [p for p in hard_successes if not p.from_facts_only]
        depth_gt_zero_hard = [p for p in hard_successes if p.depth > 0]
        
        derived_proof_rate = len(derived_hard) / len(hard_successes) if hard_successes else 0.0
        depth_gt_zero_rate = len(depth_gt_zero_hard) / len(hard_successes) if hard_successes else 0.0
        # Hard goal overall success rate
        hard_goal_success_rate = len(hard_successes) / len(hard_proofs) if hard_proofs else 0.0
        
        # Calculate family coverage
        family_coverage = {}
        for family in self.enabled_families:
            family_users = [p for p in successful_proofs if family in p.families_used]
            family_coverage[family] = len(family_users) / successful_count
        
        # Calculate required family compliance
        required_family_proofs = [p for p in successful_proofs if p.requires_family]
        compliant_proofs = [p for p in required_family_proofs if p.requires_family in p.families_used]
        required_family_compliance = len(compliant_proofs) / len(required_family_proofs) if required_family_proofs else 1.0
        
        # Check gates
        passes_dpr_gate = derived_proof_rate >= self.dpr_threshold
        passes_depth_gate = depth_gt_zero_rate >= self.depth_threshold
        passes_family_gates = all(rate >= self.family_coverage_threshold for rate in family_coverage.values())
        passes_hard_goal_gate = hard_goal_success_rate >= self.hard_goal_threshold
        passes_required_family_gate = required_family_compliance >= self.required_family_threshold
        
        # Determine overall health
        all_gates_pass = all([
            passes_dpr_gate,
            passes_depth_gate, 
            passes_family_gates,
            passes_hard_goal_gate,
            passes_required_family_gate
        ])
        
        if all_gates_pass:
            overall_health = "HEALTHY"
        elif derived_proof_rate < 0.30 or depth_gt_zero_rate < 0.50:
            overall_health = "THEATER_CODE"
        else:
            overall_health = "CONCERN"
        
        return ReasoningHealthMetrics(
            total_proofs=total_proofs,
            successful_proofs=successful_count,
            derived_proof_rate=derived_proof_rate,
            depth_gt_zero_rate=depth_gt_zero_rate,
            hard_goal_success_rate=hard_goal_success_rate,
            family_coverage=family_coverage,
            required_family_compliance=required_family_compliance,
            passes_dpr_gate=passes_dpr_gate,
            passes_depth_gate=passes_depth_gate,
            passes_family_gates=passes_family_gates,
            passes_hard_goal_gate=passes_hard_goal_gate,
            passes_required_family_gate=passes_required_family_gate,
            overall_health=overall_health
        )
    
    def _empty_metrics(self) -> ReasoningHealthMetrics:
        """Return empty metrics for edge cases."""
        return ReasoningHealthMetrics(
            total_proofs=0,
            successful_proofs=0,
            derived_proof_rate=0.0,
            depth_gt_zero_rate=0.0,
            hard_goal_success_rate=0.0,
            family_coverage={f: 0.0 for f in self.enabled_families},
            required_family_compliance=0.0,
            passes_dpr_gate=False,
            passes_depth_gate=False,
            passes_family_gates=False,
            passes_hard_goal_gate=False,
            passes_required_family_gate=False,
            overall_health="THEATER_CODE"
        )
    
    def generate_hard_goals(self) -> List[Dict[str, Any]]:
        """Generate hard goals that require derivation (cannot be satisfied by raw facts)."""
        return [
            # Aspect-derived goals
            {
                'goal': 'PAST(finish) ∧ close(now,finish)',
                'requires_family': 'aspect',
                'description': 'Must derive from RECENT_PAST(finish) via A1 rule',
                'setup_facts': ['RECENT_PAST(finish)'],
                'is_hard_goal': True
            },
            {
                'goal': 'DURING(work, now−3h..now)',
                'requires_family': 'aspect',
                'description': 'Must derive from ONGOING_FOR(work, 3h) via A1 rule',
                'setup_facts': ['ONGOING_FOR(work, 3h)'],
                'is_hard_goal': True
            },
            {
                'goal': '¬finish ∧ near(finish)',
                'requires_family': 'aspect',
                'description': 'Must derive from ALMOST_DO(finish) via A1 rule',
                'setup_facts': ['ALMOST_DO(finish)'],
                'is_hard_goal': True
            },
            
            # Quantifier-derived goals
            {
                'goal': 'NOT EXISTS[x] study(x)',
                'requires_family': 'quantifier',
                'description': 'Must derive from NARROW(¬∀) via Q1 rule',
                'setup_facts': ['NARROW(¬∀)'],
                'is_hard_goal': True
            },
            {
                'goal': 'ALL[x] NOT study(x)',
                'requires_family': 'quantifier',
                'description': 'Must derive from WIDE(∀¬) via Q1 rule',
                'setup_facts': ['WIDE(∀¬)'],
                'is_hard_goal': True
            },
            
            # Easy goals (for contrast - these should succeed without derivation)
            {
                'goal': 'RECENT_PAST(finish)',
                'requires_family': None,
                'description': 'Raw fact match - should succeed immediately',
                'setup_facts': ['RECENT_PAST(finish)'],
                'is_hard_goal': False
            }
        ]
    
    def check_ci_gates(self, metrics: ReasoningHealthMetrics) -> Dict[str, Any]:
        """Check CI gates and generate failure reasons."""
        failures = []
        
        if not metrics.passes_dpr_gate:
            failures.append(f"DPR too low: {metrics.derived_proof_rate:.1%} < {self.dpr_threshold:.1%}")
        
        if not metrics.passes_depth_gate:
            failures.append(f"Depth>0 rate too low: {metrics.depth_gt_zero_rate:.1%} < {self.depth_threshold:.1%}")
        
        if not metrics.passes_family_gates:
            for family, rate in metrics.family_coverage.items():
                if rate < self.family_coverage_threshold:
                    failures.append(f"{family.value} coverage too low: {rate:.1%} < {self.family_coverage_threshold:.1%}")
        
        if not metrics.passes_hard_goal_gate:
            failures.append(f"Hard goal success too low: {metrics.hard_goal_success_rate:.1%} < {self.hard_goal_threshold:.1%}")
        
        if not metrics.passes_required_family_gate:
            failures.append(f"Required family compliance too low: {metrics.required_family_compliance:.1%} < {self.required_family_threshold:.1%}")
        
        return {
            'passes_all_gates': len(failures) == 0,
            'failures': failures,
            'overall_health': metrics.overall_health,
            'recommendation': self._get_recommendation(metrics)
        }
    
    def _get_recommendation(self, metrics: ReasoningHealthMetrics) -> str:
        """Get recommendation based on health metrics."""
        if metrics.overall_health == "THEATER_CODE":
            return "CRITICAL: Theater code detected. System is matching facts without reasoning. Implement proper derivation rules."
        elif metrics.overall_health == "CONCERN":
            return "WARNING: Some reasoning quality issues detected. Review failing gates and improve rule coverage."
        else:
            return "GOOD: All reasoning health gates pass. System demonstrates genuine inference."


def main():
    """Demonstrate reasoning health analysis."""
    logger.info("Starting reasoning health analysis demonstration...")
    
    analyzer = ReasoningHealthAnalyzer(is_ci=False)
    
    # Generate example proof analyses (simulating current theater code issue)
    proofs = [
        # Theater code examples (fact matching without derivation)
        ProofAnalysis(
            proof_id="proof_1",
            success=True,
            steps=0,  # No derivation steps
            depth=0,  # No inference depth
            from_facts_only=True,  # Just matched facts
            rules_used=[],  # No rules applied
            families_used=set(),  # No families used
            requires_family=RuleFamily.A1,  # Should have used A1
            is_hard_goal=False
        ),
        ProofAnalysis(
            proof_id="proof_2", 
            success=True,
            steps=0,
            depth=0,
            from_facts_only=True,
            rules_used=[],
            families_used=set(),
            requires_family=RuleFamily.Q1,
            is_hard_goal=False
        ),
        
        # Good reasoning example
        ProofAnalysis(
            proof_id="proof_3",
            success=True,
            steps=2,
            depth=2,
            from_facts_only=False,
            rules_used=["A1_RECENT_PAST_TO_PAST_CLOSE"],
            families_used={RuleFamily.A1},
            requires_family=RuleFamily.A1,
            is_hard_goal=True
        )
    ]
    
    # Analyze proofs
    metrics = analyzer.analyze_proofs(proofs)
    
    print("\n" + "="*80)
    print("REASONING HEALTH ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nCore Metrics:")
    print(f"  Total Proofs: {metrics.total_proofs}")
    print(f"  Successful Proofs: {metrics.successful_proofs}")
    print(f"  Derived Proof Rate: {metrics.derived_proof_rate:.1%}")
    print(f"  Depth>0 Rate: {metrics.depth_gt_zero_rate:.1%}")
    print(f"  Hard Goal Success Rate: {metrics.hard_goal_success_rate:.1%}")
    
    print(f"\nFamily Coverage:")
    for family, rate in metrics.family_coverage.items():
        status = "✅" if rate >= analyzer.family_coverage_threshold else "❌"
        print(f"  {family.value}: {rate:.1%} {status}")
    
    print(f"\nRequired Family Compliance: {metrics.required_family_compliance:.1%}")
    
    print(f"\nGate Status:")
    print(f"  DPR Gate: {'✅' if metrics.passes_dpr_gate else '❌'}")
    print(f"  Depth Gate: {'✅' if metrics.passes_depth_gate else '❌'}")
    print(f"  Family Gates: {'✅' if metrics.passes_family_gates else '❌'}")
    print(f"  Hard Goal Gate: {'✅' if metrics.passes_hard_goal_gate else '❌'}")
    print(f"  Required Family Gate: {'✅' if metrics.passes_required_family_gate else '❌'}")
    
    print(f"\nOverall Health: {metrics.overall_health}")
    
    # Check CI gates
    ci_check = analyzer.check_ci_gates(metrics)
    print(f"\nCI Gate Check:")
    print(f"  Passes All Gates: {'✅' if ci_check['passes_all_gates'] else '❌'}")
    
    if ci_check['failures']:
        print(f"  Failures:")
        for failure in ci_check['failures']:
            print(f"    - {failure}")
    
    print(f"\nRecommendation: {ci_check['recommendation']}")
    
    # Generate hard goals
    hard_goals = analyzer.generate_hard_goals()
    print(f"\nGenerated Hard Goals:")
    for i, goal in enumerate(hard_goals):
        print(f"  {i+1}. {goal['description']}")
        print(f"     Goal: {goal['goal']}")
        print(f"     Setup: {goal['setup_facts']}")
        print(f"     Requires: {goal['requires_family'] if goal['requires_family'] else 'None'}")
        print(f"     Hard: {goal['is_hard_goal']}")
    
    # Save results
    output_path = Path("data/reasoning_health_analysis.json")
    output_path.parent.mkdir(exist_ok=True)
    
    results = {
        'metrics': metrics.to_dict(),
        'ci_check': ci_check,
        'hard_goals': hard_goals,
        'proof_analyses': [p.to_dict() for p in proofs]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Reasoning health analysis saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Reasoning health analysis completed!")
    print("="*80)


if __name__ == "__main__":
    main()

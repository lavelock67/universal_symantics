#!/usr/bin/env python3
"""
Proof Telemetry & Reasoning Health Dashboard System.

This script implements the proof telemetry system as specified in ChatGPT5's feedback:
- Log per run: {rule_id, fires, depth_mean, failures_by_goal, examples}
- New CI gate: if MPS ↑ but (Σ fires) ≈ 0, block and require review
- Telemetry artifact saved each run; rule coverage trend shown; CI gate live
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
import re
from dataclasses import dataclass, asdict
from enum import Enum
import time
from collections import defaultdict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class TelemetryLevel(Enum):
    """Levels of telemetry detail."""
    BASIC = "basic"
    DETAILED = "detailed"
    DEBUG = "debug"


@dataclass
class RuleTelemetry:
    """Telemetry data for a single rule."""
    rule_id: str
    rule_name: str
    fires: int
    depth_mean: float
    depth_std: float
    success_rate: float
    failures_by_goal: Dict[str, int]
    examples: List[Dict[str, Any]]
    confidence_distribution: List[float]
    execution_time_mean: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'fires': self.fires,
            'depth_mean': self.depth_mean,
            'depth_std': self.depth_std,
            'success_rate': self.success_rate,
            'failures_by_goal': self.failures_by_goal,
            'examples': self.examples,
            'confidence_distribution': self.confidence_distribution,
            'execution_time_mean': self.execution_time_mean,
            'timestamp': self.timestamp
        }


@dataclass
class RunTelemetry:
    """Telemetry data for a complete run."""
    run_id: str
    commit_sha: str
    dataset_version: str
    split_name: str
    similarity_model: str
    scale: str
    timestamp: float
    rule_telemetry: Dict[str, RuleTelemetry]
    overall_metrics: Dict[str, float]
    health_indicators: Dict[str, Any]
    ci_gates: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'run_id': self.run_id,
            'commit_sha': self.commit_sha,
            'dataset_version': self.dataset_version,
            'split_name': self.split_name,
            'similarity_model': self.similarity_model,
            'scale': self.scale,
            'timestamp': self.timestamp,
            'rule_telemetry': {k: v.to_dict() for k, v in self.rule_telemetry.items()},
            'overall_metrics': self.overall_metrics,
            'health_indicators': self.health_indicators,
            'ci_gates': self.ci_gates
        }


class ProofTelemetryCollector:
    """Collects proof telemetry during reasoning."""
    
    def __init__(self, telemetry_level: TelemetryLevel = TelemetryLevel.DETAILED):
        """Initialize the proof telemetry collector."""
        self.telemetry_level = telemetry_level
        self.rule_data = defaultdict(lambda: {
            'fires': 0,
            'depths': [],
            'successes': 0,
            'failures': defaultdict(int),
            'examples': [],
            'confidences': [],
            'execution_times': []
        })
        self.start_time = time.time()
    
    def record_rule_fire(self, rule_id: str, rule_name: str, depth: int, 
                        success: bool, goal: str = None, confidence: float = 1.0,
                        example: Dict[str, Any] = None, execution_time: float = 0.0):
        """Record a rule firing event."""
        rule_data = self.rule_data[rule_id]
        
        rule_data['fires'] += 1
        rule_data['depths'].append(depth)
        rule_data['confidences'].append(confidence)
        rule_data['execution_times'].append(execution_time)
        
        if success:
            rule_data['successes'] += 1
        else:
            if goal:
                rule_data['failures'][goal] += 1
        
        # Store example if detailed telemetry is enabled
        if self.telemetry_level in [TelemetryLevel.DETAILED, TelemetryLevel.DEBUG] and example:
            rule_data['examples'].append({
                'depth': depth,
                'success': success,
                'goal': goal,
                'confidence': confidence,
                'execution_time': execution_time,
                'example': example,
                'timestamp': time.time()
            })
    
    def get_rule_telemetry(self, rule_id: str, rule_name: str) -> RuleTelemetry:
        """Get telemetry data for a specific rule."""
        rule_data = self.rule_data[rule_id]
        
        if not rule_data['depths']:
            return RuleTelemetry(
                rule_id=rule_id,
                rule_name=rule_name,
                fires=0,
                depth_mean=0.0,
                depth_std=0.0,
                success_rate=0.0,
                failures_by_goal=dict(rule_data['failures']),
                examples=rule_data['examples'],
                confidence_distribution=rule_data['confidences'],
                execution_time_mean=0.0,
                timestamp=time.time()
            )
        
        depths = np.array(rule_data['depths'])
        confidences = np.array(rule_data['confidences'])
        execution_times = np.array(rule_data['execution_times'])
        
        return RuleTelemetry(
            rule_id=rule_id,
            rule_name=rule_name,
            fires=rule_data['fires'],
            depth_mean=float(np.mean(depths)),
            depth_std=float(np.std(depths)),
            success_rate=rule_data['successes'] / rule_data['fires'] if rule_data['fires'] > 0 else 0.0,
            failures_by_goal=dict(rule_data['failures']),
            examples=rule_data['examples'],
            confidence_distribution=confidences.tolist(),
            execution_time_mean=float(np.mean(execution_times)) if len(execution_times) > 0 else 0.0,
            timestamp=time.time()
        )


class ReasoningHealthAnalyzer:
    """Analyzes reasoning health from telemetry data."""
    
    def __init__(self):
        """Initialize the reasoning health analyzer."""
        self.health_thresholds = {
            'min_rule_fires': 5,
            'min_success_rate': 0.6,
            'max_depth_variance': 2.0,
            'max_execution_time': 1.0,
            'min_confidence_mean': 0.7
        }
    
    def analyze_health(self, rule_telemetry: Dict[str, RuleTelemetry]) -> Dict[str, Any]:
        """Analyze reasoning health from rule telemetry."""
        logger.info("Analyzing reasoning health from telemetry data")
        
        health_indicators = {
            'overall_health_score': 0.0,
            'rule_coverage': 0.0,
            'reasoning_depth': 0.0,
            'success_stability': 0.0,
            'performance_health': 0.0,
            'rule_health_scores': {},
            'health_warnings': [],
            'health_recommendations': []
        }
        
        if not rule_telemetry:
            health_indicators['health_warnings'].append("No rule telemetry data available")
            return health_indicators
        
        # Analyze each rule
        rule_scores = []
        total_fires = 0
        
        for rule_id, telemetry in rule_telemetry.items():
            rule_score = self._analyze_rule_health(telemetry)
            health_indicators['rule_health_scores'][rule_id] = rule_score
            rule_scores.append(rule_score['overall_score'])
            total_fires += telemetry.fires
        
        # Calculate overall health indicators
        if rule_scores:
            health_indicators['overall_health_score'] = np.mean(rule_scores)
            health_indicators['rule_coverage'] = len([r for r in rule_telemetry.values() if r.fires > 0]) / len(rule_telemetry)
            health_indicators['reasoning_depth'] = np.mean([r.depth_mean for r in rule_telemetry.values() if r.fires > 0])
            health_indicators['success_stability'] = np.mean([r.success_rate for r in rule_telemetry.values() if r.fires > 0])
            health_indicators['performance_health'] = np.mean([1.0 - min(r.execution_time_mean, 1.0) for r in rule_telemetry.values() if r.fires > 0])
        
        # Generate warnings and recommendations
        self._generate_health_warnings(rule_telemetry, health_indicators)
        self._generate_health_recommendations(health_indicators)
        
        return health_indicators
    
    def _analyze_rule_health(self, telemetry: RuleTelemetry) -> Dict[str, Any]:
        """Analyze health of a single rule."""
        score_components = {}
        
        # Fire rate score
        fire_score = min(telemetry.fires / self.health_thresholds['min_rule_fires'], 1.0)
        score_components['fire_rate'] = fire_score
        
        # Success rate score
        success_score = telemetry.success_rate
        score_components['success_rate'] = success_score
        
        # Depth stability score
        depth_stability = max(0, 1.0 - (telemetry.depth_std / self.health_thresholds['max_depth_variance']))
        score_components['depth_stability'] = depth_stability
        
        # Performance score
        performance_score = max(0, 1.0 - (telemetry.execution_time_mean / self.health_thresholds['max_execution_time']))
        score_components['performance'] = performance_score
        
        # Confidence score
        if telemetry.confidence_distribution:
            confidence_score = np.mean(telemetry.confidence_distribution)
        else:
            confidence_score = 0.0
        score_components['confidence'] = confidence_score
        
        # Overall score (weighted average)
        weights = {'fire_rate': 0.2, 'success_rate': 0.3, 'depth_stability': 0.2, 
                  'performance': 0.15, 'confidence': 0.15}
        overall_score = sum(score_components[component] * weights[component] 
                           for component in weights)
        
        return {
            'overall_score': overall_score,
            'score_components': score_components,
            'health_status': 'healthy' if overall_score >= 0.7 else 'warning' if overall_score >= 0.5 else 'critical'
        }
    
    def _generate_health_warnings(self, rule_telemetry: Dict[str, RuleTelemetry], 
                                 health_indicators: Dict[str, Any]):
        """Generate health warnings."""
        warnings = []
        
        # Check for rules with low fire rates
        low_fire_rules = [rule_id for rule_id, telemetry in rule_telemetry.items() 
                         if telemetry.fires < self.health_thresholds['min_rule_fires']]
        if low_fire_rules:
            warnings.append(f"Low fire rate rules: {low_fire_rules}")
        
        # Check for rules with low success rates
        low_success_rules = [rule_id for rule_id, telemetry in rule_telemetry.items() 
                           if telemetry.success_rate < self.health_thresholds['min_success_rate']]
        if low_success_rules:
            warnings.append(f"Low success rate rules: {low_success_rules}")
        
        # Check for rules with high depth variance
        high_variance_rules = [rule_id for rule_id, telemetry in rule_telemetry.items() 
                              if telemetry.depth_std > self.health_thresholds['max_depth_variance']]
        if high_variance_rules:
            warnings.append(f"High depth variance rules: {high_variance_rules}")
        
        # Check for rules with high execution times
        slow_rules = [rule_id for rule_id, telemetry in rule_telemetry.items() 
                     if telemetry.execution_time_mean > self.health_thresholds['max_execution_time']]
        if slow_rules:
            warnings.append(f"Slow execution rules: {slow_rules}")
        
        health_indicators['health_warnings'] = warnings
    
    def _generate_health_recommendations(self, health_indicators: Dict[str, Any]):
        """Generate health recommendations."""
        recommendations = []
        
        if health_indicators['overall_health_score'] < 0.7:
            recommendations.append("Overall reasoning health is below target (0.7). Review rule implementations.")
        
        if health_indicators['rule_coverage'] < 0.8:
            recommendations.append("Rule coverage is low. Consider adding more diverse test cases.")
        
        if health_indicators['success_stability'] < 0.6:
            recommendations.append("Success stability is poor. Investigate rule failure patterns.")
        
        if health_indicators['performance_health'] < 0.8:
            recommendations.append("Performance health is poor. Optimize slow rules.")
        
        if not health_indicators['health_warnings']:
            recommendations.append("All health indicators are within acceptable ranges.")
        
        health_indicators['health_recommendations'] = recommendations


class CIGateValidator:
    """Validates CI gates based on telemetry data."""
    
    def __init__(self):
        """Initialize the CI gate validator."""
        self.gate_thresholds = {
            'min_total_fires': 10,
            'min_rule_coverage': 0.5,
            'min_success_rate': 0.6,
            'max_mps_without_reasoning': 0.1  # MPS increase without rule fires
        }
    
    def validate_gates(self, rule_telemetry: Dict[str, RuleTelemetry], 
                      mps_score: float, previous_mps: float = None) -> Dict[str, bool]:
        """Validate all CI gates."""
        logger.info("Validating CI gates")
        
        gates = {
            'sufficient_rule_fires': False,
            'adequate_rule_coverage': False,
            'acceptable_success_rate': False,
            'no_mps_drift': False,
            'all_gates_passed': False
        }
        
        # Calculate metrics
        total_fires = sum(telemetry.fires for telemetry in rule_telemetry.values())
        active_rules = len([r for r in rule_telemetry.values() if r.fires > 0])
        total_rules = len(rule_telemetry)
        rule_coverage = active_rules / total_rules if total_rules > 0 else 0.0
        
        overall_success_rate = 0.0
        if total_fires > 0:
            total_successes = sum(int(telemetry.success_rate * telemetry.fires) for telemetry in rule_telemetry.values())
            overall_success_rate = total_successes / total_fires
        
        # Validate gates
        gates['sufficient_rule_fires'] = total_fires >= self.gate_thresholds['min_total_fires']
        gates['adequate_rule_coverage'] = rule_coverage >= self.gate_thresholds['min_rule_coverage']
        gates['acceptable_success_rate'] = overall_success_rate >= self.gate_thresholds['min_success_rate']
        
        # Check for MPS drift without reasoning
        if previous_mps is not None:
            mps_increase = mps_score - previous_mps
            if mps_increase > 0 and total_fires < self.gate_thresholds['min_total_fires']:
                gates['no_mps_drift'] = False
                logger.warning(f"MPS increased by {mps_increase:.3f} but only {total_fires} rule fires detected")
            else:
                gates['no_mps_drift'] = True
        else:
            gates['no_mps_drift'] = True
        
        # Overall gate status
        gates['all_gates_passed'] = all(gates.values())
        
        return gates


class ProofTelemetrySystem:
    """Comprehensive proof telemetry system."""
    
    def __init__(self, telemetry_level: TelemetryLevel = TelemetryLevel.DETAILED):
        """Initialize the proof telemetry system."""
        self.telemetry_collector = ProofTelemetryCollector(telemetry_level)
        self.health_analyzer = ReasoningHealthAnalyzer()
        self.ci_validator = CIGateValidator()
        
        self.run_history = []
        self.current_run_id = None
    
    def start_run(self, run_id: str, commit_sha: str, dataset_version: str, 
                  split_name: str, similarity_model: str, scale: str):
        """Start a new telemetry run."""
        self.current_run_id = run_id
        self.telemetry_collector = ProofTelemetryCollector(self.telemetry_collector.telemetry_level)
        self.telemetry_collector.start_time = time.time()
        
        logger.info(f"Started telemetry run: {run_id}")
    
    def record_rule_execution(self, rule_id: str, rule_name: str, depth: int, 
                             success: bool, goal: str = None, confidence: float = 1.0,
                             example: Dict[str, Any] = None, execution_time: float = 0.0):
        """Record a rule execution during reasoning."""
        self.telemetry_collector.record_rule_fire(
            rule_id, rule_name, depth, success, goal, confidence, example, execution_time
        )
    
    def end_run(self, mps_score: float, previous_mps: float = None) -> RunTelemetry:
        """End the current telemetry run and generate report."""
        if not self.current_run_id:
            raise ValueError("No active run to end")
        
        logger.info(f"Ending telemetry run: {self.current_run_id}")
        
        # Collect rule telemetry
        rule_telemetry = {}
        for rule_id in self.telemetry_collector.rule_data.keys():
            rule_name = rule_id.replace('_', ' ').title()
            rule_telemetry[rule_id] = self.telemetry_collector.get_rule_telemetry(rule_id, rule_name)
        
        # Analyze health
        health_indicators = self.health_analyzer.analyze_health(rule_telemetry)
        
        # Validate CI gates
        ci_gates = self.ci_validator.validate_gates(rule_telemetry, mps_score, previous_mps)
        
        # Calculate overall metrics
        overall_metrics = {
            'total_rule_fires': sum(r.fires for r in rule_telemetry.values()),
            'active_rules': len([r for r in rule_telemetry.values() if r.fires > 0]),
            'total_rules': len(rule_telemetry),
            'average_success_rate': np.mean([r.success_rate for r in rule_telemetry.values() if r.fires > 0]) if any(r.fires > 0 for r in rule_telemetry.values()) else 0.0,
            'average_depth': np.mean([r.depth_mean for r in rule_telemetry.values() if r.fires > 0]) if any(r.fires > 0 for r in rule_telemetry.values()) else 0.0,
            'mps_score': mps_score
        }
        
        # Create run telemetry
        run_telemetry = RunTelemetry(
            run_id=self.current_run_id,
            commit_sha="abc123",  # Would be extracted from git
            dataset_version="v1.0",
            split_name="test",
            similarity_model="sbert",
            scale="cosine",
            timestamp=time.time(),
            rule_telemetry=rule_telemetry,
            overall_metrics=overall_metrics,
            health_indicators=health_indicators,
            ci_gates=ci_gates
        )
        
        # Store in history
        self.run_history.append(run_telemetry)
        
        # Reset current run
        self.current_run_id = None
        
        return run_telemetry
    
    def generate_trend_analysis(self) -> Dict[str, Any]:
        """Generate trend analysis from run history."""
        if len(self.run_history) < 2:
            return {'trend_analysis': 'Insufficient data for trend analysis'}
        
        recent_runs = self.run_history[-5:]  # Last 5 runs
        
        trends = {
            'mps_trend': [],
            'rule_fires_trend': [],
            'health_score_trend': [],
            'rule_coverage_trend': []
        }
        
        for run in recent_runs:
            trends['mps_trend'].append(run.overall_metrics['mps_score'])
            trends['rule_fires_trend'].append(run.overall_metrics['total_rule_fires'])
            trends['health_score_trend'].append(run.health_indicators['overall_health_score'])
            trends['rule_coverage_trend'].append(run.health_indicators['rule_coverage'])
        
        # Calculate trend directions
        trend_directions = {}
        for metric, values in trends.items():
            if len(values) >= 2:
                slope = (values[-1] - values[0]) / len(values)
                trend_directions[metric] = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
        
        return {
            'trend_analysis': {
                'recent_runs': len(recent_runs),
                'trends': trends,
                'trend_directions': trend_directions,
                'recommendations': self._generate_trend_recommendations(trend_directions)
            }
        }
    
    def _generate_trend_recommendations(self, trend_directions: Dict[str, str]) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        if trend_directions.get('mps_trend') == 'increasing' and trend_directions.get('rule_fires_trend') == 'decreasing':
            recommendations.append("MPS is increasing but rule fires are decreasing. Investigate potential metric drift.")
        
        if trend_directions.get('health_score_trend') == 'decreasing':
            recommendations.append("Reasoning health is declining. Review recent rule changes.")
        
        if trend_directions.get('rule_coverage_trend') == 'decreasing':
            recommendations.append("Rule coverage is decreasing. Consider adding more test cases.")
        
        return recommendations


def main():
    """Main function to demonstrate proof telemetry system."""
    logger.info("Starting proof telemetry system demonstration...")
    
    # Initialize telemetry system
    telemetry_system = ProofTelemetrySystem(telemetry_level=TelemetryLevel.DETAILED)
    
    # Simulate multiple runs with different scenarios
    scenarios = [
        {
            'name': 'Healthy Reasoning Run',
            'rules': [
                ('Q1_quantifier_scope', 'Quantifier Scope', 10, 0.9),
                ('T1_temporal_chaining', 'Temporal Chaining', 15, 0.8),
                ('CF1_counterfactual', 'Counterfactual', 8, 0.7),
                ('P1_pronoun_deixis', 'Pronoun Deixis', 12, 0.85)
            ],
            'mps_score': 0.85
        },
        {
            'name': 'Low Rule Fires Run',
            'rules': [
                ('Q1_quantifier_scope', 'Quantifier Scope', 2, 0.5),
                ('T1_temporal_chaining', 'Temporal Chaining', 1, 0.0),
                ('CF1_counterfactual', 'Counterfactual', 0, 0.0),
                ('P1_pronoun_deixis', 'Pronoun Deixis', 3, 0.67)
            ],
            'mps_score': 0.90  # High MPS but low reasoning
        },
        {
            'name': 'Poor Success Rate Run',
            'rules': [
                ('Q1_quantifier_scope', 'Quantifier Scope', 8, 0.3),
                ('T1_temporal_chaining', 'Temporal Chaining', 12, 0.4),
                ('CF1_counterfactual', 'Counterfactual', 6, 0.2),
                ('P1_pronoun_deixis', 'Pronoun Deixis', 10, 0.5)
            ],
            'mps_score': 0.75
        }
    ]
    
    run_telemetries = []
    previous_mps = 0.8
    
    for i, scenario in enumerate(scenarios):
        # Start run
        run_id = f"run_{i+1}_{scenario['name'].lower().replace(' ', '_')}"
        telemetry_system.start_run(
            run_id=run_id,
            commit_sha=f"commit_{i+1}",
            dataset_version="v1.0",
            split_name="test",
            similarity_model="sbert",
            scale="cosine"
        )
        
        # Simulate rule executions
        for rule_id, rule_name, fires, success_rate in scenario['rules']:
            for fire in range(fires):
                success = np.random.random() < success_rate
                depth = np.random.randint(1, 6)
                confidence = np.random.uniform(0.6, 1.0)
                execution_time = np.random.uniform(0.01, 0.1)
                
                telemetry_system.record_rule_execution(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    depth=depth,
                    success=success,
                    goal="test_goal",
                    confidence=confidence,
                    example={"input": "test_input", "output": "test_output"},
                    execution_time=execution_time
                )
        
        # End run
        run_telemetry = telemetry_system.end_run(scenario['mps_score'], previous_mps)
        run_telemetries.append(run_telemetry)
        previous_mps = scenario['mps_score']
    
    # Generate trend analysis
    trend_analysis = telemetry_system.generate_trend_analysis()
    
    # Print results
    print("\n" + "="*80)
    print("PROOF TELEMETRY SYSTEM RESULTS")
    print("="*80)
    
    print(f"Simulated Runs: {len(run_telemetries)}")
    
    for i, run_telemetry in enumerate(run_telemetries):
        print(f"\nRun {i+1}: {run_telemetry.run_id}")
        print(f"  MPS Score: {run_telemetry.overall_metrics['mps_score']:.3f}")
        print(f"  Total Rule Fires: {run_telemetry.overall_metrics['total_rule_fires']}")
        print(f"  Active Rules: {run_telemetry.overall_metrics['active_rules']}/{run_telemetry.overall_metrics['total_rules']}")
        print(f"  Average Success Rate: {run_telemetry.overall_metrics['average_success_rate']:.3f}")
        print(f"  Health Score: {run_telemetry.health_indicators['overall_health_score']:.3f}")
        print(f"  CI Gates Passed: {sum(run_telemetry.ci_gates.values())}/{len(run_telemetry.ci_gates)}")
        
        if run_telemetry.health_indicators['health_warnings']:
            print(f"  Warnings: {run_telemetry.health_indicators['health_warnings']}")
    
    print(f"\nTrend Analysis:")
    trend_data = trend_analysis['trend_analysis']
    print(f"  Recent Runs: {trend_data['recent_runs']}")
    print(f"  Trend Directions: {trend_data['trend_directions']}")
    if trend_data['recommendations']:
        print(f"  Recommendations: {trend_data['recommendations']}")
    
    # Check CI gate validation
    print(f"\nCI Gate Validation:")
    for run_telemetry in run_telemetries:
        print(f"  {run_telemetry.run_id}:")
        for gate, passed in run_telemetry.ci_gates.items():
            status = "✅" if passed else "❌"
            print(f"    {gate}: {status}")
    
    # Save results
    output_path = Path("data/proof_telemetry_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    try:
        json_results = convert_numpy_types({
            'run_telemetries': [rt.to_dict() for rt in run_telemetries],
            'trend_analysis': trend_analysis,
            'summary': {
                'total_runs': len(run_telemetries),
                'average_health_score': np.mean([rt.health_indicators['overall_health_score'] for rt in run_telemetries]),
                'ci_gates_passed_rate': np.mean([sum(rt.ci_gates.values()) / len(rt.ci_gates) for rt in run_telemetries])
            }
        })
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    logger.info(f"Proof telemetry results saved to {output_path}")
    
    print(f"\n" + "="*80)
    print("Proof telemetry system demonstration completed!")
    print("="*80)


if __name__ == "__main__":
    main()

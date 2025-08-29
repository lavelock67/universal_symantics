"""
Fail Triage Tool
Analyze and categorize evaluation failures to identify patterns and prioritize fixes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass

@dataclass
class FailureAnalysis:
    """Analysis of evaluation failures"""
    suite_name: str
    total_failures: int
    failure_categories: Dict[str, int]
    top_failing_ids: List[Dict]
    prime_failures: Dict[str, int]
    pattern_failures: Dict[str, int]
    recommendations: List[str]

class FailTriage:
    """Analyze evaluation failures and provide actionable insights"""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.logger = logging.getLogger(__name__)
        
        # Failure categories
        self.failure_categories = {
            'missing_prime': 'Expected prime not detected',
            'wrong_scope': 'Scope incorrectly attached',
            'adapter_violation': 'Cultural adapter changed factual content',
            'glossary_violation': 'Glossary term not preserved',
            'timeout': 'Translation timed out',
            'error': 'System error occurred',
            'low_confidence': 'Low confidence score',
            'graph_f1_low': 'Graph-F1 score below threshold',
            'scope_accuracy_low': 'Scope accuracy below threshold'
        }
    
    def load_evaluation_report(self, report_file: str = "summary.json") -> Dict:
        """Load evaluation report from JSON file"""
        report_path = self.reports_dir / report_file
        
        if not report_path.exists():
            raise FileNotFoundError(f"Report file not found: {report_path}")
        
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_failures(self, report: Dict) -> Dict[str, FailureAnalysis]:
        """Analyze failures from evaluation report"""
        analyses = {}
        
        if 'failures' not in report:
            self.logger.warning("No failures found in report")
            return analyses
        
        failures = report['failures']
        
        # Group failures by suite
        suite_failures = defaultdict(list)
        for failure in failures:
            suite = failure.get('suite', 'unknown')
            suite_failures[suite].append(failure)
        
        # Analyze each suite
        for suite_name, suite_failures_list in suite_failures.items():
            analysis = self._analyze_suite_failures(suite_name, suite_failures_list)
            analyses[suite_name] = analysis
        
        return analyses
    
    def _analyze_suite_failures(self, suite_name: str, failures: List[Dict]) -> FailureAnalysis:
        """Analyze failures for a specific suite"""
        total_failures = len(failures)
        
        # Categorize failures
        failure_categories = Counter()
        prime_failures = Counter()
        pattern_failures = Counter()
        
        # Analyze each failure
        for failure in failures:
            reason = failure.get('reason', 'unknown')
            test_id = failure.get('test_id', 'unknown')
            
            # Categorize by reason
            category = self._categorize_failure(reason)
            failure_categories[category] += 1
            
            # Extract prime information
            if 'expected_primes' in failure and 'detected_primes' in failure:
                expected = set(failure['expected_primes'])
                detected = set(failure['detected_primes'])
                missing = expected - detected
                
                for prime in missing:
                    prime_failures[prime] += 1
            
            # Extract pattern information
            pattern = self._extract_pattern_from_id(test_id)
            if pattern:
                pattern_failures[pattern] += 1
        
        # Get top 20 failing IDs
        top_failing_ids = sorted(failures, key=lambda x: self._calculate_failure_priority(x))[:20]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            suite_name, failure_categories, prime_failures, pattern_failures
        )
        
        return FailureAnalysis(
            suite_name=suite_name,
            total_failures=total_failures,
            failure_categories=dict(failure_categories),
            top_failing_ids=top_failing_ids,
            prime_failures=dict(prime_failures),
            pattern_failures=dict(pattern_failures),
            recommendations=recommendations
        )
    
    def _categorize_failure(self, reason: str) -> str:
        """Categorize failure based on reason"""
        reason_lower = reason.lower()
        
        if 'prime' in reason_lower and ('missing' in reason_lower or 'not detected' in reason_lower):
            return 'missing_prime'
        elif 'scope' in reason_lower:
            return 'wrong_scope'
        elif 'adapter' in reason_lower or 'invariant' in reason_lower:
            return 'adapter_violation'
        elif 'glossary' in reason_lower:
            return 'glossary_violation'
        elif 'timeout' in reason_lower:
            return 'timeout'
        elif 'error' in reason_lower:
            return 'error'
        elif 'confidence' in reason_lower:
            return 'low_confidence'
        elif 'graph' in reason_lower and 'f1' in reason_lower:
            return 'graph_f1_low'
        elif 'scope' in reason_lower and 'accuracy' in reason_lower:
            return 'scope_accuracy_low'
        else:
            return 'unknown'
    
    def _extract_pattern_from_id(self, test_id: str) -> str:
        """Extract pattern from test ID"""
        if '_' not in test_id:
            return None
        
        parts = test_id.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        
        return None
    
    def _calculate_failure_priority(self, failure: Dict) -> float:
        """Calculate priority score for failure (lower = higher priority)"""
        priority = 0.0
        
        # Higher priority for missing primes
        if 'expected_primes' in failure and 'detected_primes' in failure:
            expected = set(failure['expected_primes'])
            detected = set(failure['detected_primes'])
            missing_count = len(expected - detected)
            priority -= missing_count * 10  # Higher priority for more missing primes
        
        # Higher priority for scope failures
        if 'scope' in failure.get('reason', '').lower():
            priority -= 5
        
        # Higher priority for adapter violations
        if 'adapter' in failure.get('reason', '').lower():
            priority -= 3
        
        # Higher priority for glossary violations
        if 'glossary' in failure.get('reason', '').lower():
            priority -= 3
        
        return priority
    
    def _generate_recommendations(self, suite_name: str, failure_categories: Counter, 
                                prime_failures: Counter, pattern_failures: Counter) -> List[str]:
        """Generate actionable recommendations based on failure analysis"""
        recommendations = []
        
        # Suite-specific recommendations
        if suite_name == 'prime':
            if prime_failures:
                top_missing_primes = prime_failures.most_common(3)
                recommendations.append(f"Focus on missing primes: {', '.join([p[0] for p in top_missing_primes])}")
            
            if pattern_failures:
                top_patterns = pattern_failures.most_common(3)
                recommendations.append(f"Check patterns: {', '.join([p[0] for p in top_patterns])}")
        
        elif suite_name == 'scope':
            if failure_categories['wrong_scope'] > 0:
                recommendations.append("Improve scope attachment logic for negation/quantifiers")
            if failure_categories['missing_prime'] > 0:
                recommendations.append("Ensure scope-related primes (NOT, MUST, HALF) are detected")
        
        elif suite_name == 'idiom':
            if failure_categories['adapter_violation'] > 0:
                recommendations.append("Strengthen cultural adapter invariant checking")
            if failure_categories['missing_prime'] > 0:
                recommendations.append("Improve idiom-to-molecule mapping")
        
        elif suite_name == 'gloss':
            if failure_categories['glossary_violation'] > 0:
                recommendations.append("Fix glossary term preservation logic")
        
        elif suite_name == 'roundtrip':
            if failure_categories['graph_f1_low'] > 0:
                recommendations.append("Improve semantic preservation in round-trip translation")
        
        elif suite_name == 'robust':
            if failure_categories['error'] > 0:
                recommendations.append("Improve error handling for malformed input")
        
        elif suite_name == 'baseline':
            if failure_categories['graph_f1_low'] > 0:
                recommendations.append("System not outperforming baseline - investigate semantic preservation")
        
        elif suite_name == 'perf':
            if failure_categories['timeout'] > 0:
                recommendations.append("Optimize performance - reduce translation latency")
            if failure_categories['low_confidence'] > 0:
                recommendations.append("Improve confidence scoring")
        
        # General recommendations
        if failure_categories['missing_prime'] > failure_categories['wrong_scope']:
            recommendations.append("Prime detection needs improvement - focus on recall")
        elif failure_categories['wrong_scope'] > failure_categories['missing_prime']:
            recommendations.append("Scope accuracy needs improvement - focus on precision")
        
        if failure_categories['adapter_violation'] > 0:
            recommendations.append("Cultural adapter is changing factual content - fix invariant checking")
        
        if failure_categories['glossary_violation'] > 0:
            recommendations.append("Glossary terms not being preserved - fix binding logic")
        
        return recommendations
    
    def print_triage_report(self, analyses: Dict[str, FailureAnalysis]):
        """Print comprehensive triage report"""
        print("=" * 80)
        print("FAILURE TRIAGE REPORT")
        print("=" * 80)
        
        total_failures = sum(analysis.total_failures for analysis in analyses.values())
        print(f"Total Failures: {total_failures}")
        print()
        
        # Print analysis for each suite
        for suite_name, analysis in analyses.items():
            print(f"📊 SUITE: {suite_name.upper()}")
            print(f"   Total Failures: {analysis.total_failures}")
            
            if analysis.failure_categories:
                print("   Failure Categories:")
                for category, count in sorted(analysis.failure_categories.items(), key=lambda x: x[1], reverse=True):
                    description = self.failure_categories.get(category, category)
                    print(f"     - {category}: {count} ({description})")
            
            if analysis.prime_failures:
                print("   Top Missing Primes:")
                for prime, count in analysis.prime_failures.most_common(5):
                    print(f"     - {prime}: {count} failures")
            
            if analysis.pattern_failures:
                print("   Top Failing Patterns:")
                for pattern, count in analysis.pattern_failures.most_common(5):
                    print(f"     - {pattern}: {count} failures")
            
            if analysis.recommendations:
                print("   Recommendations:")
                for rec in analysis.recommendations:
                    print(f"     - {rec}")
            
            print("   Top 20 Failing IDs:")
            for i, failure in enumerate(analysis.top_failing_ids[:20], 1):
                test_id = failure.get('test_id', 'unknown')
                reason = failure.get('reason', 'unknown')
                print(f"     {i:2d}. {test_id}: {reason}")
            
            print()
        
        # Print overall summary
        print("🎯 OVERALL RECOMMENDATIONS:")
        all_recommendations = set()
        for analysis in analyses.values():
            all_recommendations.update(analysis.recommendations)
        
        for rec in sorted(all_recommendations):
            print(f"   - {rec}")
        
        print("=" * 80)
    
    def save_triage_report(self, analyses: Dict[str, FailureAnalysis], output_file: str = "triage_report.json"):
        """Save triage report to JSON file"""
        report = {
            'timestamp': str(Path().cwd()),
            'total_failures': sum(analysis.total_failures for analysis in analyses.values()),
            'suite_analyses': {}
        }
        
        for suite_name, analysis in analyses.items():
            report['suite_analyses'][suite_name] = {
                'total_failures': analysis.total_failures,
                'failure_categories': analysis.failure_categories,
                'top_failing_ids': analysis.top_failing_ids,
                'prime_failures': analysis.prime_failures,
                'pattern_failures': analysis.pattern_failures,
                'recommendations': analysis.recommendations
            }
        
        output_path = self.reports_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Triage report saved to {output_path}")

def main():
    """Main triage function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fail Triage Tool")
    parser.add_argument('--reports-dir', default='reports', help='Reports directory')
    parser.add_argument('--report-file', default='summary.json', help='Report file name')
    parser.add_argument('--output', default='triage_report.json', help='Output file name')
    parser.add_argument('--suite', help='Analyze specific suite only')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    triage = FailTriage(args.reports_dir)
    
    try:
        # Load evaluation report
        report = triage.load_evaluation_report(args.report_file)
        
        # Analyze failures
        analyses = triage.analyze_failures(report)
        
        # Filter by suite if specified
        if args.suite:
            if args.suite in analyses:
                analyses = {args.suite: analyses[args.suite]}
            else:
                print(f"Suite '{args.suite}' not found in report")
                return
        
        # Print and save report
        triage.print_triage_report(analyses)
        triage.save_triage_report(analyses, args.output)
        
    except Exception as e:
        print(f"Error during triage: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

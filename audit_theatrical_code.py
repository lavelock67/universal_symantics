#!/usr/bin/env python3
"""
Audit Script for Theatrical Code and Mock Data.

This script identifies all theatrical code, mock data, and artificial results
that could be providing false numbers and getting in the way of proper honest testing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TheatricalCodeAuditor:
    """Auditor for identifying theatrical code and mock data."""
    
    def __init__(self):
        """Initialize the auditor."""
        self.theatrical_issues = []
        self.mock_data_issues = []
        self.artificial_results = []
        self.fixed_issues = []
        
    def audit_codebase(self):
        """Audit the entire codebase for theatrical code."""
        logger.info("Starting comprehensive theatrical code audit...")
        
        # Audit specific files and patterns
        self._audit_algebra_file()
        self._audit_compression_validation()
        self._audit_demo_file()
        self._audit_test_data()
        self._audit_evaluation_results()
        self._audit_placeholder_warnings()
        
        return self._generate_audit_report()
    
    def _audit_algebra_file(self):
        """Audit the algebra file for fake factorization."""
        logger.info("Auditing algebra.py for fake factorization...")
        
        algebra_path = Path("src/table/algebra.py")
        if not algebra_path.exists():
            self.theatrical_issues.append({
                "file": "src/table/algebra.py",
                "issue": "File not found",
                "severity": "high",
                "description": "Algebra file missing"
            })
            return
        
        with open(algebra_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for fake GenericFactor primitive
        if "GenericFactor" in content:
            self.theatrical_issues.append({
                "file": "src/table/algebra.py",
                "issue": "Fake GenericFactor primitive",
                "severity": "critical",
                "description": "GenericFactor creates fake primitives for any input"
            })
        else:
            self.fixed_issues.append({
                "file": "src/table/algebra.py",
                "issue": "GenericFactor removed",
                "status": "fixed",
                "description": "Fake factorization has been removed"
            })
        
        # Check for placeholder factorization
        if "_factor_generic" in content:
            # Check if it returns empty list (fixed) or fake primitives
            if "return []" in content:
                self.fixed_issues.append({
                    "file": "src/table/algebra.py",
                    "issue": "_factor_generic returns empty list",
                    "status": "fixed",
                    "description": "No longer creates fake primitives"
                })
            else:
                self.theatrical_issues.append({
                    "file": "src/table/algebra.py",
                    "issue": "_factor_generic creates fake primitives",
                    "severity": "critical",
                    "description": "Still creating fake primitives"
                })
    
    def _audit_compression_validation(self):
        """Audit compression validation for theatrical test data."""
        logger.info("Auditing compression validation...")
        
        compression_path = Path("src/validate/compression.py")
        if not compression_path.exists():
            return
        
        with open(compression_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for simple test patterns
        simple_patterns = [
            "The cat is on the mat",
            "A dog runs in the park",
            "Simple, repetitive patterns"
        ]
        
        for pattern in simple_patterns:
            if pattern in content:
                self.theatrical_issues.append({
                    "file": "src/validate/compression.py",
                    "issue": "Simple test patterns",
                    "severity": "high",
                    "description": f"Using simple pattern: {pattern}"
                })
        
        # Check for artificial compression ratios (but ignore comments about avoiding them)
        if "6-23×" in content:
            self.theatrical_issues.append({
                "file": "src/validate/compression.py",
                "issue": "Artificial compression ratios",
                "severity": "high",
                "description": "Mention of artificial 6-23× compression ratios"
            })
        elif "compression ratios" in content and "avoid" not in content:
            self.theatrical_issues.append({
                "file": "src/validate/compression.py",
                "issue": "Artificial compression ratios",
                "severity": "high",
                "description": "Mention of artificial compression ratios"
            })
    
    def _audit_demo_file(self):
        """Audit demo.py for artificial test signals."""
        logger.info("Auditing demo.py for artificial signals...")
        
        demo_path = Path("demo.py")
        if not demo_path.exists():
            return
        
        with open(demo_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for correlated signals
        if "np.array([0.8, 0.2, 0.9, 0.1, 0.7])" in content:
            self.theatrical_issues.append({
                "file": "demo.py",
                "issue": "Artificial correlated signals",
                "severity": "medium",
                "description": "Using artificially correlated multi-modal signals"
            })
        
        # Check for placeholder warnings
        if "placeholder" in content.lower() or "warning" in content.lower():
            self.fixed_issues.append({
                "file": "demo.py",
                "issue": "Placeholder warnings added",
                "status": "fixed",
                "description": "Demo includes warnings about placeholder implementations"
            })
    
    def _audit_test_data(self):
        """Audit test data for mock data and artificial examples."""
        logger.info("Auditing test data for mock data...")
        
        # Check for repetitive test examples
        test_files = [
            "data/parallel_test_data.json",
            "data/expanded_parallel_test_data.json",
            "data/comprehensive_dataset_1k_plus.json"
        ]
        
        for file_path in test_files:
            path = Path(file_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count occurrences of common test phrases
                test_phrases = [
                    "The cat is on the mat",
                    "The book is on the table",
                    "The car is in the garage"
                ]
                
                for phrase in test_phrases:
                    count = content.count(phrase)
                    if count > 5:  # More than 5 occurrences suggests overuse
                        self.mock_data_issues.append({
                            "file": file_path,
                            "issue": "Overused test phrases",
                            "severity": "medium",
                            "description": f"'{phrase}' appears {count} times"
                        })
    
    def _audit_evaluation_results(self):
        """Audit evaluation results for artificial performance."""
        logger.info("Auditing evaluation results...")
        
        # Check recent evaluation reports
        eval_files = [
            "data/enhanced_nsm_metrics_report.json",
            "data/enhanced_nsm_comparison_report.json",
            "data/enhanced_ud_patterns_report.json"
        ]
        
        for file_path in eval_files:
            path = Path(file_path)
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check for suspiciously high scores
                    self._check_suspicious_scores(data, file_path)
                    
                except Exception as e:
                    logger.warning(f"Could not parse {file_path}: {e}")
    
    def _check_suspicious_scores(self, data: Dict[str, Any], file_path: str):
        """Check for suspiciously high or artificial scores."""
        
        def check_nested_scores(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check for suspiciously high scores
                    if isinstance(value, (int, float)):
                        if value > 0.95 and "legality" in current_path.lower():
                            self.artificial_results.append({
                                "file": file_path,
                                "issue": "Suspiciously high legality score",
                                "severity": "high",
                                "description": f"{current_path}: {value} (suspiciously high)"
                            })
                        elif value > 0.9 and "confidence" in current_path.lower():
                            self.artificial_results.append({
                                "file": file_path,
                                "issue": "Suspiciously high confidence score",
                                "severity": "medium",
                                "description": f"{current_path}: {value} (suspiciously high)"
                            })
                        elif value == 1.0 and "cross_translatability" in current_path.lower():
                            self.artificial_results.append({
                                "file": file_path,
                                "issue": "Perfect cross-translatability score",
                                "severity": "high",
                                "description": f"{current_path}: {value} (suspiciously perfect)"
                            })
                    
                    elif isinstance(value, (dict, list)):
                        check_nested_scores(value, current_path)
        
        check_nested_scores(data)
    
    def _audit_placeholder_warnings(self):
        """Audit placeholder warnings file."""
        logger.info("Auditing placeholder warnings...")
        
        warnings_path = Path("PLACEHOLDER_WARNINGS.md")
        if warnings_path.exists():
            self.fixed_issues.append({
                "file": "PLACEHOLDER_WARNINGS.md",
                "issue": "Placeholder warnings documented",
                "status": "good",
                "description": "Theatrical code issues are properly documented"
            })
        
        honest_audit_path = Path("honest_audit_report.md")
        if honest_audit_path.exists():
            self.fixed_issues.append({
                "file": "honest_audit_report.md",
                "issue": "Honest audit report exists",
                "status": "good",
                "description": "Honest assessment of system limitations documented"
            })
    
    def _generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        
        report = {
            "metadata": {
                "audit_type": "Theatrical Code and Mock Data Audit",
                "timestamp": "2025-08-22",
                "description": "Comprehensive audit of theatrical code, mock data, and artificial results"
            },
            "summary": {
                "total_theatrical_issues": len(self.theatrical_issues),
                "total_mock_data_issues": len(self.mock_data_issues),
                "total_artificial_results": len(self.artificial_results),
                "total_fixed_issues": len(self.fixed_issues),
                "overall_status": self._determine_overall_status()
            },
            "theatrical_issues": self.theatrical_issues,
            "mock_data_issues": self.mock_data_issues,
            "artificial_results": self.artificial_results,
            "fixed_issues": self.fixed_issues,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _determine_overall_status(self) -> str:
        """Determine overall audit status."""
        critical_issues = sum(1 for issue in self.theatrical_issues 
                            if issue.get("severity") == "critical")
        
        if critical_issues > 0:
            return "CRITICAL - Major theatrical code found"
        elif len(self.theatrical_issues) > 0:
            return "WARNING - Some theatrical code found"
        elif len(self.artificial_results) > 0:
            return "CAUTION - Suspicious results found"
        else:
            return "CLEAN - No major theatrical code found"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on audit findings."""
        recommendations = []
        
        if len(self.theatrical_issues) > 0:
            recommendations.append("Fix all theatrical code issues before proceeding with TODOs")
            recommendations.append("Replace placeholder implementations with real algorithms")
            recommendations.append("Use realistic test data instead of simple examples")
        
        if len(self.artificial_results) > 0:
            recommendations.append("Investigate suspiciously high performance scores")
            recommendations.append("Validate evaluation metrics with independent testing")
            recommendations.append("Use more diverse and realistic evaluation datasets")
        
        if len(self.mock_data_issues) > 0:
            recommendations.append("Replace repetitive test examples with diverse data")
            recommendations.append("Use real-world datasets for evaluation")
            recommendations.append("Implement proper data generation pipelines")
        
        if len(self.fixed_issues) > 0:
            recommendations.append("Good progress on fixing theatrical code")
            recommendations.append("Continue monitoring for new theatrical code")
        
        if len(recommendations) == 0:
            recommendations.append("System appears clean - safe to proceed with TODOs")
        
        return recommendations

def main():
    """Run the theatrical code audit."""
    logger.info("Starting theatrical code audit...")
    
    auditor = TheatricalCodeAuditor()
    report = auditor.audit_codebase()
    
    # Save audit report
    output_path = Path("data/theatrical_code_audit_report.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved audit report to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("THEATRICAL CODE AUDIT SUMMARY")
    print("="*80)
    print(f"Overall Status: {report['summary']['overall_status']}")
    print(f"Theatrical Issues: {report['summary']['total_theatrical_issues']}")
    print(f"Mock Data Issues: {report['summary']['total_mock_data_issues']}")
    print(f"Artificial Results: {report['summary']['total_artificial_results']}")
    print(f"Fixed Issues: {report['summary']['total_fixed_issues']}")
    print("="*80)
    
    if report['summary']['total_theatrical_issues'] > 0:
        print("\n🚨 THEATRICAL ISSUES FOUND:")
        for issue in report['theatrical_issues']:
            print(f"  {issue['severity'].upper()}: {issue['file']} - {issue['description']}")
    
    if report['summary']['total_artificial_results'] > 0:
        print("\n⚠️ SUSPICIOUS RESULTS FOUND:")
        for result in report['artificial_results'][:5]:  # Show first 5
            print(f"  {result['severity'].upper()}: {result['file']} - {result['description']}")
    
    if report['summary']['total_fixed_issues'] > 0:
        print("\n✅ FIXED ISSUES:")
        for issue in report['fixed_issues']:
            print(f"  {issue['status'].upper()}: {issue['file']} - {issue['description']}")
    
    print("\n📋 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("="*80)

if __name__ == "__main__":
    main()



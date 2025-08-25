#!/usr/bin/env python3
"""
Specific Audit for Robust Aspect Mapper - Check for Theater Code
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AspectMapperAuditor:
    """Specific auditor for robust aspect mapper."""
    
    def __init__(self):
        self.issues = []
        self.suspicious_patterns = []
        self.test_rigging = []
        
    def audit_aspect_mapper(self):
        """Audit the robust aspect mapper for theater code."""
        logger.info("Auditing robust aspect mapper for theater code...")
        
        # Read the aspect mapper file
        aspect_mapper_path = Path("robust_aspect_mapper.py")
        if not aspect_mapper_path.exists():
            self.issues.append({
                "issue": "File not found",
                "severity": "critical",
                "description": "robust_aspect_mapper.py does not exist"
            })
            return self._generate_report()
        
        with open(aspect_mapper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for suspicious patterns
        self._check_hardcoded_test_matching(content)
        self._check_overly_simple_patterns(content)
        self._check_confidence_rigging(content)
        self._check_test_case_rigging(content)
        self._check_verb_extraction_rigging(content)
        
        return self._generate_report()
    
    def _check_hardcoded_test_matching(self, content: str):
        """Check if patterns are hardcoded to match specific test cases."""
        logger.info("Checking for hardcoded test matching...")
        
        # Look for patterns that exactly match test cases
        test_cases = [
            "acabo de",
            "viens d",
            "just",
            "almost",
            "failli",
            "casi",
            "por poco"
        ]
        
        for test_case in test_cases:
            if test_case in content:
                # Check if it's in a pattern list vs hardcoded logic
                pattern_context = self._get_context_around(content, test_case)
                if "hardcoded" in pattern_context.lower() or "test" in pattern_context.lower():
                    self.suspicious_patterns.append({
                        "issue": "Hardcoded test matching",
                        "severity": "high",
                        "description": f"Pattern '{test_case}' appears to be hardcoded for tests",
                        "context": pattern_context[:100]
                    })
    
    def _check_overly_simple_patterns(self, content: str):
        """Check if patterns are too simple and might be rigged."""
        logger.info("Checking for overly simple patterns...")
        
        # Look for single-word patterns that might be too broad
        simple_patterns = [
            "just",
            "almost",
            "again",
            "stop",
            "been"
        ]
        
        for pattern in simple_patterns:
            if pattern in content:
                # Check if it's used as a standalone pattern
                if f"'{pattern}'" in content or f'"{pattern}"' in content:
                    self.suspicious_patterns.append({
                        "issue": "Overly simple pattern",
                        "severity": "medium",
                        "description": f"Single-word pattern '{pattern}' might be too broad",
                        "suggestion": "Consider multi-word patterns for better specificity"
                    })
    
    def _check_confidence_rigging(self, content: str):
        """Check for artificial confidence scoring."""
        logger.info("Checking for confidence rigging...")
        
        # Look for hardcoded high confidence values
        confidence_patterns = [
            "0.9",
            "0.95",
            "0.8"
        ]
        
        for conf in confidence_patterns:
            if conf in content:
                context = self._get_context_around(content, conf)
                if "confidence" in context.lower():
                    # Check if it's a reasonable confidence assignment
                    if "base_confidence" in context or "confidence = " in context:
                        self.suspicious_patterns.append({
                            "issue": "Hardcoded confidence",
                            "severity": "medium",
                            "description": f"Hardcoded confidence value {conf}",
                            "suggestion": "Consider dynamic confidence calculation"
                        })
    
    def _check_test_case_rigging(self, content: str):
        """Check if test cases are rigged to pass."""
        logger.info("Checking for test case rigging...")
        
        # Look for test cases that might be cherry-picked
        test_section = self._extract_test_section(content)
        
        if test_section:
            # Check for diversity in test cases
            languages = ["en", "es", "fr"]
            aspect_types = ["recent_past", "ongoing_for", "almost_do", "stop", "resume"]
            
            for lang in languages:
                lang_count = test_section.count(lang)
                if lang_count < 3:  # Should have multiple test cases per language
                    self.test_rigging.append({
                        "issue": "Insufficient test diversity",
                        "severity": "medium",
                        "description": f"Only {lang_count} test cases for {lang}",
                        "suggestion": "Add more diverse test cases"
                    })
            
            # Check for negative controls
            negative_controls = ["cat is on the mat", "gato está", "chat est"]
            negative_count = sum(1 for control in negative_controls if control in test_section)
            
            if negative_count < 2:
                self.test_rigging.append({
                    "issue": "Insufficient negative controls",
                    "severity": "high",
                    "description": f"Only {negative_count} negative control test cases",
                    "suggestion": "Add more negative controls to prevent false positives"
                })
    
    def _check_verb_extraction_rigging(self, content: str):
        """Check if verb extraction is rigged."""
        logger.info("Checking for verb extraction rigging...")
        
        # Look for hardcoded verb mappings
        verb_mapping_section = self._extract_verb_mapping_section(content)
        
        if verb_mapping_section:
            # Check if mappings are too specific to test cases
            test_verbs = ["arrive", "llegar", "arriver", "work", "trabajar", "travailler"]
            
            for verb in test_verbs:
                if verb in verb_mapping_section:
                    # This is actually good - it shows real verb mapping
                    pass
                else:
                    self.suspicious_patterns.append({
                        "issue": "Missing verb mapping",
                        "severity": "low",
                        "description": f"Test verb '{verb}' not in verb mapping",
                        "suggestion": "Add comprehensive verb mappings"
                    })
    
    def _get_context_around(self, content: str, pattern: str, context_size: int = 50) -> str:
        """Get context around a pattern in the content."""
        try:
            index = content.find(pattern)
            if index != -1:
                start = max(0, index - context_size)
                end = min(len(content), index + len(pattern) + context_size)
                return content[start:end]
        except:
            pass
        return ""
    
    def _extract_test_section(self, content: str) -> str:
        """Extract the test cases section from the content."""
        try:
            # Look for test_cases = [
            start = content.find("test_cases = [")
            if start != -1:
                # Find the end of the test cases list
                bracket_count = 0
                for i, char in enumerate(content[start:], start):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            return content[start:i+1]
        except:
            pass
        return ""
    
    def _extract_verb_mapping_section(self, content: str) -> str:
        """Extract the verb mapping section from the content."""
        try:
            # Look for verb_mapping = {
            start = content.find("verb_mapping = {")
            if start != -1:
                # Find the end of the verb mapping dict
                brace_count = 0
                for i, char in enumerate(content[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return content[start:i+1]
        except:
            pass
        return ""
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate audit report."""
        total_issues = len(self.issues) + len(self.suspicious_patterns) + len(self.test_rigging)
        
        report = {
            "metadata": {
                "audit_type": "Robust Aspect Mapper Theater Code Audit",
                "timestamp": "2025-08-22",
                "target_file": "robust_aspect_mapper.py"
            },
            "summary": {
                "total_issues": total_issues,
                "critical_issues": len([i for i in self.issues if i.get("severity") == "critical"]),
                "high_issues": len([i for i in self.issues + self.suspicious_patterns if i.get("severity") == "high"]),
                "medium_issues": len([i for i in self.issues + self.suspicious_patterns + self.test_rigging if i.get("severity") == "medium"]),
                "overall_status": self._determine_status()
            },
            "issues": self.issues,
            "suspicious_patterns": self.suspicious_patterns,
            "test_rigging": self.test_rigging,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _determine_status(self) -> str:
        """Determine overall audit status."""
        critical_count = len([i for i in self.issues if i.get("severity") == "critical"])
        high_count = len([i for i in self.issues + self.suspicious_patterns if i.get("severity") == "high"])
        
        if critical_count > 0:
            return "CRITICAL - Major theater code found"
        elif high_count > 0:
            return "HIGH RISK - Suspicious patterns found"
        elif len(self.suspicious_patterns) + len(self.test_rigging) > 0:
            return "CAUTION - Potential issues found"
        else:
            return "CLEAN - No theater code detected"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        if len(self.issues) > 0:
            recommendations.append("Fix all critical issues before proceeding")
        
        if len(self.suspicious_patterns) > 0:
            recommendations.append("Review and improve pattern specificity")
            recommendations.append("Implement dynamic confidence calculation")
        
        if len(self.test_rigging) > 0:
            recommendations.append("Add more diverse test cases")
            recommendations.append("Increase negative control coverage")
        
        if len(recommendations) == 0:
            recommendations.append("System appears clean - proceed with confidence")
        
        return recommendations


def main():
    """Run the aspect mapper audit."""
    logger.info("Starting robust aspect mapper audit...")
    
    auditor = AspectMapperAuditor()
    report = auditor.audit_aspect_mapper()
    
    # Save audit report
    output_path = Path("data/aspect_mapper_audit_report.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved audit report to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ROBUST ASPECT MAPPER AUDIT SUMMARY")
    print("="*80)
    print(f"Overall Status: {report['summary']['overall_status']}")
    print(f"Total Issues: {report['summary']['total_issues']}")
    print(f"Critical Issues: {report['summary']['critical_issues']}")
    print(f"High Issues: {report['summary']['high_issues']}")
    print(f"Medium Issues: {report['summary']['medium_issues']}")
    print("="*80)
    
    if report['issues']:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in report['issues']:
            print(f"  {issue['severity'].upper()}: {issue['description']}")
    
    if report['suspicious_patterns']:
        print("\n⚠️ SUSPICIOUS PATTERNS:")
        for pattern in report['suspicious_patterns'][:3]:  # Show first 3
            print(f"  {pattern['severity'].upper()}: {pattern['description']}")
    
    if report['test_rigging']:
        print("\n🔍 TEST RIGGING CONCERNS:")
        for rigging in report['test_rigging'][:3]:  # Show first 3
            print(f"  {rigging['severity'].upper()}: {rigging['description']}")
    
    print("\n📋 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("="*80)


if __name__ == "__main__":
    main()

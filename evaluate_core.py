#!/usr/bin/env python3
"""
Core Evaluation Script for Universal Translator
Tests the core functionality without full production pipeline dependencies.
"""

import json
import time
from typing import Dict, List, Any
from pathlib import Path

from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language
from src.semgen.timing import get_metrics


def evaluate_canaries() -> Dict[str, Any]:
    """Evaluate the 11 critical canary tests."""
    print("🔍 Running 11 Critical Canary Tests...")
    
    # Initialize service
    service = NSMDetectionService()
    
    # Test cases from the canary grid
    test_cases = [
        # English tests
        (Language.ENGLISH, "The book is inside the box.", ["INSIDE"]),
        (Language.ENGLISH, "He lives near the station.", ["NEAR"]),
        (Language.ENGLISH, "At most half the students read a lot.", ["NOT", "MORE", "HALF", "MANY"]),
        
        # Spanish tests
        (Language.SPANISH, "El libro está dentro de la caja.", ["INSIDE"]),
        (Language.SPANISH, "Vive cerca de la estación.", ["NEAR"]),
        (Language.SPANISH, "Es falso que el medicamento no funcione.", ["FALSE", "NOT"]),
        
        # French tests
        (Language.FRENCH, "La lampe est au-dessus de la table.", ["ABOVE"]),
        (Language.FRENCH, "Les gens pensent que c'est très bon.", ["PEOPLE", "THINK", "THIS", "VERY", "GOOD"]),
        
        # German tests
        (Language.GERMAN, "Das Buch ist in der Kiste.", ["INSIDE"]),
        (Language.GERMAN, "Die Lampe ist über dem Tisch.", ["ABOVE"]),
        (Language.GERMAN, "Er wohnt in der Nähe von dem Bahnhof.", ["NEAR"]),
    ]
    
    results = []
    passed = 0
    total = len(test_cases)
    
    for i, (lang, text, expected_primes) in enumerate(test_cases, 1):
        try:
            print(f"  {i:2d}. Testing {lang.value.upper()}: {text}")
            
            # Detect primes
            result = service.detect_primes(text, lang)
            detected_primes = [p.text for p in result.primes]
            
            # Check if expected primes are present
            expected_found = all(prime in detected_primes for prime in expected_primes)
            
            if expected_found:
                passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            results.append({
                "test_id": i,
                "language": lang.value,
                "text": text,
                "expected_primes": expected_primes,
                "detected_primes": detected_primes,
                "passed": expected_found,
                "status": status
            })
            
            print(f"      {status} - Expected: {expected_primes}, Got: {detected_primes}")
            
        except Exception as e:
            print(f"      ❌ ERROR: {e}")
            results.append({
                "test_id": i,
                "language": lang.value,
                "text": text,
                "expected_primes": expected_primes,
                "detected_primes": [],
                "passed": False,
                "status": f"❌ ERROR: {e}"
            })
    
    return {
        "canary_results": {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total,
            "tests": results
        }
    }


def evaluate_performance() -> Dict[str, Any]:
    """Evaluate performance metrics."""
    print("⚡ Evaluating Performance Metrics...")
    
    metrics = get_metrics()
    all_metrics = metrics.get_all_metrics()
    
    # Calculate performance summary
    performance_summary = {
        "total_stages": len(all_metrics),
        "stage_metrics": {}
    }
    
    for stage_metric in all_metrics:
        stage = stage_metric["stage"]
        mode = stage_metric["mode"]
        key = f"{stage}_{mode}"
        
        performance_summary["stage_metrics"][key] = {
            "count": stage_metric["count"],
            "avg_time_ms": stage_metric["avg_time_ms"],
            "p95_time_ms": stage_metric["p95_time_ms"],
            "max_time_ms": stage_metric["max_time_ms"]
        }
    
    return {
        "performance": performance_summary,
        "histograms": metrics.export_histograms(),
        "counters": dict(metrics.counters)
    }


def evaluate_pipeline_integrity() -> Dict[str, Any]:
    """Evaluate pipeline integrity."""
    print("🔒 Evaluating Pipeline Integrity...")
    
    health_panel = get_metrics().get_health_panel()
    
    return {
        "traces_not_ending_with_generator": health_panel["pipeline_integrity"]["traces_not_ending_with_generator"],
        "manual_detector_violations": health_panel["pipeline_integrity"]["manual_detector_violations"],
        "invalid_pipeline_paths": health_panel["pipeline_integrity"]["invalid_pipeline_paths"],
        "compliance_status": "compliant" if sum(health_panel["pipeline_integrity"].values()) == 0 else "non_compliant",
        "error_rates": health_panel["error_rates"],
        "performance_alerts": health_panel["performance_alerts"]
    }


def generate_summary_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive summary report."""
    canary = results["canary_results"]
    performance = results["performance"]
    integrity = results["pipeline_integrity"]
    
    # Overall status
    overall_status = "healthy"
    if canary["success_rate"] < 0.85:
        overall_status = "degraded"
    if integrity["compliance_status"] != "compliant":
        overall_status = "critical"
    
    summary = {
        "evaluation_timestamp": time.time(),
        "overall_status": overall_status,
        "key_metrics": {
            "canary_success_rate": canary["success_rate"],
            "pipeline_compliance": integrity["compliance_status"],
            "total_requests": integrity["error_rates"]["total_requests"],
            "error_rate_percent": integrity["error_rates"]["error_rate_percent"]
        },
        "acceptance_gates": {
            "canaries_11_11": canary["passed"] == 11,
            "pipeline_compliant": integrity["compliance_status"] == "compliant",
            "error_rate_acceptable": integrity["error_rates"]["error_rate_percent"] < 5.0,
            "performance_healthy": len(integrity["performance_alerts"]["slow_stages"]) == 0
        },
        "recommendations": []
    }
    
    # Generate recommendations
    if canary["success_rate"] < 1.0:
        summary["recommendations"].append("Fix failing canary tests to achieve 100% success rate")
    
    if integrity["compliance_status"] != "compliant":
        summary["recommendations"].append("Address pipeline integrity violations")
    
    if integrity["error_rates"]["error_rate_percent"] > 1.0:
        summary["recommendations"].append("Investigate high error rates")
    
    if len(integrity["performance_alerts"]["slow_stages"]) > 0:
        summary["recommendations"].append("Address performance bottlenecks")
    
    return summary


def main():
    """Main evaluation function."""
    print("🚀 Universal Translator Core Evaluation")
    print("=" * 50)
    
    # Run evaluations
    results = {}
    results.update(evaluate_canaries())
    results.update(evaluate_performance())
    results["pipeline_integrity"] = evaluate_pipeline_integrity()
    
    # Generate summary
    summary = generate_summary_report(results)
    results["summary"] = summary
    
    # Save reports
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(reports_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    with open(reports_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate markdown summary
    markdown_summary = f"""# Universal Translator Evaluation Summary

## Overall Status: {summary['overall_status'].upper()}

### Key Metrics
- **Canary Success Rate**: {summary['key_metrics']['canary_success_rate']:.1%}
- **Pipeline Compliance**: {summary['key_metrics']['pipeline_compliance']}
- **Total Requests**: {summary['key_metrics']['total_requests']}
- **Error Rate**: {summary['key_metrics']['error_rate_percent']:.2f}%

### Acceptance Gates
- ✅ Canaries 11/11: {summary['acceptance_gates']['canaries_11_11']}
- ✅ Pipeline Compliant: {summary['acceptance_gates']['pipeline_compliant']}
- ✅ Error Rate Acceptable: {summary['acceptance_gates']['error_rate_acceptable']}
- ✅ Performance Healthy: {summary['acceptance_gates']['performance_healthy']}

### Recommendations
"""
    
    if summary['recommendations']:
        for rec in summary['recommendations']:
            markdown_summary += f"- {rec}\n"
    else:
        markdown_summary += "- All systems operational\n"
    
    markdown_summary += f"""
### Detailed Results
- **Canary Tests**: {results['canary_results']['passed']}/{results['canary_results']['total_tests']} passed
- **Pipeline Stages**: {results['performance']['total_stages']} monitored
- **Integrity Violations**: {results['pipeline_integrity']['traces_not_ending_with_generator'] + results['pipeline_integrity']['manual_detector_violations'] + results['pipeline_integrity']['invalid_pipeline_paths']}

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(reports_dir / "summary.md", "w") as f:
        f.write(markdown_summary)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Overall Status: {summary['overall_status'].upper()}")
    print(f"Canary Success Rate: {summary['key_metrics']['canary_success_rate']:.1%}")
    print(f"Pipeline Compliance: {summary['key_metrics']['pipeline_compliance']}")
    print(f"Error Rate: {summary['key_metrics']['error_rate_percent']:.2f}%")
    
    print(f"\nReports saved to: {reports_dir.absolute()}")
    print("- detailed_results.json")
    print("- summary.json") 
    print("- summary.md")


if __name__ == "__main__":
    main()

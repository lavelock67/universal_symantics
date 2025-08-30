"""
Health panel for pipeline monitoring and metrics.

This module provides endpoints and utilities for monitoring
pipeline health, performance, and integrity.
"""

from typing import Dict, Any
from src.semgen.timing import get_metrics
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class HealthPanel:
    """Health panel for pipeline monitoring."""
    
    def __init__(self):
        self.metrics = get_metrics()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_panel = self.metrics.get_health_panel()
        
        # Add overall health status
        health_status = "healthy"
        alerts = []
        
        # Check pipeline integrity
        integrity = health_panel["pipeline_integrity"]
        if integrity["traces_not_ending_with_generator"] > 0:
            health_status = "degraded"
            alerts.append(f"Pipeline integrity issue: {integrity['traces_not_ending_with_generator']} traces not ending with generator")
        
        if integrity["manual_detector_violations"] > 0:
            health_status = "critical"
            alerts.append(f"ADR-001 violation: {integrity['manual_detector_violations']} manual detector violations")
        
        if integrity["invalid_pipeline_paths"] > 0:
            health_status = "degraded"
            alerts.append(f"Invalid pipeline paths: {integrity['invalid_pipeline_paths']} violations")
        
        # Check error rates
        error_rates = health_panel["error_rates"]
        if error_rates["error_rate_percent"] > 5.0:
            health_status = "degraded"
            alerts.append(f"High error rate: {error_rates['error_rate_percent']:.2f}%")
        
        # Check performance
        performance = health_panel["performance_alerts"]
        if performance["high_latency_alerts"] > 10:
            health_status = "degraded"
            alerts.append(f"Performance issues: {performance['high_latency_alerts']} high latency alerts")
        
        return {
            "status": health_status,
            "alerts": alerts,
            "metrics": health_panel,
            "summary": {
                "total_requests": error_rates["total_requests"],
                "error_rate_percent": error_rates["error_rate_percent"],
                "pipeline_integrity_violations": sum(integrity.values()),
                "performance_alerts": performance["high_latency_alerts"]
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        return {
            "stage_metrics": self.metrics.get_all_metrics(),
            "histograms": self.metrics.export_histograms(),
            "counters": dict(self.metrics.counters)
        }
    
    def get_pipeline_integrity_report(self) -> Dict[str, Any]:
        """Get pipeline integrity report."""
        health_panel = self.metrics.get_health_panel()
        integrity = health_panel["pipeline_integrity"]
        
        return {
            "integrity_checks": {
                "traces_ending_with_generator": integrity["traces_not_ending_with_generator"] == 0,
                "no_manual_detectors": integrity["manual_detector_violations"] == 0,
                "valid_pipeline_paths": integrity["invalid_pipeline_paths"] == 0
            },
            "violations": integrity,
            "compliance_status": "compliant" if sum(integrity.values()) == 0 else "non_compliant"
        }


# Global health panel instance
_health_panel = HealthPanel()


def get_health_panel() -> HealthPanel:
    """Get the global health panel instance."""
    return _health_panel

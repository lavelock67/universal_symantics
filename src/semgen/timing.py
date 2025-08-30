"""
Pipeline timing instrumentation with monotonic timers.

This module provides timing decorators and metrics collection for
measuring pipeline stage performance.
"""

import time
import functools
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage with histogram buckets."""
    stage: str
    mode: str
    count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    times_ms: List[float] = field(default_factory=list)
    
    # Histogram buckets for performance analysis
    buckets: List[float] = field(default_factory=lambda: [5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000])
    bucket_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def add_measurement(self, time_ms: float):
        """Add a timing measurement with histogram bucketing."""
        self.count += 1
        self.total_time_ms += time_ms
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)
        self.times_ms.append(time_ms)
        
        # Update histogram buckets
        for bucket in self.buckets:
            if time_ms <= bucket:
                self.bucket_counts[f"le_{bucket}"] += 1
                break
        else:
            # If time_ms > all buckets, count in the last bucket
            self.bucket_counts[f"le_{self.buckets[-1]}"] += 1
    
    def get_p50(self) -> float:
        """Get 50th percentile."""
        if not self.times_ms:
            return 0.0
        sorted_times = sorted(self.times_ms)
        return sorted_times[len(sorted_times) // 2]
    
    def get_p95(self) -> float:
        """Get 95th percentile."""
        if not self.times_ms:
            return 0.0
        sorted_times = sorted(self.times_ms)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[index]
    
    def get_p99(self) -> float:
        """Get 99th percentile."""
        if not self.times_ms:
            return 0.0
        sorted_times = sorted(self.times_ms)
        index = int(len(sorted_times) * 0.99)
        return sorted_times[index]
    
    def get_avg(self) -> float:
        """Get average time."""
        if self.count == 0:
            return 0.0
        return self.total_time_ms / self.count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export with histogram data."""
        return {
            "stage": self.stage,
            "mode": self.mode,
            "count": self.count,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.get_avg(),
            "min_time_ms": self.min_time_ms if self.min_time_ms != float('inf') else 0.0,
            "max_time_ms": self.max_time_ms,
            "p50_time_ms": self.get_p50(),
            "p95_time_ms": self.get_p95(),
            "p99_time_ms": self.get_p99(),
            "histogram_buckets": dict(self.bucket_counts)
        }


class PipelineMetrics:
    """Global pipeline metrics collector with health panel."""
    
    def __init__(self):
        self.metrics: Dict[str, StageMetrics] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.health_metrics: Dict[str, Any] = {
            "pipeline_integrity": {
                "traces_not_ending_with_generator": 0,
                "manual_detector_violations": 0,
                "invalid_pipeline_paths": 0
            },
            "error_rates": {
                "total_requests": 0,
                "failed_requests": 0,
                "error_rate_percent": 0.0
            },
            "performance_alerts": {
                "slow_stages": [],
                "high_latency_alerts": 0
            }
        }
        self.logger = logging.getLogger(__name__)
    
    def observe_stage(self, stage: str, mode: str, time_ms: float):
        """Record a stage timing measurement."""
        key = f"{stage}_{mode}"
        if key not in self.metrics:
            self.metrics[key] = StageMetrics(stage=stage, mode=mode)
        
        self.metrics[key].add_measurement(time_ms)
        
        # Log if timing is unusually high
        if time_ms > 1000:  # More than 1 second
            self.logger.warning(f"Slow stage: {stage} ({mode}) took {time_ms:.2f}ms")
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a counter."""
        self.counters[counter_name] += value
    
    def get_counter(self, counter_name: str) -> int:
        """Get counter value."""
        return self.counters.get(counter_name, 0)
    
    def record_pipeline_integrity(self, semantic_trace: List[str], manual_detector_count: int, pipeline_path: str):
        """Record pipeline integrity metrics."""
        self.health_metrics["error_rates"]["total_requests"] += 1
        
        # Check if trace ends with generator
        if not semantic_trace or semantic_trace[-1] != "generator":
            self.health_metrics["pipeline_integrity"]["traces_not_ending_with_generator"] += 1
        
        # Check for manual detector violations
        if manual_detector_count > 0:
            self.health_metrics["pipeline_integrity"]["manual_detector_violations"] += 1
        
        # Check for invalid pipeline paths
        if pipeline_path not in ["semantic", "semantic+umr"]:
            self.health_metrics["pipeline_integrity"]["invalid_pipeline_paths"] += 1
    
    def record_error(self):
        """Record a failed request."""
        self.health_metrics["error_rates"]["failed_requests"] += 1
        self.health_metrics["error_rates"]["error_rate_percent"] = (
            self.health_metrics["error_rates"]["failed_requests"] / 
            max(self.health_metrics["error_rates"]["total_requests"], 1) * 100
        )
    
    def record_slow_stage(self, stage: str, mode: str, time_ms: float):
        """Record a slow stage alert."""
        self.health_metrics["performance_alerts"]["high_latency_alerts"] += 1
        self.health_metrics["performance_alerts"]["slow_stages"].append({
            "stage": stage,
            "mode": mode,
            "time_ms": time_ms,
            "timestamp": time.time()
        })
        # Keep only last 100 slow stage alerts
        if len(self.health_metrics["performance_alerts"]["slow_stages"]) > 100:
            self.health_metrics["performance_alerts"]["slow_stages"] = \
                self.health_metrics["performance_alerts"]["slow_stages"][-100:]
    
    def get_health_panel(self) -> Dict[str, Any]:
        """Get health panel metrics."""
        return {
            "pipeline_integrity": self.health_metrics["pipeline_integrity"],
            "error_rates": self.health_metrics["error_rates"],
            "performance_alerts": self.health_metrics["performance_alerts"],
            "stage_performance": {
                stage: metrics.to_dict() for stage, metrics in self.metrics.items()
            }
        }
    
    def get_stage_metrics(self, stage: str, mode: str) -> Optional[StageMetrics]:
        """Get metrics for a specific stage and mode."""
        key = f"{stage}_{mode}"
        return self.metrics.get(key)
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get all metrics as dictionaries."""
        return [metrics.to_dict() for metrics in self.metrics.values()]
    
    def export_histograms(self) -> Dict[str, Any]:
        """Export histograms for monitoring."""
        histograms = {}
        for metrics in self.metrics.values():
            key = f"stage_latency_ms{{stage=\"{metrics.stage}\",mode=\"{metrics.mode}\"}}"
            histograms[key] = {
                "buckets": self._create_histogram_buckets(metrics.times_ms),
                "sum": metrics.total_time_ms,
                "count": metrics.count
            }
        return histograms
    
    def _create_histogram_buckets(self, times_ms: List[float]) -> List[Dict[str, Any]]:
        """Create histogram buckets from timing data."""
        if not times_ms:
            return []
        
        # Define bucket boundaries (in ms)
        boundaries = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        buckets = []
        
        for i, boundary in enumerate(boundaries):
            count = sum(1 for t in times_ms if t <= boundary)
            if i > 0:
                count -= sum(1 for t in times_ms if t <= boundaries[i-1])
            
            buckets.append({
                "le": boundary,
                "count": count
            })
        
        return buckets
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.health_metrics = {
            "pipeline_integrity": {
                "traces_not_ending_with_generator": 0,
                "manual_detector_violations": 0,
                "invalid_pipeline_paths": 0
            },
            "error_rates": {
                "total_requests": 0,
                "failed_requests": 0,
                "error_rate_percent": 0.0
            },
            "performance_alerts": {
                "slow_stages": [],
                "high_latency_alerts": 0
            }
        }


# Global metrics instance
_global_metrics = PipelineMetrics()


def get_metrics() -> PipelineMetrics:
    """Get the global metrics instance."""
    return _global_metrics


def timed(stage: str, mode: str = "default"):
    """Decorator to time pipeline stages."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter_ns()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter_ns()
                time_ms = (end_time - start_time) / 1_000_000  # Convert to milliseconds
                _global_metrics.observe_stage(stage, mode, time_ms)
        return wrapper
    return decorator


def timed_method(stage: str, mode: str = "default"):
    """Decorator to time class methods."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter_ns()
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter_ns()
                time_ms = (end_time - start_time) / 1_000_000  # Convert to milliseconds
                _global_metrics.observe_stage(stage, mode, time_ms)
        return wrapper
    return decorator

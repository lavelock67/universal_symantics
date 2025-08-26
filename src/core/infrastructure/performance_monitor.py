#!/usr/bin/env python3
"""
Performance Monitoring Service

This module provides comprehensive performance monitoring for the NSM system,
including metrics collection, resource monitoring, and performance optimization.
"""

import time
import threading
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from ...shared.config.settings import get_settings
from ...shared.logging.logger import get_logger, PerformanceContext
from ...shared.exceptions.exceptions import PerformanceError, create_error_context


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    process_count: int
    load_average: List[float]
    network_io: Dict[str, float]
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.settings = get_settings()
        self.logger = get_logger("performance_monitor")
        
        # Metrics storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._snapshots: deque = deque(maxlen=100)
        self._alerts: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._operation_times: Dict[str, List[float]] = defaultdict(list)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._success_counts: Dict[str, int] = defaultdict(int)
        
        # Resource monitoring
        self._last_snapshot = None
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Thresholds and alerts
        self._thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time_ms": 5000.0,
            "error_rate": 0.05  # 5%
        }
        
        # Start monitoring if enabled
        if self.settings.performance.enable_performance_metrics:
            self.start_monitoring()
    
    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                self._collect_system_metrics()
                self._check_thresholds()
                self._cleanup_old_metrics()
                
                time.sleep(self.settings.performance.metrics_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                time.sleep(10)  # Wait before retrying
    
    def _collect_system_metrics(self) -> None:
        """Collect system-wide performance metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load_avg = psutil.getloadavg()
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory.used / 1024 / 1024,
                disk_percent=disk.percent,
                process_count=len(psutil.pids()),
                load_average=list(load_avg),
                network_io=network_io
            )
            
            self._snapshots.append(snapshot)
            self._last_snapshot = snapshot
            
            # Store metrics
            self.record_metric("system.cpu_percent", cpu_percent)
            self.record_metric("system.memory_percent", memory.percent)
            self.record_metric("system.memory_mb", memory.used / 1024 / 1024)
            self.record_metric("system.disk_percent", disk.percent)
            self.record_metric("system.process_count", len(psutil.pids()))
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {str(e)}")
    
    def _check_thresholds(self) -> None:
        """Check if any metrics exceed thresholds."""
        if not self._last_snapshot:
            return
        
        alerts = []
        
        # CPU threshold
        if self._last_snapshot.cpu_percent > self._thresholds["cpu_percent"]:
            alerts.append({
                "type": "high_cpu",
                "value": self._last_snapshot.cpu_percent,
                "threshold": self._thresholds["cpu_percent"],
                "timestamp": self._last_snapshot.timestamp
            })
        
        # Memory threshold
        if self._last_snapshot.memory_percent > self._thresholds["memory_percent"]:
            alerts.append({
                "type": "high_memory",
                "value": self._last_snapshot.memory_percent,
                "threshold": self._thresholds["memory_percent"],
                "timestamp": self._last_snapshot.timestamp
            })
        
        # Disk threshold
        if self._last_snapshot.disk_percent > self._thresholds["disk_percent"]:
            alerts.append({
                "type": "high_disk",
                "value": self._last_snapshot.disk_percent,
                "threshold": self._thresholds["disk_percent"],
                "timestamp": self._last_snapshot.timestamp
            })
        
        # Add alerts
        for alert in alerts:
            self._alerts.append(alert)
            self.logger.warning(f"Performance alert: {alert['type']} = {alert['value']:.1f}%")
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics to prevent memory bloat."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Clean up old snapshots
        while self._snapshots and self._snapshots[0].timestamp < cutoff_time:
            self._snapshots.popleft()
        
        # Clean up old metrics
        for metric_name, metric_queue in self._metrics.items():
            while metric_queue and metric_queue[0].timestamp < cutoff_time:
                metric_queue.popleft()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self._metrics[name].append(metric)
    
    def record_operation_time(self, operation: str, duration_ms: float) -> None:
        """Record operation execution time."""
        self._operation_times[operation].append(duration_ms)
        self.record_metric(f"operation.{operation}.duration_ms", duration_ms)
        
        # Check response time threshold
        if duration_ms > self._thresholds["response_time_ms"]:
            self.logger.warning(f"Slow operation: {operation} took {duration_ms:.1f}ms")
    
    def record_success(self, operation: str) -> None:
        """Record successful operation."""
        self._success_counts[operation] += 1
        self.record_metric(f"operation.{operation}.success_count", self._success_counts[operation])
    
    def record_error(self, operation: str, error: Exception) -> None:
        """Record operation error."""
        self._error_counts[operation] += 1
        self.record_metric(f"operation.{operation}.error_count", self._error_counts[operation])
        
        # Check error rate
        total_ops = self._success_counts[operation] + self._error_counts[operation]
        if total_ops > 10:  # Only check after some operations
            error_rate = self._error_counts[operation] / total_ops
            if error_rate > self._thresholds["error_rate"]:
                self.logger.error(f"High error rate for {operation}: {error_rate:.1%}")
    
    def get_metrics(self, metric_name: str = None, hours: int = 1) -> Dict[str, Any]:
        """Get metrics for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if metric_name:
            # Get specific metric
            if metric_name not in self._metrics:
                return {"error": f"Metric {metric_name} not found"}
            
            metrics = [m for m in self._metrics[metric_name] if m.timestamp >= cutoff_time]
            
            if not metrics:
                return {"metric": metric_name, "data": [], "count": 0}
            
            values = [m.value for m in metrics]
            return {
                "metric": metric_name,
                "data": [{"timestamp": m.timestamp.isoformat(), "value": m.value} for m in metrics],
                "count": len(metrics),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values)
            }
        else:
            # Get all metrics
            result = {}
            for name, metric_queue in self._metrics.items():
                metrics = [m for m in metric_queue if m.timestamp >= cutoff_time]
                if metrics:
                    values = [m.value for m in metrics]
                    result[name] = {
                        "count": len(metrics),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values)
                    }
            return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            # Current system state
            current_state = {
                "timestamp": datetime.utcnow().isoformat(),
                "monitoring_active": self._monitoring_active,
                "metrics_count": sum(len(q) for q in self._metrics.values()),
                "snapshots_count": len(self._snapshots),
                "alerts_count": len(self._alerts)
            }
            
            # Latest snapshot
            latest_snapshot = None
            if self._last_snapshot:
                latest_snapshot = {
                    "timestamp": self._last_snapshot.timestamp.isoformat(),
                    "cpu_percent": self._last_snapshot.cpu_percent,
                    "memory_percent": self._last_snapshot.memory_percent,
                    "memory_mb": self._last_snapshot.memory_mb,
                    "disk_percent": self._last_snapshot.disk_percent,
                    "process_count": self._last_snapshot.process_count,
                    "load_average": self._last_snapshot.load_average
                }
            
            # Operation statistics
            operation_stats = {}
            for operation, times in self._operation_times.items():
                if times:
                    operation_stats[operation] = {
                        "count": len(times),
                        "avg_time_ms": sum(times) / len(times),
                        "min_time_ms": min(times),
                        "max_time_ms": max(times),
                        "success_count": self._success_counts[operation],
                        "error_count": self._error_counts[operation]
                    }
            
            # Recent alerts
            recent_alerts = self._alerts[-10:] if self._alerts else []
            
            return {
                "current_state": current_state,
                "latest_snapshot": latest_snapshot,
                "operation_stats": operation_stats,
                "recent_alerts": recent_alerts,
                "thresholds": self._thresholds
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {str(e)}")
            return {"error": str(e)}
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        try:
            process = psutil.Process()
            
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "disk_percent": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids()),
                "load_average": list(psutil.getloadavg())
            }
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {str(e)}")
            return {"error": str(e)}
    
    def set_threshold(self, metric: str, value: float) -> None:
        """Set a performance threshold."""
        if metric in self._thresholds:
            self._thresholds[metric] = value
            self.logger.info(f"Updated threshold for {metric}: {value}")
        else:
            self.logger.warning(f"Unknown threshold metric: {metric}")
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self._alerts if alert["timestamp"] >= cutoff_time]
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()
        self.logger.info("All alerts cleared")
    
    def cleanup(self) -> None:
        """Clean up performance monitor resources."""
        self.stop_monitoring()
        self._metrics.clear()
        self._snapshots.clear()
        self._alerts.clear()
        self._operation_times.clear()
        self._error_counts.clear()
        self._success_counts.clear()
        self.logger.info("Performance monitor cleanup completed")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def initialize_performance_monitor() -> PerformanceMonitor:
    """Initialize the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is not None:
        _performance_monitor.cleanup()
    _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def cleanup_performance_monitor() -> None:
    """Clean up the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is not None:
        _performance_monitor.cleanup()
        _performance_monitor = None

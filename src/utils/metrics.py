"""
Metrics Collection Module for SQL Agent System
Provides metrics collection, aggregation, and export capabilities
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional
import json
import statistics


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Single metric value with metadata"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


@dataclass
class HistogramBucket:
    """Histogram bucket for distribution tracking"""
    le: float  # less than or equal
    count: int = 0


class Histogram:
    """Histogram for tracking value distributions"""
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    def __init__(self, name: str, buckets: Optional[List[float]] = None, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.labels = labels or {}
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts = {b: 0 for b in self.buckets}
        self._counts[float('inf')] = 0
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()
    
    def observe(self, value: float) -> None:
        """Record an observation"""
        with self._lock:
            self._sum += value
            self._count += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[bucket] += 1
            self._counts[float('inf')] += 1
    
    def get_percentile(self, percentile: float) -> float:
        """Get approximate percentile value"""
        if self._count == 0:
            return 0.0
        
        target_count = percentile * self._count
        for bucket in self.buckets:
            if self._counts[bucket] >= target_count:
                return bucket
        return self.buckets[-1] if self.buckets else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "histogram",
            "labels": self.labels,
            "buckets": {str(k): v for k, v in self._counts.items()},
            "sum": self._sum,
            "count": self._count,
            "p50": self.get_percentile(0.5),
            "p90": self.get_percentile(0.9),
            "p99": self.get_percentile(0.99),
        }


class MetricsCollector:
    """Thread-safe metrics collector"""
    
    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._labels: Dict[str, Dict[str, str]] = {}
        self._data_lock = threading.Lock()
        self._enabled = True
        self._initialized = True
    
    def enable(self) -> None:
        """Enable metrics collection"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable metrics collection"""
        self._enabled = False
    
    def counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter"""
        if not self._enabled:
            return
        
        key = self._make_key(name, labels)
        with self._data_lock:
            self._counters[key] += value
            if labels:
                self._labels[key] = labels
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value"""
        if not self._enabled:
            return
        
        key = self._make_key(name, labels)
        with self._data_lock:
            self._gauges[key] = value
            if labels:
                self._labels[key] = labels
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, buckets: Optional[List[float]] = None) -> None:
        """Record a histogram observation"""
        if not self._enabled:
            return
        
        key = self._make_key(name, labels)
        with self._data_lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(name, buckets, labels)
            self._histograms[key].observe(value)
    
    def timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timer value"""
        if not self._enabled:
            return
        
        key = self._make_key(name, labels)
        with self._data_lock:
            self._timers[key].append(duration)
            if labels:
                self._labels[key] = labels
        
        # Also record in histogram for distribution
        self.histogram(f"{name}_histogram", duration, labels)
    
    @contextmanager
    def time_operation(self, name: str, labels: Optional[Dict[str, str]] = None) -> Generator[None, None, None]:
        """Context manager for timing operations"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.timer(name, duration, labels)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for metric with labels"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        with self._data_lock:
            metrics = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: v.to_dict() for k, v in self._histograms.items()},
                "timers": {},
            }
            
            # Compute timer statistics
            for key, values in self._timers.items():
                if values:
                    metrics["timers"][key] = {
                        "count": len(values),
                        "sum": sum(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                    }
            
            return metrics
    
    def reset(self) -> None:
        """Reset all metrics"""
        with self._data_lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._labels.clear()
    
    def export_json(self) -> str:
        """Export metrics as JSON string"""
        return json.dumps(self.get_metrics(), indent=2, default=str)
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        with self._data_lock:
            # Export counters
            for key, value in self._counters.items():
                lines.append(f"# TYPE {key.split('{')[0]} counter")
                lines.append(f"{key} {value}")
            
            # Export gauges
            for key, value in self._gauges.items():
                lines.append(f"# TYPE {key.split('{')[0]} gauge")
                lines.append(f"{key} {value}")
            
            # Export histograms
            for key, hist in self._histograms.items():
                base_name = key.split('{')[0]
                lines.append(f"# TYPE {base_name} histogram")
                label_str = key[len(base_name):] if '{' in key else ''
                
                for bucket, count in hist._counts.items():
                    if bucket == float('inf'):
                        bucket_str = '+Inf'
                    else:
                        bucket_str = str(bucket)
                    
                    if label_str:
                        bucket_label = label_str[:-1] + f',le="{bucket_str}"' + '}'
                    else:
                        bucket_label = f'{{le="{bucket_str}"}}'
                    lines.append(f"{base_name}_bucket{bucket_label} {count}")
                
                lines.append(f"{base_name}_sum{label_str} {hist._sum}")
                lines.append(f"{base_name}_count{label_str} {hist._count}")
        
        return "\n".join(lines)


# Convenience functions for global metrics collector
def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    return MetricsCollector()


def counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter"""
    get_metrics_collector().counter(name, value, labels)


def gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Set a gauge value"""
    get_metrics_collector().gauge(name, value, labels)


def histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Record a histogram observation"""
    get_metrics_collector().histogram(name, value, labels)


def timer(name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Record a timer value"""
    get_metrics_collector().timer(name, duration, labels)


@contextmanager
def time_operation(name: str, labels: Optional[Dict[str, str]] = None) -> Generator[None, None, None]:
    """Context manager for timing operations"""
    with get_metrics_collector().time_operation(name, labels):
        yield


# SQL Agent specific metrics
class SQLAgentMetrics:
    """SQL Agent specific metrics helper"""
    
    @staticmethod
    def record_sql_generation(duration: float, success: bool, db_type: str) -> None:
        """Record SQL generation metrics"""
        labels = {"db_type": db_type, "success": str(success).lower()}
        timer("sql_generation_duration", duration, labels)
        counter("sql_generation_total", 1.0, labels)
    
    @staticmethod
    def record_sql_validation(duration: float, success: bool, db_type: str) -> None:
        """Record SQL validation metrics"""
        labels = {"db_type": db_type, "success": str(success).lower()}
        timer("sql_validation_duration", duration, labels)
        counter("sql_validation_total", 1.0, labels)
    
    @staticmethod
    def record_self_healing(duration: float, success: bool, iterations: int, db_type: str) -> None:
        """Record self-healing metrics"""
        labels = {"db_type": db_type, "success": str(success).lower()}
        timer("self_healing_duration", duration, labels)
        counter("self_healing_total", 1.0, labels)
        histogram("self_healing_iterations", float(iterations), labels)
    
    @staticmethod
    def record_llm_call(duration: float, model_id: str, input_tokens: int, output_tokens: int) -> None:
        """Record LLM call metrics"""
        labels = {"model_id": model_id}
        timer("llm_call_duration", duration, labels)
        counter("llm_call_total", 1.0, labels)
        counter("llm_input_tokens_total", float(input_tokens), labels)
        counter("llm_output_tokens_total", float(output_tokens), labels)
    
    @staticmethod
    def record_database_query(duration: float, db_type: str, query_type: str) -> None:
        """Record database query metrics"""
        labels = {"db_type": db_type, "query_type": query_type}
        timer("database_query_duration", duration, labels)
        counter("database_query_total", 1.0, labels)
    
    @staticmethod
    def record_error(error_type: str, category: str, db_type: str) -> None:
        """Record error metrics"""
        labels = {"error_type": error_type, "category": category, "db_type": db_type}
        counter("errors_total", 1.0, labels)
    
    @staticmethod
    def set_active_connections(count: int, db_type: str) -> None:
        """Set active database connections gauge"""
        gauge("active_connections", float(count), {"db_type": db_type})
    
    @staticmethod
    def set_cache_stats(hits: int, misses: int, cache_type: str) -> None:
        """Set cache statistics"""
        counter("cache_hits_total", float(hits), {"cache_type": cache_type})
        counter("cache_misses_total", float(misses), {"cache_type": cache_type})

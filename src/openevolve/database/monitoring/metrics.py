"""
Database metrics collection and analysis.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    
    name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    sum_value: float
    last_value: float
    last_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "avg": round(self.avg_value, 2),
            "sum": self.sum_value,
            "last_value": self.last_value,
            "last_timestamp": self.last_timestamp.isoformat()
        }


class MetricsCollector:
    """
    Database metrics collector with aggregation and analysis capabilities.
    """
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.max_points_per_metric = max_points_per_metric
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self._metric_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._collection_task: Optional[asyncio.Task] = None
        self._is_collecting = False
        self._collection_interval = 60  # seconds
        
        logger.info("Metrics collector initialized")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if tags is None:
            tags = {}
        
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp,
            tags=tags
        )
        
        self._metrics[name].append(metric_point)
        
        # Trigger callbacks
        for callback in self._metric_callbacks[name]:
            try:
                callback(metric_point)
            except Exception as e:
                logger.error(f"Metric callback failed for {name}: {e}")
        
        logger.debug(f"Recorded metric {name}: {value}")
    
    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        self.record_metric(f"{name}.duration_ms", duration_ms, tags)
    
    def record_counter(self, name: str, increment: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        self.record_metric(f"{name}.count", increment, tags)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self.record_metric(f"{name}.gauge", value, tags)
    
    def add_metric_callback(self, metric_name: str, callback: Callable) -> None:
        """
        Add a callback for when a specific metric is recorded.
        
        Args:
            metric_name: Name of the metric to watch
            callback: Function to call when metric is recorded
        """
        self._metric_callbacks[metric_name].append(callback)
    
    def get_metric_summary(
        self,
        name: str,
        hours: int = 24
    ) -> Optional[MetricSummary]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            hours: Number of hours to include in summary
        
        Returns:
            MetricSummary or None if no data
        """
        if name not in self._metrics:
            return None
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter points within time range
        points = [
            point for point in self._metrics[name]
            if point.timestamp > cutoff_time
        ]
        
        if not points:
            return None
        
        values = [point.value for point in points]
        
        return MetricSummary(
            name=name,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            avg_value=sum(values) / len(values),
            sum_value=sum(values),
            last_value=values[-1],
            last_timestamp=points[-1].timestamp
        )
    
    def get_all_metrics_summary(self, hours: int = 24) -> Dict[str, MetricSummary]:
        """Get summary for all metrics."""
        summaries = {}
        
        for metric_name in self._metrics.keys():
            summary = self.get_metric_summary(metric_name, hours)
            if summary:
                summaries[metric_name] = summary
        
        return summaries
    
    def get_metric_points(
        self,
        name: str,
        hours: int = 24,
        limit: Optional[int] = None
    ) -> List[MetricPoint]:
        """
        Get raw metric points.
        
        Args:
            name: Metric name
            hours: Number of hours to include
            limit: Maximum number of points to return
        
        Returns:
            List of metric points
        """
        if name not in self._metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        points = [
            point for point in self._metrics[name]
            if point.timestamp > cutoff_time
        ]
        
        # Sort by timestamp
        points.sort(key=lambda p: p.timestamp)
        
        if limit:
            points = points[-limit:]
        
        return points
    
    def get_metric_rate(self, name: str, hours: int = 1) -> float:
        """
        Calculate the rate of a metric (events per hour).
        
        Args:
            name: Metric name
            hours: Time window in hours
        
        Returns:
            Rate per hour
        """
        points = self.get_metric_points(name, hours)
        
        if not points:
            return 0.0
        
        # For counters, sum the values
        if name.endswith('.count'):
            total = sum(point.value for point in points)
            return total / hours
        
        # For other metrics, count the number of points
        return len(points) / hours
    
    def get_percentile(self, name: str, percentile: float, hours: int = 24) -> Optional[float]:
        """
        Calculate percentile for a metric.
        
        Args:
            name: Metric name
            percentile: Percentile to calculate (0-100)
            hours: Time window in hours
        
        Returns:
            Percentile value or None
        """
        points = self.get_metric_points(name, hours)
        
        if not points:
            return None
        
        values = sorted([point.value for point in points])
        
        if not values:
            return None
        
        index = int((percentile / 100) * (len(values) - 1))
        return values[index]
    
    def start_collection(self, interval: int = 60) -> None:
        """
        Start automatic metrics collection.
        
        Args:
            interval: Collection interval in seconds
        """
        if self._is_collecting:
            return
        
        self._collection_interval = interval
        self._is_collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info(f"Started metrics collection with {interval}s interval")
    
    def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        if not self._is_collecting:
            return
        
        self._is_collecting = False
        
        if self._collection_task:
            self._collection_task.cancel()
        
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._is_collecting:
            try:
                await asyncio.sleep(self._collection_interval)
                
                if not self._is_collecting:
                    break
                
                # Collect system metrics
                await self._collect_system_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # Record collection timestamp
            self.record_metric("system.collection_timestamp", time.time())
            
            # Record number of metrics being tracked
            self.record_metric("system.tracked_metrics", len(self._metrics))
            
            # Record total data points
            total_points = sum(len(points) for points in self._metrics.values())
            self.record_metric("system.total_data_points", total_points)
            
            # Memory usage estimation (rough)
            estimated_memory_mb = total_points * 0.001  # Rough estimate
            self.record_metric("system.estimated_memory_mb", estimated_memory_mb)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def clear_old_metrics(self, hours: int = 168) -> int:
        """
        Clear metrics older than specified hours.
        
        Args:
            hours: Age threshold in hours (default: 1 week)
        
        Returns:
            Number of points removed
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        removed_count = 0
        
        for metric_name, points in self._metrics.items():
            original_count = len(points)
            
            # Filter out old points
            new_points = deque(
                (point for point in points if point.timestamp > cutoff_time),
                maxlen=self.max_points_per_metric
            )
            
            self._metrics[metric_name] = new_points
            removed_count += original_count - len(new_points)
        
        logger.info(f"Removed {removed_count} old metric points")
        return removed_count
    
    def export_metrics(self, format: str = "dict") -> Any:
        """
        Export metrics in various formats.
        
        Args:
            format: Export format ("dict", "json", "csv")
        
        Returns:
            Exported data
        """
        if format == "dict":
            return {
                name: [point.to_dict() for point in points]
                for name, points in self._metrics.items()
            }
        elif format == "json":
            import json
            return json.dumps(self.export_metrics("dict"), indent=2)
        elif format == "csv":
            # Simple CSV export
            lines = ["metric_name,value,timestamp,tags"]
            
            for name, points in self._metrics.items():
                for point in points:
                    tags_str = ";".join(f"{k}={v}" for k, v in point.tags.items())
                    lines.append(f"{name},{point.value},{point.timestamp.isoformat()},{tags_str}")
            
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display."""
        summaries = self.get_all_metrics_summary(hours=24)
        
        # Group metrics by category
        categories = defaultdict(list)
        
        for name, summary in summaries.items():
            if "." in name:
                category = name.split(".")[0]
            else:
                category = "general"
            
            categories[category].append(summary.to_dict())
        
        # Calculate overall statistics
        total_metrics = len(summaries)
        total_points = sum(summary.count for summary in summaries.values())
        
        return {
            "overview": {
                "total_metrics": total_metrics,
                "total_data_points": total_points,
                "collection_active": self._is_collecting,
                "collection_interval": self._collection_interval
            },
            "categories": dict(categories),
            "recent_activity": self._get_recent_activity()
        }
    
    def _get_recent_activity(self, minutes: int = 30) -> List[Dict[str, Any]]:
        """Get recent metric activity."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_points = []
        
        for name, points in self._metrics.items():
            for point in points:
                if point.timestamp > cutoff_time:
                    recent_points.append({
                        "metric": name,
                        "value": point.value,
                        "timestamp": point.timestamp.isoformat(),
                        "tags": point.tags
                    })
        
        # Sort by timestamp, most recent first
        recent_points.sort(key=lambda p: p["timestamp"], reverse=True)
        
        return recent_points[:100]  # Limit to 100 most recent


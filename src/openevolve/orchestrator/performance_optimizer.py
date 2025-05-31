"""
Performance Optimizer - System performance monitoring and optimization.

This module provides performance monitoring, bottleneck detection, and
automated optimization capabilities for the autonomous development pipeline.
"""

import asyncio
import logging
import time
import statistics
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import psutil
import gc

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    component: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    operation: str
    component: str
    duration: float
    throughput: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    success_rate: float = 1.0
    iterations: int = 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation."""
    component: str
    issue: str
    recommendation: str
    priority: str  # "low", "medium", "high", "critical"
    estimated_improvement: Optional[str] = None
    implementation_effort: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profile for a component or operation."""
    name: str
    metrics: List[PerformanceMetric] = field(default_factory=list)
    benchmarks: List[PerformanceBenchmark] = field(default_factory=list)
    recommendations: List[OptimizationRecommendation] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class PerformanceOptimizer:
    """
    Monitors system performance and provides optimization recommendations.
    
    Tracks performance metrics, identifies bottlenecks, and suggests
    optimizations for improved system efficiency.
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.MODERATE):
        """
        Initialize the performance optimizer.
        
        Args:
            optimization_level: Level of optimization aggressiveness
        """
        self.optimization_level = optimization_level
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.global_metrics: List[PerformanceMetric] = []
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.baseline_metrics: Dict[str, float] = {}
        
        # Performance thresholds
        self.thresholds = {
            "response_time_warning": 1000,  # ms
            "response_time_critical": 5000,  # ms
            "memory_usage_warning": 80,  # %
            "memory_usage_critical": 95,  # %
            "cpu_usage_warning": 70,  # %
            "cpu_usage_critical": 90,  # %
            "throughput_degradation": 0.2  # 20% degradation
        }
        
        logger.info(f"Performance Optimizer initialized with {optimization_level.value} optimization level")
    
    async def start_monitoring(self, interval: float = 60.0):
        """
        Start continuous performance monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self._collect_system_metrics()
                await self._analyze_performance()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric("system_cpu_usage", cpu_percent, "%", "system")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._add_metric("system_memory_usage", memory.percent, "%", "system")
            self._add_metric("system_memory_available", memory.available / (1024**3), "GB", "system")
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self._add_metric("disk_read_bytes", disk_io.read_bytes, "bytes", "system")
                self._add_metric("disk_write_bytes", disk_io.write_bytes, "bytes", "system")
            
            # Network I/O metrics
            net_io = psutil.net_io_counters()
            if net_io:
                self._add_metric("network_bytes_sent", net_io.bytes_sent, "bytes", "system")
                self._add_metric("network_bytes_recv", net_io.bytes_recv, "bytes", "system")
            
            # Process-specific metrics
            process = psutil.Process()
            self._add_metric("process_cpu_usage", process.cpu_percent(), "%", "process")
            self._add_metric("process_memory_usage", process.memory_info().rss / (1024**2), "MB", "process")
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _add_metric(self, name: str, value: float, unit: str, component: str, operation: str = None):
        """Add a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            component=component,
            operation=operation
        )
        
        self.global_metrics.append(metric)
        
        # Add to component profile
        if component not in self.profiles:
            self.profiles[component] = PerformanceProfile(name=component)
        
        self.profiles[component].metrics.append(metric)
        self.profiles[component].last_updated = time.time()
        
        # Keep only recent metrics (last 1000 per component)
        if len(self.profiles[component].metrics) > 1000:
            self.profiles[component].metrics = self.profiles[component].metrics[-1000:]
    
    async def benchmark_component(self, component: Any, component_name: str, 
                                operations: List[str] = None, iterations: int = 10) -> List[PerformanceBenchmark]:
        """
        Benchmark a component's performance.
        
        Args:
            component: Component to benchmark
            component_name: Name of the component
            operations: List of operations to benchmark
            iterations: Number of iterations per operation
            
        Returns:
            List of benchmark results
        """
        logger.info(f"Starting performance benchmark for {component_name}")
        
        benchmarks = []
        operations = operations or ['execute']  # Default operation
        
        for operation in operations:
            if not hasattr(component, operation):
                logger.warning(f"Operation {operation} not found on {component_name}")
                continue
            
            benchmark = await self._benchmark_operation(
                component, component_name, operation, iterations
            )
            benchmarks.append(benchmark)
            
            # Add to profile
            if component_name not in self.profiles:
                self.profiles[component_name] = PerformanceProfile(name=component_name)
            
            self.profiles[component_name].benchmarks.append(benchmark)
        
        logger.info(f"Benchmark completed for {component_name}: {len(benchmarks)} operations tested")
        return benchmarks
    
    async def _benchmark_operation(self, component: Any, component_name: str, 
                                 operation: str, iterations: int) -> PerformanceBenchmark:
        """Benchmark a specific operation."""
        method = getattr(component, operation)
        durations = []
        memory_usages = []
        cpu_usages = []
        successes = 0
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
        
        for i in range(iterations):
            try:
                # Measure CPU before
                cpu_before = process.cpu_percent()
                
                # Measure memory before
                memory_before = process.memory_info().rss
                
                # Execute operation
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()
                
                duration = (time.time() - start_time) * 1000  # Convert to ms
                durations.append(duration)
                
                # Measure memory after
                memory_after = process.memory_info().rss
                memory_usage = (memory_after - baseline_memory) / (1024**2)  # MB
                memory_usages.append(memory_usage)
                
                # Measure CPU after
                cpu_after = process.cpu_percent()
                cpu_usage = max(0, cpu_after - cpu_before)
                cpu_usages.append(cpu_usage)
                
                successes += 1
                
                # Small delay between iterations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed for {component_name}.{operation}: {e}")
                continue
        
        # Calculate statistics
        avg_duration = statistics.mean(durations) if durations else 0
        avg_memory = statistics.mean(memory_usages) if memory_usages else 0
        avg_cpu = statistics.mean(cpu_usages) if cpu_usages else 0
        success_rate = successes / iterations
        throughput = (successes / (avg_duration / 1000)) if avg_duration > 0 else 0
        
        return PerformanceBenchmark(
            operation=operation,
            component=component_name,
            duration=avg_duration,
            throughput=throughput,
            memory_usage=avg_memory,
            cpu_usage=avg_cpu,
            success_rate=success_rate,
            iterations=iterations,
            metadata={
                "duration_std": statistics.stdev(durations) if len(durations) > 1 else 0,
                "duration_min": min(durations) if durations else 0,
                "duration_max": max(durations) if durations else 0,
                "memory_std": statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0,
                "cpu_std": statistics.stdev(cpu_usages) if len(cpu_usages) > 1 else 0
            }
        )
    
    async def _analyze_performance(self):
        """Analyze performance and generate recommendations."""
        for component_name, profile in self.profiles.items():
            recommendations = []
            
            # Analyze recent metrics
            recent_metrics = [m for m in profile.metrics if time.time() - m.timestamp < 300]  # Last 5 minutes
            
            if not recent_metrics:
                continue
            
            # Group metrics by name
            metric_groups = {}
            for metric in recent_metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric.value)
            
            # Analyze each metric type
            for metric_name, values in metric_groups.items():
                if not values:
                    continue
                
                avg_value = statistics.mean(values)
                
                # Check for performance issues
                if "response_time" in metric_name or "duration" in metric_name:
                    if avg_value > self.thresholds["response_time_critical"]:
                        recommendations.append(OptimizationRecommendation(
                            component=component_name,
                            issue=f"Critical response time: {avg_value:.2f}ms",
                            recommendation="Investigate bottlenecks, consider caching, optimize algorithms",
                            priority="critical",
                            estimated_improvement="50-80% response time reduction",
                            implementation_effort="medium"
                        ))
                    elif avg_value > self.thresholds["response_time_warning"]:
                        recommendations.append(OptimizationRecommendation(
                            component=component_name,
                            issue=f"High response time: {avg_value:.2f}ms",
                            recommendation="Profile code, optimize hot paths, consider async operations",
                            priority="medium",
                            estimated_improvement="20-50% response time reduction",
                            implementation_effort="low"
                        ))
                
                elif "memory" in metric_name and "usage" in metric_name:
                    if avg_value > self.thresholds["memory_usage_critical"]:
                        recommendations.append(OptimizationRecommendation(
                            component=component_name,
                            issue=f"Critical memory usage: {avg_value:.2f}%",
                            recommendation="Implement memory pooling, reduce object creation, add garbage collection",
                            priority="critical",
                            estimated_improvement="30-60% memory reduction",
                            implementation_effort="high"
                        ))
                    elif avg_value > self.thresholds["memory_usage_warning"]:
                        recommendations.append(OptimizationRecommendation(
                            component=component_name,
                            issue=f"High memory usage: {avg_value:.2f}%",
                            recommendation="Review data structures, implement lazy loading, optimize caching",
                            priority="medium",
                            estimated_improvement="15-30% memory reduction",
                            implementation_effort="medium"
                        ))
                
                elif "cpu" in metric_name and "usage" in metric_name:
                    if avg_value > self.thresholds["cpu_usage_critical"]:
                        recommendations.append(OptimizationRecommendation(
                            component=component_name,
                            issue=f"Critical CPU usage: {avg_value:.2f}%",
                            recommendation="Parallelize operations, optimize algorithms, consider worker processes",
                            priority="critical",
                            estimated_improvement="40-70% CPU reduction",
                            implementation_effort="high"
                        ))
                    elif avg_value > self.thresholds["cpu_usage_warning"]:
                        recommendations.append(OptimizationRecommendation(
                            component=component_name,
                            issue=f"High CPU usage: {avg_value:.2f}%",
                            recommendation="Profile CPU usage, optimize loops, use more efficient algorithms",
                            priority="medium",
                            estimated_improvement="20-40% CPU reduction",
                            implementation_effort="medium"
                        ))
            
            # Update profile with new recommendations
            profile.recommendations = recommendations
            profile.last_updated = time.time()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of system performance."""
        summary = {
            "overall_status": "healthy",
            "components": {},
            "global_metrics": {},
            "recommendations_count": 0,
            "last_updated": time.time()
        }
        
        # Analyze component performance
        critical_issues = 0
        high_issues = 0
        
        for component_name, profile in self.profiles.items():
            component_summary = {
                "status": "healthy",
                "metrics_count": len(profile.metrics),
                "benchmarks_count": len(profile.benchmarks),
                "recommendations": len(profile.recommendations),
                "last_updated": profile.last_updated
            }
            
            # Check for critical recommendations
            critical_recs = [r for r in profile.recommendations if r.priority == "critical"]
            high_recs = [r for r in profile.recommendations if r.priority == "high"]
            
            if critical_recs:
                component_summary["status"] = "critical"
                critical_issues += len(critical_recs)
            elif high_recs:
                component_summary["status"] = "warning"
                high_issues += len(high_recs)
            
            summary["components"][component_name] = component_summary
            summary["recommendations_count"] += len(profile.recommendations)
        
        # Determine overall status
        if critical_issues > 0:
            summary["overall_status"] = "critical"
        elif high_issues > 0:
            summary["overall_status"] = "warning"
        
        # Add recent global metrics
        recent_metrics = [m for m in self.global_metrics if time.time() - m.timestamp < 300]
        metric_groups = {}
        for metric in recent_metrics:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = []
            metric_groups[metric.name].append(metric.value)
        
        for metric_name, values in metric_groups.items():
            if values:
                summary["global_metrics"][metric_name] = {
                    "current": values[-1],
                    "average": statistics.mean(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary
    
    def get_detailed_performance(self) -> Dict[str, Any]:
        """Get detailed performance information."""
        return {
            "profiles": {
                name: {
                    "name": profile.name,
                    "metrics": [
                        {
                            "name": m.name,
                            "value": m.value,
                            "unit": m.unit,
                            "timestamp": m.timestamp,
                            "component": m.component,
                            "operation": m.operation
                        }
                        for m in profile.metrics[-100:]  # Last 100 metrics
                    ],
                    "benchmarks": [
                        {
                            "operation": b.operation,
                            "component": b.component,
                            "duration": b.duration,
                            "throughput": b.throughput,
                            "memory_usage": b.memory_usage,
                            "cpu_usage": b.cpu_usage,
                            "success_rate": b.success_rate,
                            "iterations": b.iterations,
                            "timestamp": b.timestamp,
                            "metadata": b.metadata
                        }
                        for b in profile.benchmarks
                    ],
                    "recommendations": [
                        {
                            "component": r.component,
                            "issue": r.issue,
                            "recommendation": r.recommendation,
                            "priority": r.priority,
                            "estimated_improvement": r.estimated_improvement,
                            "implementation_effort": r.implementation_effort,
                            "metadata": r.metadata
                        }
                        for r in profile.recommendations
                    ],
                    "last_updated": profile.last_updated
                }
                for name, profile in self.profiles.items()
            },
            "optimization_level": self.optimization_level.value,
            "thresholds": self.thresholds,
            "monitoring_active": self.monitoring_active
        }
    
    async def optimize_component(self, component_name: str, 
                               recommendations: List[str] = None) -> Dict[str, Any]:
        """
        Apply optimizations to a component.
        
        Args:
            component_name: Name of the component to optimize
            recommendations: Specific recommendations to apply
            
        Returns:
            Optimization results
        """
        if component_name not in self.profiles:
            raise ValueError(f"Component not found: {component_name}")
        
        profile = self.profiles[component_name]
        results = {
            "component": component_name,
            "optimizations_applied": [],
            "before_metrics": {},
            "after_metrics": {},
            "improvement": {}
        }
        
        # Get baseline metrics
        recent_metrics = [m for m in profile.metrics if time.time() - m.timestamp < 300]
        if recent_metrics:
            metric_groups = {}
            for metric in recent_metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric.value)
            
            for metric_name, values in metric_groups.items():
                if values:
                    results["before_metrics"][metric_name] = statistics.mean(values)
        
        # Apply optimizations based on recommendations
        applied_optimizations = []
        
        for recommendation in profile.recommendations:
            if recommendations and recommendation.recommendation not in recommendations:
                continue
            
            try:
                # Apply specific optimizations
                if "garbage collection" in recommendation.recommendation.lower():
                    gc.collect()
                    applied_optimizations.append("garbage_collection")
                
                # Add more optimization implementations here
                
            except Exception as e:
                logger.error(f"Failed to apply optimization: {e}")
        
        results["optimizations_applied"] = applied_optimizations
        
        # Wait a bit and collect new metrics
        await asyncio.sleep(30)
        await self._collect_system_metrics()
        
        # Get after metrics
        recent_metrics = [m for m in profile.metrics if time.time() - m.timestamp < 60]
        if recent_metrics:
            metric_groups = {}
            for metric in recent_metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric.value)
            
            for metric_name, values in metric_groups.items():
                if values:
                    results["after_metrics"][metric_name] = statistics.mean(values)
        
        # Calculate improvements
        for metric_name in results["before_metrics"]:
            if metric_name in results["after_metrics"]:
                before = results["before_metrics"][metric_name]
                after = results["after_metrics"][metric_name]
                improvement = ((before - after) / before) * 100 if before > 0 else 0
                results["improvement"][metric_name] = improvement
        
        logger.info(f"Optimization completed for {component_name}: {len(applied_optimizations)} optimizations applied")
        return results


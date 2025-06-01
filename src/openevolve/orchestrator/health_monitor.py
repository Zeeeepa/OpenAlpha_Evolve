"""
Health Monitor - System health monitoring and alerting.

This module provides comprehensive health monitoring capabilities including
component health checks, performance monitoring, and automated alerting.
"""

import asyncio
import logging
import time
import psutil
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import sys

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Any
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_name: str
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    last_check: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    uptime: float = 0.0
    
    def add_metric(self, metric: HealthMetric):
        """Add a health metric."""
        self.metrics.append(metric)
        
        # Update overall status based on metric status
        if metric.status == HealthStatus.CRITICAL:
            self.status = HealthStatus.CRITICAL
        elif metric.status == HealthStatus.WARNING and self.status != HealthStatus.CRITICAL:
            self.status = HealthStatus.WARNING


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    system_metrics: List[HealthMetric] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    alerts: List[str] = field(default_factory=list)


class HealthMonitor:
    """
    Monitors system health and component status.
    
    Provides real-time health monitoring, alerting, and performance tracking
    for all system components.
    """
    
    def __init__(self, check_interval: float = 30.0):
        """
        Initialize the health monitor.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.health_status = SystemHealth()
        self.component_checkers: Dict[str, Callable] = {}
        self.alert_handlers: List[Callable] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        self.start_time = time.time()
        
        logger.info("Health Monitor initialized")
    
    def register_component_checker(self, component_name: str, checker: Callable):
        """
        Register a health checker for a component.
        
        Args:
            component_name: Name of the component
            checker: Async function that returns ComponentHealth
        """
        self.component_checkers[component_name] = checker
        logger.info(f"Registered health checker for: {component_name}")
    
    def register_alert_handler(self, handler: Callable):
        """
        Register an alert handler.
        
        Args:
            handler: Function to handle alerts
        """
        self.alert_handlers.append(handler)
        logger.info("Registered alert handler")
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._running:
            logger.warning("Health monitoring already running")
            return
        
        self._running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self.check_system_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_system_health(self) -> SystemHealth:
        """
        Perform comprehensive system health check.
        
        Returns:
            Current system health status
        """
        logger.debug("Performing system health check")
        
        # Reset health status
        self.health_status = SystemHealth()
        
        # Check system metrics
        await self._check_system_metrics()
        
        # Check component health
        await self._check_component_health()
        
        # Determine overall status
        self._determine_overall_status()
        
        # Process alerts
        await self._process_alerts()
        
        self.health_status.last_update = time.time()
        
        logger.debug(f"System health check completed: {self.health_status.overall_status.value}")
        return self.health_status
    
    async def _check_system_metrics(self):
        """Check system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                status=self._get_metric_status(cpu_percent, 70, 90),
                threshold_warning=70.0,
                threshold_critical=90.0,
                unit="%",
                description="CPU usage percentage"
            )
            self.health_status.system_metrics.append(cpu_metric)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_metric = HealthMetric(
                name="memory_usage",
                value=memory_percent,
                status=self._get_metric_status(memory_percent, 80, 95),
                threshold_warning=80.0,
                threshold_critical=95.0,
                unit="%",
                description="Memory usage percentage"
            )
            self.health_status.system_metrics.append(memory_metric)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_metric = HealthMetric(
                name="disk_usage",
                value=disk_percent,
                status=self._get_metric_status(disk_percent, 80, 95),
                threshold_warning=80.0,
                threshold_critical=95.0,
                unit="%",
                description="Disk usage percentage"
            )
            self.health_status.system_metrics.append(disk_metric)
            
            # System uptime
            uptime = time.time() - self.start_time
            uptime_metric = HealthMetric(
                name="uptime",
                value=uptime,
                status=HealthStatus.HEALTHY,
                unit="seconds",
                description="System uptime"
            )
            self.health_status.system_metrics.append(uptime_metric)
            
            # Load average (Unix systems only)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()[0]  # 1-minute load average
                cpu_count = psutil.cpu_count()
                load_percent = (load_avg / cpu_count) * 100
                load_metric = HealthMetric(
                    name="load_average",
                    value=load_avg,
                    status=self._get_metric_status(load_percent, 70, 90),
                    threshold_warning=cpu_count * 0.7,
                    threshold_critical=cpu_count * 0.9,
                    unit="",
                    description="1-minute load average"
                )
                self.health_status.system_metrics.append(load_metric)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            error_metric = HealthMetric(
                name="system_metrics_error",
                value=str(e),
                status=HealthStatus.CRITICAL,
                description="Error collecting system metrics"
            )
            self.health_status.system_metrics.append(error_metric)
    
    async def _check_component_health(self):
        """Check health of all registered components."""
        for component_name, checker in self.component_checkers.items():
            try:
                component_health = await checker()
                if not isinstance(component_health, ComponentHealth):
                    # Convert simple boolean or dict response
                    if isinstance(component_health, bool):
                        component_health = ComponentHealth(
                            component_name=component_name,
                            status=HealthStatus.HEALTHY if component_health else HealthStatus.CRITICAL
                        )
                    elif isinstance(component_health, dict):
                        component_health = ComponentHealth(
                            component_name=component_name,
                            status=HealthStatus.HEALTHY if component_health.get('healthy', False) else HealthStatus.CRITICAL,
                            error_message=component_health.get('error')
                        )
                
                self.health_status.components[component_name] = component_health
                
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                error_health = ComponentHealth(
                    component_name=component_name,
                    status=HealthStatus.CRITICAL,
                    error_message=str(e)
                )
                self.health_status.components[component_name] = error_health
    
    def _get_metric_status(self, value: float, warning_threshold: float, critical_threshold: float) -> HealthStatus:
        """Determine metric status based on thresholds."""
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _determine_overall_status(self):
        """Determine overall system health status."""
        # Check system metrics
        system_critical = any(m.status == HealthStatus.CRITICAL for m in self.health_status.system_metrics)
        system_warning = any(m.status == HealthStatus.WARNING for m in self.health_status.system_metrics)
        
        # Check component health
        component_critical = any(c.status == HealthStatus.CRITICAL for c in self.health_status.components.values())
        component_warning = any(c.status == HealthStatus.WARNING for c in self.health_status.components.values())
        
        if system_critical or component_critical:
            self.health_status.overall_status = HealthStatus.CRITICAL
        elif system_warning or component_warning:
            self.health_status.overall_status = HealthStatus.WARNING
        else:
            self.health_status.overall_status = HealthStatus.HEALTHY
    
    async def _process_alerts(self):
        """Process and send alerts based on health status."""
        alerts = []
        
        # Check for critical system metrics
        for metric in self.health_status.system_metrics:
            if metric.status == HealthStatus.CRITICAL:
                alerts.append(f"CRITICAL: {metric.name} is {metric.value}{metric.unit or ''} "
                            f"(threshold: {metric.threshold_critical})")
            elif metric.status == HealthStatus.WARNING:
                alerts.append(f"WARNING: {metric.name} is {metric.value}{metric.unit or ''} "
                            f"(threshold: {metric.threshold_warning})")
        
        # Check for component issues
        for component in self.health_status.components.values():
            if component.status == HealthStatus.CRITICAL:
                alerts.append(f"CRITICAL: Component {component.component_name} is unhealthy"
                            f"{': ' + component.error_message if component.error_message else ''}")
            elif component.status == HealthStatus.WARNING:
                alerts.append(f"WARNING: Component {component.component_name} has issues"
                            f"{': ' + component.error_message if component.error_message else ''}")
        
        # Send alerts
        if alerts:
            self.health_status.alerts = alerts
            for handler in self.alert_handlers:
                try:
                    await handler(alerts)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        return {
            "overall_status": self.health_status.overall_status.value,
            "last_update": self.health_status.last_update,
            "uptime": time.time() - self.start_time,
            "components": {
                name: {
                    "status": component.status.value,
                    "last_check": component.last_check,
                    "uptime": component.uptime,
                    "error": component.error_message,
                    "metrics_count": len(component.metrics)
                }
                for name, component in self.health_status.components.items()
            },
            "system_metrics": {
                metric.name: {
                    "value": metric.value,
                    "status": metric.status.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp
                }
                for metric in self.health_status.system_metrics
            },
            "alerts": self.health_status.alerts
        }
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information."""
        return {
            "overall_status": self.health_status.overall_status.value,
            "last_update": self.health_status.last_update,
            "uptime": time.time() - self.start_time,
            "components": {
                name: {
                    "status": component.status.value,
                    "last_check": component.last_check,
                    "uptime": component.uptime,
                    "error": component.error_message,
                    "metrics": [
                        {
                            "name": metric.name,
                            "value": metric.value,
                            "status": metric.status.value,
                            "threshold_warning": metric.threshold_warning,
                            "threshold_critical": metric.threshold_critical,
                            "unit": metric.unit,
                            "description": metric.description,
                            "timestamp": metric.timestamp
                        }
                        for metric in component.metrics
                    ]
                }
                for name, component in self.health_status.components.items()
            },
            "system_metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "status": metric.status.value,
                    "threshold_warning": metric.threshold_warning,
                    "threshold_critical": metric.threshold_critical,
                    "unit": metric.unit,
                    "description": metric.description,
                    "timestamp": metric.timestamp
                }
                for metric in self.health_status.system_metrics
            ],
            "alerts": self.health_status.alerts
        }
    
    async def create_component_health_checker(self, agent: Any, component_name: str) -> Callable:
        """
        Create a health checker for a component/agent.
        
        Args:
            agent: The agent instance to check
            component_name: Name of the component
            
        Returns:
            Health checker function
        """
        async def health_checker() -> ComponentHealth:
            component_health = ComponentHealth(
                component_name=component_name,
                status=HealthStatus.HEALTHY
            )
            
            try:
                # Check if agent has a health_check method
                if hasattr(agent, 'health_check'):
                    result = await agent.health_check()
                    if not result:
                        component_health.status = HealthStatus.CRITICAL
                        component_health.error_message = "Health check returned False"
                else:
                    # Basic health check - verify agent is responsive
                    if not hasattr(agent, 'execute'):
                        component_health.status = HealthStatus.CRITICAL
                        component_health.error_message = "Agent missing execute method"
                
                # Add basic metrics
                response_time_start = time.time()
                if hasattr(agent, 'ping') or hasattr(agent, 'health_check'):
                    # Measure response time
                    if hasattr(agent, 'ping'):
                        await agent.ping()
                    response_time = (time.time() - response_time_start) * 1000
                    
                    response_metric = HealthMetric(
                        name="response_time",
                        value=response_time,
                        status=self._get_metric_status(response_time, 1000, 5000),
                        threshold_warning=1000.0,
                        threshold_critical=5000.0,
                        unit="ms",
                        description="Component response time"
                    )
                    component_health.add_metric(response_metric)
                
            except Exception as e:
                component_health.status = HealthStatus.CRITICAL
                component_health.error_message = str(e)
            
            return component_health
        
        return health_checker


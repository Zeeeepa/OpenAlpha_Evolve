"""
Database health monitoring system.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..config import DatabaseConfig
from ..connectors.postgresql import PostgreSQLConnector
from ..connectors.pool_manager import ConnectionPoolManager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    name: str
    status: HealthStatus
    response_time_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "response_time_ms": round(self.response_time_ms, 2),
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class HealthThresholds:
    """Health check thresholds."""
    
    response_time_warning_ms: float = 1000.0
    response_time_critical_ms: float = 5000.0
    connection_usage_warning: float = 0.8  # 80%
    connection_usage_critical: float = 0.95  # 95%
    error_rate_warning: float = 0.05  # 5%
    error_rate_critical: float = 0.1  # 10%


class HealthMonitor:
    """
    Comprehensive database health monitoring system with
    configurable checks and alerting.
    """
    
    def __init__(
        self,
        connector: PostgreSQLConnector,
        pool_manager: Optional[ConnectionPoolManager] = None,
        config: Optional[DatabaseConfig] = None
    ):
        self.connector = connector
        self.pool_manager = pool_manager
        self.config = config or DatabaseConfig()
        self.thresholds = HealthThresholds()
        
        self._health_checks: Dict[str, Callable] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        self._health_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000
        self._alert_callbacks: List[Callable] = []
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info("Health monitor initialized")
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self._health_checks = {
            "database_connectivity": self._check_database_connectivity,
            "connection_pool": self._check_connection_pool,
            "query_performance": self._check_query_performance,
            "disk_space": self._check_disk_space,
            "active_connections": self._check_active_connections,
            "replication_lag": self._check_replication_lag
        }
    
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """
        Register a custom health check.
        
        Args:
            name: Name of the health check
            check_func: Async function that returns HealthCheckResult
        """
        self._health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """
        Add an alert callback function.
        
        Args:
            callback: Function to call when health status changes
        """
        self._alert_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if not self._is_monitoring:
                    break
                
                # Perform health checks
                health_status = await self.check_health()
                
                # Store in history
                self._store_health_result(health_status)
                
                # Check for alerts
                await self._check_alerts(health_status)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Perform all health checks.
        
        Returns:
            Comprehensive health status
        """
        start_time = asyncio.get_event_loop().time()
        
        check_results = {}
        overall_status = HealthStatus.HEALTHY
        
        # Run all health checks
        for check_name, check_func in self._health_checks.items():
            try:
                result = await check_func()
                check_results[check_name] = result.to_dict()
                
                # Update overall status
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                check_results[check_name] = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0,
                    message=f"Check failed: {e}"
                ).to_dict()
                
                overall_status = HealthStatus.UNHEALTHY
        
        total_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return {
            "overall_status": overall_status.value,
            "total_check_time_ms": round(total_time, 2),
            "timestamp": datetime.now().isoformat(),
            "checks": check_results
        }
    
    async def _check_database_connectivity(self) -> HealthCheckResult:
        """Check basic database connectivity."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simple connectivity test
            result = await self.connector.execute_query("SELECT 1", fetch_mode="val")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if result == 1:
                status = HealthStatus.HEALTHY
                message = "Database connection successful"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Unexpected query result"
            
            # Check response time thresholds
            if response_time > self.thresholds.response_time_critical_ms:
                status = HealthStatus.UNHEALTHY
                message = f"Critical response time: {response_time:.2f}ms"
            elif response_time > self.thresholds.response_time_warning_ms:
                status = HealthStatus.DEGRADED
                message = f"Slow response time: {response_time:.2f}ms"
            
            return HealthCheckResult(
                name="database_connectivity",
                status=status,
                response_time_ms=response_time,
                message=message
            )
            
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return HealthCheckResult(
                name="database_connectivity",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Connection failed: {e}"
            )
    
    async def _check_connection_pool(self) -> HealthCheckResult:
        """Check connection pool health."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if self.pool_manager:
                pool_status = await self.pool_manager.get_pool_status()
                metrics = pool_status.get("metrics", {})
            else:
                pool_status = await self.connector.get_pool_status()
                metrics = {}
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Analyze pool health
            if pool_status.get("status") == "healthy":
                # Check connection usage
                total_connections = pool_status.get("size", 0)
                idle_connections = pool_status.get("idle_connections", 0)
                active_connections = total_connections - idle_connections
                
                if total_connections > 0:
                    usage_ratio = active_connections / total_connections
                    
                    if usage_ratio > self.thresholds.connection_usage_critical:
                        status = HealthStatus.UNHEALTHY
                        message = f"Critical connection usage: {usage_ratio:.1%}"
                    elif usage_ratio > self.thresholds.connection_usage_warning:
                        status = HealthStatus.DEGRADED
                        message = f"High connection usage: {usage_ratio:.1%}"
                    else:
                        status = HealthStatus.HEALTHY
                        message = f"Connection pool healthy: {usage_ratio:.1%} usage"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = "No active connections"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Connection pool unhealthy"
            
            return HealthCheckResult(
                name="connection_pool",
                status=status,
                response_time_ms=response_time,
                message=message,
                details=pool_status
            )
            
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return HealthCheckResult(
                name="connection_pool",
                status=HealthStatus.UNKNOWN,
                response_time_ms=response_time,
                message=f"Pool check failed: {e}"
            )
    
    async def _check_query_performance(self) -> HealthCheckResult:
        """Check query performance."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Test query performance with a more complex query
            test_query = """
            SELECT 
                COUNT(*) as count,
                AVG(EXTRACT(EPOCH FROM NOW() - NOW())) as avg_time
            FROM generate_series(1, 1000) as s(i)
            """
            
            result = await self.connector.execute_query(test_query, fetch_mode="one")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Determine status based on response time
            if response_time > self.thresholds.response_time_critical_ms:
                status = HealthStatus.UNHEALTHY
                message = f"Critical query performance: {response_time:.2f}ms"
            elif response_time > self.thresholds.response_time_warning_ms:
                status = HealthStatus.DEGRADED
                message = f"Slow query performance: {response_time:.2f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Query performance good: {response_time:.2f}ms"
            
            return HealthCheckResult(
                name="query_performance",
                status=status,
                response_time_ms=response_time,
                message=message,
                details={"test_result": result}
            )
            
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return HealthCheckResult(
                name="query_performance",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Query performance check failed: {e}"
            )
    
    async def _check_disk_space(self) -> HealthCheckResult:
        """Check database disk space."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Query disk space information
            disk_query = """
            SELECT 
                pg_size_pretty(pg_database_size(current_database())) as db_size,
                pg_database_size(current_database()) as db_size_bytes
            """
            
            result = await self.connector.execute_query(disk_query, fetch_mode="one")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # For now, just report the size - in production, you'd compare against limits
            status = HealthStatus.HEALTHY
            message = f"Database size: {result['db_size']}"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                response_time_ms=response_time,
                message=message,
                details=result
            )
            
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                response_time_ms=response_time,
                message=f"Disk space check failed: {e}"
            )
    
    async def _check_active_connections(self) -> HealthCheckResult:
        """Check active database connections."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Query active connections
            connections_query = """
            SELECT 
                COUNT(*) as total_connections,
                COUNT(*) FILTER (WHERE state = 'active') as active_connections,
                COUNT(*) FILTER (WHERE state = 'idle') as idle_connections
            FROM pg_stat_activity
            WHERE datname = current_database()
            """
            
            result = await self.connector.execute_query(connections_query, fetch_mode="one")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            total = result["total_connections"]
            active = result["active_connections"]
            
            # Determine status based on connection count
            # These thresholds should be configurable based on your setup
            if total > 100:  # Adjust based on your max_connections setting
                status = HealthStatus.DEGRADED
                message = f"High connection count: {total} total, {active} active"
            elif total > 150:
                status = HealthStatus.UNHEALTHY
                message = f"Critical connection count: {total} total, {active} active"
            else:
                status = HealthStatus.HEALTHY
                message = f"Connection count normal: {total} total, {active} active"
            
            return HealthCheckResult(
                name="active_connections",
                status=status,
                response_time_ms=response_time,
                message=message,
                details=result
            )
            
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return HealthCheckResult(
                name="active_connections",
                status=HealthStatus.UNKNOWN,
                response_time_ms=response_time,
                message=f"Active connections check failed: {e}"
            )
    
    async def _check_replication_lag(self) -> HealthCheckResult:
        """Check replication lag (if applicable)."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Check if this is a replica
            replica_query = "SELECT pg_is_in_recovery()"
            is_replica = await self.connector.execute_query(replica_query, fetch_mode="val")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if not is_replica:
                return HealthCheckResult(
                    name="replication_lag",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time,
                    message="Primary database - no replication lag",
                    details={"is_replica": False}
                )
            
            # If it's a replica, check lag
            lag_query = """
            SELECT 
                EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) as lag_seconds
            """
            
            lag_result = await self.connector.execute_query(lag_query, fetch_mode="one")
            lag_seconds = lag_result["lag_seconds"] or 0
            
            # Determine status based on lag
            if lag_seconds > 60:  # 1 minute
                status = HealthStatus.UNHEALTHY
                message = f"Critical replication lag: {lag_seconds:.1f}s"
            elif lag_seconds > 10:  # 10 seconds
                status = HealthStatus.DEGRADED
                message = f"High replication lag: {lag_seconds:.1f}s"
            else:
                status = HealthStatus.HEALTHY
                message = f"Replication lag normal: {lag_seconds:.1f}s"
            
            return HealthCheckResult(
                name="replication_lag",
                status=status,
                response_time_ms=response_time,
                message=message,
                details={"is_replica": True, "lag_seconds": lag_seconds}
            )
            
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return HealthCheckResult(
                name="replication_lag",
                status=HealthStatus.UNKNOWN,
                response_time_ms=response_time,
                message=f"Replication lag check failed: {e}"
            )
    
    def _store_health_result(self, health_status: Dict[str, Any]) -> None:
        """Store health check result in history."""
        self._health_history.append(health_status)
        
        # Limit history size
        if len(self._health_history) > self._max_history_size:
            self._health_history = self._health_history[-self._max_history_size:]
    
    async def _check_alerts(self, health_status: Dict[str, Any]) -> None:
        """Check for alert conditions and notify callbacks."""
        overall_status = health_status.get("overall_status")
        
        # Only alert on status changes or critical issues
        if overall_status in ["unhealthy", "degraded"]:
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(health_status)
                    else:
                        callback(health_status)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get health check history.
        
        Args:
            hours: Number of hours of history to return
        
        Returns:
            List of health check results
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            result for result in self._health_history
            if datetime.fromisoformat(result["timestamp"]) > cutoff_time
        ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary statistics."""
        if not self._health_history:
            return {"status": "no_data"}
        
        recent_results = self.get_health_history(hours=1)  # Last hour
        
        if not recent_results:
            return {"status": "no_recent_data"}
        
        # Calculate uptime percentage
        healthy_count = sum(
            1 for result in recent_results
            if result.get("overall_status") == "healthy"
        )
        
        uptime_percentage = (healthy_count / len(recent_results)) * 100
        
        # Get latest status
        latest_result = self._health_history[-1]
        
        return {
            "current_status": latest_result.get("overall_status"),
            "uptime_percentage_1h": round(uptime_percentage, 2),
            "total_checks_1h": len(recent_results),
            "healthy_checks_1h": healthy_count,
            "last_check": latest_result.get("timestamp"),
            "average_response_time_ms": round(
                sum(result.get("total_check_time_ms", 0) for result in recent_results) / 
                len(recent_results), 2
            )
        }


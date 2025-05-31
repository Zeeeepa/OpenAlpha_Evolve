"""
Connection pool manager for optimized database connection handling.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import weakref

from ..config import DatabaseConfig
from ..exceptions import DatabaseConnectionError, DatabasePoolExhaustedError
from .postgresql import PostgreSQLConnector

logger = logging.getLogger(__name__)


@dataclass
class PoolMetrics:
    """Connection pool metrics."""
    
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    peak_connections: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    
    def reset(self):
        """Reset metrics."""
        self.total_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0
        self.peak_connections = 0
        self.last_reset = datetime.now()


class ConnectionPoolManager:
    """
    Advanced connection pool manager with monitoring, optimization,
    and automatic scaling capabilities.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connectors: Dict[str, PostgreSQLConnector] = {}
        self.metrics = PoolMetrics()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._connection_refs: Set[weakref.ref] = set()
        
        logger.info("Connection pool manager initialized")
    
    async def start(self) -> None:
        """Start the pool manager and monitoring tasks."""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Create default connector
        await self.add_connector("default", self.config)
        
        # Start monitoring tasks
        if self.config.enable_monitoring:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info("Connection pool manager started")
    
    async def stop(self) -> None:
        """Stop the pool manager and close all connections."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Cancel monitoring tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        # Close all connectors
        for connector in self.connectors.values():
            await connector.close()
        
        self.connectors.clear()
        logger.info("Connection pool manager stopped")
    
    async def add_connector(self, name: str, config: DatabaseConfig) -> None:
        """Add a new database connector."""
        if name in self.connectors:
            logger.warning(f"Connector '{name}' already exists, replacing")
            await self.connectors[name].close()
        
        connector = PostgreSQLConnector(config)
        await connector.initialize()
        self.connectors[name] = connector
        
        logger.info(f"Added connector '{name}' for {config.host}:{config.port}")
    
    async def remove_connector(self, name: str) -> None:
        """Remove a database connector."""
        if name not in self.connectors:
            logger.warning(f"Connector '{name}' not found")
            return
        
        await self.connectors[name].close()
        del self.connectors[name]
        
        logger.info(f"Removed connector '{name}'")
    
    def get_connector(self, name: str = "default") -> PostgreSQLConnector:
        """Get a database connector by name."""
        if name not in self.connectors:
            raise DatabaseConnectionError(f"Connector '{name}' not found")
        
        return self.connectors[name]
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_mode: str = "all",
        connector_name: str = "default"
    ) -> Any:
        """Execute a query using the specified connector."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            connector = self.get_connector(connector_name)
            result = await connector.execute_query(query, parameters, fetch_mode)
            
            # Update metrics
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self._update_metrics(response_time, success=True)
            
            return result
            
        except Exception as e:
            self._update_metrics(0, success=False)
            raise
    
    async def execute_transaction(
        self,
        queries: List[tuple],
        connector_name: str = "default"
    ) -> List[Any]:
        """Execute a transaction using the specified connector."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            connector = self.get_connector(connector_name)
            result = await connector.execute_transaction(queries)
            
            # Update metrics
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self._update_metrics(response_time, success=True)
            
            return result
            
        except Exception as e:
            self._update_metrics(0, success=False)
            raise
    
    def _update_metrics(self, response_time: float, success: bool) -> None:
        """Update pool metrics."""
        self.metrics.total_requests += 1
        
        if success:
            # Update average response time
            if self.metrics.average_response_time == 0:
                self.metrics.average_response_time = response_time
            else:
                # Exponential moving average
                alpha = 0.1
                self.metrics.average_response_time = (
                    alpha * response_time + 
                    (1 - alpha) * self.metrics.average_response_time
                )
        else:
            self.metrics.failed_requests += 1
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive pool status."""
        connector_status = {}
        total_connections = 0
        total_idle = 0
        
        for name, connector in self.connectors.items():
            status = await connector.get_pool_status()
            connector_status[name] = status
            
            if isinstance(status.get("size"), int):
                total_connections += status["size"]
            if isinstance(status.get("idle_connections"), int):
                total_idle += status["idle_connections"]
        
        self.metrics.total_connections = total_connections
        self.metrics.idle_connections = total_idle
        self.metrics.active_connections = total_connections - total_idle
        
        if total_connections > self.metrics.peak_connections:
            self.metrics.peak_connections = total_connections
        
        return {
            "status": "healthy" if total_connections > 0 else "unhealthy",
            "total_connectors": len(self.connectors),
            "metrics": {
                "total_connections": self.metrics.total_connections,
                "active_connections": self.metrics.active_connections,
                "idle_connections": self.metrics.idle_connections,
                "peak_connections": self.metrics.peak_connections,
                "total_requests": self.metrics.total_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": (
                    (self.metrics.total_requests - self.metrics.failed_requests) / 
                    max(self.metrics.total_requests, 1) * 100
                ),
                "average_response_time_ms": round(self.metrics.average_response_time, 2)
            },
            "connectors": connector_status
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all connectors."""
        results = {}
        overall_healthy = True
        
        for name, connector in self.connectors.items():
            health = await connector.health_check()
            results[name] = health
            
            if health["status"] != "healthy":
                overall_healthy = False
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "connectors": results,
            "pool_metrics": (await self.get_pool_status())["metrics"]
        }
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval)
                
                if not self._is_running:
                    break
                
                # Collect metrics
                status = await self.get_pool_status()
                
                # Log metrics
                metrics = status["metrics"]
                logger.info(
                    f"Pool metrics - Connections: {metrics['active_connections']}/{metrics['total_connections']}, "
                    f"Requests: {metrics['total_requests']}, "
                    f"Success rate: {metrics['success_rate']:.1f}%, "
                    f"Avg response: {metrics['average_response_time_ms']:.1f}ms"
                )
                
                # Check for issues
                if metrics["success_rate"] < 95:
                    logger.warning(f"Low success rate: {metrics['success_rate']:.1f}%")
                
                if metrics["average_response_time_ms"] > 1000:
                    logger.warning(f"High response time: {metrics['average_response_time_ms']:.1f}ms")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if not self._is_running:
                    break
                
                await self._optimize_pools()
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
    
    async def _optimize_pools(self) -> None:
        """Optimize connection pools based on usage patterns."""
        status = await self.get_pool_status()
        metrics = status["metrics"]
        
        # Reset metrics periodically
        if datetime.now() - self.metrics.last_reset > timedelta(hours=1):
            logger.info("Resetting pool metrics")
            self.metrics.reset()
        
        # Log optimization recommendations
        if metrics["active_connections"] > metrics["total_connections"] * 0.8:
            logger.info("High connection usage detected - consider increasing pool size")
        
        if metrics["idle_connections"] > metrics["total_connections"] * 0.5:
            logger.info("Many idle connections detected - consider decreasing pool size")
        
        if metrics["average_response_time_ms"] > 500:
            logger.info("High response times detected - consider optimizing queries or increasing resources")
    
    async def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics.reset()
        logger.info("Pool metrics reset")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "total_connections": self.metrics.total_connections,
            "active_connections": self.metrics.active_connections,
            "idle_connections": self.metrics.idle_connections,
            "peak_connections": self.metrics.peak_connections,
            "total_requests": self.metrics.total_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (
                (self.metrics.total_requests - self.metrics.failed_requests) / 
                max(self.metrics.total_requests, 1) * 100
            ),
            "average_response_time_ms": round(self.metrics.average_response_time, 2),
            "last_reset": self.metrics.last_reset.isoformat()
        }


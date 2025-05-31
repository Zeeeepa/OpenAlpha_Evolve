"""
OpenEvolve Database Connector System

This module provides comprehensive database connectivity, connection pooling,
multi-tenant support, and advanced database operations for the OpenEvolve
autonomous development pipeline.

Key Features:
- PostgreSQL integration with connection pooling
- Multi-tenant schema isolation
- Async query execution
- Migration system with versioning
- Health monitoring and metrics
- Security and audit logging
- Caching layer integration
"""

from .connectors.postgresql import PostgreSQLConnector
from .connectors.pool_manager import ConnectionPoolManager
from .connectors.multi_tenant import MultiTenantManager
from .migrations.manager import MigrationManager
from .monitoring.health import HealthMonitor
from .monitoring.metrics import MetricsCollector
from .security.access_control import AccessControlManager
from .security.audit import AuditLogger
from .cache.redis_cache import RedisCache
from .query_builder import QueryBuilder
from .exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseMigrationError,
    DatabaseSecurityError,
    DatabaseCacheError
)

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"

__all__ = [
    # Core connectors
    "PostgreSQLConnector",
    "ConnectionPoolManager", 
    "MultiTenantManager",
    
    # Migration system
    "MigrationManager",
    
    # Monitoring
    "HealthMonitor",
    "MetricsCollector",
    
    # Security
    "AccessControlManager",
    "AuditLogger",
    
    # Caching
    "RedisCache",
    
    # Query building
    "QueryBuilder",
    
    # Exceptions
    "DatabaseConnectionError",
    "DatabaseQueryError", 
    "DatabaseMigrationError",
    "DatabaseSecurityError",
    "DatabaseCacheError"
]


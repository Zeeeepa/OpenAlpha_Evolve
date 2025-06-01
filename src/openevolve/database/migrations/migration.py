"""
Database migration system for OpenEvolve.
"""

import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Migration status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationInfo:
    """Migration information."""
    version: str
    name: str
    description: str
    applied_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    checksum: Optional[str] = None
    execution_time: Optional[float] = None


class Migration(ABC):
    """Base migration class."""
    
    def __init__(self):
        self.version: str = ""
        self.name: str = ""
        self.description: str = ""
        self.dependencies: List[str] = []
    
    @abstractmethod
    async def up(self, connector) -> None:
        """Apply the migration."""
        pass
    
    @abstractmethod
    async def down(self, connector) -> None:
        """Rollback the migration."""
        pass
    
    def get_checksum(self) -> str:
        """Generate a checksum for the migration."""
        content = f"{self.version}{self.name}{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class SQLMigration(Migration):
    """SQL-based migration."""
    
    def __init__(self, up_sql: str, down_sql: str):
        super().__init__()
        self.up_sql = up_sql
        self.down_sql = down_sql
    
    async def up(self, connector) -> None:
        """Execute the up SQL."""
        if self.up_sql:
            await connector.execute_query(self.up_sql)
    
    async def down(self, connector) -> None:
        """Execute the down SQL."""
        if self.down_sql:
            await connector.execute_query(self.down_sql)
    
    def get_checksum(self) -> str:
        """Generate checksum including SQL content."""
        content = f"{self.version}{self.name}{self.up_sql}{self.down_sql}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class CreateTasksMigration(SQLMigration):
    """Create the tasks table for autonomous development tracking."""
    
    def __init__(self):
        super().__init__(
            up_sql="""
            CREATE TABLE IF NOT EXISTS tasks (
                id SERIAL PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                status VARCHAR(50) NOT NULL DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP WITH TIME ZONE,
                metadata JSONB DEFAULT '{}'::jsonb
            );
            
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
            CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
            """,
            down_sql="""
            DROP INDEX IF EXISTS idx_tasks_created_at;
            DROP INDEX IF EXISTS idx_tasks_priority;
            DROP INDEX IF EXISTS idx_tasks_status;
            DROP TABLE IF EXISTS tasks;
            """
        )
        
        self.version = "001"
        self.name = "create_tasks"
        self.description = "Create tasks table for autonomous development tracking"


class CreateAnalyticsEventsMigration(SQLMigration):
    """Create analytics events table for autonomous development insights."""
    
    def __init__(self):
        super().__init__(
            up_sql="""
            CREATE TABLE IF NOT EXISTS analytics_events (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(100) NOT NULL,
                event_data JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR(255),
                metadata JSONB DEFAULT '{}'::jsonb
            );
            
            CREATE INDEX IF NOT EXISTS idx_analytics_events_type ON analytics_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_analytics_events_created_at ON analytics_events(created_at);
            CREATE INDEX IF NOT EXISTS idx_analytics_events_session_id ON analytics_events(session_id);
            """,
            down_sql="""
            DROP INDEX IF EXISTS idx_analytics_events_session_id;
            DROP INDEX IF EXISTS idx_analytics_events_created_at;
            DROP INDEX IF EXISTS idx_analytics_events_type;
            DROP TABLE IF EXISTS analytics_events;
            """
        )
        
        self.version = "002"
        self.name = "create_analytics_events"
        self.description = "Create analytics events table for autonomous development insights"
        self.dependencies = ["001"]


class CreatePerformanceMetricsMigration(SQLMigration):
    """Create performance metrics table for system monitoring."""
    
    def __init__(self):
        super().__init__(
            up_sql="""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL,
                metric_type VARCHAR(50) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                tags JSONB DEFAULT '{}'::jsonb
            );
            
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name);
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_created_at ON performance_metrics(created_at);
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);
            """,
            down_sql="""
            DROP INDEX IF EXISTS idx_performance_metrics_type;
            DROP INDEX IF EXISTS idx_performance_metrics_created_at;
            DROP INDEX IF EXISTS idx_performance_metrics_name;
            DROP TABLE IF EXISTS performance_metrics;
            """
        )
        
        self.version = "003"
        self.name = "create_performance_metrics"
        self.description = "Create performance metrics table for system monitoring"
        self.dependencies = ["002"]


class CreateAuditLogsMigration(SQLMigration):
    """Create audit logs table for system event tracking."""
    
    def __init__(self):
        super().__init__(
            up_sql="""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id SERIAL PRIMARY KEY,
                action VARCHAR(50) NOT NULL,
                resource_type VARCHAR(100) NOT NULL,
                resource_id VARCHAR(255),
                details JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR(255),
                ip_address INET,
                user_agent TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
            CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_type ON audit_logs(resource_type);
            CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_logs_session_id ON audit_logs(session_id);
            """,
            down_sql="""
            DROP INDEX IF EXISTS idx_audit_logs_session_id;
            DROP INDEX IF EXISTS idx_audit_logs_created_at;
            DROP INDEX IF EXISTS idx_audit_logs_resource_type;
            DROP INDEX IF EXISTS idx_audit_logs_action;
            DROP TABLE IF EXISTS audit_logs;
            """
        )
        
        self.version = "004"
        self.name = "create_audit_logs"
        self.description = "Create audit logs table for system event tracking"
        self.dependencies = ["003"]


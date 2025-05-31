"""
Migration base class and utilities.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Migration execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationInfo:
    """Migration metadata."""
    
    version: str
    name: str
    description: str
    status: MigrationStatus
    applied_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    checksum: Optional[str] = None


class Migration(ABC):
    """
    Base class for database migrations.
    
    Each migration should inherit from this class and implement
    the up() and down() methods.
    """
    
    def __init__(self):
        self.version: str = ""
        self.name: str = ""
        self.description: str = ""
        self.dependencies: List[str] = []
        self.reversible: bool = True
    
    @abstractmethod
    async def up(self, connector) -> None:
        """
        Apply the migration.
        
        Args:
            connector: Database connector instance
        """
        pass
    
    async def down(self, connector) -> None:
        """
        Rollback the migration.
        
        Args:
            connector: Database connector instance
        """
        if not self.reversible:
            raise NotImplementedError(f"Migration {self.version} is not reversible")
        
        raise NotImplementedError(f"Rollback not implemented for migration {self.version}")
    
    async def validate(self, connector) -> bool:
        """
        Validate that the migration was applied correctly.
        
        Args:
            connector: Database connector instance
        
        Returns:
            True if validation passes
        """
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get migration information."""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "dependencies": self.dependencies,
            "reversible": self.reversible
        }


class SQLMigration(Migration):
    """
    Migration that executes raw SQL statements.
    """
    
    def __init__(self, up_sql: str, down_sql: Optional[str] = None):
        super().__init__()
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.reversible = down_sql is not None
    
    async def up(self, connector) -> None:
        """Execute the up SQL."""
        if not self.up_sql:
            return
        
        # Split SQL into individual statements
        statements = self._split_sql(self.up_sql)
        
        for statement in statements:
            if statement.strip():
                await connector.execute_query(statement, fetch_mode="none")
                logger.debug(f"Executed SQL: {statement[:100]}...")
    
    async def down(self, connector) -> None:
        """Execute the down SQL."""
        if not self.reversible:
            raise NotImplementedError(f"Migration {self.version} is not reversible")
        
        if not self.down_sql:
            return
        
        # Split SQL into individual statements
        statements = self._split_sql(self.down_sql)
        
        for statement in statements:
            if statement.strip():
                await connector.execute_query(statement, fetch_mode="none")
                logger.debug(f"Executed rollback SQL: {statement[:100]}...")
    
    def _split_sql(self, sql: str) -> List[str]:
        """Split SQL into individual statements."""
        # Simple split on semicolon - could be enhanced for more complex cases
        statements = []
        current_statement = ""
        in_string = False
        escape_next = False
        
        for char in sql:
            if escape_next:
                current_statement += char
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                current_statement += char
                continue
            
            if char in ("'", '"') and not escape_next:
                in_string = not in_string
            
            if char == ';' and not in_string:
                if current_statement.strip():
                    statements.append(current_statement.strip())
                current_statement = ""
            else:
                current_statement += char
        
        # Add the last statement if it doesn't end with semicolon
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements


class PythonMigration(Migration):
    """
    Migration that executes Python code.
    """
    
    def __init__(self, up_func, down_func=None):
        super().__init__()
        self.up_func = up_func
        self.down_func = down_func
        self.reversible = down_func is not None
    
    async def up(self, connector) -> None:
        """Execute the up function."""
        if self.up_func:
            if asyncio.iscoroutinefunction(self.up_func):
                await self.up_func(connector)
            else:
                self.up_func(connector)
    
    async def down(self, connector) -> None:
        """Execute the down function."""
        if not self.reversible:
            raise NotImplementedError(f"Migration {self.version} is not reversible")
        
        if self.down_func:
            if asyncio.iscoroutinefunction(self.down_func):
                await self.down_func(connector)
            else:
                self.down_func(connector)


# Example migrations for the OpenEvolve system

class CreateTenantRegistryMigration(SQLMigration):
    """Create the tenant registry table."""
    
    def __init__(self):
        up_sql = """
        CREATE TABLE IF NOT EXISTS tenant_registry (
            tenant_id VARCHAR(255) PRIMARY KEY,
            schema_name VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        CREATE INDEX IF NOT EXISTS idx_tenant_registry_schema 
        ON tenant_registry(schema_name);
        
        CREATE INDEX IF NOT EXISTS idx_tenant_registry_active 
        ON tenant_registry(is_active) WHERE is_active = TRUE;
        """
        
        down_sql = """
        DROP INDEX IF EXISTS idx_tenant_registry_active;
        DROP INDEX IF EXISTS idx_tenant_registry_schema;
        DROP TABLE IF EXISTS tenant_registry;
        """
        
        super().__init__(up_sql, down_sql)
        self.version = "001"
        self.name = "create_tenant_registry"
        self.description = "Create tenant registry table for multi-tenant support"


class CreateTaskManagementMigration(SQLMigration):
    """Create task management tables."""
    
    def __init__(self):
        up_sql = """
        CREATE TABLE IF NOT EXISTS tasks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id VARCHAR(255) REFERENCES tenant_registry(tenant_id),
            title VARCHAR(500) NOT NULL,
            description TEXT,
            status VARCHAR(50) DEFAULT 'pending',
            priority INTEGER DEFAULT 0,
            assigned_to VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            completed_at TIMESTAMP WITH TIME ZONE,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        CREATE INDEX IF NOT EXISTS idx_tasks_tenant_id ON tasks(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_assigned_to ON tasks(assigned_to);
        CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
        
        CREATE TABLE IF NOT EXISTS task_executions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
            execution_type VARCHAR(100) NOT NULL,
            status VARCHAR(50) DEFAULT 'pending',
            started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            completed_at TIMESTAMP WITH TIME ZONE,
            error_message TEXT,
            result_data JSONB,
            metrics JSONB DEFAULT '{}'::jsonb
        );
        
        CREATE INDEX IF NOT EXISTS idx_task_executions_task_id ON task_executions(task_id);
        CREATE INDEX IF NOT EXISTS idx_task_executions_status ON task_executions(status);
        """
        
        down_sql = """
        DROP INDEX IF EXISTS idx_task_executions_status;
        DROP INDEX IF EXISTS idx_task_executions_task_id;
        DROP TABLE IF EXISTS task_executions;
        
        DROP INDEX IF EXISTS idx_tasks_created_at;
        DROP INDEX IF EXISTS idx_tasks_assigned_to;
        DROP INDEX IF EXISTS idx_tasks_status;
        DROP INDEX IF EXISTS idx_tasks_tenant_id;
        DROP TABLE IF EXISTS tasks;
        """
        
        super().__init__(up_sql, down_sql)
        self.version = "002"
        self.name = "create_task_management"
        self.description = "Create task management tables"
        self.dependencies = ["001"]


class CreateAnalyticsMigration(SQLMigration):
    """Create analytics and metrics tables."""
    
    def __init__(self):
        up_sql = """
        CREATE TABLE IF NOT EXISTS analytics_events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id VARCHAR(255) REFERENCES tenant_registry(tenant_id),
            event_type VARCHAR(100) NOT NULL,
            event_name VARCHAR(200) NOT NULL,
            event_data JSONB NOT NULL DEFAULT '{}'::jsonb,
            user_id VARCHAR(255),
            session_id VARCHAR(255),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            ip_address INET,
            user_agent TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_analytics_events_tenant_id ON analytics_events(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_events_type ON analytics_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_analytics_events_timestamp ON analytics_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_analytics_events_user_id ON analytics_events(user_id);
        
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id VARCHAR(255) REFERENCES tenant_registry(tenant_id),
            metric_name VARCHAR(200) NOT NULL,
            metric_value NUMERIC NOT NULL,
            metric_unit VARCHAR(50),
            tags JSONB DEFAULT '{}'::jsonb,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_tenant_id ON performance_metrics(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name);
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);
        """
        
        down_sql = """
        DROP INDEX IF EXISTS idx_performance_metrics_timestamp;
        DROP INDEX IF EXISTS idx_performance_metrics_name;
        DROP INDEX IF EXISTS idx_performance_metrics_tenant_id;
        DROP TABLE IF EXISTS performance_metrics;
        
        DROP INDEX IF EXISTS idx_analytics_events_user_id;
        DROP INDEX IF EXISTS idx_analytics_events_timestamp;
        DROP INDEX IF EXISTS idx_analytics_events_type;
        DROP INDEX IF EXISTS idx_analytics_events_tenant_id;
        DROP TABLE IF EXISTS analytics_events;
        """
        
        super().__init__(up_sql, down_sql)
        self.version = "003"
        self.name = "create_analytics"
        self.description = "Create analytics and performance metrics tables"
        self.dependencies = ["001"]


class CreateAuditLogMigration(SQLMigration):
    """Create audit log table."""
    
    def __init__(self):
        up_sql = """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id VARCHAR(255) REFERENCES tenant_registry(tenant_id),
            user_id VARCHAR(255),
            action VARCHAR(200) NOT NULL,
            resource_type VARCHAR(100),
            resource_id VARCHAR(255),
            old_values JSONB,
            new_values JSONB,
            ip_address INET,
            user_agent TEXT,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_id ON audit_logs(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
        """
        
        down_sql = """
        DROP INDEX IF EXISTS idx_audit_logs_resource;
        DROP INDEX IF EXISTS idx_audit_logs_timestamp;
        DROP INDEX IF EXISTS idx_audit_logs_action;
        DROP INDEX IF EXISTS idx_audit_logs_user_id;
        DROP INDEX IF EXISTS idx_audit_logs_tenant_id;
        DROP TABLE IF EXISTS audit_logs;
        """
        
        super().__init__(up_sql, down_sql)
        self.version = "004"
        self.name = "create_audit_log"
        self.description = "Create audit log table for security tracking"
        self.dependencies = ["001"]


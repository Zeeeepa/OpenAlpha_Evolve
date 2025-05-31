"""
Database audit logging system.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from ..connectors.postgresql import PostgreSQLConnector
from ..exceptions import DatabaseQueryError

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """Audit action types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"
    SCHEMA_CREATE = "schema_create"
    SCHEMA_DROP = "schema_drop"
    MIGRATION_APPLY = "migration_apply"
    MIGRATION_ROLLBACK = "migration_rollback"


@dataclass
class AuditEvent:
    """Audit event record."""
    
    user_id: str
    action: AuditAction
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['action'] = self.action.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AuditLogger:
    """
    Comprehensive audit logging system for database operations
    and security events.
    """
    
    def __init__(self, connector: Optional[PostgreSQLConnector] = None):
        self.connector = connector
        self._local_buffer: List[AuditEvent] = []
        self._buffer_size = 100
        self._auto_flush = True
        
        logger.info("Audit logger initialized")
    
    async def initialize(self) -> None:
        """Initialize audit logging system."""
        if self.connector:
            await self._ensure_audit_table()
    
    async def _ensure_audit_table(self) -> None:
        """Ensure audit log table exists."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id VARCHAR(255),
            user_id VARCHAR(255) NOT NULL,
            action VARCHAR(200) NOT NULL,
            resource_type VARCHAR(100),
            resource_id VARCHAR(255),
            old_values JSONB,
            new_values JSONB,
            ip_address INET,
            user_agent TEXT,
            session_id VARCHAR(255),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_id ON audit_logs(tenant_id);
        """
        
        try:
            await self.connector.execute_query(create_table_sql, fetch_mode="none")
            logger.info("Audit log table created/verified")
        except Exception as e:
            logger.error(f"Failed to create audit log table: {e}")
            raise
    
    def log_event(
        self,
        user_id: str,
        action: AuditAction,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an audit event.
        
        Args:
            user_id: User performing the action
            action: Type of action performed
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            old_values: Previous values (for updates)
            new_values: New values (for creates/updates)
            ip_address: Client IP address
            user_agent: Client user agent
            tenant_id: Tenant context
            session_id: Session identifier
            metadata: Additional metadata
        """
        event = AuditEvent(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values,
            ip_address=ip_address,
            user_agent=user_agent,
            tenant_id=tenant_id,
            session_id=session_id,
            metadata=metadata
        )
        
        self._local_buffer.append(event)
        
        # Auto-flush if buffer is full
        if self._auto_flush and len(self._local_buffer) >= self._buffer_size:
            if self.connector:
                import asyncio
                asyncio.create_task(self.flush_buffer())
        
        logger.debug(f"Logged audit event: {user_id} {action.value} {resource_type}/{resource_id}")
    
    def log_create(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        new_values: Dict[str, Any],
        **kwargs
    ) -> None:
        """Log a CREATE operation."""
        self.log_event(
            user_id=user_id,
            action=AuditAction.CREATE,
            resource_type=resource_type,
            resource_id=resource_id,
            new_values=new_values,
            **kwargs
        )
    
    def log_read(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        **kwargs
    ) -> None:
        """Log a READ operation."""
        self.log_event(
            user_id=user_id,
            action=AuditAction.READ,
            resource_type=resource_type,
            resource_id=resource_id,
            **kwargs
        )
    
    def log_update(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        old_values: Dict[str, Any],
        new_values: Dict[str, Any],
        **kwargs
    ) -> None:
        """Log an UPDATE operation."""
        self.log_event(
            user_id=user_id,
            action=AuditAction.UPDATE,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values,
            **kwargs
        )
    
    def log_delete(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        old_values: Dict[str, Any],
        **kwargs
    ) -> None:
        """Log a DELETE operation."""
        self.log_event(
            user_id=user_id,
            action=AuditAction.DELETE,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            **kwargs
        )
    
    def log_login(self, user_id: str, **kwargs) -> None:
        """Log a user login."""
        self.log_event(
            user_id=user_id,
            action=AuditAction.LOGIN,
            **kwargs
        )
    
    def log_logout(self, user_id: str, **kwargs) -> None:
        """Log a user logout."""
        self.log_event(
            user_id=user_id,
            action=AuditAction.LOGOUT,
            **kwargs
        )
    
    def log_permission_change(
        self,
        user_id: str,
        action: AuditAction,
        target_user: str,
        permission_details: Dict[str, Any],
        **kwargs
    ) -> None:
        """Log permission grant/revoke."""
        self.log_event(
            user_id=user_id,
            action=action,
            resource_type="permission",
            resource_id=target_user,
            new_values=permission_details,
            **kwargs
        )
    
    def log_schema_operation(
        self,
        user_id: str,
        action: AuditAction,
        schema_name: str,
        **kwargs
    ) -> None:
        """Log schema create/drop operations."""
        self.log_event(
            user_id=user_id,
            action=action,
            resource_type="schema",
            resource_id=schema_name,
            **kwargs
        )
    
    def log_migration(
        self,
        user_id: str,
        action: AuditAction,
        migration_version: str,
        migration_details: Dict[str, Any],
        **kwargs
    ) -> None:
        """Log migration operations."""
        self.log_event(
            user_id=user_id,
            action=action,
            resource_type="migration",
            resource_id=migration_version,
            new_values=migration_details,
            **kwargs
        )
    
    async def flush_buffer(self) -> None:
        """Flush buffered audit events to database."""
        if not self.connector or not self._local_buffer:
            return
        
        events_to_flush = self._local_buffer.copy()
        self._local_buffer.clear()
        
        try:
            # Prepare batch insert
            insert_sql = """
            INSERT INTO audit_logs (
                tenant_id, user_id, action, resource_type, resource_id,
                old_values, new_values, ip_address, user_agent, session_id,
                timestamp, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """
            
            # Execute batch insert
            queries = []
            for event in events_to_flush:
                parameters = {
                    "tenant_id": event.tenant_id,
                    "user_id": event.user_id,
                    "action": event.action.value,
                    "resource_type": event.resource_type,
                    "resource_id": event.resource_id,
                    "old_values": json.dumps(event.old_values) if event.old_values else None,
                    "new_values": json.dumps(event.new_values) if event.new_values else None,
                    "ip_address": event.ip_address,
                    "user_agent": event.user_agent,
                    "session_id": event.session_id,
                    "timestamp": event.timestamp,
                    "metadata": json.dumps(event.metadata) if event.metadata else None
                }
                queries.append((insert_sql, parameters, "none"))
            
            await self.connector.execute_transaction(queries)
            
            logger.info(f"Flushed {len(events_to_flush)} audit events to database")
            
        except Exception as e:
            logger.error(f"Failed to flush audit events: {e}")
            # Put events back in buffer for retry
            self._local_buffer.extend(events_to_flush)
            raise DatabaseQueryError(f"Failed to flush audit events: {e}")
    
    async def search_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search audit logs with filters.
        
        Args:
            user_id: Filter by user ID
            action: Filter by action type
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            tenant_id: Filter by tenant ID
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of results
            offset: Offset for pagination
        
        Returns:
            List of audit log entries
        """
        if not self.connector:
            return []
        
        # Build query
        conditions = []
        parameters = {}
        param_counter = 0
        
        if user_id:
            param_counter += 1
            conditions.append(f"user_id = $param_{param_counter}")
            parameters[f"param_{param_counter}"] = user_id
        
        if action:
            param_counter += 1
            conditions.append(f"action = $param_{param_counter}")
            parameters[f"param_{param_counter}"] = action.value
        
        if resource_type:
            param_counter += 1
            conditions.append(f"resource_type = $param_{param_counter}")
            parameters[f"param_{param_counter}"] = resource_type
        
        if resource_id:
            param_counter += 1
            conditions.append(f"resource_id = $param_{param_counter}")
            parameters[f"param_{param_counter}"] = resource_id
        
        if tenant_id:
            param_counter += 1
            conditions.append(f"tenant_id = $param_{param_counter}")
            parameters[f"param_{param_counter}"] = tenant_id
        
        if start_time:
            param_counter += 1
            conditions.append(f"timestamp >= $param_{param_counter}")
            parameters[f"param_{param_counter}"] = start_time
        
        if end_time:
            param_counter += 1
            conditions.append(f"timestamp <= $param_{param_counter}")
            parameters[f"param_{param_counter}"] = end_time
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        SELECT * FROM audit_logs
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT {limit} OFFSET {offset}
        """
        
        try:
            results = await self.connector.execute_query(query, parameters, fetch_mode="all")
            return results
        except Exception as e:
            logger.error(f"Failed to search audit logs: {e}")
            raise DatabaseQueryError(f"Failed to search audit logs: {e}")
    
    async def get_audit_summary(
        self,
        tenant_id: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get audit summary statistics.
        
        Args:
            tenant_id: Filter by tenant ID
            hours: Number of hours to include in summary
        
        Returns:
            Summary statistics
        """
        if not self.connector:
            return {}
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        # Build base query conditions
        conditions = ["timestamp >= $1"]
        parameters = {"start_time": start_time}
        
        if tenant_id:
            conditions.append("tenant_id = $2")
            parameters["tenant_id"] = tenant_id
        
        where_clause = " WHERE " + " AND ".join(conditions)
        
        # Get action counts
        action_query = f"""
        SELECT action, COUNT(*) as count
        FROM audit_logs
        {where_clause}
        GROUP BY action
        ORDER BY count DESC
        """
        
        # Get user activity
        user_query = f"""
        SELECT user_id, COUNT(*) as count
        FROM audit_logs
        {where_clause}
        GROUP BY user_id
        ORDER BY count DESC
        LIMIT 10
        """
        
        # Get resource activity
        resource_query = f"""
        SELECT resource_type, COUNT(*) as count
        FROM audit_logs
        {where_clause}
        AND resource_type IS NOT NULL
        GROUP BY resource_type
        ORDER BY count DESC
        """
        
        try:
            action_results = await self.connector.execute_query(action_query, parameters, fetch_mode="all")
            user_results = await self.connector.execute_query(user_query, parameters, fetch_mode="all")
            resource_results = await self.connector.execute_query(resource_query, parameters, fetch_mode="all")
            
            return {
                "time_range_hours": hours,
                "actions": {row["action"]: row["count"] for row in action_results},
                "top_users": [{"user_id": row["user_id"], "count": row["count"]} for row in user_results],
                "resource_types": {row["resource_type"]: row["count"] for row in resource_results},
                "total_events": sum(row["count"] for row in action_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit summary: {e}")
            raise DatabaseQueryError(f"Failed to get audit summary: {e}")
    
    def set_auto_flush(self, enabled: bool, buffer_size: int = 100) -> None:
        """
        Configure automatic buffer flushing.
        
        Args:
            enabled: Whether to enable auto-flush
            buffer_size: Buffer size threshold for auto-flush
        """
        self._auto_flush = enabled
        self._buffer_size = buffer_size
        logger.info(f"Auto-flush {'enabled' if enabled else 'disabled'}, buffer size: {buffer_size}")
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        return {
            "buffer_size": len(self._local_buffer),
            "max_buffer_size": self._buffer_size,
            "auto_flush_enabled": self._auto_flush,
            "connector_available": self.connector is not None
        }


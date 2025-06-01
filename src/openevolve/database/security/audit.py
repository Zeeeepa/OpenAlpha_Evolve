"""
Simplified audit logging for autonomous development operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..exceptions import DatabaseSecurityError

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """Audit action types for autonomous operations."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Audit event for autonomous operations."""
    action: AuditAction
    resource_type: str
    resource_id: Optional[str] = None
    details: Dict[str, Any] = None
    created_at: datetime = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.details is None:
            self.details = {}


class AuditLogger:
    """
    Simplified audit logger for autonomous development operations.
    Tracks system events, task execution, and resource operations.
    """
    
    def __init__(self, connector, buffer_size: int = 1000, flush_interval: int = 30):
        self.connector = connector
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._event_buffer: List[AuditEvent] = []
        self._flush_task: Optional[asyncio.Task] = None
        self._is_initialized = False
        
        logger.info(f"Audit logger initialized with buffer size {buffer_size}")
    
    async def initialize(self) -> None:
        """Initialize the audit logging system."""
        try:
            # Create audit logs table
            create_table_sql = """
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
            """
            
            await self.connector.execute_query(create_table_sql)
            
            # Start background flush task
            self._flush_task = asyncio.create_task(self._background_flush())
            self._is_initialized = True
            
            logger.info("Audit logging system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit logging: {e}")
            raise DatabaseSecurityError(f"Failed to initialize audit logging: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the audit logging system."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush any remaining events
        await self._flush_events()
        
        logger.info("Audit logging system shutdown")
    
    def log_event(
        self,
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log an audit event.
        
        Args:
            action: Action being performed
            resource_type: Type of resource
            resource_id: ID of the resource
            details: Additional event details
            session_id: Session identifier
            **kwargs: Additional event data
        """
        if not self._is_initialized:
            logger.warning("Audit logger not initialized, skipping event")
            return
        
        # Merge kwargs into details
        event_details = details or {}
        event_details.update(kwargs)
        
        event = AuditEvent(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=event_details,
            session_id=session_id
        )
        
        self._event_buffer.append(event)
        
        # Flush if buffer is full
        if len(self._event_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_events())
        
        logger.debug(f"Logged audit event: {action.value} {resource_type}/{resource_id}")
    
    def log_create(self, resource_type: str, resource_id: str, **kwargs) -> None:
        """Log a create operation."""
        self.log_event(AuditAction.CREATE, resource_type, resource_id, **kwargs)
    
    def log_read(self, resource_type: str, resource_id: str, **kwargs) -> None:
        """Log a read operation."""
        self.log_event(AuditAction.READ, resource_type, resource_id, **kwargs)
    
    def log_update(self, resource_type: str, resource_id: str, old_values: Dict = None, new_values: Dict = None, **kwargs) -> None:
        """Log an update operation."""
        details = kwargs.get('details', {})
        if old_values:
            details['old_values'] = old_values
        if new_values:
            details['new_values'] = new_values
        kwargs['details'] = details
        self.log_event(AuditAction.UPDATE, resource_type, resource_id, **kwargs)
    
    def log_delete(self, resource_type: str, resource_id: str, old_values: Dict = None, **kwargs) -> None:
        """Log a delete operation."""
        details = kwargs.get('details', {})
        if old_values:
            details['deleted_values'] = old_values
        kwargs['details'] = details
        self.log_event(AuditAction.DELETE, resource_type, resource_id, **kwargs)
    
    def log_execute(self, resource_type: str, resource_id: str, **kwargs) -> None:
        """Log an execute operation."""
        self.log_event(AuditAction.EXECUTE, resource_type, resource_id, **kwargs)
    
    def log_system_start(self, **kwargs) -> None:
        """Log system startup."""
        self.log_event(AuditAction.SYSTEM_START, "system", "main", **kwargs)
    
    def log_system_stop(self, **kwargs) -> None:
        """Log system shutdown."""
        self.log_event(AuditAction.SYSTEM_STOP, "system", "main", **kwargs)
    
    def log_task_start(self, task_id: str, **kwargs) -> None:
        """Log task start."""
        self.log_event(AuditAction.TASK_START, "task", task_id, **kwargs)
    
    def log_task_complete(self, task_id: str, **kwargs) -> None:
        """Log task completion."""
        self.log_event(AuditAction.TASK_COMPLETE, "task", task_id, **kwargs)
    
    def log_error(self, error_type: str, error_message: str, **kwargs) -> None:
        """Log an error event."""
        details = kwargs.get('details', {})
        details.update({
            'error_type': error_type,
            'error_message': error_message
        })
        kwargs['details'] = details
        self.log_event(AuditAction.ERROR, "system", "error", **kwargs)
    
    async def _background_flush(self) -> None:
        """Background task to flush events periodically."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background flush: {e}")
    
    async def _flush_events(self) -> None:
        """Flush buffered events to database."""
        if not self._event_buffer:
            return
        
        events_to_flush = self._event_buffer.copy()
        self._event_buffer.clear()
        
        try:
            # Prepare batch insert
            insert_sql = """
            INSERT INTO audit_logs (action, resource_type, resource_id, details, created_at, session_id, ip_address, user_agent)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            # Execute batch insert
            for event in events_to_flush:
                parameters = {
                    "action": event.action.value,
                    "resource_type": event.resource_type,
                    "resource_id": event.resource_id,
                    "details": event.details,
                    "created_at": event.created_at,
                    "session_id": event.session_id,
                    "ip_address": event.ip_address,
                    "user_agent": event.user_agent
                }
                
                await self.connector.execute_query(insert_sql, parameters)
            
            logger.debug(f"Flushed {len(events_to_flush)} audit events to database")
            
        except Exception as e:
            logger.error(f"Failed to flush audit events: {e}")
            # Re-add events to buffer for retry
            self._event_buffer.extend(events_to_flush)
    
    async def search_audit_logs(
        self,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search audit logs with filters.
        
        Args:
            action: Filter by action type
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            session_id: Filter by session ID
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of results
        
        Returns:
            List of audit log entries
        """
        conditions = []
        parameters = {}
        param_counter = 1
        
        if action:
            conditions.append(f"action = $param_{param_counter}")
            parameters[f"param_{param_counter}"] = action.value
            param_counter += 1
        
        if resource_type:
            conditions.append(f"resource_type = $param_{param_counter}")
            parameters[f"param_{param_counter}"] = resource_type
            param_counter += 1
        
        if resource_id:
            conditions.append(f"resource_id = $param_{param_counter}")
            parameters[f"param_{param_counter}"] = resource_id
            param_counter += 1
        
        if session_id:
            conditions.append(f"session_id = $param_{param_counter}")
            parameters[f"param_{param_counter}"] = session_id
            param_counter += 1
        
        if start_time:
            conditions.append(f"created_at >= $param_{param_counter}")
            parameters[f"param_{param_counter}"] = start_time
            param_counter += 1
        
        if end_time:
            conditions.append(f"created_at <= $param_{param_counter}")
            parameters[f"param_{param_counter}"] = end_time
            param_counter += 1
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT id, action, resource_type, resource_id, details, created_at, session_id, ip_address, user_agent
        FROM audit_logs
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        
        try:
            results = await self.connector.execute_query(query, parameters, fetch_mode="all")
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to search audit logs: {e}")
            raise DatabaseSecurityError(f"Failed to search audit logs: {e}")
    
    async def get_audit_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get audit statistics for the specified number of days.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with audit statistics
        """
        start_time = datetime.utcnow() - timedelta(days=days)
        
        try:
            # Get action counts
            action_query = """
            SELECT action, COUNT(*) as count
            FROM audit_logs
            WHERE created_at >= $1
            GROUP BY action
            ORDER BY count DESC
            """
            action_results = await self.connector.execute_query(
                action_query, 
                {"start_time": start_time}, 
                fetch_mode="all"
            )
            
            # Get resource type counts
            resource_query = """
            SELECT resource_type, COUNT(*) as count
            FROM audit_logs
            WHERE created_at >= $1
            GROUP BY resource_type
            ORDER BY count DESC
            """
            resource_results = await self.connector.execute_query(
                resource_query, 
                {"start_time": start_time}, 
                fetch_mode="all"
            )
            
            # Get total count
            total_query = """
            SELECT COUNT(*) as total
            FROM audit_logs
            WHERE created_at >= $1
            """
            total_result = await self.connector.execute_query(
                total_query, 
                {"start_time": start_time}, 
                fetch_mode="one"
            )
            
            return {
                "period_days": days,
                "total_events": total_result["total"] if total_result else 0,
                "actions": [{"action": row["action"], "count": row["count"]} for row in action_results],
                "resource_types": [{"resource_type": row["resource_type"], "count": row["count"]} for row in resource_results],
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit stats: {e}")
            raise DatabaseSecurityError(f"Failed to get audit stats: {e}")


"""
Assignment Router

Routes assignment events to appropriate handlers and manages assignment workflows.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from ..interfaces import AssignmentEvent, AssignmentAction

logger = logging.getLogger(__name__)


class AssignmentRouter:
    """Routes assignment events to appropriate handlers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Handler registry
        self.assignment_handlers: List[Callable] = []
        self.unassignment_handlers: List[Callable] = []
        self.reassignment_handlers: List[Callable] = []
        
        # Routing configuration
        self.enable_parallel_processing = self.config.get("enable_parallel_processing", False)
        self.max_concurrent_assignments = self.config.get("max_concurrent_assignments", 5)
        self.assignment_timeout = self.config.get("assignment_timeout", 300)  # 5 minutes
        
        # Active assignments tracking
        self.active_assignments: Dict[str, Dict[str, Any]] = {}
        
        logger.info("AssignmentRouter initialized")
    
    def register_assignment_handler(self, handler: Callable) -> None:
        """Register handler for assignment events"""
        self.assignment_handlers.append(handler)
        logger.info(f"Registered assignment handler: {handler.__name__}")
    
    def register_unassignment_handler(self, handler: Callable) -> None:
        """Register handler for unassignment events"""
        self.unassignment_handlers.append(handler)
        logger.info(f"Registered unassignment handler: {handler.__name__}")
    
    def register_reassignment_handler(self, handler: Callable) -> None:
        """Register handler for reassignment events"""
        self.reassignment_handlers.append(handler)
        logger.info(f"Registered reassignment handler: {handler.__name__}")
    
    async def route_assignment(self, assignment_event: AssignmentEvent) -> bool:
        """Route assignment event to appropriate handlers"""
        try:
            issue_id = assignment_event.issue_id
            action = assignment_event.action
            
            logger.info(f"Routing assignment event: {action.value} for issue {issue_id}")
            
            # Check if assignment is already being processed
            if issue_id in self.active_assignments:
                logger.warning(f"Assignment for issue {issue_id} already being processed")
                return False
            
            # Track active assignment
            self.active_assignments[issue_id] = {
                "assignment_event": assignment_event,
                "started_at": datetime.now(),
                "status": "processing"
            }
            
            try:
                # Route based on action type
                if action == AssignmentAction.ASSIGNED:
                    success = await self._handle_assignment(assignment_event)
                elif action == AssignmentAction.UNASSIGNED:
                    success = await self._handle_unassignment(assignment_event)
                elif action == AssignmentAction.REASSIGNED:
                    success = await self._handle_reassignment(assignment_event)
                else:
                    logger.error(f"Unknown assignment action: {action}")
                    success = False
                
                # Update status
                self.active_assignments[issue_id]["status"] = "completed" if success else "failed"
                self.active_assignments[issue_id]["completed_at"] = datetime.now()
                
                return success
                
            except Exception as e:
                logger.error(f"Error processing assignment for issue {issue_id}: {e}")
                self.active_assignments[issue_id]["status"] = "error"
                self.active_assignments[issue_id]["error"] = str(e)
                return False
            
        except Exception as e:
            logger.error(f"Error routing assignment event: {e}")
            return False
    
    async def _handle_assignment(self, assignment_event: AssignmentEvent) -> bool:
        """Handle assignment events"""
        success_count = 0
        
        for handler in self.assignment_handlers:
            try:
                result = await handler(assignment_event)
                if result:
                    success_count += 1
                    logger.debug(f"Assignment handler {handler.__name__} succeeded")
                else:
                    logger.warning(f"Assignment handler {handler.__name__} failed")
            except Exception as e:
                logger.error(f"Assignment handler {handler.__name__} error: {e}")
        
        # Consider successful if at least one handler succeeded
        return success_count > 0
    
    async def _handle_unassignment(self, assignment_event: AssignmentEvent) -> bool:
        """Handle unassignment events"""
        success_count = 0
        
        for handler in self.unassignment_handlers:
            try:
                result = await handler(assignment_event)
                if result:
                    success_count += 1
                    logger.debug(f"Unassignment handler {handler.__name__} succeeded")
                else:
                    logger.warning(f"Unassignment handler {handler.__name__} failed")
            except Exception as e:
                logger.error(f"Unassignment handler {handler.__name__} error: {e}")
        
        return success_count > 0
    
    async def _handle_reassignment(self, assignment_event: AssignmentEvent) -> bool:
        """Handle reassignment events"""
        success_count = 0
        
        for handler in self.reassignment_handlers:
            try:
                result = await handler(assignment_event)
                if result:
                    success_count += 1
                    logger.debug(f"Reassignment handler {handler.__name__} succeeded")
                else:
                    logger.warning(f"Reassignment handler {handler.__name__} failed")
            except Exception as e:
                logger.error(f"Reassignment handler {handler.__name__} error: {e}")
        
        return success_count > 0
    
    async def cancel_assignment(self, issue_id: str) -> bool:
        """Cancel active assignment processing"""
        try:
            if issue_id not in self.active_assignments:
                logger.warning(f"No active assignment found for issue {issue_id}")
                return False
            
            assignment_info = self.active_assignments[issue_id]
            assignment_info["status"] = "cancelled"
            assignment_info["cancelled_at"] = datetime.now()
            
            logger.info(f"Cancelled assignment processing for issue {issue_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling assignment for issue {issue_id}: {e}")
            return False
    
    def get_active_assignments(self) -> Dict[str, Dict[str, Any]]:
        """Get all active assignments"""
        return self.active_assignments.copy()
    
    def get_assignment_status(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific assignment"""
        return self.active_assignments.get(issue_id)
    
    async def cleanup_completed_assignments(self, max_age_hours: int = 24) -> int:
        """Clean up old completed assignments"""
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        completed_assignments = []
        
        for issue_id, assignment_info in self.active_assignments.items():
            status = assignment_info["status"]
            completed_at = assignment_info.get("completed_at")
            
            if status in ["completed", "failed", "error", "cancelled"] and completed_at:
                if completed_at < cutoff_time:
                    completed_assignments.append(issue_id)
        
        for issue_id in completed_assignments:
            del self.active_assignments[issue_id]
        
        logger.info(f"Cleaned up {len(completed_assignments)} completed assignments")
        return len(completed_assignments)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get assignment routing statistics"""
        total_assignments = len(self.active_assignments)
        status_counts = {}
        
        for assignment_info in self.active_assignments.values():
            status = assignment_info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_active_assignments": total_assignments,
            "status_breakdown": status_counts,
            "registered_handlers": {
                "assignment": len(self.assignment_handlers),
                "unassignment": len(self.unassignment_handlers),
                "reassignment": len(self.reassignment_handlers)
            },
            "configuration": {
                "parallel_processing": self.enable_parallel_processing,
                "max_concurrent": self.max_concurrent_assignments,
                "timeout": self.assignment_timeout
            }
        }


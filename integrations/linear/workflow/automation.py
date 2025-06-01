"""
Workflow Automation

Handles automated workflows for Linear integration including task creation,
status updates, and progress synchronization.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..interfaces import (
    WorkflowAutomationInterface, 
    AssignmentEvent, 
    LinearIssue,
    LinearGraphQLClientInterface
)
from core.interfaces import TaskDefinition, TaskManagerAgent
from .task_creator import TaskCreator
from .progress_sync import ProgressSync

logger = logging.getLogger(__name__)


class WorkflowAutomation(WorkflowAutomationInterface):
    """Main workflow automation engine for Linear integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.linear_client: Optional[LinearGraphQLClientInterface] = None
        self.task_creator = TaskCreator(config)
        self.progress_sync = ProgressSync(config)
        
        # Workflow configuration
        self.auto_start_tasks = self.config.get("auto_start_tasks", True)
        self.auto_update_status = self.config.get("auto_update_status", True)
        self.status_mapping = self.config.get("status_mapping", {
            "assigned": "In Progress",
            "started": "In Progress", 
            "completed": "Done",
            "failed": "Backlog"
        })
        
        # Active tasks tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        logger.info("WorkflowAutomation initialized")
    
    def set_linear_client(self, client: LinearGraphQLClientInterface) -> None:
        """Set Linear GraphQL client"""
        self.linear_client = client
        self.task_creator.set_linear_client(client)
        self.progress_sync.set_linear_client(client)
    
    async def handle_assignment(self, assignment: AssignmentEvent) -> bool:
        """Handle new assignment"""
        try:
            issue_id = assignment.issue_id
            
            logger.info(f"Handling assignment: {assignment.action.value} for issue {issue_id}")
            
            if assignment.action.value == "assigned":
                return await self._handle_new_assignment(assignment)
            elif assignment.action.value == "unassigned":
                return await self._handle_unassignment(assignment)
            elif assignment.action.value == "reassigned":
                return await self._handle_reassignment(assignment)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling assignment: {e}")
            return False
    
    async def _handle_new_assignment(self, assignment: AssignmentEvent) -> bool:
        """Handle new assignment to bot"""
        try:
            issue_id = assignment.issue_id
            
            # Get issue details
            if not self.linear_client:
                logger.error("Linear client not set")
                return False
            
            issue = await self.linear_client.get_issue(issue_id)
            if not issue:
                logger.error(f"Could not fetch issue {issue_id}")
                return False
            
            # Update issue status to indicate work has started
            if self.auto_update_status:
                await self.update_issue_status(issue_id, "started")
            
            # Create OpenAlpha_Evolve task from Linear issue
            task_id = await self.create_task_from_issue(issue)
            if not task_id:
                logger.error(f"Failed to create task from issue {issue_id}")
                return False
            
            # Track active task
            self.active_tasks[issue_id] = {
                "task_id": task_id,
                "issue": issue,
                "assignment": assignment,
                "status": "created",
                "created_at": datetime.now(),
                "last_update": datetime.now()
            }
            
            # Start task if auto-start is enabled
            if self.auto_start_tasks:
                await self._start_task(issue_id, task_id)
            
            # Add initial comment to Linear issue
            await self._add_assignment_comment(issue, task_id)
            
            logger.info(f"Successfully handled new assignment for issue {issue_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling new assignment: {e}")
            return False
    
    async def _handle_unassignment(self, assignment: AssignmentEvent) -> bool:
        """Handle unassignment from bot"""
        try:
            issue_id = assignment.issue_id
            
            # Stop any active task
            if issue_id in self.active_tasks:
                task_info = self.active_tasks[issue_id]
                await self._stop_task(issue_id, task_info["task_id"])
                
                # Remove from active tasks
                del self.active_tasks[issue_id]
            
            # Update issue status
            if self.auto_update_status:
                await self.update_issue_status(issue_id, "unassigned")
            
            logger.info(f"Successfully handled unassignment for issue {issue_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling unassignment: {e}")
            return False
    
    async def _handle_reassignment(self, assignment: AssignmentEvent) -> bool:
        """Handle reassignment"""
        # For now, treat as unassignment if bot was previous assignee
        # or new assignment if bot is new assignee
        if assignment.previous_assignee_id:
            await self._handle_unassignment(assignment)
        
        if assignment.assignee_id:
            await self._handle_new_assignment(assignment)
        
        return True
    
    async def update_issue_status(self, issue_id: str, status: str) -> bool:
        """Update issue status"""
        try:
            if not self.linear_client:
                logger.error("Linear client not set")
                return False
            
            # Map internal status to Linear state
            linear_status = self.status_mapping.get(status, status)
            
            # This would need to be implemented with proper state ID mapping
            # For now, just log the action
            logger.info(f"Would update issue {issue_id} status to: {linear_status}")
            
            # TODO: Implement actual status update with proper state IDs
            # success = await self.linear_client.update_issue(issue_id, {"stateId": state_id})
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating issue status: {e}")
            return False
    
    async def create_task_from_issue(self, issue: LinearIssue) -> Optional[str]:
        """Create OpenAlpha_Evolve task from Linear issue"""
        try:
            return await self.task_creator.create_task_from_issue(issue)
        except Exception as e:
            logger.error(f"Error creating task from issue: {e}")
            return None
    
    async def sync_progress(self, issue_id: str, progress: Dict[str, Any]) -> bool:
        """Sync progress back to Linear"""
        try:
            return await self.progress_sync.sync_progress(issue_id, progress)
        except Exception as e:
            logger.error(f"Error syncing progress: {e}")
            return False
    
    async def _start_task(self, issue_id: str, task_id: str) -> bool:
        """Start OpenAlpha_Evolve task"""
        try:
            logger.info(f"Starting task {task_id} for issue {issue_id}")
            
            # Update task status
            if issue_id in self.active_tasks:
                self.active_tasks[issue_id]["status"] = "started"
                self.active_tasks[issue_id]["last_update"] = datetime.now()
            
            # TODO: Integrate with TaskManagerAgent to start the evolutionary process
            # This would involve:
            # 1. Loading the task definition
            # 2. Creating a TaskManagerAgent instance
            # 3. Starting the evolutionary cycle
            # 4. Monitoring progress and updating Linear
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting task: {e}")
            return False
    
    async def _stop_task(self, issue_id: str, task_id: str) -> bool:
        """Stop OpenAlpha_Evolve task"""
        try:
            logger.info(f"Stopping task {task_id} for issue {issue_id}")
            
            # Update task status
            if issue_id in self.active_tasks:
                self.active_tasks[issue_id]["status"] = "stopped"
                self.active_tasks[issue_id]["last_update"] = datetime.now()
            
            # TODO: Implement task stopping logic
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping task: {e}")
            return False
    
    async def _add_assignment_comment(self, issue: LinearIssue, task_id: str) -> bool:
        """Add comment to Linear issue about assignment"""
        try:
            if not self.linear_client:
                return False
            
            comment_body = f"""🤖 **OpenAlpha_Evolve Bot Assigned**

I've been assigned to work on this issue and have created task `{task_id}` to begin the evolutionary code generation process.

**What happens next:**
1. 🧬 I'll analyze the requirements and create initial code solutions
2. 🔄 Multiple generations of code will evolve to optimize the solution
3. 🧪 Each generation will be tested and evaluated for correctness and efficiency
4. 📊 I'll provide regular updates on progress and results

You can monitor the progress and I'll update this issue with results when complete.
"""
            
            comment_id = await self.linear_client.create_comment(issue.id, comment_body)
            if comment_id:
                logger.info(f"Added assignment comment {comment_id} to issue {issue.id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding assignment comment: {e}")
            return False
    
    async def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all active tasks"""
        return self.active_tasks.copy()
    
    async def get_task_status(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific task"""
        return self.active_tasks.get(issue_id)
    
    async def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old completed tasks"""
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        completed_tasks = []
        
        for issue_id, task_info in self.active_tasks.items():
            if (task_info["status"] in ["completed", "failed", "stopped"] and 
                task_info["last_update"] < cutoff_time):
                completed_tasks.append(issue_id)
        
        for issue_id in completed_tasks:
            del self.active_tasks[issue_id]
        
        logger.info(f"Cleaned up {len(completed_tasks)} completed tasks")
        return len(completed_tasks)
    
    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow automation statistics"""
        total_tasks = len(self.active_tasks)
        status_counts = {}
        
        for task_info in self.active_tasks.values():
            status = task_info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_active_tasks": total_tasks,
            "status_breakdown": status_counts,
            "auto_start_enabled": self.auto_start_tasks,
            "auto_update_enabled": self.auto_update_status,
            "recent_tasks": [
                {
                    "issue_id": issue_id,
                    "task_id": info["task_id"],
                    "status": info["status"],
                    "created_at": info["created_at"].isoformat(),
                    "last_update": info["last_update"].isoformat()
                }
                for issue_id, info in sorted(
                    self.active_tasks.items(),
                    key=lambda x: x[1]["last_update"],
                    reverse=True
                )[:10]
            ]
        }
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute method for BaseAgent interface"""
        stats = await self.get_workflow_stats()
        return {
            "status": "active",
            "linear_client_connected": self.linear_client is not None,
            "stats": stats
        }


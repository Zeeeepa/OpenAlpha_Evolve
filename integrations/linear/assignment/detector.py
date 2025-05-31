"""
Assignment Detector

Detects assignment changes from Linear webhook events and determines
if the bot should process them.
"""

import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta

from ..interfaces import (
    AssignmentDetectorInterface, 
    WebhookEvent, 
    AssignmentEvent, 
    AssignmentAction,
    LinearIssue
)

logger = logging.getLogger(__name__)


class AssignmentDetector(AssignmentDetectorInterface):
    """Detects and processes assignment changes from webhook events"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Bot configuration
        self.bot_user_id = self.config.get("bot_user_id")
        self.bot_email = self.config.get("bot_email")
        self.bot_names = self.config.get("bot_names", ["codegen", "openalpha", "bot"])
        
        # Assignment processing rules
        self.auto_assign_labels = self.config.get("auto_assign_labels", ["ai", "automation", "codegen"])
        self.auto_assign_keywords = self.config.get("auto_assign_keywords", ["generate", "evolve", "optimize", "automate"])
        self.excluded_teams = self.config.get("excluded_teams", [])
        self.excluded_projects = self.config.get("excluded_projects", [])
        
        # Tracking
        self.recent_assignments: Dict[str, AssignmentEvent] = {}
        self.processed_assignments: Set[str] = set()
        
        logger.info("AssignmentDetector initialized")
    
    async def detect_assignment_change(self, event: WebhookEvent) -> Optional[AssignmentEvent]:
        """Detect assignment changes from webhook events"""
        try:
            if event.type.value not in ["Issue", "IssueUpdate"]:
                return None
            
            data = event.data
            issue_id = data.get("id")
            
            if not issue_id:
                logger.warning("No issue ID in webhook event")
                return None
            
            # Check for assignment changes
            assignment_event = await self._analyze_assignment_change(event)
            
            if assignment_event:
                # Store recent assignment for tracking
                self.recent_assignments[issue_id] = assignment_event
                logger.info(f"Assignment change detected: {assignment_event.action.value} for issue {issue_id}")
            
            return assignment_event
            
        except Exception as e:
            logger.error(f"Error detecting assignment change: {e}")
            return None
    
    async def _analyze_assignment_change(self, event: WebhookEvent) -> Optional[AssignmentEvent]:
        """Analyze webhook event for assignment changes"""
        data = event.data
        issue_id = data.get("id")
        action = event.action
        
        # For issue creation
        if action == "create":
            assignee = data.get("assignee")
            if assignee and await self._is_bot_user(assignee):
                return AssignmentEvent(
                    issue_id=issue_id,
                    action=AssignmentAction.ASSIGNED,
                    assignee_id=assignee.get("id"),
                    timestamp=event.timestamp,
                    metadata={"event_type": "create", "initial_assignment": True}
                )
        
        # For issue updates
        elif action == "update":
            current_assignee = data.get("assignee")
            
            # Try to get previous assignee from updatedFrom field
            updated_from = data.get("updatedFrom", {})
            previous_assignee = updated_from.get("assignee")
            
            # Determine assignment action
            if current_assignee and not previous_assignee:
                # New assignment
                if await self._is_bot_user(current_assignee):
                    return AssignmentEvent(
                        issue_id=issue_id,
                        action=AssignmentAction.ASSIGNED,
                        assignee_id=current_assignee.get("id"),
                        timestamp=event.timestamp,
                        metadata={"event_type": "update", "assignment_type": "new"}
                    )
            
            elif not current_assignee and previous_assignee:
                # Unassignment
                if await self._is_bot_user(previous_assignee):
                    return AssignmentEvent(
                        issue_id=issue_id,
                        action=AssignmentAction.UNASSIGNED,
                        previous_assignee_id=previous_assignee.get("id"),
                        timestamp=event.timestamp,
                        metadata={"event_type": "update", "assignment_type": "removed"}
                    )
            
            elif current_assignee and previous_assignee:
                current_id = current_assignee.get("id")
                previous_id = previous_assignee.get("id")
                
                if current_id != previous_id:
                    # Reassignment
                    if await self._is_bot_user(current_assignee):
                        return AssignmentEvent(
                            issue_id=issue_id,
                            action=AssignmentAction.ASSIGNED,
                            assignee_id=current_id,
                            previous_assignee_id=previous_id,
                            timestamp=event.timestamp,
                            metadata={"event_type": "update", "assignment_type": "reassigned"}
                        )
                    elif await self._is_bot_user(previous_assignee):
                        return AssignmentEvent(
                            issue_id=issue_id,
                            action=AssignmentAction.UNASSIGNED,
                            assignee_id=current_id,
                            previous_assignee_id=previous_id,
                            timestamp=event.timestamp,
                            metadata={"event_type": "update", "assignment_type": "reassigned_away"}
                        )
        
        return None
    
    async def _is_bot_user(self, user: Dict[str, Any]) -> bool:
        """Check if user is the bot"""
        if not user:
            return False
        
        user_id = user.get("id")
        user_email = user.get("email", "").lower()
        user_name = user.get("name", "").lower()
        
        # Check by user ID
        if self.bot_user_id and user_id == self.bot_user_id:
            return True
        
        # Check by email
        if self.bot_email and user_email == self.bot_email.lower():
            return True
        
        # Check by name patterns
        for bot_name in self.bot_names:
            if bot_name.lower() in user_name or bot_name.lower() in user_email:
                return True
        
        return False
    
    async def is_bot_assigned(self, issue: LinearIssue) -> bool:
        """Check if bot is assigned to issue"""
        if not issue.assignee_id:
            return False
        
        # Create user dict for checking
        user = {
            "id": issue.assignee_id,
            "name": issue.assignee_name or "",
            "email": ""  # Would need to fetch from API if needed
        }
        
        return await self._is_bot_user(user)
    
    async def should_process_assignment(self, assignment: AssignmentEvent) -> bool:
        """Determine if assignment should be processed"""
        try:
            issue_id = assignment.issue_id
            
            # Check if already processed recently
            if issue_id in self.processed_assignments:
                logger.debug(f"Assignment for issue {issue_id} already processed recently")
                return False
            
            # Only process assignments to the bot
            if assignment.action == AssignmentAction.UNASSIGNED:
                # Always process unassignments for cleanup
                return True
            elif assignment.action != AssignmentAction.ASSIGNED:
                return False
            
            # Check assignment rules
            if not await self._check_assignment_rules(assignment):
                return False
            
            # Check rate limiting
            if not await self._check_rate_limits(assignment):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if assignment should be processed: {e}")
            return False
    
    async def _check_assignment_rules(self, assignment: AssignmentEvent) -> bool:
        """Check assignment processing rules"""
        # This would typically fetch issue details to check rules
        # For now, return True - rules would be implemented based on requirements
        return True
    
    async def _check_rate_limits(self, assignment: AssignmentEvent) -> bool:
        """Check rate limits for assignment processing"""
        # Implement rate limiting logic
        # For example, limit assignments per hour/day
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Count recent assignments
        recent_count = sum(
            1 for event in self.recent_assignments.values()
            if event.timestamp > hour_ago and event.action == AssignmentAction.ASSIGNED
        )
        
        max_assignments_per_hour = self.config.get("max_assignments_per_hour", 10)
        
        if recent_count >= max_assignments_per_hour:
            logger.warning(f"Rate limit exceeded: {recent_count} assignments in the last hour")
            return False
        
        return True
    
    async def detect_auto_assignment_candidates(self, event: WebhookEvent) -> bool:
        """Detect if issue should be auto-assigned to bot"""
        try:
            if event.type.value not in ["Issue"] or event.action != "create":
                return False
            
            data = event.data
            
            # Check if already assigned
            if data.get("assignee"):
                return False
            
            # Check labels
            labels = data.get("labels", {}).get("nodes", [])
            label_names = [label.get("name", "").lower() for label in labels]
            
            if any(label in label_names for label in self.auto_assign_labels):
                logger.info(f"Auto-assignment candidate found by label: {data.get('id')}")
                return True
            
            # Check title and description for keywords
            title = data.get("title", "").lower()
            description = data.get("description", "").lower()
            text_content = f"{title} {description}"
            
            if any(keyword in text_content for keyword in self.auto_assign_keywords):
                logger.info(f"Auto-assignment candidate found by keywords: {data.get('id')}")
                return True
            
            # Check team exclusions
            team = data.get("team", {})
            if team.get("id") in self.excluded_teams or team.get("key") in self.excluded_teams:
                return False
            
            # Check project exclusions
            project = data.get("project", {})
            if project and project.get("id") in self.excluded_projects:
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting auto-assignment candidate: {e}")
            return False
    
    async def mark_assignment_processed(self, issue_id: str) -> None:
        """Mark assignment as processed"""
        self.processed_assignments.add(issue_id)
        logger.debug(f"Marked assignment for issue {issue_id} as processed")
    
    async def get_recent_assignments(self, hours: int = 24) -> List[AssignmentEvent]:
        """Get recent assignments within specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            event for event in self.recent_assignments.values()
            if event.timestamp > cutoff_time
        ]
    
    async def cleanup_old_assignments(self, hours: int = 48) -> int:
        """Clean up old assignment tracking data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Clean recent assignments
        old_assignments = [
            issue_id for issue_id, event in self.recent_assignments.items()
            if event.timestamp < cutoff_time
        ]
        
        for issue_id in old_assignments:
            del self.recent_assignments[issue_id]
        
        # Clean processed assignments (keep some for deduplication)
        # This is a simple approach - in production, you might want more sophisticated cleanup
        if len(self.processed_assignments) > 1000:
            # Keep only the most recent 500
            self.processed_assignments = set(list(self.processed_assignments)[-500:])
        
        logger.info(f"Cleaned up {len(old_assignments)} old assignment records")
        return len(old_assignments)
    
    async def get_assignment_stats(self) -> Dict[str, Any]:
        """Get assignment detection statistics"""
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        hour_ago = now - timedelta(hours=1)
        
        recent_assignments = list(self.recent_assignments.values())
        
        assignments_last_day = [e for e in recent_assignments if e.timestamp > day_ago]
        assignments_last_hour = [e for e in recent_assignments if e.timestamp > hour_ago]
        
        return {
            "total_tracked_assignments": len(self.recent_assignments),
            "processed_assignments": len(self.processed_assignments),
            "assignments_last_24h": len(assignments_last_day),
            "assignments_last_hour": len(assignments_last_hour),
            "assignment_actions": {
                "assigned": len([e for e in assignments_last_day if e.action == AssignmentAction.ASSIGNED]),
                "unassigned": len([e for e in assignments_last_day if e.action == AssignmentAction.UNASSIGNED]),
                "reassigned": len([e for e in assignments_last_day if e.action == AssignmentAction.REASSIGNED])
            }
        }
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute method for BaseAgent interface"""
        stats = await self.get_assignment_stats()
        return {
            "status": "active",
            "bot_user_id": self.bot_user_id,
            "bot_email": self.bot_email,
            "stats": stats
        }


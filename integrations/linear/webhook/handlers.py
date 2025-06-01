"""
Webhook Event Handlers

Contains handlers for different types of Linear webhook events.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from ..interfaces import WebhookEvent, LinearEventType, AssignmentEvent, AssignmentAction

logger = logging.getLogger(__name__)


class WebhookHandlers:
    """Collection of webhook event handlers"""
    
    def __init__(self):
        self.handlers: Dict[LinearEventType, List[Callable]] = {}
        self.global_handlers: List[Callable] = []
        
    def register_handler(self, event_type: LinearEventType, handler: Callable) -> None:
        """Register a handler for a specific event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type.value}")
    
    def register_global_handler(self, handler: Callable) -> None:
        """Register a handler that receives all events"""
        self.global_handlers.append(handler)
        logger.info("Registered global event handler")
    
    async def handle_event(self, event: WebhookEvent) -> bool:
        """Handle a webhook event by calling appropriate handlers"""
        try:
            success = True
            
            # Call global handlers first
            for handler in self.global_handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Global handler failed: {e}")
                    success = False
            
            # Call specific handlers
            if event.type in self.handlers:
                for handler in self.handlers[event.type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Handler for {event.type.value} failed: {e}")
                        success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error handling webhook event: {e}")
            return False
    
    async def handle_issue_event(self, event: WebhookEvent) -> None:
        """Handle issue-related events"""
        try:
            data = event.data
            action = event.action
            
            logger.info(f"Handling issue event: {action} for issue {data.get('id')}")
            
            if action == "create":
                await self._handle_issue_created(data)
            elif action == "update":
                await self._handle_issue_updated(data)
            elif action == "remove":
                await self._handle_issue_deleted(data)
            
        except Exception as e:
            logger.error(f"Error handling issue event: {e}")
    
    async def handle_comment_event(self, event: WebhookEvent) -> None:
        """Handle comment-related events"""
        try:
            data = event.data
            action = event.action
            
            logger.info(f"Handling comment event: {action} for comment {data.get('id')}")
            
            if action == "create":
                await self._handle_comment_created(data)
            elif action == "update":
                await self._handle_comment_updated(data)
            elif action == "remove":
                await self._handle_comment_deleted(data)
            
        except Exception as e:
            logger.error(f"Error handling comment event: {e}")
    
    async def handle_project_event(self, event: WebhookEvent) -> None:
        """Handle project-related events"""
        try:
            data = event.data
            action = event.action
            
            logger.info(f"Handling project event: {action} for project {data.get('id')}")
            
            if action == "create":
                await self._handle_project_created(data)
            elif action == "update":
                await self._handle_project_updated(data)
            elif action == "remove":
                await self._handle_project_deleted(data)
            
        except Exception as e:
            logger.error(f"Error handling project event: {e}")
    
    async def _handle_issue_created(self, data: Dict[str, Any]) -> None:
        """Handle issue creation"""
        issue_id = data.get("id")
        title = data.get("title")
        assignee = data.get("assignee")
        
        logger.info(f"Issue created: {title} (ID: {issue_id})")
        
        # Check if issue is assigned to bot
        if assignee and self._is_bot_user(assignee):
            logger.info(f"Bot assigned to new issue: {issue_id}")
            # Trigger assignment handling
            await self._trigger_assignment_event(issue_id, AssignmentAction.ASSIGNED, assignee.get("id"))
    
    async def _handle_issue_updated(self, data: Dict[str, Any]) -> None:
        """Handle issue updates"""
        issue_id = data.get("id")
        title = data.get("title")
        
        logger.info(f"Issue updated: {title} (ID: {issue_id})")
        
        # Check for assignment changes
        await self._check_assignment_change(data)
        
        # Check for state changes
        await self._check_state_change(data)
        
        # Check for priority changes
        await self._check_priority_change(data)
    
    async def _handle_issue_deleted(self, data: Dict[str, Any]) -> None:
        """Handle issue deletion"""
        issue_id = data.get("id")
        logger.info(f"Issue deleted: {issue_id}")
        
        # Clean up any related tasks or data
        await self._cleanup_issue_data(issue_id)
    
    async def _handle_comment_created(self, data: Dict[str, Any]) -> None:
        """Handle comment creation"""
        comment_id = data.get("id")
        issue_id = data.get("issueId")
        body = data.get("body", "")
        user = data.get("user", {})
        
        logger.info(f"Comment created on issue {issue_id}: {comment_id}")
        
        # Check if comment mentions bot or contains commands
        if self._contains_bot_mention(body) or self._contains_commands(body):
            logger.info(f"Bot mentioned in comment {comment_id}")
            await self._handle_bot_mention(issue_id, comment_id, body, user)
    
    async def _handle_comment_updated(self, data: Dict[str, Any]) -> None:
        """Handle comment updates"""
        comment_id = data.get("id")
        issue_id = data.get("issueId")
        body = data.get("body", "")
        
        logger.info(f"Comment updated on issue {issue_id}: {comment_id}")
        
        # Re-check for bot mentions or commands
        if self._contains_bot_mention(body) or self._contains_commands(body):
            await self._handle_bot_mention(issue_id, comment_id, body, data.get("user", {}))
    
    async def _handle_comment_deleted(self, data: Dict[str, Any]) -> None:
        """Handle comment deletion"""
        comment_id = data.get("id")
        logger.info(f"Comment deleted: {comment_id}")
    
    async def _handle_project_created(self, data: Dict[str, Any]) -> None:
        """Handle project creation"""
        project_id = data.get("id")
        name = data.get("name")
        logger.info(f"Project created: {name} (ID: {project_id})")
    
    async def _handle_project_updated(self, data: Dict[str, Any]) -> None:
        """Handle project updates"""
        project_id = data.get("id")
        name = data.get("name")
        logger.info(f"Project updated: {name} (ID: {project_id})")
    
    async def _handle_project_deleted(self, data: Dict[str, Any]) -> None:
        """Handle project deletion"""
        project_id = data.get("id")
        logger.info(f"Project deleted: {project_id}")
    
    async def _check_assignment_change(self, data: Dict[str, Any]) -> None:
        """Check for assignment changes in issue update"""
        issue_id = data.get("id")
        current_assignee = data.get("assignee")
        previous_assignee = data.get("previousAssignee")  # If available
        
        # Determine assignment action
        if current_assignee and not previous_assignee:
            # New assignment
            if self._is_bot_user(current_assignee):
                await self._trigger_assignment_event(
                    issue_id, 
                    AssignmentAction.ASSIGNED, 
                    current_assignee.get("id")
                )
        elif not current_assignee and previous_assignee:
            # Unassignment
            if self._is_bot_user(previous_assignee):
                await self._trigger_assignment_event(
                    issue_id, 
                    AssignmentAction.UNASSIGNED, 
                    None, 
                    previous_assignee.get("id")
                )
        elif current_assignee and previous_assignee and current_assignee.get("id") != previous_assignee.get("id"):
            # Reassignment
            if self._is_bot_user(current_assignee):
                await self._trigger_assignment_event(
                    issue_id, 
                    AssignmentAction.ASSIGNED, 
                    current_assignee.get("id"),
                    previous_assignee.get("id")
                )
            elif self._is_bot_user(previous_assignee):
                await self._trigger_assignment_event(
                    issue_id, 
                    AssignmentAction.UNASSIGNED, 
                    current_assignee.get("id"),
                    previous_assignee.get("id")
                )
    
    async def _check_state_change(self, data: Dict[str, Any]) -> None:
        """Check for state changes in issue update"""
        issue_id = data.get("id")
        current_state = data.get("state", {})
        previous_state = data.get("previousState", {})  # If available
        
        if current_state.get("id") != previous_state.get("id"):
            logger.info(f"Issue {issue_id} state changed from {previous_state.get('name')} to {current_state.get('name')}")
            
            # Handle specific state transitions
            state_name = current_state.get("name", "").lower()
            if state_name in ["done", "completed", "closed"]:
                await self._handle_issue_completion(issue_id)
            elif state_name in ["in progress", "started", "active"]:
                await self._handle_issue_started(issue_id)
    
    async def _check_priority_change(self, data: Dict[str, Any]) -> None:
        """Check for priority changes in issue update"""
        issue_id = data.get("id")
        current_priority = data.get("priority")
        previous_priority = data.get("previousPriority")  # If available
        
        if current_priority != previous_priority:
            logger.info(f"Issue {issue_id} priority changed from {previous_priority} to {current_priority}")
            
            # Handle high priority issues
            if current_priority and current_priority >= 1:  # High priority
                await self._handle_high_priority_issue(issue_id)
    
    def _is_bot_user(self, user: Dict[str, Any]) -> bool:
        """Check if user is the bot"""
        # This should be configured based on the bot's user ID or email
        bot_identifiers = [
            "codegen@bot.linear.app",  # Example bot email
            "bot-user-id"  # Example bot user ID
        ]
        
        user_email = user.get("email", "")
        user_id = user.get("id", "")
        
        return user_email in bot_identifiers or user_id in bot_identifiers
    
    def _contains_bot_mention(self, text: str) -> bool:
        """Check if text contains bot mention"""
        bot_mentions = ["@codegen", "@bot", "@openalpha"]
        text_lower = text.lower()
        return any(mention in text_lower for mention in bot_mentions)
    
    def _contains_commands(self, text: str) -> bool:
        """Check if text contains bot commands"""
        commands = ["/evolve", "/generate", "/optimize", "/test", "/help"]
        text_lower = text.lower()
        return any(command in text_lower for command in commands)
    
    async def _trigger_assignment_event(
        self, 
        issue_id: str, 
        action: AssignmentAction, 
        assignee_id: Optional[str] = None,
        previous_assignee_id: Optional[str] = None
    ) -> None:
        """Trigger assignment event for processing"""
        assignment_event = AssignmentEvent(
            issue_id=issue_id,
            action=action,
            assignee_id=assignee_id,
            previous_assignee_id=previous_assignee_id,
            timestamp=datetime.now(),
            metadata={}
        )
        
        # This would typically be sent to the assignment detector or workflow automation
        logger.info(f"Assignment event triggered: {action.value} for issue {issue_id}")
        
        # TODO: Send to assignment detector/workflow automation
        # await self.assignment_detector.process_assignment(assignment_event)
    
    async def _handle_bot_mention(
        self, 
        issue_id: str, 
        comment_id: str, 
        body: str, 
        user: Dict[str, Any]
    ) -> None:
        """Handle bot mention in comment"""
        logger.info(f"Bot mentioned in comment {comment_id} on issue {issue_id}")
        
        # Parse commands from comment
        commands = self._parse_commands(body)
        
        # TODO: Process commands
        # await self.command_processor.process_commands(issue_id, commands, user)
    
    def _parse_commands(self, text: str) -> List[str]:
        """Parse commands from text"""
        commands = []
        text_lower = text.lower()
        
        command_patterns = [
            "/evolve", "/generate", "/optimize", "/test", "/help",
            "/start", "/stop", "/status", "/report"
        ]
        
        for pattern in command_patterns:
            if pattern in text_lower:
                commands.append(pattern)
        
        return commands
    
    async def _cleanup_issue_data(self, issue_id: str) -> None:
        """Clean up data related to deleted issue"""
        logger.info(f"Cleaning up data for deleted issue: {issue_id}")
        # TODO: Implement cleanup logic
    
    async def _handle_issue_completion(self, issue_id: str) -> None:
        """Handle issue completion"""
        logger.info(f"Issue completed: {issue_id}")
        # TODO: Handle completion logic
    
    async def _handle_issue_started(self, issue_id: str) -> None:
        """Handle issue started"""
        logger.info(f"Issue started: {issue_id}")
        # TODO: Handle start logic
    
    async def _handle_high_priority_issue(self, issue_id: str) -> None:
        """Handle high priority issue"""
        logger.info(f"High priority issue: {issue_id}")
        # TODO: Handle high priority logic


"""
Linear Integration Agent

Main orchestrator for Linear API integration that coordinates all components
including GraphQL client, webhook processing, assignment detection, and workflow automation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .interfaces import LinearIntegrationAgentInterface
from .client.graphql_client import LinearGraphQLClient
from .webhook.processor import WebhookProcessor
from .assignment.detector import AssignmentDetector
from .workflow.automation import WorkflowAutomation
from .events.manager import EventManager

logger = logging.getLogger(__name__)


class LinearIntegrationAgent(LinearIntegrationAgentInterface):
    """Main Linear integration agent that orchestrates all components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Configuration
        self.api_key = self.config.get("api_key")
        self.webhook_secret = self.config.get("webhook_secret")
        self.bot_user_id = self.config.get("bot_user_id")
        self.bot_email = self.config.get("bot_email")
        
        # Component configuration
        client_config = {
            "api_key": self.api_key,
            "timeout": self.config.get("timeout", 30),
            "max_retries": self.config.get("max_retries", 3),
            "rate_limit_requests": self.config.get("rate_limit_requests", 100),
            "rate_limit_window": self.config.get("rate_limit_window", 60),
            "cache_ttl": self.config.get("cache_ttl", 300)
        }
        
        webhook_config = {
            "webhook_secret": self.webhook_secret,
            "max_retries": self.config.get("webhook_max_retries", 3),
            "retry_delay": self.config.get("webhook_retry_delay", 5)
        }
        
        assignment_config = {
            "bot_user_id": self.bot_user_id,
            "bot_email": self.bot_email,
            "bot_names": self.config.get("bot_names", ["codegen", "openalpha"]),
            "auto_assign_labels": self.config.get("auto_assign_labels", ["ai", "automation"]),
            "auto_assign_keywords": self.config.get("auto_assign_keywords", ["generate", "evolve"]),
            "max_assignments_per_hour": self.config.get("max_assignments_per_hour", 10)
        }
        
        workflow_config = {
            "auto_start_tasks": self.config.get("auto_start_tasks", True),
            "auto_update_status": self.config.get("auto_update_status", True)
        }
        
        # Initialize components
        self.linear_client = LinearGraphQLClient(client_config)
        self.webhook_processor = WebhookProcessor(webhook_config)
        self.assignment_detector = AssignmentDetector(assignment_config)
        self.workflow_automation = WorkflowAutomation(workflow_config)
        self.event_manager = EventManager(self.config.get("event_manager", {}))
        
        # Set up component relationships
        self.workflow_automation.set_linear_client(self.linear_client)
        
        # State tracking
        self.initialized = False
        self.monitoring_active = False
        self.last_sync = None
        
        # Register webhook handlers
        self._register_webhook_handlers()
        
        logger.info("LinearIntegrationAgent initialized")
    
    def _register_webhook_handlers(self) -> None:
        """Register webhook event handlers"""
        # Register assignment detection handler
        async def assignment_handler(event):
            assignment_event = await self.assignment_detector.detect_assignment_change(event)
            if assignment_event and await self.assignment_detector.should_process_assignment(assignment_event):
                await self.workflow_automation.handle_assignment(assignment_event)
                await self.assignment_detector.mark_assignment_processed(assignment_event.issue_id)
        
        # Register auto-assignment handler
        async def auto_assignment_handler(event):
            if await self.assignment_detector.detect_auto_assignment_candidates(event):
                # TODO: Implement auto-assignment logic
                logger.info(f"Auto-assignment candidate detected: {event.data.get('id')}")
        
        self.webhook_processor.register_global_handler(assignment_handler)
        self.webhook_processor.register_global_handler(auto_assignment_handler)
    
    async def initialize(self) -> bool:
        """Initialize the integration"""
        try:
            logger.info("Initializing Linear integration...")
            
            # Authenticate with Linear
            if not self.api_key:
                logger.error("No Linear API key provided")
                return False
            
            if not await self.linear_client.authenticate(self.api_key):
                logger.error("Failed to authenticate with Linear API")
                return False
            
            # Verify bot user configuration
            if not await self._verify_bot_configuration():
                logger.warning("Bot configuration verification failed")
            
            # Initialize event manager
            await self.event_manager.initialize()
            
            self.initialized = True
            self.last_sync = datetime.now()
            
            logger.info("Linear integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Linear integration: {e}")
            return False
    
    async def _verify_bot_configuration(self) -> bool:
        """Verify bot user configuration"""
        try:
            # Get current user info
            user_info = self.linear_client.user_info
            if not user_info:
                logger.error("Could not get user info from Linear")
                return False
            
            # Update bot configuration if needed
            if not self.bot_user_id:
                self.bot_user_id = user_info.get("id")
                logger.info(f"Set bot user ID: {self.bot_user_id}")
            
            if not self.bot_email:
                self.bot_email = user_info.get("email")
                logger.info(f"Set bot email: {self.bot_email}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying bot configuration: {e}")
            return False
    
    async def handle_webhook(self, payload: Dict[str, Any], signature: str) -> bool:
        """Handle incoming webhook"""
        try:
            if not self.initialized:
                logger.error("Integration not initialized")
                return False
            
            # Convert payload to bytes for processing
            import json
            payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
            
            # Process webhook
            success = await self.webhook_processor.process_webhook(
                payload_bytes, 
                signature,
                headers={"content-type": "application/json"}
            )
            
            if success:
                logger.info("Webhook processed successfully")
            else:
                logger.error("Webhook processing failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return False
    
    async def monitor_assignments(self) -> None:
        """Monitor for new assignments"""
        try:
            if not self.initialized:
                logger.error("Integration not initialized")
                return
            
            self.monitoring_active = True
            logger.info("Starting assignment monitoring...")
            
            while self.monitoring_active:
                try:
                    # Get assigned issues
                    if self.bot_user_id:
                        assigned_issues = await self.linear_client.get_user_assigned_issues(
                            self.bot_user_id, 
                            limit=50
                        )
                        
                        # Process each assigned issue
                        for issue in assigned_issues:
                            await self._process_assigned_issue(issue)
                    
                    # Wait before next check
                    await asyncio.sleep(self.config.get("monitoring_interval", 60))
                    
                except Exception as e:
                    logger.error(f"Error in assignment monitoring loop: {e}")
                    await asyncio.sleep(30)  # Wait before retrying
            
            logger.info("Assignment monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error in assignment monitoring: {e}")
        finally:
            self.monitoring_active = False
    
    async def _process_assigned_issue(self, issue) -> None:
        """Process an assigned issue"""
        try:
            # Check if we've already processed this assignment
            if issue.id in self.assignment_detector.processed_assignments:
                return
            
            # Check if issue should be processed
            if await self.assignment_detector.is_bot_assigned(issue):
                # Create assignment event
                from .interfaces import AssignmentEvent, AssignmentAction
                assignment_event = AssignmentEvent(
                    issue_id=issue.id,
                    action=AssignmentAction.ASSIGNED,
                    assignee_id=issue.assignee_id,
                    timestamp=datetime.now(),
                    metadata={"source": "monitoring"}
                )
                
                # Process assignment
                if await self.assignment_detector.should_process_assignment(assignment_event):
                    await self.workflow_automation.handle_assignment(assignment_event)
                    await self.assignment_detector.mark_assignment_processed(issue.id)
            
        except Exception as e:
            logger.error(f"Error processing assigned issue {issue.id}: {e}")
    
    async def sync_with_linear(self) -> bool:
        """Sync state with Linear"""
        try:
            logger.info("Syncing with Linear...")
            
            # Get active tasks
            active_tasks = await self.workflow_automation.get_active_tasks()
            
            # Sync progress for each active task
            sync_count = 0
            for issue_id, task_info in active_tasks.items():
                try:
                    # TODO: Get actual progress from task execution
                    progress = {
                        "status": task_info["status"],
                        "last_update": task_info["last_update"].isoformat(),
                        "generation": 0,  # Would come from actual task
                        "best_fitness": 0.0  # Would come from actual task
                    }
                    
                    if await self.workflow_automation.sync_progress(issue_id, progress):
                        sync_count += 1
                
                except Exception as e:
                    logger.error(f"Error syncing progress for issue {issue_id}: {e}")
            
            self.last_sync = datetime.now()
            logger.info(f"Synced {sync_count} tasks with Linear")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing with Linear: {e}")
            return False
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status"""
        try:
            # Get component statuses
            client_status = await self.linear_client.execute()
            webhook_status = await self.webhook_processor.execute()
            assignment_status = await self.assignment_detector.execute()
            workflow_status = await self.workflow_automation.execute()
            
            # Get statistics
            webhook_stats = await self.webhook_processor.get_processing_stats()
            assignment_stats = await self.assignment_detector.get_assignment_stats()
            workflow_stats = await self.workflow_automation.get_workflow_stats()
            
            return {
                "initialized": self.initialized,
                "monitoring_active": self.monitoring_active,
                "last_sync": self.last_sync.isoformat() if self.last_sync else None,
                "components": {
                    "linear_client": client_status,
                    "webhook_processor": webhook_status,
                    "assignment_detector": assignment_status,
                    "workflow_automation": workflow_status
                },
                "statistics": {
                    "webhooks": webhook_stats,
                    "assignments": assignment_stats,
                    "workflows": workflow_stats
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {
                "initialized": self.initialized,
                "error": str(e)
            }
    
    async def stop_monitoring(self) -> None:
        """Stop assignment monitoring"""
        self.monitoring_active = False
        logger.info("Assignment monitoring stopped")
    
    async def cleanup(self) -> None:
        """Clean up resources and old data"""
        try:
            logger.info("Cleaning up Linear integration...")
            
            # Clean up old events
            await self.webhook_processor.cleanup_old_events()
            
            # Clean up old assignments
            await self.assignment_detector.cleanup_old_assignments()
            
            # Clean up completed tasks
            await self.workflow_automation.cleanup_completed_tasks()
            
            # Clear caches
            await self.linear_client.cache.clear()
            
            logger.info("Linear integration cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute method for BaseAgent interface"""
        return await self.get_integration_status()


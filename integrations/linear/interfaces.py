"""
Linear Integration Interfaces

Defines abstract base classes and data structures for Linear API integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time

from core.interfaces import BaseAgent


class LinearEventType(Enum):
    """Linear webhook event types"""
    ISSUE_CREATE = "Issue"
    ISSUE_UPDATE = "IssueUpdate"
    COMMENT_CREATE = "Comment"
    COMMENT_UPDATE = "CommentUpdate"
    PROJECT_UPDATE = "ProjectUpdate"
    CYCLE_UPDATE = "CycleUpdate"


class AssignmentAction(Enum):
    """Assignment detection actions"""
    ASSIGNED = "assigned"
    UNASSIGNED = "unassigned"
    REASSIGNED = "reassigned"


@dataclass
class LinearIssue:
    """Linear issue data structure"""
    id: str
    title: str
    description: Optional[str] = None
    state: Optional[str] = None
    assignee_id: Optional[str] = None
    assignee_name: Optional[str] = None
    team_id: Optional[str] = None
    team_name: Optional[str] = None
    project_id: Optional[str] = None
    cycle_id: Optional[str] = None
    priority: Optional[int] = None
    labels: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    url: Optional[str] = None
    branch_name: Optional[str] = None


@dataclass
class LinearComment:
    """Linear comment data structure"""
    id: str
    body: str
    issue_id: str
    user_id: str
    user_name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class WebhookEvent:
    """Webhook event data structure"""
    id: str
    type: LinearEventType
    action: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False
    retry_count: int = 0


@dataclass
class AssignmentEvent:
    """Assignment detection event"""
    issue_id: str
    action: AssignmentAction
    assignee_id: Optional[str] = None
    previous_assignee_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LinearGraphQLClientInterface(BaseAgent):
    """Interface for Linear GraphQL client"""
    
    @abstractmethod
    async def authenticate(self, api_key: str) -> bool:
        """Authenticate with Linear API"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute GraphQL query"""
        pass
    
    @abstractmethod
    async def get_issue(self, issue_id: str) -> Optional[LinearIssue]:
        """Get issue by ID"""
        pass
    
    @abstractmethod
    async def update_issue(self, issue_id: str, updates: Dict[str, Any]) -> bool:
        """Update issue"""
        pass
    
    @abstractmethod
    async def create_comment(self, issue_id: str, body: str) -> Optional[str]:
        """Create comment on issue"""
        pass
    
    @abstractmethod
    async def get_team_issues(self, team_id: str, limit: int = 50) -> List[LinearIssue]:
        """Get issues for a team"""
        pass


class WebhookProcessorInterface(BaseAgent):
    """Interface for webhook event processing"""
    
    @abstractmethod
    async def process_event(self, event: WebhookEvent) -> bool:
        """Process a webhook event"""
        pass
    
    @abstractmethod
    async def validate_webhook(self, payload: Dict[str, Any], signature: str) -> bool:
        """Validate webhook signature"""
        pass
    
    @abstractmethod
    def register_handler(self, event_type: LinearEventType, handler: Callable) -> None:
        """Register event handler"""
        pass


class AssignmentDetectorInterface(BaseAgent):
    """Interface for assignment detection"""
    
    @abstractmethod
    async def detect_assignment_change(self, event: WebhookEvent) -> Optional[AssignmentEvent]:
        """Detect assignment changes from webhook events"""
        pass
    
    @abstractmethod
    async def is_bot_assigned(self, issue: LinearIssue) -> bool:
        """Check if bot is assigned to issue"""
        pass
    
    @abstractmethod
    async def should_process_assignment(self, assignment: AssignmentEvent) -> bool:
        """Determine if assignment should be processed"""
        pass


class WorkflowAutomationInterface(BaseAgent):
    """Interface for workflow automation"""
    
    @abstractmethod
    async def handle_assignment(self, assignment: AssignmentEvent) -> bool:
        """Handle new assignment"""
        pass
    
    @abstractmethod
    async def update_issue_status(self, issue_id: str, status: str) -> bool:
        """Update issue status"""
        pass
    
    @abstractmethod
    async def create_task_from_issue(self, issue: LinearIssue) -> Optional[str]:
        """Create OpenAlpha_Evolve task from Linear issue"""
        pass
    
    @abstractmethod
    async def sync_progress(self, issue_id: str, progress: Dict[str, Any]) -> bool:
        """Sync progress back to Linear"""
        pass


class EventManagerInterface(BaseAgent):
    """Interface for event management"""
    
    @abstractmethod
    async def queue_event(self, event: WebhookEvent) -> bool:
        """Queue event for processing"""
        pass
    
    @abstractmethod
    async def process_queue(self) -> int:
        """Process queued events"""
        pass
    
    @abstractmethod
    async def retry_failed_events(self) -> int:
        """Retry failed events"""
        pass
    
    @abstractmethod
    async def get_event_status(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get event processing status"""
        pass


class LinearIntegrationAgentInterface(BaseAgent):
    """Main Linear integration agent interface"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the integration"""
        pass
    
    @abstractmethod
    async def handle_webhook(self, payload: Dict[str, Any], signature: str) -> bool:
        """Handle incoming webhook"""
        pass
    
    @abstractmethod
    async def monitor_assignments(self) -> None:
        """Monitor for new assignments"""
        pass
    
    @abstractmethod
    async def sync_with_linear(self) -> bool:
        """Sync state with Linear"""
        pass
    
    @abstractmethod
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status"""
        pass


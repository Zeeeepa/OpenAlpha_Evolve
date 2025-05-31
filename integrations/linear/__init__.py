"""
Linear API Integration System for OpenAlpha_Evolve

This module provides comprehensive Linear API integration including:
- GraphQL client for Linear API communication
- Webhook event processing system
- Assignment detection and routing logic
- Issue lifecycle management automation
- Real-time monitoring and notifications
"""

from .client.graphql_client import LinearGraphQLClient
from .webhook.processor import WebhookProcessor
from .assignment.detector import AssignmentDetector
from .workflow.automation import WorkflowAutomation
from .events.manager import EventManager
from .agent import LinearIntegrationAgent

__version__ = "1.0.0"
__all__ = [
    "LinearGraphQLClient",
    "WebhookProcessor", 
    "AssignmentDetector",
    "WorkflowAutomation",
    "EventManager",
    "LinearIntegrationAgent"
]


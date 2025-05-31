"""
Linear Integration Example

Demonstrates how to use the Linear API integration system with OpenAlpha_Evolve.
"""

import asyncio
import logging
from typing import Dict, Any

from .agent import LinearIntegrationAgent
from .interfaces import LinearIssue, WebhookEvent, LinearEventType
from config.settings import get_linear_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_basic_setup():
    """Example: Basic Linear integration setup"""
    print("🚀 Setting up Linear integration...")
    
    # Get configuration
    config = get_linear_config()
    
    # Create integration agent
    agent = LinearIntegrationAgent(config)
    
    # Initialize the integration
    if await agent.initialize():
        print("✅ Linear integration initialized successfully!")
        
        # Get integration status
        status = await agent.get_integration_status()
        print(f"📊 Integration Status: {status}")
        
        return agent
    else:
        print("❌ Failed to initialize Linear integration")
        return None


async def example_webhook_handling(agent: LinearIntegrationAgent):
    """Example: Handle webhook events"""
    print("\\n📡 Webhook handling example...")
    
    # Example webhook payload (issue assignment)
    webhook_payload = {
        "action": "update",
        "type": "Issue",
        "data": {
            "id": "example_issue_id",
            "title": "Implement binary search algorithm",
            "description": """
            Create a function that implements binary search algorithm.
            
            Example:
            Input: [1, 2, 3, 4, 5], target=3
            Output: 2 (index of target)
            
            Input: [1, 2, 3, 4, 5], target=6  
            Output: -1 (not found)
            """,
            "assignee": {
                "id": "bot_user_id",
                "name": "OpenAlpha Bot",
                "email": "bot@openalpha.com"
            },
            "team": {
                "id": "team_id",
                "name": "AI Team"
            },
            "labels": {
                "nodes": [
                    {"name": "ai", "color": "#ff0000"},
                    {"name": "algorithm", "color": "#00ff00"}
                ]
            }
        }
    }
    
    # Mock webhook signature (in real usage, this comes from Linear)
    webhook_signature = "sha256=mock_signature"
    
    print(f"📥 Processing webhook for issue: {webhook_payload['data']['title']}")
    
    # Handle webhook (will fail signature validation in this example)
    success = await agent.handle_webhook(webhook_payload, webhook_signature)
    
    if success:
        print("✅ Webhook processed successfully!")
    else:
        print("⚠️ Webhook processing failed (expected due to mock signature)")


async def example_manual_task_creation(agent: LinearIntegrationAgent):
    """Example: Manually create task from Linear issue"""
    print("\\n🔧 Manual task creation example...")
    
    # Create example Linear issue
    issue = LinearIssue(
        id="manual_issue_id",
        title="Calculate factorial",
        description="""
        Create a function that calculates the factorial of a number.
        
        Examples:
        factorial(5) should return 120
        factorial(0) should return 1
        factorial(3) should return 6
        
        The function should handle edge cases like negative numbers.
        """,
        assignee_id="bot_user_id",
        assignee_name="OpenAlpha Bot",
        team_id="team_id",
        team_name="AI Team",
        labels=["algorithm", "math"]
    )
    
    print(f"📝 Creating task for issue: {issue.title}")
    
    # Create task from issue
    task_id = await agent.workflow_automation.create_task_from_issue(issue)
    
    if task_id:
        print(f"✅ Task created successfully: {task_id}")
        
        # Get task status
        task_status = await agent.workflow_automation.get_task_status(issue.id)
        if task_status:
            print(f"📊 Task Status: {task_status}")
    else:
        print("❌ Failed to create task")


async def example_progress_monitoring(agent: LinearIntegrationAgent):
    """Example: Monitor progress and sync with Linear"""
    print("\\n📈 Progress monitoring example...")
    
    # Simulate progress updates
    progress_updates = [
        {
            "status": "started",
            "generation": 0,
            "best_fitness": 0.0,
            "details": {"tests_passed": 0, "tests_total": 3}
        },
        {
            "status": "evolving", 
            "generation": 5,
            "best_fitness": 0.3,
            "details": {"tests_passed": 1, "tests_total": 3}
        },
        {
            "status": "evolving",
            "generation": 15,
            "best_fitness": 0.7,
            "details": {"tests_passed": 2, "tests_total": 3}
        },
        {
            "status": "completed",
            "generation": 23,
            "best_fitness": 1.0,
            "details": {"tests_passed": 3, "tests_total": 3},
            "solution": {
                "function_name": "factorial",
                "lines_of_code": 8,
                "fitness_score": 1.0
            }
        }
    ]
    
    issue_id = "progress_example_issue"
    
    for i, progress in enumerate(progress_updates):
        print(f"📊 Progress Update {i+1}: {progress['status']} (Gen: {progress.get('generation', 0)})")
        
        # Sync progress to Linear
        success = await agent.workflow_automation.sync_progress(issue_id, progress)
        
        if success:
            print(f"✅ Progress synced to Linear")
        else:
            print(f"⚠️ Failed to sync progress")
        
        # Wait a bit between updates
        await asyncio.sleep(1)


async def example_assignment_detection():
    """Example: Assignment detection from webhook events"""
    print("\\n🎯 Assignment detection example...")
    
    # Create test webhook event
    event = WebhookEvent(
        id="assignment_test_event",
        type=LinearEventType.ISSUE_UPDATE,
        action="update",
        data={
            "id": "assignment_test_issue",
            "title": "Test Assignment Detection",
            "assignee": {
                "id": "bot_user_id",
                "name": "OpenAlpha Bot",
                "email": "bot@openalpha.com"
            },
            "updatedFrom": {
                "assignee": None  # Previously unassigned
            }
        }
    )
    
    # Create assignment detector with test config
    from .assignment.detector import AssignmentDetector
    
    detector_config = {
        "bot_user_id": "bot_user_id",
        "bot_email": "bot@openalpha.com",
        "bot_names": ["openalpha", "bot"]
    }
    
    detector = AssignmentDetector(detector_config)
    
    print(f"🔍 Detecting assignment change in event: {event.id}")
    
    # Detect assignment change
    assignment_event = await detector.detect_assignment_change(event)
    
    if assignment_event:
        print(f"✅ Assignment detected: {assignment_event.action.value} for issue {assignment_event.issue_id}")
        
        # Check if should process
        should_process = await detector.should_process_assignment(assignment_event)
        print(f"📋 Should process assignment: {should_process}")
    else:
        print("⚠️ No assignment change detected")


async def example_integration_monitoring(agent: LinearIntegrationAgent):
    """Example: Monitor integration health and statistics"""
    print("\\n🏥 Integration monitoring example...")
    
    # Get comprehensive status
    status = await agent.get_integration_status()
    
    print("📊 Integration Status:")
    print(f"  - Initialized: {status.get('initialized', False)}")
    print(f"  - Monitoring Active: {status.get('monitoring_active', False)}")
    print(f"  - Last Sync: {status.get('last_sync', 'Never')}")
    
    # Component statuses
    components = status.get('components', {})
    print("\\n🔧 Component Status:")
    for component, comp_status in components.items():
        print(f"  - {component}: {comp_status.get('status', 'unknown')}")
    
    # Statistics
    statistics = status.get('statistics', {})
    print("\\n📈 Statistics:")
    
    if 'webhooks' in statistics:
        webhook_stats = statistics['webhooks']
        print(f"  Webhooks:")
        print(f"    - Processed: {webhook_stats.get('processed_events', 0)}")
        print(f"    - Failed: {webhook_stats.get('failed_events', 0)}")
        print(f"    - Success Rate: {webhook_stats.get('success_rate', 0):.2%}")
    
    if 'assignments' in statistics:
        assignment_stats = statistics['assignments']
        print(f"  Assignments:")
        print(f"    - Total Tracked: {assignment_stats.get('total_tracked_assignments', 0)}")
        print(f"    - Last 24h: {assignment_stats.get('assignments_last_24h', 0)}")
    
    if 'workflows' in statistics:
        workflow_stats = statistics['workflows']
        print(f"  Workflows:")
        print(f"    - Active Tasks: {workflow_stats.get('total_active_tasks', 0)}")


async def main():
    """Main example function"""
    print("🎯 Linear Integration Examples")
    print("=" * 50)
    
    try:
        # Basic setup
        agent = await example_basic_setup()
        
        if agent:
            # Run examples
            await example_webhook_handling(agent)
            await example_manual_task_creation(agent)
            await example_progress_monitoring(agent)
            await example_assignment_detection()
            await example_integration_monitoring(agent)
            
            # Cleanup
            print("\\n🧹 Cleaning up...")
            await agent.cleanup()
            print("✅ Cleanup completed")
        
    except Exception as e:
        logger.error(f"Error in examples: {e}")
        print(f"❌ Example failed: {e}")
    
    print("\\n🏁 Examples completed!")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())


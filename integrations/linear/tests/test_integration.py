"""
Integration Tests for Linear API Integration System

Comprehensive tests to validate the complete Linear integration functionality.
"""

import asyncio
import unittest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import the Linear integration components
from ..agent import LinearIntegrationAgent
from ..interfaces import WebhookEvent, LinearEventType, AssignmentEvent, AssignmentAction
from config.settings import get_linear_config


class TestLinearIntegration(unittest.TestCase):
    """Test complete Linear integration system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "api_key": "test_api_key",
            "webhook_secret": "test_webhook_secret",
            "bot_user_id": "test_bot_user_id",
            "bot_email": "test@bot.com",
            "bot_names": ["testbot", "codegen"],
            "enabled": True,
            "auto_start_tasks": True,
            "auto_update_status": True,
            "monitoring_interval": 30,
            "timeout": 10,
            "max_retries": 2,
            "rate_limit_requests": 50,
            "rate_limit_window": 60,
            "cache_ttl": 300
        }
        
        self.agent = LinearIntegrationAgent(self.test_config)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.linear_client)
        self.assertIsNotNone(self.agent.webhook_processor)
        self.assertIsNotNone(self.agent.assignment_detector)
        self.assertIsNotNone(self.agent.workflow_automation)
        self.assertFalse(self.agent.initialized)
    
    @patch('integrations.linear.client.graphql_client.requests.Session.post')
    async def test_authentication(self, mock_post):
        """Test Linear API authentication"""
        # Mock successful authentication response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "viewer": {
                    "id": "test_user_id",
                    "name": "Test Bot",
                    "email": "test@bot.com"
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Test authentication
        success = await self.agent.initialize()
        self.assertTrue(success)
        self.assertTrue(self.agent.initialized)
    
    async def test_webhook_processing(self):
        """Test webhook event processing"""
        # Create test webhook payload
        webhook_payload = {
            "action": "create",
            "type": "Issue",
            "data": {
                "id": "test_issue_id",
                "title": "Test Issue",
                "description": "Test issue description",
                "assignee": {
                    "id": "test_bot_user_id",
                    "name": "Test Bot",
                    "email": "test@bot.com"
                },
                "team": {
                    "id": "test_team_id",
                    "name": "Test Team"
                }
            }
        }
        
        # Mock webhook signature
        test_signature = "sha256=test_signature"
        
        # Process webhook (will fail signature validation but test structure)
        result = await self.agent.handle_webhook(webhook_payload, test_signature)
        
        # Should fail due to signature validation, but structure should be intact
        self.assertFalse(result)  # Expected to fail signature validation
    
    async def test_assignment_detection(self):
        """Test assignment detection from webhook events"""
        # Create test webhook event
        event = WebhookEvent(
            id="test_event_id",
            type=LinearEventType.ISSUE_CREATE,
            action="create",
            data={
                "id": "test_issue_id",
                "title": "Test Issue",
                "assignee": {
                    "id": "test_bot_user_id",
                    "name": "testbot",
                    "email": "test@bot.com"
                }
            },
            timestamp=datetime.now()
        )
        
        # Test assignment detection
        assignment_event = await self.agent.assignment_detector.detect_assignment_change(event)
        
        self.assertIsNotNone(assignment_event)
        self.assertEqual(assignment_event.action, AssignmentAction.ASSIGNED)
        self.assertEqual(assignment_event.issue_id, "test_issue_id")
    
    async def test_task_creation(self):
        """Test task creation from Linear issue"""
        from ..interfaces import LinearIssue
        
        # Create test issue
        test_issue = LinearIssue(
            id="test_issue_id",
            title="Implement fibonacci function",
            description="Create a function that calculates fibonacci numbers\\n\\nExample: fibonacci(5) should return 5",
            assignee_id="test_bot_user_id",
            assignee_name="Test Bot",
            team_id="test_team_id",
            team_name="Test Team"
        )
        
        # Test task creation
        task_id = await self.agent.workflow_automation.task_creator.create_task_from_issue(test_issue)
        
        self.assertIsNotNone(task_id)
        self.assertTrue(task_id.startswith("linear_"))
    
    async def test_workflow_automation(self):
        """Test complete workflow automation"""
        # Create test assignment event
        assignment_event = AssignmentEvent(
            issue_id="test_issue_id",
            action=AssignmentAction.ASSIGNED,
            assignee_id="test_bot_user_id",
            timestamp=datetime.now()
        )
        
        # Mock Linear client for workflow automation
        mock_client = AsyncMock()
        mock_client.get_issue.return_value = Mock(
            id="test_issue_id",
            title="Test Issue",
            description="Test description",
            assignee_id="test_bot_user_id"
        )
        mock_client.create_comment.return_value = "test_comment_id"
        
        self.agent.workflow_automation.set_linear_client(mock_client)
        
        # Test assignment handling
        result = await self.agent.workflow_automation.handle_assignment(assignment_event)
        
        # Should succeed with mocked client
        self.assertTrue(result)
    
    async def test_integration_status(self):
        """Test integration status reporting"""
        status = await self.agent.get_integration_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("initialized", status)
        self.assertIn("monitoring_active", status)
        self.assertIn("components", status)
        self.assertIn("statistics", status)
    
    def test_configuration_loading(self):
        """Test configuration loading from settings"""
        config = get_linear_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn("api_key", config)
        self.assertIn("webhook_secret", config)
        self.assertIn("bot_user_id", config)
        self.assertIn("enabled", config)


class TestLinearIntegrationAsync(unittest.IsolatedAsyncioTestCase):
    """Async test cases for Linear integration"""
    
    async def asyncSetUp(self):
        """Set up async test environment"""
        self.test_config = {
            "api_key": "test_api_key",
            "webhook_secret": "test_webhook_secret",
            "bot_user_id": "test_bot_user_id",
            "bot_email": "test@bot.com",
            "enabled": True
        }
        
        self.agent = LinearIntegrationAgent(self.test_config)
    
    async def test_event_queue_processing(self):
        """Test event queue processing"""
        # Create test events
        events = []
        for i in range(5):
            event = WebhookEvent(
                id=f"test_event_{i}",
                type=LinearEventType.ISSUE_UPDATE,
                action="update",
                data={"id": f"issue_{i}", "title": f"Issue {i}"},
                timestamp=datetime.now()
            )
            events.append(event)
        
        # Queue events
        for event in events:
            await self.agent.event_manager.queue_event(event)
        
        # Process queue
        processed_count = await self.agent.event_manager.process_queue()
        
        # Should process all events
        self.assertEqual(processed_count, len(events))
    
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Test rate limiter
        rate_limiter = self.agent.linear_client.rate_limiter
        
        # Should allow initial requests
        await rate_limiter.acquire()
        await rate_limiter.acquire()
        
        # Test that rate limiter tracks requests
        self.assertGreater(len(rate_limiter.requests), 0)
    
    async def test_caching(self):
        """Test response caching"""
        cache = self.agent.linear_client.cache
        
        # Test cache operations
        test_query = "query { viewer { id } }"
        test_response = {"data": {"viewer": {"id": "test_id"}}}
        
        # Cache response
        await cache.set(test_query, None, test_response)
        
        # Retrieve cached response
        cached_response = await cache.get(test_query, None)
        
        self.assertEqual(cached_response, test_response)
    
    async def test_error_handling(self):
        """Test error handling and recovery"""
        # Test webhook processing with invalid payload
        invalid_payload = {"invalid": "payload"}
        
        result = await self.agent.handle_webhook(invalid_payload, "invalid_signature")
        
        # Should handle error gracefully
        self.assertFalse(result)
    
    async def test_cleanup_operations(self):
        """Test cleanup operations"""
        # Add some test data
        test_event = WebhookEvent(
            id="test_cleanup_event",
            type=LinearEventType.ISSUE_CREATE,
            action="create",
            data={"id": "test_issue"},
            timestamp=datetime.now()
        )
        
        await self.agent.event_manager.queue_event(test_event)
        
        # Test cleanup
        await self.agent.cleanup()
        
        # Verify cleanup completed without errors
        self.assertTrue(True)  # If we get here, cleanup succeeded


def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestLinearIntegration))
    suite.addTest(unittest.makeSuite(TestLinearIntegrationAsync))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests
    success = run_integration_tests()
    
    if success:
        print("\\n✅ All Linear integration tests passed!")
    else:
        print("\\n❌ Some Linear integration tests failed!")
        exit(1)


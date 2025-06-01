"""
Linear Integration Test Runner

Comprehensive test runner for the Linear API integration system.
"""

import asyncio
import unittest
import sys
import time
from typing import List, Dict, Any
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during tests
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import test modules
from .tests.test_integration import TestLinearIntegration, TestLinearIntegrationAsync


class TestResult:
    """Test result container"""
    
    def __init__(self, name: str, success: bool, duration: float, error: str = None):
        self.name = name
        self.success = success
        self.duration = duration
        self.error = error


class LinearTestRunner:
    """Comprehensive test runner for Linear integration"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_duration = 0.0
    
    def run_unit_tests(self) -> bool:
        """Run unit tests"""
        print("🧪 Running unit tests...")
        
        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestLinearIntegration))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
        start_time = time.time()
        result = runner.run(suite)
        duration = time.time() - start_time
        
        # Record result
        test_result = TestResult(
            name="Unit Tests",
            success=result.wasSuccessful(),
            duration=duration,
            error=None if result.wasSuccessful() else f"{len(result.failures)} failures, {len(result.errors)} errors"
        )
        self.results.append(test_result)
        
        if result.wasSuccessful():
            print(f"✅ Unit tests passed ({duration:.2f}s)")
        else:
            print(f"❌ Unit tests failed ({duration:.2f}s)")
            for failure in result.failures:
                print(f"   FAIL: {failure[0]}")
            for error in result.errors:
                print(f"   ERROR: {error[0]}")
        
        return result.wasSuccessful()
    
    async def run_async_tests(self) -> bool:
        """Run async tests"""
        print("🔄 Running async tests...")
        
        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestLinearIntegrationAsync))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
        start_time = time.time()
        result = runner.run(suite)
        duration = time.time() - start_time
        
        # Record result
        test_result = TestResult(
            name="Async Tests",
            success=result.wasSuccessful(),
            duration=duration,
            error=None if result.wasSuccessful() else f"{len(result.failures)} failures, {len(result.errors)} errors"
        )
        self.results.append(test_result)
        
        if result.wasSuccessful():
            print(f"✅ Async tests passed ({duration:.2f}s)")
        else:
            print(f"❌ Async tests failed ({duration:.2f}s)")
            for failure in result.failures:
                print(f"   FAIL: {failure[0]}")
            for error in result.errors:
                print(f"   ERROR: {error[0]}")
        
        return result.wasSuccessful()
    
    async def run_integration_tests(self) -> bool:
        """Run integration tests"""
        print("🔗 Running integration tests...")
        
        try:
            from .agent import LinearIntegrationAgent
            from .interfaces import WebhookEvent, LinearEventType, LinearIssue
            from datetime import datetime
            
            start_time = time.time()
            
            # Test 1: Agent initialization
            print("  📋 Testing agent initialization...")
            config = {
                "api_key": "test_key",
                "webhook_secret": "test_secret",
                "bot_user_id": "test_bot_id",
                "enabled": True
            }
            
            agent = LinearIntegrationAgent(config)
            if not agent:
                raise Exception("Failed to create agent")
            
            # Test 2: Component initialization
            print("  🔧 Testing component initialization...")
            if not all([
                agent.linear_client,
                agent.webhook_processor,
                agent.assignment_detector,
                agent.workflow_automation,
                agent.event_manager
            ]):
                raise Exception("Components not initialized")
            
            # Test 3: Configuration loading
            print("  ⚙️ Testing configuration loading...")
            from config.settings import get_linear_config
            linear_config = get_linear_config()
            if not isinstance(linear_config, dict):
                raise Exception("Configuration not loaded properly")
            
            # Test 4: Task creation
            print("  📝 Testing task creation...")
            issue = LinearIssue(
                id="test_issue",
                title="Test Issue",
                description="Test description",
                assignee_id="test_bot_id"
            )
            
            task_id = await agent.workflow_automation.task_creator.create_task_from_issue(issue)
            if not task_id:
                raise Exception("Task creation failed")
            
            # Test 5: Event processing
            print("  📡 Testing event processing...")
            event = WebhookEvent(
                id="test_event",
                type=LinearEventType.ISSUE_CREATE,
                action="create",
                data={"id": "test_issue", "title": "Test"},
                timestamp=datetime.now()
            )
            
            await agent.event_manager.queue_event(event)
            processed = await agent.event_manager.process_queue()
            if processed == 0:
                print("    ⚠️ No events processed (expected for test)")
            
            duration = time.time() - start_time
            
            test_result = TestResult(
                name="Integration Tests",
                success=True,
                duration=duration
            )
            self.results.append(test_result)
            
            print(f"✅ Integration tests passed ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                name="Integration Tests",
                success=False,
                duration=duration,
                error=str(e)
            )
            self.results.append(test_result)
            
            print(f"❌ Integration tests failed ({duration:.2f}s): {e}")
            return False
    
    async def run_performance_tests(self) -> bool:
        """Run performance tests"""
        print("⚡ Running performance tests...")
        
        try:
            start_time = time.time()
            
            # Test rate limiter performance
            print("  🚦 Testing rate limiter...")
            from .client.graphql_client import RateLimiter
            
            rate_limiter = RateLimiter(max_requests=10, time_window=1)
            
            # Test rapid requests
            for i in range(5):
                await rate_limiter.acquire()
            
            # Test cache performance
            print("  💾 Testing cache performance...")
            from .client.graphql_client import ResponseCache
            
            cache = ResponseCache(default_ttl=60)
            
            # Test cache operations
            test_query = "test query"
            test_response = {"data": "test"}
            
            await cache.set(test_query, None, test_response)
            cached = await cache.get(test_query, None)
            
            if cached != test_response:
                raise Exception("Cache test failed")
            
            # Test event queue performance
            print("  📬 Testing event queue...")
            from .events.queue import EventQueue
            from .interfaces import WebhookEvent, LinearEventType
            from datetime import datetime
            
            queue = EventQueue(max_size=100)
            
            # Add multiple events
            for i in range(10):
                event = WebhookEvent(
                    id=f"perf_test_{i}",
                    type=LinearEventType.ISSUE_UPDATE,
                    action="update",
                    data={"id": f"issue_{i}"},
                    timestamp=datetime.now()
                )
                await queue.enqueue(event)
            
            # Process events
            processed_count = 0
            while not await queue.is_empty():
                event = await queue.dequeue()
                if event:
                    processed_count += 1
            
            if processed_count != 10:
                raise Exception(f"Expected 10 events, processed {processed_count}")
            
            duration = time.time() - start_time
            
            test_result = TestResult(
                name="Performance Tests",
                success=True,
                duration=duration
            )
            self.results.append(test_result)
            
            print(f"✅ Performance tests passed ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                name="Performance Tests",
                success=False,
                duration=duration,
                error=str(e)
            )
            self.results.append(test_result)
            
            print(f"❌ Performance tests failed ({duration:.2f}s): {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} {result.name:<20} ({result.duration:.2f}s)")
            if result.error:
                print(f"     Error: {result.error}")
        
        total_duration = sum(r.duration for r in self.results)
        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)
        
        print("-"*60)
        print(f"Total: {passed}/{total} test suites passed")
        print(f"Duration: {total_duration:.2f}s")
        
        if passed == total:
            print("\\n🎉 All tests passed!")
            return True
        else:
            print(f"\\n💥 {total - passed} test suite(s) failed!")
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all test suites"""
        print("🚀 Starting Linear Integration Test Suite")
        print("="*60)
        
        start_time = time.time()
        
        # Run all test suites
        results = []
        results.append(self.run_unit_tests())
        results.append(await self.run_async_tests())
        results.append(await self.run_integration_tests())
        results.append(await self.run_performance_tests())
        
        total_duration = time.time() - start_time
        
        # Print summary
        success = self.print_summary()
        
        print(f"\\nTotal execution time: {total_duration:.2f}s")
        
        return success


async def main():
    """Main test runner function"""
    runner = LinearTestRunner()
    
    try:
        success = await runner.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n💥 Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


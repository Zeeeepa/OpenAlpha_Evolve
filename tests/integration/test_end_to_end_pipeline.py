"""
End-to-end integration tests for the autonomous development pipeline.

These tests validate the complete workflow from task definition
through code generation, evaluation, and evolution.
"""

import pytest
import asyncio
import os
import sys
import tempfile
import json
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.interfaces import TaskDefinition, Program
from src.openevolve.orchestrator.integration_manager import IntegrationManager, PipelineStatus
from src.openevolve.orchestrator.workflow_coordinator import WorkflowCoordinator
from src.openevolve.orchestrator.health_monitor import HealthMonitor
from src.openevolve.orchestrator.performance_optimizer import PerformanceOptimizer


class TestEndToEndPipeline:
    """Test the complete end-to-end pipeline functionality."""
    
    @pytest.fixture
    async def integration_manager(self):
        """Create an integration manager for testing."""
        manager = IntegrationManager()
        yield manager
        # Cleanup
        if hasattr(manager, '_shutdown_event'):
            manager._shutdown_event.set()
    
    @pytest.fixture
    def sample_task_definition(self):
        """Create a sample task definition for testing."""
        return TaskDefinition(
            id="test_task_001",
            description="Write a function that calculates the factorial of a number",
            function_name_to_evolve="factorial",
            input_output_examples=[
                {"input": {"n": 5}, "output": 120},
                {"input": {"n": 0}, "output": 1},
                {"input": {"n": 1}, "output": 1},
                {"input": {"n": 3}, "output": 6}
            ],
            evaluation_criteria={
                "correctness": 0.7,
                "efficiency": 0.2,
                "readability": 0.1
            },
            tests=[
                {
                    "name": "basic_tests",
                    "test_cases": [
                        {"input": {"n": 5}, "expected": 120},
                        {"input": {"n": 0}, "expected": 1}
                    ]
                }
            ]
        )
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, integration_manager, sample_task_definition):
        """Test that the pipeline initializes correctly."""
        # Test initial state
        assert integration_manager.state.status == PipelineStatus.IDLE
        assert integration_manager.state.task_definition is None
        assert len(integration_manager.agents) == 0
        
        # Test component initialization
        integration_manager.state.task_definition = sample_task_definition
        await integration_manager._initialize_components()
        
        # Verify all required agents are initialized
        required_agents = [
            'task_manager', 'prompt_designer', 'code_generator',
            'evaluator', 'database', 'selection_controller'
        ]
        
        for agent_name in required_agents:
            assert agent_name in integration_manager.agents
            assert integration_manager.agents[agent_name] is not None
    
    @pytest.mark.asyncio
    async def test_health_validation(self, integration_manager, sample_task_definition):
        """Test system health validation."""
        integration_manager.state.task_definition = sample_task_definition
        await integration_manager._initialize_components()
        
        # Test health validation
        await integration_manager._validate_system_health()
        
        # Verify health checks are registered
        assert len(integration_manager.health_checks) > 0
        
        # Test individual health checks
        for name, health_check in integration_manager.health_checks.items():
            result = await health_check()
            assert result is True or result is False  # Should return boolean
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_execution(self, integration_manager, sample_task_definition):
        """Test complete pipeline execution with mocked components."""
        
        # Mock the task manager to avoid actual LLM calls
        with patch('task_manager.agent.TaskManagerAgent') as mock_task_manager_class:
            mock_task_manager = Mock()
            mock_task_manager.manage_evolutionary_cycle = Mock(return_value=asyncio.coroutine(lambda: None)())
            mock_task_manager_class.return_value = mock_task_manager
            
            # Mock other components
            with patch('prompt_designer.agent.PromptDesignerAgent'), \
                 patch('code_generator.agent.CodeGeneratorAgent'), \
                 patch('evaluator_agent.agent.EvaluatorAgent'), \
                 patch('database_agent.agent.InMemoryDatabaseAgent') as mock_db_class, \
                 patch('selection_controller.agent.SelectionControllerAgent'):
                
                # Setup mock database
                mock_db = Mock()
                mock_db.get_best_programs = Mock(return_value=asyncio.coroutine(lambda task_id, limit=1000: [])())
                mock_db_class.return_value = mock_db
                
                # Execute pipeline
                result = await integration_manager.execute(sample_task_definition)
                
                # Verify execution completed
                assert result.status == PipelineStatus.COMPLETED
                assert result.task_definition == sample_task_definition
                assert result.metrics.end_time is not None
                
                # Verify task manager was called
                mock_task_manager.manage_evolutionary_cycle.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, integration_manager, sample_task_definition):
        """Test pipeline error handling and recovery."""
        
        # Mock components to raise an error
        with patch('task_manager.agent.TaskManagerAgent') as mock_task_manager_class:
            mock_task_manager = Mock()
            mock_task_manager.manage_evolutionary_cycle = Mock(
                side_effect=Exception("Simulated pipeline error")
            )
            mock_task_manager_class.return_value = mock_task_manager
            
            with patch('prompt_designer.agent.PromptDesignerAgent'), \
                 patch('code_generator.agent.CodeGeneratorAgent'), \
                 patch('evaluator_agent.agent.EvaluatorAgent'), \
                 patch('database_agent.agent.InMemoryDatabaseAgent'), \
                 patch('selection_controller.agent.SelectionControllerAgent'):
                
                # Execute pipeline and expect failure
                with pytest.raises(Exception, match="Simulated pipeline error"):
                    await integration_manager.execute(sample_task_definition)
                
                # Verify error state
                assert integration_manager.state.status == PipelineStatus.FAILED
                assert len(integration_manager.state.metrics.errors) > 0
                assert "Simulated pipeline error" in integration_manager.state.metrics.errors[0]
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics_collection(self, integration_manager, sample_task_definition):
        """Test metrics collection during pipeline execution."""
        
        # Create mock programs for metrics testing
        mock_programs = [
            Program(
                id="prog_1",
                code="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                fitness_scores={"correctness": 0.9, "efficiency": 0.7},
                generation=1,
                status="evaluated"
            ),
            Program(
                id="prog_2",
                code="def factorial(n): return 1",
                fitness_scores={"correctness": 0.3, "efficiency": 0.9},
                generation=1,
                status="evaluated",
                errors=["Incorrect result for n > 1"]
            )
        ]
        
        with patch('task_manager.agent.TaskManagerAgent') as mock_task_manager_class:
            mock_task_manager = Mock()
            mock_task_manager.manage_evolutionary_cycle = Mock(return_value=asyncio.coroutine(lambda: None)())
            mock_task_manager_class.return_value = mock_task_manager
            
            with patch('prompt_designer.agent.PromptDesignerAgent'), \
                 patch('code_generator.agent.CodeGeneratorAgent'), \
                 patch('evaluator_agent.agent.EvaluatorAgent'), \
                 patch('database_agent.agent.InMemoryDatabaseAgent') as mock_db_class, \
                 patch('selection_controller.agent.SelectionControllerAgent'):
                
                # Setup mock database with programs
                mock_db = Mock()
                mock_db.get_best_programs = Mock(return_value=asyncio.coroutine(lambda task_id, limit=1000: mock_programs)())
                mock_db_class.return_value = mock_db
                
                # Execute pipeline
                result = await integration_manager.execute(sample_task_definition)
                
                # Verify metrics were collected
                metrics = result.metrics
                assert metrics.total_programs_generated == 2
                assert metrics.total_programs_evaluated == 2
                assert metrics.successful_evaluations == 1
                assert metrics.failed_evaluations == 1
                assert metrics.best_fitness_score == 0.9
                assert metrics.success_rate == 0.5
    
    @pytest.mark.asyncio
    async def test_event_handling(self, integration_manager, sample_task_definition):
        """Test event emission and handling."""
        events_received = []
        
        async def event_handler(data):
            events_received.append(data)
        
        # Register event handlers
        integration_manager.register_event_handler("pipeline_completed", event_handler)
        integration_manager.register_event_handler("health_check_completed", event_handler)
        
        with patch('task_manager.agent.TaskManagerAgent') as mock_task_manager_class:
            mock_task_manager = Mock()
            mock_task_manager.manage_evolutionary_cycle = Mock(return_value=asyncio.coroutine(lambda: None)())
            mock_task_manager_class.return_value = mock_task_manager
            
            with patch('prompt_designer.agent.PromptDesignerAgent'), \
                 patch('code_generator.agent.CodeGeneratorAgent'), \
                 patch('evaluator_agent.agent.EvaluatorAgent'), \
                 patch('database_agent.agent.InMemoryDatabaseAgent') as mock_db_class, \
                 patch('selection_controller.agent.SelectionControllerAgent'):
                
                mock_db = Mock()
                mock_db.get_best_programs = Mock(return_value=asyncio.coroutine(lambda task_id, limit=1000: [])())
                mock_db_class.return_value = mock_db
                
                # Execute pipeline
                await integration_manager.execute(sample_task_definition)
                
                # Verify events were emitted
                assert len(events_received) >= 2  # At least health_check and pipeline_completed
    
    @pytest.mark.asyncio
    async def test_pipeline_status_tracking(self, integration_manager, sample_task_definition):
        """Test pipeline status tracking throughout execution."""
        
        status_changes = []
        
        async def status_tracker():
            """Track status changes during execution."""
            while integration_manager.state.status != PipelineStatus.COMPLETED:
                status_changes.append(integration_manager.state.status)
                await asyncio.sleep(0.1)
        
        with patch('task_manager.agent.TaskManagerAgent') as mock_task_manager_class:
            mock_task_manager = Mock()
            
            async def slow_execution():
                await asyncio.sleep(0.5)  # Simulate some work
            
            mock_task_manager.manage_evolutionary_cycle = Mock(return_value=slow_execution())
            mock_task_manager_class.return_value = mock_task_manager
            
            with patch('prompt_designer.agent.PromptDesignerAgent'), \
                 patch('code_generator.agent.CodeGeneratorAgent'), \
                 patch('evaluator_agent.agent.EvaluatorAgent'), \
                 patch('database_agent.agent.InMemoryDatabaseAgent') as mock_db_class, \
                 patch('selection_controller.agent.SelectionControllerAgent'):
                
                mock_db = Mock()
                mock_db.get_best_programs = Mock(return_value=asyncio.coroutine(lambda task_id, limit=1000: [])())
                mock_db_class.return_value = mock_db
                
                # Start status tracking
                tracker_task = asyncio.create_task(status_tracker())
                
                # Execute pipeline
                result = await integration_manager.execute(sample_task_definition)
                
                # Stop tracking
                tracker_task.cancel()
                
                # Verify status progression
                assert PipelineStatus.IDLE in status_changes
                assert PipelineStatus.INITIALIZING in status_changes
                assert PipelineStatus.RUNNING in status_changes
                assert result.status == PipelineStatus.COMPLETED


class TestWorkflowCoordination:
    """Test workflow coordination functionality."""
    
    @pytest.fixture
    def workflow_coordinator(self):
        """Create a workflow coordinator for testing."""
        return WorkflowCoordinator()
    
    @pytest.mark.asyncio
    async def test_workflow_template_registration(self, workflow_coordinator):
        """Test workflow template registration and listing."""
        # Check built-in templates
        templates = workflow_coordinator.list_templates()
        assert len(templates) >= 3  # Should have built-in templates
        
        template_ids = [t["id"] for t in templates]
        assert "evolutionary_cycle" in template_ids
        assert "system_health_check" in template_ids
        assert "performance_benchmark" in template_ids
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, workflow_coordinator):
        """Test agent registration for workflow execution."""
        mock_agent = Mock()
        mock_agent.health_check = Mock(return_value=asyncio.coroutine(lambda: True)())
        
        workflow_coordinator.register_agent("test_agent", mock_agent)
        
        assert "test_agent" in workflow_coordinator.agents
        assert workflow_coordinator.agents["test_agent"] == mock_agent
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, workflow_coordinator):
        """Test workflow execution with mocked agents."""
        # Register mock agents
        mock_agents = {}
        for agent_type in ["task_manager", "code_generator", "evaluator", "database"]:
            mock_agent = Mock()
            mock_agent.health_check = Mock(return_value=asyncio.coroutine(lambda: True)())
            mock_agents[agent_type] = mock_agent
            workflow_coordinator.register_agent(agent_type, mock_agent)
        
        # Execute health check workflow
        execution = await workflow_coordinator.execute_workflow("system_health_check")
        
        # Verify execution completed
        assert execution.status.value == "completed"
        assert len(execution.steps) == 4  # Should have 4 health check steps
        
        # Verify all agents were called
        for agent in mock_agents.values():
            agent.health_check.assert_called_once()


class TestSystemIntegration:
    """Test integration between different system components."""
    
    @pytest.mark.asyncio
    async def test_health_monitor_integration(self):
        """Test health monitor integration with other components."""
        health_monitor = HealthMonitor(check_interval=1.0)
        
        # Create mock components
        mock_component = Mock()
        mock_component.health_check = Mock(return_value=asyncio.coroutine(lambda: True)())
        
        # Register component checker
        checker = await health_monitor.create_component_health_checker(mock_component, "test_component")
        health_monitor.register_component_checker("test_component", checker)
        
        # Perform health check
        health_status = await health_monitor.check_system_health()
        
        # Verify health check results
        assert health_status.overall_status.value in ["healthy", "warning", "critical"]
        assert "test_component" in health_status.components
        assert health_status.components["test_component"].status.value in ["healthy", "warning", "critical"]
    
    @pytest.mark.asyncio
    async def test_performance_optimizer_integration(self):
        """Test performance optimizer integration."""
        optimizer = PerformanceOptimizer()
        
        # Create mock component
        mock_component = Mock()
        mock_component.execute = Mock(return_value=asyncio.coroutine(lambda: "result")())
        
        # Benchmark component
        benchmarks = await optimizer.benchmark_component(
            mock_component, "test_component", ["execute"], iterations=3
        )
        
        # Verify benchmark results
        assert len(benchmarks) == 1
        benchmark = benchmarks[0]
        assert benchmark.component == "test_component"
        assert benchmark.operation == "execute"
        assert benchmark.iterations == 3
        assert benchmark.success_rate > 0
        
        # Verify component was called
        assert mock_component.execute.call_count == 3
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full system integration with all components."""
        # Create integration manager
        integration_manager = IntegrationManager()
        
        # Create workflow coordinator
        workflow_coordinator = WorkflowCoordinator()
        
        # Create health monitor
        health_monitor = HealthMonitor(check_interval=5.0)
        
        # Create performance optimizer
        performance_optimizer = PerformanceOptimizer()
        
        # Test that all components can be initialized together
        assert integration_manager is not None
        assert workflow_coordinator is not None
        assert health_monitor is not None
        assert performance_optimizer is not None
        
        # Test basic functionality
        status = integration_manager.get_status()
        assert "status" in status
        assert "id" in status
        
        templates = workflow_coordinator.list_templates()
        assert len(templates) > 0
        
        health_summary = health_monitor.get_health_summary()
        assert "overall_status" in health_summary
        
        perf_summary = performance_optimizer.get_performance_summary()
        assert "overall_status" in perf_summary


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])


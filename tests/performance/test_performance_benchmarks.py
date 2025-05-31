"""
Performance benchmark tests for the autonomous development pipeline.

These tests measure system performance, identify bottlenecks, and ensure
the system meets performance requirements under various load conditions.
"""

import pytest
import asyncio
import time
import statistics
import os
import sys
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.interfaces import TaskDefinition, Program
from src.openevolve.orchestrator.integration_manager import IntegrationManager
from src.openevolve.orchestrator.performance_optimizer import PerformanceOptimizer, PerformanceBenchmark


class TestPerformanceBenchmarks:
    """Performance benchmark tests for system components."""
    
    @pytest.fixture
    def performance_optimizer(self):
        """Create a performance optimizer for testing."""
        return PerformanceOptimizer()
    
    @pytest.fixture
    def sample_task_definition(self):
        """Create a sample task definition for performance testing."""
        return TaskDefinition(
            id="perf_test_task",
            description="Performance test task for benchmarking",
            function_name_to_evolve="test_function",
            input_output_examples=[
                {"input": {"x": i}, "output": i * 2} for i in range(10)
            ]
        )
    
    @pytest.mark.asyncio
    async def test_integration_manager_performance(self, sample_task_definition):
        """Test integration manager performance under normal load."""
        integration_manager = IntegrationManager()
        
        # Mock components to avoid actual LLM calls
        with patch('task_manager.agent.TaskManagerAgent') as mock_task_manager_class:
            mock_task_manager = Mock()
            
            # Simulate realistic execution time
            async def mock_execution():
                await asyncio.sleep(0.1)  # 100ms simulation
            
            mock_task_manager.manage_evolutionary_cycle = Mock(return_value=mock_execution())
            mock_task_manager_class.return_value = mock_task_manager
            
            with patch('prompt_designer.agent.PromptDesignerAgent'), \
                 patch('code_generator.agent.CodeGeneratorAgent'), \
                 patch('evaluator_agent.agent.EvaluatorAgent'), \
                 patch('database_agent.agent.InMemoryDatabaseAgent') as mock_db_class, \
                 patch('selection_controller.agent.SelectionControllerAgent'):
                
                mock_db = Mock()
                mock_db.get_best_programs = Mock(return_value=asyncio.coroutine(lambda task_id, limit=1000: [])())
                mock_db_class.return_value = mock_db
                
                # Measure execution time
                start_time = time.time()
                result = await integration_manager.execute(sample_task_definition)
                execution_time = time.time() - start_time
                
                # Performance assertions
                assert execution_time < 5.0  # Should complete within 5 seconds
                assert result.metrics.duration < 5.0
                
                # Verify successful completion
                assert result.status.value == "completed"
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_execution(self, sample_task_definition):
        """Test performance with multiple concurrent pipeline executions."""
        num_concurrent = 3
        integration_managers = [IntegrationManager() for _ in range(num_concurrent)]
        
        # Mock components
        with patch('task_manager.agent.TaskManagerAgent') as mock_task_manager_class:
            mock_task_manager = Mock()
            
            async def mock_execution():
                await asyncio.sleep(0.2)  # 200ms simulation
            
            mock_task_manager.manage_evolutionary_cycle = Mock(return_value=mock_execution())
            mock_task_manager_class.return_value = mock_task_manager
            
            with patch('prompt_designer.agent.PromptDesignerAgent'), \
                 patch('code_generator.agent.CodeGeneratorAgent'), \
                 patch('evaluator_agent.agent.EvaluatorAgent'), \
                 patch('database_agent.agent.InMemoryDatabaseAgent') as mock_db_class, \
                 patch('selection_controller.agent.SelectionControllerAgent'):
                
                mock_db = Mock()
                mock_db.get_best_programs = Mock(return_value=asyncio.coroutine(lambda task_id, limit=1000: [])())
                mock_db_class.return_value = mock_db
                
                # Execute pipelines concurrently
                start_time = time.time()
                tasks = [
                    manager.execute(sample_task_definition) 
                    for manager in integration_managers
                ]
                results = await asyncio.gather(*tasks)
                total_time = time.time() - start_time
                
                # Performance assertions
                assert total_time < 10.0  # Should complete within 10 seconds
                assert len(results) == num_concurrent
                
                # Verify all completed successfully
                for result in results:
                    assert result.status.value == "completed"
                
                # Check that concurrent execution is more efficient than sequential
                expected_sequential_time = num_concurrent * 0.2
                efficiency_ratio = expected_sequential_time / total_time
                assert efficiency_ratio > 1.5  # Should be at least 50% more efficient
    
    @pytest.mark.asyncio
    async def test_component_benchmarking(self, performance_optimizer):
        """Test component performance benchmarking."""
        # Create mock component
        mock_component = Mock()
        
        # Simulate different performance characteristics
        async def fast_operation():
            await asyncio.sleep(0.01)  # 10ms
            return "fast_result"
        
        async def slow_operation():
            await asyncio.sleep(0.1)   # 100ms
            return "slow_result"
        
        mock_component.fast_op = fast_operation
        mock_component.slow_op = slow_operation
        
        # Benchmark both operations
        benchmarks = await performance_optimizer.benchmark_component(
            mock_component, 
            "test_component", 
            ["fast_op", "slow_op"], 
            iterations=5
        )
        
        # Verify benchmark results
        assert len(benchmarks) == 2
        
        fast_benchmark = next(b for b in benchmarks if b.operation == "fast_op")
        slow_benchmark = next(b for b in benchmarks if b.operation == "slow_op")
        
        # Performance assertions
        assert fast_benchmark.duration < slow_benchmark.duration
        assert fast_benchmark.success_rate == 1.0
        assert slow_benchmark.success_rate == 1.0
        assert fast_benchmark.throughput > slow_benchmark.throughput
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, performance_optimizer):
        """Test memory usage monitoring during operations."""
        # Start monitoring
        await performance_optimizer.start_monitoring(interval=1.0)
        
        # Wait for some metrics to be collected
        await asyncio.sleep(2.0)
        
        # Stop monitoring
        await performance_optimizer.stop_monitoring()
        
        # Check collected metrics
        summary = performance_optimizer.get_performance_summary()
        
        assert "global_metrics" in summary
        assert len(performance_optimizer.global_metrics) > 0
        
        # Check for system metrics
        metric_names = [m.name for m in performance_optimizer.global_metrics]
        assert "system_memory_usage" in metric_names
        assert "system_cpu_usage" in metric_names
    
    @pytest.mark.asyncio
    async def test_load_testing(self, sample_task_definition):
        """Test system performance under high load."""
        num_requests = 10
        integration_manager = IntegrationManager()
        
        # Mock components for fast execution
        with patch('task_manager.agent.TaskManagerAgent') as mock_task_manager_class:
            mock_task_manager = Mock()
            
            async def fast_execution():
                await asyncio.sleep(0.05)  # 50ms simulation
            
            mock_task_manager.manage_evolutionary_cycle = Mock(return_value=fast_execution())
            mock_task_manager_class.return_value = mock_task_manager
            
            with patch('prompt_designer.agent.PromptDesignerAgent'), \
                 patch('code_generator.agent.CodeGeneratorAgent'), \
                 patch('evaluator_agent.agent.EvaluatorAgent'), \
                 patch('database_agent.agent.InMemoryDatabaseAgent') as mock_db_class, \
                 patch('selection_controller.agent.SelectionControllerAgent'):
                
                mock_db = Mock()
                mock_db.get_best_programs = Mock(return_value=asyncio.coroutine(lambda task_id, limit=1000: [])())
                mock_db_class.return_value = mock_db
                
                # Execute multiple requests sequentially
                execution_times = []
                
                for i in range(num_requests):
                    start_time = time.time()
                    result = await integration_manager.execute(sample_task_definition)
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                    
                    assert result.status.value == "completed"
                
                # Analyze performance
                avg_time = statistics.mean(execution_times)
                max_time = max(execution_times)
                min_time = min(execution_times)
                std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                
                # Performance assertions
                assert avg_time < 2.0  # Average should be under 2 seconds
                assert max_time < 5.0  # No request should take more than 5 seconds
                assert std_dev < 1.0   # Standard deviation should be reasonable
                
                # Check for performance degradation
                first_half = execution_times[:num_requests//2]
                second_half = execution_times[num_requests//2:]
                
                if len(first_half) > 0 and len(second_half) > 0:
                    first_avg = statistics.mean(first_half)
                    second_avg = statistics.mean(second_half)
                    degradation = (second_avg - first_avg) / first_avg
                    
                    # Should not degrade by more than 50%
                    assert degradation < 0.5
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, sample_task_definition):
        """Test that resources are properly cleaned up after execution."""
        integration_manager = IntegrationManager()
        
        # Get initial resource usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_threads = process.num_threads()
        
        # Mock components
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
                
                # Execute multiple times
                for _ in range(5):
                    await integration_manager.execute(sample_task_definition)
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Check resource usage after execution
                final_memory = process.memory_info().rss
                final_threads = process.num_threads()
                
                # Memory should not grow excessively (allow 50MB growth)
                memory_growth = (final_memory - initial_memory) / (1024 * 1024)
                assert memory_growth < 50  # Less than 50MB growth
                
                # Thread count should not grow excessively
                thread_growth = final_threads - initial_threads
                assert thread_growth < 10  # Less than 10 additional threads
    
    @pytest.mark.asyncio
    async def test_performance_optimization_recommendations(self, performance_optimizer):
        """Test performance optimization recommendations."""
        # Create mock component with performance issues
        mock_component = Mock()
        
        # Simulate slow operation
        async def slow_operation():
            await asyncio.sleep(0.5)  # 500ms - should trigger warnings
            return "result"
        
        mock_component.slow_execute = slow_operation
        
        # Benchmark the component
        benchmarks = await performance_optimizer.benchmark_component(
            mock_component, "slow_component", ["slow_execute"], iterations=3
        )
        
        # Add some metrics to trigger recommendations
        performance_optimizer._add_metric("response_time", 2000, "ms", "slow_component", "slow_execute")
        performance_optimizer._add_metric("memory_usage", 85, "%", "slow_component")
        performance_optimizer._add_metric("cpu_usage", 75, "%", "slow_component")
        
        # Analyze performance to generate recommendations
        await performance_optimizer._analyze_performance()
        
        # Check for recommendations
        profile = performance_optimizer.profiles.get("slow_component")
        assert profile is not None
        assert len(profile.recommendations) > 0
        
        # Verify recommendation content
        recommendations = profile.recommendations
        response_time_recs = [r for r in recommendations if "response time" in r.issue.lower()]
        memory_recs = [r for r in recommendations if "memory" in r.issue.lower()]
        cpu_recs = [r for r in recommendations if "cpu" in r.issue.lower()]
        
        assert len(response_time_recs) > 0
        assert len(memory_recs) > 0
        assert len(cpu_recs) > 0
        
        # Check recommendation priorities
        priorities = [r.priority for r in recommendations]
        assert "medium" in priorities or "high" in priorities or "critical" in priorities


class TestScalabilityBenchmarks:
    """Test system scalability under various conditions."""
    
    @pytest.mark.asyncio
    async def test_population_size_scaling(self, sample_task_definition):
        """Test performance scaling with different population sizes."""
        population_sizes = [5, 10, 20]
        execution_times = []
        
        for pop_size in population_sizes:
            # Mock settings for different population sizes
            with patch('config.settings.POPULATION_SIZE', pop_size):
                integration_manager = IntegrationManager()
                
                with patch('task_manager.agent.TaskManagerAgent') as mock_task_manager_class:
                    mock_task_manager = Mock()
                    
                    # Simulate execution time proportional to population size
                    async def scaled_execution():
                        await asyncio.sleep(0.01 * pop_size)  # 10ms per individual
                    
                    mock_task_manager.manage_evolutionary_cycle = Mock(return_value=scaled_execution())
                    mock_task_manager_class.return_value = mock_task_manager
                    
                    with patch('prompt_designer.agent.PromptDesignerAgent'), \
                         patch('code_generator.agent.CodeGeneratorAgent'), \
                         patch('evaluator_agent.agent.EvaluatorAgent'), \
                         patch('database_agent.agent.InMemoryDatabaseAgent') as mock_db_class, \
                         patch('selection_controller.agent.SelectionControllerAgent'):
                        
                        mock_db = Mock()
                        mock_db.get_best_programs = Mock(return_value=asyncio.coroutine(lambda task_id, limit=1000: [])())
                        mock_db_class.return_value = mock_db
                        
                        start_time = time.time()
                        result = await integration_manager.execute(sample_task_definition)
                        execution_time = time.time() - start_time
                        execution_times.append(execution_time)
                        
                        assert result.status.value == "completed"
        
        # Verify scaling behavior
        assert len(execution_times) == len(population_sizes)
        
        # Execution time should scale reasonably with population size
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[i-1]
            pop_ratio = population_sizes[i] / population_sizes[i-1]
            
            # Execution time should not scale worse than linearly
            assert ratio <= pop_ratio * 1.5  # Allow 50% overhead
    
    @pytest.mark.asyncio
    async def test_generation_scaling(self, sample_task_definition):
        """Test performance scaling with different generation counts."""
        generation_counts = [1, 2, 5]
        execution_times = []
        
        for gen_count in generation_counts:
            with patch('config.settings.GENERATIONS', gen_count):
                integration_manager = IntegrationManager()
                
                with patch('task_manager.agent.TaskManagerAgent') as mock_task_manager_class:
                    mock_task_manager = Mock()
                    
                    # Simulate execution time proportional to generations
                    async def scaled_execution():
                        await asyncio.sleep(0.05 * gen_count)  # 50ms per generation
                    
                    mock_task_manager.manage_evolutionary_cycle = Mock(return_value=scaled_execution())
                    mock_task_manager_class.return_value = mock_task_manager
                    
                    with patch('prompt_designer.agent.PromptDesignerAgent'), \
                         patch('code_generator.agent.CodeGeneratorAgent'), \
                         patch('evaluator_agent.agent.EvaluatorAgent'), \
                         patch('database_agent.agent.InMemoryDatabaseAgent') as mock_db_class, \
                         patch('selection_controller.agent.SelectionControllerAgent'):
                        
                        mock_db = Mock()
                        mock_db.get_best_programs = Mock(return_value=asyncio.coroutine(lambda task_id, limit=1000: [])())
                        mock_db_class.return_value = mock_db
                        
                        start_time = time.time()
                        result = await integration_manager.execute(sample_task_definition)
                        execution_time = time.time() - start_time
                        execution_times.append(execution_time)
                        
                        assert result.status.value == "completed"
        
        # Verify scaling behavior
        assert len(execution_times) == len(generation_counts)
        
        # Execution time should scale linearly with generation count
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[i-1]
            gen_ratio = generation_counts[i] / generation_counts[i-1]
            
            # Allow some overhead but should be roughly linear
            assert ratio <= gen_ratio * 1.3  # Allow 30% overhead


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])


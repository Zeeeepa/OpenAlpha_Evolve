"""
Integration Manager - Core orchestrator for the autonomous development pipeline.

This module coordinates all system components and manages the end-to-end workflow
for autonomous code generation, evaluation, and evolution.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.interfaces import (
    TaskDefinition, Program, BaseAgent,
    TaskManagerInterface, PromptDesignerInterface, 
    CodeGeneratorInterface, EvaluatorAgentInterface,
    DatabaseAgentInterface, SelectionControllerInterface
)
from config import settings

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_programs_generated: int = 0
    total_programs_evaluated: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    best_fitness_score: float = 0.0
    average_fitness_score: float = 0.0
    generations_completed: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Calculate pipeline duration."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Calculate evaluation success rate."""
        total = self.total_programs_evaluated
        return (self.successful_evaluations / total) if total > 0 else 0.0


@dataclass
class PipelineState:
    """Current state of the pipeline."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: PipelineStatus = PipelineStatus.IDLE
    task_definition: Optional[TaskDefinition] = None
    current_generation: int = 0
    current_population: List[Program] = field(default_factory=list)
    best_programs: List[Program] = field(default_factory=list)
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class IntegrationManager(BaseAgent):
    """
    Core integration manager that orchestrates the entire autonomous development pipeline.
    
    This class coordinates all system components and manages the end-to-end workflow
    for autonomous code generation, evaluation, and evolution.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the integration manager."""
        super().__init__(config)
        self.state = PipelineState()
        self.agents: Dict[str, BaseAgent] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.health_checks: Dict[str, Callable] = {}
        self._shutdown_event = asyncio.Event()
        
        logger.info("Integration Manager initialized")
    
    async def execute(self, task_definition: TaskDefinition) -> PipelineState:
        """
        Execute the complete autonomous development pipeline.
        
        Args:
            task_definition: The task to be solved
            
        Returns:
            Final pipeline state
        """
        try:
            self.state.task_definition = task_definition
            self.state.status = PipelineStatus.INITIALIZING
            self.state.updated_at = time.time()
            
            logger.info(f"Starting pipeline execution for task: {task_definition.id}")
            
            # Initialize all components
            await self._initialize_components()
            
            # Validate system health
            await self._validate_system_health()
            
            # Execute the main pipeline
            await self._execute_pipeline()
            
            self.state.status = PipelineStatus.COMPLETED
            self.state.metrics.end_time = time.time()
            
            logger.info(f"Pipeline completed successfully in {self.state.metrics.duration:.2f}s")
            
        except Exception as e:
            self.state.status = PipelineStatus.FAILED
            self.state.metrics.errors.append(str(e))
            self.state.metrics.end_time = time.time()
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
        
        finally:
            self.state.updated_at = time.time()
            await self._emit_event("pipeline_completed", self.state)
        
        return self.state
    
    async def _initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing system components...")
        
        try:
            # Import and initialize agents
            from task_manager.agent import TaskManagerAgent
            from prompt_designer.agent import PromptDesignerAgent
            from code_generator.agent import CodeGeneratorAgent
            from evaluator_agent.agent import EvaluatorAgent
            from database_agent.agent import InMemoryDatabaseAgent
            from selection_controller.agent import SelectionControllerAgent
            
            # Initialize agents
            self.agents['task_manager'] = TaskManagerAgent(
                task_definition=self.state.task_definition
            )
            self.agents['prompt_designer'] = PromptDesignerAgent(
                task_definition=self.state.task_definition
            )
            self.agents['code_generator'] = CodeGeneratorAgent()
            self.agents['evaluator'] = EvaluatorAgent(
                task_definition=self.state.task_definition
            )
            self.agents['database'] = InMemoryDatabaseAgent()
            self.agents['selection_controller'] = SelectionControllerAgent()
            
            # Register health checks
            for name, agent in self.agents.items():
                self.health_checks[name] = self._create_health_check(agent)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def _validate_system_health(self):
        """Validate that all system components are healthy."""
        logger.info("Validating system health...")
        
        health_results = {}
        for name, health_check in self.health_checks.items():
            try:
                result = await health_check()
                health_results[name] = result
                if not result:
                    raise Exception(f"Health check failed for component: {name}")
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                raise
        
        logger.info("System health validation passed")
        await self._emit_event("health_check_completed", health_results)
    
    async def _execute_pipeline(self):
        """Execute the main evolutionary pipeline."""
        logger.info("Starting evolutionary pipeline execution...")
        
        self.state.status = PipelineStatus.RUNNING
        task_manager = self.agents['task_manager']
        
        try:
            # Execute the evolutionary cycle
            await task_manager.manage_evolutionary_cycle()
            
            # Update metrics
            await self._update_metrics()
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            raise
    
    async def _update_metrics(self):
        """Update pipeline metrics from component states."""
        try:
            database = self.agents['database']
            task_id = self.state.task_definition.id
            
            # Get all programs for this task
            all_programs = await database.get_best_programs(task_id, limit=1000)
            
            if all_programs:
                self.state.metrics.total_programs_generated = len(all_programs)
                self.state.metrics.total_programs_evaluated = len([
                    p for p in all_programs if p.status == "evaluated"
                ])
                self.state.metrics.successful_evaluations = len([
                    p for p in all_programs 
                    if p.status == "evaluated" and not p.errors
                ])
                self.state.metrics.failed_evaluations = len([
                    p for p in all_programs 
                    if p.status == "evaluated" and p.errors
                ])
                
                # Calculate fitness statistics
                evaluated_programs = [
                    p for p in all_programs 
                    if p.status == "evaluated" and p.fitness_scores
                ]
                
                if evaluated_programs:
                    fitness_scores = [
                        max(p.fitness_scores.values()) 
                        for p in evaluated_programs
                    ]
                    self.state.metrics.best_fitness_score = max(fitness_scores)
                    self.state.metrics.average_fitness_score = sum(fitness_scores) / len(fitness_scores)
                
                # Update best programs
                self.state.best_programs = await database.get_best_programs(task_id, limit=10)
                
                # Update generation count
                if all_programs:
                    self.state.metrics.generations_completed = max(p.generation for p in all_programs)
            
            logger.info(f"Metrics updated: {self.state.metrics.total_programs_generated} programs generated, "
                       f"{self.state.metrics.successful_evaluations} successful evaluations")
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
            self.state.metrics.errors.append(f"Metrics update failed: {e}")
    
    def _create_health_check(self, agent: BaseAgent) -> Callable:
        """Create a health check function for an agent."""
        async def health_check():
            try:
                # Basic health check - verify agent is responsive
                if hasattr(agent, 'health_check'):
                    return await agent.health_check()
                else:
                    # Default health check - just verify the agent exists and is callable
                    return agent is not None and hasattr(agent, 'execute')
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return False
        
        return health_check
    
    async def _emit_event(self, event_type: str, data: Any):
        """Emit an event to registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Event handler failed for {event_type}: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def pause_pipeline(self):
        """Pause the pipeline execution."""
        self.state.status = PipelineStatus.PAUSED
        self.state.updated_at = time.time()
        logger.info("Pipeline paused")
        await self._emit_event("pipeline_paused", self.state)
    
    async def resume_pipeline(self):
        """Resume the pipeline execution."""
        if self.state.status == PipelineStatus.PAUSED:
            self.state.status = PipelineStatus.RUNNING
            self.state.updated_at = time.time()
            logger.info("Pipeline resumed")
            await self._emit_event("pipeline_resumed", self.state)
    
    async def cancel_pipeline(self):
        """Cancel the pipeline execution."""
        self.state.status = PipelineStatus.CANCELLED
        self.state.updated_at = time.time()
        self._shutdown_event.set()
        logger.info("Pipeline cancelled")
        await self._emit_event("pipeline_cancelled", self.state)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'id': self.state.id,
            'status': self.state.status.value,
            'task_id': self.state.task_definition.id if self.state.task_definition else None,
            'current_generation': self.state.current_generation,
            'metrics': {
                'duration': self.state.metrics.duration,
                'programs_generated': self.state.metrics.total_programs_generated,
                'programs_evaluated': self.state.metrics.total_programs_evaluated,
                'success_rate': self.state.metrics.success_rate,
                'best_fitness': self.state.metrics.best_fitness_score,
                'average_fitness': self.state.metrics.average_fitness_score,
                'generations_completed': self.state.metrics.generations_completed,
                'errors': len(self.state.metrics.errors)
            },
            'created_at': self.state.created_at,
            'updated_at': self.state.updated_at
        }
    
    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed pipeline metrics."""
        status = self.get_status()
        
        # Add component health status
        health_status = {}
        for name, health_check in self.health_checks.items():
            try:
                health_status[name] = await health_check()
            except Exception as e:
                health_status[name] = False
                logger.error(f"Health check failed for {name}: {e}")
        
        status['component_health'] = health_status
        status['best_programs'] = [
            {
                'id': p.id,
                'fitness_scores': p.fitness_scores,
                'generation': p.generation,
                'status': p.status,
                'errors': len(p.errors)
            }
            for p in self.state.best_programs[:5]  # Top 5 programs
        ]
        
        return status


"""
Workflow Coordinator - Manages cross-component communication and workflow automation.

This module provides advanced workflow coordination capabilities including
event-driven communication, workflow templates, and automated task sequencing.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class WorkflowStepStatus(Enum):
    """Status of individual workflow steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Definition of a workflow step."""
    id: str
    name: str
    agent_type: str
    method: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[str] = None


@dataclass
class WorkflowTemplate:
    """Template for defining reusable workflows."""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Runtime execution state of a workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    template_id: str = ""
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error: Optional[str] = None


class WorkflowCoordinator:
    """
    Coordinates complex workflows across multiple agents and components.
    
    Provides event-driven communication, dependency management, and
    automated workflow execution capabilities.
    """
    
    def __init__(self):
        """Initialize the workflow coordinator."""
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.agents: Dict[str, Any] = {}
        self.event_bus: Dict[str, List[Callable]] = {}
        self._running = False
        
        # Register built-in workflow templates
        self._register_builtin_templates()
        
        logger.info("Workflow Coordinator initialized")
    
    def register_agent(self, agent_type: str, agent_instance: Any):
        """Register an agent for workflow execution."""
        self.agents[agent_type] = agent_instance
        logger.info(f"Registered agent: {agent_type}")
    
    def register_template(self, template: WorkflowTemplate):
        """Register a workflow template."""
        self.templates[template.id] = template
        logger.info(f"Registered workflow template: {template.id}")
    
    def subscribe_to_event(self, event_type: str, handler: Callable):
        """Subscribe to workflow events."""
        if event_type not in self.event_bus:
            self.event_bus[event_type] = []
        self.event_bus[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: Any):
        """Emit a workflow event."""
        if event_type in self.event_bus:
            for handler in self.event_bus[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Event handler failed for {event_type}: {e}")
    
    async def execute_workflow(self, template_id: str, context: Dict[str, Any] = None) -> WorkflowExecution:
        """
        Execute a workflow from a template.
        
        Args:
            template_id: ID of the workflow template to execute
            context: Initial context data for the workflow
            
        Returns:
            WorkflowExecution instance with results
        """
        if template_id not in self.templates:
            raise ValueError(f"Workflow template not found: {template_id}")
        
        template = self.templates[template_id]
        execution = WorkflowExecution(
            template_id=template_id,
            context=context or {},
            steps={step.id: WorkflowStep(**step.__dict__) for step in template.steps}
        )
        
        self.executions[execution.id] = execution
        
        logger.info(f"Starting workflow execution: {template.name} ({execution.id})")
        
        try:
            execution.status = WorkflowStepStatus.RUNNING
            await self.emit_event("workflow_started", execution)
            
            # Execute steps in dependency order
            await self._execute_workflow_steps(execution)
            
            execution.status = WorkflowStepStatus.COMPLETED
            execution.end_time = time.time()
            
            logger.info(f"Workflow completed: {execution.id}")
            await self.emit_event("workflow_completed", execution)
            
        except Exception as e:
            execution.status = WorkflowStepStatus.FAILED
            execution.error = str(e)
            execution.end_time = time.time()
            
            logger.error(f"Workflow failed: {execution.id} - {e}")
            await self.emit_event("workflow_failed", execution)
            raise
        
        return execution
    
    async def _execute_workflow_steps(self, execution: WorkflowExecution):
        """Execute workflow steps in dependency order."""
        completed_steps = set()
        
        while len(completed_steps) < len(execution.steps):
            # Find steps that can be executed (dependencies satisfied)
            ready_steps = []
            for step_id, step in execution.steps.items():
                if (step.status == WorkflowStepStatus.PENDING and 
                    all(dep in completed_steps for dep in step.dependencies)):
                    ready_steps.append(step)
            
            if not ready_steps:
                # Check for circular dependencies or other issues
                pending_steps = [s for s in execution.steps.values() 
                               if s.status == WorkflowStepStatus.PENDING]
                if pending_steps:
                    raise Exception(f"Circular dependency or missing dependencies detected. "
                                  f"Pending steps: {[s.id for s in pending_steps]}")
                break
            
            # Execute ready steps in parallel
            tasks = [self._execute_step(execution, step) for step in ready_steps]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(ready_steps, results):
                if isinstance(result, Exception):
                    step.status = WorkflowStepStatus.FAILED
                    step.error = str(result)
                    step.end_time = time.time()
                    
                    # Decide whether to fail the entire workflow
                    if step.max_retries <= step.retry_count:
                        raise Exception(f"Step {step.id} failed after {step.retry_count} retries: {result}")
                else:
                    step.status = WorkflowStepStatus.COMPLETED
                    step.result = result
                    step.end_time = time.time()
                    completed_steps.add(step.id)
                    
                    # Update execution context with step result
                    execution.context[f"step_{step.id}_result"] = result
    
    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep):
        """Execute a single workflow step."""
        logger.info(f"Executing step: {step.name} ({step.id})")
        
        step.status = WorkflowStepStatus.RUNNING
        step.start_time = time.time()
        
        await self.emit_event("step_started", {"execution": execution, "step": step})
        
        try:
            # Get the agent for this step
            if step.agent_type not in self.agents:
                raise Exception(f"Agent not registered: {step.agent_type}")
            
            agent = self.agents[step.agent_type]
            
            # Get the method to call
            if not hasattr(agent, step.method):
                raise Exception(f"Method {step.method} not found on agent {step.agent_type}")
            
            method = getattr(agent, step.method)
            
            # Prepare parameters with context substitution
            params = self._substitute_context_variables(step.parameters, execution.context)
            
            # Execute with timeout if specified
            if step.timeout:
                result = await asyncio.wait_for(method(**params), timeout=step.timeout)
            else:
                result = await method(**params)
            
            await self.emit_event("step_completed", {"execution": execution, "step": step, "result": result})
            return result
            
        except Exception as e:
            step.retry_count += 1
            await self.emit_event("step_failed", {"execution": execution, "step": step, "error": e})
            
            if step.retry_count <= step.max_retries:
                logger.warning(f"Step {step.id} failed, retrying ({step.retry_count}/{step.max_retries}): {e}")
                await asyncio.sleep(2 ** step.retry_count)  # Exponential backoff
                return await self._execute_step(execution, step)
            else:
                raise
    
    def _substitute_context_variables(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute context variables in parameters."""
        result = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Context variable substitution
                var_name = value[2:-1]
                if var_name in context:
                    result[key] = context[var_name]
                else:
                    raise Exception(f"Context variable not found: {var_name}")
            else:
                result[key] = value
        return result
    
    def _register_builtin_templates(self):
        """Register built-in workflow templates."""
        
        # Standard evolutionary cycle workflow
        evolutionary_cycle = WorkflowTemplate(
            id="evolutionary_cycle",
            name="Standard Evolutionary Cycle",
            description="Complete evolutionary algorithm cycle with population initialization, evaluation, and selection",
            steps=[
                WorkflowStep(
                    id="initialize_population",
                    name="Initialize Population",
                    agent_type="task_manager",
                    method="initialize_population",
                    parameters={}
                ),
                WorkflowStep(
                    id="evaluate_population",
                    name="Evaluate Population",
                    agent_type="task_manager",
                    method="evaluate_population",
                    parameters={"programs": "${step_initialize_population_result}"},
                    dependencies=["initialize_population"]
                ),
                WorkflowStep(
                    id="select_parents",
                    name="Select Parents",
                    agent_type="selection_controller",
                    method="select_parents",
                    parameters={
                        "evaluated_programs": "${step_evaluate_population_result}",
                        "num_parents": 5
                    },
                    dependencies=["evaluate_population"]
                ),
                WorkflowStep(
                    id="generate_offspring",
                    name="Generate Offspring",
                    agent_type="task_manager",
                    method="generate_offspring",
                    parameters={"parents": "${step_select_parents_result}"},
                    dependencies=["select_parents"]
                )
            ]
        )
        
        # Health check workflow
        health_check = WorkflowTemplate(
            id="system_health_check",
            name="System Health Check",
            description="Comprehensive health check of all system components",
            steps=[
                WorkflowStep(
                    id="check_task_manager",
                    name="Check Task Manager",
                    agent_type="task_manager",
                    method="health_check",
                    parameters={}
                ),
                WorkflowStep(
                    id="check_code_generator",
                    name="Check Code Generator",
                    agent_type="code_generator",
                    method="health_check",
                    parameters={}
                ),
                WorkflowStep(
                    id="check_evaluator",
                    name="Check Evaluator",
                    agent_type="evaluator",
                    method="health_check",
                    parameters={}
                ),
                WorkflowStep(
                    id="check_database",
                    name="Check Database",
                    agent_type="database",
                    method="health_check",
                    parameters={}
                )
            ]
        )
        
        # Performance benchmark workflow
        performance_benchmark = WorkflowTemplate(
            id="performance_benchmark",
            name="Performance Benchmark",
            description="Benchmark system performance and collect metrics",
            steps=[
                WorkflowStep(
                    id="benchmark_code_generation",
                    name="Benchmark Code Generation",
                    agent_type="code_generator",
                    method="benchmark_performance",
                    parameters={"iterations": 10},
                    timeout=300.0
                ),
                WorkflowStep(
                    id="benchmark_evaluation",
                    name="Benchmark Evaluation",
                    agent_type="evaluator",
                    method="benchmark_performance",
                    parameters={"iterations": 10},
                    timeout=300.0
                ),
                WorkflowStep(
                    id="collect_metrics",
                    name="Collect System Metrics",
                    agent_type="task_manager",
                    method="collect_performance_metrics",
                    parameters={
                        "code_gen_results": "${step_benchmark_code_generation_result}",
                        "eval_results": "${step_benchmark_evaluation_result}"
                    },
                    dependencies=["benchmark_code_generation", "benchmark_evaluation"]
                )
            ]
        )
        
        self.register_template(evolutionary_cycle)
        self.register_template(health_check)
        self.register_template(performance_benchmark)
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get the status of a workflow execution."""
        if execution_id not in self.executions:
            raise ValueError(f"Execution not found: {execution_id}")
        
        execution = self.executions[execution_id]
        
        return {
            "id": execution.id,
            "template_id": execution.template_id,
            "status": execution.status.value,
            "start_time": execution.start_time,
            "end_time": execution.end_time,
            "duration": (execution.end_time or time.time()) - execution.start_time,
            "steps": {
                step_id: {
                    "name": step.name,
                    "status": step.status.value,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "duration": (step.end_time - step.start_time) if step.start_time and step.end_time else None,
                    "retry_count": step.retry_count,
                    "error": step.error
                }
                for step_id, step in execution.steps.items()
            },
            "error": execution.error
        }
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available workflow templates."""
        return [
            {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "steps": len(template.steps),
                "metadata": template.metadata
            }
            for template in self.templates.values()
        ]
    
    def list_executions(self) -> List[Dict[str, Any]]:
        """List all workflow executions."""
        return [
            {
                "id": execution.id,
                "template_id": execution.template_id,
                "status": execution.status.value,
                "start_time": execution.start_time,
                "end_time": execution.end_time,
                "duration": (execution.end_time or time.time()) - execution.start_time
            }
            for execution in self.executions.values()
        ]


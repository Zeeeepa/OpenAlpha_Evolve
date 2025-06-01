"""
Pipeline Orchestrator - Central coordinator for autonomous development pipeline.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.interfaces import BaseAgent, TaskDefinition, Program
from context_analysis.engine import ContextAnalysisEngine, CodebaseSnapshot
from error_analysis.error_classifier import ErrorClassifier
from debugging_system.auto_debugger import AutoDebugger
from learning_system.learning_engine import LearningEngine
from task_manager.agent import TaskManagerAgent

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Stages in the autonomous development pipeline."""
    INITIALIZATION = "initialization"
    CONTEXT_ANALYSIS = "context_analysis"
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    TASK_DECOMPOSITION = "task_decomposition"
    SOLUTION_GENERATION = "solution_generation"
    VALIDATION = "validation"
    ERROR_ANALYSIS = "error_analysis"
    DEBUGGING = "debugging"
    LEARNING = "learning"
    OPTIMIZATION = "optimization"
    COMPLETION = "completion"

@dataclass
class PipelineState:
    """Represents the current state of the pipeline."""
    current_stage: PipelineStage
    task_definition: Optional[TaskDefinition] = None
    codebase_context: Optional[CodebaseSnapshot] = None
    generated_programs: List[Program] = field(default_factory=list)
    errors_encountered: List[Dict[str, Any]] = field(default_factory=list)
    debug_results: List[Dict[str, Any]] = field(default_factory=list)
    learning_insights: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    final_programs: List[Program]
    execution_time: float
    stages_completed: List[PipelineStage]
    errors_resolved: int
    learning_insights: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]

class PipelineOrchestrator(BaseAgent):
    """
    Central orchestrator for the autonomous development pipeline.
    Coordinates all components and manages the end-to-end development process.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize component agents
        self.context_engine = ContextAnalysisEngine(config)
        self.error_classifier = ErrorClassifier(config)
        self.auto_debugger = AutoDebugger(config)
        self.learning_engine = LearningEngine(config)
        
        # Pipeline configuration
        self.enable_learning = self.config.get('enable_learning', True)
        self.enable_auto_debugging = self.config.get('enable_auto_debugging', True)
        self.enable_context_analysis = self.config.get('enable_context_analysis', True)
        self.max_pipeline_retries = self.config.get('max_pipeline_retries', 3)
        
        # Callbacks for monitoring
        self.stage_callbacks: Dict[PipelineStage, List[Callable]] = {}
        self.progress_callback: Optional[Callable] = None
        
    async def execute(self, task_definition: TaskDefinition) -> PipelineResult:
        """Main execution method - runs the complete autonomous development pipeline."""
        return await self.run_pipeline(task_definition)
    
    async def run_pipeline(self, task_definition: TaskDefinition) -> PipelineResult:
        """
        Run the complete autonomous development pipeline.
        
        Args:
            task_definition: The task to be solved autonomously
        """
        start_time = datetime.now()
        logger.info(f"Starting autonomous development pipeline for task: {task_definition.id}")
        
        # Initialize pipeline state
        state = PipelineState(
            current_stage=PipelineStage.INITIALIZATION,
            task_definition=task_definition,
            max_retries=self.max_pipeline_retries
        )
        
        try:
            # Execute pipeline stages
            await self._execute_stage(PipelineStage.INITIALIZATION, state)
            
            if self.enable_context_analysis:
                await self._execute_stage(PipelineStage.CONTEXT_ANALYSIS, state)
            
            await self._execute_stage(PipelineStage.REQUIREMENT_ANALYSIS, state)
            await self._execute_stage(PipelineStage.TASK_DECOMPOSITION, state)
            await self._execute_stage(PipelineStage.SOLUTION_GENERATION, state)
            await self._execute_stage(PipelineStage.VALIDATION, state)
            
            # Error handling and debugging loop
            while state.errors_encountered and state.retry_count < state.max_retries:
                if self.enable_auto_debugging:
                    await self._execute_stage(PipelineStage.ERROR_ANALYSIS, state)
                    await self._execute_stage(PipelineStage.DEBUGGING, state)
                    await self._execute_stage(PipelineStage.VALIDATION, state)
                else:
                    break
                state.retry_count += 1
            
            # Learning and optimization
            if self.enable_learning:
                await self._execute_stage(PipelineStage.LEARNING, state)
                await self._execute_stage(PipelineStage.OPTIMIZATION, state)
            
            await self._execute_stage(PipelineStage.COMPLETION, state)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Generate final result
            result = PipelineResult(
                success=len(state.generated_programs) > 0 and not state.errors_encountered,
                final_programs=state.generated_programs,
                execution_time=execution_time,
                stages_completed=[record['stage'] for record in state.stage_history],
                errors_resolved=len(state.debug_results),
                learning_insights=state.learning_insights,
                performance_metrics=state.performance_metrics,
                recommendations=self._generate_recommendations(state)
            )
            
            logger.info(f"Pipeline completed: success={result.success}, "
                       f"programs={len(result.final_programs)}, time={execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                success=False,
                final_programs=state.generated_programs,
                execution_time=execution_time,
                stages_completed=[record['stage'] for record in state.stage_history],
                errors_resolved=0,
                learning_insights={},
                performance_metrics={},
                recommendations=[f"Pipeline failed with error: {str(e)}"]
            )
    
    async def _execute_stage(self, stage: PipelineStage, state: PipelineState):
        """Execute a specific pipeline stage."""
        logger.debug(f"Executing pipeline stage: {stage.value}")
        state.current_stage = stage
        
        stage_start = datetime.now()
        
        try:
            # Execute stage-specific logic
            if stage == PipelineStage.INITIALIZATION:
                await self._stage_initialization(state)
            elif stage == PipelineStage.CONTEXT_ANALYSIS:
                await self._stage_context_analysis(state)
            elif stage == PipelineStage.REQUIREMENT_ANALYSIS:
                await self._stage_requirement_analysis(state)
            elif stage == PipelineStage.TASK_DECOMPOSITION:
                await self._stage_task_decomposition(state)
            elif stage == PipelineStage.SOLUTION_GENERATION:
                await self._stage_solution_generation(state)
            elif stage == PipelineStage.VALIDATION:
                await self._stage_validation(state)
            elif stage == PipelineStage.ERROR_ANALYSIS:
                await self._stage_error_analysis(state)
            elif stage == PipelineStage.DEBUGGING:
                await self._stage_debugging(state)
            elif stage == PipelineStage.LEARNING:
                await self._stage_learning(state)
            elif stage == PipelineStage.OPTIMIZATION:
                await self._stage_optimization(state)
            elif stage == PipelineStage.COMPLETION:
                await self._stage_completion(state)
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            # Record stage completion
            state.stage_history.append({
                'stage': stage,
                'duration': stage_duration,
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            # Call stage callbacks
            if stage in self.stage_callbacks:
                for callback in self.stage_callbacks[stage]:
                    try:
                        await callback(stage, state)
                    except Exception as e:
                        logger.warning(f"Stage callback failed: {e}")
            
            # Update progress
            if self.progress_callback:
                try:
                    progress = len(state.stage_history) / len(PipelineStage)
                    await self.progress_callback(progress, f"Completed {stage.value}")
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
            
            logger.debug(f"Stage {stage.value} completed in {stage_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}", exc_info=True)
            
            state.stage_history.append({
                'stage': stage,
                'duration': (datetime.now() - stage_start).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            })
            
            raise
    
    async def _stage_initialization(self, state: PipelineState):
        """Initialize the pipeline."""
        logger.info("Initializing autonomous development pipeline")
        
        # Validate task definition
        if not state.task_definition:
            raise ValueError("Task definition is required")
        
        # Initialize performance metrics
        state.performance_metrics = {
            'start_time': datetime.now().isoformat(),
            'task_id': state.task_definition.id,
            'task_complexity': len(state.task_definition.description.split()),
            'initial_examples': len(state.task_definition.input_output_examples or [])
        }
        
        logger.info(f"Pipeline initialized for task: {state.task_definition.id}")
    
    async def _stage_context_analysis(self, state: PipelineState):
        """Analyze codebase context."""
        logger.info("Analyzing codebase context")
        
        # Perform context analysis
        state.codebase_context = await self.context_engine.execute()
        
        # Update performance metrics
        state.performance_metrics.update({
            'codebase_files': state.codebase_context.total_files,
            'codebase_lines': state.codebase_context.total_lines,
            'codebase_complexity': state.codebase_context.complexity_metrics.get('total_complexity', 0)
        })
        
        logger.info(f"Context analysis complete: {state.codebase_context.total_files} files analyzed")
    
    async def _stage_requirement_analysis(self, state: PipelineState):
        """Analyze requirements and constraints."""
        logger.info("Analyzing requirements")
        
        # Analyze task requirements
        requirements = {
            'function_name': state.task_definition.function_name_to_evolve,
            'input_output_examples': len(state.task_definition.input_output_examples or []),
            'allowed_imports': state.task_definition.allowed_imports or [],
            'complexity_estimate': self._estimate_task_complexity(state.task_definition)
        }
        
        state.performance_metrics['requirements'] = requirements
        
        logger.info(f"Requirements analyzed: complexity={requirements['complexity_estimate']}")
    
    async def _stage_task_decomposition(self, state: PipelineState):
        """Decompose task into manageable components."""
        logger.info("Decomposing task")
        
        # Simple task decomposition (can be enhanced)
        decomposition = {
            'main_task': state.task_definition.description,
            'sub_tasks': self._decompose_task(state.task_definition),
            'dependencies': self._analyze_task_dependencies(state.task_definition)
        }
        
        state.performance_metrics['task_decomposition'] = decomposition
        
        logger.info(f"Task decomposed into {len(decomposition['sub_tasks'])} sub-tasks")
    
    async def _stage_solution_generation(self, state: PipelineState):
        """Generate solutions using the evolutionary algorithm."""
        logger.info("Generating solutions")
        
        # Create task manager and run evolution
        task_manager = TaskManagerAgent(task_definition=state.task_definition)
        
        # Set progress callback if available
        if self.progress_callback:
            task_manager.progress_callback = self.progress_callback
        
        # Execute evolutionary algorithm
        best_programs = await task_manager.execute()
        
        if best_programs:
            state.generated_programs = best_programs
            
            # Update performance metrics
            state.performance_metrics.update({
                'solutions_generated': len(best_programs),
                'best_fitness': max(p.fitness_scores.get('correctness', 0) for p in best_programs),
                'average_fitness': sum(p.fitness_scores.get('correctness', 0) for p in best_programs) / len(best_programs)
            })
        
        logger.info(f"Solution generation complete: {len(state.generated_programs)} programs generated")
    
    async def _stage_validation(self, state: PipelineState):
        """Validate generated solutions."""
        logger.info("Validating solutions")
        
        validation_results = []
        errors_found = []
        
        for program in state.generated_programs:
            # Check for errors
            if program.errors:
                for error in program.errors:
                    errors_found.append({
                        'program_id': program.id,
                        'error_message': error,
                        'error_type': 'runtime_error'
                    })
            
            # Validate fitness
            correctness = program.fitness_scores.get('correctness', 0)
            validation_results.append({
                'program_id': program.id,
                'correctness': correctness,
                'valid': correctness > 0.5
            })
        
        state.errors_encountered = errors_found
        state.performance_metrics['validation'] = {
            'programs_validated': len(validation_results),
            'valid_programs': sum(1 for r in validation_results if r['valid']),
            'errors_found': len(errors_found)
        }
        
        logger.info(f"Validation complete: {len(errors_found)} errors found")
    
    async def _stage_error_analysis(self, state: PipelineState):
        """Analyze encountered errors."""
        logger.info("Analyzing errors")
        
        error_classifications = []
        
        for error_info in state.errors_encountered:
            classification = await self.error_classifier.classify_error(error_info)
            error_classifications.append({
                'program_id': error_info['program_id'],
                'classification': classification,
                'suggested_fixes': classification.suggested_fixes
            })
        
        state.performance_metrics['error_analysis'] = {
            'errors_classified': len(error_classifications),
            'error_categories': [c['classification'].category.value for c in error_classifications]
        }
        
        logger.info(f"Error analysis complete: {len(error_classifications)} errors classified")
    
    async def _stage_debugging(self, state: PipelineState):
        """Attempt to debug and fix errors."""
        logger.info("Debugging errors")
        
        debug_results = []
        
        for error_info in state.errors_encountered:
            # Find the program with this error
            program = next((p for p in state.generated_programs if p.id == error_info['program_id']), None)
            
            if program:
                debug_result = await self.auto_debugger.debug_program(program, error_info)
                debug_results.append({
                    'program_id': program.id,
                    'debug_result': debug_result,
                    'fixed': debug_result.success
                })
                
                # Update program if fixed
                if debug_result.success and debug_result.fixed_code:
                    program.code = debug_result.fixed_code
                    program.errors = []  # Clear errors
        
        state.debug_results = debug_results
        state.performance_metrics['debugging'] = {
            'debug_attempts': len(debug_results),
            'successful_fixes': sum(1 for r in debug_results if r['fixed'])
        }
        
        # Remove fixed errors
        fixed_program_ids = {r['program_id'] for r in debug_results if r['fixed']}
        state.errors_encountered = [e for e in state.errors_encountered 
                                  if e['program_id'] not in fixed_program_ids]
        
        logger.info(f"Debugging complete: {len(debug_results)} attempts, "
                   f"{sum(1 for r in debug_results if r['fixed'])} fixes")
    
    async def _stage_learning(self, state: PipelineState):
        """Learn from the development process."""
        logger.info("Learning from development process")
        
        learning_data = {
            'task_definition': state.task_definition,
            'programs': state.generated_programs,
            'errors': state.errors_encountered,
            'debug_results': state.debug_results,
            'performance_metrics': state.performance_metrics,
            'success_indicators': {
                'task_completed': len(state.generated_programs) > 0,
                'errors_resolved': len(state.debug_results) > 0,
                'high_fitness_achieved': any(p.fitness_scores.get('correctness', 0) > 0.8 
                                           for p in state.generated_programs)
            }
        }
        
        learning_result = await self.learning_engine.learn_from_data(learning_data)
        state.learning_insights = learning_result
        
        logger.info(f"Learning complete: {learning_result.get('new_patterns', 0)} new patterns learned")
    
    async def _stage_optimization(self, state: PipelineState):
        """Optimize based on learning insights."""
        logger.info("Optimizing based on insights")
        
        # Get recommendations from learning engine
        task_context = {
            'description': state.task_definition.description,
            'complexity': state.performance_metrics.get('requirements', {}).get('complexity_estimate', 'medium')
        }
        
        recommendations = self.learning_engine.get_recommendations(task_context)
        
        state.performance_metrics['optimization'] = {
            'recommendations_generated': len(recommendations),
            'recommendations': recommendations
        }
        
        logger.info(f"Optimization complete: {len(recommendations)} recommendations generated")
    
    async def _stage_completion(self, state: PipelineState):
        """Complete the pipeline."""
        logger.info("Completing pipeline")
        
        # Final performance metrics
        state.performance_metrics.update({
            'end_time': datetime.now().isoformat(),
            'total_stages': len(state.stage_history),
            'successful_stages': sum(1 for s in state.stage_history if s['success']),
            'total_retries': state.retry_count
        })
        
        logger.info("Pipeline completed successfully")
    
    def _estimate_task_complexity(self, task_definition: TaskDefinition) -> str:
        """Estimate task complexity based on various factors."""
        complexity_score = 0
        
        # Description length
        desc_words = len(task_definition.description.split())
        if desc_words > 50:
            complexity_score += 2
        elif desc_words > 20:
            complexity_score += 1
        
        # Number of examples
        examples = len(task_definition.input_output_examples or [])
        if examples > 10:
            complexity_score += 2
        elif examples > 5:
            complexity_score += 1
        
        # Allowed imports (more imports = more complex)
        imports = len(task_definition.allowed_imports or [])
        if imports > 5:
            complexity_score += 2
        elif imports > 2:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score >= 5:
            return 'high'
        elif complexity_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _decompose_task(self, task_definition: TaskDefinition) -> List[str]:
        """Decompose task into sub-tasks."""
        sub_tasks = []
        
        # Basic decomposition based on task description
        description = task_definition.description.lower()
        
        if 'function' in description:
            sub_tasks.append('Define function signature')
            sub_tasks.append('Implement core logic')
            sub_tasks.append('Handle edge cases')
        
        if 'input' in description and 'output' in description:
            sub_tasks.append('Parse input parameters')
            sub_tasks.append('Process data')
            sub_tasks.append('Format output')
        
        if 'error' in description or 'exception' in description:
            sub_tasks.append('Add error handling')
        
        if not sub_tasks:
            sub_tasks = ['Analyze requirements', 'Implement solution', 'Test and validate']
        
        return sub_tasks
    
    def _analyze_task_dependencies(self, task_definition: TaskDefinition) -> List[str]:
        """Analyze task dependencies."""
        dependencies = []
        
        # Add allowed imports as dependencies
        if task_definition.allowed_imports:
            dependencies.extend(task_definition.allowed_imports)
        
        # Analyze description for implicit dependencies
        description = task_definition.description.lower()
        
        if 'math' in description or 'calculate' in description:
            dependencies.append('math')
        if 'list' in description or 'array' in description:
            dependencies.append('collections')
        if 'string' in description or 'text' in description:
            dependencies.append('string processing')
        
        return list(set(dependencies))
    
    def _generate_recommendations(self, state: PipelineState) -> List[str]:
        """Generate recommendations based on pipeline execution."""
        recommendations = []
        
        # Performance-based recommendations
        if state.performance_metrics.get('validation', {}).get('errors_found', 0) > 0:
            recommendations.append("Consider adding more comprehensive error handling")
        
        if state.performance_metrics.get('debugging', {}).get('successful_fixes', 0) > 0:
            recommendations.append("Automated debugging was successful - consider enabling it by default")
        
        # Learning-based recommendations
        if state.learning_insights.get('new_patterns', 0) > 0:
            recommendations.append("New patterns learned - apply insights to future tasks")
        
        # Add optimization recommendations
        optimization_recs = state.performance_metrics.get('optimization', {}).get('recommendations', [])
        recommendations.extend(optimization_recs[:3])  # Top 3 recommendations
        
        return recommendations
    
    def add_stage_callback(self, stage: PipelineStage, callback: Callable):
        """Add a callback for a specific pipeline stage."""
        if stage not in self.stage_callbacks:
            self.stage_callbacks[stage] = []
        self.stage_callbacks[stage].append(callback)
    
    def set_progress_callback(self, callback: Callable):
        """Set a progress callback for the entire pipeline."""
        self.progress_callback = callback


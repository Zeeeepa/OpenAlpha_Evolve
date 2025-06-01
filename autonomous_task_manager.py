"""
Enhanced Task Manager with Autonomous Development Capabilities
Integrates all autonomous components for intelligent task management.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from core.interfaces import TaskManagerInterface, TaskDefinition, Program
from autonomous_pipeline.pipeline_orchestrator import PipelineOrchestrator, PipelineResult
from context_analysis.engine import ContextAnalysisEngine
from error_analysis.error_classifier import ErrorClassifier
from debugging_system.auto_debugger import AutoDebugger
from learning_system.learning_engine import LearningEngine
from task_manager.agent import TaskManagerAgent

logger = logging.getLogger(__name__)

class AutonomousTaskManager(TaskManagerInterface):
    """
    Enhanced task manager that leverages autonomous development capabilities
    for intelligent, self-improving task execution.
    """
    
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_definition = task_definition
        
        # Initialize autonomous components
        self.pipeline_orchestrator = PipelineOrchestrator(config)
        self.context_engine = ContextAnalysisEngine(config)
        self.error_classifier = ErrorClassifier(config)
        self.auto_debugger = AutoDebugger(config)
        self.learning_engine = LearningEngine(config)
        
        # Fallback to original task manager
        self.fallback_manager = TaskManagerAgent(task_definition, config)
        
        # Configuration
        self.use_autonomous_pipeline = self.config.get('use_autonomous_pipeline', True)
        self.enable_context_analysis = self.config.get('enable_context_analysis', True)
        self.enable_learning = self.config.get('enable_learning', True)
        self.enable_auto_debugging = self.config.get('enable_auto_debugging', True)
        
        # Callbacks
        self.progress_callback = None
        
    async def execute(self) -> List[Program]:
        """Main execution method using autonomous development pipeline."""
        logger.info(f"Starting autonomous task execution for: {self.task_definition.id}")
        
        try:
            if self.use_autonomous_pipeline:
                return await self._execute_autonomous_pipeline()
            else:
                return await self._execute_fallback()
        except Exception as e:
            logger.error(f"Autonomous execution failed, falling back to standard approach: {e}")
            return await self._execute_fallback()
    
    async def _execute_autonomous_pipeline(self) -> List[Program]:
        """Execute using the autonomous development pipeline."""
        logger.info("Executing autonomous development pipeline")
        
        # Set up progress callback
        if self.progress_callback:
            self.pipeline_orchestrator.set_progress_callback(self.progress_callback)
        
        # Pre-execution context analysis
        if self.enable_context_analysis:
            await self._perform_pre_execution_analysis()
        
        # Apply learning insights
        if self.enable_learning:
            await self._apply_learning_insights()
        
        # Run the autonomous pipeline
        pipeline_result = await self.pipeline_orchestrator.run_pipeline(self.task_definition)
        
        # Post-execution learning
        if self.enable_learning and pipeline_result.success:
            await self._perform_post_execution_learning(pipeline_result)
        
        # Return results
        if pipeline_result.success and pipeline_result.final_programs:
            logger.info(f"Autonomous pipeline succeeded with {len(pipeline_result.final_programs)} programs")
            return pipeline_result.final_programs
        else:
            logger.warning("Autonomous pipeline did not produce valid results, falling back")
            return await self._execute_fallback()
    
    async def _execute_fallback(self) -> List[Program]:
        """Execute using the original task manager as fallback."""
        logger.info("Executing fallback task manager")
        
        # Set progress callback if available
        if self.progress_callback:
            self.fallback_manager.progress_callback = self.progress_callback
        
        return await self.fallback_manager.execute()
    
    async def _perform_pre_execution_analysis(self):
        """Perform context analysis before execution."""
        logger.info("Performing pre-execution context analysis")
        
        try:
            # Analyze current codebase context
            codebase_snapshot = await self.context_engine.execute()
            
            # Enhance task definition with context insights
            self._enhance_task_with_context(codebase_snapshot)
            
            logger.info(f"Context analysis complete: {codebase_snapshot.total_files} files analyzed")
            
        except Exception as e:
            logger.warning(f"Pre-execution analysis failed: {e}")
    
    async def _apply_learning_insights(self):
        """Apply insights from previous learning."""
        logger.info("Applying learning insights")
        
        try:
            # Get recommendations for this task
            task_context = {
                'description': self.task_definition.description,
                'complexity': self.task_definition.complexity_level or 'medium',
                'domain_tags': self.task_definition.domain_tags
            }
            
            recommendations = self.learning_engine.get_recommendations(task_context)
            
            if recommendations:
                logger.info(f"Applied {len(recommendations)} learning insights")
                
                # Store recommendations in task definition for pipeline use
                if not hasattr(self.task_definition, 'learning_recommendations'):
                    self.task_definition.learning_recommendations = []
                self.task_definition.learning_recommendations.extend(recommendations)
            
        except Exception as e:
            logger.warning(f"Failed to apply learning insights: {e}")
    
    async def _perform_post_execution_learning(self, pipeline_result: PipelineResult):
        """Perform learning after successful execution."""
        logger.info("Performing post-execution learning")
        
        try:
            # Prepare learning data
            learning_data = {
                'task_definition': self.task_definition,
                'programs': pipeline_result.final_programs,
                'errors': [],  # Errors would be captured in pipeline_result
                'debug_results': [],
                'performance_metrics': pipeline_result.performance_metrics,
                'success_indicators': {
                    'task_completed': pipeline_result.success,
                    'execution_time': pipeline_result.execution_time,
                    'stages_completed': len(pipeline_result.stages_completed),
                    'errors_resolved': pipeline_result.errors_resolved
                }
            }
            
            # Learn from the execution
            learning_result = await self.learning_engine.learn_from_data(learning_data)
            
            logger.info(f"Post-execution learning complete: {learning_result.get('new_patterns', 0)} new patterns")
            
        except Exception as e:
            logger.warning(f"Post-execution learning failed: {e}")
    
    def _enhance_task_with_context(self, codebase_snapshot):
        """Enhance task definition with context insights."""
        
        # Analyze task complexity based on codebase
        if codebase_snapshot.complexity_metrics:
            avg_complexity = codebase_snapshot.complexity_metrics.get('average_complexity', 0)
            
            if avg_complexity > 10:
                self.task_definition.complexity_level = 'high'
            elif avg_complexity > 5:
                self.task_definition.complexity_level = 'medium'
            else:
                self.task_definition.complexity_level = 'low'
        
        # Extract domain tags from semantic clusters
        if codebase_snapshot.semantic_clusters:
            domain_tags = []
            for cluster_name, contexts in codebase_snapshot.semantic_clusters.items():
                if cluster_name.startswith('tag:'):
                    tag = cluster_name[4:]  # Remove 'tag:' prefix
                    if len(contexts) > 2:  # Only include significant clusters
                        domain_tags.append(tag)
            
            self.task_definition.domain_tags.extend(domain_tags[:5])  # Limit to top 5
        
        # Set context requirements
        self.task_definition.context_requirements = {
            'codebase_files': codebase_snapshot.total_files,
            'codebase_complexity': codebase_snapshot.complexity_metrics.get('total_complexity', 0),
            'available_patterns': len(codebase_snapshot.semantic_clusters)
        }
    
    async def manage_evolutionary_cycle(self):
        """Legacy method for compatibility."""
        return await self.execute()
    
    def set_progress_callback(self, callback):
        """Set progress callback for monitoring."""
        self.progress_callback = callback
        
        # Also set on fallback manager
        if hasattr(self.fallback_manager, 'progress_callback'):
            self.fallback_manager.progress_callback = callback
    
    async def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about autonomous execution."""
        stats = {
            'autonomous_mode_enabled': self.use_autonomous_pipeline,
            'context_analysis_enabled': self.enable_context_analysis,
            'learning_enabled': self.enable_learning,
            'auto_debugging_enabled': self.enable_auto_debugging
        }
        
        # Add component statistics
        if self.enable_learning:
            try:
                learning_stats = self.learning_engine.get_learning_statistics()
                stats['learning_statistics'] = learning_stats
            except Exception as e:
                logger.warning(f"Could not get learning statistics: {e}")
        
        if self.enable_auto_debugging:
            try:
                debug_stats = self.auto_debugger.get_debug_statistics()
                stats['debugging_statistics'] = debug_stats
            except Exception as e:
                logger.warning(f"Could not get debugging statistics: {e}")
        
        return stats
    
    async def reset_autonomous_components(self):
        """Reset all autonomous components (use with caution)."""
        logger.warning("Resetting all autonomous components")
        
        try:
            if hasattr(self.learning_engine, 'reset_learning_data'):
                self.learning_engine.reset_learning_data()
            
            # Clear other component caches
            if hasattr(self.context_engine, 'context_cache'):
                self.context_engine.context_cache.clear()
            
            if hasattr(self.auto_debugger, 'debug_history'):
                self.auto_debugger.debug_history.clear()
            
            logger.info("Autonomous components reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset autonomous components: {e}")
    
    def configure_autonomous_features(self, **kwargs):
        """Configure autonomous features dynamically."""
        
        if 'use_autonomous_pipeline' in kwargs:
            self.use_autonomous_pipeline = kwargs['use_autonomous_pipeline']
        
        if 'enable_context_analysis' in kwargs:
            self.enable_context_analysis = kwargs['enable_context_analysis']
        
        if 'enable_learning' in kwargs:
            self.enable_learning = kwargs['enable_learning']
        
        if 'enable_auto_debugging' in kwargs:
            self.enable_auto_debugging = kwargs['enable_auto_debugging']
        
        logger.info(f"Autonomous features configured: {kwargs}")
    
    async def validate_autonomous_setup(self) -> Dict[str, bool]:
        """Validate that all autonomous components are properly set up."""
        validation_results = {}
        
        # Test context analysis
        try:
            await self.context_engine.execute('.')
            validation_results['context_analysis'] = True
        except Exception as e:
            logger.error(f"Context analysis validation failed: {e}")
            validation_results['context_analysis'] = False
        
        # Test error classification
        try:
            test_error = {
                'error_message': 'Test error',
                'error_type': 'TestError',
                'code': 'test code'
            }
            await self.error_classifier.classify_error(test_error)
            validation_results['error_classification'] = True
        except Exception as e:
            logger.error(f"Error classification validation failed: {e}")
            validation_results['error_classification'] = False
        
        # Test learning engine
        try:
            test_data = {
                'task_definition': self.task_definition,
                'programs': [],
                'errors': [],
                'debug_results': [],
                'performance_metrics': {},
                'success_indicators': {}
            }
            await self.learning_engine.learn_from_data(test_data)
            validation_results['learning_engine'] = True
        except Exception as e:
            logger.error(f"Learning engine validation failed: {e}")
            validation_results['learning_engine'] = False
        
        # Test pipeline orchestrator
        try:
            # Just check if it can be initialized
            validation_results['pipeline_orchestrator'] = self.pipeline_orchestrator is not None
        except Exception as e:
            logger.error(f"Pipeline orchestrator validation failed: {e}")
            validation_results['pipeline_orchestrator'] = False
        
        all_valid = all(validation_results.values())
        logger.info(f"Autonomous setup validation: {'PASSED' if all_valid else 'FAILED'}")
        
        return validation_results


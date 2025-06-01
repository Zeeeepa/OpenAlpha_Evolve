"""
Learning Engine - Core component for continuous improvement and pattern recognition.
"""

import json
import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

from core.interfaces import BaseAgent, Program, TaskDefinition
from error_analysis.error_classifier import ErrorClassification
from debugging_system.auto_debugger import DebugResult

logger = logging.getLogger(__name__)

@dataclass
class LearningPattern:
    """Represents a learned pattern from development cycles."""
    pattern_id: str
    pattern_type: str  # 'success', 'failure', 'optimization', 'error_prevention'
    context: Dict[str, Any]
    conditions: List[str]
    outcomes: Dict[str, Any]
    confidence: float
    frequency: int = 1
    last_seen: datetime = field(default_factory=datetime.now)
    effectiveness_score: float = 0.0

@dataclass
class LearningInsight:
    """Represents an actionable insight derived from learning."""
    insight_id: str
    insight_type: str
    description: str
    recommended_actions: List[str]
    confidence: float
    supporting_evidence: List[str]
    applicable_contexts: List[str]

class LearningEngine(BaseAgent):
    """
    Advanced learning engine that analyzes development patterns,
    learns from successes and failures, and provides continuous improvement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.knowledge_base_path = self.config.get('knowledge_base_path', 'learning_data')
        self.patterns: Dict[str, LearningPattern] = {}
        self.insights: Dict[str, LearningInsight] = {}
        self.development_history: List[Dict[str, Any]] = []
        self.success_metrics: Dict[str, List[float]] = defaultdict(list)
        self.learning_enabled = True
        
        # Initialize knowledge base
        self._load_knowledge_base()
    
    async def execute(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method - processes learning data and generates insights."""
        return await self.learn_from_data(learning_data)
    
    async def learn_from_data(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from development cycle data.
        
        Args:
            learning_data: Dictionary containing:
                - task_definition: The task that was worked on
                - programs: List of programs generated
                - errors: List of errors encountered
                - debug_results: List of debugging results
                - performance_metrics: Performance data
                - success_indicators: Success/failure indicators
        """
        if not self.learning_enabled:
            return {'status': 'learning_disabled'}
        
        logger.info("Processing learning data for pattern recognition")
        
        # Extract and store development history
        self._record_development_cycle(learning_data)
        
        # Analyze patterns
        new_patterns = await self._analyze_patterns(learning_data)
        
        # Update existing patterns
        self._update_patterns(new_patterns)
        
        # Generate insights
        new_insights = await self._generate_insights()
        
        # Update success metrics
        self._update_success_metrics(learning_data)
        
        # Save knowledge base
        self._save_knowledge_base()
        
        logger.info(f"Learning complete: {len(new_patterns)} new patterns, "
                   f"{len(new_insights)} new insights")
        
        return {
            'status': 'success',
            'new_patterns': len(new_patterns),
            'new_insights': len(new_insights),
            'total_patterns': len(self.patterns),
            'total_insights': len(self.insights)
        }
    
    def _record_development_cycle(self, learning_data: Dict[str, Any]):
        """Record a development cycle in history."""
        cycle_record = {
            'timestamp': datetime.now().isoformat(),
            'task_id': learning_data.get('task_definition', {}).get('id', 'unknown'),
            'task_description': learning_data.get('task_definition', {}).get('description', ''),
            'num_programs': len(learning_data.get('programs', [])),
            'num_errors': len(learning_data.get('errors', [])),
            'num_debug_attempts': len(learning_data.get('debug_results', [])),
            'success_rate': learning_data.get('performance_metrics', {}).get('success_rate', 0.0),
            'best_fitness': learning_data.get('performance_metrics', {}).get('best_fitness', 0.0),
            'generation_count': learning_data.get('performance_metrics', {}).get('generations', 0),
            'success_indicators': learning_data.get('success_indicators', {})
        }
        
        self.development_history.append(cycle_record)
        
        # Keep only recent history (last 1000 cycles)
        if len(self.development_history) > 1000:
            self.development_history = self.development_history[-1000:]
    
    async def _analyze_patterns(self, learning_data: Dict[str, Any]) -> List[LearningPattern]:
        """Analyze learning data to identify patterns."""
        new_patterns = []
        
        # Analyze success patterns
        success_patterns = self._analyze_success_patterns(learning_data)
        new_patterns.extend(success_patterns)
        
        # Analyze failure patterns
        failure_patterns = self._analyze_failure_patterns(learning_data)
        new_patterns.extend(failure_patterns)
        
        # Analyze optimization patterns
        optimization_patterns = self._analyze_optimization_patterns(learning_data)
        new_patterns.extend(optimization_patterns)
        
        # Analyze error prevention patterns
        error_prevention_patterns = self._analyze_error_prevention_patterns(learning_data)
        new_patterns.extend(error_prevention_patterns)
        
        return new_patterns
    
    def _analyze_success_patterns(self, learning_data: Dict[str, Any]) -> List[LearningPattern]:
        """Analyze patterns that lead to success."""
        patterns = []
        
        successful_programs = [
            p for p in learning_data.get('programs', [])
            if p.fitness_scores.get('correctness', 0) > 0.8
        ]
        
        if not successful_programs:
            return patterns
        
        # Analyze code patterns in successful programs
        code_patterns = self._extract_code_patterns(successful_programs)
        
        for pattern_key, programs in code_patterns.items():
            if len(programs) >= 2:  # Pattern must appear in multiple programs
                pattern = LearningPattern(
                    pattern_id=f"success_code_{hash(pattern_key)}",
                    pattern_type="success",
                    context={
                        'task_type': learning_data.get('task_definition', {}).get('description', ''),
                        'code_pattern': pattern_key
                    },
                    conditions=[
                        f"code_contains:{pattern_key}",
                        "fitness_score > 0.8"
                    ],
                    outcomes={
                        'average_fitness': np.mean([p.fitness_scores.get('correctness', 0) for p in programs]),
                        'success_rate': 1.0,
                        'program_count': len(programs)
                    },
                    confidence=min(len(programs) / 5.0, 1.0),  # Higher confidence with more examples
                    frequency=len(programs)
                )
                patterns.append(pattern)
        
        # Analyze generation patterns
        if successful_programs:
            avg_generation = np.mean([p.generation for p in successful_programs])
            if avg_generation < 3:  # Early success pattern
                pattern = LearningPattern(
                    pattern_id=f"early_success_{learning_data.get('task_definition', {}).get('id', 'unknown')}",
                    pattern_type="success",
                    context={
                        'task_type': learning_data.get('task_definition', {}).get('description', ''),
                        'success_timing': 'early'
                    },
                    conditions=[
                        "generation < 3",
                        "fitness_score > 0.8"
                    ],
                    outcomes={
                        'average_generation': avg_generation,
                        'success_probability': len(successful_programs) / len(learning_data.get('programs', [1]))
                    },
                    confidence=0.8,
                    frequency=len(successful_programs)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_failure_patterns(self, learning_data: Dict[str, Any]) -> List[LearningPattern]:
        """Analyze patterns that lead to failure."""
        patterns = []
        
        failed_programs = [
            p for p in learning_data.get('programs', [])
            if p.fitness_scores.get('correctness', 0) < 0.3 or p.errors
        ]
        
        if not failed_programs:
            return patterns
        
        # Analyze error patterns
        error_types = Counter()
        for program in failed_programs:
            for error in program.errors:
                error_type = error.split(':')[0] if ':' in error else error
                error_types[error_type] += 1
        
        for error_type, count in error_types.most_common(5):
            if count >= 2:  # Pattern must appear multiple times
                pattern = LearningPattern(
                    pattern_id=f"failure_error_{hash(error_type)}",
                    pattern_type="failure",
                    context={
                        'task_type': learning_data.get('task_definition', {}).get('description', ''),
                        'error_type': error_type
                    },
                    conditions=[
                        f"error_contains:{error_type}"
                    ],
                    outcomes={
                        'failure_rate': count / len(failed_programs),
                        'error_frequency': count
                    },
                    confidence=min(count / 10.0, 1.0),
                    frequency=count
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_optimization_patterns(self, learning_data: Dict[str, Any]) -> List[LearningPattern]:
        """Analyze patterns for optimization opportunities."""
        patterns = []
        
        programs = learning_data.get('programs', [])
        if len(programs) < 5:
            return patterns
        
        # Analyze performance trends
        performance_by_generation = defaultdict(list)
        for program in programs:
            performance_by_generation[program.generation].append(
                program.fitness_scores.get('correctness', 0)
            )
        
        # Check for improvement patterns
        generations = sorted(performance_by_generation.keys())
        if len(generations) >= 3:
            early_avg = np.mean(performance_by_generation[generations[0]])
            late_avg = np.mean(performance_by_generation[generations[-1]])
            
            if late_avg > early_avg + 0.2:  # Significant improvement
                pattern = LearningPattern(
                    pattern_id=f"optimization_improvement_{learning_data.get('task_definition', {}).get('id', 'unknown')}",
                    pattern_type="optimization",
                    context={
                        'task_type': learning_data.get('task_definition', {}).get('description', ''),
                        'improvement_type': 'generational'
                    },
                    conditions=[
                        "generation > 2",
                        "fitness_improvement > 0.2"
                    ],
                    outcomes={
                        'improvement_rate': late_avg - early_avg,
                        'optimal_generations': len(generations)
                    },
                    confidence=0.7,
                    frequency=1
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_error_prevention_patterns(self, learning_data: Dict[str, Any]) -> List[LearningPattern]:
        """Analyze patterns for error prevention."""
        patterns = []
        
        debug_results = learning_data.get('debug_results', [])
        if not debug_results:
            return patterns
        
        # Analyze successful debugging patterns
        successful_debugs = [dr for dr in debug_results if dr.success]
        
        if successful_debugs:
            action_success_rates = Counter()
            for debug_result in successful_debugs:
                for action in debug_result.actions_taken:
                    action_success_rates[action.value] += 1
            
            for action, count in action_success_rates.most_common(3):
                pattern = LearningPattern(
                    pattern_id=f"error_prevention_{hash(action)}",
                    pattern_type="error_prevention",
                    context={
                        'debug_action': action
                    },
                    conditions=[
                        f"debug_action:{action}"
                    ],
                    outcomes={
                        'success_rate': count / len(debug_results),
                        'usage_frequency': count
                    },
                    confidence=min(count / 5.0, 1.0),
                    frequency=count
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_code_patterns(self, programs: List[Program]) -> Dict[str, List[Program]]:
        """Extract common code patterns from programs."""
        patterns = defaultdict(list)
        
        for program in programs:
            code_lines = program.code.split('\n')
            
            # Extract function definitions
            for line in code_lines:
                line = line.strip()
                if line.startswith('def '):
                    func_signature = line.split('(')[0] if '(' in line else line
                    patterns[f"function:{func_signature}"].append(program)
                
                # Extract common constructs
                if 'for ' in line and ' in ' in line:
                    patterns["construct:for_loop"].append(program)
                if 'if ' in line:
                    patterns["construct:conditional"].append(program)
                if 'return ' in line:
                    patterns["construct:return_statement"].append(program)
                if 'try:' in line or 'except' in line:
                    patterns["construct:error_handling"].append(program)
        
        return patterns
    
    def _update_patterns(self, new_patterns: List[LearningPattern]):
        """Update existing patterns with new observations."""
        for new_pattern in new_patterns:
            if new_pattern.pattern_id in self.patterns:
                # Update existing pattern
                existing = self.patterns[new_pattern.pattern_id]
                existing.frequency += new_pattern.frequency
                existing.last_seen = datetime.now()
                
                # Update confidence based on frequency
                existing.confidence = min(existing.frequency / 10.0, 1.0)
                
                # Update effectiveness score
                if new_pattern.outcomes.get('success_rate', 0) > 0.5:
                    existing.effectiveness_score = min(existing.effectiveness_score + 0.1, 1.0)
                else:
                    existing.effectiveness_score = max(existing.effectiveness_score - 0.05, 0.0)
            else:
                # Add new pattern
                self.patterns[new_pattern.pattern_id] = new_pattern
    
    async def _generate_insights(self) -> List[LearningInsight]:
        """Generate actionable insights from learned patterns."""
        insights = []
        
        # Insight 1: Most effective success patterns
        success_patterns = [p for p in self.patterns.values() if p.pattern_type == "success"]
        if success_patterns:
            best_success_pattern = max(success_patterns, key=lambda p: p.confidence * p.effectiveness_score)
            
            insight = LearningInsight(
                insight_id="best_success_strategy",
                insight_type="optimization",
                description=f"Most effective success pattern: {best_success_pattern.context}",
                recommended_actions=[
                    f"Prioritize code patterns that include: {best_success_pattern.context.get('code_pattern', 'N/A')}",
                    "Apply this pattern early in the evolutionary process",
                    "Use this pattern as a template for initial population generation"
                ],
                confidence=best_success_pattern.confidence,
                supporting_evidence=[
                    f"Pattern seen {best_success_pattern.frequency} times",
                    f"Average success rate: {best_success_pattern.outcomes.get('success_rate', 0):.2f}"
                ],
                applicable_contexts=[best_success_pattern.context.get('task_type', 'general')]
            )
            insights.append(insight)
        
        # Insight 2: Common failure prevention
        failure_patterns = [p for p in self.patterns.values() if p.pattern_type == "failure"]
        if failure_patterns:
            most_common_failure = max(failure_patterns, key=lambda p: p.frequency)
            
            insight = LearningInsight(
                insight_id="failure_prevention",
                insight_type="error_prevention",
                description=f"Most common failure pattern: {most_common_failure.context}",
                recommended_actions=[
                    f"Add validation to prevent: {most_common_failure.context.get('error_type', 'unknown errors')}",
                    "Implement early error detection for this pattern",
                    "Add specific test cases to catch this error type"
                ],
                confidence=most_common_failure.confidence,
                supporting_evidence=[
                    f"Error occurred {most_common_failure.frequency} times",
                    f"Failure rate: {most_common_failure.outcomes.get('failure_rate', 0):.2f}"
                ],
                applicable_contexts=[most_common_failure.context.get('task_type', 'general')]
            )
            insights.append(insight)
        
        # Insight 3: Optimization opportunities
        optimization_patterns = [p for p in self.patterns.values() if p.pattern_type == "optimization"]
        if optimization_patterns:
            best_optimization = max(optimization_patterns, key=lambda p: p.outcomes.get('improvement_rate', 0))
            
            insight = LearningInsight(
                insight_id="optimization_opportunity",
                insight_type="performance",
                description=f"Best optimization strategy: {best_optimization.context}",
                recommended_actions=[
                    f"Extend evolution to {best_optimization.outcomes.get('optimal_generations', 5)} generations",
                    "Focus on iterative improvement strategies",
                    "Monitor fitness trends for early stopping"
                ],
                confidence=best_optimization.confidence,
                supporting_evidence=[
                    f"Improvement rate: {best_optimization.outcomes.get('improvement_rate', 0):.2f}",
                    f"Optimal generations: {best_optimization.outcomes.get('optimal_generations', 0)}"
                ],
                applicable_contexts=[best_optimization.context.get('task_type', 'general')]
            )
            insights.append(insight)
        
        # Store insights
        for insight in insights:
            self.insights[insight.insight_id] = insight
        
        return insights
    
    def _update_success_metrics(self, learning_data: Dict[str, Any]):
        """Update success metrics for trend analysis."""
        performance_metrics = learning_data.get('performance_metrics', {})
        
        for metric_name, value in performance_metrics.items():
            if isinstance(value, (int, float)):
                self.success_metrics[metric_name].append(value)
                
                # Keep only recent metrics (last 100 values)
                if len(self.success_metrics[metric_name]) > 100:
                    self.success_metrics[metric_name] = self.success_metrics[metric_name][-100:]
    
    def get_recommendations(self, task_context: Dict[str, Any]) -> List[str]:
        """Get recommendations based on learned patterns."""
        recommendations = []
        
        task_type = task_context.get('description', '').lower()
        
        # Find applicable patterns
        applicable_patterns = []
        for pattern in self.patterns.values():
            pattern_task_type = pattern.context.get('task_type', '').lower()
            if pattern_task_type in task_type or task_type in pattern_task_type:
                applicable_patterns.append(pattern)
        
        # Sort by effectiveness and confidence
        applicable_patterns.sort(key=lambda p: p.confidence * p.effectiveness_score, reverse=True)
        
        # Generate recommendations from top patterns
        for pattern in applicable_patterns[:5]:
            if pattern.pattern_type == "success":
                recommendations.append(
                    f"Use successful pattern: {pattern.context.get('code_pattern', 'N/A')} "
                    f"(confidence: {pattern.confidence:.2f})"
                )
            elif pattern.pattern_type == "failure":
                recommendations.append(
                    f"Avoid failure pattern: {pattern.context.get('error_type', 'N/A')} "
                    f"(seen {pattern.frequency} times)"
                )
        
        # Add insights-based recommendations
        for insight in self.insights.values():
            if insight.confidence > 0.7:
                recommendations.extend(insight.recommended_actions[:2])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning system."""
        total_patterns = len(self.patterns)
        pattern_types = Counter(p.pattern_type for p in self.patterns.values())
        
        avg_confidence = np.mean([p.confidence for p in self.patterns.values()]) if self.patterns else 0
        avg_effectiveness = np.mean([p.effectiveness_score for p in self.patterns.values()]) if self.patterns else 0
        
        recent_cycles = len([h for h in self.development_history 
                           if datetime.fromisoformat(h['timestamp']) > datetime.now() - timedelta(days=7)])
        
        return {
            'total_patterns': total_patterns,
            'pattern_types': dict(pattern_types),
            'total_insights': len(self.insights),
            'average_pattern_confidence': avg_confidence,
            'average_pattern_effectiveness': avg_effectiveness,
            'development_cycles_recorded': len(self.development_history),
            'recent_cycles_7_days': recent_cycles,
            'success_metrics_tracked': list(self.success_metrics.keys())
        }
    
    def _load_knowledge_base(self):
        """Load knowledge base from disk."""
        kb_path = Path(self.knowledge_base_path)
        
        try:
            if (kb_path / 'patterns.pkl').exists():
                with open(kb_path / 'patterns.pkl', 'rb') as f:
                    self.patterns = pickle.load(f)
            
            if (kb_path / 'insights.pkl').exists():
                with open(kb_path / 'insights.pkl', 'rb') as f:
                    self.insights = pickle.load(f)
            
            if (kb_path / 'history.json').exists():
                with open(kb_path / 'history.json', 'r') as f:
                    self.development_history = json.load(f)
            
            if (kb_path / 'metrics.json').exists():
                with open(kb_path / 'metrics.json', 'r') as f:
                    self.success_metrics = defaultdict(list, json.load(f))
            
            logger.info(f"Loaded knowledge base: {len(self.patterns)} patterns, "
                       f"{len(self.insights)} insights, {len(self.development_history)} history records")
        
        except Exception as e:
            logger.warning(f"Could not load knowledge base: {e}")
    
    def _save_knowledge_base(self):
        """Save knowledge base to disk."""
        kb_path = Path(self.knowledge_base_path)
        kb_path.mkdir(exist_ok=True)
        
        try:
            with open(kb_path / 'patterns.pkl', 'wb') as f:
                pickle.dump(self.patterns, f)
            
            with open(kb_path / 'insights.pkl', 'wb') as f:
                pickle.dump(self.insights, f)
            
            with open(kb_path / 'history.json', 'w') as f:
                json.dump(self.development_history, f, indent=2)
            
            with open(kb_path / 'metrics.json', 'w') as f:
                json.dump(dict(self.success_metrics), f, indent=2)
            
            logger.debug("Knowledge base saved successfully")
        
        except Exception as e:
            logger.error(f"Could not save knowledge base: {e}")
    
    def reset_learning_data(self):
        """Reset all learning data (use with caution)."""
        self.patterns.clear()
        self.insights.clear()
        self.development_history.clear()
        self.success_metrics.clear()
        logger.warning("All learning data has been reset")


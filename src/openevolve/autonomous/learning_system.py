"""
Learning System for Autonomous Development.

Provides pattern recognition, continuous improvement, and adaptive
capabilities for autonomous development systems.
"""

import logging
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class LearningType(Enum):
    """Types of learning in the autonomous system."""
    PATTERN_RECOGNITION = "pattern_recognition"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ERROR_PREVENTION = "error_prevention"
    TASK_OPTIMIZATION = "task_optimization"
    CODE_QUALITY = "code_quality"
    RESOURCE_MANAGEMENT = "resource_management"


class ConfidenceLevel(Enum):
    """Confidence levels for learned patterns."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


@dataclass
class LearningPattern:
    """A learned pattern in the autonomous system."""
    id: str
    pattern_type: LearningType
    description: str
    conditions: Dict[str, Any]
    actions: List[str]
    confidence: ConfidenceLevel
    success_rate: float
    usage_count: int = 0
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningEvent:
    """An event that contributes to learning."""
    timestamp: datetime
    event_type: str
    context: Dict[str, Any]
    outcome: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback: Optional[str] = None


@dataclass
class PerformanceMetric:
    """Performance metric for learning analysis."""
    name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class LearningSystem:
    """
    Intelligent learning system for autonomous development.
    
    Provides pattern recognition, continuous improvement, and adaptive
    capabilities that help the system learn from experience and optimize
    its behavior over time.
    """
    
    def __init__(self, database_connector=None, audit_logger=None):
        self.connector = database_connector
        self.audit_logger = audit_logger
        
        # Learning data storage
        self._patterns: Dict[str, LearningPattern] = {}
        self._events: deque = deque(maxlen=10000)  # Keep last 10k events
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Learning configuration
        self._min_events_for_pattern = 5
        self._min_success_rate = 0.7
        self._pattern_confidence_threshold = 0.6
        self._learning_rate = 0.1
        
        # Performance tracking
        self._performance_baselines: Dict[str, float] = {}
        self._improvement_targets: Dict[str, float] = {}
        
        logger.info("Learning system initialized")
    
    async def initialize(self) -> None:
        """Initialize the learning system."""
        try:
            # Load existing patterns and data
            await self._load_learning_data()
            
            # Initialize performance baselines
            await self._initialize_baselines()
            
            logger.info(f"Learning system initialized with {len(self._patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to initialize learning system: {e}")
            raise
    
    async def record_event(
        self,
        event_type: str,
        context: Dict[str, Any],
        outcome: str,
        success: bool,
        metrics: Optional[Dict[str, float]] = None,
        feedback: Optional[str] = None
    ) -> None:
        """
        Record a learning event.
        
        Args:
            event_type: Type of event (e.g., "task_execution", "error_resolution")
            context: Context information about the event
            outcome: Description of the outcome
            success: Whether the event was successful
            metrics: Performance metrics associated with the event
            feedback: Optional feedback about the event
        """
        event = LearningEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            context=context,
            outcome=outcome,
            success=success,
            metrics=metrics or {},
            feedback=feedback
        )
        
        self._events.append(event)
        
        # Record metrics
        for metric_name, value in (metrics or {}).items():
            metric = PerformanceMetric(
                name=metric_name,
                value=value,
                timestamp=event.timestamp,
                context=context
            )
            self._metrics[metric_name].append(metric)
        
        # Trigger pattern analysis if we have enough events
        if len(self._events) % 100 == 0:  # Analyze every 100 events
            await self._analyze_patterns()
        
        logger.debug(f"Recorded learning event: {event_type} - {outcome}")
    
    async def learn_from_success(
        self,
        task_type: str,
        context: Dict[str, Any],
        actions_taken: List[str],
        performance_metrics: Dict[str, float]
    ) -> None:
        """
        Learn from a successful operation.
        
        Args:
            task_type: Type of task that succeeded
            context: Context in which the task was performed
            actions_taken: Actions that led to success
            performance_metrics: Performance metrics from the operation
        """
        await self.record_event(
            event_type=f"success_{task_type}",
            context={**context, "actions": actions_taken},
            outcome="success",
            success=True,
            metrics=performance_metrics
        )
        
        # Look for patterns in successful operations
        await self._identify_success_patterns(task_type, context, actions_taken, performance_metrics)
    
    async def learn_from_failure(
        self,
        task_type: str,
        context: Dict[str, Any],
        error_info: Dict[str, Any],
        attempted_solutions: List[str]
    ) -> None:
        """
        Learn from a failed operation.
        
        Args:
            task_type: Type of task that failed
            context: Context in which the task was performed
            error_info: Information about the error that occurred
            attempted_solutions: Solutions that were attempted
        """
        await self.record_event(
            event_type=f"failure_{task_type}",
            context={**context, "error": error_info, "attempted_solutions": attempted_solutions},
            outcome="failure",
            success=False
        )
        
        # Analyze failure patterns to prevent future occurrences
        await self._identify_failure_patterns(task_type, context, error_info, attempted_solutions)
    
    async def get_recommendations(
        self,
        task_type: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on learned patterns.
        
        Args:
            task_type: Type of task for which to get recommendations
            context: Current context
        
        Returns:
            List of recommendations with confidence scores
        """
        recommendations = []
        
        # Find relevant patterns
        relevant_patterns = self._find_relevant_patterns(task_type, context)
        
        for pattern in relevant_patterns:
            if pattern.confidence.value >= 2:  # Medium confidence or higher
                recommendation = {
                    "pattern_id": pattern.id,
                    "description": pattern.description,
                    "actions": pattern.actions,
                    "confidence": pattern.confidence.name,
                    "success_rate": pattern.success_rate,
                    "usage_count": pattern.usage_count,
                    "reasoning": self._explain_recommendation(pattern, context)
                }
                recommendations.append(recommendation)
        
        # Sort by confidence and success rate
        recommendations.sort(
            key=lambda x: (x["confidence"], x["success_rate"]),
            reverse=True
        )
        
        return recommendations[:5]  # Return top 5 recommendations
    
    async def optimize_performance(
        self,
        metric_name: str,
        target_improvement: float = 0.1
    ) -> Dict[str, Any]:
        """
        Analyze performance data and suggest optimizations.
        
        Args:
            metric_name: Name of the metric to optimize
            target_improvement: Target improvement percentage (0.1 = 10%)
        
        Returns:
            Optimization suggestions
        """
        if metric_name not in self._metrics:
            return {"error": f"No data available for metric: {metric_name}"}
        
        metric_data = list(self._metrics[metric_name])
        if len(metric_data) < 10:
            return {"error": f"Insufficient data for optimization: {len(metric_data)} samples"}
        
        # Analyze trends
        recent_values = [m.value for m in metric_data[-50:]]  # Last 50 values
        historical_values = [m.value for m in metric_data[:-50]] if len(metric_data) > 50 else []
        
        current_avg = statistics.mean(recent_values)
        historical_avg = statistics.mean(historical_values) if historical_values else current_avg
        
        # Calculate trend
        trend = "improving" if current_avg > historical_avg else "declining" if current_avg < historical_avg else "stable"
        
        # Find best performing contexts
        best_contexts = self._find_best_performing_contexts(metric_name, metric_data)
        
        # Generate optimization suggestions
        suggestions = []
        
        if trend == "declining":
            suggestions.append({
                "type": "trend_reversal",
                "description": f"Performance is declining. Current: {current_avg:.2f}, Historical: {historical_avg:.2f}",
                "actions": ["Review recent changes", "Check for resource constraints", "Analyze error patterns"]
            })
        
        if best_contexts:
            suggestions.append({
                "type": "context_optimization",
                "description": "Apply successful context patterns",
                "best_contexts": best_contexts[:3],
                "actions": ["Replicate successful conditions", "Standardize best practices"]
            })
        
        # Calculate potential improvement
        best_values = sorted(recent_values, reverse=True)[:10]  # Top 10 values
        potential_target = statistics.mean(best_values) if best_values else current_avg
        potential_improvement = (potential_target - current_avg) / current_avg if current_avg > 0 else 0
        
        return {
            "metric_name": metric_name,
            "current_performance": current_avg,
            "trend": trend,
            "potential_improvement": potential_improvement,
            "target_value": current_avg * (1 + target_improvement),
            "suggestions": suggestions,
            "data_points": len(metric_data)
        }
    
    async def get_learning_insights(self, days: int = 7) -> Dict[str, Any]:
        """
        Get insights about the learning system's performance.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Learning insights and statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_events = [e for e in self._events if e.timestamp >= cutoff_date]
        
        # Event statistics
        total_events = len(recent_events)
        successful_events = len([e for e in recent_events if e.success])
        success_rate = (successful_events / total_events * 100) if total_events > 0 else 0
        
        # Event type distribution
        event_types = defaultdict(int)
        for event in recent_events:
            event_types[event.event_type] += 1
        
        # Pattern usage statistics
        pattern_usage = {}
        for pattern in self._patterns.values():
            if pattern.last_used and pattern.last_used >= cutoff_date:
                pattern_usage[pattern.id] = {
                    "description": pattern.description,
                    "usage_count": pattern.usage_count,
                    "success_rate": pattern.success_rate,
                    "confidence": pattern.confidence.name
                }
        
        # Learning effectiveness
        new_patterns = len([p for p in self._patterns.values() if p.created_at >= cutoff_date])
        high_confidence_patterns = len([p for p in self._patterns.values() if p.confidence.value >= 3])
        
        # Performance trends
        performance_trends = {}
        for metric_name, metric_data in self._metrics.items():
            recent_metrics = [m for m in metric_data if m.timestamp >= cutoff_date]
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                performance_trends[metric_name] = {
                    "average": statistics.mean(values),
                    "trend": "improving" if len(values) > 1 and values[-1] > values[0] else "stable",
                    "data_points": len(values)
                }
        
        return {
            "period_days": days,
            "total_events": total_events,
            "success_rate": success_rate,
            "event_types": dict(event_types),
            "pattern_usage": pattern_usage,
            "new_patterns": new_patterns,
            "high_confidence_patterns": high_confidence_patterns,
            "total_patterns": len(self._patterns),
            "performance_trends": performance_trends,
            "learning_effectiveness": {
                "pattern_creation_rate": new_patterns / days,
                "pattern_confidence_avg": statistics.mean([p.confidence.value for p in self._patterns.values()]) if self._patterns else 0,
                "pattern_success_rate_avg": statistics.mean([p.success_rate for p in self._patterns.values()]) if self._patterns else 0
            }
        }
    
    async def _analyze_patterns(self) -> None:
        """Analyze recent events to identify new patterns."""
        # Group events by type and context similarity
        event_groups = self._group_similar_events()
        
        for group_key, events in event_groups.items():
            if len(events) >= self._min_events_for_pattern:
                await self._extract_pattern_from_events(group_key, events)
    
    def _group_similar_events(self) -> Dict[str, List[LearningEvent]]:
        """Group similar events together for pattern analysis."""
        groups = defaultdict(list)
        
        for event in list(self._events)[-1000:]:  # Analyze last 1000 events
            # Create a key based on event type and key context elements
            context_key = self._create_context_key(event.context)
            group_key = f"{event.event_type}:{context_key}"
            groups[group_key].append(event)
        
        return groups
    
    def _create_context_key(self, context: Dict[str, Any]) -> str:
        """Create a key from context for grouping similar events."""
        # Extract key context elements (customize based on your needs)
        key_elements = []
        
        for key in ["task_type", "file_type", "complexity", "size"]:
            if key in context:
                key_elements.append(f"{key}:{context[key]}")
        
        return "|".join(sorted(key_elements))
    
    async def _extract_pattern_from_events(
        self,
        group_key: str,
        events: List[LearningEvent]
    ) -> None:
        """Extract a learning pattern from a group of similar events."""
        successful_events = [e for e in events if e.success]
        success_rate = len(successful_events) / len(events)
        
        if success_rate < self._min_success_rate:
            return  # Not successful enough to create a pattern
        
        # Analyze successful events to extract common actions and conditions
        common_conditions = self._find_common_conditions(successful_events)
        common_actions = self._find_common_actions(successful_events)
        
        if not common_conditions and not common_actions:
            return  # No clear pattern found
        
        # Determine pattern type
        pattern_type = self._determine_pattern_type(group_key, events)
        
        # Calculate confidence based on success rate and sample size
        confidence = self._calculate_confidence(success_rate, len(events))
        
        # Create pattern
        pattern_id = f"pattern_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(group_key) % 10000}"
        
        pattern = LearningPattern(
            id=pattern_id,
            pattern_type=pattern_type,
            description=f"Pattern for {group_key} with {success_rate:.1%} success rate",
            conditions=common_conditions,
            actions=common_actions,
            confidence=confidence,
            success_rate=success_rate,
            metadata={
                "sample_size": len(events),
                "group_key": group_key,
                "extracted_at": datetime.utcnow().isoformat()
            }
        )
        
        self._patterns[pattern_id] = pattern
        
        logger.info(f"Extracted new pattern: {pattern_id} with {success_rate:.1%} success rate")
    
    def _find_common_conditions(self, events: List[LearningEvent]) -> Dict[str, Any]:
        """Find common conditions across successful events."""
        if not events:
            return {}
        
        # Count occurrences of each condition
        condition_counts = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            for key, value in event.context.items():
                if isinstance(value, (str, int, float, bool)):
                    condition_counts[key][value] += 1
        
        # Find conditions that appear in most events
        common_conditions = {}
        threshold = len(events) * 0.7  # Must appear in 70% of events
        
        for key, value_counts in condition_counts.items():
            for value, count in value_counts.items():
                if count >= threshold:
                    common_conditions[key] = value
        
        return common_conditions
    
    def _find_common_actions(self, events: List[LearningEvent]) -> List[str]:
        """Find common actions across successful events."""
        if not events:
            return []
        
        # Extract actions from context
        all_actions = []
        for event in events:
            if "actions" in event.context:
                actions = event.context["actions"]
                if isinstance(actions, list):
                    all_actions.extend(actions)
                elif isinstance(actions, str):
                    all_actions.append(actions)
        
        # Count action occurrences
        action_counts = defaultdict(int)
        for action in all_actions:
            action_counts[action] += 1
        
        # Find actions that appear frequently
        threshold = len(events) * 0.5  # Must appear in 50% of events
        common_actions = [
            action for action, count in action_counts.items()
            if count >= threshold
        ]
        
        return common_actions
    
    def _determine_pattern_type(self, group_key: str, events: List[LearningEvent]) -> LearningType:
        """Determine the type of pattern based on the events."""
        if "error" in group_key.lower() or "failure" in group_key.lower():
            return LearningType.ERROR_PREVENTION
        elif "performance" in group_key.lower() or "optimization" in group_key.lower():
            return LearningType.PERFORMANCE_OPTIMIZATION
        elif "task" in group_key.lower():
            return LearningType.TASK_OPTIMIZATION
        elif "code" in group_key.lower() or "quality" in group_key.lower():
            return LearningType.CODE_QUALITY
        else:
            return LearningType.PATTERN_RECOGNITION
    
    def _calculate_confidence(self, success_rate: float, sample_size: int) -> ConfidenceLevel:
        """Calculate confidence level based on success rate and sample size."""
        # Adjust confidence based on both success rate and sample size
        confidence_score = success_rate * min(sample_size / 20, 1.0)  # Max boost at 20 samples
        
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _find_relevant_patterns(
        self,
        task_type: str,
        context: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Find patterns relevant to the current task and context."""
        relevant_patterns = []
        
        for pattern in self._patterns.values():
            relevance_score = self._calculate_pattern_relevance(pattern, task_type, context)
            if relevance_score > self._pattern_confidence_threshold:
                relevant_patterns.append(pattern)
        
        # Sort by relevance and confidence
        relevant_patterns.sort(
            key=lambda p: (self._calculate_pattern_relevance(p, task_type, context), p.confidence.value),
            reverse=True
        )
        
        return relevant_patterns
    
    def _calculate_pattern_relevance(
        self,
        pattern: LearningPattern,
        task_type: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate how relevant a pattern is to the current situation."""
        relevance = 0.0
        
        # Check if pattern conditions match current context
        matching_conditions = 0
        total_conditions = len(pattern.conditions)
        
        if total_conditions > 0:
            for key, value in pattern.conditions.items():
                if key in context and context[key] == value:
                    matching_conditions += 1
            
            relevance += (matching_conditions / total_conditions) * 0.7
        
        # Check task type similarity
        if task_type in pattern.description.lower():
            relevance += 0.3
        
        return min(relevance, 1.0)
    
    def _explain_recommendation(
        self,
        pattern: LearningPattern,
        context: Dict[str, Any]
    ) -> str:
        """Generate an explanation for why a pattern is recommended."""
        explanations = []
        
        # Explain based on success rate
        explanations.append(f"This approach has a {pattern.success_rate:.1%} success rate")
        
        # Explain based on usage
        if pattern.usage_count > 10:
            explanations.append(f"Successfully used {pattern.usage_count} times")
        
        # Explain based on matching conditions
        matching_conditions = sum(
            1 for key, value in pattern.conditions.items()
            if key in context and context[key] == value
        )
        if matching_conditions > 0:
            explanations.append(f"Matches {matching_conditions} contextual conditions")
        
        return ". ".join(explanations)
    
    def _find_best_performing_contexts(
        self,
        metric_name: str,
        metric_data: List[PerformanceMetric]
    ) -> List[Dict[str, Any]]:
        """Find contexts that correlate with best performance."""
        # Group metrics by context and calculate averages
        context_performance = defaultdict(list)
        
        for metric in metric_data:
            context_key = self._create_context_key(metric.context)
            context_performance[context_key].append(metric.value)
        
        # Calculate average performance for each context
        context_averages = {}
        for context_key, values in context_performance.items():
            if len(values) >= 3:  # Need at least 3 samples
                context_averages[context_key] = statistics.mean(values)
        
        # Sort by performance
        sorted_contexts = sorted(
            context_averages.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top performing contexts with details
        best_contexts = []
        for context_key, avg_performance in sorted_contexts[:5]:
            best_contexts.append({
                "context": context_key,
                "average_performance": avg_performance,
                "sample_size": len(context_performance[context_key])
            })
        
        return best_contexts
    
    async def _identify_success_patterns(
        self,
        task_type: str,
        context: Dict[str, Any],
        actions_taken: List[str],
        performance_metrics: Dict[str, float]
    ) -> None:
        """Identify patterns from successful operations."""
        # This is handled by the general pattern analysis
        # Could add specific success pattern logic here
        pass
    
    async def _identify_failure_patterns(
        self,
        task_type: str,
        context: Dict[str, Any],
        error_info: Dict[str, Any],
        attempted_solutions: List[str]
    ) -> None:
        """Identify patterns from failed operations to prevent future failures."""
        # This is handled by the general pattern analysis
        # Could add specific failure pattern logic here
        pass
    
    async def _load_learning_data(self) -> None:
        """Load learning data from storage."""
        # Placeholder - would load from database in real implementation
        pass
    
    async def _initialize_baselines(self) -> None:
        """Initialize performance baselines."""
        # Placeholder - would initialize from historical data
        pass


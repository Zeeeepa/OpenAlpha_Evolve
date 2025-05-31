"""
Recommendation Engine for intelligent code suggestions and optimizations.

Generates intelligent recommendations based on analysis results including:
- Code optimizations
- Refactoring suggestions
- Best practice recommendations
- Performance improvements
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter

from ..core.interfaces import (
    RecommendationEngineInterface, AnalysisResult, Recommendation,
    ImpactAnalysis, DependencyGraph, CodeContext, AnalysisConfig
)


logger = logging.getLogger(__name__)


class RecommendationEngine(RecommendationEngineInterface):
    """
    Engine for generating intelligent recommendations based on code analysis.
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the recommendation engine."""
        self.config = config
        
        # Define recommendation rules and patterns
        self.optimization_rules = {
            'high_complexity': {
                'threshold': 10.0,
                'recommendation': 'Consider breaking down complex functions into smaller, more manageable pieces',
                'priority': 4,
                'type': 'refactoring'
            },
            'long_functions': {
                'threshold': 50,  # lines
                'recommendation': 'Function is too long. Consider extracting smaller functions',
                'priority': 3,
                'type': 'refactoring'
            },
            'too_many_parameters': {
                'threshold': 5,
                'recommendation': 'Function has too many parameters. Consider using a configuration object',
                'priority': 3,
                'type': 'refactoring'
            },
            'deep_nesting': {
                'threshold': 4,
                'recommendation': 'Deep nesting detected. Consider early returns or guard clauses',
                'priority': 3,
                'type': 'refactoring'
            },
            'low_comment_ratio': {
                'threshold': 0.1,
                'recommendation': 'Low comment ratio. Consider adding more documentation',
                'priority': 2,
                'type': 'documentation'
            },
            'no_error_handling': {
                'patterns': ['try', 'except', 'catch', 'throw'],
                'recommendation': 'Consider adding proper error handling',
                'priority': 4,
                'type': 'reliability'
            }
        }
        
        # Performance optimization patterns
        self.performance_patterns = {
            'inefficient_loops': {
                'patterns': [r'for.*in.*range\(len\(', r'while.*len\('],
                'recommendation': 'Consider using more efficient iteration patterns',
                'priority': 3
            },
            'string_concatenation': {
                'patterns': [r'\+.*\+.*\+', r'str\s*\+='],
                'recommendation': 'Consider using join() for multiple string concatenations',
                'priority': 2
            },
            'repeated_calculations': {
                'patterns': [r'len\([^)]+\).*len\([^)]+\)'],
                'recommendation': 'Consider caching repeated calculations',
                'priority': 2
            }
        }
        
        # Security patterns
        self.security_patterns = {
            'sql_injection': {
                'patterns': [r'execute\s*\(\s*["\'].*%', r'query\s*\+'],
                'recommendation': 'Potential SQL injection vulnerability. Use parameterized queries',
                'priority': 5
            },
            'hardcoded_secrets': {
                'patterns': [r'password\s*=\s*["\'][^"\']+["\']', r'api_key\s*=\s*["\'][^"\']+["\']'],
                'recommendation': 'Hardcoded secrets detected. Use environment variables or secure storage',
                'priority': 5
            },
            'unsafe_eval': {
                'patterns': [r'\beval\s*\(', r'\bexec\s*\('],
                'recommendation': 'Unsafe eval/exec usage. Consider safer alternatives',
                'priority': 5
            }
        }
        
        logger.info("Recommendation engine initialized")
    
    async def generate_recommendations(self, analysis_result: AnalysisResult) -> List[Recommendation]:
        """
        Generate comprehensive recommendations based on analysis results.
        
        Args:
            analysis_result: Complete analysis result
            
        Returns:
            List of prioritized recommendations
        """
        logger.debug(f"Generating recommendations for analysis {analysis_result.analysis_id}")
        
        try:
            recommendations = []
            
            # Generate recommendations for each code context
            for code_context in analysis_result.code_contexts:
                context_recommendations = await self._generate_context_recommendations(code_context)
                recommendations.extend(context_recommendations)
            
            # Generate dependency-based recommendations
            if analysis_result.dependency_graph:
                dependency_recommendations = await self._generate_dependency_recommendations(
                    analysis_result.dependency_graph
                )
                recommendations.extend(dependency_recommendations)
            
            # Generate metric-based recommendations
            if analysis_result.metrics:
                metric_recommendations = await self._generate_metric_recommendations(
                    analysis_result.metrics
                )
                recommendations.extend(metric_recommendations)
            
            # Generate pattern-based recommendations
            pattern_recommendations = await self._generate_pattern_recommendations(
                analysis_result.code_contexts
            )
            recommendations.extend(pattern_recommendations)
            
            # Remove duplicates and sort by priority
            recommendations = await self._deduplicate_recommendations(recommendations)
            recommendations.sort(key=lambda x: x.priority, reverse=True)
            
            logger.debug(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed", exc_info=e)
            return []
    
    async def analyze_impact(self, changes: List[str], 
                           dependency_graph: DependencyGraph) -> ImpactAnalysis:
        """
        Analyze the impact of proposed changes.
        
        Args:
            changes: List of proposed changes
            dependency_graph: Code dependency graph
            
        Returns:
            Impact analysis results
        """
        logger.debug("Analyzing impact of proposed changes")
        
        try:
            change_id = f"impact_{int(time.time())}"
            
            affected_files = set()
            affected_functions = set()
            affected_classes = set()
            risk_level = "low"
            confidence_score = 0.8
            
            # Analyze each change
            for change in changes:
                change_lower = change.lower()
                
                # Extract file references
                if any(ext in change_lower for ext in ['.py', '.js', '.ts', '.java', '.cpp']):
                    # Extract file paths
                    import re
                    file_matches = re.findall(r'[\w/]+\.\w+', change)
                    affected_files.update(file_matches)
                
                # Extract function/class references
                func_matches = re.findall(r'\b(?:function|def|class)\s+(\w+)', change_lower)
                affected_functions.update(func_matches)
                
                # Assess risk level
                if any(keyword in change_lower for keyword in ['delete', 'remove', 'drop']):
                    risk_level = "high"
                elif any(keyword in change_lower for keyword in ['modify', 'change', 'update']):
                    if risk_level == "low":
                        risk_level = "medium"
            
            # Use dependency graph to find additional impacts
            if dependency_graph:
                for file_path in list(affected_files):
                    # Find nodes that depend on this file
                    for node_id, node in dependency_graph.nodes.items():
                        if node.file_path == file_path:
                            dependents = dependency_graph.get_dependents(node_id)
                            for dependent_id in dependents:
                                dependent_node = dependency_graph.nodes.get(dependent_id)
                                if dependent_node and dependent_node.file_path:
                                    affected_files.add(dependent_node.file_path)
            
            # Generate recommendations
            recommendations = []
            if risk_level == "high":
                recommendations.append("Conduct thorough testing before deployment")
                recommendations.append("Consider implementing feature flags for gradual rollout")
                recommendations.append("Ensure proper backup and rollback procedures")
            elif risk_level == "medium":
                recommendations.append("Review changes with team members")
                recommendations.append("Run comprehensive test suite")
            
            recommendations.append("Update documentation if interfaces change")
            
            # Generate test suggestions
            test_suggestions = []
            if affected_functions:
                test_suggestions.append("Add unit tests for modified functions")
            if len(affected_files) > 1:
                test_suggestions.append("Add integration tests for cross-file interactions")
            if risk_level in ["medium", "high"]:
                test_suggestions.append("Add regression tests to prevent future issues")
            
            return ImpactAnalysis(
                change_id=change_id,
                affected_files=list(affected_files),
                affected_functions=list(affected_functions),
                affected_classes=list(affected_classes),
                risk_level=risk_level,
                confidence_score=confidence_score,
                recommendations=recommendations,
                test_suggestions=test_suggestions,
                metadata={
                    'total_changes': len(changes),
                    'analysis_timestamp': time.time()
                }
            )
            
        except Exception as e:
            logger.error(f"Impact analysis failed", exc_info=e)
            return ImpactAnalysis(
                change_id=f"error_{int(time.time())}",
                risk_level="unknown",
                confidence_score=0.0,
                recommendations=["Impact analysis failed - proceed with caution"],
                metadata={'error': str(e)}
            )
    
    async def suggest_optimizations(self, code_context: CodeContext) -> List[Recommendation]:
        """
        Suggest specific optimizations for a code context.
        
        Args:
            code_context: Code context to optimize
            
        Returns:
            List of optimization recommendations
        """
        logger.debug(f"Suggesting optimizations for {code_context.file_path}")
        
        try:
            recommendations = []
            
            # Performance optimizations
            perf_recommendations = await self._suggest_performance_optimizations(code_context)
            recommendations.extend(perf_recommendations)
            
            # Memory optimizations
            memory_recommendations = await self._suggest_memory_optimizations(code_context)
            recommendations.extend(memory_recommendations)
            
            # Code quality optimizations
            quality_recommendations = await self._suggest_quality_optimizations(code_context)
            recommendations.extend(quality_recommendations)
            
            # Security optimizations
            security_recommendations = await self._suggest_security_optimizations(code_context)
            recommendations.extend(security_recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Optimization suggestion failed", exc_info=e)
            return []
    
    async def _generate_context_recommendations(self, code_context: CodeContext) -> List[Recommendation]:
        """Generate recommendations for a specific code context."""
        recommendations = []
        
        try:
            # Check complexity metrics
            if code_context.complexity_metrics:
                complexity = code_context.complexity_metrics.get('cyclomatic_complexity', 0)
                if complexity > self.optimization_rules['high_complexity']['threshold']:
                    rec = await self._create_recommendation(
                        'high_complexity',
                        self.optimization_rules['high_complexity'],
                        code_context.file_path
                    )
                    recommendations.append(rec)
                
                # Check nesting depth
                nesting = code_context.complexity_metrics.get('max_nesting_depth', 0)
                if nesting > self.optimization_rules['deep_nesting']['threshold']:
                    rec = await self._create_recommendation(
                        'deep_nesting',
                        self.optimization_rules['deep_nesting'],
                        code_context.file_path
                    )
                    recommendations.append(rec)
            
            # Check quality metrics
            if code_context.quality_metrics:
                comment_ratio = code_context.quality_metrics.get('comment_ratio', 0)
                if comment_ratio < self.optimization_rules['low_comment_ratio']['threshold']:
                    rec = await self._create_recommendation(
                        'low_comment_ratio',
                        self.optimization_rules['low_comment_ratio'],
                        code_context.file_path
                    )
                    recommendations.append(rec)
            
            # Check function metrics
            for element in code_context.elements:
                if element.type == 'function':
                    # Check function length
                    func_length = element.end_line - element.start_line
                    if func_length > self.optimization_rules['long_functions']['threshold']:
                        rec = await self._create_recommendation(
                            f'long_function_{element.name}',
                            self.optimization_rules['long_functions'],
                            code_context.file_path,
                            f"Function '{element.name}' is {func_length} lines long"
                        )
                        recommendations.append(rec)
                    
                    # Check parameter count
                    if 'args' in element.metadata:
                        param_count = len(element.metadata['args'])
                        if param_count > self.optimization_rules['too_many_parameters']['threshold']:
                            rec = await self._create_recommendation(
                                f'too_many_params_{element.name}',
                                self.optimization_rules['too_many_parameters'],
                                code_context.file_path,
                                f"Function '{element.name}' has {param_count} parameters"
                            )
                            recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Context recommendation generation failed", exc_info=e)
            return []
    
    async def _generate_dependency_recommendations(self, dependency_graph: DependencyGraph) -> List[Recommendation]:
        """Generate recommendations based on dependency analysis."""
        recommendations = []
        
        try:
            # Analyze circular dependencies
            circular_deps = await self._detect_circular_dependencies(dependency_graph)
            if circular_deps:
                rec = Recommendation(
                    id=f"circular_deps_{int(time.time())}",
                    type="architecture",
                    title="Circular Dependencies Detected",
                    description=f"Found {len(circular_deps)} circular dependencies that should be resolved",
                    priority=4,
                    confidence=0.9,
                    affected_files=[],
                    suggested_changes=[
                        "Refactor code to eliminate circular dependencies",
                        "Consider using dependency injection",
                        "Extract common functionality to shared modules"
                    ],
                    rationale="Circular dependencies make code harder to test and maintain"
                )
                recommendations.append(rec)
            
            # Analyze highly coupled modules
            high_coupling = await self._detect_high_coupling(dependency_graph)
            if high_coupling:
                rec = Recommendation(
                    id=f"high_coupling_{int(time.time())}",
                    type="architecture",
                    title="High Coupling Detected",
                    description="Some modules are highly coupled and may benefit from refactoring",
                    priority=3,
                    confidence=0.8,
                    affected_files=high_coupling,
                    suggested_changes=[
                        "Reduce coupling between modules",
                        "Use interfaces to decouple implementations",
                        "Apply single responsibility principle"
                    ],
                    rationale="High coupling makes code harder to modify and test"
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Dependency recommendation generation failed", exc_info=e)
            return []
    
    async def _generate_metric_recommendations(self, metrics: Dict[str, Any]) -> List[Recommendation]:
        """Generate recommendations based on overall metrics."""
        recommendations = []
        
        try:
            # Check overall complexity
            overall_complexity = metrics.get('overall_complexity', 0)
            if overall_complexity > 0.7:
                rec = Recommendation(
                    id=f"overall_complexity_{int(time.time())}",
                    type="refactoring",
                    title="High Overall Complexity",
                    description="The codebase has high overall complexity",
                    priority=3,
                    confidence=0.8,
                    suggested_changes=[
                        "Identify and refactor the most complex functions",
                        "Break down large classes and functions",
                        "Improve code organization and structure"
                    ],
                    rationale="High complexity makes code harder to understand and maintain"
                )
                recommendations.append(rec)
            
            # Check function count vs complexity
            function_count = metrics.get('total_functions', 0)
            if function_count > 50 and overall_complexity > 0.5:
                rec = Recommendation(
                    id=f"large_codebase_{int(time.time())}",
                    type="architecture",
                    title="Large Codebase Management",
                    description="Large codebase with moderate complexity may benefit from better organization",
                    priority=2,
                    confidence=0.7,
                    suggested_changes=[
                        "Consider splitting into smaller modules",
                        "Implement clear architectural boundaries",
                        "Add comprehensive documentation"
                    ],
                    rationale="Large codebases need good organization to remain maintainable"
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Metric recommendation generation failed", exc_info=e)
            return []
    
    async def _generate_pattern_recommendations(self, code_contexts: List[CodeContext]) -> List[Recommendation]:
        """Generate recommendations based on code patterns."""
        recommendations = []
        
        try:
            all_patterns = []
            for context in code_contexts:
                all_patterns.extend(context.patterns)
            
            pattern_counts = Counter(all_patterns)
            
            # Check for missing patterns
            if 'error_handling' not in pattern_counts:
                rec = Recommendation(
                    id=f"missing_error_handling_{int(time.time())}",
                    type="reliability",
                    title="Missing Error Handling",
                    description="No error handling patterns detected in the codebase",
                    priority=4,
                    confidence=0.8,
                    suggested_changes=[
                        "Add try-catch blocks for error-prone operations",
                        "Implement proper error logging",
                        "Add input validation"
                    ],
                    rationale="Proper error handling improves application reliability"
                )
                recommendations.append(rec)
            
            if 'logging' not in pattern_counts:
                rec = Recommendation(
                    id=f"missing_logging_{int(time.time())}",
                    type="observability",
                    title="Missing Logging",
                    description="No logging patterns detected in the codebase",
                    priority=3,
                    confidence=0.8,
                    suggested_changes=[
                        "Add logging statements for important operations",
                        "Implement structured logging",
                        "Add error and debug logging"
                    ],
                    rationale="Logging is essential for debugging and monitoring"
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Pattern recommendation generation failed", exc_info=e)
            return []
    
    async def _suggest_performance_optimizations(self, code_context: CodeContext) -> List[Recommendation]:
        """Suggest performance-specific optimizations."""
        recommendations = []
        
        try:
            content = code_context.content.lower()
            
            # Check for performance anti-patterns
            import re
            
            for pattern_name, pattern_info in self.performance_patterns.items():
                for pattern in pattern_info['patterns']:
                    if re.search(pattern, content):
                        rec = Recommendation(
                            id=f"perf_{pattern_name}_{int(time.time())}",
                            type="performance",
                            title=f"Performance Issue: {pattern_name.replace('_', ' ').title()}",
                            description=pattern_info['recommendation'],
                            priority=pattern_info['priority'],
                            confidence=0.7,
                            affected_files=[code_context.file_path],
                            suggested_changes=[pattern_info['recommendation']],
                            rationale="Performance optimization can improve user experience"
                        )
                        recommendations.append(rec)
                        break  # Only add one recommendation per pattern type
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Performance optimization suggestion failed", exc_info=e)
            return []
    
    async def _suggest_memory_optimizations(self, code_context: CodeContext) -> List[Recommendation]:
        """Suggest memory-specific optimizations."""
        recommendations = []
        
        try:
            # Simple memory optimization checks
            content = code_context.content.lower()
            
            # Check for potential memory leaks
            if 'global' in content and 'list' in content:
                rec = Recommendation(
                    id=f"memory_global_list_{int(time.time())}",
                    type="performance",
                    title="Potential Memory Issue: Global Lists",
                    description="Global lists detected that may grow unbounded",
                    priority=3,
                    confidence=0.6,
                    affected_files=[code_context.file_path],
                    suggested_changes=[
                        "Consider using local variables instead of globals",
                        "Implement proper cleanup mechanisms",
                        "Use generators for large datasets"
                    ],
                    rationale="Unbounded global collections can cause memory leaks"
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Memory optimization suggestion failed", exc_info=e)
            return []
    
    async def _suggest_quality_optimizations(self, code_context: CodeContext) -> List[Recommendation]:
        """Suggest code quality optimizations."""
        recommendations = []
        
        try:
            # Check for code quality issues
            lines = code_context.content.split('\n')
            long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 120]
            
            if long_lines:
                rec = Recommendation(
                    id=f"quality_long_lines_{int(time.time())}",
                    type="quality",
                    title="Long Lines Detected",
                    description=f"Found {len(long_lines)} lines longer than 120 characters",
                    priority=2,
                    confidence=0.9,
                    affected_files=[code_context.file_path],
                    suggested_changes=[
                        "Break long lines into multiple lines",
                        "Extract complex expressions into variables",
                        "Use proper line continuation"
                    ],
                    rationale="Shorter lines improve code readability"
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Quality optimization suggestion failed", exc_info=e)
            return []
    
    async def _suggest_security_optimizations(self, code_context: CodeContext) -> List[Recommendation]:
        """Suggest security-specific optimizations."""
        recommendations = []
        
        try:
            content = code_context.content
            
            # Check for security anti-patterns
            import re
            
            for pattern_name, pattern_info in self.security_patterns.items():
                for pattern in pattern_info['patterns']:
                    if re.search(pattern, content, re.IGNORECASE):
                        rec = Recommendation(
                            id=f"security_{pattern_name}_{int(time.time())}",
                            type="security",
                            title=f"Security Issue: {pattern_name.replace('_', ' ').title()}",
                            description=pattern_info['recommendation'],
                            priority=pattern_info['priority'],
                            confidence=0.8,
                            affected_files=[code_context.file_path],
                            suggested_changes=[pattern_info['recommendation']],
                            rationale="Security vulnerabilities can lead to data breaches"
                        )
                        recommendations.append(rec)
                        break  # Only add one recommendation per pattern type
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Security optimization suggestion failed", exc_info=e)
            return []
    
    async def _create_recommendation(self, rec_id: str, rule: Dict[str, Any], 
                                   file_path: str, custom_description: str = None) -> Recommendation:
        """Create a recommendation from a rule."""
        return Recommendation(
            id=f"{rec_id}_{int(time.time())}",
            type=rule['type'],
            title=rule['recommendation'].split('.')[0],  # First sentence as title
            description=custom_description or rule['recommendation'],
            priority=rule['priority'],
            confidence=0.8,
            affected_files=[file_path],
            suggested_changes=[rule['recommendation']],
            rationale=f"Based on {rule['type']} analysis"
        )
    
    async def _deduplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate recommendations."""
        seen_titles = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    async def _detect_circular_dependencies(self, dependency_graph: DependencyGraph) -> List[List[str]]:
        """Detect circular dependencies in the graph."""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node_id: str, path: List[str]) -> bool:
            if node_id in rec_stack:
                # Found a cycle
                cycle_start = path.index(node_id)
                cycles.append(path[cycle_start:] + [node_id])
                return True
            
            if node_id in visited:
                return False
            
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # Visit all dependencies
            dependencies = dependency_graph.get_dependencies(node_id)
            for dep_id in dependencies:
                if dfs(dep_id, path + [node_id]):
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        # Check all nodes
        for node_id in dependency_graph.nodes:
            if node_id not in visited:
                dfs(node_id, [])
        
        return cycles
    
    async def _detect_high_coupling(self, dependency_graph: DependencyGraph) -> List[str]:
        """Detect highly coupled modules."""
        coupling_scores = {}
        
        for node_id, node in dependency_graph.nodes.items():
            if node.file_path:
                # Count incoming and outgoing dependencies
                incoming = len(dependency_graph.get_dependents(node_id))
                outgoing = len(dependency_graph.get_dependencies(node_id))
                
                # Simple coupling score
                coupling_score = incoming + outgoing
                coupling_scores[node.file_path] = coupling_score
        
        # Return files with high coupling (top 20% or score > 10)
        if coupling_scores:
            threshold = max(10, sorted(coupling_scores.values())[-len(coupling_scores)//5])
            return [file_path for file_path, score in coupling_scores.items() if score >= threshold]
        
        return []


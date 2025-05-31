"""
Main Context Analysis Engine implementation.

This is the central orchestrator for all context analysis operations,
coordinating semantic analysis, requirement processing, and recommendation generation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .interfaces import (
    AnalysisResult, AnalysisType, AnalysisConfig, CodeContext, RequirementContext,
    DependencyGraph, LanguageType, ContextAnalyzerInterface
)
from ..semantic.analyzer import SemanticAnalyzer
from ..intelligence.processor import RequirementProcessor
from ..intelligence.recommender import RecommendationEngine
from ..integration.graph_sitter import GraphSitterParser
from ..utils.cache import CacheManager
from ..utils.language_detector import LanguageDetector


logger = logging.getLogger(__name__)


class ContextAnalysisEngine(ContextAnalyzerInterface):
    """
    Main Context Analysis Engine for autonomous development pipelines.
    
    Provides comprehensive analysis capabilities including:
    - Semantic code analysis
    - Requirement processing
    - Dependency mapping
    - Impact analysis
    - Intelligent recommendations
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize the Context Analysis Engine."""
        self.config = config or AnalysisConfig()
        self.semantic_analyzer = SemanticAnalyzer(self.config)
        self.requirement_processor = RequirementProcessor(self.config)
        self.recommendation_engine = RecommendationEngine(self.config)
        self.graph_sitter_parser = GraphSitterParser(self.config)
        self.cache_manager = CacheManager(self.config)
        self.language_detector = LanguageDetector()
        
        logger.info("Context Analysis Engine initialized")
    
    async def analyze(self, content: str, file_path: str, 
                     config: Optional[AnalysisConfig] = None) -> AnalysisResult:
        """
        Perform comprehensive context analysis on the provided content.
        
        Args:
            content: The source code or text content to analyze
            file_path: Path to the file being analyzed
            config: Optional configuration override
            
        Returns:
            AnalysisResult containing all analysis findings
        """
        start_time = time.time()
        analysis_config = config or self.config
        analysis_id = f"analysis_{int(start_time)}"
        
        logger.info(f"Starting analysis {analysis_id} for {file_path}")
        
        try:
            # Detect language
            language = self.language_detector.detect_language(content, file_path)
            
            if not self.supports_language(language):
                logger.warning(f"Language {language} not supported, using basic analysis")
                language = LanguageType.UNKNOWN
            
            # Create initial code context
            code_context = CodeContext(
                file_path=file_path,
                content=content,
                language=language
            )
            
            # Check cache first
            cache_key = self._generate_cache_key(content, file_path, analysis_config)
            if analysis_config.enable_caching:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    logger.info(f"Returning cached result for {analysis_id}")
                    return cached_result
            
            # Perform analysis based on enabled types
            analysis_tasks = []
            
            if AnalysisType.SEMANTIC in analysis_config.enabled_analyses:
                analysis_tasks.append(self._perform_semantic_analysis(code_context))
            
            if AnalysisType.DEPENDENCY in analysis_config.enabled_analyses:
                analysis_tasks.append(self._perform_dependency_analysis(code_context))
            
            if AnalysisType.COMPLEXITY in analysis_config.enabled_analyses:
                analysis_tasks.append(self._perform_complexity_analysis(code_context))
            
            if AnalysisType.QUALITY in analysis_config.enabled_analyses:
                analysis_tasks.append(self._perform_quality_analysis(code_context))
            
            if AnalysisType.PATTERN in analysis_config.enabled_analyses:
                analysis_tasks.append(self._perform_pattern_analysis(code_context))
            
            # Execute analysis tasks
            if analysis_config.parallel_processing and len(analysis_tasks) > 1:
                analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            else:
                analysis_results = []
                for task in analysis_tasks:
                    try:
                        result = await task
                        analysis_results.append(result)
                    except Exception as e:
                        analysis_results.append(e)
            
            # Process results and handle exceptions
            errors = []
            warnings = []
            metrics = {}
            dependency_graph = None
            
            for i, result in enumerate(analysis_results):
                if isinstance(result, Exception):
                    errors.append(f"Analysis task {i} failed: {str(result)}")
                    logger.error(f"Analysis task {i} failed", exc_info=result)
                elif isinstance(result, dict):
                    if 'dependency_graph' in result:
                        dependency_graph = result['dependency_graph']
                    if 'metrics' in result:
                        metrics.update(result['metrics'])
                    if 'errors' in result:
                        errors.extend(result['errors'])
                    if 'warnings' in result:
                        warnings.extend(result['warnings'])
            
            # Generate recommendations
            recommendations = []
            if not errors:  # Only generate recommendations if analysis succeeded
                try:
                    temp_result = AnalysisResult(
                        analysis_id=analysis_id,
                        analysis_type=AnalysisType.SEMANTIC,
                        code_contexts=[code_context],
                        dependency_graph=dependency_graph,
                        metrics=metrics
                    )
                    recommendations = await self.recommendation_engine.generate_recommendations(temp_result)
                except Exception as e:
                    errors.append(f"Recommendation generation failed: {str(e)}")
                    logger.error("Recommendation generation failed", exc_info=e)
            
            # Create final result
            execution_time = time.time() - start_time
            result = AnalysisResult(
                analysis_id=analysis_id,
                analysis_type=AnalysisType.SEMANTIC,
                code_contexts=[code_context],
                dependency_graph=dependency_graph,
                recommendations=recommendations,
                metrics=metrics,
                errors=errors,
                warnings=warnings,
                execution_time=execution_time
            )
            
            # Cache result if successful
            if analysis_config.enable_caching and not errors:
                await self.cache_manager.set(cache_key, result, analysis_config.cache_ttl)
            
            logger.info(f"Analysis {analysis_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Analysis {analysis_id} failed", exc_info=e)
            
            return AnalysisResult(
                analysis_id=analysis_id,
                analysis_type=AnalysisType.SEMANTIC,
                errors=[f"Analysis failed: {str(e)}"],
                execution_time=execution_time
            )
    
    async def analyze_multiple_files(self, file_paths: List[str], 
                                   config: Optional[AnalysisConfig] = None) -> List[AnalysisResult]:
        """
        Analyze multiple files and return combined results.
        
        Args:
            file_paths: List of file paths to analyze
            config: Optional configuration override
            
        Returns:
            List of AnalysisResult objects
        """
        analysis_config = config or self.config
        
        # Read file contents
        file_contents = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > analysis_config.max_file_size:
                        logger.warning(f"File {file_path} exceeds max size, truncating")
                        content = content[:analysis_config.max_file_size]
                    file_contents.append((content, file_path))
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                file_contents.append(("", file_path))
        
        # Analyze files
        if analysis_config.parallel_processing:
            tasks = [
                self.analyze(content, file_path, analysis_config)
                for content, file_path in file_contents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Analysis of file {file_paths[i]} failed", exc_info=result)
                    final_results.append(AnalysisResult(
                        analysis_id=f"failed_{i}",
                        analysis_type=AnalysisType.SEMANTIC,
                        errors=[f"Analysis failed: {str(result)}"]
                    ))
                else:
                    final_results.append(result)
            
            return final_results
        else:
            results = []
            for content, file_path in file_contents:
                result = await self.analyze(content, file_path, analysis_config)
                results.append(result)
            return results
    
    async def analyze_requirements(self, requirements: str) -> RequirementContext:
        """
        Analyze natural language requirements.
        
        Args:
            requirements: Natural language requirements text
            
        Returns:
            RequirementContext with analyzed requirements
        """
        return await self.requirement_processor.process_requirements(requirements)
    
    async def map_requirements_to_code(self, requirement: RequirementContext,
                                     code_contexts: List[CodeContext]) -> Dict[str, Any]:
        """
        Map requirements to existing code components.
        
        Args:
            requirement: Analyzed requirement context
            code_contexts: List of code contexts to map against
            
        Returns:
            Mapping results
        """
        return await self.requirement_processor.map_to_code(requirement, code_contexts)
    
    def supports_language(self, language: LanguageType) -> bool:
        """Check if the engine supports analysis for a specific language."""
        return language in self.config.supported_languages or language == LanguageType.UNKNOWN
    
    async def _perform_semantic_analysis(self, code_context: CodeContext) -> Dict[str, Any]:
        """Perform semantic analysis on code context."""
        try:
            # Parse with tree-sitter
            if code_context.language != LanguageType.UNKNOWN:
                tree = await self.graph_sitter_parser.parse(code_context.content, code_context.language)
                elements = await self.graph_sitter_parser.extract_elements(tree, code_context.content)
                code_context.elements = elements
            
            # Semantic analysis
            semantics = await self.semantic_analyzer.analyze_semantics(code_context)
            
            return {
                'semantics': semantics,
                'metrics': semantics.get('metrics', {}),
                'errors': [],
                'warnings': []
            }
        except Exception as e:
            return {
                'errors': [f"Semantic analysis failed: {str(e)}"],
                'warnings': [],
                'metrics': {}
            }
    
    async def _perform_dependency_analysis(self, code_context: CodeContext) -> Dict[str, Any]:
        """Perform dependency analysis."""
        try:
            if code_context.elements:
                dependency_graph = await self.graph_sitter_parser.build_dependency_graph(code_context.elements)
            else:
                dependency_graph = DependencyGraph()
            
            return {
                'dependency_graph': dependency_graph,
                'metrics': {
                    'dependency_count': len(dependency_graph.edges),
                    'node_count': len(dependency_graph.nodes)
                },
                'errors': [],
                'warnings': []
            }
        except Exception as e:
            return {
                'errors': [f"Dependency analysis failed: {str(e)}"],
                'warnings': [],
                'metrics': {}
            }
    
    async def _perform_complexity_analysis(self, code_context: CodeContext) -> Dict[str, Any]:
        """Perform complexity analysis."""
        try:
            complexity_metrics = await self.semantic_analyzer.calculate_complexity(code_context)
            code_context.complexity_metrics = complexity_metrics
            
            return {
                'metrics': complexity_metrics,
                'errors': [],
                'warnings': []
            }
        except Exception as e:
            return {
                'errors': [f"Complexity analysis failed: {str(e)}"],
                'warnings': [],
                'metrics': {}
            }
    
    async def _perform_quality_analysis(self, code_context: CodeContext) -> Dict[str, Any]:
        """Perform code quality analysis."""
        try:
            # Basic quality metrics
            lines = code_context.content.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            
            quality_metrics = {
                'total_lines': len(lines),
                'code_lines': len(non_empty_lines),
                'comment_lines': len(comment_lines),
                'comment_ratio': len(comment_lines) / max(len(non_empty_lines), 1),
                'avg_line_length': sum(len(line) for line in lines) / max(len(lines), 1)
            }
            
            code_context.quality_metrics = quality_metrics
            
            return {
                'metrics': quality_metrics,
                'errors': [],
                'warnings': []
            }
        except Exception as e:
            return {
                'errors': [f"Quality analysis failed: {str(e)}"],
                'warnings': [],
                'metrics': {}
            }
    
    async def _perform_pattern_analysis(self, code_context: CodeContext) -> Dict[str, Any]:
        """Perform pattern analysis."""
        try:
            patterns = await self.semantic_analyzer.extract_patterns(code_context)
            code_context.patterns = patterns
            
            return {
                'patterns': patterns,
                'metrics': {'pattern_count': len(patterns)},
                'errors': [],
                'warnings': []
            }
        except Exception as e:
            return {
                'errors': [f"Pattern analysis failed: {str(e)}"],
                'warnings': [],
                'metrics': {}
            }
    
    def _generate_cache_key(self, content: str, file_path: str, config: AnalysisConfig) -> str:
        """Generate a cache key for the analysis."""
        import hashlib
        
        content_hash = hashlib.md5(content.encode()).hexdigest()
        config_hash = hashlib.md5(str(config).encode()).hexdigest()
        
        return f"analysis_{content_hash}_{config_hash}_{file_path}"


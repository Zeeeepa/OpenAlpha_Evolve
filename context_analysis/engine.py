"""
Context Analysis Engine - Core component for comprehensive codebase understanding.
"""

import ast
import os
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from datetime import datetime

from core.interfaces import BaseAgent

logger = logging.getLogger(__name__)

@dataclass
class CodeContext:
    """Represents the context of a code element."""
    file_path: str
    element_type: str  # 'function', 'class', 'module', 'variable'
    name: str
    line_start: int
    line_end: int
    dependencies: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    semantic_tags: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    parent_context: Optional[str] = None
    children_contexts: List[str] = field(default_factory=list)

@dataclass
class CodebaseSnapshot:
    """Represents a snapshot of the entire codebase context."""
    timestamp: datetime
    total_files: int
    total_lines: int
    contexts: Dict[str, CodeContext] = field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    semantic_clusters: Dict[str, List[str]] = field(default_factory=dict)
    change_patterns: List[Dict[str, Any]] = field(default_factory=list)

class ContextAnalysisEngine(BaseAgent):
    """
    Advanced context analysis engine for comprehensive codebase understanding.
    Provides semantic analysis, dependency mapping, and pattern detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.codebase_root = self.config.get('codebase_root', '.')
        self.exclude_patterns = self.config.get('exclude_patterns', [
            '__pycache__', '.git', '.pytest_cache', 'node_modules', '.venv'
        ])
        self.include_extensions = self.config.get('include_extensions', ['.py'])
        self.context_cache: Dict[str, CodeContext] = {}
        self.dependency_cache: Dict[str, Set[str]] = {}
        self.last_analysis_hash: Optional[str] = None
        
    async def execute(self, target_path: Optional[str] = None) -> CodebaseSnapshot:
        """Main execution method - performs comprehensive codebase analysis."""
        logger.info("Starting comprehensive codebase analysis")
        
        target = target_path or self.codebase_root
        
        # Check if re-analysis is needed
        current_hash = self._calculate_codebase_hash(target)
        if current_hash == self.last_analysis_hash and self.context_cache:
            logger.info("Codebase unchanged, using cached analysis")
            return self._build_snapshot_from_cache()
        
        # Perform fresh analysis
        self.context_cache.clear()
        self.dependency_cache.clear()
        
        # Discover and analyze all files
        python_files = self._discover_python_files(target)
        logger.info(f"Discovered {len(python_files)} Python files for analysis")
        
        total_lines = 0
        for file_path in python_files:
            try:
                contexts, lines = await self._analyze_file(file_path)
                total_lines += lines
                for context in contexts:
                    self.context_cache[context.name] = context
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {e}")
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph()
        
        # Calculate complexity metrics
        complexity_metrics = self._calculate_complexity_metrics()
        
        # Detect semantic clusters
        semantic_clusters = self._detect_semantic_clusters()
        
        # Detect change patterns
        change_patterns = self._detect_change_patterns()
        
        snapshot = CodebaseSnapshot(
            timestamp=datetime.now(),
            total_files=len(python_files),
            total_lines=total_lines,
            contexts=self.context_cache.copy(),
            dependency_graph=dependency_graph,
            complexity_metrics=complexity_metrics,
            semantic_clusters=semantic_clusters,
            change_patterns=change_patterns
        )
        
        self.last_analysis_hash = current_hash
        logger.info(f"Codebase analysis complete: {len(self.context_cache)} contexts analyzed")
        
        return snapshot
    
    def _discover_python_files(self, root_path: str) -> List[str]:
        """Discover all Python files in the codebase."""
        python_files = []
        root = Path(root_path)
        
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.suffix in self.include_extensions:
                # Check if file should be excluded
                if any(pattern in str(file_path) for pattern in self.exclude_patterns):
                    continue
                python_files.append(str(file_path))
        
        return python_files
    
    async def _analyze_file(self, file_path: str) -> Tuple[List[CodeContext], int]:
        """Analyze a single Python file and extract contexts."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = len(content.splitlines())
            
            contexts = []
            
            # Analyze module-level context
            module_context = CodeContext(
                file_path=file_path,
                element_type='module',
                name=f"module:{os.path.basename(file_path)}",
                line_start=1,
                line_end=lines,
                docstring=ast.get_docstring(tree)
            )
            contexts.append(module_context)
            
            # Walk AST to find functions, classes, etc.
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_context = self._analyze_function(node, file_path)
                    contexts.append(func_context)
                elif isinstance(node, ast.ClassDef):
                    class_context = self._analyze_class(node, file_path)
                    contexts.append(class_context)
            
            return contexts, lines
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return [], 0
    
    def _analyze_function(self, node: ast.FunctionDef, file_path: str) -> CodeContext:
        """Analyze a function definition."""
        dependencies = []
        
        # Extract function calls and imports
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(child.func.attr)
        
        # Calculate complexity (simplified cyclomatic complexity)
        complexity = self._calculate_cyclomatic_complexity(node)
        
        # Extract semantic tags
        semantic_tags = self._extract_semantic_tags(node)
        
        return CodeContext(
            file_path=file_path,
            element_type='function',
            name=f"function:{node.name}",
            line_start=node.lineno,
            line_end=getattr(node, 'end_lineno', node.lineno),
            dependencies=list(set(dependencies)),
            complexity_score=complexity,
            semantic_tags=semantic_tags,
            docstring=ast.get_docstring(node)
        )
    
    def _analyze_class(self, node: ast.ClassDef, file_path: str) -> CodeContext:
        """Analyze a class definition."""
        dependencies = []
        
        # Extract base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                dependencies.append(base.id)
        
        # Extract method calls and attributes
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
        
        # Calculate complexity
        complexity = sum(self._calculate_cyclomatic_complexity(child) 
                        for child in node.body if isinstance(child, ast.FunctionDef))
        
        # Extract semantic tags
        semantic_tags = self._extract_semantic_tags(node)
        
        return CodeContext(
            file_path=file_path,
            element_type='class',
            name=f"class:{node.name}",
            line_start=node.lineno,
            line_end=getattr(node, 'end_lineno', node.lineno),
            dependencies=list(set(dependencies)),
            complexity_score=complexity,
            semantic_tags=semantic_tags,
            docstring=ast.get_docstring(node)
        )
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> float:
        """Calculate simplified cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return float(complexity)
    
    def _extract_semantic_tags(self, node: ast.AST) -> List[str]:
        """Extract semantic tags from AST node."""
        tags = []
        
        # Check for common patterns
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith('test_'):
                tags.append('test')
            if node.name.startswith('_'):
                tags.append('private')
            if any(isinstance(d, ast.AsyncFunctionDef) for d in [node]):
                tags.append('async')
            
            # Check for decorators
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    tags.append(f'decorator:{decorator.id}')
        
        elif isinstance(node, ast.ClassDef):
            if any(base.id == 'Exception' for base in node.bases if isinstance(base, ast.Name)):
                tags.append('exception')
            if node.name.endswith('Test'):
                tags.append('test_class')
        
        return tags
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build a dependency graph from analyzed contexts."""
        graph = {}
        
        for context_name, context in self.context_cache.items():
            graph[context_name] = []
            
            # Add dependencies to other contexts
            for dep in context.dependencies:
                for other_name in self.context_cache:
                    if dep in other_name or other_name.endswith(f":{dep}"):
                        graph[context_name].append(other_name)
        
        return graph
    
    def _calculate_complexity_metrics(self) -> Dict[str, float]:
        """Calculate various complexity metrics."""
        metrics = {}
        
        total_complexity = sum(ctx.complexity_score for ctx in self.context_cache.values())
        avg_complexity = total_complexity / len(self.context_cache) if self.context_cache else 0
        
        metrics['total_complexity'] = total_complexity
        metrics['average_complexity'] = avg_complexity
        metrics['max_complexity'] = max((ctx.complexity_score for ctx in self.context_cache.values()), default=0)
        
        # Calculate coupling metrics
        dependency_counts = [len(ctx.dependencies) for ctx in self.context_cache.values()]
        metrics['average_coupling'] = sum(dependency_counts) / len(dependency_counts) if dependency_counts else 0
        
        return metrics
    
    def _detect_semantic_clusters(self) -> Dict[str, List[str]]:
        """Detect semantic clusters in the codebase."""
        clusters = {}
        
        # Group by semantic tags
        for context_name, context in self.context_cache.items():
            for tag in context.semantic_tags:
                if tag not in clusters:
                    clusters[tag] = []
                clusters[tag].append(context_name)
        
        # Group by file location
        file_clusters = {}
        for context_name, context in self.context_cache.items():
            file_key = os.path.dirname(context.file_path)
            if file_key not in file_clusters:
                file_clusters[file_key] = []
            file_clusters[file_key].append(context_name)
        
        clusters.update({f"file:{k}": v for k, v in file_clusters.items()})
        
        return clusters
    
    def _detect_change_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns in code changes (placeholder for future implementation)."""
        # This would integrate with version control to analyze change patterns
        return []
    
    def _calculate_codebase_hash(self, root_path: str) -> str:
        """Calculate a hash of the codebase for change detection."""
        hasher = hashlib.md5()
        
        python_files = self._discover_python_files(root_path)
        for file_path in sorted(python_files):
            try:
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
            except Exception:
                continue
        
        return hasher.hexdigest()
    
    def _build_snapshot_from_cache(self) -> CodebaseSnapshot:
        """Build a snapshot from cached data."""
        return CodebaseSnapshot(
            timestamp=datetime.now(),
            total_files=len(set(ctx.file_path for ctx in self.context_cache.values())),
            total_lines=sum(ctx.line_end - ctx.line_start + 1 for ctx in self.context_cache.values()),
            contexts=self.context_cache.copy(),
            dependency_graph=self._build_dependency_graph(),
            complexity_metrics=self._calculate_complexity_metrics(),
            semantic_clusters=self._detect_semantic_clusters(),
            change_patterns=[]
        )
    
    def get_context_for_element(self, element_name: str) -> Optional[CodeContext]:
        """Get context for a specific code element."""
        return self.context_cache.get(element_name)
    
    def get_related_contexts(self, element_name: str, max_depth: int = 2) -> List[CodeContext]:
        """Get contexts related to a specific element through dependencies."""
        related = []
        visited = set()
        queue = [(element_name, 0)]
        
        while queue:
            current_name, depth = queue.pop(0)
            if current_name in visited or depth > max_depth:
                continue
            
            visited.add(current_name)
            context = self.context_cache.get(current_name)
            if context:
                related.append(context)
                
                # Add dependencies to queue
                for dep in context.dependencies:
                    for ctx_name in self.context_cache:
                        if dep in ctx_name and ctx_name not in visited:
                            queue.append((ctx_name, depth + 1))
        
        return related


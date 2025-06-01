"""
Code Analysis Engine for Autonomous Development.

Provides comprehensive code understanding, pattern recognition,
and intelligent analysis capabilities for autonomous development.
"""

import ast
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class CodeComplexity(Enum):
    """Code complexity levels."""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4
    EXTREMELY_COMPLEX = 5


class IssueType(Enum):
    """Types of code issues."""
    SYNTAX_ERROR = "syntax_error"
    LOGIC_ERROR = "logic_error"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TESTING = "testing"


@dataclass
class CodeIssue:
    """Represents a code issue found during analysis."""
    issue_type: IssueType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    file_path: str
    line_number: int
    column: Optional[int] = None
    suggestion: Optional[str] = None
    rule_id: Optional[str] = None


@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    file_path: str
    line_number: int
    parameters: List[str]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    complexity: int = 0
    lines_of_code: int = 0
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    file_path: str
    line_number: int
    base_classes: List[str]
    methods: List[FunctionInfo]
    attributes: List[str]
    docstring: Optional[str] = None
    is_abstract: bool = False


@dataclass
class ModuleInfo:
    """Information about a module."""
    name: str
    file_path: str
    imports: List[str]
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    global_variables: List[str]
    docstring: Optional[str] = None
    lines_of_code: int = 0
    complexity_score: float = 0.0


@dataclass
class CodebaseAnalysis:
    """Complete codebase analysis results."""
    total_files: int
    total_lines: int
    modules: List[ModuleInfo]
    issues: List[CodeIssue]
    complexity_distribution: Dict[CodeComplexity, int]
    language_stats: Dict[str, int]
    dependency_graph: Dict[str, List[str]]
    test_coverage: Optional[float] = None
    documentation_coverage: float = 0.0
    maintainability_index: float = 0.0


class CodeAnalyzer:
    """
    Comprehensive code analyzer for autonomous development.
    
    Provides deep code understanding, pattern recognition,
    and intelligent analysis capabilities.
    """
    
    def __init__(self, database_connector=None):
        self.connector = database_connector
        self._analysis_cache: Dict[str, Any] = {}
        self._file_hashes: Dict[str, str] = {}
        
        # Analysis rules and patterns
        self._security_patterns = [
            (r'eval\s*\(', "Dangerous use of eval()"),
            (r'exec\s*\(', "Dangerous use of exec()"),
            (r'__import__\s*\(', "Dynamic import detected"),
            (r'subprocess\.call\s*\(', "Subprocess call without shell=False"),
            (r'os\.system\s*\(', "Dangerous use of os.system()"),
        ]
        
        self._performance_patterns = [
            (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', "Use enumerate() instead of range(len())"),
            (r'\.append\s*\(\s*\)\s*in\s+for', "Consider list comprehension"),
            (r'time\.sleep\s*\(\s*0\s*\)', "Unnecessary sleep(0)"),
        ]
        
        logger.info("Code analyzer initialized")
    
    async def analyze_codebase(
        self,
        root_path: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> CodebaseAnalysis:
        """
        Perform comprehensive analysis of a codebase.
        
        Args:
            root_path: Root directory to analyze
            include_patterns: File patterns to include (e.g., ["*.py", "*.js"])
            exclude_patterns: File patterns to exclude (e.g., ["test_*", "__pycache__"])
            use_cache: Whether to use cached analysis results
        
        Returns:
            Complete codebase analysis
        """
        logger.info(f"Starting codebase analysis of {root_path}")
        
        # Default patterns
        if include_patterns is None:
            include_patterns = ["*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.c", "*.h"]
        
        if exclude_patterns is None:
            exclude_patterns = [
                "__pycache__", "*.pyc", "node_modules", ".git", 
                "*.min.js", "dist", "build", "target"
            ]
        
        # Find all relevant files
        files_to_analyze = self._find_files(root_path, include_patterns, exclude_patterns)
        
        logger.info(f"Found {len(files_to_analyze)} files to analyze")
        
        # Analyze each file
        modules = []
        all_issues = []
        total_lines = 0
        language_stats = {}
        
        for file_path in files_to_analyze:
            try:
                # Check if file has changed (for caching)
                file_hash = self._get_file_hash(file_path)
                cache_key = f"{file_path}:{file_hash}"
                
                if use_cache and cache_key in self._analysis_cache:
                    module_info = self._analysis_cache[cache_key]
                    logger.debug(f"Using cached analysis for {file_path}")
                else:
                    # Analyze the file
                    module_info = await self._analyze_file(file_path)
                    
                    # Cache the result
                    if use_cache:
                        self._analysis_cache[cache_key] = module_info
                        self._file_hashes[file_path] = file_hash
                
                if module_info:
                    modules.append(module_info)
                    total_lines += module_info.lines_of_code
                    
                    # Update language stats
                    ext = Path(file_path).suffix.lower()
                    language_stats[ext] = language_stats.get(ext, 0) + 1
                    
                    # Collect issues from this module
                    file_issues = await self._analyze_file_issues(file_path)
                    all_issues.extend(file_issues)
                
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                # Create issue for analysis failure
                all_issues.append(CodeIssue(
                    issue_type=IssueType.SYNTAX_ERROR,
                    severity="high",
                    message=f"Failed to analyze file: {e}",
                    file_path=file_path,
                    line_number=1
                ))
        
        # Calculate complexity distribution
        complexity_dist = self._calculate_complexity_distribution(modules)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(modules)
        
        # Calculate metrics
        doc_coverage = self._calculate_documentation_coverage(modules)
        maintainability = self._calculate_maintainability_index(modules, all_issues)
        
        analysis = CodebaseAnalysis(
            total_files=len(files_to_analyze),
            total_lines=total_lines,
            modules=modules,
            issues=all_issues,
            complexity_distribution=complexity_dist,
            language_stats=language_stats,
            dependency_graph=dependency_graph,
            documentation_coverage=doc_coverage,
            maintainability_index=maintainability
        )
        
        logger.info(f"Codebase analysis completed: {len(modules)} modules, {len(all_issues)} issues")
        
        return analysis
    
    async def analyze_file(self, file_path: str) -> Optional[ModuleInfo]:
        """
        Analyze a single file.
        
        Args:
            file_path: Path to the file to analyze
        
        Returns:
            Module information or None if analysis failed
        """
        return await self._analyze_file(file_path)
    
    async def suggest_improvements(self, analysis: CodebaseAnalysis) -> List[Dict[str, Any]]:
        """
        Generate improvement suggestions based on analysis.
        
        Args:
            analysis: Codebase analysis results
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # High-priority issues
        critical_issues = [i for i in analysis.issues if i.severity == "critical"]
        if critical_issues:
            suggestions.append({
                "type": "critical_issues",
                "priority": "urgent",
                "title": f"Fix {len(critical_issues)} critical issues",
                "description": "Critical issues that need immediate attention",
                "issues": critical_issues[:5]  # Top 5
            })
        
        # Complexity issues
        complex_functions = []
        for module in analysis.modules:
            for func in module.functions:
                if func.complexity > 10:
                    complex_functions.append(func)
        
        if complex_functions:
            suggestions.append({
                "type": "complexity",
                "priority": "high",
                "title": f"Reduce complexity in {len(complex_functions)} functions",
                "description": "Functions with high cyclomatic complexity",
                "functions": complex_functions[:5]
            })
        
        # Documentation coverage
        if analysis.documentation_coverage < 0.7:
            suggestions.append({
                "type": "documentation",
                "priority": "medium",
                "title": "Improve documentation coverage",
                "description": f"Current coverage: {analysis.documentation_coverage:.1%}",
                "target": "80%"
            })
        
        # Security issues
        security_issues = [i for i in analysis.issues if i.issue_type == IssueType.SECURITY]
        if security_issues:
            suggestions.append({
                "type": "security",
                "priority": "high",
                "title": f"Address {len(security_issues)} security issues",
                "description": "Security vulnerabilities found",
                "issues": security_issues
            })
        
        # Performance issues
        perf_issues = [i for i in analysis.issues if i.issue_type == IssueType.PERFORMANCE]
        if perf_issues:
            suggestions.append({
                "type": "performance",
                "priority": "medium",
                "title": f"Optimize {len(perf_issues)} performance issues",
                "description": "Performance improvements available",
                "issues": perf_issues
            })
        
        return suggestions
    
    def _find_files(
        self,
        root_path: str,
        include_patterns: List[str],
        exclude_patterns: List[str]
    ) -> List[str]:
        """Find files matching the include/exclude patterns."""
        import fnmatch
        
        files = []
        root = Path(root_path)
        
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Check exclude patterns
            relative_path = str(file_path.relative_to(root))
            if any(fnmatch.fnmatch(relative_path, pattern) for pattern in exclude_patterns):
                continue
            
            # Check include patterns
            if any(fnmatch.fnmatch(file_path.name, pattern) for pattern in include_patterns):
                files.append(str(file_path))
        
        return sorted(files)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file content for caching."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    async def _analyze_file(self, file_path: str) -> Optional[ModuleInfo]:
        """Analyze a single file and extract information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine file type and analyze accordingly
            ext = Path(file_path).suffix.lower()
            
            if ext == '.py':
                return self._analyze_python_file(file_path, content)
            else:
                # Basic analysis for other file types
                return self._analyze_generic_file(file_path, content)
                
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _analyze_python_file(self, file_path: str, content: str) -> ModuleInfo:
        """Analyze a Python file using AST."""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return ModuleInfo(
                name=Path(file_path).stem,
                file_path=file_path,
                imports=[],
                functions=[],
                classes=[],
                global_variables=[],
                lines_of_code=len(content.splitlines())
            )
        
        # Extract module information
        module_name = Path(file_path).stem
        imports = []
        functions = []
        classes = []
        global_vars = []
        docstring = None
        
        # Get module docstring
        if (tree.body and isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Str)):
            docstring = tree.body[0].value.s
        
        # Walk the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
            
            elif isinstance(node, ast.FunctionDef):
                func_info = self._extract_function_info(node, file_path)
                functions.append(func_info)
            
            elif isinstance(node, ast.AsyncFunctionDef):
                func_info = self._extract_function_info(node, file_path, is_async=True)
                functions.append(func_info)
            
            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node, file_path)
                classes.append(class_info)
            
            elif isinstance(node, ast.Assign):
                # Global variable assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        global_vars.append(target.id)
        
        lines_of_code = len([line for line in content.splitlines() if line.strip()])
        complexity_score = self._calculate_file_complexity(functions, classes)
        
        return ModuleInfo(
            name=module_name,
            file_path=file_path,
            imports=list(set(imports)),
            functions=functions,
            classes=classes,
            global_variables=global_vars,
            docstring=docstring,
            lines_of_code=lines_of_code,
            complexity_score=complexity_score
        )
    
    def _analyze_generic_file(self, file_path: str, content: str) -> ModuleInfo:
        """Basic analysis for non-Python files."""
        lines = content.splitlines()
        lines_of_code = len([line for line in lines if line.strip()])
        
        return ModuleInfo(
            name=Path(file_path).stem,
            file_path=file_path,
            imports=[],
            functions=[],
            classes=[],
            global_variables=[],
            lines_of_code=lines_of_code,
            complexity_score=0.0
        )
    
    def _extract_function_info(
        self,
        node: ast.FunctionDef,
        file_path: str,
        is_async: bool = False
    ) -> FunctionInfo:
        """Extract information from a function AST node."""
        # Get parameters
        params = [arg.arg for arg in node.args.args]
        
        # Get docstring
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            docstring = node.body[0].value.s
        
        # Calculate complexity (simplified cyclomatic complexity)
        complexity = self._calculate_function_complexity(node)
        
        # Count lines of code
        lines_of_code = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1
        
        # Get decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(decorator.attr)
        
        return FunctionInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            parameters=params,
            docstring=docstring,
            complexity=complexity,
            lines_of_code=lines_of_code,
            is_async=is_async,
            decorators=decorators
        )
    
    def _extract_class_info(self, node: ast.ClassDef, file_path: str) -> ClassInfo:
        """Extract information from a class AST node."""
        # Get base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(base.attr)
        
        # Get docstring
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            docstring = node.body[0].value.s
        
        # Extract methods
        methods = []
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item, file_path)
                methods.append(method_info)
            elif isinstance(item, ast.AsyncFunctionDef):
                method_info = self._extract_function_info(item, file_path, is_async=True)
                methods.append(method_info)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        
        # Check if abstract
        is_abstract = any(
            isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod'
            for method in methods
            for decorator in getattr(method, 'decorators', [])
        )
        
        return ClassInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            base_classes=base_classes,
            methods=methods,
            attributes=attributes,
            docstring=docstring,
            is_abstract=is_abstract
        )
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _calculate_file_complexity(
        self,
        functions: List[FunctionInfo],
        classes: List[ClassInfo]
    ) -> float:
        """Calculate overall file complexity score."""
        total_complexity = 0
        total_functions = len(functions)
        
        # Add function complexities
        for func in functions:
            total_complexity += func.complexity
        
        # Add method complexities from classes
        for cls in classes:
            for method in cls.methods:
                total_complexity += method.complexity
                total_functions += 1
        
        if total_functions == 0:
            return 0.0
        
        return total_complexity / total_functions
    
    async def _analyze_file_issues(self, file_path: str) -> List[CodeIssue]:
        """Analyze a file for various issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            
            # Check security patterns
            for i, line in enumerate(lines, 1):
                for pattern, message in self._security_patterns:
                    if re.search(pattern, line):
                        issues.append(CodeIssue(
                            issue_type=IssueType.SECURITY,
                            severity="high",
                            message=message,
                            file_path=file_path,
                            line_number=i,
                            suggestion="Review and secure this code"
                        ))
                
                # Check performance patterns
                for pattern, message in self._performance_patterns:
                    if re.search(pattern, line):
                        issues.append(CodeIssue(
                            issue_type=IssueType.PERFORMANCE,
                            severity="medium",
                            message=message,
                            file_path=file_path,
                            line_number=i
                        ))
                
                # Check line length
                if len(line) > 120:
                    issues.append(CodeIssue(
                        issue_type=IssueType.STYLE,
                        severity="low",
                        message=f"Line too long ({len(line)} characters)",
                        file_path=file_path,
                        line_number=i,
                        suggestion="Break line into multiple lines"
                    ))
        
        except Exception as e:
            logger.error(f"Failed to analyze issues in {file_path}: {e}")
        
        return issues
    
    def _calculate_complexity_distribution(
        self,
        modules: List[ModuleInfo]
    ) -> Dict[CodeComplexity, int]:
        """Calculate distribution of code complexity."""
        distribution = {complexity: 0 for complexity in CodeComplexity}
        
        for module in modules:
            for func in module.functions:
                if func.complexity <= 5:
                    distribution[CodeComplexity.SIMPLE] += 1
                elif func.complexity <= 10:
                    distribution[CodeComplexity.MODERATE] += 1
                elif func.complexity <= 15:
                    distribution[CodeComplexity.COMPLEX] += 1
                elif func.complexity <= 25:
                    distribution[CodeComplexity.VERY_COMPLEX] += 1
                else:
                    distribution[CodeComplexity.EXTREMELY_COMPLEX] += 1
            
            for cls in module.classes:
                for method in cls.methods:
                    if method.complexity <= 5:
                        distribution[CodeComplexity.SIMPLE] += 1
                    elif method.complexity <= 10:
                        distribution[CodeComplexity.MODERATE] += 1
                    elif method.complexity <= 15:
                        distribution[CodeComplexity.COMPLEX] += 1
                    elif method.complexity <= 25:
                        distribution[CodeComplexity.VERY_COMPLEX] += 1
                    else:
                        distribution[CodeComplexity.EXTREMELY_COMPLEX] += 1
        
        return distribution
    
    def _build_dependency_graph(self, modules: List[ModuleInfo]) -> Dict[str, List[str]]:
        """Build dependency graph between modules."""
        graph = {}
        
        for module in modules:
            module_name = module.name
            dependencies = []
            
            # Extract local dependencies (imports from same codebase)
            module_names = {m.name for m in modules}
            
            for import_name in module.imports:
                # Check if it's a local module
                if import_name in module_names:
                    dependencies.append(import_name)
                # Check for relative imports
                elif import_name.startswith('.'):
                    # Handle relative imports
                    base_name = import_name.lstrip('.')
                    if base_name in module_names:
                        dependencies.append(base_name)
            
            graph[module_name] = dependencies
        
        return graph
    
    def _calculate_documentation_coverage(self, modules: List[ModuleInfo]) -> float:
        """Calculate documentation coverage percentage."""
        total_items = 0
        documented_items = 0
        
        for module in modules:
            # Module docstring
            total_items += 1
            if module.docstring:
                documented_items += 1
            
            # Function docstrings
            for func in module.functions:
                total_items += 1
                if func.docstring:
                    documented_items += 1
            
            # Class and method docstrings
            for cls in module.classes:
                total_items += 1
                if cls.docstring:
                    documented_items += 1
                
                for method in cls.methods:
                    total_items += 1
                    if method.docstring:
                        documented_items += 1
        
        if total_items == 0:
            return 0.0
        
        return documented_items / total_items
    
    def _calculate_maintainability_index(
        self,
        modules: List[ModuleInfo],
        issues: List[CodeIssue]
    ) -> float:
        """Calculate maintainability index (0-100 scale)."""
        if not modules:
            return 0.0
        
        # Base score
        score = 100.0
        
        # Penalize for complexity
        total_complexity = sum(m.complexity_score for m in modules)
        avg_complexity = total_complexity / len(modules)
        score -= min(avg_complexity * 2, 30)  # Max 30 point penalty
        
        # Penalize for issues
        critical_issues = len([i for i in issues if i.severity == "critical"])
        high_issues = len([i for i in issues if i.severity == "high"])
        medium_issues = len([i for i in issues if i.severity == "medium"])
        
        score -= critical_issues * 10
        score -= high_issues * 5
        score -= medium_issues * 2
        
        # Bonus for documentation
        doc_coverage = self._calculate_documentation_coverage(modules)
        score += doc_coverage * 10  # Max 10 point bonus
        
        return max(0.0, min(100.0, score))


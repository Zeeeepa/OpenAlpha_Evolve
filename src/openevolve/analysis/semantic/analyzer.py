"""
Semantic Analyzer for code understanding and analysis.

Provides comprehensive semantic analysis including:
- Code structure analysis
- Pattern recognition
- Complexity calculation
- Quality assessment
"""

import ast
import re
import logging
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from ..core.interfaces import (
    SemanticAnalyzerInterface, CodeContext, AnalysisConfig, 
    LanguageType, CodeElement
)
from .ast_processor import ASTProcessor
from .complexity_calculator import ComplexityCalculator


logger = logging.getLogger(__name__)


class SemanticAnalyzer(SemanticAnalyzerInterface):
    """
    Semantic analyzer for code understanding and pattern recognition.
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the semantic analyzer."""
        self.config = config
        self.ast_processor = ASTProcessor(config)
        self.complexity_calculator = ComplexityCalculator(config)
        
        # Common code patterns
        self.patterns = {
            'singleton': [
                r'class\s+\w+.*:\s*\n.*__new__.*\n.*if.*not.*hasattr',
                r'_instance\s*=\s*None',
                r'if\s+cls\._instance\s+is\s+None'
            ],
            'factory': [
                r'def\s+create_\w+',
                r'def\s+make_\w+',
                r'class\s+\w*Factory'
            ],
            'observer': [
                r'def\s+notify',
                r'def\s+subscribe',
                r'def\s+unsubscribe',
                r'observers?\s*=\s*\[\]'
            ],
            'decorator': [
                r'@\w+',
                r'def\s+\w+\(.*func.*\):',
                r'return\s+wrapper'
            ],
            'context_manager': [
                r'def\s+__enter__',
                r'def\s+__exit__',
                r'with\s+\w+'
            ],
            'async_pattern': [
                r'async\s+def',
                r'await\s+',
                r'asyncio\.'
            ],
            'error_handling': [
                r'try:',
                r'except\s+\w*Error',
                r'finally:',
                r'raise\s+\w*Error'
            ],
            'logging': [
                r'logger\.',
                r'logging\.',
                r'\.log\(',
                r'\.debug\(',
                r'\.info\(',
                r'\.warning\(',
                r'\.error\('
            ]
        }
        
        logger.info("Semantic analyzer initialized")
    
    async def analyze_semantics(self, code_context: CodeContext) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis on code context.
        
        Args:
            code_context: The code context to analyze
            
        Returns:
            Dictionary containing semantic analysis results
        """
        logger.debug(f"Starting semantic analysis for {code_context.file_path}")
        
        try:
            results = {
                'structure': await self._analyze_structure(code_context),
                'patterns': await self.extract_patterns(code_context),
                'complexity': await self.calculate_complexity(code_context),
                'quality': await self._analyze_quality(code_context),
                'dependencies': await self._analyze_dependencies(code_context),
                'metrics': {}
            }
            
            # Aggregate metrics
            results['metrics'] = {
                'total_functions': len([e for e in code_context.elements if e.type == 'function']),
                'total_classes': len([e for e in code_context.elements if e.type == 'class']),
                'total_patterns': len(results['patterns']),
                'complexity_score': results['complexity'].get('cyclomatic_complexity', 0),
                'quality_score': results['quality'].get('overall_score', 0)
            }
            
            logger.debug(f"Semantic analysis completed for {code_context.file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Semantic analysis failed for {code_context.file_path}", exc_info=e)
            return {
                'error': str(e),
                'structure': {},
                'patterns': [],
                'complexity': {},
                'quality': {},
                'dependencies': {},
                'metrics': {}
            }
    
    async def extract_patterns(self, code_context: CodeContext) -> List[str]:
        """
        Extract code patterns from the context.
        
        Args:
            code_context: The code context to analyze
            
        Returns:
            List of detected patterns
        """
        detected_patterns = []
        content = code_context.content
        
        try:
            for pattern_name, pattern_regexes in self.patterns.items():
                pattern_score = 0
                for regex in pattern_regexes:
                    if re.search(regex, content, re.MULTILINE | re.IGNORECASE):
                        pattern_score += 1
                
                # If more than half the pattern indicators are found, consider it detected
                if pattern_score >= len(pattern_regexes) * 0.5:
                    detected_patterns.append(pattern_name)
            
            # Language-specific patterns
            if code_context.language == LanguageType.PYTHON:
                detected_patterns.extend(await self._extract_python_patterns(content))
            elif code_context.language == LanguageType.JAVASCRIPT:
                detected_patterns.extend(await self._extract_javascript_patterns(content))
            
            logger.debug(f"Detected patterns: {detected_patterns}")
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Pattern extraction failed", exc_info=e)
            return []
    
    async def calculate_complexity(self, code_context: CodeContext) -> Dict[str, float]:
        """
        Calculate complexity metrics for the code.
        
        Args:
            code_context: The code context to analyze
            
        Returns:
            Dictionary of complexity metrics
        """
        try:
            return await self.complexity_calculator.calculate_all_metrics(code_context)
        except Exception as e:
            logger.error(f"Complexity calculation failed", exc_info=e)
            return {}
    
    async def _analyze_structure(self, code_context: CodeContext) -> Dict[str, Any]:
        """Analyze the structural aspects of the code."""
        structure = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'constants': []
        }
        
        try:
            if code_context.language == LanguageType.PYTHON:
                structure = await self._analyze_python_structure(code_context.content)
            elif code_context.language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
                structure = await self._analyze_javascript_structure(code_context.content)
            
            return structure
            
        except Exception as e:
            logger.error(f"Structure analysis failed", exc_info=e)
            return structure
    
    async def _analyze_quality(self, code_context: CodeContext) -> Dict[str, Any]:
        """Analyze code quality metrics."""
        quality_metrics = {
            'readability_score': 0.0,
            'maintainability_score': 0.0,
            'documentation_score': 0.0,
            'overall_score': 0.0
        }
        
        try:
            content = code_context.content
            lines = content.split('\n')
            
            # Basic readability metrics
            total_lines = len(lines)
            non_empty_lines = [line for line in lines if line.strip()]
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            
            # Calculate scores
            if total_lines > 0:
                comment_ratio = len(comment_lines) / total_lines
                quality_metrics['documentation_score'] = min(comment_ratio * 5, 1.0)  # Cap at 1.0
                
                avg_line_length = sum(len(line) for line in lines) / total_lines
                readability_score = 1.0 - min(avg_line_length / 120, 1.0)  # Penalize very long lines
                quality_metrics['readability_score'] = max(readability_score, 0.0)
                
                # Maintainability based on complexity and structure
                complexity = code_context.complexity_metrics.get('cyclomatic_complexity', 0)
                maintainability = 1.0 - min(complexity / 20, 1.0)  # Penalize high complexity
                quality_metrics['maintainability_score'] = max(maintainability, 0.0)
                
                # Overall score
                quality_metrics['overall_score'] = (
                    quality_metrics['readability_score'] * 0.4 +
                    quality_metrics['maintainability_score'] * 0.4 +
                    quality_metrics['documentation_score'] * 0.2
                )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality analysis failed", exc_info=e)
            return quality_metrics
    
    async def _analyze_dependencies(self, code_context: CodeContext) -> Dict[str, Any]:
        """Analyze code dependencies."""
        dependencies = {
            'imports': [],
            'external_calls': [],
            'internal_calls': []
        }
        
        try:
            content = code_context.content
            
            # Extract imports
            import_patterns = [
                r'import\s+(\w+)',
                r'from\s+(\w+)\s+import',
                r'require\([\'"]([^\'"]+)[\'"]\)',
                r'import\s+[\'"]([^\'"]+)[\'"]'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                dependencies['imports'].extend(matches)
            
            # Extract function calls
            function_call_pattern = r'(\w+)\s*\('
            function_calls = re.findall(function_call_pattern, content)
            
            # Classify as external or internal based on common patterns
            external_indicators = ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set']
            for call in function_calls:
                if call in external_indicators or call.isupper():
                    dependencies['external_calls'].append(call)
                else:
                    dependencies['internal_calls'].append(call)
            
            # Remove duplicates
            dependencies['imports'] = list(set(dependencies['imports']))
            dependencies['external_calls'] = list(set(dependencies['external_calls']))
            dependencies['internal_calls'] = list(set(dependencies['internal_calls']))
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Dependency analysis failed", exc_info=e)
            return dependencies
    
    async def _analyze_python_structure(self, content: str) -> Dict[str, Any]:
        """Analyze Python code structure using AST."""
        structure = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'constants': []
        }
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    structure['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    })
                
                elif isinstance(node, ast.ClassDef):
                    structure['classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                        'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                    })
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        structure['imports'].append({
                            'name': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })
                
                elif isinstance(node, ast.ImportFrom):
                    structure['imports'].append({
                        'module': node.module,
                        'names': [alias.name for alias in node.names],
                        'line': node.lineno
                    })
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_info = {
                                'name': target.id,
                                'line': node.lineno
                            }
                            
                            # Check if it's a constant (uppercase)
                            if target.id.isupper():
                                structure['constants'].append(var_info)
                            else:
                                structure['variables'].append(var_info)
            
            return structure
            
        except SyntaxError as e:
            logger.warning(f"Python syntax error during structure analysis: {e}")
            return structure
        except Exception as e:
            logger.error(f"Python structure analysis failed", exc_info=e)
            return structure
    
    async def _analyze_javascript_structure(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript code structure using regex patterns."""
        structure = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'constants': []
        }
        
        try:
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Function declarations
                func_match = re.match(r'(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)', line)
                if func_match:
                    structure['functions'].append({
                        'name': func_match.group(1),
                        'line': i,
                        'args': [arg.strip() for arg in func_match.group(2).split(',') if arg.strip()],
                        'is_async': 'async' in line
                    })
                
                # Arrow functions
                arrow_match = re.match(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>', line)
                if arrow_match:
                    structure['functions'].append({
                        'name': arrow_match.group(1),
                        'line': i,
                        'args': [],
                        'is_async': 'async' in line
                    })
                
                # Class declarations
                class_match = re.match(r'class\s+(\w+)(?:\s+extends\s+(\w+))?', line)
                if class_match:
                    structure['classes'].append({
                        'name': class_match.group(1),
                        'line': i,
                        'extends': class_match.group(2)
                    })
                
                # Import statements
                import_match = re.match(r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]', line)
                if import_match:
                    structure['imports'].append({
                        'module': import_match.group(1),
                        'line': i
                    })
                
                # Variable declarations
                var_match = re.match(r'(?:const|let|var)\s+(\w+)', line)
                if var_match:
                    var_name = var_match.group(1)
                    var_info = {
                        'name': var_name,
                        'line': i
                    }
                    
                    if var_name.isupper() or 'const' in line:
                        structure['constants'].append(var_info)
                    else:
                        structure['variables'].append(var_info)
            
            return structure
            
        except Exception as e:
            logger.error(f"JavaScript structure analysis failed", exc_info=e)
            return structure
    
    async def _extract_python_patterns(self, content: str) -> List[str]:
        """Extract Python-specific patterns."""
        patterns = []
        
        # Check for common Python patterns
        if 'if __name__ == "__main__"' in content:
            patterns.append('main_guard')
        
        if re.search(r'class\s+\w+\(.*Exception.*\)', content):
            patterns.append('custom_exception')
        
        if 'yield' in content:
            patterns.append('generator')
        
        if re.search(r'@property', content):
            patterns.append('property_decorator')
        
        if re.search(r'@staticmethod|@classmethod', content):
            patterns.append('class_methods')
        
        return patterns
    
    async def _extract_javascript_patterns(self, content: str) -> List[str]:
        """Extract JavaScript-specific patterns."""
        patterns = []
        
        # Check for common JavaScript patterns
        if re.search(r'\.prototype\.', content):
            patterns.append('prototype_pattern')
        
        if re.search(r'function\s*\([^)]*\)\s*{[^}]*return\s+function', content):
            patterns.append('closure')
        
        if 'Promise' in content or '.then(' in content:
            patterns.append('promise_pattern')
        
        if re.search(r'module\.exports|export\s+', content):
            patterns.append('module_pattern')
        
        return patterns


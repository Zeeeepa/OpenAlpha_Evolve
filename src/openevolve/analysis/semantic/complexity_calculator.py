"""
Complexity Calculator for code quality assessment.

Provides various complexity metrics including:
- Cyclomatic complexity
- Cognitive complexity
- Halstead metrics
- Maintainability index
"""

import ast
import re
import math
import logging
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, Counter

from ..core.interfaces import AnalysisConfig, CodeContext, LanguageType


logger = logging.getLogger(__name__)


class ComplexityCalculator:
    """
    Calculator for various code complexity metrics.
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the complexity calculator."""
        self.config = config
        logger.debug("Complexity calculator initialized")
    
    async def calculate_all_metrics(self, code_context: CodeContext) -> Dict[str, float]:
        """
        Calculate all available complexity metrics.
        
        Args:
            code_context: The code context to analyze
            
        Returns:
            Dictionary of complexity metrics
        """
        metrics = {}
        
        try:
            if code_context.language == LanguageType.PYTHON:
                metrics.update(await self._calculate_python_metrics(code_context.content))
            elif code_context.language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
                metrics.update(await self._calculate_javascript_metrics(code_context.content))
            else:
                metrics.update(await self._calculate_generic_metrics(code_context.content))
            
            # Calculate overall complexity score
            metrics['overall_complexity'] = await self._calculate_overall_complexity(metrics)
            
            logger.debug(f"Calculated complexity metrics: {list(metrics.keys())}")
            return metrics
            
        except Exception as e:
            logger.error(f"Complexity calculation failed", exc_info=e)
            return {'error': str(e)}
    
    async def _calculate_python_metrics(self, content: str) -> Dict[str, float]:
        """Calculate complexity metrics for Python code."""
        metrics = {}
        
        try:
            tree = ast.parse(content)
            
            # Cyclomatic complexity
            metrics['cyclomatic_complexity'] = await self._calculate_cyclomatic_complexity_python(tree)
            
            # Cognitive complexity
            metrics['cognitive_complexity'] = await self._calculate_cognitive_complexity_python(tree)
            
            # Halstead metrics
            halstead_metrics = await self._calculate_halstead_metrics_python(tree, content)
            metrics.update(halstead_metrics)
            
            # Maintainability index
            metrics['maintainability_index'] = await self._calculate_maintainability_index(
                metrics.get('cyclomatic_complexity', 0),
                metrics.get('halstead_volume', 0),
                len(content.split('\n'))
            )
            
            # Nesting depth
            metrics['max_nesting_depth'] = await self._calculate_max_nesting_depth_python(tree)
            
            # Function metrics
            function_metrics = await self._calculate_function_metrics_python(tree)
            metrics.update(function_metrics)
            
        except SyntaxError as e:
            logger.warning(f"Python syntax error during complexity calculation: {e}")
            metrics = await self._calculate_generic_metrics(content)
        except Exception as e:
            logger.error(f"Python complexity calculation failed", exc_info=e)
            metrics = await self._calculate_generic_metrics(content)
        
        return metrics
    
    async def _calculate_javascript_metrics(self, content: str) -> Dict[str, float]:
        """Calculate complexity metrics for JavaScript/TypeScript code."""
        metrics = {}
        
        try:
            # Basic cyclomatic complexity using regex patterns
            metrics['cyclomatic_complexity'] = await self._calculate_cyclomatic_complexity_regex(content)
            
            # Cognitive complexity (simplified)
            metrics['cognitive_complexity'] = await self._calculate_cognitive_complexity_regex(content)
            
            # Nesting depth
            metrics['max_nesting_depth'] = await self._calculate_max_nesting_depth_regex(content)
            
            # Function metrics
            function_metrics = await self._calculate_function_metrics_regex(content)
            metrics.update(function_metrics)
            
            # Generic metrics
            generic_metrics = await self._calculate_generic_metrics(content)
            metrics.update(generic_metrics)
            
        except Exception as e:
            logger.error(f"JavaScript complexity calculation failed", exc_info=e)
            metrics = await self._calculate_generic_metrics(content)
        
        return metrics
    
    async def _calculate_generic_metrics(self, content: str) -> Dict[str, float]:
        """Calculate language-agnostic complexity metrics."""
        metrics = {}
        
        try:
            lines = content.split('\n')
            
            # Basic line metrics
            metrics['total_lines'] = len(lines)
            metrics['code_lines'] = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            metrics['comment_lines'] = len([line for line in lines if line.strip().startswith('#')])
            metrics['blank_lines'] = len([line for line in lines if not line.strip()])
            
            # Character metrics
            metrics['total_characters'] = len(content)
            metrics['avg_line_length'] = sum(len(line) for line in lines) / max(len(lines), 1)
            
            # Indentation complexity
            metrics['avg_indentation'] = await self._calculate_avg_indentation(lines)
            
            # Keyword density
            metrics['keyword_density'] = await self._calculate_keyword_density(content)
            
        except Exception as e:
            logger.error(f"Generic complexity calculation failed", exc_info=e)
        
        return metrics
    
    async def _calculate_cyclomatic_complexity_python(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity for Python AST."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.ListComp):
                complexity += 1
            elif isinstance(node, ast.DictComp):
                complexity += 1
            elif isinstance(node, ast.SetComp):
                complexity += 1
            elif isinstance(node, ast.GeneratorExp):
                complexity += 1
        
        return float(complexity)
    
    async def _calculate_cognitive_complexity_python(self, tree: ast.AST) -> float:
        """Calculate cognitive complexity for Python AST."""
        complexity = 0
        nesting_level = 0
        
        def visit_node(node: ast.AST, level: int) -> int:
            nonlocal complexity
            current_complexity = 0
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                current_complexity += 1 + level
            elif isinstance(node, ast.ExceptHandler):
                current_complexity += 1 + level
            elif isinstance(node, ast.BoolOp):
                current_complexity += len(node.values) - 1
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                current_complexity += 1
            
            # Increase nesting level for certain constructs
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                level += 1
            
            for child in ast.iter_child_nodes(node):
                current_complexity += visit_node(child, level)
            
            return current_complexity
        
        complexity = visit_node(tree, 0)
        return float(complexity)
    
    async def _calculate_halstead_metrics_python(self, tree: ast.AST, content: str) -> Dict[str, float]:
        """Calculate Halstead complexity metrics for Python code."""
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0
        
        # Define Python operators
        python_operators = {
            '+', '-', '*', '/', '//', '%', '**', '=', '+=', '-=', '*=', '/=',
            '==', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not', 'in', 'is',
            '&', '|', '^', '~', '<<', '>>', 'if', 'else', 'elif', 'while',
            'for', 'def', 'class', 'return', 'yield', 'import', 'from', 'as',
            'try', 'except', 'finally', 'with', 'lambda', 'global', 'nonlocal'
        }
        
        for node in ast.walk(tree):
            # Count operators
            node_type = type(node).__name__
            if node_type in ['Add', 'Sub', 'Mult', 'Div', 'Mod', 'Pow', 'LShift', 'RShift',
                           'BitOr', 'BitXor', 'BitAnd', 'FloorDiv', 'Eq', 'NotEq', 'Lt',
                           'LtE', 'Gt', 'GtE', 'Is', 'IsNot', 'In', 'NotIn', 'And', 'Or']:
                operators.add(node_type)
                operator_count += 1
            
            # Count operands (names, constants)
            elif isinstance(node, ast.Name):
                operands.add(node.id)
                operand_count += 1
            elif isinstance(node, (ast.Constant, ast.Num, ast.Str)):
                operands.add(str(getattr(node, 'value', getattr(node, 'n', getattr(node, 's', '')))))
                operand_count += 1
        
        # Calculate Halstead metrics
        n1 = len(operators)  # Number of distinct operators
        n2 = len(operands)   # Number of distinct operands
        N1 = operator_count  # Total number of operators
        N2 = operand_count   # Total number of operands
        
        vocabulary = n1 + n2
        length = N1 + N2
        
        if vocabulary > 0 and length > 0:
            volume = length * math.log2(vocabulary) if vocabulary > 1 else 0
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = difficulty * volume
            time = effort / 18  # Stroud number
            bugs = volume / 3000  # Estimated bugs
        else:
            volume = difficulty = effort = time = bugs = 0
        
        return {
            'halstead_vocabulary': float(vocabulary),
            'halstead_length': float(length),
            'halstead_volume': float(volume),
            'halstead_difficulty': float(difficulty),
            'halstead_effort': float(effort),
            'halstead_time': float(time),
            'halstead_bugs': float(bugs)
        }
    
    async def _calculate_maintainability_index(self, cyclomatic_complexity: float,
                                             halstead_volume: float, lines_of_code: int) -> float:
        """Calculate maintainability index."""
        if lines_of_code == 0:
            return 0.0
        
        # Microsoft's maintainability index formula
        mi = 171 - 5.2 * math.log(halstead_volume) - 0.23 * cyclomatic_complexity - 16.2 * math.log(lines_of_code)
        
        # Normalize to 0-100 scale
        mi = max(0, min(100, mi))
        
        return float(mi)
    
    async def _calculate_max_nesting_depth_python(self, tree: ast.AST) -> float:
        """Calculate maximum nesting depth for Python AST."""
        max_depth = 0
        
        def calculate_depth(node: ast.AST, current_depth: int) -> int:
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            # Increase depth for nesting constructs
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With,
                               ast.AsyncWith, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
                current_depth += 1
            
            for child in ast.iter_child_nodes(node):
                calculate_depth(child, current_depth)
            
            return max_depth
        
        calculate_depth(tree, 0)
        return float(max_depth)
    
    async def _calculate_function_metrics_python(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate function-related metrics for Python code."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Count parameters
                param_count = len(node.args.args)
                
                # Count lines in function
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    function_lines = node.end_lineno - node.lineno + 1
                else:
                    function_lines = 1  # Estimate
                
                functions.append({
                    'param_count': param_count,
                    'line_count': function_lines
                })
        
        if not functions:
            return {
                'function_count': 0.0,
                'avg_function_length': 0.0,
                'avg_parameter_count': 0.0,
                'max_function_length': 0.0,
                'max_parameter_count': 0.0
            }
        
        return {
            'function_count': float(len(functions)),
            'avg_function_length': float(sum(f['line_count'] for f in functions) / len(functions)),
            'avg_parameter_count': float(sum(f['param_count'] for f in functions) / len(functions)),
            'max_function_length': float(max(f['line_count'] for f in functions)),
            'max_parameter_count': float(max(f['param_count'] for f in functions))
        }
    
    async def _calculate_cyclomatic_complexity_regex(self, content: str) -> float:
        """Calculate cyclomatic complexity using regex patterns."""
        complexity = 1  # Base complexity
        
        # Decision points
        patterns = [
            r'\bif\b', r'\belse\b', r'\belif\b', r'\bwhile\b', r'\bfor\b',
            r'\bswitch\b', r'\bcase\b', r'\bcatch\b', r'\btry\b',
            r'\?\s*:', r'\&\&', r'\|\|'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            complexity += len(matches)
        
        return float(complexity)
    
    async def _calculate_cognitive_complexity_regex(self, content: str) -> float:
        """Calculate cognitive complexity using regex patterns."""
        complexity = 0
        
        # Simplified cognitive complexity
        patterns = {
            r'\bif\b': 1,
            r'\belse\b': 1,
            r'\bwhile\b': 1,
            r'\bfor\b': 1,
            r'\bswitch\b': 1,
            r'\bcatch\b': 1,
            r'\&\&': 1,
            r'\|\|': 1
        }
        
        for pattern, weight in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            complexity += len(matches) * weight
        
        return float(complexity)
    
    async def _calculate_max_nesting_depth_regex(self, content: str) -> float:
        """Calculate maximum nesting depth using brace counting."""
        max_depth = 0
        current_depth = 0
        
        for char in content:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        return float(max_depth)
    
    async def _calculate_function_metrics_regex(self, content: str) -> Dict[str, float]:
        """Calculate function metrics using regex patterns."""
        # Find function declarations
        function_patterns = [
            r'function\s+\w+\s*\([^)]*\)',
            r'\w+\s*:\s*function\s*\([^)]*\)',
            r'const\s+\w+\s*=\s*\([^)]*\)\s*=>',
            r'let\s+\w+\s*=\s*\([^)]*\)\s*=>',
            r'var\s+\w+\s*=\s*\([^)]*\)\s*=>'
        ]
        
        functions = []
        for pattern in function_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                # Count parameters
                param_part = re.search(r'\(([^)]*)\)', match.group())
                if param_part:
                    params = [p.strip() for p in param_part.group(1).split(',') if p.strip()]
                    param_count = len(params)
                else:
                    param_count = 0
                
                functions.append({'param_count': param_count})
        
        if not functions:
            return {
                'function_count': 0.0,
                'avg_parameter_count': 0.0,
                'max_parameter_count': 0.0
            }
        
        return {
            'function_count': float(len(functions)),
            'avg_parameter_count': float(sum(f['param_count'] for f in functions) / len(functions)),
            'max_parameter_count': float(max(f['param_count'] for f in functions))
        }
    
    async def _calculate_avg_indentation(self, lines: List[str]) -> float:
        """Calculate average indentation level."""
        indentations = []
        
        for line in lines:
            if line.strip():  # Skip empty lines
                indent = 0
                for char in line:
                    if char == ' ':
                        indent += 1
                    elif char == '\t':
                        indent += 4  # Assume tab = 4 spaces
                    else:
                        break
                indentations.append(indent)
        
        if not indentations:
            return 0.0
        
        return float(sum(indentations) / len(indentations))
    
    async def _calculate_keyword_density(self, content: str) -> float:
        """Calculate density of programming keywords."""
        keywords = [
            'if', 'else', 'elif', 'while', 'for', 'def', 'class', 'return',
            'import', 'from', 'try', 'except', 'finally', 'with', 'as',
            'function', 'var', 'let', 'const', 'return', 'switch', 'case'
        ]
        
        words = re.findall(r'\b\w+\b', content.lower())
        if not words:
            return 0.0
        
        keyword_count = sum(1 for word in words if word in keywords)
        return float(keyword_count / len(words))
    
    async def _calculate_overall_complexity(self, metrics: Dict[str, float]) -> float:
        """Calculate an overall complexity score."""
        weights = {
            'cyclomatic_complexity': 0.3,
            'cognitive_complexity': 0.25,
            'max_nesting_depth': 0.2,
            'halstead_difficulty': 0.15,
            'avg_function_length': 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                # Normalize metrics to 0-1 scale
                normalized_value = min(metrics[metric] / 20, 1.0)  # Assume max reasonable value is 20
                score += normalized_value * weight
                total_weight += weight
        
        if total_weight > 0:
            return score / total_weight
        else:
            return 0.0


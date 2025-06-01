"""
AST Processor for detailed code structure analysis.

Provides advanced AST processing capabilities for multiple programming languages.
"""

import ast
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from ..core.interfaces import AnalysisConfig, LanguageType, CodeElement


logger = logging.getLogger(__name__)


@dataclass
class ASTNode:
    """Represents a node in the Abstract Syntax Tree."""
    type: str
    name: Optional[str]
    line: int
    column: int
    children: List['ASTNode']
    metadata: Dict[str, Any]


class ASTProcessor:
    """
    Advanced AST processor for code structure analysis.
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the AST processor."""
        self.config = config
        logger.debug("AST processor initialized")
    
    async def process_python_ast(self, content: str) -> List[CodeElement]:
        """
        Process Python AST and extract code elements.
        
        Args:
            content: Python source code
            
        Returns:
            List of extracted code elements
        """
        elements = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                element = await self._process_python_node(node, content)
                if element:
                    elements.append(element)
            
            logger.debug(f"Extracted {len(elements)} elements from Python AST")
            return elements
            
        except SyntaxError as e:
            logger.warning(f"Python syntax error: {e}")
            return []
        except Exception as e:
            logger.error(f"Python AST processing failed", exc_info=e)
            return []
    
    async def _process_python_node(self, node: ast.AST, content: str) -> Optional[CodeElement]:
        """Process a single Python AST node."""
        try:
            if isinstance(node, ast.FunctionDef):
                return await self._create_function_element(node, content, LanguageType.PYTHON)
            
            elif isinstance(node, ast.AsyncFunctionDef):
                element = await self._create_function_element(node, content, LanguageType.PYTHON)
                if element:
                    element.metadata['is_async'] = True
                return element
            
            elif isinstance(node, ast.ClassDef):
                return await self._create_class_element(node, content, LanguageType.PYTHON)
            
            elif isinstance(node, ast.Import):
                return await self._create_import_element(node, content, LanguageType.PYTHON)
            
            elif isinstance(node, ast.ImportFrom):
                return await self._create_import_from_element(node, content, LanguageType.PYTHON)
            
            elif isinstance(node, ast.Assign):
                return await self._create_variable_element(node, content, LanguageType.PYTHON)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing Python node {type(node)}", exc_info=e)
            return None
    
    async def _create_function_element(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                                     content: str, language: LanguageType) -> CodeElement:
        """Create a CodeElement for a function."""
        lines = content.split('\n')
        
        # Calculate end line (approximate)
        end_line = node.lineno
        if hasattr(node, 'end_lineno') and node.end_lineno:
            end_line = node.end_lineno
        else:
            # Estimate end line by finding the next function or class
            for i in range(node.lineno, len(lines)):
                line = lines[i].strip()
                if line and not line.startswith(' ') and not line.startswith('\t'):
                    if line.startswith(('def ', 'class ', 'async def ')):
                        end_line = i
                        break
            else:
                end_line = len(lines)
        
        # Extract function content
        function_lines = lines[node.lineno - 1:end_line]
        function_content = '\n'.join(function_lines)
        
        # Extract dependencies (function calls within the function)
        dependencies = set()
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Call):
                if isinstance(child_node.func, ast.Name):
                    dependencies.add(child_node.func.id)
                elif isinstance(child_node.func, ast.Attribute):
                    dependencies.add(child_node.func.attr)
        
        # Calculate complexity (basic cyclomatic complexity)
        complexity = await self._calculate_function_complexity(node)
        
        return CodeElement(
            name=node.name,
            type='function',
            start_line=node.lineno,
            end_line=end_line,
            start_column=node.col_offset,
            end_column=getattr(node, 'end_col_offset', 0),
            content=function_content,
            language=language,
            dependencies=dependencies,
            complexity_score=complexity,
            metadata={
                'args': [arg.arg for arg in node.args.args],
                'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                'returns': self._get_return_annotation(node),
                'docstring': ast.get_docstring(node),
                'is_async': isinstance(node, ast.AsyncFunctionDef)
            }
        )
    
    async def _create_class_element(self, node: ast.ClassDef, content: str, 
                                  language: LanguageType) -> CodeElement:
        """Create a CodeElement for a class."""
        lines = content.split('\n')
        
        # Calculate end line
        end_line = node.lineno
        if hasattr(node, 'end_lineno') and node.end_lineno:
            end_line = node.end_lineno
        else:
            # Estimate end line
            for i in range(node.lineno, len(lines)):
                line = lines[i].strip()
                if line and not line.startswith(' ') and not line.startswith('\t'):
                    if line.startswith(('def ', 'class ', 'async def ')):
                        end_line = i
                        break
            else:
                end_line = len(lines)
        
        # Extract class content
        class_lines = lines[node.lineno - 1:end_line]
        class_content = '\n'.join(class_lines)
        
        # Extract methods
        methods = []
        for child_node in node.body:
            if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(child_node.name)
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
        
        return CodeElement(
            name=node.name,
            type='class',
            start_line=node.lineno,
            end_line=end_line,
            start_column=node.col_offset,
            end_column=getattr(node, 'end_col_offset', 0),
            content=class_content,
            language=language,
            dependencies=set(bases),
            metadata={
                'bases': bases,
                'methods': methods,
                'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                'docstring': ast.get_docstring(node)
            }
        )
    
    async def _create_import_element(self, node: ast.Import, content: str, 
                                   language: LanguageType) -> CodeElement:
        """Create a CodeElement for an import statement."""
        import_names = [alias.name for alias in node.names]
        
        return CodeElement(
            name=', '.join(import_names),
            type='import',
            start_line=node.lineno,
            end_line=node.lineno,
            start_column=node.col_offset,
            end_column=getattr(node, 'end_col_offset', 0),
            content=f"import {', '.join(import_names)}",
            language=language,
            metadata={
                'modules': import_names,
                'aliases': {alias.name: alias.asname for alias in node.names if alias.asname}
            }
        )
    
    async def _create_import_from_element(self, node: ast.ImportFrom, content: str, 
                                        language: LanguageType) -> CodeElement:
        """Create a CodeElement for a from-import statement."""
        module = node.module or ''
        import_names = [alias.name for alias in node.names]
        
        return CodeElement(
            name=f"{module}.{', '.join(import_names)}",
            type='import',
            start_line=node.lineno,
            end_line=node.lineno,
            start_column=node.col_offset,
            end_column=getattr(node, 'end_col_offset', 0),
            content=f"from {module} import {', '.join(import_names)}",
            language=language,
            dependencies={module} if module else set(),
            metadata={
                'module': module,
                'names': import_names,
                'aliases': {alias.name: alias.asname for alias in node.names if alias.asname}
            }
        )
    
    async def _create_variable_element(self, node: ast.Assign, content: str, 
                                     language: LanguageType) -> Optional[CodeElement]:
        """Create a CodeElement for a variable assignment."""
        # Only process simple name assignments
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
            
            # Get the assignment line
            lines = content.split('\n')
            if node.lineno <= len(lines):
                assignment_content = lines[node.lineno - 1].strip()
            else:
                assignment_content = f"{target.id} = ..."
            
            return CodeElement(
                name=target.id,
                type='variable',
                start_line=node.lineno,
                end_line=node.lineno,
                start_column=node.col_offset,
                end_column=getattr(node, 'end_col_offset', 0),
                content=assignment_content,
                language=language,
                metadata={
                    'is_constant': target.id.isupper(),
                    'value_type': type(node.value).__name__
                }
            )
        
        return None
    
    async def _calculate_function_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> float:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child_node in ast.walk(node):
            if isinstance(child_node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child_node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child_node, ast.With):
                complexity += 1
            elif isinstance(child_node, ast.BoolOp):
                complexity += len(child_node.values) - 1
        
        return float(complexity)
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return str(decorator)
    
    def _get_return_annotation(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Extract return type annotation."""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return node.returns.attr
            else:
                return str(node.returns)
        return None


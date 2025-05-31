"""
Graph-sitter integration for advanced code parsing and analysis.

Provides tree-sitter based parsing capabilities for multiple programming languages
with AST extraction and dependency graph construction.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path

try:
    import tree_sitter
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    tree_sitter = None
    Language = None
    Parser = None
    Node = None

from ..core.interfaces import (
    GraphSitterParserInterface, CodeElement, DependencyGraph, 
    DependencyNode, DependencyEdge, LanguageType, AnalysisConfig
)


logger = logging.getLogger(__name__)


class GraphSitterParser(GraphSitterParserInterface):
    """
    Graph-sitter based parser for advanced code analysis.
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the graph-sitter parser."""
        self.config = config
        self.parsers = {}
        self.languages = {}
        
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available. Using fallback parsing.")
            return
        
        # Initialize language parsers
        asyncio.create_task(self._initialize_languages())
        
        logger.info("Graph-sitter parser initialized")
    
    async def _initialize_languages(self):
        """Initialize tree-sitter languages."""
        if not TREE_SITTER_AVAILABLE:
            return
        
        try:
            # Language mappings
            language_configs = {
                LanguageType.PYTHON: {
                    'library_name': 'python',
                    'parser_name': 'python'
                },
                LanguageType.JAVASCRIPT: {
                    'library_name': 'javascript',
                    'parser_name': 'javascript'
                },
                LanguageType.TYPESCRIPT: {
                    'library_name': 'typescript',
                    'parser_name': 'typescript'
                },
                LanguageType.JAVA: {
                    'library_name': 'java',
                    'parser_name': 'java'
                },
                LanguageType.CPP: {
                    'library_name': 'cpp',
                    'parser_name': 'cpp'
                },
                LanguageType.RUST: {
                    'library_name': 'rust',
                    'parser_name': 'rust'
                },
                LanguageType.GO: {
                    'library_name': 'go',
                    'parser_name': 'go'
                }
            }
            
            for lang_type, config in language_configs.items():
                try:
                    # Try to load the language
                    # Note: In a real implementation, you would need to build or download
                    # the tree-sitter language libraries
                    parser = Parser()
                    # language = Language(library_path, config['parser_name'])
                    # parser.set_language(language)
                    
                    # For now, we'll use a mock implementation
                    self.parsers[lang_type] = parser
                    # self.languages[lang_type] = language
                    
                    logger.debug(f"Initialized parser for {lang_type}")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize parser for {lang_type}: {e}")
            
        except Exception as e:
            logger.error(f"Language initialization failed", exc_info=e)
    
    async def parse(self, content: str, language: LanguageType) -> Any:
        """
        Parse content using tree-sitter.
        
        Args:
            content: Source code content
            language: Programming language type
            
        Returns:
            Parsed AST tree or None if parsing fails
        """
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available, returning None")
            return None
        
        try:
            parser = self.parsers.get(language)
            if not parser:
                logger.warning(f"No parser available for {language}")
                return await self._fallback_parse(content, language)
            
            # Parse the content
            tree = parser.parse(bytes(content, 'utf8'))
            
            if tree.root_node.has_error:
                logger.warning(f"Parse errors detected for {language}")
                # Still return the tree as it might have partial information
            
            return tree
            
        except Exception as e:
            logger.error(f"Parsing failed for {language}", exc_info=e)
            return await self._fallback_parse(content, language)
    
    async def extract_elements(self, tree: Any, content: str) -> List[CodeElement]:
        """
        Extract code elements from parsed AST.
        
        Args:
            tree: Parsed AST tree
            content: Original source code
            
        Returns:
            List of extracted code elements
        """
        if not tree or not TREE_SITTER_AVAILABLE:
            return await self._fallback_extract_elements(content)
        
        try:
            elements = []
            content_bytes = bytes(content, 'utf8')
            
            # Traverse the AST and extract elements
            await self._traverse_node(tree.root_node, content_bytes, elements)
            
            logger.debug(f"Extracted {len(elements)} elements from AST")
            return elements
            
        except Exception as e:
            logger.error(f"Element extraction failed", exc_info=e)
            return await self._fallback_extract_elements(content)
    
    async def build_dependency_graph(self, elements: List[CodeElement]) -> DependencyGraph:
        """
        Build dependency graph from code elements.
        
        Args:
            elements: List of code elements
            
        Returns:
            Constructed dependency graph
        """
        try:
            graph = DependencyGraph()
            
            # Create nodes for each element
            for element in elements:
                node = DependencyNode(
                    id=f"{element.name}_{element.type}_{element.start_line}",
                    name=element.name,
                    type=element.type,
                    file_path=getattr(element, 'file_path', None),
                    line_number=element.start_line,
                    metadata={
                        'language': element.language.value,
                        'complexity': element.complexity_score,
                        'element_metadata': element.metadata
                    }
                )
                graph.add_node(node)
            
            # Create edges based on dependencies
            for element in elements:
                source_id = f"{element.name}_{element.type}_{element.start_line}"
                
                for dependency in element.dependencies:
                    # Find the target element
                    target_element = self._find_element_by_name(elements, dependency)
                    if target_element:
                        target_id = f"{target_element.name}_{target_element.type}_{target_element.start_line}"
                        
                        edge = DependencyEdge(
                            source=source_id,
                            target=target_id,
                            relationship_type=self._determine_relationship_type(element, target_element),
                            weight=1.0,
                            metadata={
                                'source_type': element.type,
                                'target_type': target_element.type
                            }
                        )
                        graph.add_edge(edge)
            
            logger.debug(f"Built dependency graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph
            
        except Exception as e:
            logger.error(f"Dependency graph construction failed", exc_info=e)
            return DependencyGraph()
    
    async def _traverse_node(self, node: Any, content_bytes: bytes, elements: List[CodeElement]):
        """Traverse AST node and extract code elements."""
        if not node:
            return
        
        try:
            # Extract element based on node type
            element = await self._extract_element_from_node(node, content_bytes)
            if element:
                elements.append(element)
            
            # Recursively traverse children
            for child in node.children:
                await self._traverse_node(child, content_bytes, elements)
                
        except Exception as e:
            logger.error(f"Node traversal failed", exc_info=e)
    
    async def _extract_element_from_node(self, node: Any, content_bytes: bytes) -> Optional[CodeElement]:
        """Extract a code element from an AST node."""
        try:
            node_type = node.type
            
            # Map tree-sitter node types to our element types
            element_type_mapping = {
                'function_definition': 'function',
                'async_function_definition': 'function',
                'method_definition': 'function',
                'class_definition': 'class',
                'import_statement': 'import',
                'import_from_statement': 'import',
                'assignment': 'variable',
                'variable_declaration': 'variable'
            }
            
            element_type = element_type_mapping.get(node_type)
            if not element_type:
                return None
            
            # Extract node content
            start_byte = node.start_byte
            end_byte = node.end_byte
            node_content = content_bytes[start_byte:end_byte].decode('utf8')
            
            # Extract name (this would need language-specific logic)
            name = await self._extract_node_name(node, content_bytes)
            if not name:
                return None
            
            # Extract dependencies (simplified)
            dependencies = await self._extract_node_dependencies(node, content_bytes)
            
            # Calculate complexity (simplified)
            complexity = await self._calculate_node_complexity(node)
            
            return CodeElement(
                name=name,
                type=element_type,
                start_line=node.start_point[0] + 1,  # Convert to 1-based
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                content=node_content,
                language=LanguageType.PYTHON,  # Would need to be determined
                dependencies=dependencies,
                complexity_score=complexity,
                metadata={
                    'node_type': node_type,
                    'start_byte': start_byte,
                    'end_byte': end_byte
                }
            )
            
        except Exception as e:
            logger.error(f"Element extraction from node failed", exc_info=e)
            return None
    
    async def _extract_node_name(self, node: Any, content_bytes: bytes) -> Optional[str]:
        """Extract the name of a node (function name, class name, etc.)."""
        try:
            # This would need language-specific logic
            # For now, we'll use a simple approach
            
            # Look for identifier children
            for child in node.children:
                if child.type == 'identifier':
                    return content_bytes[child.start_byte:child.end_byte].decode('utf8')
            
            return None
            
        except Exception as e:
            logger.error(f"Node name extraction failed", exc_info=e)
            return None
    
    async def _extract_node_dependencies(self, node: Any, content_bytes: bytes) -> Set[str]:
        """Extract dependencies from a node."""
        dependencies = set()
        
        try:
            # Look for function calls, attribute access, etc.
            for child in node.children:
                if child.type in ['call', 'attribute', 'identifier']:
                    dep_name = await self._extract_node_name(child, content_bytes)
                    if dep_name:
                        dependencies.add(dep_name)
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Dependency extraction failed", exc_info=e)
            return set()
    
    async def _calculate_node_complexity(self, node: Any) -> float:
        """Calculate complexity for a node."""
        try:
            # Simple complexity based on node count and depth
            node_count = len(list(node.children))
            depth = self._calculate_node_depth(node)
            
            return float(node_count * 0.1 + depth * 0.5)
            
        except Exception as e:
            logger.error(f"Complexity calculation failed", exc_info=e)
            return 1.0
    
    def _calculate_node_depth(self, node: Any, current_depth: int = 0) -> int:
        """Calculate the maximum depth of a node."""
        if not node.children:
            return current_depth
        
        max_depth = current_depth
        for child in node.children:
            child_depth = self._calculate_node_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _find_element_by_name(self, elements: List[CodeElement], name: str) -> Optional[CodeElement]:
        """Find an element by name."""
        for element in elements:
            if element.name == name:
                return element
        return None
    
    def _determine_relationship_type(self, source: CodeElement, target: CodeElement) -> str:
        """Determine the type of relationship between two elements."""
        if source.type == 'function' and target.type == 'function':
            return 'calls'
        elif source.type == 'class' and target.type == 'class':
            return 'inherits'
        elif source.type in ['function', 'class'] and target.type == 'import':
            return 'imports'
        else:
            return 'depends_on'
    
    async def _fallback_parse(self, content: str, language: LanguageType) -> Dict[str, Any]:
        """Fallback parsing when tree-sitter is not available."""
        logger.debug(f"Using fallback parsing for {language}")
        
        return {
            'type': 'fallback',
            'language': language,
            'content': content,
            'lines': content.split('\n')
        }
    
    async def _fallback_extract_elements(self, content: str) -> List[CodeElement]:
        """Fallback element extraction using regex patterns."""
        elements = []
        
        try:
            import re
            lines = content.split('\n')
            
            # Simple regex patterns for common constructs
            patterns = {
                'function': [
                    r'def\s+(\w+)\s*\(',
                    r'function\s+(\w+)\s*\(',
                    r'(\w+)\s*:\s*function\s*\('
                ],
                'class': [
                    r'class\s+(\w+)',
                    r'class\s+(\w+)\s*\('
                ],
                'import': [
                    r'import\s+(\w+)',
                    r'from\s+(\w+)\s+import'
                ]
            }
            
            for line_num, line in enumerate(lines, 1):
                for element_type, type_patterns in patterns.items():
                    for pattern in type_patterns:
                        match = re.search(pattern, line)
                        if match:
                            element = CodeElement(
                                name=match.group(1),
                                type=element_type,
                                start_line=line_num,
                                end_line=line_num,
                                start_column=match.start(),
                                end_column=match.end(),
                                content=line.strip(),
                                language=LanguageType.UNKNOWN,
                                dependencies=set(),
                                metadata={'fallback_parsing': True}
                            )
                            elements.append(element)
                            break
            
            logger.debug(f"Fallback extraction found {len(elements)} elements")
            return elements
            
        except Exception as e:
            logger.error(f"Fallback extraction failed", exc_info=e)
            return []


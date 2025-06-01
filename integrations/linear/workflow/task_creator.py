"""
Task Creator

Creates OpenAlpha_Evolve tasks from Linear issues with intelligent parsing
of requirements and test case generation.
"""

import logging
import re
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..interfaces import LinearIssue, LinearGraphQLClientInterface
from core.interfaces import TaskDefinition

logger = logging.getLogger(__name__)


class TaskCreator:
    """Creates OpenAlpha_Evolve tasks from Linear issues"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.linear_client: Optional[LinearGraphQLClientInterface] = None
        
        # Task creation configuration
        self.default_function_name = self.config.get("default_function_name", "solve")
        self.default_imports = self.config.get("default_imports", ["math", "itertools", "collections"])
        self.max_description_length = self.config.get("max_description_length", 2000)
        
        # Pattern matching for requirements extraction
        self.function_name_patterns = [
            r"function\s+(?:called\s+)?[`'\"]?(\w+)[`'\"]?",
            r"implement\s+(?:a\s+)?(?:function\s+)?[`'\"]?(\w+)[`'\"]?",
            r"create\s+(?:a\s+)?(?:function\s+)?[`'\"]?(\w+)[`'\"]?",
            r"write\s+(?:a\s+)?(?:function\s+)?[`'\"]?(\w+)[`'\"]?"
        ]
        
        self.test_case_patterns = [
            r"input:\s*(.+?)\s*(?:output|expected|result):\s*(.+?)(?:\n|$)",
            r"example:\s*(.+?)\s*(?:->|=>|returns?)\s*(.+?)(?:\n|$)",
            r"test:\s*(.+?)\s*(?:should\s+return|returns?)\s*(.+?)(?:\n|$)"
        ]
        
        logger.info("TaskCreator initialized")
    
    def set_linear_client(self, client: LinearGraphQLClientInterface) -> None:
        """Set Linear GraphQL client"""
        self.linear_client = client
    
    async def create_task_from_issue(self, issue: LinearIssue) -> Optional[str]:
        """Create OpenAlpha_Evolve task from Linear issue"""
        try:
            logger.info(f"Creating task from Linear issue: {issue.id}")
            
            # Parse issue content
            task_definition = await self._parse_issue_to_task(issue)
            
            if not task_definition:
                logger.error(f"Failed to parse issue {issue.id} into task definition")
                return None
            
            # Generate unique task ID
            task_id = f"linear_{issue.id}_{uuid.uuid4().hex[:8]}"
            task_definition.id = task_id
            
            # TODO: Store task definition and integrate with TaskManagerAgent
            # For now, just log the created task
            logger.info(f"Created task definition: {task_id}")
            logger.debug(f"Task definition: {task_definition}")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating task from issue: {e}")
            return None
    
    async def _parse_issue_to_task(self, issue: LinearIssue) -> Optional[TaskDefinition]:
        """Parse Linear issue into TaskDefinition"""
        try:
            # Combine title and description
            full_text = f"{issue.title}\n\n{issue.description or ''}"
            
            # Extract function name
            function_name = self._extract_function_name(full_text)
            
            # Extract test cases
            test_cases = self._extract_test_cases(full_text)
            
            # Extract allowed imports
            allowed_imports = self._extract_imports(full_text)
            
            # Clean and format description
            description = self._format_description(issue)
            
            # Create task definition
            task_definition = TaskDefinition(
                id="",  # Will be set by caller
                description=description,
                function_name_to_evolve=function_name,
                input_output_examples=test_cases,
                allowed_imports=allowed_imports,
                tests=self._create_test_groups(test_cases),
                expert_knowledge=self._extract_expert_knowledge(full_text)
            )
            
            return task_definition
            
        except Exception as e:
            logger.error(f"Error parsing issue to task: {e}")
            return None
    
    def _extract_function_name(self, text: str) -> str:
        """Extract function name from issue text"""
        text_lower = text.lower()
        
        for pattern in self.function_name_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                function_name = match.group(1)
                # Validate function name
                if function_name.isidentifier():
                    logger.debug(f"Extracted function name: {function_name}")
                    return function_name
        
        # Default function name
        logger.debug(f"Using default function name: {self.default_function_name}")
        return self.default_function_name
    
    def _extract_test_cases(self, text: str) -> List[Dict[str, Any]]:
        """Extract test cases from issue text"""
        test_cases = []
        
        for pattern in self.test_case_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                try:
                    input_str = match.group(1).strip()
                    output_str = match.group(2).strip()
                    
                    # Parse input and output
                    input_value = self._parse_value(input_str)
                    output_value = self._parse_value(output_str)
                    
                    if input_value is not None and output_value is not None:
                        test_cases.append({
                            "input": input_value if isinstance(input_value, list) else [input_value],
                            "output": output_value
                        })
                        logger.debug(f"Extracted test case: {input_value} -> {output_value}")
                
                except Exception as e:
                    logger.warning(f"Failed to parse test case: {e}")
                    continue
        
        # If no test cases found, create some basic ones
        if not test_cases:
            test_cases = self._generate_default_test_cases(text)
        
        return test_cases
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse string value into Python object"""
        value_str = value_str.strip()
        
        # Remove common prefixes/suffixes
        value_str = re.sub(r'^(input|output|expected|result):\s*', '', value_str, flags=re.IGNORECASE)
        value_str = value_str.strip('`"\'')
        
        try:
            # Try to evaluate as Python literal
            import ast
            return ast.literal_eval(value_str)
        except:
            # If that fails, try some common patterns
            if value_str.lower() in ['true', 'false']:
                return value_str.lower() == 'true'
            elif value_str.lower() in ['null', 'none']:
                return None
            elif value_str.isdigit():
                return int(value_str)
            elif re.match(r'^\d+\.\d+$', value_str):
                return float(value_str)
            elif value_str.startswith('[') and value_str.endswith(']'):
                # Try to parse as list
                try:
                    return eval(value_str)
                except:
                    pass
            
            # Return as string if all else fails
            return value_str
    
    def _extract_imports(self, text: str) -> List[str]:
        """Extract allowed imports from issue text"""
        imports = set(self.default_imports)
        
        # Look for import statements in the text
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import',
            r'using\s+(\w+)',
            r'library:\s*(\w+)',
            r'module:\s*(\w+)'
        ]
        
        for pattern in import_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                module_name = match.group(1)
                if module_name.isidentifier():
                    imports.add(module_name)
        
        return list(imports)
    
    def _format_description(self, issue: LinearIssue) -> str:
        """Format issue description for task"""
        description_parts = []
        
        # Add title
        description_parts.append(f"# {issue.title}")
        
        # Add issue description if available
        if issue.description:
            # Clean up description
            clean_desc = issue.description.strip()
            if len(clean_desc) > self.max_description_length:
                clean_desc = clean_desc[:self.max_description_length] + "..."
            description_parts.append(clean_desc)
        
        # Add metadata
        metadata_parts = []
        if issue.team_name:
            metadata_parts.append(f"Team: {issue.team_name}")
        if issue.priority:
            metadata_parts.append(f"Priority: {issue.priority}")
        if issue.labels:
            metadata_parts.append(f"Labels: {', '.join(issue.labels)}")
        
        if metadata_parts:
            description_parts.append(f"\\n**Issue Metadata:**\\n{'; '.join(metadata_parts)}")
        
        # Add Linear URL
        if issue.url:
            description_parts.append(f"\\n**Linear Issue:** {issue.url}")
        
        return "\\n\\n".join(description_parts)
    
    def _create_test_groups(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create test groups from test cases"""
        if not test_cases:
            return []
        
        return [{
            "name": "Linear Issue Test Cases",
            "description": "Test cases extracted from Linear issue",
            "test_cases": test_cases
        }]
    
    def _extract_expert_knowledge(self, text: str) -> Optional[str]:
        """Extract expert knowledge or algorithms from issue text"""
        knowledge_patterns = [
            r'algorithm:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'approach:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'solution:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'hint:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'note:\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        knowledge_parts = []
        
        for pattern in knowledge_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                knowledge = match.group(1).strip()
                if knowledge and len(knowledge) > 10:  # Minimum length filter
                    knowledge_parts.append(knowledge)
        
        if knowledge_parts:
            return "\\n\\n".join(knowledge_parts)
        
        return None
    
    def _generate_default_test_cases(self, text: str) -> List[Dict[str, Any]]:
        """Generate default test cases when none are found"""
        # This is a fallback - in practice, you'd want more sophisticated generation
        # based on the problem type
        
        default_cases = []
        
        # Look for numeric patterns to guess input types
        if re.search(r'\b(?:number|integer|int|digit)\b', text, re.IGNORECASE):
            default_cases.extend([
                {"input": [0], "output": 0},
                {"input": [1], "output": 1},
                {"input": [5], "output": 5}
            ])
        elif re.search(r'\b(?:list|array|sequence)\b', text, re.IGNORECASE):
            default_cases.extend([
                {"input": [[]], "output": []},
                {"input": [[1, 2, 3]], "output": [1, 2, 3]},
                {"input": [[1, 2, 3, 4, 5]], "output": [1, 2, 3, 4, 5]}
            ])
        elif re.search(r'\b(?:string|text|word)\b', text, re.IGNORECASE):
            default_cases.extend([
                {"input": [""], "output": ""},
                {"input": ["hello"], "output": "hello"},
                {"input": ["test"], "output": "test"}
            ])
        else:
            # Generic default cases
            default_cases.extend([
                {"input": [None], "output": None},
                {"input": [True], "output": True},
                {"input": [False], "output": False}
            ])
        
        logger.info(f"Generated {len(default_cases)} default test cases")
        return default_cases
    
    async def validate_task_definition(self, task_definition: TaskDefinition) -> bool:
        """Validate created task definition"""
        try:
            # Check required fields
            if not task_definition.description:
                logger.error("Task definition missing description")
                return False
            
            if not task_definition.function_name_to_evolve:
                logger.error("Task definition missing function name")
                return False
            
            # Validate function name
            if not task_definition.function_name_to_evolve.isidentifier():
                logger.error(f"Invalid function name: {task_definition.function_name_to_evolve}")
                return False
            
            # Check test cases
            if not task_definition.input_output_examples:
                logger.warning("Task definition has no test cases")
            
            # Validate test cases format
            for i, test_case in enumerate(task_definition.input_output_examples or []):
                if "input" not in test_case or "output" not in test_case:
                    logger.error(f"Test case {i} missing input or output")
                    return False
            
            logger.info("Task definition validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating task definition: {e}")
            return False


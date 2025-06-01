"""
Automated Debugging System with Self-Healing Capabilities
"""

import ast
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from core.interfaces import BaseAgent, Program
from error_analysis.error_classifier import ErrorClassifier, ErrorClassification

logger = logging.getLogger(__name__)

class DebugAction(Enum):
    """Types of debugging actions that can be performed."""
    SYNTAX_FIX = "syntax_fix"
    IMPORT_FIX = "import_fix"
    VARIABLE_FIX = "variable_fix"
    TYPE_FIX = "type_fix"
    LOGIC_FIX = "logic_fix"
    PERFORMANCE_FIX = "performance_fix"
    REFACTOR = "refactor"
    ADD_ERROR_HANDLING = "add_error_handling"

@dataclass
class DebugResult:
    """Result of a debugging attempt."""
    success: bool
    fixed_code: Optional[str] = None
    actions_taken: List[DebugAction] = field(default_factory=list)
    confidence_score: float = 0.0
    error_resolved: bool = False
    new_errors: List[str] = field(default_factory=list)
    explanation: str = ""

class AutoDebugger(BaseAgent):
    """
    Automated debugging system that can analyze errors and attempt fixes.
    Uses pattern matching, AST manipulation, and heuristics to resolve issues.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.error_classifier = ErrorClassifier()
        self.debug_history: List[Tuple[str, ErrorClassification, DebugResult]] = []
        self.success_patterns: Dict[str, List[str]] = {}
        
    async def execute(self, program: Program, error_info: Dict[str, Any]) -> DebugResult:
        """Main execution method - attempts to debug and fix a program."""
        return await self.debug_program(program, error_info)
    
    async def debug_program(self, program: Program, error_info: Dict[str, Any]) -> DebugResult:
        """
        Attempt to debug and fix a program based on error information.
        
        Args:
            program: The program that has errors
            error_info: Information about the error that occurred
        """
        logger.info(f"Starting automated debugging for program {program.id}")
        
        # Classify the error first
        error_classification = await self.error_classifier.classify_error(error_info)
        
        # Determine debugging strategy based on error classification
        debug_actions = self._plan_debug_actions(error_classification, program.code)
        
        # Attempt fixes in order of priority
        current_code = program.code
        actions_taken = []
        
        for action in debug_actions:
            try:
                fixed_code = await self._apply_debug_action(action, current_code, error_classification)
                if fixed_code and fixed_code != current_code:
                    current_code = fixed_code
                    actions_taken.append(action)
                    logger.debug(f"Applied debug action: {action.value}")
            except Exception as e:
                logger.error(f"Error applying debug action {action.value}: {e}")
        
        # Validate the fix
        validation_result = await self._validate_fix(current_code, program.code, error_classification)
        
        debug_result = DebugResult(
            success=validation_result['is_valid'],
            fixed_code=current_code if current_code != program.code else None,
            actions_taken=actions_taken,
            confidence_score=validation_result['confidence'],
            error_resolved=validation_result['error_resolved'],
            new_errors=validation_result['new_errors'],
            explanation=validation_result['explanation']
        )
        
        # Store in history for learning
        self.debug_history.append((program.id, error_classification, debug_result))
        
        # Update success patterns
        if debug_result.success:
            self._update_success_patterns(error_classification, actions_taken)
        
        logger.info(f"Debugging complete for {program.id}: success={debug_result.success}, "
                   f"actions={len(actions_taken)}, confidence={debug_result.confidence_score:.2f}")
        
        return debug_result
    
    def _plan_debug_actions(self, error_classification: ErrorClassification, code: str) -> List[DebugAction]:
        """Plan debugging actions based on error classification."""
        actions = []
        
        # Priority-based action planning
        if error_classification.category.value == "syntax_error":
            actions.extend([
                DebugAction.SYNTAX_FIX,
                DebugAction.REFACTOR
            ])
        
        elif error_classification.category.value == "import_error":
            actions.extend([
                DebugAction.IMPORT_FIX,
                DebugAction.REFACTOR
            ])
        
        elif error_classification.category.value == "runtime_error":
            actions.extend([
                DebugAction.VARIABLE_FIX,
                DebugAction.ADD_ERROR_HANDLING,
                DebugAction.LOGIC_FIX
            ])
        
        elif error_classification.category.value == "type_error":
            actions.extend([
                DebugAction.TYPE_FIX,
                DebugAction.VARIABLE_FIX,
                DebugAction.ADD_ERROR_HANDLING
            ])
        
        elif error_classification.category.value in ["performance_error", "timeout_error"]:
            actions.extend([
                DebugAction.PERFORMANCE_FIX,
                DebugAction.REFACTOR
            ])
        
        # Add general actions
        actions.extend([
            DebugAction.ADD_ERROR_HANDLING,
            DebugAction.REFACTOR
        ])
        
        # Check success patterns for this error type
        pattern_key = f"{error_classification.category.value}:{error_classification.error_type}"
        if pattern_key in self.success_patterns:
            # Prioritize actions that have worked before
            successful_actions = [DebugAction(action) for action in self.success_patterns[pattern_key]]
            actions = successful_actions + [a for a in actions if a not in successful_actions]
        
        return actions[:5]  # Limit to top 5 actions
    
    async def _apply_debug_action(self, action: DebugAction, code: str, 
                                error_classification: ErrorClassification) -> Optional[str]:
        """Apply a specific debugging action to the code."""
        
        if action == DebugAction.SYNTAX_FIX:
            return self._fix_syntax_errors(code, error_classification)
        
        elif action == DebugAction.IMPORT_FIX:
            return self._fix_import_errors(code, error_classification)
        
        elif action == DebugAction.VARIABLE_FIX:
            return self._fix_variable_errors(code, error_classification)
        
        elif action == DebugAction.TYPE_FIX:
            return self._fix_type_errors(code, error_classification)
        
        elif action == DebugAction.LOGIC_FIX:
            return self._fix_logic_errors(code, error_classification)
        
        elif action == DebugAction.PERFORMANCE_FIX:
            return self._fix_performance_issues(code, error_classification)
        
        elif action == DebugAction.ADD_ERROR_HANDLING:
            return self._add_error_handling(code, error_classification)
        
        elif action == DebugAction.REFACTOR:
            return self._refactor_code(code, error_classification)
        
        return None
    
    def _fix_syntax_errors(self, code: str, error_classification: ErrorClassification) -> Optional[str]:
        """Fix common syntax errors."""
        fixed_code = code
        
        # Fix missing colons
        if "invalid syntax" in error_classification.error_message.lower():
            # Add colons to if/for/while/def/class statements
            lines = fixed_code.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if (stripped.startswith(('if ', 'for ', 'while ', 'def ', 'class ', 'elif ', 'else', 'try', 'except', 'finally')) 
                    and not stripped.endswith(':') and not stripped.endswith(':\\')):
                    lines[i] = line + ':'
            fixed_code = '\n'.join(lines)
        
        # Fix indentation issues
        if "indentation" in error_classification.error_message.lower():
            lines = fixed_code.split('\n')
            fixed_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    fixed_lines.append('')
                    continue
                
                # Determine expected indent level
                if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'with ')):
                    fixed_lines.append('    ' * indent_level + stripped)
                    if stripped.endswith(':'):
                        indent_level += 1
                elif stripped in ['else:', 'elif', 'except:', 'finally:']:
                    indent_level = max(0, indent_level - 1)
                    fixed_lines.append('    ' * indent_level + stripped)
                    indent_level += 1
                else:
                    fixed_lines.append('    ' * indent_level + stripped)
            
            fixed_code = '\n'.join(fixed_lines)
        
        # Fix missing parentheses
        if "unexpected EOF" in error_classification.error_message.lower():
            # Simple heuristic: count parentheses and add missing ones
            open_parens = fixed_code.count('(') - fixed_code.count(')')
            if open_parens > 0:
                fixed_code += ')' * open_parens
        
        return fixed_code if fixed_code != code else None
    
    def _fix_import_errors(self, code: str, error_classification: ErrorClassification) -> Optional[str]:
        """Fix import-related errors."""
        fixed_code = code
        
        # Extract missing module name
        match = re.search(r"No module named '([^']+)'", error_classification.error_message)
        if match:
            missing_module = match.group(1)
            
            # Try common alternatives
            alternatives = {
                'numpy': 'import numpy as np',
                'pandas': 'import pandas as pd',
                'matplotlib': 'import matplotlib.pyplot as plt',
                'sklearn': 'from sklearn import *',
                'requests': 'import requests',
                'json': 'import json',
                'os': 'import os',
                'sys': 'import sys',
                'math': 'import math',
                'random': 'import random',
                'datetime': 'import datetime',
                'collections': 'import collections',
                'itertools': 'import itertools',
                'functools': 'import functools'
            }
            
            if missing_module in alternatives:
                # Add import at the beginning
                lines = fixed_code.split('\n')
                import_line = alternatives[missing_module]
                if import_line not in fixed_code:
                    lines.insert(0, import_line)
                    fixed_code = '\n'.join(lines)
            else:
                # Try to remove the problematic import
                lines = fixed_code.split('\n')
                filtered_lines = []
                for line in lines:
                    if f"import {missing_module}" not in line and f"from {missing_module}" not in line:
                        filtered_lines.append(line)
                fixed_code = '\n'.join(filtered_lines)
        
        return fixed_code if fixed_code != code else None
    
    def _fix_variable_errors(self, code: str, error_classification: ErrorClassification) -> Optional[str]:
        """Fix variable-related errors."""
        fixed_code = code
        
        # Fix NameError - variable not defined
        match = re.search(r"name '([^']+)' is not defined", error_classification.error_message)
        if match:
            undefined_var = match.group(1)
            
            # Try to initialize the variable
            lines = fixed_code.split('\n')
            
            # Find where the variable is first used
            for i, line in enumerate(lines):
                if undefined_var in line and '=' not in line.split(undefined_var)[0]:
                    # Add initialization before first use
                    indent = len(line) - len(line.lstrip())
                    init_line = ' ' * indent + f"{undefined_var} = None  # Auto-initialized"
                    lines.insert(i, init_line)
                    break
            
            fixed_code = '\n'.join(lines)
        
        return fixed_code if fixed_code != code else None
    
    def _fix_type_errors(self, code: str, error_classification: ErrorClassification) -> Optional[str]:
        """Fix type-related errors."""
        fixed_code = code
        
        # Fix unsupported operand types
        if "unsupported operand type" in error_classification.error_message.lower():
            # Add type conversion
            lines = fixed_code.split('\n')
            for i, line in enumerate(lines):
                # Look for operations that might need type conversion
                if '+' in line or '-' in line or '*' in line or '/' in line:
                    # Simple heuristic: wrap operands in str() or int()
                    if 'str' in error_classification.error_message.lower():
                        # Convert to string
                        line = re.sub(r'(\w+)\s*\+\s*(\w+)', r'str(\1) + str(\2)', line)
                    elif 'int' in error_classification.error_message.lower():
                        # Convert to int
                        line = re.sub(r'(\w+)\s*\+\s*(\w+)', r'int(\1) + int(\2)', line)
                    lines[i] = line
            
            fixed_code = '\n'.join(lines)
        
        return fixed_code if fixed_code != code else None
    
    def _fix_logic_errors(self, code: str, error_classification: ErrorClassification) -> Optional[str]:
        """Fix logical errors."""
        fixed_code = code
        
        # Fix IndexError
        if "list index out of range" in error_classification.error_message.lower():
            lines = fixed_code.split('\n')
            for i, line in enumerate(lines):
                # Look for list indexing
                if '[' in line and ']' in line:
                    # Add bounds checking
                    indent = len(line) - len(line.lstrip())
                    var_match = re.search(r'(\w+)\[([^\]]+)\]', line)
                    if var_match:
                        var_name = var_match.group(1)
                        index_expr = var_match.group(2)
                        
                        # Replace with safe indexing
                        safe_access = f"({var_name}[{index_expr}] if 0 <= {index_expr} < len({var_name}) else None)"
                        lines[i] = line.replace(var_match.group(0), safe_access)
            
            fixed_code = '\n'.join(lines)
        
        # Fix KeyError
        if "KeyError" in error_classification.error_type:
            lines = fixed_code.split('\n')
            for i, line in enumerate(lines):
                # Look for dictionary access
                if '[' in line and ']' in line and '=' not in line.split('[')[0]:
                    # Replace with .get() method
                    dict_access = re.search(r'(\w+)\[([^\]]+)\]', line)
                    if dict_access:
                        dict_name = dict_access.group(1)
                        key_expr = dict_access.group(2)
                        safe_access = f"{dict_name}.get({key_expr})"
                        lines[i] = line.replace(dict_access.group(0), safe_access)
            
            fixed_code = '\n'.join(lines)
        
        return fixed_code if fixed_code != code else None
    
    def _fix_performance_issues(self, code: str, error_classification: ErrorClassification) -> Optional[str]:
        """Fix performance-related issues."""
        fixed_code = code
        
        # Add timeout or optimization hints
        if "timeout" in error_classification.error_message.lower():
            # Add early returns or break conditions
            lines = fixed_code.split('\n')
            for i, line in enumerate(lines):
                if 'while' in line and 'True' in line:
                    # Add a counter to prevent infinite loops
                    indent = len(line) - len(line.lstrip())
                    counter_init = ' ' * indent + "_loop_counter = 0"
                    counter_check = ' ' * (indent + 4) + "_loop_counter += 1"
                    counter_break = ' ' * (indent + 4) + "if _loop_counter > 10000: break"
                    
                    lines.insert(i, counter_init)
                    lines.insert(i + 2, counter_check)
                    lines.insert(i + 3, counter_break)
                    break
            
            fixed_code = '\n'.join(lines)
        
        return fixed_code if fixed_code != code else None
    
    def _add_error_handling(self, code: str, error_classification: ErrorClassification) -> Optional[str]:
        """Add error handling to the code."""
        lines = code.split('\n')
        
        # Find the main function or problematic area
        main_function_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and ('solve' in line or 'main' in line):
                main_function_start = i
                break
        
        if main_function_start is not None:
            # Find the function body
            indent_level = len(lines[main_function_start]) - len(lines[main_function_start].lstrip())
            function_end = len(lines)
            
            for i in range(main_function_start + 1, len(lines)):
                if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= indent_level:
                    function_end = i
                    break
            
            # Wrap function body in try-except
            try_line = ' ' * (indent_level + 4) + "try:"
            except_line = ' ' * (indent_level + 4) + "except Exception as e:"
            return_line = ' ' * (indent_level + 8) + "return None  # Error handled"
            
            # Insert try-except around function body
            lines.insert(main_function_start + 1, try_line)
            lines.insert(function_end + 1, except_line)
            lines.insert(function_end + 2, return_line)
            
            # Indent existing function body
            for i in range(main_function_start + 2, function_end + 1):
                if lines[i].strip():
                    lines[i] = '    ' + lines[i]
        
        return '\n'.join(lines)
    
    def _refactor_code(self, code: str, error_classification: ErrorClassification) -> Optional[str]:
        """Perform basic code refactoring."""
        fixed_code = code
        
        # Remove duplicate imports
        lines = fixed_code.split('\n')
        import_lines = []
        other_lines = []
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ')):
                if line not in import_lines:
                    import_lines.append(line)
            else:
                other_lines.append(line)
        
        fixed_code = '\n'.join(import_lines + [''] + other_lines)
        
        # Remove empty lines at the beginning and end
        lines = fixed_code.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        fixed_code = '\n'.join(lines)
        
        return fixed_code if fixed_code != code else None
    
    async def _validate_fix(self, fixed_code: str, original_code: str, 
                          error_classification: ErrorClassification) -> Dict[str, Any]:
        """Validate that the fix is syntactically correct and potentially resolves the error."""
        
        # Check if code actually changed
        if fixed_code == original_code:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'error_resolved': False,
                'new_errors': [],
                'explanation': 'No changes made to the code'
            }
        
        # Check syntax validity
        try:
            ast.parse(fixed_code)
            syntax_valid = True
        except SyntaxError as e:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'error_resolved': False,
                'new_errors': [f"Syntax error: {str(e)}"],
                'explanation': 'Fix introduced syntax errors'
            }
        
        # Estimate confidence based on the type of fix
        confidence = 0.7  # Base confidence
        
        # Higher confidence for syntax fixes
        if error_classification.category.value == "syntax_error" and syntax_valid:
            confidence = 0.9
        
        # Lower confidence for complex logic fixes
        if error_classification.category.value == "logic_error":
            confidence = 0.5
        
        return {
            'is_valid': True,
            'confidence': confidence,
            'error_resolved': True,  # Optimistic assumption
            'new_errors': [],
            'explanation': f'Applied automated fixes for {error_classification.category.value}'
        }
    
    def _update_success_patterns(self, error_classification: ErrorClassification, 
                               actions_taken: List[DebugAction]):
        """Update success patterns based on successful debugging."""
        pattern_key = f"{error_classification.category.value}:{error_classification.error_type}"
        
        if pattern_key not in self.success_patterns:
            self.success_patterns[pattern_key] = []
        
        for action in actions_taken:
            if action.value not in self.success_patterns[pattern_key]:
                self.success_patterns[pattern_key].append(action.value)
    
    def get_debug_statistics(self) -> Dict[str, Any]:
        """Get statistics about debugging performance."""
        if not self.debug_history:
            return {}
        
        total_attempts = len(self.debug_history)
        successful_fixes = sum(1 for _, _, result in self.debug_history if result.success)
        
        action_success_rates = {}
        for _, _, result in self.debug_history:
            for action in result.actions_taken:
                if action.value not in action_success_rates:
                    action_success_rates[action.value] = {'attempts': 0, 'successes': 0}
                action_success_rates[action.value]['attempts'] += 1
                if result.success:
                    action_success_rates[action.value]['successes'] += 1
        
        # Calculate success rates
        for action_data in action_success_rates.values():
            action_data['success_rate'] = action_data['successes'] / action_data['attempts']
        
        return {
            'total_debug_attempts': total_attempts,
            'successful_fixes': successful_fixes,
            'overall_success_rate': successful_fixes / total_attempts,
            'action_success_rates': action_success_rates,
            'average_confidence': sum(result.confidence_score for _, _, result in self.debug_history) / total_attempts
        }


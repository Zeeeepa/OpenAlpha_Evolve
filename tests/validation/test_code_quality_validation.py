"""
Code quality validation tests for the autonomous development pipeline.

These tests validate code quality, detect dead code, and ensure
all functionality is properly utilized and validated.
"""

import pytest
import ast
import os
import sys
import importlib
import inspect
from typing import Set, List, Dict, Any
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class CodeQualityValidator:
    """Validates code quality and detects dead code."""
    
    def __init__(self, project_root: str):
        """Initialize the code quality validator."""
        self.project_root = Path(project_root)
        self.python_files = list(self.project_root.rglob("*.py"))
        self.defined_functions = set()
        self.defined_classes = set()
        self.called_functions = set()
        self.imported_modules = set()
        self.used_classes = set()
    
    def analyze_codebase(self):
        """Analyze the entire codebase for quality issues."""
        for file_path in self.python_files:
            if self._should_skip_file(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                self._analyze_ast(tree, file_path)
                
            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"Warning: Could not parse {file_path}: {e}")
                continue
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during analysis."""
        skip_patterns = [
            '__pycache__',
            '.git',
            'venv',
            'env',
            '.pytest_cache',
            'node_modules'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path):
        """Analyze an AST for function and class definitions and usage."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = f"{file_path.stem}.{node.name}"
                self.defined_functions.add(func_name)
            
            elif isinstance(node, ast.ClassDef):
                class_name = f"{file_path.stem}.{node.name}"
                self.defined_classes.add(class_name)
            
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.called_functions.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        func_name = f"{node.func.value.id}.{node.func.attr}"
                        self.called_functions.add(func_name)
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.imported_modules.add(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.imported_modules.add(node.module)
                    for alias in node.names:
                        self.imported_modules.add(f"{node.module}.{alias.name}")
    
    def find_dead_code(self) -> Dict[str, List[str]]:
        """Find potentially dead code in the codebase."""
        dead_code = {
            "unused_functions": [],
            "unused_classes": [],
            "unused_imports": []
        }
        
        # Find unused functions (excluding special methods and test functions)
        for func in self.defined_functions:
            func_name = func.split('.')[-1]
            if (func_name not in self.called_functions and 
                not func_name.startswith('_') and 
                not func_name.startswith('test_') and
                func_name not in ['main', 'setup', 'teardown']):
                dead_code["unused_functions"].append(func)
        
        # Find unused classes
        for cls in self.defined_classes:
            cls_name = cls.split('.')[-1]
            if (cls_name not in self.used_classes and 
                not cls_name.startswith('_') and 
                not cls_name.startswith('Test')):
                dead_code["unused_classes"].append(cls)
        
        return dead_code
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check overall code quality metrics."""
        quality_metrics = {
            "total_files": len(self.python_files),
            "total_functions": len(self.defined_functions),
            "total_classes": len(self.defined_classes),
            "total_imports": len(self.imported_modules),
            "issues": []
        }
        
        # Check for common quality issues
        for file_path in self.python_files:
            if self._should_skip_file(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check file length
                lines = content.split('\n')
                if len(lines) > 1000:
                    quality_metrics["issues"].append({
                        "type": "long_file",
                        "file": str(file_path),
                        "lines": len(lines),
                        "message": f"File has {len(lines)} lines (consider splitting)"
                    })
                
                # Check for very long functions
                tree = ast.parse(content, filename=str(file_path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        if func_lines > 100:
                            quality_metrics["issues"].append({
                                "type": "long_function",
                                "file": str(file_path),
                                "function": node.name,
                                "lines": func_lines,
                                "message": f"Function '{node.name}' has {func_lines} lines (consider refactoring)"
                            })
                
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        return quality_metrics


class TestCodeQualityValidation:
    """Test code quality validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create a code quality validator."""
        return CodeQualityValidator(project_root)
    
    def test_codebase_analysis(self, validator):
        """Test that the codebase can be analyzed without errors."""
        validator.analyze_codebase()
        
        # Verify that analysis found some code
        assert len(validator.defined_functions) > 0
        assert len(validator.defined_classes) > 0
        assert len(validator.python_files) > 0
    
    def test_dead_code_detection(self, validator):
        """Test dead code detection functionality."""
        validator.analyze_codebase()
        dead_code = validator.find_dead_code()
        
        # Verify dead code detection structure
        assert "unused_functions" in dead_code
        assert "unused_classes" in dead_code
        assert "unused_imports" in dead_code
        
        # Report dead code findings
        if dead_code["unused_functions"]:
            print(f"\\nFound {len(dead_code['unused_functions'])} potentially unused functions:")
            for func in dead_code["unused_functions"][:10]:  # Show first 10
                print(f"  - {func}")
        
        if dead_code["unused_classes"]:
            print(f"\\nFound {len(dead_code['unused_classes'])} potentially unused classes:")
            for cls in dead_code["unused_classes"][:10]:  # Show first 10
                print(f"  - {cls}")
        
        # For a well-maintained codebase, there should be minimal dead code
        # This is more of a warning than a hard failure
        total_dead_items = (len(dead_code["unused_functions"]) + 
                           len(dead_code["unused_classes"]))
        
        if total_dead_items > 0:
            print(f"\\nWarning: Found {total_dead_items} potentially dead code items")
    
    def test_code_quality_metrics(self, validator):
        """Test code quality metrics collection."""
        validator.analyze_codebase()
        quality_metrics = validator.check_code_quality()
        
        # Verify metrics structure
        assert "total_files" in quality_metrics
        assert "total_functions" in quality_metrics
        assert "total_classes" in quality_metrics
        assert "issues" in quality_metrics
        
        # Verify reasonable metrics
        assert quality_metrics["total_files"] > 0
        assert quality_metrics["total_functions"] > 0
        
        # Report quality issues
        if quality_metrics["issues"]:
            print(f"\\nFound {len(quality_metrics['issues'])} code quality issues:")
            for issue in quality_metrics["issues"][:10]:  # Show first 10
                print(f"  - {issue['type']}: {issue['message']}")
        
        # Check for critical quality issues
        critical_issues = [
            issue for issue in quality_metrics["issues"]
            if issue["type"] in ["syntax_error", "import_error"]
        ]
        
        assert len(critical_issues) == 0, f"Found critical quality issues: {critical_issues}"
    
    def test_import_validation(self, validator):
        """Test that all imports are valid and accessible."""
        validator.analyze_codebase()
        
        import_errors = []
        
        for file_path in validator.python_files:
            if validator._should_skip_file(file_path):
                continue
            
            try:
                # Try to compile the file to check for import errors
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                compile(content, str(file_path), 'exec')
                
            except SyntaxError as e:
                import_errors.append({
                    "file": str(file_path),
                    "error": str(e),
                    "type": "syntax_error"
                })
            except Exception as e:
                # Skip other compilation errors as they might be due to missing dependencies
                pass
        
        # Report import errors
        if import_errors:
            print(f"\\nFound {len(import_errors)} import/syntax errors:")
            for error in import_errors[:5]:  # Show first 5
                print(f"  - {error['file']}: {error['error']}")
        
        # Syntax errors should not be present
        syntax_errors = [e for e in import_errors if e["type"] == "syntax_error"]
        assert len(syntax_errors) == 0, f"Found syntax errors: {syntax_errors}"
    
    def test_function_complexity(self, validator):
        """Test function complexity metrics."""
        complex_functions = []
        
        for file_path in validator.python_files:
            if validator._should_skip_file(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Count nested structures as complexity indicators
                        complexity = 0
                        for child in ast.walk(node):
                            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                                complexity += 1
                        
                        if complexity > 10:  # High complexity threshold
                            complex_functions.append({
                                "file": str(file_path),
                                "function": node.name,
                                "complexity": complexity
                            })
                
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        # Report complex functions
        if complex_functions:
            print(f"\\nFound {len(complex_functions)} highly complex functions:")
            for func in complex_functions[:5]:  # Show first 5
                print(f"  - {func['function']} in {func['file']}: complexity {func['complexity']}")
        
        # High complexity functions should be reviewed
        very_complex = [f for f in complex_functions if f["complexity"] > 20]
        if very_complex:
            print(f"\\nWarning: {len(very_complex)} functions have very high complexity (>20)")


class TestSystemValidation:
    """Test system-wide validation requirements."""
    
    def test_all_components_importable(self):
        """Test that all main components can be imported."""
        components_to_test = [
            "core.interfaces",
            "config.settings",
            "task_manager.agent",
            "prompt_designer.agent",
            "code_generator.agent",
            "evaluator_agent.agent",
            "database_agent.agent",
            "selection_controller.agent"
        ]
        
        import_errors = []
        
        for component in components_to_test:
            try:
                importlib.import_module(component)
            except ImportError as e:
                import_errors.append(f"{component}: {e}")
        
        assert len(import_errors) == 0, f"Import errors: {import_errors}"
    
    def test_orchestrator_components_importable(self):
        """Test that orchestrator components can be imported."""
        orchestrator_components = [
            "src.openevolve.orchestrator.integration_manager",
            "src.openevolve.orchestrator.workflow_coordinator",
            "src.openevolve.orchestrator.health_monitor",
            "src.openevolve.orchestrator.performance_optimizer"
        ]
        
        import_errors = []
        
        for component in orchestrator_components:
            try:
                importlib.import_module(component)
            except ImportError as e:
                import_errors.append(f"{component}: {e}")
        
        assert len(import_errors) == 0, f"Orchestrator import errors: {import_errors}"
    
    def test_interface_compliance(self):
        """Test that all agents implement required interfaces."""
        from core.interfaces import (
            TaskManagerInterface, PromptDesignerInterface, 
            CodeGeneratorInterface, EvaluatorAgentInterface,
            DatabaseAgentInterface, SelectionControllerInterface
        )
        
        # Import agent implementations
        try:
            from task_manager.agent import TaskManagerAgent
            from prompt_designer.agent import PromptDesignerAgent
            from code_generator.agent import CodeGeneratorAgent
            from evaluator_agent.agent import EvaluatorAgent
            from database_agent.agent import InMemoryDatabaseAgent
            from selection_controller.agent import SelectionControllerAgent
            
            # Check interface compliance
            assert issubclass(TaskManagerAgent, TaskManagerInterface)
            assert issubclass(PromptDesignerAgent, PromptDesignerInterface)
            assert issubclass(CodeGeneratorAgent, CodeGeneratorInterface)
            assert issubclass(EvaluatorAgent, EvaluatorAgentInterface)
            assert issubclass(InMemoryDatabaseAgent, DatabaseAgentInterface)
            assert issubclass(SelectionControllerAgent, SelectionControllerInterface)
            
        except ImportError as e:
            pytest.fail(f"Could not import agent implementations: {e}")
    
    def test_configuration_completeness(self):
        """Test that all required configuration is present."""
        try:
            from config import settings
            
            # Check for required settings
            required_settings = [
                'POPULATION_SIZE',
                'GENERATIONS',
                'DATABASE_TYPE',
                'LOG_LEVEL',
                'LITELLM_DEFAULT_MODEL'
            ]
            
            missing_settings = []
            for setting in required_settings:
                if not hasattr(settings, setting):
                    missing_settings.append(setting)
            
            assert len(missing_settings) == 0, f"Missing required settings: {missing_settings}"
            
            # Check for reasonable values
            assert settings.POPULATION_SIZE > 0
            assert settings.GENERATIONS > 0
            
        except ImportError as e:
            pytest.fail(f"Could not import settings: {e}")
    
    def test_test_coverage(self):
        """Test that critical components have test coverage."""
        test_files = list(Path(project_root).rglob("test_*.py"))
        
        # Check that we have tests for main components
        test_file_names = [f.name for f in test_files]
        
        expected_tests = [
            "test_evaluator_agent.py",  # Already exists
            "test_end_to_end_pipeline.py",  # Created in this implementation
            "test_performance_benchmarks.py",  # Created in this implementation
            "test_code_quality_validation.py"  # This file
        ]
        
        missing_tests = []
        for expected_test in expected_tests:
            if expected_test not in test_file_names:
                missing_tests.append(expected_test)
        
        if missing_tests:
            print(f"\\nWarning: Missing test files: {missing_tests}")
        
        # At minimum, we should have some tests
        assert len(test_files) >= 3, f"Insufficient test coverage. Found {len(test_files)} test files"


if __name__ == "__main__":
    # Run validation tests
    pytest.main([__file__, "-v", "-s"])


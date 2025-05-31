"""
Comprehensive tests for the Context Analysis Engine.

Tests all major components including semantic analysis, requirement processing,
recommendation generation, and integration capabilities.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

# Import the Context Analysis Engine components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from openevolve.analysis import (
    ContextAnalysisEngine, AnalysisConfig, AnalysisType, LanguageType,
    CodeContext, RequirementContext, AnalysisResult
)


class TestContextAnalysisEngine:
    """Test suite for the Context Analysis Engine."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AnalysisConfig(
            enabled_analyses=[AnalysisType.SEMANTIC, AnalysisType.COMPLEXITY, AnalysisType.QUALITY],
            supported_languages=[LanguageType.PYTHON, LanguageType.JAVASCRIPT],
            enable_caching=False,  # Disable caching for tests
            parallel_processing=False,  # Disable parallel processing for deterministic tests
            max_workers=1
        )
    
    @pytest.fixture
    def engine(self, config):
        """Create Context Analysis Engine instance."""
        return ContextAnalysisEngine(config)
    
    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code for testing."""
        return '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class MathUtils:
    """Utility class for mathematical operations."""
    
    @staticmethod
    def factorial(n):
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n-1)
    
    def power(self, base, exponent):
        result = 1
        for i in range(exponent):
            result *= base
        return result

# Main execution
if __name__ == "__main__":
    print(calculate_fibonacci(10))
    utils = MathUtils()
    print(utils.power(2, 8))
'''
    
    @pytest.fixture
    def sample_javascript_code(self):
        """Sample JavaScript code for testing."""
        return '''
function calculateSum(numbers) {
    let sum = 0;
    for (let i = 0; i < numbers.length; i++) {
        sum += numbers[i];
    }
    return sum;
}

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
    
    getHistory() {
        return this.history;
    }
}

const calc = new Calculator();
console.log(calc.add(5, 3));
'''
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test that the engine initializes correctly."""
        assert engine is not None
        assert engine.config is not None
        assert engine.semantic_analyzer is not None
        assert engine.requirement_processor is not None
        assert engine.recommendation_engine is not None
        assert engine.graph_sitter_parser is not None
    
    @pytest.mark.asyncio
    async def test_python_code_analysis(self, engine, sample_python_code):
        """Test analysis of Python code."""
        result = await engine.analyze(sample_python_code, "test.py")
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_id is not None
        assert len(result.code_contexts) == 1
        assert result.code_contexts[0].language == LanguageType.PYTHON
        assert len(result.errors) == 0
        
        # Check that elements were extracted
        code_context = result.code_contexts[0]
        assert len(code_context.elements) > 0
        
        # Check for specific elements
        element_names = [elem.name for elem in code_context.elements]
        assert 'calculate_fibonacci' in element_names
        assert 'MathUtils' in element_names
    
    @pytest.mark.asyncio
    async def test_javascript_code_analysis(self, engine, sample_javascript_code):
        """Test analysis of JavaScript code."""
        result = await engine.analyze(sample_javascript_code, "test.js")
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_id is not None
        assert len(result.code_contexts) == 1
        assert result.code_contexts[0].language == LanguageType.JAVASCRIPT
        assert len(result.errors) == 0
        
        # Check that elements were extracted
        code_context = result.code_contexts[0]
        assert len(code_context.elements) > 0
        
        # Check for specific elements
        element_names = [elem.name for elem in code_context.elements]
        assert 'calculateSum' in element_names
        assert 'Calculator' in element_names
    
    @pytest.mark.asyncio
    async def test_complexity_analysis(self, engine, sample_python_code):
        """Test complexity analysis functionality."""
        result = await engine.analyze(sample_python_code, "test.py")
        
        code_context = result.code_contexts[0]
        assert code_context.complexity_metrics is not None
        assert 'cyclomatic_complexity' in code_context.complexity_metrics
        assert code_context.complexity_metrics['cyclomatic_complexity'] > 0
    
    @pytest.mark.asyncio
    async def test_quality_analysis(self, engine, sample_python_code):
        """Test code quality analysis."""
        result = await engine.analyze(sample_python_code, "test.py")
        
        code_context = result.code_contexts[0]
        assert code_context.quality_metrics is not None
        assert 'comment_ratio' in code_context.quality_metrics
        assert 'total_lines' in code_context.quality_metrics
    
    @pytest.mark.asyncio
    async def test_pattern_detection(self, engine, sample_python_code):
        """Test pattern detection in code."""
        result = await engine.analyze(sample_python_code, "test.py")
        
        code_context = result.code_contexts[0]
        assert len(code_context.patterns) > 0
        
        # Should detect main guard pattern
        assert 'main_guard' in code_context.patterns
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, engine):
        """Test recommendation generation."""
        # Create code with high complexity to trigger recommendations
        complex_code = '''
def complex_function(a, b, c, d, e, f, g, h, i, j):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            if g > 0:
                                if h > 0:
                                    if i > 0:
                                        if j > 0:
                                            return a + b + c + d + e + f + g + h + i + j
                                        else:
                                            return 0
                                    else:
                                        return 0
                                else:
                                    return 0
                            else:
                                return 0
                        else:
                            return 0
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0
        else:
            return 0
    else:
        return 0
'''
        
        result = await engine.analyze(complex_code, "complex.py")
        
        # Should generate recommendations for high complexity
        assert len(result.recommendations) > 0
        
        # Check for specific recommendation types
        rec_types = [rec.type for rec in result.recommendations]
        assert 'refactoring' in rec_types
    
    @pytest.mark.asyncio
    async def test_requirement_analysis(self, engine):
        """Test requirement analysis functionality."""
        requirements = """
        Add a new user authentication feature that allows users to login with email and password.
        The system should validate credentials against the database and return a JWT token.
        This is a high priority feature that needs to be implemented securely.
        """
        
        requirement_context = await engine.analyze_requirements(requirements)
        
        assert isinstance(requirement_context, RequirementContext)
        assert requirement_context.type in ['feature', 'security']
        assert requirement_context.priority >= 3  # High priority
        assert len(requirement_context.affected_components) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_file_analysis(self, engine, sample_python_code, sample_javascript_code):
        """Test analysis of multiple files."""
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            python_file = os.path.join(temp_dir, "test.py")
            js_file = os.path.join(temp_dir, "test.js")
            
            with open(python_file, 'w') as f:
                f.write(sample_python_code)
            
            with open(js_file, 'w') as f:
                f.write(sample_javascript_code)
            
            results = await engine.analyze_multiple_files([python_file, js_file])
            
            assert len(results) == 2
            assert all(isinstance(result, AnalysisResult) for result in results)
            
            # Check that different languages were detected
            languages = [result.code_contexts[0].language for result in results]
            assert LanguageType.PYTHON in languages
            assert LanguageType.JAVASCRIPT in languages
    
    @pytest.mark.asyncio
    async def test_dependency_graph_construction(self, engine):
        """Test dependency graph construction."""
        code_with_dependencies = '''
import math
from datetime import datetime

def calculate_area(radius):
    return math.pi * radius ** 2

def get_current_time():
    return datetime.now()

class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return calculate_area(self.radius)
    
    def info(self):
        return f"Circle area: {self.area()} at {get_current_time()}"
'''
        
        result = await engine.analyze(code_with_dependencies, "dependencies.py")
        
        assert result.dependency_graph is not None
        assert len(result.dependency_graph.nodes) > 0
        assert len(result.dependency_graph.edges) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling with invalid code."""
        invalid_code = '''
def broken_function(
    # Missing closing parenthesis and body
'''
        
        result = await engine.analyze(invalid_code, "broken.py")
        
        # Should still return a result, possibly with errors
        assert isinstance(result, AnalysisResult)
        # May have errors but should not crash
    
    @pytest.mark.asyncio
    async def test_language_detection(self, engine):
        """Test automatic language detection."""
        # Test with no file extension
        result = await engine.analyze("print('Hello, World!')", "unknown_file")
        
        # Should detect Python based on content
        assert result.code_contexts[0].language == LanguageType.PYTHON
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, engine, sample_python_code):
        """Test that performance metrics are collected."""
        result = await engine.analyze(sample_python_code, "test.py")
        
        assert result.execution_time > 0
        assert result.created_at > 0
    
    @pytest.mark.asyncio
    async def test_configuration_override(self, sample_python_code):
        """Test configuration override functionality."""
        # Create engine with default config
        engine = ContextAnalysisEngine()
        
        # Override config for specific analysis
        custom_config = AnalysisConfig(
            enabled_analyses=[AnalysisType.SEMANTIC],  # Only semantic analysis
            enable_caching=False
        )
        
        result = await engine.analyze(sample_python_code, "test.py", custom_config)
        
        assert isinstance(result, AnalysisResult)
        # Should only have semantic analysis results
        code_context = result.code_contexts[0]
        # Complexity metrics might not be calculated with limited analysis
    
    @pytest.mark.asyncio
    async def test_requirement_to_code_mapping(self, engine, sample_python_code):
        """Test mapping requirements to existing code."""
        # First analyze the code
        code_result = await engine.analyze(sample_python_code, "math_utils.py")
        
        # Create a requirement related to the code
        requirement = RequirementContext(
            id="req_1",
            description="Optimize the Fibonacci calculation to use memoization for better performance",
            type="enhancement",
            affected_components=["fibonacci", "calculation"]
        )
        
        # Map requirement to code
        mapping_result = await engine.map_requirements_to_code(
            requirement, code_result.code_contexts
        )
        
        assert mapping_result is not None
        assert 'mappings' in mapping_result
        assert len(mapping_result['mappings']) > 0
        
        # Should find relevance to fibonacci function
        relevant_files = [m['file_path'] for m in mapping_result['mappings']]
        assert "math_utils.py" in relevant_files


class TestSemanticAnalyzer:
    """Test suite for the Semantic Analyzer component."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AnalysisConfig()
    
    @pytest.mark.asyncio
    async def test_pattern_extraction(self, config):
        """Test pattern extraction functionality."""
        from openevolve.analysis.semantic.analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer(config)
        
        code_with_patterns = '''
@decorator
def singleton_function():
    pass

class Factory:
    def create_object(self):
        return Object()

async def async_function():
    await some_operation()

try:
    risky_operation()
except Exception as e:
    logger.error(f"Error: {e}")
finally:
    cleanup()
'''
        
        code_context = CodeContext(
            file_path="patterns.py",
            content=code_with_patterns,
            language=LanguageType.PYTHON
        )
        
        patterns = await analyzer.extract_patterns(code_context)
        
        assert 'decorator' in patterns
        assert 'factory' in patterns
        assert 'async_pattern' in patterns
        assert 'error_handling' in patterns


class TestRequirementProcessor:
    """Test suite for the Requirement Processor component."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AnalysisConfig()
    
    @pytest.mark.asyncio
    async def test_requirement_classification(self, config):
        """Test requirement type classification."""
        from openevolve.analysis.intelligence.processor import RequirementProcessor
        
        processor = RequirementProcessor(config)
        
        # Test different requirement types
        test_cases = [
            ("Fix the login bug that prevents users from accessing their accounts", "bug_fix"),
            ("Add a new dashboard feature for analytics", "feature"),
            ("Improve the performance of the search functionality", "enhancement"),
            ("Implement secure authentication with two-factor verification", "security")
        ]
        
        for requirement_text, expected_type in test_cases:
            result = await processor.process_requirements(requirement_text)
            assert result.type == expected_type
    
    @pytest.mark.asyncio
    async def test_priority_extraction(self, config):
        """Test priority extraction from requirements."""
        from openevolve.analysis.intelligence.processor import RequirementProcessor
        
        processor = RequirementProcessor(config)
        
        high_priority_req = "URGENT: Fix critical security vulnerability in authentication system"
        result = await processor.process_requirements(high_priority_req)
        
        assert result.priority >= 4  # Should be high priority
    
    @pytest.mark.asyncio
    async def test_task_decomposition(self, config):
        """Test task decomposition functionality."""
        from openevolve.analysis.intelligence.processor import RequirementProcessor
        
        processor = RequirementProcessor(config)
        
        complex_requirement = RequirementContext(
            id="complex_req",
            description="""
            Implement user management system with the following features:
            1. User registration with email verification
            2. User login with password authentication
            3. Password reset functionality
            4. User profile management
            5. Admin panel for user administration
            """,
            type="feature"
        )
        
        subtasks = await processor.decompose_task(complex_requirement)
        
        assert len(subtasks) >= 3  # Should break down into multiple subtasks
        assert all(subtask.metadata.get('is_subtask', False) for subtask in subtasks)


class TestRecommendationEngine:
    """Test suite for the Recommendation Engine component."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AnalysisConfig()
    
    @pytest.mark.asyncio
    async def test_optimization_suggestions(self, config):
        """Test optimization suggestion generation."""
        from openevolve.analysis.intelligence.recommender import RecommendationEngine
        
        engine = RecommendationEngine(config)
        
        # Create code context with optimization opportunities
        inefficient_code = '''
def inefficient_function():
    result = ""
    for i in range(1000):
        result += str(i)  # Inefficient string concatenation
    return result

def another_function():
    data = [1, 2, 3, 4, 5]
    for i in range(len(data)):  # Inefficient iteration
        print(data[i])
'''
        
        code_context = CodeContext(
            file_path="inefficient.py",
            content=inefficient_code,
            language=LanguageType.PYTHON
        )
        
        recommendations = await engine.suggest_optimizations(code_context)
        
        assert len(recommendations) > 0
        assert any(rec.type == "performance" for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_security_recommendations(self, config):
        """Test security recommendation generation."""
        from openevolve.analysis.intelligence.recommender import RecommendationEngine
        
        engine = RecommendationEngine(config)
        
        # Code with security issues
        insecure_code = '''
password = "hardcoded_password"
api_key = "sk-1234567890abcdef"

def execute_query(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    return execute(query)

def unsafe_operation(code):
    return eval(code)  # Unsafe eval usage
'''
        
        code_context = CodeContext(
            file_path="insecure.py",
            content=insecure_code,
            language=LanguageType.PYTHON
        )
        
        recommendations = await engine.suggest_optimizations(code_context)
        
        assert len(recommendations) > 0
        assert any(rec.type == "security" for rec in recommendations)
        assert any(rec.priority == 5 for rec in recommendations)  # High priority security issues


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])


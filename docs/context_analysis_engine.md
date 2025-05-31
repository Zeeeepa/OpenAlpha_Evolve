# Context Analysis Engine Documentation

## Overview

The Context Analysis Engine is a comprehensive code analysis and intelligence system designed for autonomous development pipelines. It provides semantic code understanding, requirement processing, and intelligent recommendations to support automated software development workflows.

## Features

### 🔍 **Semantic Code Analysis**
- Multi-language AST parsing and traversal
- Code element extraction (functions, classes, variables)
- Pattern recognition and code quality assessment
- Complexity metrics calculation (cyclomatic, cognitive, Halstead)

### 🧠 **Intelligence & Automation**
- Natural language requirement processing
- Requirement-to-code mapping
- Task decomposition and prioritization
- Impact analysis for code changes

### 💡 **Recommendation Engine**
- Code optimization suggestions
- Refactoring recommendations
- Security vulnerability detection
- Performance improvement suggestions

### 🕸️ **Dependency Analysis**
- Dependency graph construction
- Circular dependency detection
- Module coupling analysis
- Change impact assessment

### 🚀 **Integration Capabilities**
- Tree-sitter parser integration
- Multi-language support framework
- Caching system for performance
- Configurable analysis pipeline

## Architecture

```
Context Analysis Engine
├── Core Components
│   ├── ContextAnalysisEngine (Main orchestrator)
│   ├── AnalysisConfig (Configuration management)
│   └── Interfaces (Data structures and contracts)
├── Semantic Analysis
│   ├── SemanticAnalyzer (Code understanding)
│   ├── ASTProcessor (AST traversal)
│   └── ComplexityCalculator (Metrics calculation)
├── Intelligence Layer
│   ├── RequirementProcessor (NLP processing)
│   ├── RecommendationEngine (Suggestion generation)
│   └── ImpactAnalyzer (Change analysis)
├── Integration
│   ├── GraphSitterParser (Tree-sitter integration)
│   └── LanguageParserFactory (Multi-language support)
└── Utilities
    ├── CacheManager (Performance optimization)
    ├── LanguageDetector (Auto-detection)
    └── MetricsCollector (Analytics)
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install tree-sitter languages (optional)
pip install tree-sitter-python tree-sitter-javascript tree-sitter-typescript
```

### Basic Usage

```python
import asyncio
from openevolve.analysis import ContextAnalysisEngine, AnalysisConfig

async def analyze_code():
    # Create configuration
    config = AnalysisConfig(
        enabled_analyses=['semantic', 'complexity', 'quality'],
        supported_languages=['python', 'javascript'],
        enable_caching=True
    )
    
    # Initialize engine
    engine = ContextAnalysisEngine(config)
    
    # Analyze code
    code = '''
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    '''
    
    result = await engine.analyze(code, "fibonacci.py")
    
    # Access results
    print(f"Language: {result.code_contexts[0].language}")
    print(f"Elements: {len(result.code_contexts[0].elements)}")
    print(f"Recommendations: {len(result.recommendations)}")

# Run analysis
asyncio.run(analyze_code())
```

## Configuration

### AnalysisConfig Options

```python
config = AnalysisConfig(
    # Analysis settings
    enabled_analyses=[
        AnalysisType.SEMANTIC,
        AnalysisType.COMPLEXITY,
        AnalysisType.QUALITY,
        AnalysisType.PATTERN,
        AnalysisType.DEPENDENCY
    ],
    supported_languages=[
        LanguageType.PYTHON,
        LanguageType.JAVASCRIPT,
        LanguageType.TYPESCRIPT
    ],
    
    # Performance settings
    max_file_size=10 * 1024 * 1024,  # 10MB
    max_analysis_time=300,  # 5 minutes
    parallel_processing=True,
    max_workers=4,
    
    # Caching settings
    enable_caching=True,
    cache_ttl=3600,  # 1 hour
    cache_backend="memory",  # memory, redis, file
    
    # Quality thresholds
    complexity_threshold=10.0,
    maintainability_threshold=0.7,
    
    # Output settings
    include_source_code=False,
    verbose_output=False
)
```

## API Reference

### ContextAnalysisEngine

The main orchestrator for all analysis operations.

#### Methods

##### `analyze(content: str, file_path: str, config: Optional[AnalysisConfig] = None) -> AnalysisResult`

Perform comprehensive analysis on source code.

**Parameters:**
- `content`: Source code content to analyze
- `file_path`: Path to the file (used for language detection)
- `config`: Optional configuration override

**Returns:**
- `AnalysisResult`: Complete analysis results

**Example:**
```python
result = await engine.analyze(python_code, "example.py")
```

##### `analyze_multiple_files(file_paths: List[str], config: Optional[AnalysisConfig] = None) -> List[AnalysisResult]`

Analyze multiple files in parallel.

**Parameters:**
- `file_paths`: List of file paths to analyze
- `config`: Optional configuration override

**Returns:**
- `List[AnalysisResult]`: Results for each file

##### `analyze_requirements(requirements: str) -> RequirementContext`

Process natural language requirements.

**Parameters:**
- `requirements`: Natural language requirements text

**Returns:**
- `RequirementContext`: Structured requirement analysis

##### `map_requirements_to_code(requirement: RequirementContext, code_contexts: List[CodeContext]) -> Dict[str, Any]`

Map requirements to existing code components.

**Parameters:**
- `requirement`: Analyzed requirement context
- `code_contexts`: List of code contexts to map against

**Returns:**
- `Dict[str, Any]`: Mapping results with relevance scores

### Data Structures

#### AnalysisResult

Complete result of context analysis.

```python
@dataclass
class AnalysisResult:
    analysis_id: str
    analysis_type: AnalysisType
    code_contexts: List[CodeContext]
    requirement_contexts: List[RequirementContext]
    dependency_graph: Optional[DependencyGraph]
    impact_analysis: Optional[ImpactAnalysis]
    recommendations: List[Recommendation]
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    execution_time: float
    created_at: float
```

#### CodeContext

Represents the context of analyzed code.

```python
@dataclass
class CodeContext:
    file_path: str
    content: str
    language: LanguageType
    elements: List[CodeElement]
    imports: List[str]
    exports: List[str]
    complexity_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    patterns: List[str]
    metadata: Dict[str, Any]
    created_at: float
```

#### RequirementContext

Represents analyzed requirements.

```python
@dataclass
class RequirementContext:
    id: str
    description: str
    type: str  # feature, bug_fix, enhancement, etc.
    priority: int
    complexity_estimate: Optional[float]
    affected_components: List[str]
    dependencies: List[str]
    acceptance_criteria: List[str]
    technical_constraints: List[str]
    suggested_approach: Optional[str]
    estimated_effort: Optional[float]
    risk_factors: List[str]
    metadata: Dict[str, Any]
    created_at: float
```

#### Recommendation

Represents a recommendation from the analysis engine.

```python
@dataclass
class Recommendation:
    id: str
    type: str  # optimization, refactoring, bug_fix, etc.
    title: str
    description: str
    priority: int
    confidence: float
    affected_files: List[str]
    suggested_changes: List[str]
    rationale: str
    metadata: Dict[str, Any]
```

## Supported Languages

| Language   | Extension | AST Support | Pattern Recognition | Complexity Analysis |
|------------|-----------|-------------|-------------------|-------------------|
| Python     | .py       | ✅ Full     | ✅ Advanced       | ✅ Complete       |
| JavaScript | .js, .jsx | ✅ Full     | ✅ Advanced       | ✅ Complete       |
| TypeScript | .ts, .tsx | ✅ Full     | ✅ Advanced       | ✅ Complete       |
| Java       | .java     | ⚠️ Basic    | ✅ Advanced       | ✅ Complete       |
| C++        | .cpp, .h  | ⚠️ Basic    | ⚠️ Limited        | ⚠️ Basic          |
| Rust       | .rs       | ⚠️ Basic    | ⚠️ Limited        | ⚠️ Basic          |
| Go         | .go       | ⚠️ Basic    | ⚠️ Limited        | ⚠️ Basic          |

## Analysis Types

### Semantic Analysis
- **Purpose**: Understand code structure and meaning
- **Output**: Code elements, imports, exports, relationships
- **Use Cases**: Code navigation, refactoring, documentation

### Complexity Analysis
- **Metrics**: Cyclomatic complexity, cognitive complexity, Halstead metrics
- **Purpose**: Identify complex code that may need refactoring
- **Thresholds**: Configurable complexity thresholds for recommendations

### Quality Analysis
- **Metrics**: Comment ratio, line length, maintainability index
- **Purpose**: Assess code quality and maintainability
- **Output**: Quality scores and improvement suggestions

### Pattern Analysis
- **Patterns**: Design patterns, anti-patterns, language-specific patterns
- **Purpose**: Identify architectural patterns and potential issues
- **Examples**: Singleton, Factory, Observer, Error handling patterns

### Dependency Analysis
- **Purpose**: Map code dependencies and relationships
- **Output**: Dependency graph with nodes and edges
- **Features**: Circular dependency detection, coupling analysis

## Caching

The Context Analysis Engine supports multiple caching backends for improved performance:

### Memory Cache
- **Use Case**: Single-process applications
- **Performance**: Fastest access
- **Persistence**: No persistence across restarts

### File Cache
- **Use Case**: Development environments
- **Performance**: Good performance
- **Persistence**: Survives restarts

### Redis Cache
- **Use Case**: Production environments, distributed systems
- **Performance**: Good performance with network overhead
- **Persistence**: Configurable persistence

### Configuration

```python
# Memory cache (default)
config = AnalysisConfig(
    enable_caching=True,
    cache_backend="memory",
    cache_ttl=3600
)

# File cache
config = AnalysisConfig(
    enable_caching=True,
    cache_backend="file",
    cache_ttl=3600
)

# Redis cache
config = AnalysisConfig(
    enable_caching=True,
    cache_backend="redis",
    cache_ttl=3600,
    redis_url="redis://localhost:6379"
)
```

## Performance Optimization

### Best Practices

1. **Enable Caching**: Use appropriate cache backend for your environment
2. **Parallel Processing**: Enable for multiple file analysis
3. **File Size Limits**: Set reasonable limits to avoid memory issues
4. **Selective Analysis**: Only enable needed analysis types
5. **Language Detection**: Provide file paths for faster language detection

### Performance Metrics

The engine tracks performance metrics:

```python
result = await engine.analyze(code, "file.py")
print(f"Analysis time: {result.execution_time:.2f}s")

# Cache statistics
cache_stats = engine.cache_manager.get_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
```

## Error Handling

The engine provides comprehensive error handling:

### Error Types
- **Syntax Errors**: Invalid code syntax
- **Analysis Errors**: Failures in specific analysis components
- **Configuration Errors**: Invalid configuration parameters
- **Resource Errors**: Memory or time limit exceeded

### Error Recovery
- **Graceful Degradation**: Continue analysis even if some components fail
- **Fallback Parsing**: Use regex-based parsing when tree-sitter fails
- **Error Reporting**: Detailed error messages with context

### Example

```python
result = await engine.analyze(invalid_code, "broken.py")

if result.errors:
    print("Analysis errors:")
    for error in result.errors:
        print(f"  - {error}")

if result.warnings:
    print("Analysis warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")
```

## Integration Examples

### With Linear API

```python
from openevolve.analysis import ContextAnalysisEngine
import linear_api

async def process_linear_issue(issue_id):
    # Get issue from Linear
    issue = await linear_api.get_issue(issue_id)
    
    # Analyze requirements
    engine = ContextAnalysisEngine()
    requirement = await engine.analyze_requirements(issue.description)
    
    # Get related code files
    code_files = await get_related_files(requirement.affected_components)
    
    # Analyze code
    code_results = await engine.analyze_multiple_files(code_files)
    
    # Map requirements to code
    mapping = await engine.map_requirements_to_code(requirement, 
                                                   [r.code_contexts[0] for r in code_results])
    
    # Generate recommendations
    recommendations = []
    for result in code_results:
        recommendations.extend(result.recommendations)
    
    return {
        'requirement': requirement,
        'code_analysis': code_results,
        'mapping': mapping,
        'recommendations': recommendations
    }
```

### With GitHub API

```python
import github_api
from openevolve.analysis import ContextAnalysisEngine

async def analyze_pull_request(pr_number):
    # Get PR changes
    pr = await github_api.get_pull_request(pr_number)
    changed_files = await github_api.get_pr_files(pr_number)
    
    # Analyze changed files
    engine = ContextAnalysisEngine()
    results = []
    
    for file_info in changed_files:
        if file_info.status in ['added', 'modified']:
            content = await github_api.get_file_content(file_info.filename)
            result = await engine.analyze(content, file_info.filename)
            results.append(result)
    
    # Generate impact analysis
    changes = [f"Modified {f.filename}" for f in changed_files]
    
    # Combine dependency graphs
    combined_graph = combine_dependency_graphs([r.dependency_graph for r in results])
    
    impact = await engine.recommendation_engine.analyze_impact(changes, combined_graph)
    
    return {
        'pr_number': pr_number,
        'analysis_results': results,
        'impact_analysis': impact,
        'recommendations': [rec for result in results for rec in result.recommendations]
    }
```

## Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/analysis/

# Run specific test file
pytest tests/analysis/test_context_analysis_engine.py -v

# Run with coverage
pytest tests/analysis/ --cov=openevolve.analysis --cov-report=html
```

### Test Structure

```
tests/analysis/
├── test_context_analysis_engine.py    # Main engine tests
├── test_semantic_analyzer.py          # Semantic analysis tests
├── test_requirement_processor.py      # Requirement processing tests
├── test_recommendation_engine.py      # Recommendation tests
├── test_graph_sitter_parser.py       # Parser tests
├── test_cache_manager.py             # Caching tests
└── test_language_detector.py         # Language detection tests
```

### Example Test

```python
import pytest
from openevolve.analysis import ContextAnalysisEngine, AnalysisConfig

@pytest.mark.asyncio
async def test_python_analysis():
    config = AnalysisConfig(enable_caching=False)
    engine = ContextAnalysisEngine(config)
    
    code = '''
    def hello_world():
        print("Hello, World!")
    '''
    
    result = await engine.analyze(code, "hello.py")
    
    assert len(result.code_contexts) == 1
    assert result.code_contexts[0].language.value == "python"
    assert len(result.code_contexts[0].elements) > 0
    assert result.errors == []
```

## Troubleshooting

### Common Issues

#### 1. Tree-sitter Not Available
**Problem**: Tree-sitter libraries not installed
**Solution**: 
```bash
pip install tree-sitter tree-sitter-python tree-sitter-javascript
```

#### 2. Memory Issues with Large Files
**Problem**: Out of memory when analyzing large files
**Solution**: Adjust configuration
```python
config = AnalysisConfig(
    max_file_size=5 * 1024 * 1024,  # Reduce to 5MB
    parallel_processing=False        # Disable parallel processing
)
```

#### 3. Slow Analysis Performance
**Problem**: Analysis takes too long
**Solution**: Enable caching and optimize configuration
```python
config = AnalysisConfig(
    enable_caching=True,
    cache_backend="redis",
    enabled_analyses=[AnalysisType.SEMANTIC],  # Reduce analysis types
    parallel_processing=True
)
```

#### 4. Language Detection Issues
**Problem**: Wrong language detected
**Solution**: Provide explicit file extensions or use language hints
```python
# Provide clear file extension
result = await engine.analyze(code, "script.py")

# Or check language detection confidence
detector = LanguageDetector()
confidence = detector.get_language_confidence(code, "script.py")
```

### Debug Mode

Enable verbose logging for debugging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create config with verbose output
config = AnalysisConfig(verbose_output=True)
engine = ContextAnalysisEngine(config)
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/Zeeeepa/openevolve.git
cd openevolve

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints for all public APIs
- Add docstrings for all public methods
- Write comprehensive tests for new features

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For questions, issues, or contributions:

- **GitHub Issues**: [Report bugs or request features](https://github.com/Zeeeepa/openevolve/issues)
- **Documentation**: [Full documentation](https://github.com/Zeeeepa/openevolve/docs)
- **Examples**: [Usage examples](https://github.com/Zeeeepa/openevolve/examples)

## Changelog

### Version 1.0.0 (Current)
- Initial implementation of Context Analysis Engine
- Multi-language semantic analysis support
- Requirement processing and mapping
- Recommendation generation
- Dependency graph construction
- Caching system implementation
- Comprehensive test suite
- Documentation and examples


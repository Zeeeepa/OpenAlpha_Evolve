"""
Core interfaces and data structures for the Context Analysis Engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from enum import Enum
import time


class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    SEMANTIC = "semantic"
    SYNTACTIC = "syntactic"
    DEPENDENCY = "dependency"
    COMPLEXITY = "complexity"
    QUALITY = "quality"
    PATTERN = "pattern"
    IMPACT = "impact"


class LanguageType(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    RUST = "rust"
    GO = "go"
    UNKNOWN = "unknown"


@dataclass
class CodeElement:
    """Represents a code element (function, class, variable, etc.)."""
    name: str
    type: str  # function, class, variable, import, etc.
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    content: str
    language: LanguageType
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    complexity_score: Optional[float] = None


@dataclass
class DependencyNode:
    """Represents a node in the dependency graph."""
    id: str
    name: str
    type: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyEdge:
    """Represents an edge in the dependency graph."""
    source: str
    target: str
    relationship_type: str  # imports, calls, inherits, etc.
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyGraph:
    """Represents the complete dependency graph of a codebase."""
    nodes: Dict[str, DependencyNode] = field(default_factory=dict)
    edges: List[DependencyEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: DependencyNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: DependencyEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
    
    def get_dependencies(self, node_id: str) -> List[str]:
        """Get all dependencies of a node."""
        return [edge.target for edge in self.edges if edge.source == node_id]
    
    def get_dependents(self, node_id: str) -> List[str]:
        """Get all nodes that depend on this node."""
        return [edge.source for edge in self.edges if edge.target == node_id]


@dataclass
class CodeContext:
    """Represents the context of a code file or snippet."""
    file_path: str
    content: str
    language: LanguageType
    elements: List[CodeElement] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())


@dataclass
class RequirementContext:
    """Represents analyzed requirements and specifications."""
    id: str
    description: str
    type: str  # feature, bug_fix, enhancement, etc.
    priority: int = 1
    complexity_estimate: Optional[float] = None
    affected_components: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    technical_constraints: List[str] = field(default_factory=list)
    suggested_approach: Optional[str] = None
    estimated_effort: Optional[float] = None
    risk_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())


@dataclass
class ImpactAnalysis:
    """Represents the impact analysis of a change."""
    change_id: str
    affected_files: List[str] = field(default_factory=list)
    affected_functions: List[str] = field(default_factory=list)
    affected_classes: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high, critical
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    test_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Represents a recommendation from the analysis engine."""
    id: str
    type: str  # optimization, refactoring, bug_fix, etc.
    title: str
    description: str
    priority: int = 1
    confidence: float = 0.0
    affected_files: List[str] = field(default_factory=list)
    suggested_changes: List[str] = field(default_factory=list)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Comprehensive result of context analysis."""
    analysis_id: str
    analysis_type: AnalysisType
    code_contexts: List[CodeContext] = field(default_factory=list)
    requirement_contexts: List[RequirementContext] = field(default_factory=list)
    dependency_graph: Optional[DependencyGraph] = None
    impact_analysis: Optional[ImpactAnalysis] = None
    recommendations: List[Recommendation] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    created_at: float = field(default_factory=lambda: time.time())


@dataclass
class AnalysisConfig:
    """Configuration for the context analysis engine."""
    # Analysis settings
    enabled_analyses: List[AnalysisType] = field(default_factory=lambda: list(AnalysisType))
    supported_languages: List[LanguageType] = field(default_factory=lambda: [
        LanguageType.PYTHON, LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT
    ])
    
    # Performance settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_analysis_time: int = 300  # 5 minutes
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Caching settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_backend: str = "memory"  # memory, redis, file
    
    # Graph-sitter settings
    tree_sitter_timeout: int = 30
    tree_sitter_max_depth: int = 1000
    
    # NLP settings
    nlp_model: str = "en_core_web_sm"
    similarity_threshold: float = 0.7
    
    # Quality thresholds
    complexity_threshold: float = 10.0
    maintainability_threshold: float = 0.7
    
    # Output settings
    include_source_code: bool = False
    include_ast: bool = False
    verbose_output: bool = False
    
    # Integration settings
    database_url: Optional[str] = None
    linear_api_key: Optional[str] = None
    github_api_key: Optional[str] = None


# Abstract Interfaces

class ContextAnalyzerInterface(ABC):
    """Interface for context analyzers."""
    
    @abstractmethod
    async def analyze(self, content: str, file_path: str, config: AnalysisConfig) -> AnalysisResult:
        """Analyze content and return analysis result."""
        pass
    
    @abstractmethod
    def supports_language(self, language: LanguageType) -> bool:
        """Check if the analyzer supports a specific language."""
        pass


class SemanticAnalyzerInterface(ABC):
    """Interface for semantic analysis."""
    
    @abstractmethod
    async def analyze_semantics(self, code_context: CodeContext) -> Dict[str, Any]:
        """Perform semantic analysis on code context."""
        pass
    
    @abstractmethod
    async def extract_patterns(self, code_context: CodeContext) -> List[str]:
        """Extract code patterns from context."""
        pass
    
    @abstractmethod
    async def calculate_complexity(self, code_context: CodeContext) -> Dict[str, float]:
        """Calculate complexity metrics."""
        pass


class RequirementProcessorInterface(ABC):
    """Interface for requirement processing."""
    
    @abstractmethod
    async def process_requirements(self, requirements: str) -> RequirementContext:
        """Process natural language requirements."""
        pass
    
    @abstractmethod
    async def map_to_code(self, requirement: RequirementContext, 
                         code_contexts: List[CodeContext]) -> Dict[str, Any]:
        """Map requirements to code components."""
        pass
    
    @abstractmethod
    async def decompose_task(self, requirement: RequirementContext) -> List[RequirementContext]:
        """Decompose complex requirements into subtasks."""
        pass


class RecommendationEngineInterface(ABC):
    """Interface for recommendation generation."""
    
    @abstractmethod
    async def generate_recommendations(self, analysis_result: AnalysisResult) -> List[Recommendation]:
        """Generate recommendations based on analysis."""
        pass
    
    @abstractmethod
    async def analyze_impact(self, changes: List[str], 
                           dependency_graph: DependencyGraph) -> ImpactAnalysis:
        """Analyze the impact of proposed changes."""
        pass
    
    @abstractmethod
    async def suggest_optimizations(self, code_context: CodeContext) -> List[Recommendation]:
        """Suggest code optimizations."""
        pass


class GraphSitterParserInterface(ABC):
    """Interface for graph-sitter integration."""
    
    @abstractmethod
    async def parse(self, content: str, language: LanguageType) -> Any:
        """Parse content using tree-sitter."""
        pass
    
    @abstractmethod
    async def extract_elements(self, tree: Any, content: str) -> List[CodeElement]:
        """Extract code elements from AST."""
        pass
    
    @abstractmethod
    async def build_dependency_graph(self, elements: List[CodeElement]) -> DependencyGraph:
        """Build dependency graph from code elements."""
        pass


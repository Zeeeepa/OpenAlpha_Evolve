"""Core components for the Context Analysis Engine."""

from .interfaces import (
    AnalysisResult,
    CodeContext,
    RequirementContext,
    DependencyGraph,
    AnalysisConfig,
    ContextAnalyzerInterface,
    SemanticAnalyzerInterface,
    RequirementProcessorInterface,
    RecommendationEngineInterface
)
from .engine import ContextAnalysisEngine

__all__ = [
    "AnalysisResult",
    "CodeContext",
    "RequirementContext", 
    "DependencyGraph",
    "AnalysisConfig",
    "ContextAnalyzerInterface",
    "SemanticAnalyzerInterface",
    "RequirementProcessorInterface",
    "RecommendationEngineInterface",
    "ContextAnalysisEngine"
]


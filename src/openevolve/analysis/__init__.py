"""
OpenEvolve Context Analysis Engine

A comprehensive context analysis system for autonomous development pipelines.
Provides semantic code analysis, requirement processing, and intelligent recommendations.
"""

from .core.engine import ContextAnalysisEngine
from .core.interfaces import (
    AnalysisResult,
    CodeContext,
    RequirementContext,
    DependencyGraph,
    AnalysisConfig
)
from .semantic.analyzer import SemanticAnalyzer
from .intelligence.processor import RequirementProcessor
from .intelligence.recommender import RecommendationEngine
from .integration.graph_sitter import GraphSitterParser

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"

__all__ = [
    "ContextAnalysisEngine",
    "AnalysisResult",
    "CodeContext", 
    "RequirementContext",
    "DependencyGraph",
    "AnalysisConfig",
    "SemanticAnalyzer",
    "RequirementProcessor",
    "RecommendationEngine",
    "GraphSitterParser"
]


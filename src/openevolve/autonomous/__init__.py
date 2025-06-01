"""
Autonomous development components for OpenEvolve.

This module provides autonomous development capabilities including:
- Task management and execution
- Code analysis and understanding
- Error detection and resolution
- Continuous improvement mechanisms
"""

from .task_manager import AutonomousTaskManager
from .code_analyzer import CodeAnalyzer
from .error_handler import ErrorHandler
from .learning_system import LearningSystem

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"

__all__ = [
    "AutonomousTaskManager",
    "CodeAnalyzer", 
    "ErrorHandler",
    "LearningSystem"
]


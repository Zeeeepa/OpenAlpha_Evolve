"""
Intelligent Error Classification System
Categorizes errors by type, severity, and potential solutions.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ast
import traceback

from core.interfaces import BaseAgent

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Categories of errors that can occur during code evolution."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    PERFORMANCE_ERROR = "performance_error"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    SECURITY_ERROR = "security_error"
    STYLE_ERROR = "style_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    CRITICAL = "critical"      # Prevents execution entirely
    HIGH = "high"             # Causes incorrect results
    MEDIUM = "medium"         # Performance or style issues
    LOW = "low"               # Minor style or optimization issues
    INFO = "info"             # Informational only

@dataclass
class ErrorClassification:
    """Represents a classified error with metadata."""
    category: ErrorCategory
    severity: ErrorSeverity
    error_message: str
    error_type: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    context: Optional[str] = None
    suggested_fixes: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    patterns_matched: List[str] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)

class ErrorClassifier(BaseAgent):
    """
    Intelligent error classification system that categorizes errors
    and provides actionable insights for automated debugging.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.error_patterns = self._initialize_error_patterns()
        self.classification_history: List[ErrorClassification] = []
        
    async def execute(self, error_info: Dict[str, Any]) -> ErrorClassification:
        """Main execution method - classifies an error."""
        return await self.classify_error(error_info)
    
    async def classify_error(self, error_info: Dict[str, Any]) -> ErrorClassification:
        """
        Classify an error based on error information.
        
        Args:
            error_info: Dictionary containing error details:
                - error_message: The error message
                - error_type: Type of exception
                - traceback: Full traceback if available
                - code: The code that caused the error
                - context: Additional context
        """
        error_message = error_info.get('error_message', '')
        error_type = error_info.get('error_type', '')
        traceback_str = error_info.get('traceback', '')
        code = error_info.get('code', '')
        context = error_info.get('context', '')
        
        logger.debug(f"Classifying error: {error_type}: {error_message}")
        
        # Extract line and column information
        line_number, column_number = self._extract_location_info(traceback_str, error_message)
        
        # Classify by error type first
        category = self._classify_by_type(error_type, error_message)
        
        # Determine severity
        severity = self._determine_severity(category, error_message, code)
        
        # Find matching patterns
        patterns_matched = self._find_matching_patterns(error_message, error_type, code)
        
        # Generate suggested fixes
        suggested_fixes = self._generate_suggested_fixes(category, error_message, code, patterns_matched)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(patterns_matched, category, error_type)
        
        # Find related errors
        related_errors = self._find_related_errors(error_message, error_type)
        
        classification = ErrorClassification(
            category=category,
            severity=severity,
            error_message=error_message,
            error_type=error_type,
            line_number=line_number,
            column_number=column_number,
            context=context,
            suggested_fixes=suggested_fixes,
            confidence_score=confidence_score,
            patterns_matched=patterns_matched,
            related_errors=related_errors
        )
        
        # Store in history for learning
        self.classification_history.append(classification)
        
        logger.info(f"Classified error as {category.value} with severity {severity.value} (confidence: {confidence_score:.2f})")
        
        return classification
    
    def _initialize_error_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize error patterns for classification."""
        return {
            'syntax_patterns': [
                {
                    'pattern': r'SyntaxError.*invalid syntax',
                    'category': ErrorCategory.SYNTAX_ERROR,
                    'severity': ErrorSeverity.CRITICAL,
                    'fixes': ['Check for missing colons, parentheses, or brackets',
                             'Verify proper indentation',
                             'Check for invalid characters']
                },
                {
                    'pattern': r'IndentationError',
                    'category': ErrorCategory.SYNTAX_ERROR,
                    'severity': ErrorSeverity.CRITICAL,
                    'fixes': ['Fix indentation to match Python standards',
                             'Use consistent spaces or tabs',
                             'Check for mixed indentation']
                },
                {
                    'pattern': r'EOFError.*unexpected end of file',
                    'category': ErrorCategory.SYNTAX_ERROR,
                    'severity': ErrorSeverity.CRITICAL,
                    'fixes': ['Check for unclosed parentheses, brackets, or quotes',
                             'Ensure all code blocks are properly closed']
                }
            ],
            'runtime_patterns': [
                {
                    'pattern': r'NameError.*name .* is not defined',
                    'category': ErrorCategory.RUNTIME_ERROR,
                    'severity': ErrorSeverity.HIGH,
                    'fixes': ['Define the variable before using it',
                             'Check for typos in variable names',
                             'Import required modules']
                },
                {
                    'pattern': r'AttributeError.*has no attribute',
                    'category': ErrorCategory.RUNTIME_ERROR,
                    'severity': ErrorSeverity.HIGH,
                    'fixes': ['Check if the attribute exists on the object',
                             'Verify object type before accessing attributes',
                             'Check for typos in attribute names']
                },
                {
                    'pattern': r'KeyError',
                    'category': ErrorCategory.RUNTIME_ERROR,
                    'severity': ErrorSeverity.MEDIUM,
                    'fixes': ['Check if key exists before accessing',
                             'Use dict.get() with default value',
                             'Verify dictionary structure']
                },
                {
                    'pattern': r'IndexError.*list index out of range',
                    'category': ErrorCategory.RUNTIME_ERROR,
                    'severity': ErrorSeverity.MEDIUM,
                    'fixes': ['Check list length before accessing index',
                             'Use try-except for index access',
                             'Verify loop bounds']
                }
            ],
            'type_patterns': [
                {
                    'pattern': r'TypeError.*unsupported operand type',
                    'category': ErrorCategory.TYPE_ERROR,
                    'severity': ErrorSeverity.HIGH,
                    'fixes': ['Check operand types before operations',
                             'Convert types as needed',
                             'Use type checking']
                },
                {
                    'pattern': r'TypeError.*takes .* positional arguments but .* were given',
                    'category': ErrorCategory.TYPE_ERROR,
                    'severity': ErrorSeverity.HIGH,
                    'fixes': ['Check function signature',
                             'Verify number of arguments passed',
                             'Check for missing or extra parameters']
                }
            ],
            'import_patterns': [
                {
                    'pattern': r'ImportError.*No module named',
                    'category': ErrorCategory.IMPORT_ERROR,
                    'severity': ErrorSeverity.CRITICAL,
                    'fixes': ['Install required package',
                             'Check module name spelling',
                             'Verify module is in Python path']
                },
                {
                    'pattern': r'ModuleNotFoundError',
                    'category': ErrorCategory.IMPORT_ERROR,
                    'severity': ErrorSeverity.CRITICAL,
                    'fixes': ['Install missing module',
                             'Check import path',
                             'Verify virtual environment']
                }
            ],
            'performance_patterns': [
                {
                    'pattern': r'TimeoutError|timeout',
                    'category': ErrorCategory.TIMEOUT_ERROR,
                    'severity': ErrorSeverity.MEDIUM,
                    'fixes': ['Optimize algorithm complexity',
                             'Add timeout handling',
                             'Use more efficient data structures']
                },
                {
                    'pattern': r'MemoryError|memory',
                    'category': ErrorCategory.MEMORY_ERROR,
                    'severity': ErrorSeverity.HIGH,
                    'fixes': ['Optimize memory usage',
                             'Use generators for large datasets',
                             'Implement memory-efficient algorithms']
                }
            ]
        }
    
    def _classify_by_type(self, error_type: str, error_message: str) -> ErrorCategory:
        """Classify error by its type and message."""
        error_text = f"{error_type}: {error_message}".lower()
        
        # Check all pattern categories
        for pattern_category, patterns in self.error_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info['pattern'], error_text, re.IGNORECASE):
                    return pattern_info['category']
        
        # Fallback classification based on error type
        type_mapping = {
            'syntaxerror': ErrorCategory.SYNTAX_ERROR,
            'indentationerror': ErrorCategory.SYNTAX_ERROR,
            'nameerror': ErrorCategory.RUNTIME_ERROR,
            'attributeerror': ErrorCategory.RUNTIME_ERROR,
            'keyerror': ErrorCategory.RUNTIME_ERROR,
            'indexerror': ErrorCategory.RUNTIME_ERROR,
            'valueerror': ErrorCategory.RUNTIME_ERROR,
            'typeerror': ErrorCategory.TYPE_ERROR,
            'importerror': ErrorCategory.IMPORT_ERROR,
            'modulenotfounderror': ErrorCategory.IMPORT_ERROR,
            'timeouterror': ErrorCategory.TIMEOUT_ERROR,
            'memoryerror': ErrorCategory.MEMORY_ERROR,
        }
        
        return type_mapping.get(error_type.lower(), ErrorCategory.UNKNOWN_ERROR)
    
    def _determine_severity(self, category: ErrorCategory, error_message: str, code: str) -> ErrorSeverity:
        """Determine the severity of an error."""
        # Critical errors that prevent execution
        if category in [ErrorCategory.SYNTAX_ERROR, ErrorCategory.IMPORT_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # High severity for runtime errors that affect correctness
        if category in [ErrorCategory.RUNTIME_ERROR, ErrorCategory.TYPE_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity for performance issues
        if category in [ErrorCategory.PERFORMANCE_ERROR, ErrorCategory.TIMEOUT_ERROR, ErrorCategory.MEMORY_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity for style issues
        if category == ErrorCategory.STYLE_ERROR:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _find_matching_patterns(self, error_message: str, error_type: str, code: str) -> List[str]:
        """Find patterns that match the current error."""
        matched_patterns = []
        error_text = f"{error_type}: {error_message}".lower()
        
        for pattern_category, patterns in self.error_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info['pattern'], error_text, re.IGNORECASE):
                    matched_patterns.append(pattern_info['pattern'])
        
        return matched_patterns
    
    def _generate_suggested_fixes(self, category: ErrorCategory, error_message: str, 
                                code: str, patterns_matched: List[str]) -> List[str]:
        """Generate suggested fixes based on error classification."""
        fixes = []
        
        # Get fixes from matched patterns
        for pattern_category, patterns in self.error_patterns.items():
            for pattern_info in patterns:
                if pattern_info['pattern'] in patterns_matched:
                    fixes.extend(pattern_info['fixes'])
        
        # Add category-specific fixes
        category_fixes = {
            ErrorCategory.SYNTAX_ERROR: [
                'Run code through a syntax checker',
                'Use an IDE with syntax highlighting',
                'Check Python version compatibility'
            ],
            ErrorCategory.RUNTIME_ERROR: [
                'Add error handling with try-except blocks',
                'Validate inputs before processing',
                'Add debugging print statements'
            ],
            ErrorCategory.LOGIC_ERROR: [
                'Review algorithm logic',
                'Add unit tests to verify behavior',
                'Use debugging tools to trace execution'
            ],
            ErrorCategory.PERFORMANCE_ERROR: [
                'Profile code to identify bottlenecks',
                'Optimize data structures and algorithms',
                'Consider parallel processing'
            ]
        }
        
        if category in category_fixes:
            fixes.extend(category_fixes[category])
        
        return list(set(fixes))  # Remove duplicates
    
    def _calculate_confidence(self, patterns_matched: List[str], 
                            category: ErrorCategory, error_type: str) -> float:
        """Calculate confidence score for the classification."""
        base_confidence = 0.5
        
        # Increase confidence for pattern matches
        pattern_bonus = min(len(patterns_matched) * 0.2, 0.4)
        
        # Increase confidence for exact error type matches
        type_bonus = 0.3 if error_type.lower() in category.value else 0.0
        
        # Historical accuracy bonus (placeholder for future implementation)
        history_bonus = 0.0
        
        confidence = min(base_confidence + pattern_bonus + type_bonus + history_bonus, 1.0)
        return confidence
    
    def _find_related_errors(self, error_message: str, error_type: str) -> List[str]:
        """Find related errors from classification history."""
        related = []
        
        for past_classification in self.classification_history[-10:]:  # Last 10 errors
            if (past_classification.error_type == error_type or 
                any(word in past_classification.error_message.lower() 
                    for word in error_message.lower().split() if len(word) > 3)):
                related.append(past_classification.error_message)
        
        return related
    
    def _extract_location_info(self, traceback_str: str, error_message: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract line and column information from traceback."""
        line_number = None
        column_number = None
        
        # Try to extract from traceback
        line_match = re.search(r'line (\d+)', traceback_str)
        if line_match:
            line_number = int(line_match.group(1))
        
        # Try to extract from error message
        if not line_number:
            line_match = re.search(r'line (\d+)', error_message)
            if line_match:
                line_number = int(line_match.group(1))
        
        # Extract column information if available
        col_match = re.search(r'column (\d+)', error_message)
        if col_match:
            column_number = int(col_match.group(1))
        
        return line_number, column_number
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about error classifications."""
        if not self.classification_history:
            return {}
        
        total_errors = len(self.classification_history)
        category_counts = {}
        severity_counts = {}
        avg_confidence = 0.0
        
        for classification in self.classification_history:
            category_counts[classification.category.value] = category_counts.get(classification.category.value, 0) + 1
            severity_counts[classification.severity.value] = severity_counts.get(classification.severity.value, 0) + 1
            avg_confidence += classification.confidence_score
        
        avg_confidence /= total_errors
        
        return {
            'total_errors': total_errors,
            'category_distribution': category_counts,
            'severity_distribution': severity_counts,
            'average_confidence': avg_confidence,
            'most_common_category': max(category_counts, key=category_counts.get) if category_counts else None,
            'most_common_severity': max(severity_counts, key=severity_counts.get) if severity_counts else None
        }


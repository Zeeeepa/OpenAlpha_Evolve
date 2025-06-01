"""
Error Handler for Autonomous Development.

Provides intelligent error detection, classification, and resolution
capabilities for autonomous development systems.
"""

import logging
import traceback
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import re
import json

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors in autonomous development."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    DEPENDENCY_ERROR = "dependency_error"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_ERROR = "resource_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ResolutionStatus(Enum):
    """Status of error resolution attempts."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    FAILED = "failed"
    ESCALATED = "escalated"


@dataclass
class ErrorPattern:
    """Pattern for error recognition and resolution."""
    pattern: str
    category: ErrorCategory
    severity: ErrorSeverity
    description: str
    resolution_steps: List[str]
    auto_fixable: bool = False
    confidence: float = 1.0


@dataclass
class ErrorInstance:
    """Instance of an error that occurred."""
    id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    resolution_status: ResolutionStatus = ResolutionStatus.PENDING
    resolution_attempts: List[Dict[str, Any]] = field(default_factory=list)
    pattern_match: Optional[ErrorPattern] = None


@dataclass
class ResolutionAttempt:
    """Attempt to resolve an error."""
    timestamp: datetime
    strategy: str
    actions_taken: List[str]
    success: bool
    error_message: Optional[str] = None
    duration: float = 0.0


class ErrorHandler:
    """
    Intelligent error handler for autonomous development.
    
    Provides error detection, classification, and automated resolution
    capabilities with learning and improvement over time.
    """
    
    def __init__(self, database_connector=None, audit_logger=None):
        self.connector = database_connector
        self.audit_logger = audit_logger
        self._error_patterns: List[ErrorPattern] = []
        self._error_history: Dict[str, ErrorInstance] = {}
        self._resolution_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self._learning_data: Dict[str, Any] = {}
        
        # Initialize built-in patterns and strategies
        self._initialize_error_patterns()
        self._initialize_resolution_strategies()
        
        logger.info("Error handler initialized")
    
    async def initialize(self) -> None:
        """Initialize the error handler."""
        try:
            # Load error history from database
            await self._load_error_history()
            
            # Load learning data
            await self._load_learning_data()
            
            logger.info(f"Error handler initialized with {len(self._error_history)} historical errors")
            
        except Exception as e:
            logger.error(f"Failed to initialize error handler: {e}")
            raise
    
    async def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        auto_resolve: bool = True
    ) -> ErrorInstance:
        """
        Handle an error with intelligent classification and resolution.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            auto_resolve: Whether to attempt automatic resolution
        
        Returns:
            ErrorInstance with classification and resolution status
        """
        # Create error instance
        error_instance = self._create_error_instance(error, context)
        
        # Classify the error
        await self._classify_error(error_instance)
        
        # Store error
        self._error_history[error_instance.id] = error_instance
        await self._save_error_to_db(error_instance)
        
        # Log the error
        if self.audit_logger:
            self.audit_logger.log_error(
                error_instance.category.value,
                error_instance.message,
                error_id=error_instance.id,
                severity=error_instance.severity.name,
                file_path=error_instance.file_path,
                line_number=error_instance.line_number
            )
        
        logger.error(f"Error handled: {error_instance.id} - {error_instance.message}")
        
        # Attempt automatic resolution if enabled
        if auto_resolve and error_instance.pattern_match and error_instance.pattern_match.auto_fixable:
            await self._attempt_resolution(error_instance)
        
        return error_instance
    
    async def resolve_error(self, error_id: str, strategy: Optional[str] = None) -> bool:
        """
        Attempt to resolve a specific error.
        
        Args:
            error_id: ID of the error to resolve
            strategy: Specific resolution strategy to use
        
        Returns:
            True if resolution was successful
        """
        error_instance = self._error_history.get(error_id)
        if not error_instance:
            logger.warning(f"Error {error_id} not found")
            return False
        
        return await self._attempt_resolution(error_instance, strategy)
    
    async def get_error_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get error statistics for the specified period.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with error statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_errors = [
            error for error in self._error_history.values()
            if error.occurred_at >= cutoff_date
        ]
        
        # Category distribution
        category_counts = {}
        for error in recent_errors:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Severity distribution
        severity_counts = {}
        for error in recent_errors:
            severity = error.severity.name
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Resolution statistics
        resolved_count = len([e for e in recent_errors if e.resolution_status == ResolutionStatus.RESOLVED])
        failed_count = len([e for e in recent_errors if e.resolution_status == ResolutionStatus.FAILED])
        
        resolution_rate = (resolved_count / len(recent_errors) * 100) if recent_errors else 0
        
        # Most common errors
        error_messages = {}
        for error in recent_errors:
            msg = error.message[:100]  # Truncate for grouping
            error_messages[msg] = error_messages.get(msg, 0) + 1
        
        common_errors = sorted(error_messages.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "period_days": days,
            "total_errors": len(recent_errors),
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "resolution_rate": resolution_rate,
            "resolved_errors": resolved_count,
            "failed_resolutions": failed_count,
            "common_errors": common_errors,
            "auto_fixable_rate": len([e for e in recent_errors if e.pattern_match and e.pattern_match.auto_fixable]) / len(recent_errors) * 100 if recent_errors else 0
        }
    
    async def learn_from_resolution(
        self,
        error_id: str,
        resolution_successful: bool,
        feedback: Optional[str] = None
    ) -> None:
        """
        Learn from a resolution attempt to improve future handling.
        
        Args:
            error_id: ID of the error that was resolved
            resolution_successful: Whether the resolution was successful
            feedback: Optional feedback about the resolution
        """
        error_instance = self._error_history.get(error_id)
        if not error_instance:
            return
        
        # Update learning data
        pattern_key = f"{error_instance.category.value}:{error_instance.message[:50]}"
        
        if pattern_key not in self._learning_data:
            self._learning_data[pattern_key] = {
                "success_count": 0,
                "failure_count": 0,
                "total_attempts": 0,
                "successful_strategies": [],
                "failed_strategies": [],
                "feedback": []
            }
        
        learning_entry = self._learning_data[pattern_key]
        learning_entry["total_attempts"] += 1
        
        if resolution_successful:
            learning_entry["success_count"] += 1
            if error_instance.resolution_attempts:
                last_attempt = error_instance.resolution_attempts[-1]
                strategy = last_attempt.get("strategy", "unknown")
                if strategy not in learning_entry["successful_strategies"]:
                    learning_entry["successful_strategies"].append(strategy)
        else:
            learning_entry["failure_count"] += 1
            if error_instance.resolution_attempts:
                last_attempt = error_instance.resolution_attempts[-1]
                strategy = last_attempt.get("strategy", "unknown")
                if strategy not in learning_entry["failed_strategies"]:
                    learning_entry["failed_strategies"].append(strategy)
        
        if feedback:
            learning_entry["feedback"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "feedback": feedback,
                "successful": resolution_successful
            })
        
        # Save learning data
        await self._save_learning_data()
        
        logger.info(f"Learned from resolution of error {error_id}: success={resolution_successful}")
    
    def _create_error_instance(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorInstance:
        """Create an ErrorInstance from an exception."""
        error_id = f"error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{id(error)}"
        
        # Extract traceback information
        tb = traceback.extract_tb(error.__traceback__) if error.__traceback__ else None
        traceback_str = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        
        file_path = None
        line_number = None
        function_name = None
        
        if tb and len(tb) > 0:
            frame = tb[-1]  # Last frame (where error occurred)
            file_path = frame.filename
            line_number = frame.lineno
            function_name = frame.name
        
        return ErrorInstance(
            id=error_id,
            category=ErrorCategory.UNKNOWN_ERROR,  # Will be classified later
            severity=ErrorSeverity.MEDIUM,  # Default severity
            message=str(error),
            traceback=traceback_str,
            context=context or {},
            file_path=file_path,
            line_number=line_number,
            function_name=function_name
        )
    
    async def _classify_error(self, error_instance: ErrorInstance) -> None:
        """Classify an error using patterns and heuristics."""
        best_match = None
        best_confidence = 0.0
        
        # Try to match against known patterns
        for pattern in self._error_patterns:
            confidence = self._match_pattern(error_instance, pattern)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = pattern
        
        if best_match and best_confidence > 0.5:
            error_instance.category = best_match.category
            error_instance.severity = best_match.severity
            error_instance.pattern_match = best_match
        else:
            # Fallback classification based on exception type
            error_instance.category = self._classify_by_exception_type(error_instance)
            error_instance.severity = self._estimate_severity(error_instance)
    
    def _match_pattern(self, error_instance: ErrorInstance, pattern: ErrorPattern) -> float:
        """Calculate confidence score for pattern match."""
        message = error_instance.message.lower()
        pattern_text = pattern.pattern.lower()
        
        # Simple regex matching
        try:
            if re.search(pattern_text, message):
                return pattern.confidence
        except re.error:
            # Fallback to substring matching
            if pattern_text in message:
                return pattern.confidence * 0.8
        
        return 0.0
    
    def _classify_by_exception_type(self, error_instance: ErrorInstance) -> ErrorCategory:
        """Classify error based on exception type and message."""
        message = error_instance.message.lower()
        
        # Syntax errors
        if any(keyword in message for keyword in ['syntax error', 'invalid syntax', 'unexpected token']):
            return ErrorCategory.SYNTAX_ERROR
        
        # Database errors
        if any(keyword in message for keyword in ['database', 'sql', 'connection', 'query']):
            return ErrorCategory.DATABASE_ERROR
        
        # Network errors
        if any(keyword in message for keyword in ['network', 'connection refused', 'timeout', 'dns']):
            return ErrorCategory.NETWORK_ERROR
        
        # Permission errors
        if any(keyword in message for keyword in ['permission', 'access denied', 'forbidden', 'unauthorized']):
            return ErrorCategory.PERMISSION_ERROR
        
        # Dependency errors
        if any(keyword in message for keyword in ['import', 'module', 'package', 'dependency']):
            return ErrorCategory.DEPENDENCY_ERROR
        
        # Configuration errors
        if any(keyword in message for keyword in ['config', 'setting', 'environment', 'variable']):
            return ErrorCategory.CONFIGURATION_ERROR
        
        # Resource errors
        if any(keyword in message for keyword in ['memory', 'disk', 'space', 'resource']):
            return ErrorCategory.RESOURCE_ERROR
        
        # Timeout errors
        if any(keyword in message for keyword in ['timeout', 'timed out', 'deadline']):
            return ErrorCategory.TIMEOUT_ERROR
        
        return ErrorCategory.RUNTIME_ERROR
    
    def _estimate_severity(self, error_instance: ErrorInstance) -> ErrorSeverity:
        """Estimate error severity based on context and type."""
        message = error_instance.message.lower()
        
        # Critical keywords
        if any(keyword in message for keyword in ['critical', 'fatal', 'crash', 'corruption']):
            return ErrorSeverity.CRITICAL
        
        # High severity keywords
        if any(keyword in message for keyword in ['security', 'vulnerability', 'breach', 'unauthorized']):
            return ErrorSeverity.HIGH
        
        # Category-based severity
        if error_instance.category in [ErrorCategory.SYNTAX_ERROR, ErrorCategory.DEPENDENCY_ERROR]:
            return ErrorSeverity.HIGH
        
        if error_instance.category in [ErrorCategory.CONFIGURATION_ERROR, ErrorCategory.PERMISSION_ERROR]:
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.MEDIUM
    
    async def _attempt_resolution(
        self,
        error_instance: ErrorInstance,
        strategy: Optional[str] = None
    ) -> bool:
        """Attempt to resolve an error automatically."""
        error_instance.resolution_status = ResolutionStatus.IN_PROGRESS
        
        start_time = datetime.utcnow()
        attempt = {
            "timestamp": start_time,
            "strategy": strategy or "auto",
            "actions_taken": [],
            "success": False,
            "error_message": None
        }
        
        try:
            # Get resolution strategies for this error category
            strategies = self._resolution_strategies.get(error_instance.category, [])
            
            if strategy:
                # Use specific strategy if provided
                strategy_func = next((s for s in strategies if s.__name__ == strategy), None)
                if strategy_func:
                    strategies = [strategy_func]
                else:
                    attempt["error_message"] = f"Strategy '{strategy}' not found"
                    return False
            
            # Try each strategy
            for strategy_func in strategies:
                try:
                    logger.info(f"Attempting resolution strategy: {strategy_func.__name__}")
                    
                    result = await strategy_func(error_instance)
                    attempt["actions_taken"].append(f"Executed {strategy_func.__name__}")
                    
                    if result:
                        attempt["success"] = True
                        error_instance.resolution_status = ResolutionStatus.RESOLVED
                        break
                    
                except Exception as e:
                    attempt["actions_taken"].append(f"Failed {strategy_func.__name__}: {str(e)}")
                    logger.warning(f"Resolution strategy {strategy_func.__name__} failed: {e}")
            
            if not attempt["success"]:
                error_instance.resolution_status = ResolutionStatus.FAILED
                attempt["error_message"] = "All resolution strategies failed"
            
        except Exception as e:
            attempt["success"] = False
            attempt["error_message"] = str(e)
            error_instance.resolution_status = ResolutionStatus.FAILED
            logger.error(f"Error during resolution attempt: {e}")
        
        finally:
            end_time = datetime.utcnow()
            attempt["duration"] = (end_time - start_time).total_seconds()
            error_instance.resolution_attempts.append(attempt)
            
            # Save updated error instance
            await self._save_error_to_db(error_instance)
        
        success = attempt["success"]
        logger.info(f"Resolution attempt for {error_instance.id}: {'SUCCESS' if success else 'FAILED'}")
        
        return success
    
    def _initialize_error_patterns(self) -> None:
        """Initialize built-in error patterns."""
        patterns = [
            ErrorPattern(
                pattern=r"no module named ['\"](\w+)['\"]",
                category=ErrorCategory.DEPENDENCY_ERROR,
                severity=ErrorSeverity.HIGH,
                description="Missing Python module",
                resolution_steps=["Install missing module", "Check import path", "Verify virtual environment"],
                auto_fixable=True,
                confidence=0.9
            ),
            ErrorPattern(
                pattern=r"connection refused",
                category=ErrorCategory.NETWORK_ERROR,
                severity=ErrorSeverity.HIGH,
                description="Network connection refused",
                resolution_steps=["Check service status", "Verify network connectivity", "Check firewall settings"],
                auto_fixable=False,
                confidence=0.8
            ),
            ErrorPattern(
                pattern=r"permission denied",
                category=ErrorCategory.PERMISSION_ERROR,
                severity=ErrorSeverity.MEDIUM,
                description="Permission denied error",
                resolution_steps=["Check file permissions", "Verify user privileges", "Update access rights"],
                auto_fixable=True,
                confidence=0.9
            ),
            ErrorPattern(
                pattern=r"syntax error",
                category=ErrorCategory.SYNTAX_ERROR,
                severity=ErrorSeverity.HIGH,
                description="Syntax error in code",
                resolution_steps=["Check syntax", "Verify indentation", "Review recent changes"],
                auto_fixable=False,
                confidence=0.95
            ),
            ErrorPattern(
                pattern=r"timeout|timed out",
                category=ErrorCategory.TIMEOUT_ERROR,
                severity=ErrorSeverity.MEDIUM,
                description="Operation timeout",
                resolution_steps=["Increase timeout", "Optimize operation", "Check system load"],
                auto_fixable=True,
                confidence=0.8
            )
        ]
        
        self._error_patterns.extend(patterns)
        logger.info(f"Initialized {len(patterns)} error patterns")
    
    def _initialize_resolution_strategies(self) -> None:
        """Initialize resolution strategies for different error categories."""
        
        async def resolve_dependency_error(error_instance: ErrorInstance) -> bool:
            """Attempt to resolve dependency errors."""
            # Extract module name from error message
            import re
            match = re.search(r"no module named ['\"](\w+)['\"]", error_instance.message.lower())
            if match:
                module_name = match.group(1)
                # In a real implementation, this would attempt to install the module
                logger.info(f"Would attempt to install module: {module_name}")
                return True
            return False
        
        async def resolve_permission_error(error_instance: ErrorInstance) -> bool:
            """Attempt to resolve permission errors."""
            # In a real implementation, this would attempt to fix permissions
            logger.info("Would attempt to fix file permissions")
            return False  # Usually requires manual intervention
        
        async def resolve_timeout_error(error_instance: ErrorInstance) -> bool:
            """Attempt to resolve timeout errors."""
            # In a real implementation, this would adjust timeout settings
            logger.info("Would attempt to increase timeout settings")
            return True
        
        async def resolve_configuration_error(error_instance: ErrorInstance) -> bool:
            """Attempt to resolve configuration errors."""
            # In a real implementation, this would check and fix configuration
            logger.info("Would attempt to fix configuration")
            return False
        
        # Register strategies
        self._resolution_strategies = {
            ErrorCategory.DEPENDENCY_ERROR: [resolve_dependency_error],
            ErrorCategory.PERMISSION_ERROR: [resolve_permission_error],
            ErrorCategory.TIMEOUT_ERROR: [resolve_timeout_error],
            ErrorCategory.CONFIGURATION_ERROR: [resolve_configuration_error]
        }
        
        logger.info(f"Initialized resolution strategies for {len(self._resolution_strategies)} error categories")
    
    async def _load_error_history(self) -> None:
        """Load error history from database."""
        # Placeholder - would load from database in real implementation
        pass
    
    async def _save_error_to_db(self, error_instance: ErrorInstance) -> None:
        """Save error instance to database."""
        # Placeholder - would save to database in real implementation
        pass
    
    async def _load_learning_data(self) -> None:
        """Load learning data from storage."""
        # Placeholder - would load from database in real implementation
        pass
    
    async def _save_learning_data(self) -> None:
        """Save learning data to storage."""
        # Placeholder - would save to database in real implementation
        pass


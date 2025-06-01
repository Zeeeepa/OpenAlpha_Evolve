"""
Autonomous Task Manager for OpenEvolve.

Manages autonomous development tasks including planning, execution,
validation, and continuous improvement.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


@dataclass
class TaskResult:
    """Result of task execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    duration: float = 0.0


@dataclass
class AutonomousTask:
    """Autonomous development task."""
    id: str
    title: str
    description: str
    task_type: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[TaskResult] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0  # 5 minutes default


class AutonomousTaskManager:
    """
    Manages autonomous development tasks with intelligent planning,
    execution, and continuous improvement capabilities.
    """
    
    def __init__(self, database_connector, audit_logger=None):
        self.connector = database_connector
        self.audit_logger = audit_logger
        self._tasks: Dict[str, AutonomousTask] = {}
        self._task_handlers: Dict[str, Callable] = {}
        self._execution_queue: asyncio.Queue = asyncio.Queue()
        self._worker_tasks: List[asyncio.Task] = []
        self._is_running = False
        self._max_concurrent_tasks = 3
        
        # Task execution metrics
        self._execution_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_duration": 0.0,
            "success_rate": 0.0
        }
        
        logger.info("Autonomous Task Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the task manager."""
        try:
            # Ensure tasks table exists
            await self._ensure_tasks_table()
            
            # Load existing tasks from database
            await self._load_existing_tasks()
            
            # Register built-in task handlers
            self._register_builtin_handlers()
            
            logger.info(f"Task manager initialized with {len(self._tasks)} existing tasks")
            
        except Exception as e:
            logger.error(f"Failed to initialize task manager: {e}")
            raise
    
    async def start(self) -> None:
        """Start the autonomous task execution system."""
        if self._is_running:
            logger.warning("Task manager is already running")
            return
        
        self._is_running = True
        
        # Start worker tasks
        for i in range(self._max_concurrent_tasks):
            worker = asyncio.create_task(self._task_worker(f"worker-{i}"))
            self._worker_tasks.append(worker)
        
        # Start task scheduler
        scheduler = asyncio.create_task(self._task_scheduler())
        self._worker_tasks.append(scheduler)
        
        if self.audit_logger:
            self.audit_logger.log_system_start(component="task_manager")
        
        logger.info(f"Task manager started with {self._max_concurrent_tasks} workers")
    
    async def stop(self) -> None:
        """Stop the autonomous task execution system."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        
        if self.audit_logger:
            self.audit_logger.log_system_stop(component="task_manager")
        
        logger.info("Task manager stopped")
    
    async def create_task(
        self,
        title: str,
        description: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        timeout: float = 300.0
    ) -> AutonomousTask:
        """
        Create a new autonomous task.
        
        Args:
            title: Task title
            description: Task description
            task_type: Type of task (must have registered handler)
            priority: Task priority
            parameters: Task parameters
            dependencies: List of task IDs this task depends on
            timeout: Task timeout in seconds
        
        Returns:
            Created AutonomousTask
        """
        task_id = f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self._tasks)}"
        
        task = AutonomousTask(
            id=task_id,
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            parameters=parameters or {},
            dependencies=dependencies or [],
            timeout=timeout
        )
        
        # Validate task type has handler
        if task_type not in self._task_handlers:
            raise ValueError(f"No handler registered for task type: {task_type}")
        
        # Validate dependencies exist
        for dep_id in task.dependencies:
            if dep_id not in self._tasks:
                raise ValueError(f"Dependency task not found: {dep_id}")
        
        # Store task
        self._tasks[task_id] = task
        
        # Save to database
        await self._save_task_to_db(task)
        
        # Add to execution queue if no dependencies or dependencies are completed
        if self._can_execute_task(task):
            await self._execution_queue.put(task_id)
        
        if self.audit_logger:
            self.audit_logger.log_task_start(
                task_id,
                task_type=task_type,
                priority=priority.name,
                dependencies=len(task.dependencies)
            )
        
        logger.info(f"Created task {task_id}: {title}")
        return task
    
    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific task type.
        
        Args:
            task_type: Type of task
            handler: Async function that takes (task, context) and returns TaskResult
        """
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def get_task(self, task_id: str) -> Optional[AutonomousTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AutonomousTask]:
        """
        List tasks with optional filtering.
        
        Args:
            status: Filter by status
            task_type: Filter by task type
            limit: Maximum number of tasks to return
        
        Returns:
            List of matching tasks
        """
        tasks = list(self._tasks.values())
        
        # Apply filters
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        
        # Sort by priority and creation time
        tasks.sort(key=lambda t: (t.priority.value, t.created_at), reverse=True)
        
        return tasks[:limit]
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: ID of task to cancel
        
        Returns:
            True if task was cancelled
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.updated_at = datetime.utcnow()
        
        await self._save_task_to_db(task)
        
        if self.audit_logger:
            self.audit_logger.log_event(
                "task_cancelled",
                "task",
                task_id,
                details={"reason": "user_request"}
            )
        
        logger.info(f"Cancelled task {task_id}")
        return True
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        # Update stats from current tasks
        total_tasks = len(self._tasks)
        completed_tasks = len([t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self._tasks.values() if t.status == TaskStatus.FAILED])
        
        # Calculate average duration for completed tasks
        completed_with_duration = [
            t for t in self._tasks.values() 
            if t.status == TaskStatus.COMPLETED and t.result and t.result.duration > 0
        ]
        
        avg_duration = 0.0
        if completed_with_duration:
            avg_duration = sum(t.result.duration for t in completed_with_duration) / len(completed_with_duration)
        
        success_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0.0
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": len([t for t in self._tasks.values() if t.status == TaskStatus.PENDING]),
            "executing_tasks": len([t for t in self._tasks.values() if t.status == TaskStatus.EXECUTING]),
            "average_duration": avg_duration,
            "success_rate": success_rate,
            "task_types": list(set(t.task_type for t in self._tasks.values())),
            "queue_size": self._execution_queue.qsize()
        }
    
    async def _task_worker(self, worker_name: str) -> None:
        """Worker task that executes tasks from the queue."""
        logger.info(f"Task worker {worker_name} started")
        
        while self._is_running:
            try:
                # Get task from queue with timeout
                task_id = await asyncio.wait_for(
                    self._execution_queue.get(),
                    timeout=1.0
                )
                
                task = self._tasks.get(task_id)
                if not task:
                    logger.warning(f"Task {task_id} not found")
                    continue
                
                # Execute the task
                await self._execute_task(task, worker_name)
                
            except asyncio.TimeoutError:
                # Normal timeout, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.info(f"Task worker {worker_name} stopped")
    
    async def _task_scheduler(self) -> None:
        """Scheduler that adds ready tasks to the execution queue."""
        logger.info("Task scheduler started")
        
        while self._is_running:
            try:
                # Check for tasks that are ready to execute
                for task in self._tasks.values():
                    if (task.status == TaskStatus.PENDING and 
                        self._can_execute_task(task) and
                        task.id not in [item for item in list(self._execution_queue._queue)]):
                        
                        await self._execution_queue.put(task.id)
                        logger.debug(f"Scheduled task {task.id} for execution")
                
                # Sleep before next check
                await asyncio.sleep(5.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
        
        logger.info("Task scheduler stopped")
    
    def _can_execute_task(self, task: AutonomousTask) -> bool:
        """Check if a task can be executed (all dependencies completed)."""
        if task.status != TaskStatus.PENDING:
            return False
        
        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def _execute_task(self, task: AutonomousTask, worker_name: str) -> None:
        """Execute a single task."""
        logger.info(f"Worker {worker_name} executing task {task.id}: {task.title}")
        
        task.status = TaskStatus.EXECUTING
        task.started_at = datetime.utcnow()
        task.updated_at = datetime.utcnow()
        
        await self._save_task_to_db(task)
        
        try:
            # Get task handler
            handler = self._task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")
            
            # Execute with timeout
            start_time = datetime.utcnow()
            
            result = await asyncio.wait_for(
                handler(task, {"worker": worker_name}),
                timeout=task.timeout
            )
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Update task with result
            if isinstance(result, TaskResult):
                task.result = result
                task.result.duration = duration
            else:
                task.result = TaskResult(
                    success=True,
                    output=result,
                    duration=duration
                )
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            if self.audit_logger:
                self.audit_logger.log_task_complete(
                    task.id,
                    duration=duration,
                    success=True,
                    worker=worker_name
                )
            
            logger.info(f"Task {task.id} completed successfully in {duration:.2f}s")
            
        except asyncio.TimeoutError:
            task.result = TaskResult(
                success=False,
                error=f"Task timed out after {task.timeout}s"
            )
            task.status = TaskStatus.FAILED
            
            logger.error(f"Task {task.id} timed out after {task.timeout}s")
            
        except Exception as e:
            task.result = TaskResult(
                success=False,
                error=str(e)
            )
            
            # Check if we should retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                
                # Add back to queue for retry
                await self._execution_queue.put(task.id)
                
                logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
            else:
                task.status = TaskStatus.FAILED
                logger.error(f"Task {task.id} failed permanently: {e}")
            
            if self.audit_logger:
                self.audit_logger.log_error(
                    "task_execution_error",
                    str(e),
                    task_id=task.id,
                    worker=worker_name,
                    retry_count=task.retry_count
                )
        
        finally:
            task.updated_at = datetime.utcnow()
            await self._save_task_to_db(task)
    
    def _register_builtin_handlers(self) -> None:
        """Register built-in task handlers."""
        
        async def code_analysis_handler(task: AutonomousTask, context: Dict[str, Any]) -> TaskResult:
            """Handler for code analysis tasks."""
            # Placeholder implementation
            await asyncio.sleep(1.0)  # Simulate work
            
            return TaskResult(
                success=True,
                output={"analysis": "Code analysis completed", "files_analyzed": 42},
                metrics={"lines_of_code": 1500, "complexity_score": 7.2}
            )
        
        async def test_execution_handler(task: AutonomousTask, context: Dict[str, Any]) -> TaskResult:
            """Handler for test execution tasks."""
            # Placeholder implementation
            await asyncio.sleep(2.0)  # Simulate work
            
            return TaskResult(
                success=True,
                output={"tests_run": 25, "tests_passed": 23, "tests_failed": 2},
                metrics={"coverage": 85.5, "duration": 2.1}
            )
        
        async def documentation_handler(task: AutonomousTask, context: Dict[str, Any]) -> TaskResult:
            """Handler for documentation generation tasks."""
            # Placeholder implementation
            await asyncio.sleep(1.5)  # Simulate work
            
            return TaskResult(
                success=True,
                output={"docs_generated": ["README.md", "API.md"], "pages": 12},
                artifacts=["docs/README.md", "docs/API.md"]
            )
        
        # Register handlers
        self.register_task_handler("code_analysis", code_analysis_handler)
        self.register_task_handler("test_execution", test_execution_handler)
        self.register_task_handler("documentation", documentation_handler)
    
    async def _ensure_tasks_table(self) -> None:
        """Ensure the tasks table exists."""
        # This would typically be handled by migrations, but we'll ensure it exists
        pass
    
    async def _load_existing_tasks(self) -> None:
        """Load existing tasks from database."""
        try:
            query = """
            SELECT id, title, description, status, priority, created_at, updated_at,
                   started_at, completed_at, metadata
            FROM tasks
            WHERE status NOT IN ('completed', 'failed', 'cancelled')
            ORDER BY created_at DESC
            """
            
            results = await self.connector.execute_query(query, fetch_mode="all")
            
            for row in results:
                # Reconstruct task from database row
                metadata = row.get("metadata", {})
                
                task = AutonomousTask(
                    id=row["id"],
                    title=row["title"],
                    description=row["description"],
                    task_type=metadata.get("task_type", "unknown"),
                    priority=TaskPriority(metadata.get("priority", 2)),
                    status=TaskStatus(row["status"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    started_at=row.get("started_at"),
                    completed_at=row.get("completed_at"),
                    parameters=metadata.get("parameters", {}),
                    dependencies=metadata.get("dependencies", [])
                )
                
                self._tasks[task.id] = task
            
            logger.info(f"Loaded {len(results)} existing tasks from database")
            
        except Exception as e:
            logger.warning(f"Failed to load existing tasks: {e}")
    
    async def _save_task_to_db(self, task: AutonomousTask) -> None:
        """Save task to database."""
        try:
            metadata = {
                "task_type": task.task_type,
                "priority": task.priority.value,
                "parameters": task.parameters,
                "dependencies": task.dependencies,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "timeout": task.timeout
            }
            
            if task.result:
                metadata["result"] = {
                    "success": task.result.success,
                    "output": task.result.output,
                    "error": task.result.error,
                    "metrics": task.result.metrics,
                    "artifacts": task.result.artifacts,
                    "duration": task.result.duration
                }
            
            # Use INSERT ... ON CONFLICT UPDATE for upsert
            query = """
            INSERT INTO tasks (id, title, description, status, priority, created_at, updated_at, 
                             started_at, completed_at, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                status = EXCLUDED.status,
                priority = EXCLUDED.priority,
                updated_at = EXCLUDED.updated_at,
                started_at = EXCLUDED.started_at,
                completed_at = EXCLUDED.completed_at,
                metadata = EXCLUDED.metadata
            """
            
            parameters = {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "status": task.status.value,
                "priority": task.priority.value,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "metadata": json.dumps(metadata)
            }
            
            await self.connector.execute_query(query, parameters)
            
        except Exception as e:
            logger.error(f"Failed to save task {task.id} to database: {e}")


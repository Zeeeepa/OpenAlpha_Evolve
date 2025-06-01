# OpenEvolve Database Connector System

## Overview

The OpenEvolve Database Connector System is a comprehensive, production-ready database management solution designed for autonomous development. It provides enterprise-grade features including connection pooling, security, monitoring, and caching optimized for single-user autonomous operations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Query Builder  │  Migration Manager  │  Access Control    │
├─────────────────────────────────────────────────────────────┤
│  Connection Pool Manager  │  Autonomous Task Integration    │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL Connector  │  Redis Cache  │  Health Monitor   │
├─────────────────────────────────────────────────────────────┤
│  Metrics Collector  │  Audit Logger  │  Security Manager  │
├─────────────────────────────────────────────────────────────┤
│                    Database Layer                           │
│  PostgreSQL Database  │  Redis Cache  │  Configuration     │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Database Configuration (`config.py`)

Environment-based configuration management with validation for autonomous development:

```python
from openevolve.database.config import DatabaseConfig, load_database_config

# Load from environment variables
config = load_database_config()

# Or create manually for autonomous development
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="openevolve_autonomous",
    username="autonomous_user",
    password="password",
    min_pool_size=5,
    max_pool_size=20
)
```

**Environment Variables:**
- `DATABASE_URL` - Complete database URL
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `DB_MIN_POOL_SIZE`, `DB_MAX_POOL_SIZE`
- `DB_SSL_MODE`, `DB_SSL_CERT_PATH`

### 2. PostgreSQL Connector (`connectors/postgresql.py`)

High-performance async PostgreSQL connector optimized for autonomous development:

```python
from openevolve.database.connectors import PostgreSQLConnector

connector = PostgreSQLConnector(config)
await connector.initialize()

# Execute autonomous development queries
result = await connector.execute_query(
    "SELECT * FROM autonomous_tasks WHERE status = $1",
    {"status": "pending"},
    fetch_mode="all"
)

# Execute autonomous transactions
queries = [
    ("INSERT INTO code_analysis (file_path, complexity) VALUES ($1, $2)", 
     {"file_path": "main.py", "complexity": 5}, "none"),
    ("UPDATE autonomous_tasks SET status = $1 WHERE id = $2", 
     {"status": "completed", "id": 1}, "none")
]
await connector.execute_transaction(queries)
```

**Features:**
- Async/await support with asyncpg
- Connection pooling with optimization
- Health monitoring and metrics
- Timeout handling and error recovery
- SQLAlchemy integration

### 3. Connection Pool Manager (`connectors/pool_manager.py`)

Advanced connection pool management optimized for autonomous workloads:

```python
from openevolve.database.connectors import ConnectionPoolManager

pool_manager = ConnectionPoolManager(config)
await pool_manager.start()

# Execute autonomous queries through pool
result = await pool_manager.execute_query(
    "SELECT COUNT(*) FROM learning_patterns",
    fetch_mode="val"
)

# Get pool status for autonomous monitoring
status = await pool_manager.get_pool_status()
print(f"Active connections: {status['metrics']['active_connections']}")
```

**Features:**
- Automatic pool optimization for autonomous workloads
- Real-time metrics collection
- Connection recycling and load balancing
- Health monitoring and alerting
- Performance analytics

### 4. Query Builder (`query_builder.py`)

Advanced SQL query builder with injection prevention:

```python
from openevolve.database.query_builder import QueryBuilder, InsertQueryBuilder

# SELECT queries
query, params = (QueryBuilder()
    .select("id", "title", "status")
    .from_table("tasks", "t")
    .where("t.status", "=", "pending")
    .where("t.priority", ">", 0)
    .order_by("t.created_at")
    .limit(10)
    .build())

# INSERT queries
insert_query, insert_params = (InsertQueryBuilder()
    .into("tasks")
    .values(title="Autonomous Task", status="pending")
    .on_conflict_do_nothing()
    .returning("id")
    .build())
```

**Features:**
- Fluent interface for complex queries
- SQL injection prevention
- Support for JOINs, subqueries, and CTEs
- Parameter binding and validation
- Query optimization hints

### 5. Migration System (`migrations/`)

Versioned database schema management for autonomous development:

```python
from openevolve.database.migrations import MigrationManager, SQLMigration

# Create migration
migration = SQLMigration(
    up_sql="CREATE TABLE autonomous_logs (id SERIAL PRIMARY KEY, message TEXT);",
    down_sql="DROP TABLE autonomous_logs;"
)
migration.version = "005"
migration.name = "create_autonomous_logs"

# Run migrations
manager = MigrationManager(connector, config)
await manager.initialize()
applied = await manager.migrate()  # Apply all pending migrations
```

**Features:**
- Dependency resolution
- Rollback support
- Migration validation
- Progress tracking
- Checksum verification

### 6. Health Monitoring (`monitoring/health.py`)

Comprehensive database health monitoring for autonomous systems:

```python
from openevolve.database.monitoring import HealthMonitor

monitor = HealthMonitor(connector, pool_manager, config)
await monitor.start_monitoring()

# Get health status
health = await monitor.check_health()
print(f"Overall status: {health['overall_status']}")

# Add custom health checks
async def autonomous_system_check():
    # Your custom health check logic
    return HealthCheckResult("autonomous_system", HealthStatus.HEALTHY, 50.0)

monitor.register_health_check("autonomous_check", autonomous_system_check)
```

**Features:**
- Real-time health monitoring
- Configurable thresholds and alerts
- Custom health check registration
- Historical health data
- Dashboard integration

### 7. Metrics Collection (`monitoring/metrics.py`)

Advanced metrics collection for autonomous development analytics:

```python
from openevolve.database.monitoring import MetricsCollector

metrics = MetricsCollector()
metrics.start_collection()

# Record metrics
metrics.record_timing("task.execution_time", 150.5)
metrics.record_counter("tasks.completed", 1)
metrics.record_gauge("system.load", 0.75)

# Get summaries
summary = metrics.get_metric_summary("task.execution_time", hours=24)
print(f"Average task execution time: {summary.avg_value}ms")
```

**Features:**
- Multiple metric types (timing, counter, gauge)
- Statistical analysis and percentiles
- Historical data retention
- Export capabilities (JSON, CSV)
- Dashboard data formatting

### 8. Access Control (`security/access_control.py`)

Simplified access control for autonomous operations:

```python
from openevolve.database.security import AccessControlManager, Permission

acm = AccessControlManager()

# Create resource rules
from openevolve.database.security.access_control import AccessRule
rule = AccessRule(
    resource_type="table",
    resource_name="tasks",
    permissions={Permission.READ, Permission.WRITE},
    conditions={}
)
acm.add_resource_rule(rule)

# Check permissions (always returns True in autonomous mode)
has_access = acm.check_permission(
    Permission.WRITE, 
    "table", 
    "tasks"
)
```

**Features:**
- Resource-based permissions
- Operation logging
- System validation
- Autonomous-friendly defaults

### 9. Audit Logging (`security/audit.py`)

Comprehensive audit logging for autonomous operations:

```python
from openevolve.database.security import AuditLogger, AuditAction

audit = AuditLogger(connector)
await audit.initialize()

# Log operations
audit.log_create("task", "task_123", title="New autonomous task")
audit.log_task_start("task_123", details={"priority": "high"})
audit.log_task_complete("task_123", duration=45.2)

# Search audit logs
logs = await audit.search_audit_logs(
    action=AuditAction.TASK_COMPLETE,
    start_time=datetime.now() - timedelta(days=7)
)
```

**Features:**
- Autonomous operation logging
- Searchable audit trails
- Automatic buffering and flushing
- System event tracking
- Performance reporting

### 10. Redis Caching (`cache/redis_cache.py`)

High-performance Redis caching layer for autonomous systems:

```python
from openevolve.database.cache import RedisCache

cache = RedisCache(redis_config)
await cache.connect()

# Cache operations
await cache.set("tasks", "pending_list", task_data, ttl=300)
cached_data = await cache.get("tasks", "pending_list")

# Batch operations
await cache.set_multi("session", {
    "current_task": current_task_data,
    "system_state": system_state
})
```

**Features:**
- Async Redis operations
- Automatic serialization (JSON/Pickle)
- Batch operations for efficiency
- TTL management and expiration
- Health monitoring and statistics

## Usage Examples

### Basic Setup for Autonomous Development

```python
import asyncio
from openevolve.database import *

async def setup_autonomous_database():
    # Load configuration
    config = load_database_config()
    
    # Initialize connector
    connector = PostgreSQLConnector(config)
    await connector.initialize()
    
    # Setup pool manager
    pool_manager = ConnectionPoolManager(config)
    await pool_manager.start()
    
    # Run migrations
    migration_manager = MigrationManager(connector, config)
    await migration_manager.initialize()
    await migration_manager.migrate()
    
    # Start monitoring
    health_monitor = HealthMonitor(connector, pool_manager)
    await health_monitor.start_monitoring()
    
    # Initialize audit logging
    audit_logger = AuditLogger(connector)
    await audit_logger.initialize()
    audit_logger.log_system_start(version="1.0.0")
    
    return {
        "connector": connector,
        "pool_manager": pool_manager,
        "health_monitor": health_monitor,
        "audit_logger": audit_logger
    }

# Run setup
components = asyncio.run(setup_autonomous_database())
```

### Autonomous Task Management

```python
# Create autonomous task
task_query, params = (InsertQueryBuilder()
    .into("tasks")
    .values(
        title="Analyze codebase structure",
        description="Perform autonomous code analysis",
        status="pending",
        priority=1,
        metadata={"type": "analysis", "autonomous": True}
    )
    .returning("id")
    .build())

task_result = await connector.execute_query(task_query, params, fetch_mode="one")
task_id = task_result["id"]

# Log task creation
audit_logger.log_task_start(str(task_id), task_type="analysis")

# Update task status
update_query, update_params = (UpdateQueryBuilder()
    .table("tasks")
    .set(status="completed", completed_at="NOW()")
    .where("id", "=", task_id)
    .build())

await connector.execute_query(update_query, update_params)
audit_logger.log_task_complete(str(task_id), duration=120.5)
```

### Performance Monitoring

```python
# Track autonomous system performance
metrics = MetricsCollector()

# Record task metrics
metrics.record_timing("autonomous.task.duration", 120.5)
metrics.record_counter("autonomous.tasks.completed", 1)
metrics.record_gauge("autonomous.system.cpu_usage", 0.65)

# Get performance summary
performance_summary = metrics.get_metric_summary("autonomous.task.duration", hours=24)
print(f"Average task duration: {performance_summary.avg_value}s")
print(f"Tasks completed today: {metrics.get_counter_value('autonomous.tasks.completed')}")
```

## Configuration

### Database Configuration for Autonomous Systems

```python
# Environment variables
DATABASE_URL=postgresql://autonomous_user:password@localhost:5432/openevolve_autonomous
DB_MIN_POOL_SIZE=5
DB_MAX_POOL_SIZE=20
DB_QUERY_TIMEOUT=30.0
DB_SSL_MODE=prefer
DB_ENABLE_MONITORING=true
DB_ENABLE_AUDIT_LOGGING=true
```

### Redis Configuration

```python
# Environment variables
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20
REDIS_DEFAULT_TTL=300
REDIS_SSL_ENABLED=false
```

## Performance Optimization for Autonomous Systems

### Connection Pooling
- Configure pool sizes based on autonomous workload patterns
- Monitor connection usage and adjust dynamically
- Use connection recycling to prevent stale connections
- Implement connection health checks

### Query Optimization
- Use the query builder for complex autonomous queries
- Implement proper indexing for task and analytics tables
- Monitor query performance metrics
- Use connection pooling for high-throughput autonomous operations

### Caching Strategy
- Cache frequently accessed autonomous system data
- Use appropriate TTL values for different data types
- Implement cache invalidation for real-time updates
- Monitor cache hit rates and adjust accordingly

## Security Best Practices for Autonomous Systems

### Access Control
- Implement resource-based access controls
- Use simplified permission model for autonomous operations
- Regularly validate system permissions
- Monitor resource access patterns

### Data Protection
- Enable SSL/TLS for all connections
- Use parameter binding to prevent SQL injection
- Implement comprehensive audit logging
- Regular security assessments

### Autonomous System Security
- Ensure proper resource isolation
- Validate all autonomous operations
- Implement system-level access controls
- Monitor autonomous system behavior

## Monitoring and Alerting

### Health Monitoring
- Configure appropriate health check thresholds
- Set up alerting for critical autonomous system issues
- Monitor connection pool utilization
- Track autonomous task performance trends

### Metrics Collection
- Collect autonomous system-specific metrics
- Monitor database performance indicators
- Track autonomous task execution patterns
- Generate regular performance reports

## Troubleshooting

### Common Issues

1. **Connection Pool Exhaustion**
   - Increase pool size or optimize autonomous task patterns
   - Check for connection leaks in autonomous operations
   - Monitor connection usage patterns

2. **Slow Query Performance**
   - Analyze query execution plans for autonomous operations
   - Add appropriate indexes for task and analytics tables
   - Optimize autonomous query structure

3. **Cache Performance**
   - Monitor hit rates and adjust TTL for autonomous data
   - Check Redis memory usage
   - Optimize serialization methods for autonomous objects

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger('openevolve.database').setLevel(logging.DEBUG)
```

Use health checks:

```python
health_status = await health_monitor.check_health()
print(f"Database health: {health_status}")
```

## Testing

The system includes comprehensive tests for autonomous operations:

```bash
# Run tests
cd tests
python test_database_connector.py

# Or with pytest
pytest test_database_connector.py -v
```

## Autonomous Development Features

### Task Management
- Automated task creation and tracking
- Priority-based task execution
- Task dependency resolution
- Performance analytics

### System Monitoring
- Real-time health monitoring
- Autonomous system metrics
- Performance trend analysis
- Automated alerting

### Data Analytics
- Task execution analytics
- System performance metrics
- Resource utilization tracking
- Autonomous behavior analysis

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new autonomous features
3. Update documentation
4. Ensure security best practices
5. Add appropriate logging and monitoring for autonomous operations

## License

This database connector system is part of the OpenEvolve project and follows the same licensing terms.

# OpenEvolve Database Connector System

## Overview

The OpenEvolve Database Connector System is a comprehensive, production-ready database management solution designed for the autonomous development pipeline. It provides enterprise-grade features including connection pooling, multi-tenancy, security, monitoring, and caching.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Query Builder  │  Migration Manager  │  Access Control    │
├─────────────────────────────────────────────────────────────┤
│  Connection Pool Manager  │  Multi-Tenant Manager          │
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

Environment-based configuration management with validation:

```python
from openevolve.database.config import DatabaseConfig, load_database_config

# Load from environment variables
config = load_database_config()

# Or create manually
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="openevolve",
    username="user",
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

High-performance async PostgreSQL connector with connection pooling:

```python
from openevolve.database.connectors import PostgreSQLConnector

connector = PostgreSQLConnector(config)
await connector.initialize()

# Execute queries
result = await connector.execute_query(
    "SELECT * FROM users WHERE active = $1",
    {"active": True},
    fetch_mode="all"
)

# Execute transactions
queries = [
    ("INSERT INTO users (name) VALUES ($1)", {"name": "John"}, "none"),
    ("UPDATE users SET active = $1 WHERE id = $2", {"active": True, "id": 1}, "none")
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

Advanced connection pool management with monitoring and optimization:

```python
from openevolve.database.connectors import ConnectionPoolManager

pool_manager = ConnectionPoolManager(config)
await pool_manager.start()

# Execute queries through pool
result = await pool_manager.execute_query(
    "SELECT COUNT(*) FROM users",
    fetch_mode="val"
)

# Get pool status
status = await pool_manager.get_pool_status()
print(f"Active connections: {status['metrics']['active_connections']}")
```

**Features:**
- Automatic pool optimization
- Real-time metrics collection
- Connection recycling and load balancing
- Health monitoring and alerting
- Performance analytics

### 4. Multi-Tenant Manager (`connectors/multi_tenant.py`)

Schema-level tenant isolation with security:

```python
from openevolve.database.connectors import MultiTenantManager

tenant_manager = MultiTenantManager(connector, config)
await tenant_manager.initialize()

# Create tenant
tenant_info = await tenant_manager.create_tenant(
    "tenant_123",
    metadata={"company": "Acme Corp"}
)

# Execute tenant-specific queries
result = await tenant_manager.execute_tenant_query(
    "tenant_123",
    "SELECT * FROM users",
    fetch_mode="all"
)
```

**Features:**
- Automatic schema creation and isolation
- Tenant lifecycle management
- Security controls and access validation
- Resource usage tracking
- Cleanup and maintenance tools

### 5. Query Builder (`query_builder.py`)

Advanced SQL query builder with injection prevention:

```python
from openevolve.database.query_builder import QueryBuilder, InsertQueryBuilder

# SELECT queries
query, params = (QueryBuilder()
    .select("id", "name", "email")
    .from_table("users", "u")
    .join("profiles", "p.user_id = u.id", JoinType.LEFT, "p")
    .where("u.active", "=", True)
    .where("u.age", ">", 18)
    .order_by("u.name")
    .limit(10)
    .build())

# INSERT queries
insert_query, insert_params = (InsertQueryBuilder()
    .into("users")
    .values(name="John Doe", email="john@example.com")
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

### 6. Migration System (`migrations/`)

Versioned database schema management:

```python
from openevolve.database.migrations import MigrationManager, SQLMigration

# Create migration
migration = SQLMigration(
    up_sql="CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100));",
    down_sql="DROP TABLE users;"
)
migration.version = "001"
migration.name = "create_users_table"

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

### 7. Health Monitoring (`monitoring/health.py`)

Comprehensive database health monitoring:

```python
from openevolve.database.monitoring import HealthMonitor

monitor = HealthMonitor(connector, pool_manager, config)
await monitor.start_monitoring()

# Get health status
health = await monitor.check_health()
print(f"Overall status: {health['overall_status']}")

# Add custom health checks
async def custom_check():
    # Your custom health check logic
    return HealthCheckResult("custom", HealthStatus.HEALTHY, 50.0)

monitor.register_health_check("custom_check", custom_check)
```

**Features:**
- Real-time health monitoring
- Configurable thresholds and alerts
- Custom health check registration
- Historical health data
- Dashboard integration

### 8. Metrics Collection (`monitoring/metrics.py`)

Advanced metrics collection and analysis:

```python
from openevolve.database.monitoring import MetricsCollector

metrics = MetricsCollector()
metrics.start_collection()

# Record metrics
metrics.record_timing("query.duration", 150.5)
metrics.record_counter("requests.count", 1)
metrics.record_gauge("connections.active", 15)

# Get summaries
summary = metrics.get_metric_summary("query.duration", hours=24)
print(f"Average query time: {summary.avg_value}ms")
```

**Features:**
- Multiple metric types (timing, counter, gauge)
- Statistical analysis and percentiles
- Historical data retention
- Export capabilities (JSON, CSV)
- Dashboard data formatting

### 9. Access Control (`security/access_control.py`)

Role-based access control and security:

```python
from openevolve.database.security import AccessControlManager, Permission

acm = AccessControlManager()

# Create roles
acm.create_role("developer", {Permission.READ, Permission.WRITE})
acm.create_role("analyst", {Permission.READ})

# Assign roles
acm.assign_role("user123", "developer")

# Check permissions
has_access = acm.check_permission(
    "user123", 
    Permission.WRITE, 
    "table", 
    "users"
)
```

**Features:**
- Role-based permissions
- Fine-grained access rules
- API key management
- Context-aware authorization
- Audit trail integration

### 10. Audit Logging (`security/audit.py`)

Comprehensive audit logging system:

```python
from openevolve.database.security import AuditLogger, AuditAction

audit = AuditLogger(connector)
await audit.initialize()

# Log operations
audit.log_create("user123", "table", "users", {"name": "John"})
audit.log_update("user123", "table", "users", old_values, new_values)
audit.log_delete("user123", "table", "users", old_values)

# Search audit logs
logs = await audit.search_audit_logs(
    user_id="user123",
    action=AuditAction.CREATE,
    start_time=datetime.now() - timedelta(days=7)
)
```

**Features:**
- Comprehensive operation logging
- Searchable audit trails
- Automatic buffering and flushing
- Security event tracking
- Compliance reporting

### 11. Redis Caching (`cache/redis_cache.py`)

High-performance Redis caching layer:

```python
from openevolve.database.cache import RedisCache

cache = RedisCache(redis_config)
await cache.connect()

# Cache operations
await cache.set("queries", "user_list", user_data, ttl=300)
cached_data = await cache.get("queries", "user_list")

# Batch operations
await cache.set_multi("session", {
    "user_123": session_data,
    "user_456": other_session
})
```

**Features:**
- Async Redis operations
- Automatic serialization (JSON/Pickle)
- Batch operations for efficiency
- TTL management and expiration
- Health monitoring and statistics

## Usage Examples

### Basic Setup

```python
import asyncio
from openevolve.database import *

async def setup_database():
    # Load configuration
    config = load_database_config()
    
    # Initialize connector
    connector = PostgreSQLConnector(config)
    await connector.initialize()
    
    # Setup pool manager
    pool_manager = ConnectionPoolManager(config)
    await pool_manager.start()
    
    # Initialize multi-tenant support
    tenant_manager = MultiTenantManager(connector, config)
    await tenant_manager.initialize()
    
    # Run migrations
    migration_manager = MigrationManager(connector, config)
    await migration_manager.initialize()
    await migration_manager.migrate()
    
    # Start monitoring
    health_monitor = HealthMonitor(connector, pool_manager)
    await health_monitor.start_monitoring()
    
    return {
        "connector": connector,
        "pool_manager": pool_manager,
        "tenant_manager": tenant_manager,
        "health_monitor": health_monitor
    }

# Run setup
components = asyncio.run(setup_database())
```

### Query Execution

```python
# Simple query
users = await connector.execute_query(
    "SELECT * FROM users WHERE active = $1",
    {"active": True},
    fetch_mode="all"
)

# Complex query with builder
query, params = (QueryBuilder()
    .select("u.name", "p.title", "COUNT(c.id) as comment_count")
    .from_table("users", "u")
    .join("posts", "p.user_id = u.id", JoinType.LEFT, "p")
    .join("comments", "c.post_id = p.id", JoinType.LEFT, "c")
    .where("u.active", "=", True)
    .group_by("u.id", "u.name", "p.id", "p.title")
    .having("COUNT(c.id)", ">", 5)
    .order_by("comment_count", OrderDirection.DESC)
    .limit(20)
    .build())

results = await connector.execute_query(query, params, fetch_mode="all")
```

### Multi-Tenant Operations

```python
# Create tenant
tenant = await tenant_manager.create_tenant(
    "company_123",
    metadata={"name": "Acme Corp", "plan": "enterprise"}
)

# Execute tenant-specific operations
await tenant_manager.execute_tenant_query(
    "company_123",
    "CREATE TABLE company_data (id SERIAL PRIMARY KEY, data JSONB)"
)

# Insert tenant data
insert_query, params = (InsertQueryBuilder()
    .into("company_data")
    .values(data={"key": "value"})
    .returning("id")
    .build())

result = await tenant_manager.execute_tenant_query(
    "company_123", 
    insert_query, 
    params, 
    fetch_mode="one"
)
```

## Configuration

### Database Configuration

```python
# Environment variables
DATABASE_URL=postgresql://user:pass@localhost:5432/openevolve
DB_MIN_POOL_SIZE=5
DB_MAX_POOL_SIZE=20
DB_QUERY_TIMEOUT=30.0
DB_SSL_MODE=prefer
DB_ENABLE_MULTI_TENANT=true
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

## Performance Optimization

### Connection Pooling

- Configure pool sizes based on workload
- Monitor connection usage and adjust dynamically
- Use connection recycling to prevent stale connections
- Implement connection health checks

### Query Optimization

- Use the query builder for complex queries
- Implement proper indexing strategies
- Monitor query performance metrics
- Use connection pooling for high-throughput scenarios

### Caching Strategy

- Cache frequently accessed data
- Use appropriate TTL values
- Implement cache invalidation strategies
- Monitor cache hit rates and adjust accordingly

## Security Best Practices

### Access Control

- Implement least-privilege access
- Use role-based permissions
- Regularly audit user permissions
- Implement API key rotation

### Data Protection

- Enable SSL/TLS for all connections
- Use parameter binding to prevent SQL injection
- Implement audit logging for all operations
- Regular security assessments

### Multi-Tenant Security

- Ensure proper schema isolation
- Validate tenant context in all operations
- Implement tenant-specific access controls
- Monitor cross-tenant access attempts

## Monitoring and Alerting

### Health Monitoring

- Configure appropriate health check thresholds
- Set up alerting for critical issues
- Monitor connection pool utilization
- Track query performance trends

### Metrics Collection

- Collect application-specific metrics
- Monitor database performance indicators
- Track user activity and access patterns
- Generate regular performance reports

## Troubleshooting

### Common Issues

1. **Connection Pool Exhaustion**
   - Increase pool size or optimize query patterns
   - Check for connection leaks
   - Monitor connection usage patterns

2. **Slow Query Performance**
   - Analyze query execution plans
   - Add appropriate indexes
   - Optimize query structure

3. **Multi-Tenant Issues**
   - Verify schema isolation
   - Check tenant context validation
   - Monitor cross-tenant access

4. **Cache Performance**
   - Monitor hit rates and adjust TTL
   - Check Redis memory usage
   - Optimize serialization methods

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

The system includes comprehensive tests:

```bash
# Run tests
cd tests
python test_database_connector.py

# Or with pytest
pytest test_database_connector.py -v
```

## Migration from Other Systems

### From Raw SQL

Replace direct SQL execution with the query builder:

```python
# Before
cursor.execute("SELECT * FROM users WHERE active = %s", (True,))

# After
result = await connector.execute_query(
    "SELECT * FROM users WHERE active = $1",
    {"active": True},
    fetch_mode="all"
)
```

### From SQLAlchemy

Use the connector's SQLAlchemy integration:

```python
# Use async sessions
async with connector.get_async_session() as session:
    result = await session.execute(select(User).where(User.active == True))
    users = result.scalars().all()
```

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure security best practices
5. Add appropriate logging and monitoring

## License

This database connector system is part of the OpenEvolve project and follows the same licensing terms.


"""
Comprehensive test suite for the OpenEvolve database connector system.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import the database components
import sys
sys.path.insert(0, 'src')

from openevolve.database.config import DatabaseConfig, RedisConfig, load_database_config
from openevolve.database.connectors.postgresql import PostgreSQLConnector
from openevolve.database.connectors.pool_manager import ConnectionPoolManager
from openevolve.database.connectors.multi_tenant import MultiTenantManager
from openevolve.database.query_builder import QueryBuilder, InsertQueryBuilder, UpdateQueryBuilder
from openevolve.database.migrations.migration import SQLMigration, MigrationStatus
from openevolve.database.migrations.manager import MigrationManager
from openevolve.database.monitoring.health import HealthMonitor, HealthStatus
from openevolve.database.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseMigrationError,
    DatabaseSchemaError
)


class TestDatabaseConfig:
    """Test database configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "openevolve"
        assert config.min_pool_size == 5
        assert config.max_pool_size == 20
        assert config.enable_multi_tenant is True
        assert config.enable_monitoring is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid pool size
        with pytest.raises(ValueError, match="min_pool_size must be at least 1"):
            DatabaseConfig(min_pool_size=0)
        
        with pytest.raises(ValueError, match="max_pool_size must be >= min_pool_size"):
            DatabaseConfig(min_pool_size=10, max_pool_size=5)
        
        # Test invalid port
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            DatabaseConfig(port=0)
        
        # Test invalid SSL mode
        with pytest.raises(ValueError, match="Invalid ssl_mode"):
            DatabaseConfig(ssl_mode="invalid")
    
    def test_connection_url_generation(self):
        """Test connection URL generation."""
        config = DatabaseConfig(
            host="testhost",
            port=5433,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        
        url = config.connection_url
        assert "postgresql://testuser:testpass@testhost:5433/testdb" in url
    
    def test_load_config_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'DB_HOST': 'envhost',
            'DB_PORT': '5434',
            'DB_NAME': 'envdb',
            'DB_USER': 'envuser',
            'DB_PASSWORD': 'envpass',
            'DB_MIN_POOL_SIZE': '3',
            'DB_MAX_POOL_SIZE': '15'
        }):
            config = load_database_config()
            
            assert config.host == 'envhost'
            assert config.port == 5434
            assert config.database == 'envdb'
            assert config.username == 'envuser'
            assert config.password == 'envpass'
            assert config.min_pool_size == 3
            assert config.max_pool_size == 15


class TestQueryBuilder:
    """Test query builder functionality."""
    
    def test_basic_select_query(self):
        """Test basic SELECT query building."""
        builder = QueryBuilder()
        query, params = (builder
                        .select("id", "name", "email")
                        .from_table("users")
                        .where("active", "=", True)
                        .order_by("name")
                        .limit(10)
                        .build())
        
        assert "SELECT id, name, email" in query
        assert "FROM users" in query
        assert "WHERE active = $param_1" in query
        assert "ORDER BY name ASC" in query
        assert "LIMIT 10" in query
        assert params["param_1"] is True
    
    def test_complex_query_with_joins(self):
        """Test complex query with JOINs."""
        from openevolve.database.query_builder import JoinType, OrderDirection
        
        builder = QueryBuilder()
        query, params = (builder
                        .select("u.name", "p.title")
                        .from_table("users", "u")
                        .join("posts", "p.user_id = u.id", JoinType.LEFT, "p")
                        .where("u.active", "=", True)
                        .where("p.published", "=", True, "AND")
                        .order_by("u.name", OrderDirection.ASC)
                        .order_by("p.created_at", OrderDirection.DESC)
                        .build())
        
        assert "SELECT u.name, p.title" in query
        assert "FROM users AS u" in query
        assert "LEFT JOIN posts AS p ON p.user_id = u.id" in query
        assert "WHERE u.active = $param_1 AND p.published = $param_2" in query
        assert "ORDER BY u.name ASC, p.created_at DESC" in query
    
    def test_where_in_clause(self):
        """Test WHERE IN clause."""
        builder = QueryBuilder()
        query, params = (builder
                        .select("*")
                        .from_table("users")
                        .where_in("id", [1, 2, 3, 4, 5])
                        .build())
        
        assert "WHERE id IN ($param_1, $param_2, $param_3, $param_4, $param_5)" in query
        assert params["param_1"] == 1
        assert params["param_5"] == 5
    
    def test_pagination(self):
        """Test pagination functionality."""
        builder = QueryBuilder()
        query, params = (builder
                        .select("*")
                        .from_table("users")
                        .paginate(page=3, per_page=20)
                        .build())
        
        assert "LIMIT 20" in query
        assert "OFFSET 40" in query  # (3-1) * 20
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        builder = QueryBuilder()
        
        # Test dangerous column name
        with pytest.raises(ValueError, match="Invalid column identifier"):
            builder.select("id; DROP TABLE users; --")
        
        # Test dangerous table name
        with pytest.raises(ValueError, match="Invalid table identifier"):
            builder.from_table("users; DROP TABLE posts; --")
    
    def test_insert_query_builder(self):
        """Test INSERT query builder."""
        builder = InsertQueryBuilder()
        query, params = (builder
                        .into("users")
                        .values(name="John Doe", email="john@example.com", active=True)
                        .values(name="Jane Smith", email="jane@example.com", active=False)
                        .returning("id", "name")
                        .build())
        
        assert "INSERT INTO users" in query
        assert "(name, email, active)" in query
        assert "VALUES ($param_1, $param_2, $param_3), ($param_4, $param_5, $param_6)" in query
        assert "RETURNING id, name" in query
        assert params["param_1"] == "John Doe"
        assert params["param_4"] == "Jane Smith"
    
    def test_update_query_builder(self):
        """Test UPDATE query builder."""
        builder = UpdateQueryBuilder()
        query, params = (builder
                        .table("users")
                        .set(name="Updated Name", email="updated@example.com")
                        .where("id", "=", 123)
                        .returning("id", "name", "updated_at")
                        .build())
        
        assert "UPDATE users" in query
        assert "SET name = $param_1, email = $param_2" in query
        assert "WHERE id = $param_3" in query
        assert "RETURNING id, name, updated_at" in query
        assert params["param_1"] == "Updated Name"
        assert params["param_3"] == 123


class TestMigrationSystem:
    """Test migration system."""
    
    def test_sql_migration(self):
        """Test SQL migration creation."""
        up_sql = "CREATE TABLE test_table (id SERIAL PRIMARY KEY, name VARCHAR(100));"
        down_sql = "DROP TABLE test_table;"
        
        migration = SQLMigration(up_sql, down_sql)
        migration.version = "001"
        migration.name = "create_test_table"
        migration.description = "Create test table"
        
        assert migration.version == "001"
        assert migration.name == "create_test_table"
        assert migration.reversible is True
        assert migration.up_sql == up_sql
        assert migration.down_sql == down_sql
    
    def test_migration_sql_splitting(self):
        """Test SQL statement splitting."""
        multi_sql = """
        CREATE TABLE users (id SERIAL PRIMARY KEY);
        CREATE INDEX idx_users_id ON users(id);
        INSERT INTO users (id) VALUES (1);
        """
        
        migration = SQLMigration(multi_sql)
        statements = migration._split_sql(multi_sql)
        
        assert len(statements) == 3
        assert "CREATE TABLE users" in statements[0]
        assert "CREATE INDEX" in statements[1]
        assert "INSERT INTO users" in statements[2]
    
    @pytest.mark.asyncio
    async def test_migration_manager_initialization(self):
        """Test migration manager initialization."""
        # Mock connector
        mock_connector = AsyncMock()
        mock_connector.execute_query = AsyncMock()
        
        config = DatabaseConfig()
        manager = MigrationManager(mock_connector, config)
        
        # Should have built-in migrations registered
        assert len(manager.migrations) > 0
        assert "001" in manager.migrations  # CreateTenantRegistryMigration
    
    @pytest.mark.asyncio
    async def test_dependency_resolution(self):
        """Test migration dependency resolution."""
        mock_connector = AsyncMock()
        config = DatabaseConfig()
        manager = MigrationManager(mock_connector, config)
        
        # Create test migrations with dependencies
        migration1 = SQLMigration("CREATE TABLE test1 (id SERIAL);")
        migration1.version = "001"
        migration1.name = "create_test1"
        migration1.dependencies = []
        
        migration2 = SQLMigration("CREATE TABLE test2 (id SERIAL);")
        migration2.version = "002"
        migration2.name = "create_test2"
        migration2.dependencies = ["001"]
        
        migration3 = SQLMigration("CREATE TABLE test3 (id SERIAL);")
        migration3.version = "003"
        migration3.name = "create_test3"
        migration3.dependencies = ["001", "002"]
        
        # Clear existing migrations and add test ones
        manager.migrations.clear()
        manager.register_migration(migration3)  # Register out of order
        manager.register_migration(migration1)
        manager.register_migration(migration2)
        
        # Resolve dependencies
        resolved_order = manager._resolve_dependencies()
        
        assert resolved_order == ["001", "002", "003"]


class TestHealthMonitoring:
    """Test health monitoring system."""
    
    @pytest.mark.asyncio
    async def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        mock_connector = AsyncMock()
        monitor = HealthMonitor(mock_connector)
        
        # Should have default health checks registered
        assert len(monitor._health_checks) > 0
        assert "database_connectivity" in monitor._health_checks
        assert "connection_pool" in monitor._health_checks
    
    @pytest.mark.asyncio
    async def test_database_connectivity_check(self):
        """Test database connectivity health check."""
        mock_connector = AsyncMock()
        mock_connector.execute_query = AsyncMock(return_value=1)
        
        monitor = HealthMonitor(mock_connector)
        result = await monitor._check_database_connectivity()
        
        assert result.name == "database_connectivity"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms >= 0
        assert "successful" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure handling."""
        mock_connector = AsyncMock()
        mock_connector.execute_query = AsyncMock(side_effect=Exception("Connection failed"))
        
        monitor = HealthMonitor(mock_connector)
        result = await monitor._check_database_connectivity()
        
        assert result.name == "database_connectivity"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in result.message
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self):
        """Test comprehensive health check."""
        mock_connector = AsyncMock()
        mock_connector.execute_query = AsyncMock(return_value=1)
        mock_connector.get_pool_status = AsyncMock(return_value={
            "status": "healthy",
            "size": 10,
            "idle_connections": 5
        })
        
        monitor = HealthMonitor(mock_connector)
        health_status = await monitor.check_health()
        
        assert "overall_status" in health_status
        assert "checks" in health_status
        assert "total_check_time_ms" in health_status
        assert len(health_status["checks"]) > 0


class TestErrorHandling:
    """Test error handling and exceptions."""
    
    def test_database_connection_error(self):
        """Test database connection error."""
        error = DatabaseConnectionError(
            "Connection failed",
            "postgresql://user@host:5432/db",
            Exception("Original error")
        )
        
        assert str(error) == "Connection failed"
        assert error.connection_string == "postgresql://user@host:5432/db"
        assert isinstance(error.original_error, Exception)
    
    def test_database_query_error(self):
        """Test database query error."""
        error = DatabaseQueryError(
            "Query failed",
            "SELECT * FROM users",
            {"id": 123},
            Exception("SQL error")
        )
        
        assert str(error) == "Query failed"
        assert error.query == "SELECT * FROM users"
        assert error.parameters == {"id": 123}
    
    def test_database_migration_error(self):
        """Test database migration error."""
        error = DatabaseMigrationError(
            "Migration failed",
            "001",
            Exception("Migration error")
        )
        
        assert str(error) == "Migration failed"
        assert error.migration_version == "001"
    
    def test_database_schema_error(self):
        """Test database schema error."""
        error = DatabaseSchemaError(
            "Schema operation failed",
            "test_schema",
            "tenant_123"
        )
        
        assert str(error) == "Schema operation failed"
        assert error.schema_name == "test_schema"
        assert error.tenant_id == "tenant_123"


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_mock_database_workflow(self):
        """Test complete database workflow with mocked components."""
        # Create mock connector
        mock_connector = AsyncMock()
        mock_connector.execute_query = AsyncMock()
        mock_connector.execute_transaction = AsyncMock()
        mock_connector.get_pool_status = AsyncMock(return_value={
            "status": "healthy",
            "size": 10,
            "idle_connections": 5
        })
        mock_connector.health_check = AsyncMock(return_value={
            "status": "healthy",
            "response_time_ms": 50.0
        })
        
        config = DatabaseConfig()
        
        # Test pool manager
        pool_manager = ConnectionPoolManager(config)
        pool_manager.connectors["default"] = mock_connector
        
        # Test query execution
        await pool_manager.execute_query("SELECT 1", fetch_mode="val")
        mock_connector.execute_query.assert_called_once()
        
        # Test health monitoring
        health_monitor = HealthMonitor(mock_connector, pool_manager, config)
        health_status = await health_monitor.check_health()
        
        assert health_status["overall_status"] in ["healthy", "degraded", "unhealthy"]
        assert "checks" in health_status
        
        # Test migration system
        migration_manager = MigrationManager(mock_connector, config)
        await migration_manager.initialize()
        
        # Should have built-in migrations
        assert len(migration_manager.migrations) > 0
        
        migration_status = migration_manager.get_migration_status()
        assert "total_migrations" in migration_status
        assert "applied_migrations" in migration_status


def test_system_validation():
    """Test that all components can be imported and instantiated."""
    # Test that all main components can be imported
    from openevolve.database import (
        PostgreSQLConnector,
        ConnectionPoolManager,
        MultiTenantManager,
        MigrationManager,
        HealthMonitor,
        QueryBuilder
    )
    
    # Test that configuration can be created
    config = DatabaseConfig()
    assert config is not None
    
    # Test that query builder works
    builder = QueryBuilder()
    query, params = builder.select("*").from_table("test").build()
    assert "SELECT *" in query
    assert "FROM test" in query
    
    print("✅ All database connector components validated successfully!")


if __name__ == "__main__":
    # Run basic validation
    test_system_validation()
    
    # Run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic validation only")
        
        # Run some basic tests manually
        test_config = TestDatabaseConfig()
        test_config.test_default_config()
        test_config.test_config_validation()
        
        test_query = TestQueryBuilder()
        test_query.test_basic_select_query()
        test_query.test_sql_injection_prevention()
        
        print("✅ Basic tests completed successfully!")


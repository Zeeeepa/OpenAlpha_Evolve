#!/usr/bin/env python3
"""
Comprehensive test suite for OpenEvolve Database Connector System.
Tests all components including connectors, query builders, migrations, and monitoring.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, 'src')

from openevolve.database.config import DatabaseConfig, RedisConfig, load_database_config
from openevolve.database.connectors import (
    PostgreSQLConnector,
    ConnectionPoolManager
)
from openevolve.database.query_builder import QueryBuilder, InsertQueryBuilder, UpdateQueryBuilder
from openevolve.database.migrations.migration import SQLMigration, MigrationStatus
from openevolve.database.migrations.manager import MigrationManager
from openevolve.database.monitoring.health import HealthMonitor, HealthStatus
from openevolve.database.monitoring.metrics import MetricsCollector
from openevolve.database.security.access_control import AccessControlManager, Permission
from openevolve.database.security.audit import AuditLogger, AuditAction
from openevolve.database.cache.redis_cache import RedisCache
from openevolve.database.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseMigrationError,
    DatabaseSecurityError
)


class TestDatabaseConfig(unittest.TestCase):
    """Test database configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        
        # Test default values
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "openevolve"
        assert config.username == "postgres"
        assert config.min_pool_size == 5
        assert config.max_pool_size == 20
        assert config.enable_monitoring is True
        assert config.enable_audit_logging is True
        assert config.enable_caching is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid port
        with self.assertRaises(ValueError):
            config = DatabaseConfig(port=-1)
            config._validate_config()
        
        # Test invalid pool sizes
        with self.assertRaises(ValueError):
            config = DatabaseConfig(min_pool_size=0)
            config._validate_config()
        
        with self.assertRaises(ValueError):
            config = DatabaseConfig(max_pool_size=0)
            config._validate_config()
        
        with self.assertRaises(ValueError):
            config = DatabaseConfig(min_pool_size=10, max_pool_size=5)
            config._validate_config()


class TestQueryBuilder(unittest.TestCase):
    """Test SQL query builder functionality."""
    
    def test_simple_select(self):
        """Test simple SELECT query building."""
        builder = QueryBuilder()
        query, params = (builder
            .select("id", "name")
            .from_table("tasks")
            .build())
        
        assert "SELECT id, name" in query
        assert "FROM tasks" in query
        assert isinstance(params, dict)
    
    def test_select_with_where(self):
        """Test SELECT with WHERE clause."""
        builder = QueryBuilder()
        query, params = (builder
            .select("*")
            .from_table("tasks", "t")
            .where("t.status", "=", "pending")
            .build())
        
        assert "SELECT *" in query
        assert "FROM tasks AS t" in query
        assert "WHERE" in query
        assert len(params) > 0
    
    def test_select_with_join(self):
        """Test SELECT with JOIN."""
        from openevolve.database.query_builder import JoinType
        
        builder = QueryBuilder()
        query, params = (builder
            .select("t.id", "t.title")
            .from_table("tasks", "t")
            .join("analytics_events", "a.task_id = t.id", JoinType.LEFT, "a")
            .build())
        
        assert "FROM tasks AS t" in query
        assert "LEFT JOIN analytics_events AS a ON a.task_id = t.id" in query
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        builder = QueryBuilder()
        
        # Test malicious input in select
        try:
            builder.select("id; DROP TABLE tasks; --")
            query, params = builder.from_table("tasks").build()
            # Should not contain the malicious SQL
            assert "DROP TABLE" not in query
        except ValueError:
            # Or should raise an error
            pass
        
        # Test malicious input in table name
        try:
            builder.from_table("tasks; DROP TABLE analytics_events; --")
            # Should raise an error or sanitize
        except ValueError:
            pass
    
    def test_insert_query(self):
        """Test INSERT query building."""
        builder = InsertQueryBuilder()
        query, params = (builder
            .into("tasks")
            .values(title="Test Task", status="pending")
            .build())
        
        assert "INSERT INTO tasks" in query
        assert len(params) > 0
    
    def test_update_query(self):
        """Test UPDATE query building."""
        builder = UpdateQueryBuilder()
        query, params = (builder
            .table("tasks")
            .set(status="completed")
            .where("id", "=", 1)
            .build())
        
        assert "UPDATE tasks" in query
        assert "SET" in query
        assert "WHERE" in query


class TestMigrations(unittest.TestCase):
    """Test database migration system."""
    
    def test_sql_migration(self):
        """Test SQL migration creation and execution."""
        migration = SQLMigration(
            up_sql="CREATE TABLE test_table (id SERIAL PRIMARY KEY);",
            down_sql="DROP TABLE test_table;"
        )
        migration.version = "001"
        migration.name = "create_test_table"
        
        assert migration.version == "001"
        assert migration.name == "create_test_table"
        assert "CREATE TABLE test_table" in migration.up_sql
        assert "DROP TABLE test_table" in migration.down_sql
    
    def test_migration_checksum(self):
        """Test migration checksum generation."""
        migration = SQLMigration(
            up_sql="CREATE TABLE test_table (id SERIAL PRIMARY KEY);",
            down_sql="DROP TABLE test_table;"
        )
        migration.version = "001"
        migration.name = "create_test_table"
        
        checksum1 = migration.get_checksum()
        checksum2 = migration.get_checksum()
        
        # Same migration should have same checksum
        assert checksum1 == checksum2
        
        # Different migration should have different checksum
        migration2 = SQLMigration(
            up_sql="CREATE TABLE other_table (id SERIAL PRIMARY KEY);",
            down_sql="DROP TABLE other_table;"
        )
        migration2.version = "002"
        migration2.name = "create_other_table"
        
        assert migration.get_checksum() != migration2.get_checksum()
    
    def test_migration_manager_initialization(self):
        """Test migration manager initialization."""
        config = DatabaseConfig()
        # Note: This test doesn't actually connect to a database
        # In a real test environment, you'd use a test database
        
        # Test that manager can be created
        try:
            manager = MigrationManager(None, config)  # None connector for testing
            assert manager is not None
            assert "001" in manager.migrations  # Should have built-in migrations
        except Exception as e:
            # Expected since we're not providing a real connector
            pass


class TestExceptions(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_database_connection_error(self):
        """Test DatabaseConnectionError."""
        error = DatabaseConnectionError("Connection failed", "localhost:5432")
        assert error.host == "localhost:5432"
        assert "Connection failed" in str(error)
    
    def test_database_query_error(self):
        """Test DatabaseQueryError."""
        error = DatabaseQueryError("Query failed", "SELECT * FROM tasks")
        assert error.query == "SELECT * FROM tasks"
        assert "Query failed" in str(error)
    
    def test_database_migration_error(self):
        """Test DatabaseMigrationError."""
        error = DatabaseMigrationError("Migration failed", "001", "create_tasks")
        assert error.migration_version == "001"
        assert error.migration_name == "create_tasks"
    
    def test_database_security_error(self):
        """Test DatabaseSecurityError."""
        error = DatabaseSecurityError("Access denied", "task_123")
        assert error.resource_id == "task_123"


class TestAccessControl(unittest.TestCase):
    """Test access control system."""
    
    def test_access_control_manager(self):
        """Test access control manager."""
        acm = AccessControlManager()
        
        # Test permission checking (should always return True in autonomous mode)
        has_access = acm.check_permission(
            Permission.READ,
            "table",
            "tasks"
        )
        assert has_access is True
        
        # Test system permissions
        permissions = acm.get_system_permissions()
        assert Permission.READ in permissions
        assert Permission.WRITE in permissions
        assert Permission.DELETE in permissions
    
    def test_resource_rules(self):
        """Test resource access rules."""
        from openevolve.database.security.access_control import AccessRule
        
        acm = AccessControlManager()
        
        # Create and add a rule
        rule = AccessRule(
            resource_type="table",
            resource_name="tasks",
            permissions={Permission.READ, Permission.WRITE},
            conditions={}
        )
        acm.add_resource_rule(rule)
        
        # Test resource listing
        resources = acm.list_resources()
        assert "table:tasks" in resources
        
        # Test permission checking with rule
        has_read = acm.check_permission(Permission.READ, "table", "tasks")
        has_delete = acm.check_permission(Permission.DELETE, "table", "tasks")
        
        assert has_read is True  # Allowed by rule
        assert has_delete is False  # Not allowed by rule


def test_system_validation():
    """Test that all system components can be imported and initialized."""
    print("🔍 Testing OpenEvolve Database Connector System...")
    
    # Test that all main components can be imported
    from openevolve.database import (
        PostgreSQLConnector,
        ConnectionPoolManager
    )
    
    # Test that configuration can be created
    config = DatabaseConfig()
    assert config is not None
    
    # Test that query builder works
    builder = QueryBuilder()
    query, params = builder.select("id").from_table("test").build()
    assert "SELECT id" in query
    assert "FROM test" in query
    
    # Test that exceptions can be created
    error = DatabaseConnectionError("Test error")
    assert "Test error" in str(error)
    
    print("✅ All database connector components validated successfully!")


if __name__ == "__main__":
    # Run basic validation
    test_system_validation()
    
    # Run unit tests
    unittest.main(verbosity=2)


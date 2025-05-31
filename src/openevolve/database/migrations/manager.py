"""
Database migration manager with versioning and dependency resolution.
"""

import asyncio
import logging
import hashlib
import importlib
import os
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from pathlib import Path

from ..config import DatabaseConfig
from ..exceptions import DatabaseMigrationError
from ..connectors.postgresql import PostgreSQLConnector
from .migration import Migration, MigrationInfo, MigrationStatus
from .migration import (
    CreateTenantRegistryMigration,
    CreateTaskManagementMigration,
    CreateAnalyticsMigration,
    CreateAuditLogMigration
)

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Database migration manager with versioning, dependency resolution,
    and rollback capabilities.
    """
    
    def __init__(self, connector: PostgreSQLConnector, config: DatabaseConfig):
        self.connector = connector
        self.config = config
        self.migrations: Dict[str, Migration] = {}
        self.migration_history: Dict[str, MigrationInfo] = {}
        self._lock = asyncio.Lock()
        
        # Register built-in migrations
        self._register_builtin_migrations()
        
        logger.info("Migration manager initialized")
    
    def _register_builtin_migrations(self) -> None:
        """Register built-in migrations."""
        builtin_migrations = [
            CreateTenantRegistryMigration(),
            CreateTaskManagementMigration(),
            CreateAnalyticsMigration(),
            CreateAuditLogMigration()
        ]
        
        for migration in builtin_migrations:
            self.register_migration(migration)
    
    async def initialize(self) -> None:
        """Initialize the migration system."""
        # Create migration tracking table
        await self._create_migration_table()
        
        # Load migration history
        await self._load_migration_history()
        
        logger.info(f"Migration manager initialized with {len(self.migrations)} migrations")
    
    async def _create_migration_table(self) -> None:
        """Create the migration tracking table."""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.config.migration_schema}.{self.config.migration_table} (
            version VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            checksum VARCHAR(64) NOT NULL,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            execution_time_ms NUMERIC,
            status VARCHAR(50) DEFAULT 'completed',
            error_message TEXT,
            rolled_back_at TIMESTAMP WITH TIME ZONE
        );
        
        CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at 
        ON {self.config.migration_schema}.{self.config.migration_table}(applied_at);
        
        CREATE INDEX IF NOT EXISTS idx_schema_migrations_status 
        ON {self.config.migration_schema}.{self.config.migration_table}(status);
        """
        
        try:
            await self.connector.execute_query(create_table_query, fetch_mode="none")
            logger.info("Migration tracking table created/verified")
        except Exception as e:
            logger.error(f"Failed to create migration table: {e}")
            raise DatabaseMigrationError(f"Failed to create migration table: {e}")
    
    async def _load_migration_history(self) -> None:
        """Load migration history from the database."""
        query = f"""
        SELECT version, name, description, checksum, applied_at, 
               execution_time_ms, status, error_message, rolled_back_at
        FROM {self.config.migration_schema}.{self.config.migration_table}
        ORDER BY applied_at
        """
        
        try:
            history = await self.connector.execute_query(query, fetch_mode="all")
            
            for record in history:
                migration_info = MigrationInfo(
                    version=record["version"],
                    name=record["name"],
                    description=record["description"],
                    status=MigrationStatus(record["status"]),
                    applied_at=record["applied_at"],
                    rolled_back_at=record["rolled_back_at"],
                    execution_time_ms=record["execution_time_ms"],
                    error_message=record["error_message"],
                    checksum=record["checksum"]
                )
                self.migration_history[record["version"]] = migration_info
            
            logger.info(f"Loaded {len(self.migration_history)} migration records")
            
        except Exception as e:
            logger.error(f"Failed to load migration history: {e}")
            # Don't raise here - system can continue without history
    
    def register_migration(self, migration: Migration) -> None:
        """
        Register a migration.
        
        Args:
            migration: Migration instance to register
        """
        if not migration.version:
            raise ValueError("Migration version is required")
        
        if migration.version in self.migrations:
            logger.warning(f"Migration {migration.version} already registered, replacing")
        
        self.migrations[migration.version] = migration
        logger.debug(f"Registered migration {migration.version}: {migration.name}")
    
    def load_migrations_from_directory(self, directory: str) -> None:
        """
        Load migrations from a directory.
        
        Args:
            directory: Directory path containing migration files
        """
        migration_dir = Path(directory)
        if not migration_dir.exists():
            logger.warning(f"Migration directory {directory} does not exist")
            return
        
        # Find Python files that look like migrations
        migration_files = sorted(migration_dir.glob("*.py"))
        
        for file_path in migration_files:
            if file_path.name.startswith("__"):
                continue
            
            try:
                # Import the module
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for Migration classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, Migration) and 
                        attr != Migration):
                        
                        # Instantiate and register the migration
                        migration = attr()
                        self.register_migration(migration)
                        logger.info(f"Loaded migration from {file_path}: {migration.version}")
                        
            except Exception as e:
                logger.error(f"Failed to load migration from {file_path}: {e}")
    
    def _calculate_checksum(self, migration: Migration) -> str:
        """Calculate checksum for a migration."""
        # Create a string representation of the migration
        migration_data = f"{migration.version}:{migration.name}:{migration.description}"
        
        # Add migration-specific data if available
        if hasattr(migration, 'up_sql'):
            migration_data += f":{migration.up_sql}"
        if hasattr(migration, 'down_sql'):
            migration_data += f":{migration.down_sql}"
        
        return hashlib.sha256(migration_data.encode()).hexdigest()
    
    def _resolve_dependencies(self, target_version: Optional[str] = None) -> List[str]:
        """
        Resolve migration dependencies and return ordered list of versions.
        
        Args:
            target_version: Target version to migrate to (None for latest)
        
        Returns:
            Ordered list of migration versions
        """
        if target_version and target_version not in self.migrations:
            raise DatabaseMigrationError(f"Migration {target_version} not found")
        
        # Get all migrations up to target version
        all_versions = sorted(self.migrations.keys())
        
        if target_version:
            target_index = all_versions.index(target_version)
            versions_to_apply = all_versions[:target_index + 1]
        else:
            versions_to_apply = all_versions
        
        # Check dependencies
        resolved_order = []
        remaining = set(versions_to_apply)
        
        while remaining:
            # Find migrations with no unresolved dependencies
            ready_migrations = []
            
            for version in remaining:
                migration = self.migrations[version]
                dependencies_met = all(
                    dep in resolved_order or dep in self.migration_history
                    for dep in migration.dependencies
                )
                
                if dependencies_met:
                    ready_migrations.append(version)
            
            if not ready_migrations:
                # Circular dependency or missing dependency
                missing_deps = []
                for version in remaining:
                    migration = self.migrations[version]
                    for dep in migration.dependencies:
                        if dep not in resolved_order and dep not in self.migration_history:
                            missing_deps.append(f"{version} -> {dep}")
                
                raise DatabaseMigrationError(
                    f"Circular dependency or missing dependencies: {missing_deps}"
                )
            
            # Add ready migrations to resolved order
            for version in sorted(ready_migrations):
                resolved_order.append(version)
                remaining.remove(version)
        
        return resolved_order
    
    async def migrate(self, target_version: Optional[str] = None) -> List[str]:
        """
        Apply migrations up to target version.
        
        Args:
            target_version: Target version (None for latest)
        
        Returns:
            List of applied migration versions
        """
        async with self._lock:
            try:
                # Resolve migration order
                migration_order = self._resolve_dependencies(target_version)
                
                # Filter out already applied migrations
                pending_migrations = [
                    version for version in migration_order
                    if version not in self.migration_history or 
                    self.migration_history[version].status != MigrationStatus.COMPLETED
                ]
                
                if not pending_migrations:
                    logger.info("No pending migrations to apply")
                    return []
                
                applied_migrations = []
                
                for version in pending_migrations:
                    migration = self.migrations[version]
                    
                    logger.info(f"Applying migration {version}: {migration.name}")
                    
                    # Record migration start
                    await self._record_migration_start(migration)
                    
                    try:
                        start_time = asyncio.get_event_loop().time()
                        
                        # Apply migration
                        await migration.up(self.connector)
                        
                        # Validate migration
                        if not await migration.validate(self.connector):
                            raise DatabaseMigrationError(f"Migration {version} validation failed")
                        
                        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
                        
                        # Record successful completion
                        await self._record_migration_completion(migration, execution_time)
                        
                        applied_migrations.append(version)
                        logger.info(f"Successfully applied migration {version} in {execution_time:.2f}ms")
                        
                    except Exception as e:
                        logger.error(f"Migration {version} failed: {e}")
                        
                        # Record failure
                        await self._record_migration_failure(migration, str(e))
                        
                        raise DatabaseMigrationError(f"Migration {version} failed: {e}", version, e)
                
                logger.info(f"Applied {len(applied_migrations)} migrations successfully")
                return applied_migrations
                
            except Exception as e:
                logger.error(f"Migration process failed: {e}")
                raise
    
    async def rollback(self, target_version: str) -> List[str]:
        """
        Rollback migrations to target version.
        
        Args:
            target_version: Version to rollback to
        
        Returns:
            List of rolled back migration versions
        """
        async with self._lock:
            try:
                # Get applied migrations in reverse order
                applied_versions = [
                    version for version, info in self.migration_history.items()
                    if info.status == MigrationStatus.COMPLETED
                ]
                
                applied_versions.sort(reverse=True)
                
                # Find migrations to rollback
                rollback_versions = []
                for version in applied_versions:
                    if version == target_version:
                        break
                    rollback_versions.append(version)
                
                if not rollback_versions:
                    logger.info(f"No migrations to rollback to {target_version}")
                    return []
                
                rolled_back = []
                
                for version in rollback_versions:
                    if version not in self.migrations:
                        logger.warning(f"Migration {version} not found for rollback")
                        continue
                    
                    migration = self.migrations[version]
                    
                    if not migration.reversible:
                        raise DatabaseMigrationError(f"Migration {version} is not reversible")
                    
                    logger.info(f"Rolling back migration {version}: {migration.name}")
                    
                    try:
                        # Rollback migration
                        await migration.down(self.connector)
                        
                        # Record rollback
                        await self._record_migration_rollback(migration)
                        
                        rolled_back.append(version)
                        logger.info(f"Successfully rolled back migration {version}")
                        
                    except Exception as e:
                        logger.error(f"Rollback of migration {version} failed: {e}")
                        raise DatabaseMigrationError(f"Rollback of migration {version} failed: {e}", version, e)
                
                logger.info(f"Rolled back {len(rolled_back)} migrations successfully")
                return rolled_back
                
            except Exception as e:
                logger.error(f"Rollback process failed: {e}")
                raise
    
    async def _record_migration_start(self, migration: Migration) -> None:
        """Record migration start in the database."""
        checksum = self._calculate_checksum(migration)
        
        query = f"""
        INSERT INTO {self.config.migration_schema}.{self.config.migration_table}
        (version, name, description, checksum, status, applied_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (version) DO UPDATE SET
            status = EXCLUDED.status,
            applied_at = EXCLUDED.applied_at,
            error_message = NULL
        """
        
        parameters = {
            "version": migration.version,
            "name": migration.name,
            "description": migration.description,
            "checksum": checksum,
            "status": MigrationStatus.RUNNING.value,
            "applied_at": datetime.now()
        }
        
        await self.connector.execute_query(query, parameters, fetch_mode="none")
        
        # Update local history
        migration_info = MigrationInfo(
            version=migration.version,
            name=migration.name,
            description=migration.description,
            status=MigrationStatus.RUNNING,
            applied_at=datetime.now(),
            checksum=checksum
        )
        self.migration_history[migration.version] = migration_info
    
    async def _record_migration_completion(self, migration: Migration, execution_time: float) -> None:
        """Record successful migration completion."""
        query = f"""
        UPDATE {self.config.migration_schema}.{self.config.migration_table}
        SET status = $1, execution_time_ms = $2, error_message = NULL
        WHERE version = $3
        """
        
        parameters = {
            "status": MigrationStatus.COMPLETED.value,
            "execution_time_ms": execution_time,
            "version": migration.version
        }
        
        await self.connector.execute_query(query, parameters, fetch_mode="none")
        
        # Update local history
        if migration.version in self.migration_history:
            self.migration_history[migration.version].status = MigrationStatus.COMPLETED
            self.migration_history[migration.version].execution_time_ms = execution_time
    
    async def _record_migration_failure(self, migration: Migration, error_message: str) -> None:
        """Record migration failure."""
        query = f"""
        UPDATE {self.config.migration_schema}.{self.config.migration_table}
        SET status = $1, error_message = $2
        WHERE version = $3
        """
        
        parameters = {
            "status": MigrationStatus.FAILED.value,
            "error_message": error_message,
            "version": migration.version
        }
        
        await self.connector.execute_query(query, parameters, fetch_mode="none")
        
        # Update local history
        if migration.version in self.migration_history:
            self.migration_history[migration.version].status = MigrationStatus.FAILED
            self.migration_history[migration.version].error_message = error_message
    
    async def _record_migration_rollback(self, migration: Migration) -> None:
        """Record migration rollback."""
        query = f"""
        UPDATE {self.config.migration_schema}.{self.config.migration_table}
        SET status = $1, rolled_back_at = $2
        WHERE version = $3
        """
        
        parameters = {
            "status": MigrationStatus.ROLLED_BACK.value,
            "rolled_back_at": datetime.now(),
            "version": migration.version
        }
        
        await self.connector.execute_query(query, parameters, fetch_mode="none")
        
        # Update local history
        if migration.version in self.migration_history:
            self.migration_history[migration.version].status = MigrationStatus.ROLLED_BACK
            self.migration_history[migration.version].rolled_back_at = datetime.now()
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        total_migrations = len(self.migrations)
        applied_migrations = len([
            info for info in self.migration_history.values()
            if info.status == MigrationStatus.COMPLETED
        ])
        pending_migrations = total_migrations - applied_migrations
        
        failed_migrations = [
            info for info in self.migration_history.values()
            if info.status == MigrationStatus.FAILED
        ]
        
        return {
            "total_migrations": total_migrations,
            "applied_migrations": applied_migrations,
            "pending_migrations": pending_migrations,
            "failed_migrations": len(failed_migrations),
            "latest_applied": max(
                (info.applied_at for info in self.migration_history.values() 
                 if info.status == MigrationStatus.COMPLETED and info.applied_at),
                default=None
            ),
            "failed_migration_details": [
                {
                    "version": info.version,
                    "name": info.name,
                    "error": info.error_message
                }
                for info in failed_migrations
            ]
        }
    
    def list_migrations(self) -> List[Dict[str, Any]]:
        """List all migrations with their status."""
        migrations_list = []
        
        for version in sorted(self.migrations.keys()):
            migration = self.migrations[version]
            history = self.migration_history.get(version)
            
            migration_data = {
                "version": version,
                "name": migration.name,
                "description": migration.description,
                "dependencies": migration.dependencies,
                "reversible": migration.reversible,
                "status": history.status.value if history else "pending",
                "applied_at": history.applied_at.isoformat() if history and history.applied_at else None,
                "execution_time_ms": history.execution_time_ms if history else None,
                "error_message": history.error_message if history else None
            }
            
            migrations_list.append(migration_data)
        
        return migrations_list
    
    async def validate_migrations(self) -> Dict[str, Any]:
        """Validate all applied migrations."""
        validation_results = {
            "valid_migrations": [],
            "invalid_migrations": [],
            "checksum_mismatches": []
        }
        
        for version, info in self.migration_history.items():
            if info.status != MigrationStatus.COMPLETED:
                continue
            
            if version not in self.migrations:
                validation_results["invalid_migrations"].append({
                    "version": version,
                    "reason": "Migration not found in current codebase"
                })
                continue
            
            migration = self.migrations[version]
            current_checksum = self._calculate_checksum(migration)
            
            if current_checksum != info.checksum:
                validation_results["checksum_mismatches"].append({
                    "version": version,
                    "stored_checksum": info.checksum,
                    "current_checksum": current_checksum
                })
                continue
            
            try:
                is_valid = await migration.validate(self.connector)
                if is_valid:
                    validation_results["valid_migrations"].append(version)
                else:
                    validation_results["invalid_migrations"].append({
                        "version": version,
                        "reason": "Migration validation failed"
                    })
            except Exception as e:
                validation_results["invalid_migrations"].append({
                    "version": version,
                    "reason": f"Validation error: {e}"
                })
        
        return validation_results


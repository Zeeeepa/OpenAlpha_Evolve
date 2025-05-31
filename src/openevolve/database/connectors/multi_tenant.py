"""
Multi-tenant database management with schema isolation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import hashlib

from ..config import DatabaseConfig
from ..exceptions import DatabaseSchemaError, DatabaseSecurityError
from .postgresql import PostgreSQLConnector

logger = logging.getLogger(__name__)


@dataclass
class TenantInfo:
    """Tenant information."""
    
    tenant_id: str
    schema_name: str
    created_at: datetime
    last_accessed: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MultiTenantManager:
    """
    Multi-tenant database manager providing schema-level isolation
    for different tenants while maintaining performance and security.
    """
    
    def __init__(self, connector: PostgreSQLConnector, config: DatabaseConfig):
        self.connector = connector
        self.config = config
        self.tenants: Dict[str, TenantInfo] = {}
        self._schema_cache: Set[str] = set()
        self._lock = asyncio.Lock()
        
        logger.info("Multi-tenant manager initialized")
    
    async def initialize(self) -> None:
        """Initialize multi-tenant system."""
        if not self.config.enable_multi_tenant:
            logger.info("Multi-tenant mode disabled")
            return
        
        # Create tenant management tables
        await self._create_tenant_tables()
        
        # Load existing tenants
        await self._load_existing_tenants()
        
        logger.info(f"Multi-tenant system initialized with {len(self.tenants)} tenants")
    
    async def _create_tenant_tables(self) -> None:
        """Create tables for tenant management."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS tenant_registry (
            tenant_id VARCHAR(255) PRIMARY KEY,
            schema_name VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        CREATE INDEX IF NOT EXISTS idx_tenant_registry_schema 
        ON tenant_registry(schema_name);
        
        CREATE INDEX IF NOT EXISTS idx_tenant_registry_active 
        ON tenant_registry(is_active) WHERE is_active = TRUE;
        """
        
        try:
            await self.connector.execute_query(create_table_query, fetch_mode="none")
            logger.info("Tenant registry tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create tenant tables: {e}")
            raise DatabaseSchemaError(f"Failed to create tenant tables: {e}")
    
    async def _load_existing_tenants(self) -> None:
        """Load existing tenants from the registry."""
        query = """
        SELECT tenant_id, schema_name, created_at, last_accessed, is_active, metadata
        FROM tenant_registry
        WHERE is_active = TRUE
        """
        
        try:
            tenants = await self.connector.execute_query(query, fetch_mode="all")
            
            for tenant_data in tenants:
                tenant_info = TenantInfo(
                    tenant_id=tenant_data["tenant_id"],
                    schema_name=tenant_data["schema_name"],
                    created_at=tenant_data["created_at"],
                    last_accessed=tenant_data["last_accessed"],
                    is_active=tenant_data["is_active"],
                    metadata=tenant_data["metadata"] or {}
                )
                self.tenants[tenant_info.tenant_id] = tenant_info
                self._schema_cache.add(tenant_info.schema_name)
            
            logger.info(f"Loaded {len(self.tenants)} existing tenants")
            
        except Exception as e:
            logger.error(f"Failed to load existing tenants: {e}")
            # Don't raise here - system can continue without existing tenants
    
    def _generate_schema_name(self, tenant_id: str) -> str:
        """Generate a unique schema name for a tenant."""
        # Create a hash of the tenant_id for uniqueness and security
        hash_object = hashlib.sha256(tenant_id.encode())
        hash_hex = hash_object.hexdigest()[:8]
        
        # Combine prefix with hash
        schema_name = f"{self.config.tenant_schema_prefix}{hash_hex}"
        
        # Ensure it's a valid PostgreSQL identifier
        schema_name = schema_name.lower().replace("-", "_")
        
        return schema_name
    
    async def create_tenant(
        self,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TenantInfo:
        """
        Create a new tenant with isolated schema.
        
        Args:
            tenant_id: Unique identifier for the tenant
            metadata: Optional metadata for the tenant
        
        Returns:
            TenantInfo object for the created tenant
        """
        if not self.config.enable_multi_tenant:
            raise DatabaseSchemaError("Multi-tenant mode is disabled")
        
        async with self._lock:
            # Check if tenant already exists
            if tenant_id in self.tenants:
                logger.warning(f"Tenant {tenant_id} already exists")
                return self.tenants[tenant_id]
            
            # Generate schema name
            schema_name = self._generate_schema_name(tenant_id)
            
            # Ensure schema name is unique
            counter = 1
            original_schema = schema_name
            while schema_name in self._schema_cache:
                schema_name = f"{original_schema}_{counter}"
                counter += 1
            
            try:
                # Create schema
                await self._create_tenant_schema(schema_name)
                
                # Register tenant
                tenant_info = TenantInfo(
                    tenant_id=tenant_id,
                    schema_name=schema_name,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    metadata=metadata or {}
                )
                
                await self._register_tenant(tenant_info)
                
                # Cache tenant info
                self.tenants[tenant_id] = tenant_info
                self._schema_cache.add(schema_name)
                
                logger.info(f"Created tenant {tenant_id} with schema {schema_name}")
                return tenant_info
                
            except Exception as e:
                logger.error(f"Failed to create tenant {tenant_id}: {e}")
                # Cleanup on failure
                try:
                    await self._drop_tenant_schema(schema_name)
                except:
                    pass
                raise DatabaseSchemaError(f"Failed to create tenant {tenant_id}: {e}")
    
    async def _create_tenant_schema(self, schema_name: str) -> None:
        """Create a new schema for a tenant."""
        create_schema_query = f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'
        
        # Grant appropriate permissions
        grant_query = f"""
        GRANT USAGE ON SCHEMA "{schema_name}" TO {self.config.username};
        GRANT CREATE ON SCHEMA "{schema_name}" TO {self.config.username};
        """
        
        await self.connector.execute_query(create_schema_query, fetch_mode="none")
        await self.connector.execute_query(grant_query, fetch_mode="none")
        
        logger.debug(f"Created schema {schema_name}")
    
    async def _register_tenant(self, tenant_info: TenantInfo) -> None:
        """Register tenant in the registry."""
        query = """
        INSERT INTO tenant_registry 
        (tenant_id, schema_name, created_at, last_accessed, is_active, metadata)
        VALUES ($1, $2, $3, $4, $5, $6)
        """
        
        parameters = {
            "tenant_id": tenant_info.tenant_id,
            "schema_name": tenant_info.schema_name,
            "created_at": tenant_info.created_at,
            "last_accessed": tenant_info.last_accessed,
            "is_active": tenant_info.is_active,
            "metadata": tenant_info.metadata
        }
        
        await self.connector.execute_query(query, parameters, fetch_mode="none")
    
    async def get_tenant(self, tenant_id: str) -> Optional[TenantInfo]:
        """Get tenant information."""
        if tenant_id in self.tenants:
            # Update last accessed time
            await self._update_last_accessed(tenant_id)
            return self.tenants[tenant_id]
        
        return None
    
    async def _update_last_accessed(self, tenant_id: str) -> None:
        """Update the last accessed time for a tenant."""
        if tenant_id not in self.tenants:
            return
        
        now = datetime.now()
        self.tenants[tenant_id].last_accessed = now
        
        # Update in database (async, don't wait)
        asyncio.create_task(self._update_tenant_access_time(tenant_id, now))
    
    async def _update_tenant_access_time(self, tenant_id: str, access_time: datetime) -> None:
        """Update tenant access time in database."""
        try:
            query = """
            UPDATE tenant_registry 
            SET last_accessed = $1 
            WHERE tenant_id = $2
            """
            
            parameters = {"access_time": access_time, "tenant_id": tenant_id}
            await self.connector.execute_query(query, parameters, fetch_mode="none")
            
        except Exception as e:
            logger.error(f"Failed to update access time for tenant {tenant_id}: {e}")
    
    async def delete_tenant(self, tenant_id: str, force: bool = False) -> bool:
        """
        Delete a tenant and its schema.
        
        Args:
            tenant_id: Tenant to delete
            force: If True, force deletion even if schema contains data
        
        Returns:
            True if deleted successfully
        """
        if not self.config.enable_multi_tenant:
            raise DatabaseSchemaError("Multi-tenant mode is disabled")
        
        async with self._lock:
            if tenant_id not in self.tenants:
                logger.warning(f"Tenant {tenant_id} not found")
                return False
            
            tenant_info = self.tenants[tenant_id]
            
            try:
                # Check if schema has data (unless force is True)
                if not force:
                    has_data = await self._schema_has_data(tenant_info.schema_name)
                    if has_data:
                        raise DatabaseSchemaError(
                            f"Schema {tenant_info.schema_name} contains data. Use force=True to delete anyway."
                        )
                
                # Drop schema
                await self._drop_tenant_schema(tenant_info.schema_name)
                
                # Remove from registry
                await self._unregister_tenant(tenant_id)
                
                # Remove from cache
                del self.tenants[tenant_id]
                self._schema_cache.discard(tenant_info.schema_name)
                
                logger.info(f"Deleted tenant {tenant_id} and schema {tenant_info.schema_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete tenant {tenant_id}: {e}")
                raise DatabaseSchemaError(f"Failed to delete tenant {tenant_id}: {e}")
    
    async def _schema_has_data(self, schema_name: str) -> bool:
        """Check if a schema contains any tables with data."""
        query = """
        SELECT COUNT(*) as table_count
        FROM information_schema.tables 
        WHERE table_schema = $1 AND table_type = 'BASE TABLE'
        """
        
        parameters = {"schema_name": schema_name}
        result = await self.connector.execute_query(query, parameters, fetch_mode="val")
        
        return result > 0
    
    async def _drop_tenant_schema(self, schema_name: str) -> None:
        """Drop a tenant schema."""
        drop_query = f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'
        await self.connector.execute_query(drop_query, fetch_mode="none")
        
        logger.debug(f"Dropped schema {schema_name}")
    
    async def _unregister_tenant(self, tenant_id: str) -> None:
        """Remove tenant from registry."""
        query = "DELETE FROM tenant_registry WHERE tenant_id = $1"
        parameters = {"tenant_id": tenant_id}
        await self.connector.execute_query(query, parameters, fetch_mode="none")
    
    async def execute_tenant_query(
        self,
        tenant_id: str,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_mode: str = "all"
    ) -> Any:
        """
        Execute a query in a tenant's schema context.
        
        Args:
            tenant_id: Tenant identifier
            query: SQL query
            parameters: Query parameters
            fetch_mode: Fetch mode for results
        
        Returns:
            Query results
        """
        tenant_info = await self.get_tenant(tenant_id)
        if not tenant_info:
            raise DatabaseSchemaError(f"Tenant {tenant_id} not found")
        
        # Set search_path to tenant schema
        set_path_query = f'SET search_path TO "{tenant_info.schema_name}", public'
        
        try:
            # Execute in transaction to ensure search_path is isolated
            queries = [
                (set_path_query, None, "none"),
                (query, parameters, fetch_mode)
            ]
            
            results = await self.connector.execute_transaction(queries)
            return results[1]  # Return result of the actual query
            
        except Exception as e:
            logger.error(f"Failed to execute tenant query for {tenant_id}: {e}")
            raise
    
    async def list_tenants(self, active_only: bool = True) -> List[TenantInfo]:
        """List all tenants."""
        tenants = list(self.tenants.values())
        
        if active_only:
            tenants = [t for t in tenants if t.is_active]
        
        return sorted(tenants, key=lambda t: t.created_at)
    
    async def get_tenant_stats(self) -> Dict[str, Any]:
        """Get tenant statistics."""
        total_tenants = len(self.tenants)
        active_tenants = len([t for t in self.tenants.values() if t.is_active])
        
        # Get schema sizes
        schema_sizes = {}
        for tenant_info in self.tenants.values():
            try:
                size_query = """
                SELECT pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname = $1
                """
                
                parameters = {"schema_name": tenant_info.schema_name}
                result = await self.connector.execute_query(size_query, parameters, fetch_mode="all")
                
                total_size = sum(len(row.get("size", "0")) for row in result)
                schema_sizes[tenant_info.tenant_id] = total_size
                
            except Exception as e:
                logger.warning(f"Failed to get size for tenant {tenant_info.tenant_id}: {e}")
                schema_sizes[tenant_info.tenant_id] = 0
        
        return {
            "total_tenants": total_tenants,
            "active_tenants": active_tenants,
            "inactive_tenants": total_tenants - active_tenants,
            "schema_sizes": schema_sizes,
            "total_schemas": len(self._schema_cache)
        }
    
    async def cleanup_inactive_tenants(self, days_inactive: int = 30) -> int:
        """
        Clean up tenants that haven't been accessed for a specified number of days.
        
        Args:
            days_inactive: Number of days of inactivity before cleanup
        
        Returns:
            Number of tenants cleaned up
        """
        if not self.config.enable_multi_tenant:
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days_inactive)
        cleaned_up = 0
        
        tenants_to_cleanup = [
            tenant_id for tenant_id, tenant_info in self.tenants.items()
            if tenant_info.last_accessed < cutoff_date and tenant_info.is_active
        ]
        
        for tenant_id in tenants_to_cleanup:
            try:
                # Mark as inactive instead of deleting
                await self._deactivate_tenant(tenant_id)
                cleaned_up += 1
                logger.info(f"Deactivated inactive tenant {tenant_id}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup tenant {tenant_id}: {e}")
        
        return cleaned_up
    
    async def _deactivate_tenant(self, tenant_id: str) -> None:
        """Deactivate a tenant without deleting its data."""
        if tenant_id in self.tenants:
            self.tenants[tenant_id].is_active = False
        
        query = "UPDATE tenant_registry SET is_active = FALSE WHERE tenant_id = $1"
        parameters = {"tenant_id": tenant_id}
        await self.connector.execute_query(query, parameters, fetch_mode="none")


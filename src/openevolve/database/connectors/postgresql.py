"""
PostgreSQL connector with async support and connection pooling.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager
import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text

from ..config import DatabaseConfig
from ..exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseTimeoutError,
    DatabasePoolExhaustedError
)

logger = logging.getLogger(__name__)


class PostgreSQLConnector:
    """
    High-performance PostgreSQL connector with async support, connection pooling,
    and comprehensive error handling.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._async_engine = None
        self._sync_engine = None
        self._async_session_factory = None
        self._sync_session_factory = None
        self._connection_pool = None
        self._is_initialized = False
        
        logger.info(f"PostgreSQL connector initialized for {config.host}:{config.port}")
    
    async def initialize(self) -> None:
        """Initialize database connections and pools."""
        if self._is_initialized:
            return
        
        try:
            # Create async engine with connection pooling
            self._async_engine = create_async_engine(
                self.config.async_connection_url,
                pool_size=self.config.min_pool_size,
                max_overflow=self.config.max_pool_size - self.config.min_pool_size,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.enable_query_logging,
                future=True
            )
            
            # Create sync engine for migrations and admin tasks
            self._sync_engine = create_engine(
                self.config.connection_url,
                pool_size=self.config.min_pool_size,
                max_overflow=self.config.max_pool_size - self.config.min_pool_size,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.enable_query_logging
            )
            
            # Create session factories
            self._async_session_factory = async_sessionmaker(
                self._async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self._sync_session_factory = sessionmaker(
                self._sync_engine,
                expire_on_commit=False
            )
            
            # Create asyncpg connection pool for raw queries
            self._connection_pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                command_timeout=self.config.query_timeout,
                server_settings={
                    'application_name': 'openevolve_connector',
                    'statement_timeout': str(int(self.config.statement_timeout * 1000))
                }
            )
            
            # Test connections
            await self._test_connections()
            
            self._is_initialized = True
            logger.info("PostgreSQL connector successfully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connector: {e}")
            raise DatabaseConnectionError(
                f"Failed to initialize database connection: {e}",
                self.config.connection_url,
                e
            )
    
    async def _test_connections(self) -> None:
        """Test database connections."""
        try:
            # Test asyncpg pool
            async with self._connection_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            # Test SQLAlchemy async engine
            async with self._async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("Database connection tests passed")
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise DatabaseConnectionError(
                f"Database connection test failed: {e}",
                self.config.connection_url,
                e
            )
    
    async def close(self) -> None:
        """Close all database connections."""
        try:
            if self._connection_pool:
                await self._connection_pool.close()
            
            if self._async_engine:
                await self._async_engine.dispose()
            
            if self._sync_engine:
                self._sync_engine.dispose()
            
            self._is_initialized = False
            logger.info("PostgreSQL connector closed")
            
        except Exception as e:
            logger.error(f"Error closing PostgreSQL connector: {e}")
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async SQLAlchemy session."""
        if not self._is_initialized:
            await self.initialize()
        
        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseQueryError(f"Database session error: {e}", original_error=e)
        finally:
            await session.close()
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a raw asyncpg connection from the pool."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            async with self._connection_pool.acquire() as conn:
                yield conn
        except asyncio.TimeoutError as e:
            logger.error("Connection pool timeout")
            raise DatabaseTimeoutError(
                "Connection pool timeout",
                self.config.pool_timeout,
                "acquire_connection"
            )
        except Exception as e:
            logger.error(f"Error acquiring database connection: {e}")
            raise DatabaseConnectionError(
                f"Error acquiring database connection: {e}",
                self.config.connection_url,
                e
            )
    
    def get_sync_session(self):
        """Get a synchronous SQLAlchemy session (for migrations)."""
        if not self._sync_engine:
            raise DatabaseConnectionError("Sync engine not initialized")
        
        return self._sync_session_factory()
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_mode: str = "all"
    ) -> Union[List[Dict[str, Any]], Dict[str, Any], Any]:
        """
        Execute a raw SQL query with parameters.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            fetch_mode: "all", "one", "val", or "none"
        
        Returns:
            Query results based on fetch_mode
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            async with self.get_connection() as conn:
                if parameters:
                    if fetch_mode == "all":
                        result = await conn.fetch(query, *parameters.values())
                        return [dict(row) for row in result]
                    elif fetch_mode == "one":
                        result = await conn.fetchrow(query, *parameters.values())
                        return dict(result) if result else None
                    elif fetch_mode == "val":
                        return await conn.fetchval(query, *parameters.values())
                    elif fetch_mode == "none":
                        await conn.execute(query, *parameters.values())
                        return None
                else:
                    if fetch_mode == "all":
                        result = await conn.fetch(query)
                        return [dict(row) for row in result]
                    elif fetch_mode == "one":
                        result = await conn.fetchrow(query)
                        return dict(result) if result else None
                    elif fetch_mode == "val":
                        return await conn.fetchval(query)
                    elif fetch_mode == "none":
                        await conn.execute(query)
                        return None
                        
        except asyncio.TimeoutError as e:
            logger.error(f"Query timeout: {query[:100]}...")
            raise DatabaseTimeoutError(
                f"Query timeout: {query[:100]}...",
                self.config.query_timeout,
                "execute_query"
            )
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseQueryError(
                f"Query execution failed: {e}",
                query,
                parameters,
                e
            )
    
    async def execute_transaction(
        self,
        queries: List[tuple]
    ) -> List[Any]:
        """
        Execute multiple queries in a transaction.
        
        Args:
            queries: List of (query, parameters, fetch_mode) tuples
        
        Returns:
            List of query results
        """
        if not self._is_initialized:
            await self.initialize()
        
        results = []
        
        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    for query, parameters, fetch_mode in queries:
                        if parameters:
                            if fetch_mode == "all":
                                result = await conn.fetch(query, *parameters.values())
                                results.append([dict(row) for row in result])
                            elif fetch_mode == "one":
                                result = await conn.fetchrow(query, *parameters.values())
                                results.append(dict(result) if result else None)
                            elif fetch_mode == "val":
                                result = await conn.fetchval(query, *parameters.values())
                                results.append(result)
                            elif fetch_mode == "none":
                                await conn.execute(query, *parameters.values())
                                results.append(None)
                        else:
                            if fetch_mode == "all":
                                result = await conn.fetch(query)
                                results.append([dict(row) for row in result])
                            elif fetch_mode == "one":
                                result = await conn.fetchrow(query)
                                results.append(dict(result) if result else None)
                            elif fetch_mode == "val":
                                result = await conn.fetchval(query)
                                results.append(result)
                            elif fetch_mode == "none":
                                await conn.execute(query)
                                results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"Transaction execution failed: {e}")
            raise DatabaseQueryError(
                f"Transaction execution failed: {e}",
                str(queries),
                None,
                e
            )
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status information."""
        if not self._connection_pool:
            return {"status": "not_initialized"}
        
        return {
            "size": self._connection_pool.get_size(),
            "min_size": self._connection_pool.get_min_size(),
            "max_size": self._connection_pool.get_max_size(),
            "idle_connections": self._connection_pool.get_idle_size(),
            "status": "healthy" if self._connection_pool.get_size() > 0 else "unhealthy"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Test basic connectivity
            async with self.get_connection() as conn:
                await conn.fetchval("SELECT 1")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            pool_status = await self.get_pool_status()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "pool_status": pool_status,
                "database": self.config.database,
                "host": self.config.host,
                "port": self.config.port
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "database": self.config.database,
                "host": self.config.host,
                "port": self.config.port
            }


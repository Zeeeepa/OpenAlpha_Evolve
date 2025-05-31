"""
Redis-based caching layer for database operations.
"""

import asyncio
import logging
import json
import pickle
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import hashlib

from ..config import RedisConfig
from ..exceptions import DatabaseCacheError

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based caching layer with advanced features like
    cache invalidation, compression, and analytics.
    """
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._redis_client = None
        self._is_connected = False
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        
        logger.info(f"Redis cache initialized for {config.host}:{config.port}")
    
    async def connect(self) -> None:
        """Connect to Redis server."""
        try:
            import redis.asyncio as redis
            
            self._redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.database,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_connect_timeout=self.config.connection_timeout,
                socket_timeout=self.config.socket_timeout,
                decode_responses=False  # We'll handle encoding ourselves
            )
            
            # Test connection
            await self._redis_client.ping()
            self._is_connected = True
            
            logger.info("Connected to Redis successfully")
            
        except ImportError:
            logger.warning("Redis library not available, caching disabled")
            self._redis_client = None
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis_client = None
            raise DatabaseCacheError(f"Failed to connect to Redis: {e}", original_error=e)
    
    async def disconnect(self) -> None:
        """Disconnect from Redis server."""
        if self._redis_client:
            try:
                await self._redis_client.close()
                self._is_connected = False
                logger.info("Disconnected from Redis")
            except Exception as e:
                logger.error(f"Error disconnecting from Redis: {e}")
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate a namespaced cache key."""
        return f"openevolve:db:{namespace}:{key}"
    
    def _serialize_value(self, value: Any, use_json: bool = True) -> bytes:
        """Serialize a value for storage."""
        try:
            if use_json:
                # Try JSON first (more readable, cross-language compatible)
                return json.dumps(value, default=str).encode('utf-8')
            else:
                # Fall back to pickle for complex objects
                return pickle.dumps(value)
        except (TypeError, ValueError):
            # If JSON fails, use pickle
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes, use_json: bool = True) -> Any:
        """Deserialize a value from storage."""
        try:
            if use_json:
                return json.loads(data.decode('utf-8'))
            else:
                return pickle.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If JSON fails, try pickle
            return pickle.loads(data)
    
    async def get(
        self,
        namespace: str,
        key: str,
        default: Any = None,
        use_json: bool = True
    ) -> Any:
        """
        Get a value from cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            default: Default value if not found
            use_json: Whether to use JSON serialization
        
        Returns:
            Cached value or default
        """
        if not self._is_connected:
            self._cache_stats["misses"] += 1
            return default
        
        cache_key = self._generate_key(namespace, key)
        
        try:
            data = await self._redis_client.get(cache_key)
            
            if data is None:
                self._cache_stats["misses"] += 1
                return default
            
            value = self._deserialize_value(data, use_json)
            self._cache_stats["hits"] += 1
            
            logger.debug(f"Cache hit: {cache_key}")
            return value
            
        except Exception as e:
            self._cache_stats["errors"] += 1
            logger.error(f"Cache get error for {cache_key}: {e}")
            return default
    
    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        use_json: bool = True
    ) -> bool:
        """
        Set a value in cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            use_json: Whether to use JSON serialization
        
        Returns:
            True if successful
        """
        if not self._is_connected:
            return False
        
        cache_key = self._generate_key(namespace, key)
        
        try:
            data = self._serialize_value(value, use_json)
            
            if ttl is None:
                ttl = self.config.default_ttl
            
            await self._redis_client.setex(cache_key, ttl, data)
            self._cache_stats["sets"] += 1
            
            logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            self._cache_stats["errors"] += 1
            logger.error(f"Cache set error for {cache_key}: {e}")
            raise DatabaseCacheError(f"Cache set error: {e}", cache_key, e)
    
    async def delete(self, namespace: str, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
        
        Returns:
            True if deleted
        """
        if not self._is_connected:
            return False
        
        cache_key = self._generate_key(namespace, key)
        
        try:
            result = await self._redis_client.delete(cache_key)
            self._cache_stats["deletes"] += 1
            
            logger.debug(f"Cache delete: {cache_key}")
            return result > 0
            
        except Exception as e:
            self._cache_stats["errors"] += 1
            logger.error(f"Cache delete error for {cache_key}: {e}")
            return False
    
    async def exists(self, namespace: str, key: str) -> bool:
        """
        Check if a key exists in cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
        
        Returns:
            True if exists
        """
        if not self._is_connected:
            return False
        
        cache_key = self._generate_key(namespace, key)
        
        try:
            result = await self._redis_client.exists(cache_key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache exists error for {cache_key}: {e}")
            return False
    
    async def expire(self, namespace: str, key: str, ttl: int) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            ttl: Time to live in seconds
        
        Returns:
            True if successful
        """
        if not self._is_connected:
            return False
        
        cache_key = self._generate_key(namespace, key)
        
        try:
            result = await self._redis_client.expire(cache_key, ttl)
            return result
        except Exception as e:
            logger.error(f"Cache expire error for {cache_key}: {e}")
            return False
    
    async def get_ttl(self, namespace: str, key: str) -> int:
        """
        Get remaining TTL for a key.
        
        Args:
            namespace: Cache namespace
            key: Cache key
        
        Returns:
            TTL in seconds (-1 if no expiry, -2 if key doesn't exist)
        """
        if not self._is_connected:
            return -2
        
        cache_key = self._generate_key(namespace, key)
        
        try:
            return await self._redis_client.ttl(cache_key)
        except Exception as e:
            logger.error(f"Cache TTL error for {cache_key}: {e}")
            return -2
    
    async def clear_namespace(self, namespace: str) -> int:
        """
        Clear all keys in a namespace.
        
        Args:
            namespace: Cache namespace to clear
        
        Returns:
            Number of keys deleted
        """
        if not self._is_connected:
            return 0
        
        pattern = self._generate_key(namespace, "*")
        
        try:
            keys = await self._redis_client.keys(pattern)
            if keys:
                deleted = await self._redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} keys from namespace '{namespace}'")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache clear namespace error for {namespace}: {e}")
            return 0
    
    async def get_multi(
        self,
        namespace: str,
        keys: List[str],
        use_json: bool = True
    ) -> Dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            namespace: Cache namespace
            keys: List of cache keys
            use_json: Whether to use JSON serialization
        
        Returns:
            Dictionary of key-value pairs
        """
        if not self._is_connected or not keys:
            return {}
        
        cache_keys = [self._generate_key(namespace, key) for key in keys]
        
        try:
            values = await self._redis_client.mget(cache_keys)
            result = {}
            
            for i, (original_key, value) in enumerate(zip(keys, values)):
                if value is not None:
                    try:
                        result[original_key] = self._deserialize_value(value, use_json)
                        self._cache_stats["hits"] += 1
                    except Exception as e:
                        logger.error(f"Deserialization error for {original_key}: {e}")
                        self._cache_stats["errors"] += 1
                else:
                    self._cache_stats["misses"] += 1
            
            return result
            
        except Exception as e:
            self._cache_stats["errors"] += 1
            logger.error(f"Cache mget error: {e}")
            return {}
    
    async def set_multi(
        self,
        namespace: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        use_json: bool = True
    ) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            namespace: Cache namespace
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
            use_json: Whether to use JSON serialization
        
        Returns:
            True if successful
        """
        if not self._is_connected or not data:
            return False
        
        try:
            # Use pipeline for efficiency
            pipe = self._redis_client.pipeline()
            
            for key, value in data.items():
                cache_key = self._generate_key(namespace, key)
                serialized_value = self._serialize_value(value, use_json)
                
                if ttl is None:
                    ttl = self.config.default_ttl
                
                pipe.setex(cache_key, ttl, serialized_value)
            
            await pipe.execute()
            self._cache_stats["sets"] += len(data)
            
            logger.debug(f"Cache mset: {len(data)} keys in namespace '{namespace}'")
            return True
            
        except Exception as e:
            self._cache_stats["errors"] += 1
            logger.error(f"Cache mset error: {e}")
            return False
    
    async def increment(self, namespace: str, key: str, amount: int = 1) -> int:
        """
        Increment a numeric value in cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            amount: Amount to increment
        
        Returns:
            New value after increment
        """
        if not self._is_connected:
            return 0
        
        cache_key = self._generate_key(namespace, key)
        
        try:
            result = await self._redis_client.incrby(cache_key, amount)
            return result
        except Exception as e:
            logger.error(f"Cache increment error for {cache_key}: {e}")
            return 0
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics."""
        if not self._is_connected:
            return {"status": "disconnected", "stats": self._cache_stats}
        
        try:
            info = await self._redis_client.info()
            
            return {
                "status": "connected",
                "redis_info": {
                    "version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses")
                },
                "stats": self._cache_stats,
                "config": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "database": self.config.database,
                    "default_ttl": self.config.default_ttl
                }
            }
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {"status": "error", "error": str(e), "stats": self._cache_stats}
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        if total_requests == 0:
            return 0.0
        return (self._cache_stats["hits"] / total_requests) * 100
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        logger.info("Cache statistics reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        if not self._is_connected:
            return {
                "status": "unhealthy",
                "message": "Not connected to Redis"
            }
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Test basic operations
            test_key = "health_check_test"
            test_value = {"timestamp": datetime.now().isoformat()}
            
            await self.set("health", test_key, test_value, ttl=60)
            retrieved_value = await self.get("health", test_key)
            await self.delete("health", test_key)
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if retrieved_value == test_value:
                return {
                    "status": "healthy",
                    "response_time_ms": round(response_time, 2),
                    "hit_rate": round(self.get_hit_rate(), 2)
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Cache operations failed"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}"
            }


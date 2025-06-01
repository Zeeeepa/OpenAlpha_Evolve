"""
Cache Manager for the Context Analysis Engine.

Provides caching capabilities to improve performance by storing
analysis results and avoiding redundant computations.
"""

import json
import time
import hashlib
import logging
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod

from ..core.interfaces import AnalysisConfig, AnalysisResult


logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self):
        """Initialize memory cache."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.debug("Memory cache backend initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        try:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry['expires_at'] > time.time():
                    return entry['value']
                else:
                    # Remove expired entry
                    del self._cache[key]
            
            return None
            
        except Exception as e:
            logger.error(f"Memory cache get failed for key {key}", exc_info=e)
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in memory cache."""
        try:
            expires_at = time.time() + ttl
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            return True
            
        except Exception as e:
            logger.error(f"Memory cache set failed for key {key}", exc_info=e)
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        try:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Memory cache delete failed for key {key}", exc_info=e)
            return False
    
    async def clear(self) -> bool:
        """Clear all memory cache entries."""
        try:
            self._cache.clear()
            return True
            
        except Exception as e:
            logger.error(f"Memory cache clear failed", exc_info=e)
            return False


class FileCacheBackend(CacheBackend):
    """File-based cache backend."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize file cache."""
        import os
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f"File cache backend initialized at {cache_dir}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        try:
            import os
            
            file_path = self._get_cache_file_path(key)
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if expired
                if data['expires_at'] > time.time():
                    return data['value']
                else:
                    # Remove expired file
                    os.remove(file_path)
            
            return None
            
        except Exception as e:
            logger.error(f"File cache get failed for key {key}", exc_info=e)
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in file cache."""
        try:
            file_path = self._get_cache_file_path(key)
            
            data = {
                'value': value,
                'expires_at': time.time() + ttl,
                'created_at': time.time()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=self._json_serializer)
            
            return True
            
        except Exception as e:
            logger.error(f"File cache set failed for key {key}", exc_info=e)
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from file cache."""
        try:
            import os
            
            file_path = self._get_cache_file_path(key)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"File cache delete failed for key {key}", exc_info=e)
            return False
    
    async def clear(self) -> bool:
        """Clear all file cache entries."""
        try:
            import os
            import glob
            
            cache_files = glob.glob(os.path.join(self.cache_dir, "*.json"))
            
            for file_path in cache_files:
                os.remove(file_path)
            
            return True
            
        except Exception as e:
            logger.error(f"File cache clear failed", exc_info=e)
            return False
    
    def _get_cache_file_path(self, key: str) -> str:
        """Get file path for cache key."""
        import os
        
        # Create safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.json")
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex objects."""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '_asdict'):  # namedtuple
            return obj._asdict()
        else:
            return str(obj)


class RedisCacheBackend(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis cache."""
        self.redis_url = redis_url
        self._redis = None
        logger.debug(f"Redis cache backend initialized with URL {redis_url}")
    
    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url)
            except ImportError:
                logger.error("Redis library not available")
                raise
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            redis_client = await self._get_redis()
            data = await redis_client.get(key)
            
            if data:
                return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis cache get failed for key {key}", exc_info=e)
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in Redis cache."""
        try:
            redis_client = await self._get_redis()
            
            serialized_value = json.dumps(value, default=self._json_serializer)
            await redis_client.setex(key, ttl, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache set failed for key {key}", exc_info=e)
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis cache delete failed for key {key}", exc_info=e)
            return False
    
    async def clear(self) -> bool:
        """Clear all Redis cache entries."""
        try:
            redis_client = await self._get_redis()
            await redis_client.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"Redis cache clear failed", exc_info=e)
            return False
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex objects."""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '_asdict'):  # namedtuple
            return obj._asdict()
        else:
            return str(obj)


class CacheManager:
    """
    Cache manager for the Context Analysis Engine.
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize cache manager."""
        self.config = config
        self.backend = self._create_backend()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        logger.info(f"Cache manager initialized with {config.cache_backend} backend")
    
    def _create_backend(self) -> CacheBackend:
        """Create cache backend based on configuration."""
        backend_type = self.config.cache_backend.lower()
        
        if backend_type == "memory":
            return MemoryCacheBackend()
        elif backend_type == "file":
            return FileCacheBackend()
        elif backend_type == "redis":
            redis_url = getattr(self.config, 'redis_url', 'redis://localhost:6379')
            return RedisCacheBackend(redis_url)
        else:
            logger.warning(f"Unknown cache backend {backend_type}, using memory")
            return MemoryCacheBackend()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.config.enable_caching:
            return None
        
        try:
            value = await self.backend.get(key)
            
            if value is not None:
                self.stats['hits'] += 1
                logger.debug(f"Cache hit for key: {key}")
            else:
                self.stats['misses'] += 1
                logger.debug(f"Cache miss for key: {key}")
            
            return value
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Cache get failed for key {key}", exc_info=e)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses config default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config.enable_caching:
            return False
        
        try:
            cache_ttl = ttl or self.config.cache_ttl
            success = await self.backend.set(key, value, cache_ttl)
            
            if success:
                self.stats['sets'] += 1
                logger.debug(f"Cache set for key: {key}")
            
            return success
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Cache set failed for key {key}", exc_info=e)
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config.enable_caching:
            return False
        
        try:
            success = await self.backend.delete(key)
            
            if success:
                self.stats['deletes'] += 1
                logger.debug(f"Cache delete for key: {key}")
            
            return success
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Cache delete failed for key {key}", exc_info=e)
            return False
    
    async def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.config.enable_caching:
            return False
        
        try:
            success = await self.backend.clear()
            
            if success:
                logger.info("Cache cleared")
                # Reset stats
                self.stats = {key: 0 for key in self.stats}
            
            return success
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Cache clear failed", exc_info=e)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'backend_type': self.config.cache_backend
        }
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries (for backends that don't auto-expire).
        
        Returns:
            Number of entries cleaned up
        """
        # This would be implemented for backends that don't auto-expire
        # For now, we'll just return 0 as most backends handle this automatically
        return 0


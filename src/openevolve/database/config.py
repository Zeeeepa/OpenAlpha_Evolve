"""
Database configuration management for the OpenEvolve database connector system.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "openevolve"
    username: str = "openevolve_user"
    password: str = ""
    
    # Connection pool settings
    min_pool_size: int = 5
    max_pool_size: int = 20
    pool_timeout: float = 30.0
    pool_recycle: int = 3600  # 1 hour
    
    # Query settings
    query_timeout: float = 30.0
    statement_timeout: float = 60.0
    
    # SSL settings
    ssl_mode: str = "prefer"  # disable, allow, prefer, require, verify-ca, verify-full
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    
    # Multi-tenant settings
    enable_multi_tenant: bool = True
    default_schema: str = "public"
    tenant_schema_prefix: str = "tenant_"
    
    # Monitoring settings
    enable_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    health_check_interval: int = 30  # seconds
    
    # Security settings
    enable_audit_logging: bool = True
    enable_query_logging: bool = False
    max_query_log_length: int = 1000
    
    # Cache settings
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    cache_max_size: int = 1000
    
    # Migration settings
    migration_table: str = "schema_migrations"
    migration_schema: str = "public"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.min_pool_size < 1:
            raise ValueError("min_pool_size must be at least 1")
        
        if self.max_pool_size < self.min_pool_size:
            raise ValueError("max_pool_size must be >= min_pool_size")
        
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        
        if self.query_timeout <= 0:
            raise ValueError("query_timeout must be positive")
        
        if self.ssl_mode not in ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]:
            raise ValueError(f"Invalid ssl_mode: {self.ssl_mode}")
    
    @property
    def connection_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        url = f"postgresql://{self.username}"
        if self.password:
            url += f":{self.password}"
        url += f"@{self.host}:{self.port}/{self.database}"
        
        # Add SSL parameters
        params = []
        if self.ssl_mode != "prefer":
            params.append(f"sslmode={self.ssl_mode}")
        if self.ssl_cert_path:
            params.append(f"sslcert={self.ssl_cert_path}")
        if self.ssl_key_path:
            params.append(f"sslkey={self.ssl_key_path}")
        if self.ssl_ca_path:
            params.append(f"sslrootcert={self.ssl_ca_path}")
        
        if params:
            url += "?" + "&".join(params)
        
        return url
    
    @property
    def async_connection_url(self) -> str:
        """Generate asyncpg connection URL."""
        return self.connection_url.replace("postgresql://", "postgresql+asyncpg://")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": "***" if self.password else "",
            "min_pool_size": self.min_pool_size,
            "max_pool_size": self.max_pool_size,
            "pool_timeout": self.pool_timeout,
            "query_timeout": self.query_timeout,
            "ssl_mode": self.ssl_mode,
            "enable_multi_tenant": self.enable_multi_tenant,
            "enable_monitoring": self.enable_monitoring,
            "enable_audit_logging": self.enable_audit_logging,
            "enable_caching": self.enable_caching
        }


@dataclass
class RedisConfig:
    """Redis cache configuration."""
    
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    
    # Connection pool settings
    max_connections: int = 20
    connection_timeout: float = 5.0
    socket_timeout: float = 5.0
    
    # Cache settings
    default_ttl: int = 300  # 5 minutes
    max_memory_policy: str = "allkeys-lru"
    
    # SSL settings
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    
    @property
    def connection_url(self) -> str:
        """Generate Redis connection URL."""
        scheme = "rediss" if self.ssl_enabled else "redis"
        url = f"{scheme}://"
        
        if self.password:
            url += f":{self.password}@"
        
        url += f"{self.host}:{self.port}/{self.database}"
        return url


def load_database_config() -> DatabaseConfig:
    """Load database configuration from environment variables."""
    
    # Parse DATABASE_URL if provided
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        parsed = urlparse(database_url)
        config = DatabaseConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") if parsed.path else "openevolve",
            username=parsed.username or "openevolve_user",
            password=parsed.password or ""
        )
    else:
        config = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "openevolve"),
            username=os.getenv("DB_USER", "openevolve_user"),
            password=os.getenv("DB_PASSWORD", "")
        )
    
    # Override with specific environment variables
    config.min_pool_size = int(os.getenv("DB_MIN_POOL_SIZE", str(config.min_pool_size)))
    config.max_pool_size = int(os.getenv("DB_MAX_POOL_SIZE", str(config.max_pool_size)))
    config.pool_timeout = float(os.getenv("DB_POOL_TIMEOUT", str(config.pool_timeout)))
    config.query_timeout = float(os.getenv("DB_QUERY_TIMEOUT", str(config.query_timeout)))
    
    config.ssl_mode = os.getenv("DB_SSL_MODE", config.ssl_mode)
    config.ssl_cert_path = os.getenv("DB_SSL_CERT_PATH")
    config.ssl_key_path = os.getenv("DB_SSL_KEY_PATH")
    config.ssl_ca_path = os.getenv("DB_SSL_CA_PATH")
    
    config.enable_multi_tenant = os.getenv("DB_ENABLE_MULTI_TENANT", "true").lower() == "true"
    config.enable_monitoring = os.getenv("DB_ENABLE_MONITORING", "true").lower() == "true"
    config.enable_audit_logging = os.getenv("DB_ENABLE_AUDIT_LOGGING", "true").lower() == "true"
    config.enable_caching = os.getenv("DB_ENABLE_CACHING", "true").lower() == "true"
    
    logger.info(f"Loaded database configuration: {config.to_dict()}")
    return config


def load_redis_config() -> RedisConfig:
    """Load Redis configuration from environment variables."""
    
    # Parse REDIS_URL if provided
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        parsed = urlparse(redis_url)
        config = RedisConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            database=int(parsed.path.lstrip("/")) if parsed.path else 0,
            password=parsed.password
        )
    else:
        config = RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            database=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD")
        )
    
    # Override with specific environment variables
    config.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", str(config.max_connections)))
    config.connection_timeout = float(os.getenv("REDIS_CONNECTION_TIMEOUT", str(config.connection_timeout)))
    config.default_ttl = int(os.getenv("REDIS_DEFAULT_TTL", str(config.default_ttl)))
    
    config.ssl_enabled = os.getenv("REDIS_SSL_ENABLED", "false").lower() == "true"
    config.ssl_cert_path = os.getenv("REDIS_SSL_CERT_PATH")
    config.ssl_key_path = os.getenv("REDIS_SSL_KEY_PATH")
    config.ssl_ca_path = os.getenv("REDIS_SSL_CA_PATH")
    
    logger.info(f"Loaded Redis configuration for {config.host}:{config.port}")
    return config


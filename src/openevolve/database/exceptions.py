"""
Database-specific exceptions for the OpenEvolve database connector system.
"""

class DatabaseError(Exception):
    """Base exception for all database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, message: str, connection_string: str = None, original_error: Exception = None):
        super().__init__(message)
        self.connection_string = connection_string
        self.original_error = original_error


class DatabaseQueryError(DatabaseError):
    """Raised when database query execution fails."""
    
    def __init__(self, message: str, query: str = None, parameters: dict = None, original_error: Exception = None):
        super().__init__(message)
        self.query = query
        self.parameters = parameters
        self.original_error = original_error


class DatabaseMigrationError(DatabaseError):
    """Raised when database migration fails."""
    
    def __init__(self, message: str, migration_version: str = None, original_error: Exception = None):
        super().__init__(message)
        self.migration_version = migration_version
        self.original_error = original_error


class DatabaseSecurityError(DatabaseError):
    """Raised when database security validation fails."""
    
    def __init__(self, message: str, user_id: str = None, operation: str = None, original_error: Exception = None):
        super().__init__(message)
        self.user_id = user_id
        self.operation = operation
        self.original_error = original_error


class DatabaseCacheError(DatabaseError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, cache_key: str = None, original_error: Exception = None):
        super().__init__(message)
        self.cache_key = cache_key
        self.original_error = original_error


class DatabasePoolExhaustedError(DatabaseConnectionError):
    """Raised when connection pool is exhausted."""
    
    def __init__(self, message: str = "Connection pool exhausted", pool_size: int = None):
        super().__init__(message)
        self.pool_size = pool_size


class DatabaseTimeoutError(DatabaseError):
    """Raised when database operation times out."""
    
    def __init__(self, message: str, timeout_seconds: float = None, operation: str = None):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class DatabaseSchemaError(DatabaseError):
    """Raised when schema-related operations fail."""
    
    def __init__(self, message: str, schema_name: str = None, tenant_id: str = None):
        super().__init__(message)
        self.schema_name = schema_name
        self.tenant_id = tenant_id


class DatabaseValidationError(DatabaseError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field_name: str = None, field_value: any = None):
        super().__init__(message)
        self.field_name = field_name
        self.field_value = field_value


"""
Advanced query builder with SQL injection prevention and optimization.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class JoinType(Enum):
    """SQL join types."""
    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    RIGHT = "RIGHT JOIN"
    FULL = "FULL OUTER JOIN"
    CROSS = "CROSS JOIN"


class OrderDirection(Enum):
    """SQL order directions."""
    ASC = "ASC"
    DESC = "DESC"


@dataclass
class QueryCondition:
    """Represents a WHERE condition."""
    
    column: str
    operator: str
    value: Any
    logical_operator: str = "AND"  # AND, OR
    
    def __post_init__(self):
        """Validate condition."""
        valid_operators = ["=", "!=", "<>", "<", ">", "<=", ">=", "LIKE", "ILIKE", "IN", "NOT IN", "IS NULL", "IS NOT NULL"]
        if self.operator.upper() not in valid_operators:
            raise ValueError(f"Invalid operator: {self.operator}")


@dataclass
class JoinClause:
    """Represents a JOIN clause."""
    
    table: str
    join_type: JoinType
    on_condition: str
    alias: Optional[str] = None


@dataclass
class OrderClause:
    """Represents an ORDER BY clause."""
    
    column: str
    direction: OrderDirection = OrderDirection.ASC


class QueryBuilder:
    """
    Advanced SQL query builder with security features and optimization.
    Prevents SQL injection and provides a fluent interface for building queries.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'QueryBuilder':
        """Reset the query builder to initial state."""
        self._select_columns: List[str] = []
        self._from_table: Optional[str] = None
        self._table_alias: Optional[str] = None
        self._joins: List[JoinClause] = []
        self._conditions: List[QueryCondition] = []
        self._group_by: List[str] = []
        self._having_conditions: List[QueryCondition] = []
        self._order_by: List[OrderClause] = []
        self._limit_value: Optional[int] = None
        self._offset_value: Optional[int] = None
        self._parameters: Dict[str, Any] = {}
        self._parameter_counter: int = 0
        
        return self
    
    def select(self, *columns: str) -> 'QueryBuilder':
        """
        Add columns to SELECT clause.
        
        Args:
            columns: Column names or expressions
        
        Returns:
            QueryBuilder instance for chaining
        """
        for column in columns:
            if self._is_safe_identifier(column):
                self._select_columns.append(column)
            else:
                raise ValueError(f"Invalid column identifier: {column}")
        
        return self
    
    def from_table(self, table: str, alias: Optional[str] = None) -> 'QueryBuilder':
        """
        Set the FROM table.
        
        Args:
            table: Table name
            alias: Optional table alias
        
        Returns:
            QueryBuilder instance for chaining
        """
        if not self._is_safe_identifier(table):
            raise ValueError(f"Invalid table identifier: {table}")
        
        if alias and not self._is_safe_identifier(alias):
            raise ValueError(f"Invalid alias identifier: {alias}")
        
        self._from_table = table
        self._table_alias = alias
        
        return self
    
    def join(
        self,
        table: str,
        on_condition: str,
        join_type: JoinType = JoinType.INNER,
        alias: Optional[str] = None
    ) -> 'QueryBuilder':
        """
        Add a JOIN clause.
        
        Args:
            table: Table to join
            on_condition: JOIN condition
            join_type: Type of join
            alias: Optional table alias
        
        Returns:
            QueryBuilder instance for chaining
        """
        if not self._is_safe_identifier(table):
            raise ValueError(f"Invalid table identifier: {table}")
        
        if alias and not self._is_safe_identifier(alias):
            raise ValueError(f"Invalid alias identifier: {alias}")
        
        join_clause = JoinClause(
            table=table,
            join_type=join_type,
            on_condition=on_condition,
            alias=alias
        )
        
        self._joins.append(join_clause)
        return self
    
    def where(
        self,
        column: str,
        operator: str,
        value: Any,
        logical_operator: str = "AND"
    ) -> 'QueryBuilder':
        """
        Add a WHERE condition.
        
        Args:
            column: Column name
            operator: Comparison operator
            value: Value to compare
            logical_operator: AND/OR
        
        Returns:
            QueryBuilder instance for chaining
        """
        if not self._is_safe_identifier(column):
            raise ValueError(f"Invalid column identifier: {column}")
        
        condition = QueryCondition(
            column=column,
            operator=operator,
            value=value,
            logical_operator=logical_operator.upper()
        )
        
        self._conditions.append(condition)
        return self
    
    def where_in(self, column: str, values: List[Any], logical_operator: str = "AND") -> 'QueryBuilder':
        """
        Add a WHERE IN condition.
        
        Args:
            column: Column name
            values: List of values
            logical_operator: AND/OR
        
        Returns:
            QueryBuilder instance for chaining
        """
        return self.where(column, "IN", values, logical_operator)
    
    def where_null(self, column: str, logical_operator: str = "AND") -> 'QueryBuilder':
        """
        Add a WHERE IS NULL condition.
        
        Args:
            column: Column name
            logical_operator: AND/OR
        
        Returns:
            QueryBuilder instance for chaining
        """
        return self.where(column, "IS NULL", None, logical_operator)
    
    def where_not_null(self, column: str, logical_operator: str = "AND") -> 'QueryBuilder':
        """
        Add a WHERE IS NOT NULL condition.
        
        Args:
            column: Column name
            logical_operator: AND/OR
        
        Returns:
            QueryBuilder instance for chaining
        """
        return self.where(column, "IS NOT NULL", None, logical_operator)
    
    def group_by(self, *columns: str) -> 'QueryBuilder':
        """
        Add columns to GROUP BY clause.
        
        Args:
            columns: Column names
        
        Returns:
            QueryBuilder instance for chaining
        """
        for column in columns:
            if self._is_safe_identifier(column):
                self._group_by.append(column)
            else:
                raise ValueError(f"Invalid column identifier: {column}")
        
        return self
    
    def having(
        self,
        column: str,
        operator: str,
        value: Any,
        logical_operator: str = "AND"
    ) -> 'QueryBuilder':
        """
        Add a HAVING condition.
        
        Args:
            column: Column name or aggregate function
            operator: Comparison operator
            value: Value to compare
            logical_operator: AND/OR
        
        Returns:
            QueryBuilder instance for chaining
        """
        condition = QueryCondition(
            column=column,
            operator=operator,
            value=value,
            logical_operator=logical_operator.upper()
        )
        
        self._having_conditions.append(condition)
        return self
    
    def order_by(self, column: str, direction: OrderDirection = OrderDirection.ASC) -> 'QueryBuilder':
        """
        Add an ORDER BY clause.
        
        Args:
            column: Column name
            direction: Sort direction
        
        Returns:
            QueryBuilder instance for chaining
        """
        if not self._is_safe_identifier(column):
            raise ValueError(f"Invalid column identifier: {column}")
        
        order_clause = OrderClause(column=column, direction=direction)
        self._order_by.append(order_clause)
        
        return self
    
    def limit(self, count: int) -> 'QueryBuilder':
        """
        Add a LIMIT clause.
        
        Args:
            count: Maximum number of rows
        
        Returns:
            QueryBuilder instance for chaining
        """
        if count < 0:
            raise ValueError("LIMIT count must be non-negative")
        
        self._limit_value = count
        return self
    
    def offset(self, count: int) -> 'QueryBuilder':
        """
        Add an OFFSET clause.
        
        Args:
            count: Number of rows to skip
        
        Returns:
            QueryBuilder instance for chaining
        """
        if count < 0:
            raise ValueError("OFFSET count must be non-negative")
        
        self._offset_value = count
        return self
    
    def paginate(self, page: int, per_page: int) -> 'QueryBuilder':
        """
        Add pagination (LIMIT and OFFSET).
        
        Args:
            page: Page number (1-based)
            per_page: Items per page
        
        Returns:
            QueryBuilder instance for chaining
        """
        if page < 1:
            raise ValueError("Page number must be >= 1")
        
        if per_page < 1:
            raise ValueError("Items per page must be >= 1")
        
        self._limit_value = per_page
        self._offset_value = (page - 1) * per_page
        
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """
        Build the SQL query and parameters.
        
        Returns:
            Tuple of (query_string, parameters)
        """
        if not self._from_table:
            raise ValueError("FROM table is required")
        
        query_parts = []
        
        # SELECT clause
        if self._select_columns:
            columns = ", ".join(self._select_columns)
        else:
            columns = "*"
        
        query_parts.append(f"SELECT {columns}")
        
        # FROM clause
        from_clause = f"FROM {self._from_table}"
        if self._table_alias:
            from_clause += f" AS {self._table_alias}"
        query_parts.append(from_clause)
        
        # JOIN clauses
        for join in self._joins:
            join_clause = f"{join.join_type.value} {join.table}"
            if join.alias:
                join_clause += f" AS {join.alias}"
            join_clause += f" ON {join.on_condition}"
            query_parts.append(join_clause)
        
        # WHERE clause
        if self._conditions:
            where_clause = self._build_conditions(self._conditions)
            query_parts.append(f"WHERE {where_clause}")
        
        # GROUP BY clause
        if self._group_by:
            group_clause = "GROUP BY " + ", ".join(self._group_by)
            query_parts.append(group_clause)
        
        # HAVING clause
        if self._having_conditions:
            having_clause = self._build_conditions(self._having_conditions)
            query_parts.append(f"HAVING {having_clause}")
        
        # ORDER BY clause
        if self._order_by:
            order_parts = []
            for order in self._order_by:
                order_parts.append(f"{order.column} {order.direction.value}")
            query_parts.append("ORDER BY " + ", ".join(order_parts))
        
        # LIMIT clause
        if self._limit_value is not None:
            query_parts.append(f"LIMIT {self._limit_value}")
        
        # OFFSET clause
        if self._offset_value is not None:
            query_parts.append(f"OFFSET {self._offset_value}")
        
        query = " ".join(query_parts)
        
        logger.debug(f"Built query: {query}")
        logger.debug(f"Parameters: {self._parameters}")
        
        return query, self._parameters.copy()
    
    def _build_conditions(self, conditions: List[QueryCondition]) -> str:
        """Build WHERE or HAVING conditions."""
        if not conditions:
            return ""
        
        condition_parts = []
        
        for i, condition in enumerate(conditions):
            if i > 0:
                condition_parts.append(condition.logical_operator)
            
            if condition.operator.upper() in ["IS NULL", "IS NOT NULL"]:
                condition_parts.append(f"{condition.column} {condition.operator}")
            elif condition.operator.upper() in ["IN", "NOT IN"]:
                if isinstance(condition.value, (list, tuple)):
                    placeholders = []
                    for value in condition.value:
                        param_name = self._add_parameter(value)
                        placeholders.append(f"${param_name}")
                    
                    condition_parts.append(
                        f"{condition.column} {condition.operator} ({', '.join(placeholders)})"
                    )
                else:
                    raise ValueError(f"IN/NOT IN operator requires list/tuple value")
            else:
                param_name = self._add_parameter(condition.value)
                condition_parts.append(f"{condition.column} {condition.operator} ${param_name}")
        
        return " ".join(condition_parts)
    
    def _add_parameter(self, value: Any) -> str:
        """Add a parameter and return its placeholder name."""
        self._parameter_counter += 1
        param_name = f"param_{self._parameter_counter}"
        self._parameters[param_name] = value
        return param_name
    
    def _is_safe_identifier(self, identifier: str) -> bool:
        """
        Check if an identifier is safe (prevents SQL injection).
        
        Args:
            identifier: SQL identifier to check
        
        Returns:
            True if safe, False otherwise
        """
        if not identifier:
            return False
        
        # Allow alphanumeric, underscore, dot (for table.column), and spaces (for aliases)
        # Also allow common SQL functions and expressions
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_\.\s\(\)\*,]*$'
        
        if not re.match(pattern, identifier):
            return False
        
        # Check for SQL keywords that shouldn't be used as identifiers
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER', 'TRUNCATE',
            'EXEC', 'EXECUTE', 'UNION', 'SCRIPT', 'DECLARE', 'CURSOR'
        ]
        
        identifier_upper = identifier.upper()
        for keyword in dangerous_keywords:
            if keyword in identifier_upper:
                return False
        
        return True


class InsertQueryBuilder:
    """Builder for INSERT queries."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'InsertQueryBuilder':
        """Reset the builder."""
        self._table: Optional[str] = None
        self._columns: List[str] = []
        self._values: List[Dict[str, Any]] = []
        self._on_conflict: Optional[str] = None
        self._returning: List[str] = []
        
        return self
    
    def into(self, table: str) -> 'InsertQueryBuilder':
        """Set the target table."""
        if not self._is_safe_identifier(table):
            raise ValueError(f"Invalid table identifier: {table}")
        
        self._table = table
        return self
    
    def values(self, **kwargs) -> 'InsertQueryBuilder':
        """Add values to insert."""
        if not kwargs:
            raise ValueError("Values cannot be empty")
        
        # Validate column names
        for column in kwargs.keys():
            if not self._is_safe_identifier(column):
                raise ValueError(f"Invalid column identifier: {column}")
        
        self._values.append(kwargs)
        
        # Update columns list
        for column in kwargs.keys():
            if column not in self._columns:
                self._columns.append(column)
        
        return self
    
    def on_conflict_do_nothing(self) -> 'InsertQueryBuilder':
        """Add ON CONFLICT DO NOTHING clause."""
        self._on_conflict = "DO NOTHING"
        return self
    
    def on_conflict_update(self, conflict_columns: List[str], update_columns: List[str]) -> 'InsertQueryBuilder':
        """Add ON CONFLICT DO UPDATE clause."""
        # Validate column names
        for column in conflict_columns + update_columns:
            if not self._is_safe_identifier(column):
                raise ValueError(f"Invalid column identifier: {column}")
        
        conflict_cols = ", ".join(conflict_columns)
        update_sets = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_columns])
        
        self._on_conflict = f"({conflict_cols}) DO UPDATE SET {update_sets}"
        return self
    
    def returning(self, *columns: str) -> 'InsertQueryBuilder':
        """Add RETURNING clause."""
        for column in columns:
            if self._is_safe_identifier(column):
                self._returning.append(column)
            else:
                raise ValueError(f"Invalid column identifier: {column}")
        
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the INSERT query."""
        if not self._table:
            raise ValueError("Table is required")
        
        if not self._values:
            raise ValueError("Values are required")
        
        # Build columns list
        all_columns = list(self._columns)
        
        # Build VALUES clause
        value_rows = []
        parameters = {}
        param_counter = 0
        
        for row_values in self._values:
            row_placeholders = []
            for column in all_columns:
                param_counter += 1
                param_name = f"param_{param_counter}"
                parameters[param_name] = row_values.get(column)
                row_placeholders.append(f"${param_name}")
            
            value_rows.append(f"({', '.join(row_placeholders)})")
        
        # Build query
        columns_clause = f"({', '.join(all_columns)})"
        values_clause = ", ".join(value_rows)
        
        query = f"INSERT INTO {self._table} {columns_clause} VALUES {values_clause}"
        
        # Add ON CONFLICT clause
        if self._on_conflict:
            query += f" ON CONFLICT {self._on_conflict}"
        
        # Add RETURNING clause
        if self._returning:
            query += f" RETURNING {', '.join(self._returning)}"
        
        return query, parameters
    
    def _is_safe_identifier(self, identifier: str) -> bool:
        """Check if identifier is safe."""
        if not identifier:
            return False
        
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return re.match(pattern, identifier) is not None


class UpdateQueryBuilder:
    """Builder for UPDATE queries."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'UpdateQueryBuilder':
        """Reset the builder."""
        self._table: Optional[str] = None
        self._set_values: Dict[str, Any] = {}
        self._conditions: List[QueryCondition] = []
        self._returning: List[str] = []
        
        return self
    
    def table(self, table: str) -> 'UpdateQueryBuilder':
        """Set the target table."""
        if not self._is_safe_identifier(table):
            raise ValueError(f"Invalid table identifier: {table}")
        
        self._table = table
        return self
    
    def set(self, **kwargs) -> 'UpdateQueryBuilder':
        """Set column values."""
        for column, value in kwargs.items():
            if not self._is_safe_identifier(column):
                raise ValueError(f"Invalid column identifier: {column}")
            
            self._set_values[column] = value
        
        return self
    
    def where(self, column: str, operator: str, value: Any, logical_operator: str = "AND") -> 'UpdateQueryBuilder':
        """Add WHERE condition."""
        if not self._is_safe_identifier(column):
            raise ValueError(f"Invalid column identifier: {column}")
        
        condition = QueryCondition(
            column=column,
            operator=operator,
            value=value,
            logical_operator=logical_operator.upper()
        )
        
        self._conditions.append(condition)
        return self
    
    def returning(self, *columns: str) -> 'UpdateQueryBuilder':
        """Add RETURNING clause."""
        for column in columns:
            if self._is_safe_identifier(column):
                self._returning.append(column)
            else:
                raise ValueError(f"Invalid column identifier: {column}")
        
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the UPDATE query."""
        if not self._table:
            raise ValueError("Table is required")
        
        if not self._set_values:
            raise ValueError("SET values are required")
        
        parameters = {}
        param_counter = 0
        
        # Build SET clause
        set_parts = []
        for column, value in self._set_values.items():
            param_counter += 1
            param_name = f"param_{param_counter}"
            parameters[param_name] = value
            set_parts.append(f"{column} = ${param_name}")
        
        set_clause = ", ".join(set_parts)
        
        # Build WHERE clause
        where_clause = ""
        if self._conditions:
            condition_parts = []
            for i, condition in enumerate(self._conditions):
                if i > 0:
                    condition_parts.append(condition.logical_operator)
                
                param_counter += 1
                param_name = f"param_{param_counter}"
                parameters[param_name] = condition.value
                condition_parts.append(f"{condition.column} {condition.operator} ${param_name}")
            
            where_clause = " WHERE " + " ".join(condition_parts)
        
        # Build query
        query = f"UPDATE {self._table} SET {set_clause}{where_clause}"
        
        # Add RETURNING clause
        if self._returning:
            query += f" RETURNING {', '.join(self._returning)}"
        
        return query, parameters
    
    def _is_safe_identifier(self, identifier: str) -> bool:
        """Check if identifier is safe."""
        if not identifier:
            return False
        
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return re.match(pattern, identifier) is not None


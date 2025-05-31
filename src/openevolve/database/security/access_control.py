"""
Database access control and security management.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
import secrets

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Database permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SCHEMA_CREATE = "schema_create"
    SCHEMA_DROP = "schema_drop"


@dataclass
class AccessRule:
    """Access control rule."""
    
    user_id: str
    resource_type: str  # table, schema, database
    resource_name: str
    permissions: Set[Permission]
    conditions: Dict[str, Any] = None  # Additional conditions
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


class AccessControlManager:
    """
    Database access control manager with role-based permissions
    and fine-grained access control.
    """
    
    def __init__(self):
        self._access_rules: Dict[str, List[AccessRule]] = {}
        self._roles: Dict[str, Set[Permission]] = {}
        self._user_roles: Dict[str, Set[str]] = {}
        self._api_keys: Dict[str, str] = {}  # api_key -> user_id
        
        # Initialize default roles
        self._initialize_default_roles()
        
        logger.info("Access control manager initialized")
    
    def _initialize_default_roles(self) -> None:
        """Initialize default roles."""
        self._roles = {
            "admin": {Permission.READ, Permission.WRITE, Permission.DELETE, 
                     Permission.ADMIN, Permission.SCHEMA_CREATE, Permission.SCHEMA_DROP},
            "developer": {Permission.READ, Permission.WRITE, Permission.SCHEMA_CREATE},
            "analyst": {Permission.READ},
            "service": {Permission.READ, Permission.WRITE}
        }
    
    def create_role(self, role_name: str, permissions: Set[Permission]) -> None:
        """
        Create a new role.
        
        Args:
            role_name: Name of the role
            permissions: Set of permissions for the role
        """
        self._roles[role_name] = permissions
        logger.info(f"Created role '{role_name}' with permissions: {[p.value for p in permissions]}")
    
    def assign_role(self, user_id: str, role_name: str) -> None:
        """
        Assign a role to a user.
        
        Args:
            user_id: User identifier
            role_name: Role to assign
        """
        if role_name not in self._roles:
            raise ValueError(f"Role '{role_name}' does not exist")
        
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()
        
        self._user_roles[user_id].add(role_name)
        logger.info(f"Assigned role '{role_name}' to user '{user_id}'")
    
    def revoke_role(self, user_id: str, role_name: str) -> None:
        """
        Revoke a role from a user.
        
        Args:
            user_id: User identifier
            role_name: Role to revoke
        """
        if user_id in self._user_roles:
            self._user_roles[user_id].discard(role_name)
            logger.info(f"Revoked role '{role_name}' from user '{user_id}'")
    
    def add_access_rule(self, rule: AccessRule) -> None:
        """
        Add an access control rule.
        
        Args:
            rule: Access rule to add
        """
        if rule.user_id not in self._access_rules:
            self._access_rules[rule.user_id] = []
        
        self._access_rules[rule.user_id].append(rule)
        logger.info(f"Added access rule for user '{rule.user_id}' on '{rule.resource_name}'")
    
    def remove_access_rule(self, user_id: str, resource_type: str, resource_name: str) -> None:
        """
        Remove an access control rule.
        
        Args:
            user_id: User identifier
            resource_type: Type of resource
            resource_name: Name of resource
        """
        if user_id in self._access_rules:
            self._access_rules[user_id] = [
                rule for rule in self._access_rules[user_id]
                if not (rule.resource_type == resource_type and rule.resource_name == resource_name)
            ]
            logger.info(f"Removed access rule for user '{user_id}' on '{resource_name}'")
    
    def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource_type: str,
        resource_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a user has permission for a resource.
        
        Args:
            user_id: User identifier
            permission: Permission to check
            resource_type: Type of resource
            resource_name: Name of resource
            context: Additional context for permission check
        
        Returns:
            True if permission is granted
        """
        # Check role-based permissions
        user_permissions = self._get_user_permissions(user_id)
        if permission in user_permissions:
            # Check for admin permission (grants all access)
            if Permission.ADMIN in user_permissions:
                return True
        
        # Check specific access rules
        if user_id in self._access_rules:
            for rule in self._access_rules[user_id]:
                if (rule.resource_type == resource_type and 
                    rule.resource_name == resource_name and
                    permission in rule.permissions):
                    
                    # Check additional conditions
                    if self._check_rule_conditions(rule, context):
                        return True
        
        # Check wildcard rules
        if user_id in self._access_rules:
            for rule in self._access_rules[user_id]:
                if (rule.resource_type == resource_type and 
                    rule.resource_name == "*" and
                    permission in rule.permissions):
                    
                    if self._check_rule_conditions(rule, context):
                        return True
        
        logger.warning(f"Permission denied: user '{user_id}' lacks '{permission.value}' on '{resource_name}'")
        return False
    
    def _get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user based on their roles."""
        permissions = set()
        
        if user_id in self._user_roles:
            for role_name in self._user_roles[user_id]:
                if role_name in self._roles:
                    permissions.update(self._roles[role_name])
        
        return permissions
    
    def _check_rule_conditions(self, rule: AccessRule, context: Optional[Dict[str, Any]]) -> bool:
        """Check if rule conditions are met."""
        if not rule.conditions:
            return True
        
        if not context:
            context = {}
        
        # Check time-based conditions
        if "time_range" in rule.conditions:
            # Implementation for time-based access
            pass
        
        # Check IP-based conditions
        if "allowed_ips" in rule.conditions:
            client_ip = context.get("client_ip")
            if client_ip and client_ip not in rule.conditions["allowed_ips"]:
                return False
        
        # Check tenant-based conditions
        if "tenant_id" in rule.conditions:
            request_tenant = context.get("tenant_id")
            if request_tenant != rule.conditions["tenant_id"]:
                return False
        
        return True
    
    def generate_api_key(self, user_id: str) -> str:
        """
        Generate an API key for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            Generated API key
        """
        api_key = secrets.token_urlsafe(32)
        self._api_keys[api_key] = user_id
        
        logger.info(f"Generated API key for user '{user_id}'")
        return api_key
    
    def revoke_api_key(self, api_key: str) -> None:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
        """
        if api_key in self._api_keys:
            user_id = self._api_keys[api_key]
            del self._api_keys[api_key]
            logger.info(f"Revoked API key for user '{user_id}'")
    
    def authenticate_api_key(self, api_key: str) -> Optional[str]:
        """
        Authenticate an API key and return the user ID.
        
        Args:
            api_key: API key to authenticate
        
        Returns:
            User ID if valid, None otherwise
        """
        return self._api_keys.get(api_key)
    
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get information about a user's permissions and roles.
        
        Args:
            user_id: User identifier
        
        Returns:
            User information dictionary
        """
        roles = list(self._user_roles.get(user_id, set()))
        permissions = list(self._get_user_permissions(user_id))
        access_rules = self._access_rules.get(user_id, [])
        
        return {
            "user_id": user_id,
            "roles": roles,
            "permissions": [p.value for p in permissions],
            "access_rules": [
                {
                    "resource_type": rule.resource_type,
                    "resource_name": rule.resource_name,
                    "permissions": [p.value for p in rule.permissions],
                    "conditions": rule.conditions
                }
                for rule in access_rules
            ],
            "api_keys_count": sum(1 for uid in self._api_keys.values() if uid == user_id)
        }
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users with their roles and permissions."""
        all_users = set()
        all_users.update(self._user_roles.keys())
        all_users.update(self._access_rules.keys())
        all_users.update(self._api_keys.values())
        
        return [self.get_user_info(user_id) for user_id in sorted(all_users)]
    
    def get_resource_permissions(self, resource_type: str, resource_name: str) -> Dict[str, List[str]]:
        """
        Get all users and their permissions for a specific resource.
        
        Args:
            resource_type: Type of resource
            resource_name: Name of resource
        
        Returns:
            Dictionary mapping user IDs to their permissions
        """
        resource_permissions = {}
        
        # Check all users
        for user_id in set(list(self._user_roles.keys()) + list(self._access_rules.keys())):
            user_perms = []
            
            # Check each permission type
            for permission in Permission:
                if self.check_permission(user_id, permission, resource_type, resource_name):
                    user_perms.append(permission.value)
            
            if user_perms:
                resource_permissions[user_id] = user_perms
        
        return resource_permissions
    
    def audit_permissions(self) -> Dict[str, Any]:
        """Generate an audit report of all permissions."""
        return {
            "roles": {
                name: [p.value for p in permissions]
                for name, permissions in self._roles.items()
            },
            "user_roles": {
                user_id: list(roles)
                for user_id, roles in self._user_roles.items()
            },
            "access_rules_count": sum(len(rules) for rules in self._access_rules.values()),
            "api_keys_count": len(self._api_keys),
            "total_users": len(set(
                list(self._user_roles.keys()) + 
                list(self._access_rules.keys()) + 
                list(self._api_keys.values())
            ))
        }


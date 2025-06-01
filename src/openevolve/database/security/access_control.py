"""
Simplified access control for single-user autonomous development.
"""

import logging
from enum import Enum
from typing import Dict, Set, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions for autonomous operations."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class AccessRule:
    """Access rule for resources."""
    resource_type: str
    resource_name: str
    permissions: Set[Permission]
    conditions: Dict[str, Any]


class AccessControlManager:
    """
    Simplified access control manager for single-user autonomous development.
    Provides basic resource protection and operation logging.
    """
    
    def __init__(self):
        self._resource_rules: Dict[str, AccessRule] = {}
        self._system_permissions: Set[Permission] = {
            Permission.READ,
            Permission.WRITE,
            Permission.DELETE,
            Permission.EXECUTE,
            Permission.ADMIN
        }
        
        logger.info("Access control manager initialized for autonomous mode")
    
    def add_resource_rule(self, rule: AccessRule) -> None:
        """
        Add a resource access rule.
        
        Args:
            rule: Access rule to add
        """
        resource_key = f"{rule.resource_type}:{rule.resource_name}"
        self._resource_rules[resource_key] = rule
        
        logger.info(f"Added access rule for resource '{rule.resource_name}' of type '{rule.resource_type}'")
    
    def remove_resource_rule(self, resource_type: str, resource_name: str) -> None:
        """
        Remove a resource access rule.
        
        Args:
            resource_type: Type of resource
            resource_name: Name of resource
        """
        resource_key = f"{resource_type}:{resource_name}"
        if resource_key in self._resource_rules:
            del self._resource_rules[resource_key]
            logger.info(f"Removed access rule for resource '{resource_name}' of type '{resource_type}'")
    
    def check_permission(
        self,
        permission: Permission,
        resource_type: str,
        resource_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a permission is allowed for a resource.
        In autonomous mode, this primarily serves as logging and validation.
        
        Args:
            permission: Permission to check
            resource_type: Type of resource
            resource_name: Name of resource
            context: Optional context for the check
        
        Returns:
            True if permission is granted (always True in autonomous mode)
        """
        resource_key = f"{resource_type}:{resource_name}"
        
        # Log the access attempt
        logger.debug(f"Permission check: {permission.value} on {resource_type}/{resource_name}")
        
        # Check if there's a specific rule for this resource
        if resource_key in self._resource_rules:
            rule = self._resource_rules[resource_key]
            
            # Check if permission is in the rule's allowed permissions
            if permission not in rule.permissions:
                logger.warning(f"Permission '{permission.value}' not allowed for resource '{resource_name}'")
                return False
            
            # Check any additional conditions
            if rule.conditions and context:
                for condition_key, condition_value in rule.conditions.items():
                    if context.get(condition_key) != condition_value:
                        logger.warning(f"Access condition failed: {condition_key} = {condition_value}")
                        return False
        
        # In autonomous mode, default to allowing access
        logger.debug(f"Permission granted: {permission.value} on {resource_type}/{resource_name}")
        return True
    
    def get_system_permissions(self) -> Set[Permission]:
        """
        Get all available system permissions.
        
        Returns:
            Set of all system permissions
        """
        return self._system_permissions.copy()
    
    def get_resource_permissions(self, resource_type: str, resource_name: str) -> Set[Permission]:
        """
        Get permissions for a specific resource.
        
        Args:
            resource_type: Type of resource
            resource_name: Name of resource
        
        Returns:
            Set of permissions for the resource
        """
        resource_key = f"{resource_type}:{resource_name}"
        
        if resource_key in self._resource_rules:
            return self._resource_rules[resource_key].permissions.copy()
        
        # Default to all permissions in autonomous mode
        return self._system_permissions.copy()
    
    def list_resources(self) -> Dict[str, Dict[str, Any]]:
        """
        List all resources with their access rules.
        
        Returns:
            Dictionary of resources and their rules
        """
        resources = {}
        
        for resource_key, rule in self._resource_rules.items():
            resources[resource_key] = {
                "resource_type": rule.resource_type,
                "resource_name": rule.resource_name,
                "permissions": [p.value for p in rule.permissions],
                "conditions": rule.conditions
            }
        
        return resources
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get access control statistics.
        
        Returns:
            Dictionary with access control statistics
        """
        return {
            "total_resources": len(self._resource_rules),
            "system_permissions": [p.value for p in self._system_permissions],
            "resource_types": list(set(rule.resource_type for rule in self._resource_rules.values()))
        }
    
    def validate_system(self) -> bool:
        """
        Validate the access control system configuration.
        
        Returns:
            True if system is valid
        """
        try:
            # Basic validation checks
            if not self._system_permissions:
                logger.error("No system permissions defined")
                return False
            
            # Validate resource rules
            for resource_key, rule in self._resource_rules.items():
                if not rule.permissions:
                    logger.warning(f"Resource '{resource_key}' has no permissions defined")
                
                # Check if all permissions are valid
                invalid_perms = rule.permissions - self._system_permissions
                if invalid_perms:
                    logger.error(f"Resource '{resource_key}' has invalid permissions: {invalid_perms}")
                    return False
            
            logger.info("Access control system validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Access control system validation failed: {e}")
            return False


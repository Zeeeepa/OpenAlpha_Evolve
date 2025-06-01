"""
Webhook Validator

Validates Linear webhook signatures and payloads for security.
"""

import hashlib
import hmac
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class WebhookValidator:
    """Validates Linear webhook signatures and payloads"""
    
    def __init__(self, webhook_secret: Optional[str] = None):
        self.webhook_secret = webhook_secret
        
    def validate_signature(self, payload: bytes, signature: str) -> bool:
        """
        Validate webhook signature using HMAC-SHA256
        
        Args:
            payload: Raw webhook payload bytes
            signature: Signature from Linear-Signature header
            
        Returns:
            bool: True if signature is valid
        """
        if not self.webhook_secret:
            logger.warning("No webhook secret configured, skipping signature validation")
            return True
            
        if not signature:
            logger.error("No signature provided in webhook request")
            return False
            
        try:
            # Linear sends signature in format: sha256=<hash>
            if not signature.startswith("sha256="):
                logger.error("Invalid signature format")
                return False
                
            expected_signature = signature[7:]  # Remove 'sha256=' prefix
            
            # Calculate expected signature
            calculated_signature = hmac.new(
                self.webhook_secret.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Use constant-time comparison to prevent timing attacks
            is_valid = hmac.compare_digest(calculated_signature, expected_signature)
            
            if not is_valid:
                logger.error("Webhook signature validation failed")
                
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating webhook signature: {e}")
            return False
    
    def validate_payload(self, payload: Dict[str, Any]) -> bool:
        """
        Validate webhook payload structure
        
        Args:
            payload: Parsed webhook payload
            
        Returns:
            bool: True if payload is valid
        """
        try:
            # Check required fields
            required_fields = ["action", "type", "data"]
            for field in required_fields:
                if field not in payload:
                    logger.error(f"Missing required field in webhook payload: {field}")
                    return False
            
            # Validate action
            valid_actions = ["create", "update", "remove"]
            if payload["action"] not in valid_actions:
                logger.error(f"Invalid action in webhook payload: {payload['action']}")
                return False
            
            # Validate type
            valid_types = [
                "Issue", "IssueUpdate", "Comment", "CommentUpdate",
                "Project", "ProjectUpdate", "Cycle", "CycleUpdate",
                "Team", "TeamUpdate", "User", "UserUpdate"
            ]
            if payload["type"] not in valid_types:
                logger.error(f"Invalid type in webhook payload: {payload['type']}")
                return False
            
            # Validate data structure
            if not isinstance(payload["data"], dict):
                logger.error("Invalid data structure in webhook payload")
                return False
            
            # Check for required data fields based on type
            data = payload["data"]
            if payload["type"] in ["Issue", "IssueUpdate"]:
                if "id" not in data:
                    logger.error("Missing issue ID in webhook payload")
                    return False
            elif payload["type"] in ["Comment", "CommentUpdate"]:
                if "id" not in data or "issueId" not in data:
                    logger.error("Missing comment or issue ID in webhook payload")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating webhook payload: {e}")
            return False
    
    def validate_timestamp(self, timestamp: str, max_age_seconds: int = 300) -> bool:
        """
        Validate webhook timestamp to prevent replay attacks
        
        Args:
            timestamp: Timestamp from webhook headers
            max_age_seconds: Maximum age of webhook in seconds
            
        Returns:
            bool: True if timestamp is valid
        """
        try:
            from datetime import datetime, timezone
            
            # Parse timestamp
            webhook_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            
            # Check if webhook is too old
            age_seconds = (current_time - webhook_time).total_seconds()
            if age_seconds > max_age_seconds:
                logger.error(f"Webhook too old: {age_seconds} seconds")
                return False
            
            # Check if webhook is from the future (with small tolerance)
            if age_seconds < -30:  # 30 second tolerance for clock skew
                logger.error(f"Webhook from future: {age_seconds} seconds")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating webhook timestamp: {e}")
            return False
    
    def sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize webhook payload to remove potentially dangerous content
        
        Args:
            payload: Raw webhook payload
            
        Returns:
            Dict: Sanitized payload
        """
        try:
            # Create a deep copy to avoid modifying original
            import copy
            sanitized = copy.deepcopy(payload)
            
            # Remove or sanitize potentially dangerous fields
            dangerous_fields = ["__proto__", "constructor", "prototype"]
            
            def clean_dict(d):
                if isinstance(d, dict):
                    for key in list(d.keys()):
                        if key in dangerous_fields:
                            del d[key]
                        else:
                            clean_dict(d[key])
                elif isinstance(d, list):
                    for item in d:
                        clean_dict(item)
            
            clean_dict(sanitized)
            
            # Limit string lengths to prevent memory exhaustion
            def limit_strings(obj, max_length=10000):
                if isinstance(obj, str):
                    return obj[:max_length]
                elif isinstance(obj, dict):
                    return {k: limit_strings(v, max_length) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [limit_strings(item, max_length) for item in obj]
                else:
                    return obj
            
            sanitized = limit_strings(sanitized)
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing webhook payload: {e}")
            return payload


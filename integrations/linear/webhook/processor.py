"""
Webhook Processor

Main webhook processing engine that handles incoming Linear webhooks,
validates them, and routes them to appropriate handlers.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import uuid

from ..interfaces import WebhookProcessorInterface, WebhookEvent, LinearEventType
from .validator import WebhookValidator
from .handlers import WebhookHandlers

logger = logging.getLogger(__name__)


class WebhookProcessor(WebhookProcessorInterface):
    """Main webhook processor for Linear webhooks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.webhook_secret = self.config.get("webhook_secret")
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 5)
        self.max_payload_size = self.config.get("max_payload_size", 1024 * 1024)  # 1MB
        
        # Initialize components
        self.validator = WebhookValidator(self.webhook_secret)
        self.handlers = WebhookHandlers()
        
        # Event storage for retry and monitoring
        self.processed_events: Dict[str, WebhookEvent] = {}
        self.failed_events: Dict[str, WebhookEvent] = {}
        
        # Register default handlers
        self._register_default_handlers()
        
        logger.info("WebhookProcessor initialized")
    
    def _register_default_handlers(self) -> None:
        """Register default event handlers"""
        self.handlers.register_handler(LinearEventType.ISSUE_CREATE, self.handlers.handle_issue_event)
        self.handlers.register_handler(LinearEventType.ISSUE_UPDATE, self.handlers.handle_issue_event)
        self.handlers.register_handler(LinearEventType.COMMENT_CREATE, self.handlers.handle_comment_event)
        self.handlers.register_handler(LinearEventType.COMMENT_UPDATE, self.handlers.handle_comment_event)
        self.handlers.register_handler(LinearEventType.PROJECT_UPDATE, self.handlers.handle_project_event)
        self.handlers.register_handler(LinearEventType.CYCLE_UPDATE, self.handlers.handle_project_event)
    
    async def process_webhook(
        self, 
        payload: bytes, 
        signature: Optional[str] = None,
        timestamp: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Process incoming webhook
        
        Args:
            payload: Raw webhook payload bytes
            signature: Webhook signature for validation
            timestamp: Webhook timestamp
            headers: Additional headers
            
        Returns:
            bool: True if webhook was processed successfully
        """
        try:
            # Validate payload size
            if len(payload) > self.max_payload_size:
                logger.error(f"Webhook payload too large: {len(payload)} bytes")
                return False
            
            # Validate signature
            if not self.validator.validate_signature(payload, signature or ""):
                logger.error("Webhook signature validation failed")
                return False
            
            # Validate timestamp if provided
            if timestamp and not self.validator.validate_timestamp(timestamp):
                logger.error("Webhook timestamp validation failed")
                return False
            
            # Parse payload
            try:
                payload_dict = json.loads(payload.decode('utf-8'))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse webhook payload as JSON: {e}")
                return False
            
            # Validate payload structure
            if not self.validator.validate_payload(payload_dict):
                logger.error("Webhook payload validation failed")
                return False
            
            # Sanitize payload
            payload_dict = self.validator.sanitize_payload(payload_dict)
            
            # Create webhook event
            event = self._create_webhook_event(payload_dict, headers)
            
            # Process event
            success = await self.process_event(event)
            
            if success:
                self.processed_events[event.id] = event
                logger.info(f"Successfully processed webhook event: {event.id}")
            else:
                self.failed_events[event.id] = event
                logger.error(f"Failed to process webhook event: {event.id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return False
    
    async def process_event(self, event: WebhookEvent) -> bool:
        """Process a webhook event"""
        try:
            logger.info(f"Processing webhook event: {event.type.value} - {event.action}")
            
            # Handle the event
            success = await self.handlers.handle_event(event)
            
            if success:
                event.processed = True
                logger.debug(f"Event {event.id} processed successfully")
            else:
                event.retry_count += 1
                logger.warning(f"Event {event.id} processing failed, retry count: {event.retry_count}")
                
                # Retry if under limit
                if event.retry_count < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                    return await self.process_event(event)
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing webhook event {event.id}: {e}")
            event.retry_count += 1
            return False
    
    async def validate_webhook(self, payload: Dict[str, Any], signature: str) -> bool:
        """Validate webhook signature and payload"""
        try:
            # Convert payload to bytes for signature validation
            payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
            
            # Validate signature
            if not self.validator.validate_signature(payload_bytes, signature):
                return False
            
            # Validate payload structure
            return self.validator.validate_payload(payload)
            
        except Exception as e:
            logger.error(f"Error validating webhook: {e}")
            return False
    
    def register_handler(self, event_type: LinearEventType, handler: Callable) -> None:
        """Register event handler"""
        self.handlers.register_handler(event_type, handler)
    
    def register_global_handler(self, handler: Callable) -> None:
        """Register global event handler"""
        self.handlers.register_global_handler(handler)
    
    def _create_webhook_event(
        self, 
        payload: Dict[str, Any], 
        headers: Optional[Dict[str, str]] = None
    ) -> WebhookEvent:
        """Create webhook event from payload"""
        
        # Map Linear event types
        type_mapping = {
            "Issue": LinearEventType.ISSUE_CREATE if payload.get("action") == "create" else LinearEventType.ISSUE_UPDATE,
            "Comment": LinearEventType.COMMENT_CREATE if payload.get("action") == "create" else LinearEventType.COMMENT_UPDATE,
            "Project": LinearEventType.PROJECT_UPDATE,
            "Cycle": LinearEventType.CYCLE_UPDATE
        }
        
        event_type = type_mapping.get(payload.get("type"), LinearEventType.ISSUE_UPDATE)
        
        return WebhookEvent(
            id=str(uuid.uuid4()),
            type=event_type,
            action=payload.get("action", "unknown"),
            data=payload.get("data", {}),
            timestamp=datetime.now(),
            processed=False,
            retry_count=0
        )
    
    async def retry_failed_events(self) -> int:
        """Retry failed events"""
        retry_count = 0
        failed_event_ids = list(self.failed_events.keys())
        
        for event_id in failed_event_ids:
            event = self.failed_events[event_id]
            
            if event.retry_count < self.max_retries:
                logger.info(f"Retrying failed event: {event_id}")
                success = await self.process_event(event)
                
                if success:
                    # Move from failed to processed
                    del self.failed_events[event_id]
                    self.processed_events[event_id] = event
                    retry_count += 1
        
        logger.info(f"Retried {retry_count} failed events")
        return retry_count
    
    async def get_event_status(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get event processing status"""
        if event_id in self.processed_events:
            event = self.processed_events[event_id]
            return {
                "id": event.id,
                "type": event.type.value,
                "action": event.action,
                "status": "processed",
                "processed": event.processed,
                "retry_count": event.retry_count,
                "timestamp": event.timestamp.isoformat()
            }
        elif event_id in self.failed_events:
            event = self.failed_events[event_id]
            return {
                "id": event.id,
                "type": event.type.value,
                "action": event.action,
                "status": "failed",
                "processed": event.processed,
                "retry_count": event.retry_count,
                "timestamp": event.timestamp.isoformat()
            }
        else:
            return None
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get webhook processing statistics"""
        return {
            "processed_events": len(self.processed_events),
            "failed_events": len(self.failed_events),
            "total_events": len(self.processed_events) + len(self.failed_events),
            "success_rate": len(self.processed_events) / max(1, len(self.processed_events) + len(self.failed_events)),
            "recent_events": [
                {
                    "id": event.id,
                    "type": event.type.value,
                    "action": event.action,
                    "timestamp": event.timestamp.isoformat(),
                    "processed": event.processed
                }
                for event in sorted(
                    list(self.processed_events.values()) + list(self.failed_events.values()),
                    key=lambda e: e.timestamp,
                    reverse=True
                )[:10]
            ]
        }
    
    async def cleanup_old_events(self, max_age_hours: int = 24) -> int:
        """Clean up old processed events"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        # Clean processed events
        processed_to_remove = [
            event_id for event_id, event in self.processed_events.items()
            if event.timestamp < cutoff_time
        ]
        
        for event_id in processed_to_remove:
            del self.processed_events[event_id]
            cleaned_count += 1
        
        # Clean old failed events (but keep recent ones for retry)
        failed_to_remove = [
            event_id for event_id, event in self.failed_events.items()
            if event.timestamp < cutoff_time and event.retry_count >= self.max_retries
        ]
        
        for event_id in failed_to_remove:
            del self.failed_events[event_id]
            cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old events")
        return cleaned_count
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute method for BaseAgent interface"""
        # Return processor status
        stats = await self.get_processing_stats()
        return {
            "status": "active",
            "webhook_secret_configured": bool(self.webhook_secret),
            "stats": stats
        }


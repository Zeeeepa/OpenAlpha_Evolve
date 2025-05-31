"""
Event Manager

Manages event queuing, processing, and persistence for Linear integration.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

from ..interfaces import EventManagerInterface, WebhookEvent
from .queue import EventQueue

logger = logging.getLogger(__name__)


class EventManager(EventManagerInterface):
    """Manages event processing and queuing for Linear integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Configuration
        self.max_queue_size = self.config.get("max_queue_size", 1000)
        self.batch_size = self.config.get("batch_size", 10)
        self.processing_interval = self.config.get("processing_interval", 5)
        self.retry_interval = self.config.get("retry_interval", 60)
        self.max_retries = self.config.get("max_retries", 3)
        self.persistence_enabled = self.config.get("persistence_enabled", True)
        self.persistence_file = self.config.get("persistence_file", "linear_events.json")
        
        # Event storage
        self.event_queue = EventQueue(self.max_queue_size)
        self.processing_events: Dict[str, WebhookEvent] = {}
        self.failed_events: Dict[str, WebhookEvent] = {}
        self.completed_events: Dict[str, WebhookEvent] = {}
        
        # Processing state
        self.processing_active = False
        self.processor_task: Optional[asyncio.Task] = None
        self.retry_task: Optional[asyncio.Task] = None
        
        logger.info("EventManager initialized")
    
    async def initialize(self) -> None:
        """Initialize event manager"""
        try:
            # Load persisted events if enabled
            if self.persistence_enabled:
                await self._load_persisted_events()
            
            # Start background processing
            await self._start_background_processing()
            
            logger.info("EventManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing EventManager: {e}")
            raise
    
    async def queue_event(self, event: WebhookEvent) -> bool:
        """Queue event for processing"""
        try:
            success = await self.event_queue.enqueue(event)
            
            if success:
                logger.debug(f"Queued event: {event.id}")
                
                # Persist if enabled
                if self.persistence_enabled:
                    await self._persist_event(event)
            else:
                logger.warning(f"Failed to queue event: {event.id} (queue full)")
            
            return success
            
        except Exception as e:
            logger.error(f"Error queuing event: {e}")
            return False
    
    async def process_queue(self) -> int:
        """Process queued events"""
        processed_count = 0
        
        try:
            # Process events in batches
            while not self.event_queue.is_empty() and processed_count < self.batch_size:
                event = await self.event_queue.dequeue()
                if event:
                    success = await self._process_single_event(event)
                    if success:
                        processed_count += 1
            
            logger.debug(f"Processed {processed_count} events from queue")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing event queue: {e}")
            return processed_count
    
    async def _process_single_event(self, event: WebhookEvent) -> bool:
        """Process a single event"""
        try:
            # Move to processing
            self.processing_events[event.id] = event
            
            # TODO: Integrate with actual event processors
            # For now, just simulate processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Mark as completed
            event.processed = True
            self.completed_events[event.id] = event
            
            # Remove from processing
            if event.id in self.processing_events:
                del self.processing_events[event.id]
            
            logger.debug(f"Successfully processed event: {event.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing event {event.id}: {e}")
            
            # Move to failed events
            event.retry_count += 1
            self.failed_events[event.id] = event
            
            # Remove from processing
            if event.id in self.processing_events:
                del self.processing_events[event.id]
            
            return False
    
    async def retry_failed_events(self) -> int:
        """Retry failed events"""
        retry_count = 0
        
        try:
            failed_event_ids = list(self.failed_events.keys())
            
            for event_id in failed_event_ids:
                event = self.failed_events[event_id]
                
                # Check if event should be retried
                if event.retry_count < self.max_retries:
                    # Check if enough time has passed since last attempt
                    time_since_last_attempt = datetime.now() - event.timestamp
                    if time_since_last_attempt.total_seconds() >= self.retry_interval:
                        logger.info(f"Retrying failed event: {event_id}")
                        
                        # Remove from failed events
                        del self.failed_events[event_id]
                        
                        # Re-queue for processing
                        await self.queue_event(event)
                        retry_count += 1
                else:
                    # Event has exceeded max retries, move to permanent failure
                    logger.warning(f"Event {event_id} exceeded max retries, marking as permanently failed")
                    # Could implement permanent failure handling here
            
            logger.info(f"Retried {retry_count} failed events")
            return retry_count
            
        except Exception as e:
            logger.error(f"Error retrying failed events: {e}")
            return retry_count
    
    async def get_event_status(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get event processing status"""
        # Check in queue
        if await self.event_queue.contains(event_id):
            return {
                "id": event_id,
                "status": "queued",
                "queue_position": await self.event_queue.get_position(event_id)
            }
        
        # Check in processing
        if event_id in self.processing_events:
            event = self.processing_events[event_id]
            return {
                "id": event.id,
                "type": event.type.value,
                "action": event.action,
                "status": "processing",
                "retry_count": event.retry_count,
                "timestamp": event.timestamp.isoformat()
            }
        
        # Check in completed
        if event_id in self.completed_events:
            event = self.completed_events[event_id]
            return {
                "id": event.id,
                "type": event.type.value,
                "action": event.action,
                "status": "completed",
                "processed": event.processed,
                "retry_count": event.retry_count,
                "timestamp": event.timestamp.isoformat()
            }
        
        # Check in failed
        if event_id in self.failed_events:
            event = self.failed_events[event_id]
            return {
                "id": event.id,
                "type": event.type.value,
                "action": event.action,
                "status": "failed",
                "retry_count": event.retry_count,
                "timestamp": event.timestamp.isoformat()
            }
        
        return None
    
    async def _start_background_processing(self) -> None:
        """Start background event processing"""
        if self.processing_active:
            return
        
        self.processing_active = True
        
        # Start processing task
        self.processor_task = asyncio.create_task(self._background_processor())
        
        # Start retry task
        self.retry_task = asyncio.create_task(self._background_retry())
        
        logger.info("Background event processing started")
    
    async def _background_processor(self) -> None:
        """Background event processor"""
        while self.processing_active:
            try:
                await self.process_queue()
                await asyncio.sleep(self.processing_interval)
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
                await asyncio.sleep(self.processing_interval)
    
    async def _background_retry(self) -> None:
        """Background retry processor"""
        while self.processing_active:
            try:
                await self.retry_failed_events()
                await asyncio.sleep(self.retry_interval)
            except Exception as e:
                logger.error(f"Error in background retry: {e}")
                await asyncio.sleep(self.retry_interval)
    
    async def stop_processing(self) -> None:
        """Stop background processing"""
        self.processing_active = False
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        if self.retry_task:
            self.retry_task.cancel()
            try:
                await self.retry_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Background event processing stopped")
    
    async def _persist_event(self, event: WebhookEvent) -> None:
        """Persist event to storage"""
        try:
            # Simple file-based persistence
            # In production, you'd want a proper database
            event_data = {
                "id": event.id,
                "type": event.type.value,
                "action": event.action,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
                "processed": event.processed,
                "retry_count": event.retry_count
            }
            
            # Append to file (simple approach)
            with open(self.persistence_file, "a") as f:
                f.write(json.dumps(event_data) + "\\n")
                
        except Exception as e:
            logger.error(f"Error persisting event: {e}")
    
    async def _load_persisted_events(self) -> None:
        """Load persisted events from storage"""
        try:
            import os
            if not os.path.exists(self.persistence_file):
                return
            
            with open(self.persistence_file, "r") as f:
                for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        
                        # Reconstruct event
                        from ..interfaces import LinearEventType
                        event = WebhookEvent(
                            id=event_data["id"],
                            type=LinearEventType(event_data["type"]),
                            action=event_data["action"],
                            data=event_data["data"],
                            timestamp=datetime.fromisoformat(event_data["timestamp"]),
                            processed=event_data["processed"],
                            retry_count=event_data["retry_count"]
                        )
                        
                        # Add to appropriate collection based on status
                        if event.processed:
                            self.completed_events[event.id] = event
                        elif event.retry_count >= self.max_retries:
                            self.failed_events[event.id] = event
                        else:
                            await self.queue_event(event)
                            
                    except Exception as e:
                        logger.error(f"Error loading persisted event: {e}")
            
            logger.info(f"Loaded persisted events from {self.persistence_file}")
            
        except Exception as e:
            logger.error(f"Error loading persisted events: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get event processing statistics"""
        queue_size = await self.event_queue.size()
        
        return {
            "queue_size": queue_size,
            "processing_events": len(self.processing_events),
            "completed_events": len(self.completed_events),
            "failed_events": len(self.failed_events),
            "processing_active": self.processing_active,
            "total_events": queue_size + len(self.processing_events) + len(self.completed_events) + len(self.failed_events)
        }
    
    async def cleanup_old_events(self, max_age_hours: int = 24) -> int:
        """Clean up old completed events"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        # Clean completed events
        completed_to_remove = [
            event_id for event_id, event in self.completed_events.items()
            if event.timestamp < cutoff_time
        ]
        
        for event_id in completed_to_remove:
            del self.completed_events[event_id]
            cleaned_count += 1
        
        # Clean old failed events that exceeded retries
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
        stats = await self.get_statistics()
        return {
            "status": "active" if self.processing_active else "inactive",
            "persistence_enabled": self.persistence_enabled,
            "stats": stats
        }


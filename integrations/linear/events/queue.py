"""
Event Queue

Thread-safe event queue implementation for webhook event processing.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from collections import deque

from ..interfaces import WebhookEvent

logger = logging.getLogger(__name__)


class EventQueue:
    """Thread-safe event queue for webhook events"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = deque()
        self.event_lookup: Dict[str, WebhookEvent] = {}
        self._lock = asyncio.Lock()
        
        logger.debug(f"EventQueue initialized with max_size: {max_size}")
    
    async def enqueue(self, event: WebhookEvent) -> bool:
        """Add event to queue"""
        async with self._lock:
            if len(self.queue) >= self.max_size:
                logger.warning("Event queue is full, dropping oldest event")
                # Remove oldest event
                oldest_event = self.queue.popleft()
                if oldest_event.id in self.event_lookup:
                    del self.event_lookup[oldest_event.id]
            
            # Add new event
            self.queue.append(event)
            self.event_lookup[event.id] = event
            
            logger.debug(f"Enqueued event: {event.id}, queue size: {len(self.queue)}")
            return True
    
    async def dequeue(self) -> Optional[WebhookEvent]:
        """Remove and return event from queue"""
        async with self._lock:
            if not self.queue:
                return None
            
            event = self.queue.popleft()
            if event.id in self.event_lookup:
                del self.event_lookup[event.id]
            
            logger.debug(f"Dequeued event: {event.id}, queue size: {len(self.queue)}")
            return event
    
    async def peek(self) -> Optional[WebhookEvent]:
        """Peek at next event without removing it"""
        async with self._lock:
            if not self.queue:
                return None
            return self.queue[0]
    
    async def size(self) -> int:
        """Get current queue size"""
        async with self._lock:
            return len(self.queue)
    
    async def is_empty(self) -> bool:
        """Check if queue is empty"""
        async with self._lock:
            return len(self.queue) == 0
    
    async def is_full(self) -> bool:
        """Check if queue is full"""
        async with self._lock:
            return len(self.queue) >= self.max_size
    
    async def contains(self, event_id: str) -> bool:
        """Check if event is in queue"""
        async with self._lock:
            return event_id in self.event_lookup
    
    async def get_position(self, event_id: str) -> Optional[int]:
        """Get position of event in queue (0-based)"""
        async with self._lock:
            if event_id not in self.event_lookup:
                return None
            
            for i, event in enumerate(self.queue):
                if event.id == event_id:
                    return i
            
            return None
    
    async def remove(self, event_id: str) -> bool:
        """Remove specific event from queue"""
        async with self._lock:
            if event_id not in self.event_lookup:
                return False
            
            # Find and remove event
            for i, event in enumerate(self.queue):
                if event.id == event_id:
                    del self.queue[i]
                    del self.event_lookup[event_id]
                    logger.debug(f"Removed event: {event_id}, queue size: {len(self.queue)}")
                    return True
            
            return False
    
    async def clear(self) -> int:
        """Clear all events from queue"""
        async with self._lock:
            count = len(self.queue)
            self.queue.clear()
            self.event_lookup.clear()
            logger.debug(f"Cleared queue, removed {count} events")
            return count
    
    async def get_events_by_type(self, event_type: str) -> list:
        """Get all events of specific type"""
        async with self._lock:
            return [event for event in self.queue if event.type.value == event_type]
    
    async def get_oldest_event(self) -> Optional[WebhookEvent]:
        """Get oldest event in queue"""
        async with self._lock:
            if not self.queue:
                return None
            return self.queue[0]
    
    async def get_newest_event(self) -> Optional[WebhookEvent]:
        """Get newest event in queue"""
        async with self._lock:
            if not self.queue:
                return None
            return self.queue[-1]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            if not self.queue:
                return {
                    "size": 0,
                    "max_size": self.max_size,
                    "utilization": 0.0,
                    "event_types": {}
                }
            
            # Count event types
            event_types = {}
            for event in self.queue:
                event_type = event.type.value
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            return {
                "size": len(self.queue),
                "max_size": self.max_size,
                "utilization": len(self.queue) / self.max_size,
                "event_types": event_types,
                "oldest_event_age": (
                    (self.queue[0].timestamp - self.queue[0].timestamp).total_seconds()
                    if self.queue else 0
                )
            }


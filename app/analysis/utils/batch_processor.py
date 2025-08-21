"""
Batch processing utilities for Analysis Service.

Efficient batch processing with concurrency control and error handling.
"""

import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional, Awaitable
from datetime import datetime
from dataclasses import dataclass, field

from .common_types import ContentPriority, ProcessingStatus


@dataclass
class BatchItem:
    """Individual item in a processing batch."""
    item_id: str
    data: Dict[str, Any]
    priority: ContentPriority = ContentPriority.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    status: ProcessingStatus = ProcessingStatus.PENDING
    error: Optional[str] = None
    result: Optional[Any] = None
    processing_time_ms: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now())


@dataclass
class BatchResult:
    """Result from batch processing."""
    batch_id: str
    total_items: int
    successful: int
    failed: int
    skipped: int
    total_processing_time_ms: int
    items: List[BatchItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now())
    completed_at: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_items == 0:
            return 0.0
        return self.successful / self.total_items


class AnalysisBatchProcessor:
    """Batch processor with concurrency control and error handling."""
    
    def __init__(
        self,
        max_concurrent: int = 5,
        default_timeout: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.logger = logger or logging.getLogger(__name__)
        
        # Processing statistics
        self._total_batches = 0
        self._total_items = 0
        self._successful_items = 0
        self._failed_items = 0
        
    async def process_batch(
        self,
        batch_id: str,
        items: List[Dict[str, Any]],
        processor_func: Callable[[Dict[str, Any]], Awaitable[Any]],
        priority_func: Optional[Callable[[Dict[str, Any]], ContentPriority]] = None,
        timeout: Optional[int] = None
    ) -> BatchResult:
        """Process a batch of items with concurrency control."""
        timeout = timeout or self.default_timeout
        
        # Create batch items
        batch_items = []
        for i, item_data in enumerate(items):
            priority = ContentPriority.MEDIUM
            if priority_func:
                try:
                    priority = priority_func(item_data)
                except Exception as e:
                    self.logger.warning(f"Priority function failed for item {i}: {e}")
                    
            batch_item = BatchItem(
                item_id=f"{batch_id}_{i}",
                data=item_data,
                priority=priority
            )
            batch_items.append(batch_item)
        
        # Sort by priority (critical first)
        batch_items.sort(
            key=lambda x: (
                0 if x.priority == ContentPriority.CRITICAL else
                1 if x.priority == ContentPriority.HIGH else
                2 if x.priority == ContentPriority.MEDIUM else 3
            )
        )
        
        # Create batch result
        result = BatchResult(
            batch_id=batch_id,
            total_items=len(batch_items),
            successful=0,
            failed=0,
            skipped=0,
            total_processing_time_ms=0,
            items=batch_items
        )
        
        # Process items with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_item(item: BatchItem):
            """Process individual item with error handling."""
            async with semaphore:
                start_time = asyncio.get_event_loop().time()
                
                try:
                    # Process with timeout
                    item.status = ProcessingStatus.IN_PROGRESS
                    item.result = await asyncio.wait_for(
                        processor_func(item.data),
                        timeout=timeout
                    )
                    item.status = ProcessingStatus.COMPLETED
                    result.successful += 1
                    
                except asyncio.TimeoutError:
                    item.error = f"Processing timeout after {timeout}s"
                    item.status = ProcessingStatus.FAILED
                    result.failed += 1
                    self.logger.warning(f"Item {item.item_id} timed out")
                    
                except Exception as e:
                    item.error = str(e)
                    item.status = ProcessingStatus.FAILED
                    result.failed += 1
                    self.logger.error(f"Item {item.item_id} failed: {e}")
                    
                finally:
                    end_time = asyncio.get_event_loop().time()
                    item.processing_time_ms = int((end_time - start_time) * 1000)
                    result.total_processing_time_ms += item.processing_time_ms
        
        # Process all items
        self.logger.info(f"Processing batch {batch_id} with {len(batch_items)} items")
        start_time = asyncio.get_event_loop().time()
        
        await asyncio.gather(
            *[process_item(item) for item in batch_items],
            return_exceptions=True
        )
        
        end_time = asyncio.get_event_loop().time()
        result.completed_at = datetime.now()
        
        # Update statistics
        self._total_batches += 1
        self._total_items += result.total_items
        self._successful_items += result.successful
        self._failed_items += result.failed
        
        self.logger.info(
            f"Batch {batch_id} completed: {result.successful}/{result.total_items} successful "
            f"({result.success_rate:.1%}) in {result.total_processing_time_ms}ms"
        )
        
        return result
    
    async def process_with_retries(
        self,
        batch_id: str,
        items: List[Dict[str, Any]],
        processor_func: Callable[[Dict[str, Any]], Awaitable[Any]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> BatchResult:
        """Process batch with automatic retries for failed items."""
        current_items = items.copy()
        attempt = 0
        
        while attempt <= max_retries and current_items:
            retry_suffix = f"_retry_{attempt}" if attempt > 0 else ""
            current_batch_id = f"{batch_id}{retry_suffix}"
            
            result = await self.process_batch(
                current_batch_id,
                current_items,
                processor_func,
                **kwargs
            )
            
            if attempt == max_retries:
                # Final attempt, return results
                return result
                
            # Prepare failed items for retry
            failed_items = [
                item.data for item in result.items
                if item.status == ProcessingStatus.FAILED
            ]
            
            if not failed_items:
                # All items succeeded
                return result
                
            self.logger.info(
                f"Retrying {len(failed_items)} failed items from batch {batch_id} "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            
            # Wait before retry
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)
                
            current_items = failed_items
            attempt += 1
            
        return result
    
    def create_priority_func(
        self,
        priority_rules: Dict[str, ContentPriority]
    ) -> Callable[[Dict[str, Any]], ContentPriority]:
        """Create a priority function based on rules."""
        def priority_func(item: Dict[str, Any]) -> ContentPriority:
            # Check rules in order
            for field, priority in priority_rules.items():
                if field in item and item[field]:
                    return priority
            return ContentPriority.MEDIUM
            
        return priority_func
    
    def filter_by_priority(
        self,
        items: List[Dict[str, Any]],
        min_priority: ContentPriority,
        priority_func: Optional[Callable[[Dict[str, Any]], ContentPriority]] = None
    ) -> List[Dict[str, Any]]:
        """Filter items by minimum priority level."""
        if not priority_func:
            return items  # No filtering
            
        priority_order = {
            ContentPriority.CRITICAL: 0,
            ContentPriority.HIGH: 1,
            ContentPriority.MEDIUM: 2,
            ContentPriority.LOW: 3
        }
        
        min_level = priority_order[min_priority]
        filtered_items = []
        
        for item in items:
            try:
                item_priority = priority_func(item)
                if priority_order[item_priority] <= min_level:
                    filtered_items.append(item)
            except Exception as e:
                self.logger.warning(f"Priority check failed for item: {e}")
                # Include item if priority check fails
                filtered_items.append(item)
                
        return filtered_items
    
    async def process_prioritized_batch(
        self,
        batch_id: str,
        items: List[Dict[str, Any]],
        processor_func: Callable[[Dict[str, Any]], Awaitable[Any]],
        critical_first: bool = True,
        **kwargs
    ) -> List[BatchResult]:
        """Process batch with priority-based splitting."""
        if not critical_first:
            # Process as single batch
            result = await self.process_batch(
                batch_id,
                items,
                processor_func,
                **kwargs
            )
            return [result]
        
        # Split by priority
        priority_groups = {
            ContentPriority.CRITICAL: [],
            ContentPriority.HIGH: [],
            ContentPriority.MEDIUM: [],
            ContentPriority.LOW: []
        }
        
        priority_func = kwargs.get('priority_func')
        if priority_func:
            for item in items:
                try:
                    priority = priority_func(item)
                    priority_groups[priority].append(item)
                except Exception:
                    priority_groups[ContentPriority.MEDIUM].append(item)
        else:
            priority_groups[ContentPriority.MEDIUM] = items
        
        # Process priority groups in order
        results = []
        for priority in [ContentPriority.CRITICAL, ContentPriority.HIGH, 
                        ContentPriority.MEDIUM, ContentPriority.LOW]:
            
            if priority_groups[priority]:
                group_batch_id = f"{batch_id}_{priority.value}"
                result = await self.process_batch(
                    group_batch_id,
                    priority_groups[priority],
                    processor_func,
                    **kwargs
                )
                results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        success_rate = (
            self._successful_items / self._total_items
            if self._total_items > 0 else 0.0
        )
        
        return {
            "total_batches": self._total_batches,
            "total_items": self._total_items,
            "successful_items": self._successful_items,
            "failed_items": self._failed_items,
            "success_rate": success_rate,
            "average_items_per_batch": (
                self._total_items / self._total_batches
                if self._total_batches > 0 else 0
            )
        }
    
    def reset_stats(self):
        """Reset processing statistics."""
        self._total_batches = 0
        self._total_items = 0
        self._successful_items = 0
        self._failed_items = 0
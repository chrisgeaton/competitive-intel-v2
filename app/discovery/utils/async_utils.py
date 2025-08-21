"""
Async utility functions for optimized batch processing and concurrent operations.
Standardizes async patterns across the discovery service.
"""

import asyncio
import logging
from typing import List, Callable, Awaitable, TypeVar, Optional, Any, Dict
from dataclasses import dataclass
from datetime import datetime

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    successful: List[U]
    failed: List[Exception]
    total_processed: int
    processing_time: float
    success_rate: float


class AsyncBatchProcessor:
    """Efficient batch processing for async operations with error handling."""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5, 
                 timeout: float = 30.0, logger: Optional[logging.Logger] = None):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logger or logging.getLogger("discovery.batch_processor")
    
    async def process_batch(self, items: List[T], 
                          processor_func: Callable[[T], Awaitable[U]],
                          retry_failures: bool = True,
                          max_retries: int = 2) -> BatchResult[U]:
        """Process items in concurrent batches with comprehensive error handling."""
        start_time = datetime.now()
        all_successful = []
        all_failed = []
        
        # Process items in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Process batch with concurrency control
            successful, failed = await self._process_single_batch(
                batch, processor_func, retry_failures, max_retries
            )
            
            all_successful.extend(successful)
            all_failed.extend(failed)
            
            # Log progress for large batches
            if len(items) > 50:
                processed = min(i + self.batch_size, len(items))
                self.logger.debug(f"Processed {processed}/{len(items)} items")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        success_rate = len(all_successful) / len(items) if items else 0.0
        
        return BatchResult(
            successful=all_successful,
            failed=all_failed,
            total_processed=len(items),
            processing_time=processing_time,
            success_rate=success_rate
        )
    
    async def _process_single_batch(self, batch: List[T],
                                  processor_func: Callable[[T], Awaitable[U]],
                                  retry_failures: bool,
                                  max_retries: int) -> tuple[List[U], List[Exception]]:
        """Process a single batch with concurrency control."""
        
        async def process_with_semaphore_and_retry(item: T) -> tuple[Optional[U], Optional[Exception]]:
            """Process single item with semaphore and retry logic."""
            async with self.semaphore:
                for attempt in range(max_retries + 1):
                    try:
                        # Apply timeout to individual operations
                        result = await asyncio.wait_for(
                            processor_func(item), 
                            timeout=self.timeout
                        )
                        return result, None
                        
                    except asyncio.TimeoutError as e:
                        if attempt == max_retries:
                            return None, e
                        await asyncio.sleep(0.5 * (attempt + 1))  # Brief backoff
                        
                    except Exception as e:
                        if not retry_failures or attempt == max_retries:
                            return None, e
                        await asyncio.sleep(0.5 * (attempt + 1))  # Brief backoff
                
                return None, Exception("Max retries exceeded")
        
        # Create tasks for all items in batch
        tasks = [process_with_semaphore_and_retry(item) for item in batch]
        
        # Wait for all tasks to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return [], [e] * len(batch)
        
        # Separate successful results from failures
        successful = []
        failed = []
        
        for result, error in results:
            if error is None and result is not None:
                successful.append(result)
            else:
                failed.append(error or Exception("Unknown processing error"))
        
        return successful, failed
    
    async def process_with_callback(self, items: List[T],
                                  processor_func: Callable[[T], Awaitable[U]],
                                  progress_callback: Optional[Callable[[int, int], None]] = None) -> BatchResult[U]:
        """Process items with progress callback for UI updates."""
        start_time = datetime.now()
        all_successful = []
        all_failed = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            successful, failed = await self._process_single_batch(batch, processor_func, True, 2)
            
            all_successful.extend(successful)
            all_failed.extend(failed)
            
            # Call progress callback if provided
            if progress_callback:
                processed_count = min(i + self.batch_size, len(items))
                progress_callback(processed_count, len(items))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        success_rate = len(all_successful) / len(items) if items else 0.0
        
        return BatchResult(
            successful=all_successful,
            failed=all_failed,
            total_processed=len(items),
            processing_time=processing_time,
            success_rate=success_rate
        )


class AsyncRetryHandler:
    """Standardized async retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_factor = exponential_factor
        self.logger = logging.getLogger("discovery.retry_handler")
    
    async def retry_async(self, operation: Callable[[], Awaitable[T]], 
                         operation_name: str = "operation",
                         retryable_exceptions: tuple = (Exception,)) -> T:
        """Retry an async operation with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await operation()
                if attempt > 0:
                    self.logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not isinstance(e, retryable_exceptions):
                    self.logger.error(f"{operation_name} failed with non-retryable error: {e}")
                    raise
                
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.exponential_factor ** attempt),
                        self.max_delay
                    )
                    
                    self.logger.warning(
                        f"{operation_name} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"{operation_name} failed after {self.max_retries + 1} attempts: {e}"
                    )
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"{operation_name} failed after all retries")


class AsyncThrottler:
    """Rate limiting for async operations."""
    
    def __init__(self, rate_limit: float, time_window: float = 1.0):
        """
        Initialize throttler.
        
        Args:
            rate_limit: Maximum number of operations per time window
            time_window: Time window in seconds (default 1 second)
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.semaphore = asyncio.Semaphore(int(rate_limit))
        self.call_times: List[float] = []
        self.lock = asyncio.Lock()
    
    async def throttle(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute operation with rate limiting."""
        async with self.semaphore:
            # Clean up old call times
            now = asyncio.get_event_loop().time()
            async with self.lock:
                self.call_times = [t for t in self.call_times if now - t < self.time_window]
                
                # Check if we need to wait
                if len(self.call_times) >= self.rate_limit:
                    oldest_call = min(self.call_times)
                    wait_time = self.time_window - (now - oldest_call)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                
                # Record this call
                self.call_times.append(now)
            
            # Execute operation
            return await operation()


class AsyncTimeoutManager:
    """Manages timeouts for async operations with context."""
    
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
        self.logger = logging.getLogger("discovery.timeout_manager")
    
    async def with_timeout(self, operation: Callable[[], Awaitable[T]], 
                          timeout: Optional[float] = None,
                          operation_name: str = "operation") -> T:
        """Execute operation with timeout and proper error handling."""
        actual_timeout = timeout or self.default_timeout
        
        try:
            start_time = datetime.now()
            result = await asyncio.wait_for(operation(), timeout=actual_timeout)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > actual_timeout * 0.8:  # Warn if operation took >80% of timeout
                self.logger.warning(
                    f"{operation_name} took {elapsed:.1f}s (timeout: {actual_timeout}s)"
                )
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"{operation_name} timed out after {actual_timeout}s")
            raise
        except Exception as e:
            self.logger.error(f"{operation_name} failed: {e}")
            raise


# Global utility instances for convenience
batch_processor = AsyncBatchProcessor()
retry_handler = AsyncRetryHandler()
timeout_manager = AsyncTimeoutManager()
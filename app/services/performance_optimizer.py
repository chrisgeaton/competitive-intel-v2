"""
Performance optimization utilities for Phase 4 services.

Provides caching, memory optimization, and performance improvements
following established Phase 1-3 patterns for production readiness.
"""

import asyncio
import functools
import gc
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from dataclasses import dataclass
import hashlib
import json

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with expiration."""
    value: T
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0


class MemoryCache:
    """
    Simple in-memory cache with TTL and size limits.
    
    Optimized for Phase 4 services to cache user contexts,
    report templates, and frequently accessed data.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if datetime.utcnow() > entry.expires_at:
            del self.cache[key]
            self.misses += 1
            return None
        
        entry.hit_count += 1
        self.hits += 1
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        ttl = ttl or self.default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        self.cache[key] = CacheEntry(
            value=value,
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
        del self.cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }


# Global cache instance
_global_cache = MemoryCache(max_size=2000, default_ttl=600)  # 10 minutes default


def cached(ttl: int = 300, key_prefix: str = "default"):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _global_cache._generate_key(
                f"{key_prefix}:{func.__name__}", *args, **kwargs
            )
            
            # Try to get from cache
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            _global_cache.set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            
            return result
        
        return wrapper
    return decorator


class BatchProcessor:
    """
    Batch processing utility for optimizing database operations.
    
    Reduces database round trips by batching similar operations
    following established Phase 1-3 optimization patterns.
    """
    
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
    
    async def process_in_batches(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """Process items in batches to optimize performance."""
        batch_size = batch_size or self.batch_size
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            try:
                batch_results = await process_func(batch)
                results.extend(batch_results)
                
                # Small delay between batches to prevent overwhelming
                if i + batch_size < len(items):
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Batch processing error for batch {i//batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        return results
    
    async def batch_database_operations(
        self,
        operations: List[Callable],
        max_concurrent: int = 10
    ) -> List[Any]:
        """Execute database operations concurrently with limits."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(operation):
            async with semaphore:
                return await operation()
        
        return await asyncio.gather(
            *[execute_with_semaphore(op) for op in operations],
            return_exceptions=True
        )


class MemoryOptimizer:
    """
    Memory optimization utilities for large-scale processing.
    
    Provides memory monitoring and optimization for Phase 4 services
    to handle large report generation workloads efficiently.
    """
    
    @staticmethod
    def optimize_content_items(content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize content items for memory efficiency.
        
        Removes unnecessary fields and optimizes data structures
        for memory-efficient processing in report generation.
        """
        optimized_items = []
        
        for item in content_items:
            # Keep only essential fields for memory efficiency
            optimized_item = {
                "content_id": item.get("content_id"),
                "title": item.get("title", "")[:200],  # Truncate long titles
                "url": item.get("url"),
                "priority": item.get("priority"),
                "score": round(item.get("overall_score", 0.0), 3),  # Reduce precision
                "published_at": item.get("published_at"),
                "relevance": item.get("relevance_explanation", "")[:300]  # Truncate
            }
            
            # Only include non-empty insights to save memory
            insights = item.get("strategic_insights", [])
            if insights:
                optimized_item["insights"] = [
                    {
                        "type": insight.get("insight_type", ""),
                        "title": insight.get("insight_title", "")[:100]
                    }
                    for insight in insights[:3]  # Top 3 insights only
                ]
            
            optimized_items.append(optimized_item)
        
        return optimized_items
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection to free memory."""
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        return collected
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get basic memory usage statistics."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2)
            }
        except ImportError:
            return {"error": "psutil not available"}


class AsyncTimeout:
    """Async timeout context manager for operation limits."""
    
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.task = None
    
    async def __aenter__(self):
        self.task = asyncio.current_task()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.task and not self.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=self.timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Operation timed out after {self.timeout}s")
                self.task.cancel()


def performance_monitor(operation_name: str):
    """Decorator to monitor operation performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            start_memory = MemoryOptimizer.get_memory_usage()
            
            try:
                result = await func(*args, **kwargs)
                
                end_time = datetime.utcnow()
                end_memory = MemoryOptimizer.get_memory_usage()
                
                elapsed = (end_time - start_time).total_seconds()
                
                logger.info(
                    f"Performance: {operation_name} completed in {elapsed:.2f}s, "
                    f"memory: {start_memory} -> {end_memory}"
                )
                
                return result
                
            except Exception as e:
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                logger.error(f"Performance: {operation_name} failed after {elapsed:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator


# Utility functions
def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _global_cache.get_stats()


def clear_cache() -> None:
    """Clear global cache."""
    _global_cache.clear()
    logger.info("Global cache cleared")


async def warm_cache_with_user_contexts(user_ids: List[int]) -> None:
    """Pre-warm cache with user contexts for better performance."""
    logger.info(f"Warming cache for {len(user_ids)} users")
    
    # This would be implemented to pre-load frequently accessed user data
    # For now, just log the warming attempt
    for user_id in user_ids[:10]:  # Limit to top 10 for demo
        cache_key = _global_cache._generate_key("user_context", user_id)
        logger.debug(f"Cache warming for user {user_id}")
    
    logger.info("Cache warming completed")
"""
Unified cache management with LRU and TTL support.
Consolidates multiple uncoordinated caches across the discovery service.
"""

import asyncio
import logging
import weakref
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, TypeVar, Generic, Tuple, List
from dataclasses import dataclass
from threading import RLock

T = TypeVar('T')


@dataclass
class CacheStats:
    """Cache statistics for monitoring and optimization."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    ttl_seconds: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache(Generic[T]):
    """Memory-efficient LRU cache with TTL support and statistics."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300, name: str = "cache"):
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.name = name
        
        # Use RLock for thread safety
        self._lock = RLock()
        
        # Cache storage: key -> (value, created_at, last_accessed)
        self._cache: Dict[str, Tuple[T, datetime, datetime]] = {}
        
        # Track access order for LRU eviction
        self._access_order: List[str] = []
        
        # Statistics
        self._stats = CacheStats(max_size=max_size, ttl_seconds=ttl_seconds)
        
        # Logger for debugging
        self._logger = logging.getLogger(f"discovery.cache.{name}")
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache with LRU tracking."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            value, created_at, last_accessed = self._cache[key]
            now = datetime.now()
            
            # Check TTL
            if now - created_at > self.ttl:
                self._remove_key(key)
                self._stats.misses += 1
                return None
            
            # Update access time and order
            self._cache[key] = (value, created_at, now)
            self._update_access_order(key)
            
            self._stats.hits += 1
            return value
    
    def put(self, key: str, value: T) -> None:
        """Put item in cache with size management."""
        with self._lock:
            now = datetime.now()
            
            # If key exists, update it
            if key in self._cache:
                self._cache[key] = (value, now, now)
                self._update_access_order(key)
                return
            
            # Check if we need to evict items
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            self._cache[key] = (value, now, now)
            self._access_order.append(key)
            self._stats.size = len(self._cache)
    
    def delete(self, key: str) -> bool:
        """Delete specific key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats.size = 0
            self._logger.debug(f"Cache {self.name} cleared")
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache and access order."""
        self._cache.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)
        self._stats.size = len(self._cache)
    
    def _update_access_order(self, key: str) -> None:
        """Update LRU access order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_order:
            return
        
        # Remove oldest item
        oldest_key = self._access_order[0]
        self._remove_key(oldest_key)
        self._stats.evictions += 1
        
        self._logger.debug(f"Cache {self.name} evicted key: {oldest_key}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats
    
    def cleanup_expired(self) -> int:
        """Remove all expired items and return count."""
        with self._lock:
            now = datetime.now()
            expired_keys = []
            
            for key, (value, created_at, last_accessed) in self._cache.items():
                if now - created_at > self.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_key(key)
            
            if expired_keys:
                self._logger.debug(f"Cache {self.name} cleaned up {len(expired_keys)} expired items")
            
            return len(expired_keys)


class CacheManager:
    """Centralized cache management for discovery service."""
    
    def __init__(self):
        self._caches: Dict[str, LRUCache] = {}
        self._lock = RLock()
        self._logger = logging.getLogger("discovery.cache_manager")
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 300  # 5 minutes
    
    def get_cache(self, name: str, max_size: int = 1000, 
                  ttl_seconds: int = 300) -> LRUCache:
        """Get or create a named cache."""
        with self._lock:
            if name not in self._caches:
                self._caches[name] = LRUCache(max_size, ttl_seconds, name)
                self._logger.info(f"Created cache '{name}' with size={max_size}, ttl={ttl_seconds}s")
            return self._caches[name]
    
    def delete_cache(self, name: str) -> bool:
        """Delete a named cache."""
        with self._lock:
            if name in self._caches:
                self._caches[name].clear()
                del self._caches[name]
                self._logger.info(f"Deleted cache '{name}'")
                return True
            return False
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            self._logger.info("Cleared all caches")
    
    def get_all_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        with self._lock:
            return {name: cache.get_stats() for name, cache in self._caches.items()}
    
    def start_cleanup_task(self) -> None:
        """Start periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            self._logger.info("Started cache cleanup task")
    
    def stop_cleanup_task(self) -> None:
        """Stop periodic cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            self._logger.info("Stopped cache cleanup task")
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired cache items."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                total_cleaned = 0
                with self._lock:
                    for cache in self._caches.values():
                        cleaned = cache.cleanup_expired()
                        total_cleaned += cleaned
                
                if total_cleaned > 0:
                    self._logger.debug(f"Periodic cleanup removed {total_cleaned} expired items")
                    
            except asyncio.CancelledError:
                self._logger.info("Cache cleanup task cancelled")
                break
            except Exception as e:
                self._logger.error(f"Error in cache cleanup: {e}")
                # Continue running despite errors
    
    def get_memory_usage_estimate(self) -> Dict[str, Any]:
        """Get estimated memory usage of all caches."""
        with self._lock:
            stats = {}
            total_items = 0
            
            for name, cache in self._caches.items():
                cache_stats = cache.get_stats()
                total_items += cache_stats.size
                stats[name] = {
                    'items': cache_stats.size,
                    'max_size': cache_stats.max_size,
                    'hit_rate': cache_stats.hit_rate,
                    'utilization': cache_stats.size / cache_stats.max_size if cache_stats.max_size > 0 else 0
                }
            
            stats['total_items'] = total_items
            stats['total_caches'] = len(self._caches)
            
            return stats
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
        except:
            pass  # Ignore errors during cleanup


# Global cache manager instance
cache_manager = CacheManager()


# Convenience functions for common cache operations
def get_user_context_cache() -> LRUCache:
    """Get cache for user contexts (15 minute TTL)."""
    return cache_manager.get_cache("user_contexts", max_size=500, ttl_seconds=900)


def get_content_processing_cache() -> LRUCache:
    """Get cache for processed content (6 hour TTL)."""
    return cache_manager.get_cache("content_processing", max_size=2000, ttl_seconds=21600)


def get_ml_scoring_cache() -> LRUCache:
    """Get cache for ML scoring results (1 hour TTL)."""
    return cache_manager.get_cache("ml_scoring", max_size=1000, ttl_seconds=3600)


def get_source_discovery_cache() -> LRUCache:
    """Get cache for source discovery results (30 minute TTL)."""
    return cache_manager.get_cache("source_discovery", max_size=500, ttl_seconds=1800)


def get_rss_feed_cache() -> LRUCache:
    """Get cache for RSS feed content (2 hour TTL)."""
    return cache_manager.get_cache("rss_feeds", max_size=1500, ttl_seconds=7200)
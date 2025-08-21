"""
Caching utilities for Analysis Service.

Memory-efficient caching with TTL and LRU eviction policies.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import threading


class TTLCache:
    """Time-to-live cache implementation."""
    
    def __init__(self, default_ttl_seconds: int = 900):  # 15 minutes default
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._default_ttl = default_ttl_seconds
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if time.time() < expires_at:
                    return value
                else:
                    # Remove expired entry
                    del self._cache[key]
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache with TTL."""
        ttl = ttl_seconds or self._default_ttl
        expires_at = time.time() + ttl
        
        with self._lock:
            self._cache[key] = (value, expires_at)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, (value, expires_at) in self._cache.items():
                if current_time >= expires_at:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                
        return len(expired_keys)
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = 0
        
        with self._lock:
            for _, (value, expires_at) in self._cache.items():
                if current_time >= expires_at:
                    expired_count += 1
                    
        return {
            "total_entries": len(self._cache),
            "expired_entries": expired_count,
            "active_entries": len(self._cache) - expired_count,
            "default_ttl_seconds": self._default_ttl
        }


class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value and move to end (most recently used)."""
        with self._lock:
            if key in self._cache:
                # Move to end
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    def set(self, key: str, value: Any):
        """Set value and manage cache size."""
        with self._lock:
            if key in self._cache:
                # Update existing key
                self._cache[key] = value
                self._cache.move_to_end(key)
            else:
                # Add new key
                self._cache[key] = value
                if len(self._cache) > self._max_size:
                    # Remove least recently used item
                    self._cache.popitem(last=False)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def keys(self) -> list:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())


class AnalysisCacheManager:
    """Centralized cache manager for Analysis Service."""
    
    def __init__(self):
        # Different caches for different data types
        self.user_context_cache = TTLCache(default_ttl_seconds=900)  # 15 min
        self.content_features_cache = LRUCache(max_size=2000)
        self.filter_results_cache = TTLCache(default_ttl_seconds=3600)  # 1 hour
        self.cost_estimates_cache = TTLCache(default_ttl_seconds=1800)  # 30 min
        
        # Cache hit/miss statistics
        self._stats = {
            "user_context": {"hits": 0, "misses": 0},
            "content_features": {"hits": 0, "misses": 0},
            "filter_results": {"hits": 0, "misses": 0},
            "cost_estimates": {"hits": 0, "misses": 0}
        }
    
    def get_user_context(self, user_id: int) -> Optional[Any]:
        """Get cached user context."""
        key = f"user_context_{user_id}"
        result = self.user_context_cache.get(key)
        
        if result is not None:
            self._stats["user_context"]["hits"] += 1
        else:
            self._stats["user_context"]["misses"] += 1
            
        return result
    
    def set_user_context(self, user_id: int, context: Any, ttl_seconds: int = 900):
        """Cache user context."""
        key = f"user_context_{user_id}"
        self.user_context_cache.set(key, context, ttl_seconds)
    
    def get_content_features(self, content_hash: str) -> Optional[Any]:
        """Get cached content features."""
        result = self.content_features_cache.get(content_hash)
        
        if result is not None:
            self._stats["content_features"]["hits"] += 1
        else:
            self._stats["content_features"]["misses"] += 1
            
        return result
    
    def set_content_features(self, content_hash: str, features: Any):
        """Cache content features."""
        self.content_features_cache.set(content_hash, features)
    
    def get_filter_result(self, filter_key: str) -> Optional[Any]:
        """Get cached filter result."""
        result = self.filter_results_cache.get(filter_key)
        
        if result is not None:
            self._stats["filter_results"]["hits"] += 1
        else:
            self._stats["filter_results"]["misses"] += 1
            
        return result
    
    def set_filter_result(self, filter_key: str, result: Any):
        """Cache filter result."""
        self.filter_results_cache.set(filter_key, result)
    
    def get_cost_estimate(self, estimate_key: str) -> Optional[Any]:
        """Get cached cost estimate."""
        result = self.cost_estimates_cache.get(estimate_key)
        
        if result is not None:
            self._stats["cost_estimates"]["hits"] += 1
        else:
            self._stats["cost_estimates"]["misses"] += 1
            
        return result
    
    def set_cost_estimate(self, estimate_key: str, estimate: Any):
        """Cache cost estimate."""
        self.cost_estimates_cache.set(estimate_key, estimate)
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup expired entries from all caches."""
        cleanup_results = {
            "user_context_expired": self.user_context_cache.cleanup_expired(),
            "filter_results_expired": self.filter_results_cache.cleanup_expired(),
            "cost_estimates_expired": self.cost_estimates_cache.cleanup_expired()
        }
        return cleanup_results
    
    def clear_all(self):
        """Clear all caches."""
        self.user_context_cache.clear()
        self.content_features_cache.clear()
        self.filter_results_cache.clear()
        self.cost_estimates_cache.clear()
        
        # Reset stats
        for cache_stats in self._stats.values():
            cache_stats["hits"] = 0
            cache_stats["misses"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "cache_stats": self._stats.copy(),
            "cache_sizes": {
                "user_context": self.user_context_cache.size(),
                "content_features": self.content_features_cache.size(),
                "filter_results": self.filter_results_cache.size(),
                "cost_estimates": self.cost_estimates_cache.size()
            }
        }
        
        # Calculate hit rates
        for cache_name, cache_stats in stats["cache_stats"].items():
            total_requests = cache_stats["hits"] + cache_stats["misses"]
            if total_requests > 0:
                cache_stats["hit_rate"] = cache_stats["hits"] / total_requests
            else:
                cache_stats["hit_rate"] = 0.0
                
        return stats
    
    def generate_content_hash(self, content: Dict[str, Any]) -> str:
        """Generate hash for content caching."""
        import hashlib
        
        # Create hash from content ID and key fields
        key_fields = [
            str(content.get("id", "")),
            content.get("title", ""),
            content.get("content_text", "")[:100]  # First 100 chars
        ]
        
        content_string = "|".join(key_fields)
        return hashlib.md5(content_string.encode()).hexdigest()[:16]
    
    def generate_filter_key(self, content_id: int, filter_config: Dict[str, Any]) -> str:
        """Generate key for filter result caching."""
        import hashlib
        import json
        
        # Create hash from content ID and filter configuration
        key_data = {
            "content_id": content_id,
            "config": filter_config
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
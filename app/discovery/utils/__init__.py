"""
Discovery service utilities for code optimization and shared functionality.
"""

from .session_manager import AsyncSessionManager
from .error_handler import UnifiedErrorHandler, EngineException, EngineErrorType
from .content_utils import ContentUtils
from .cache_manager import CacheManager, LRUCache, cache_manager
from .async_utils import AsyncBatchProcessor
from .config import DiscoveryConfig

# Cache instances
_config_instance = None
_content_processing_cache = None
_ml_scoring_cache = None
_source_discovery_cache = None

def get_config() -> DiscoveryConfig:
    """Get singleton discovery configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = DiscoveryConfig()
    return _config_instance

def get_content_processing_cache() -> LRUCache:
    """Get content processing cache instance."""
    global _content_processing_cache
    if _content_processing_cache is None:
        _content_processing_cache = LRUCache(max_size=1000)
    return _content_processing_cache

def get_ml_scoring_cache() -> LRUCache:
    """Get ML scoring cache instance."""
    global _ml_scoring_cache
    if _ml_scoring_cache is None:
        _ml_scoring_cache = LRUCache(max_size=5000)
    return _ml_scoring_cache

def get_source_discovery_cache() -> LRUCache:
    """Get source discovery cache instance."""
    global _source_discovery_cache
    if _source_discovery_cache is None:
        _source_discovery_cache = LRUCache(max_size=500)
    return _source_discovery_cache

def get_user_context_cache() -> LRUCache:
    """Get user context cache instance."""
    return get_content_processing_cache()  # Reuse the same cache

# Create singleton instances
batch_processor = AsyncBatchProcessor()

__all__ = [
    'AsyncSessionManager',
    'UnifiedErrorHandler', 
    'EngineException',
    'EngineErrorType',
    'ContentUtils',
    'CacheManager',
    'LRUCache', 
    'cache_manager',
    'AsyncBatchProcessor',
    'batch_processor',
    'DiscoveryConfig',
    'get_config',
    'get_content_processing_cache',
    'get_ml_scoring_cache', 
    'get_source_discovery_cache',
    'get_user_context_cache'
]
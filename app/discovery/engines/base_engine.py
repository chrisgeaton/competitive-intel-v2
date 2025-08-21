"""
Base discovery engine class for consistent interface across all source providers.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from pydantic import BaseModel

# Import new utility modules
from app.discovery.utils import (
    AsyncSessionManager, UnifiedErrorHandler, EngineException, EngineErrorType,
    ContentUtils, CacheManager, cache_manager, AsyncBatchProcessor,
    DiscoveryConfig, get_config
)


class ContentType(str, Enum):
    """Content type enumeration."""
    NEWS = "news"
    BLOG = "blog"
    RESEARCH = "research"
    SOCIAL = "social"
    PRESS_RELEASE = "press_release"
    FORUM = "forum"
    PODCAST = "podcast"
    OTHER = "other"


class SourceType(str, Enum):
    """Source type enumeration."""
    NEWS_API = "news_api"
    RSS_FEED = "rss_feed"
    WEB_SCRAPER = "web_scraper"
    SOCIAL_API = "social_api"
    PODCAST = "podcast"
    OTHER = "other"


@dataclass
class DiscoveredItem:
    """Standardized discovered content item."""
    title: str
    url: str
    content: str
    source_name: str
    source_type: SourceType
    content_type: ContentType
    published_at: datetime
    author: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    relevance_score: float = 0.0
    extracted_at: datetime = field(default_factory=datetime.now)


@dataclass
class SourceMetrics:
    """Source performance and quota tracking."""
    source_name: str
    source_type: SourceType
    requests_made: int = 0
    requests_remaining: int = 0
    quota_reset_time: Optional[datetime] = None
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    error_count: int = 0
    items_discovered: int = 0


# RateLimitManager is now handled by the centralized configuration system
# Rate limiting logic has been moved to the unified error handler and session manager


class BaseDiscoveryEngine(ABC):
    """Abstract base class for all discovery engines with optimized utilities."""
    
    def __init__(self, name: str, source_type: SourceType):
        self.name = name
        self.source_type = source_type
        self.logger = logging.getLogger(f"discovery.{name}")
        
        # Use centralized configuration
        self.global_config = get_config()
        self.is_enabled = True
        self.config = {}
        
        # Use shared utilities
        self.session_manager = AsyncSessionManager.get_instance(f"{name}_session")
        self.error_handler = UnifiedErrorHandler(self.logger)
        self.batch_processor = AsyncBatchProcessor(
            batch_size=self.global_config.performance.batch_size,
            max_concurrent=self.global_config.performance.max_concurrent_requests,
            timeout=self.global_config.performance.http_timeout
        )
        
        # Metrics and caching
        self.metrics = SourceMetrics(name, source_type)
        self.cache = cache_manager.get_cache(
            f"{name}_cache", 
            max_size=self.global_config.cache.source_discovery_cache_size,
            ttl_seconds=self.global_config.cache.source_discovery_ttl
        )
    
    @abstractmethod
    async def discover_content(
        self, 
        keywords: List[str], 
        focus_areas: List[str] = None,
        entities: List[str] = None,
        limit: int = 10
    ) -> List[DiscoveredItem]:
        """Discover content based on keywords and context."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the source is accessible and configured properly."""
        pass
    
    @abstractmethod
    def get_quota_info(self) -> Dict[str, Any]:
        """Get current quota usage and limits."""
        pass
    
    async def extract_content(self, raw_content: Dict[str, Any]) -> DiscoveredItem:
        """Extract and normalize content using shared utilities."""
        # Extract basic content
        title = raw_content.get('title', 'Untitled')
        url = raw_content.get('url', '')
        content = raw_content.get('content', '')
        
        # Use ContentUtils for content processing
        normalized_url = ContentUtils.normalize_url(url)
        cleaned_content = ContentUtils.clean_html(content)
        keywords = ContentUtils.extract_keywords(cleaned_content, 
                                                self.global_config.content.max_keywords)
        quality_score = ContentUtils.assess_content_quality(
            cleaned_content, title,
            self.global_config.content.min_content_length,
            self.global_config.content.max_content_length
        )
        
        return DiscoveredItem(
            title=title,
            url=normalized_url,
            content=cleaned_content,
            source_name=self.name,
            source_type=self.source_type,
            content_type=ContentType.OTHER,
            published_at=raw_content.get('published_at', datetime.now()),
            author=raw_content.get('author'),
            description=raw_content.get('description'),
            keywords=keywords,
            metadata=raw_content,
            quality_score=quality_score
        )
    
    async def assess_quality(self, item: DiscoveredItem) -> float:
        """Enhanced quality assessment using ContentUtils."""
        # Use centralized content quality assessment
        quality_score = ContentUtils.assess_content_quality(
            item.content, 
            item.title,
            self.global_config.content.min_content_length,
            self.global_config.content.max_content_length
        )
        
        # Add domain-specific bonuses
        if item.author:
            quality_score += 0.05
            
        if item.published_at and (datetime.now() - item.published_at).days < 7:
            quality_score += 0.05  # Fresh content bonus
            
        # Check readability
        readability = ContentUtils.calculate_readability_score(item.content)
        quality_score += readability * 0.1
        
        return min(quality_score, 1.0)
    
    async def filter_duplicates(self, items: List[DiscoveredItem]) -> List[DiscoveredItem]:
        """Remove duplicates using ContentUtils similarity calculation."""
        seen_hashes: Set[str] = set()
        seen_urls: Set[str] = set()
        filtered = []
        
        for item in items:
            # Generate content hash for deduplication
            content_hash = ContentUtils.generate_content_hash(item.content, item.url, item.title)
            
            # Skip if exact duplicate
            if content_hash in seen_hashes:
                continue
            
            # Skip if URL already seen (after normalization)
            normalized_url = ContentUtils.normalize_url(item.url)
            if normalized_url in seen_urls:
                continue
                
            # Check for title similarity with existing items
            is_duplicate = False
            for existing_item in filtered:
                similarity = ContentUtils.calculate_text_similarity(item.title, existing_item.title)
                if similarity > self.global_config.ml.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_hashes.add(content_hash)
                seen_urls.add(normalized_url)
                filtered.append(item)
        
        return filtered
    
    async def update_metrics(self, success: bool, response_time: float, items_count: int = 0):
        """Update engine performance metrics with caching."""
        self.metrics.requests_made += 1
        self.metrics.last_request_time = datetime.now()
        
        if success:
            self.metrics.items_discovered += items_count
            # Update rolling average response time
            current_avg = self.metrics.avg_response_time
            total_requests = self.metrics.requests_made
            self.metrics.avg_response_time = (
                (current_avg * (total_requests - 1)) + response_time
            ) / total_requests
        else:
            self.metrics.error_count += 1
        
        # Calculate success rate
        if self.metrics.requests_made > 0:
            self.metrics.success_rate = (
                (self.metrics.requests_made - self.metrics.error_count) / 
                self.metrics.requests_made
            )
        
        # Cache metrics for performance monitoring
        metrics_key = f"metrics_{self.name}_{datetime.now().strftime('%Y%m%d_%H')}"
        self.cache.put(metrics_key, self.metrics)
    
    def configure(self, config: Dict[str, Any]):
        """Configure engine with settings using centralized config."""
        self.config = config
        self.is_enabled = config.get('enabled', True)
        
        # Configuration is now handled by the global config system
        # Rate limiting is built into the session manager and error handler
        self.logger.info(f"Configured engine {self.name} with global settings")
    
    async def safe_request(self, request_func, *args, **kwargs):
        """Safely make a request using unified error handling and session management."""
        start_time = datetime.now()
        
        try:
            # Use the error handler for standardized request processing
            result = await self.error_handler.handle_request_with_retry(
                lambda: request_func(*args, **kwargs),
                provider=self.name
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            await self.update_metrics(True, response_time, len(result) if isinstance(result, list) else 1)
            
            return result
            
        except EngineException as e:
            response_time = (datetime.now() - start_time).total_seconds()
            await self.update_metrics(False, response_time)
            
            # Log with appropriate level based on error type
            if e.error_type == EngineErrorType.RATE_LIMITED:
                self.logger.warning(f"Rate limited for {self.name}: {e}")
            elif e.error_type == EngineErrorType.QUOTA_EXCEEDED:
                self.logger.error(f"Quota exceeded for {self.name}: {e}")
            else:
                self.logger.error(f"Request failed for {self.name}: {e}")
            
            raise
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            await self.update_metrics(False, response_time)
            self.logger.error(f"Unexpected error for {self.name}: {e}")
            raise EngineException(EngineErrorType.UNKNOWN_ERROR, str(e), provider=self.name)


# ContentExtractor functionality has been moved to ContentUtils
# All content processing now uses the centralized ContentUtils class for consistency

class ContentExtractor:
    """Base class for content extraction utilities - replaced by ContentUtils."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_text(self, html: str) -> str:
        """Extract clean text from HTML."""
        from app.discovery.utils import ContentUtils
        return ContentUtils.extract_text_from_html(html)
    
    def extract_metadata(self, html: str) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        return {}  # Basic implementation
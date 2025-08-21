"""
Centralized configuration for Discovery Service with environment variable support.
Replaces scattered hard-coded values throughout the codebase.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class LogLevel(str, Enum):
    """Logging level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class MLConfig:
    """Machine Learning configuration."""
    # Scoring thresholds
    similarity_threshold: float = 0.85
    relevance_threshold: float = 0.7
    quality_threshold: float = 0.6
    confidence_threshold: float = 0.5
    
    # ML model settings
    model_version: str = "2.0"
    enable_learning: bool = True
    learning_rate: float = 0.01
    
    # Engagement weights
    email_open_weight: float = 1.0
    email_click_weight: float = 3.0
    time_spent_weight: float = 2.0
    manual_feedback_weight: float = 5.0
    
    @classmethod
    def from_env(cls) -> 'MLConfig':
        """Load ML configuration from environment variables."""
        return cls(
            similarity_threshold=float(os.getenv('ML_SIMILARITY_THRESHOLD', 0.85)),
            relevance_threshold=float(os.getenv('ML_RELEVANCE_THRESHOLD', 0.7)),
            quality_threshold=float(os.getenv('ML_QUALITY_THRESHOLD', 0.6)),
            confidence_threshold=float(os.getenv('ML_CONFIDENCE_THRESHOLD', 0.5)),
            enable_learning=os.getenv('ML_ENABLE_LEARNING', 'true').lower() == 'true',
            learning_rate=float(os.getenv('ML_LEARNING_RATE', 0.01))
        )


@dataclass
class PerformanceConfig:
    """Performance and concurrency configuration."""
    # Concurrency limits
    max_concurrent_requests: int = 5
    max_concurrent_engines: int = 3
    max_concurrent_feeds: int = 5
    max_concurrent_scrapes: int = 2
    
    # Timeout settings (seconds)
    default_timeout: int = 60
    http_timeout: int = 30
    database_timeout: int = 10
    
    # Batch processing
    batch_size: int = 10
    max_batch_size: int = 50
    
    # Memory limits
    max_memory_mb: int = 512
    
    @classmethod
    def from_env(cls) -> 'PerformanceConfig':
        """Load performance configuration from environment variables."""
        return cls(
            max_concurrent_requests=int(os.getenv('PERF_MAX_CONCURRENT_REQUESTS', 5)),
            max_concurrent_engines=int(os.getenv('PERF_MAX_CONCURRENT_ENGINES', 3)),
            default_timeout=int(os.getenv('PERF_DEFAULT_TIMEOUT', 60)),
            batch_size=int(os.getenv('PERF_BATCH_SIZE', 10)),
            max_memory_mb=int(os.getenv('PERF_MAX_MEMORY_MB', 512))
        )


@dataclass
class CacheConfig:
    """Caching configuration."""
    # Cache sizes
    user_context_cache_size: int = 500
    content_processing_cache_size: int = 2000
    ml_scoring_cache_size: int = 1000
    source_discovery_cache_size: int = 500
    rss_feed_cache_size: int = 1500
    
    # TTL settings (seconds)
    user_context_ttl: int = 900      # 15 minutes
    content_processing_ttl: int = 21600  # 6 hours
    ml_scoring_ttl: int = 3600       # 1 hour
    source_discovery_ttl: int = 1800  # 30 minutes
    rss_feed_ttl: int = 7200         # 2 hours
    
    # Cache behavior
    enable_cleanup: bool = True
    cleanup_interval: int = 300      # 5 minutes
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Load cache configuration from environment variables."""
        return cls(
            user_context_cache_size=int(os.getenv('CACHE_USER_CONTEXT_SIZE', 500)),
            content_processing_cache_size=int(os.getenv('CACHE_CONTENT_PROCESSING_SIZE', 2000)),
            user_context_ttl=int(os.getenv('CACHE_USER_CONTEXT_TTL', 900)),
            content_processing_ttl=int(os.getenv('CACHE_CONTENT_PROCESSING_TTL', 21600)),
            enable_cleanup=os.getenv('CACHE_ENABLE_CLEANUP', 'true').lower() == 'true'
        )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    # Global rate limits (per hour)
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    
    # Provider-specific limits
    newsapi_requests_per_hour: int = 40
    newsapi_requests_per_day: int = 1000
    
    gnews_requests_per_hour: int = 4
    gnews_requests_per_day: int = 100
    
    bing_news_requests_per_hour: int = 40
    bing_news_requests_per_day: int = 1000
    
    rss_requests_per_hour: int = 1000
    rss_requests_per_day: int = 10000
    
    web_scraper_requests_per_hour: int = 120
    web_scraper_requests_per_day: int = 1000
    
    # Rate limit behavior
    enable_rate_limiting: bool = True
    rate_limit_backoff_factor: float = 2.0
    max_backoff_time: int = 300  # 5 minutes
    
    @classmethod
    def from_env(cls) -> 'RateLimitConfig':
        """Load rate limiting configuration from environment variables."""
        return cls(
            requests_per_hour=int(os.getenv('RATE_LIMIT_REQUESTS_PER_HOUR', 1000)),
            requests_per_day=int(os.getenv('RATE_LIMIT_REQUESTS_PER_DAY', 10000)),
            enable_rate_limiting=os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true',
            newsapi_requests_per_hour=int(os.getenv('RATE_LIMIT_NEWSAPI_HOUR', 40)),
            gnews_requests_per_hour=int(os.getenv('RATE_LIMIT_GNEWS_HOUR', 4)),
            bing_news_requests_per_hour=int(os.getenv('RATE_LIMIT_BING_HOUR', 40))
        )


@dataclass
class ContentConfig:
    """Content processing configuration."""
    # Content length limits
    min_content_length: int = 200
    max_content_length: int = 50000
    max_title_length: int = 500
    
    # Quality assessment
    min_quality_score: float = 0.3
    spam_penalty_factor: float = 0.1
    
    # Extraction settings
    max_keywords: int = 10
    max_sentences_for_summary: int = 5
    
    # Language settings
    default_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de"])
    
    @classmethod
    def from_env(cls) -> 'ContentConfig':
        """Load content configuration from environment variables."""
        return cls(
            min_content_length=int(os.getenv('CONTENT_MIN_LENGTH', 200)),
            max_content_length=int(os.getenv('CONTENT_MAX_LENGTH', 50000)),
            min_quality_score=float(os.getenv('CONTENT_MIN_QUALITY', 0.3)),
            max_keywords=int(os.getenv('CONTENT_MAX_KEYWORDS', 10))
        )


@dataclass
class SecurityConfig:
    """Security configuration."""
    # Web scraping security
    respect_robots_txt: bool = True
    user_agent: str = "Mozilla/5.0 (compatible; CompetitiveIntel/2.0)"
    
    # Request security
    max_redirects: int = 5
    verify_ssl: bool = True
    
    # Content security
    enable_content_sanitization: bool = True
    block_suspicious_domains: bool = True
    suspicious_domains: List[str] = field(default_factory=lambda: [
        "spam.com", "malware.com", "phishing.com"
    ])
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Load security configuration from environment variables."""
        return cls(
            respect_robots_txt=os.getenv('SECURITY_RESPECT_ROBOTS', 'true').lower() == 'true',
            user_agent=os.getenv('SECURITY_USER_AGENT', "Mozilla/5.0 (compatible; CompetitiveIntel/2.0)"),
            verify_ssl=os.getenv('SECURITY_VERIFY_SSL', 'true').lower() == 'true',
            enable_content_sanitization=os.getenv('SECURITY_SANITIZE_CONTENT', 'true').lower() == 'true'
        )


@dataclass
class DiscoveryConfig:
    """Main Discovery Service configuration container."""
    # Sub-configurations
    ml: MLConfig = field(default_factory=MLConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # General settings
    log_level: LogLevel = LogLevel.INFO
    debug_mode: bool = False
    metrics_enabled: bool = True
    
    # API keys (loaded from environment)
    newsapi_key: Optional[str] = None
    gnews_key: Optional[str] = None
    bing_news_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys and perform validation after initialization."""
        # Load API keys from environment
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.gnews_key = os.getenv('GNEWS_KEY')  
        self.bing_news_key = os.getenv('BING_NEWS_KEY')
        
        # Set debug mode from environment
        self.debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
        if self.debug_mode:
            self.log_level = LogLevel.DEBUG
    
    @classmethod
    def from_env(cls) -> 'DiscoveryConfig':
        """Load complete configuration from environment variables."""
        return cls(
            ml=MLConfig.from_env(),
            performance=PerformanceConfig.from_env(),
            cache=CacheConfig.from_env(),
            rate_limit=RateLimitConfig.from_env(),
            content=ContentConfig.from_env(),
            security=SecurityConfig.from_env(),
            log_level=LogLevel(os.getenv('LOG_LEVEL', 'INFO')),
            debug_mode=os.getenv('DEBUG', 'false').lower() == 'true',
            metrics_enabled=os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate ML config
        if not (0.0 <= self.ml.similarity_threshold <= 1.0):
            errors.append("ML similarity_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.ml.relevance_threshold <= 1.0):
            errors.append("ML relevance_threshold must be between 0.0 and 1.0")
        
        # Validate performance config
        if self.performance.max_concurrent_requests <= 0:
            errors.append("Performance max_concurrent_requests must be positive")
        
        if self.performance.default_timeout <= 0:
            errors.append("Performance default_timeout must be positive")
        
        # Validate cache config
        if self.cache.user_context_cache_size <= 0:
            errors.append("Cache sizes must be positive")
        
        # Validate content config
        if self.content.min_content_length <= 0:
            errors.append("Content min_content_length must be positive")
        
        if self.content.max_content_length <= self.content.min_content_length:
            errors.append("Content max_content_length must be greater than min_content_length")
        
        # Warn about missing API keys (not errors, as they may not be needed)
        if not any([self.newsapi_key, self.gnews_key, self.bing_news_key]):
            errors.append("Warning: No news API keys configured")
        
        return errors
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        return {
            'ml': self.ml.__dict__,
            'performance': self.performance.__dict__,
            'cache': self.cache.__dict__,
            'rate_limit': self.rate_limit.__dict__,
            'content': self.content.__dict__,
            'security': self.security.__dict__,
            'log_level': self.log_level.value,
            'debug_mode': self.debug_mode,
            'metrics_enabled': self.metrics_enabled,
            'api_keys_configured': {
                'newsapi': bool(self.newsapi_key),
                'gnews': bool(self.gnews_key),
                'bing_news': bool(self.bing_news_key)
            }
        }


# Global configuration instance
_config: Optional[DiscoveryConfig] = None


def get_config() -> DiscoveryConfig:
    """Get global configuration instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = DiscoveryConfig.from_env()
        
        # Validate configuration
        errors = _config.validate()
        if errors:
            import logging
            logger = logging.getLogger("discovery.config")
            for error in errors:
                if error.startswith("Warning"):
                    logger.warning(error)
                else:
                    logger.error(error)
    
    return _config


def reload_config() -> DiscoveryConfig:
    """Reload configuration from environment (useful for testing)."""
    global _config
    _config = DiscoveryConfig.from_env()
    return _config
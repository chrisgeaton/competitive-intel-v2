"""
Discovery Engines - Multi-provider source discovery implementations.
"""

from .base_engine import BaseDiscoveryEngine, DiscoveredItem, SourceMetrics
from .news_api_client import NewsAPIClient
from .rss_monitor import RSSMonitor
from .web_scraper import WebScraper
from .source_manager import SourceManager

__all__ = [
    'BaseDiscoveryEngine',
    'DiscoveredItem',
    'SourceMetrics',
    'NewsAPIClient',
    'RSSMonitor',
    'WebScraper', 
    'SourceManager'
]
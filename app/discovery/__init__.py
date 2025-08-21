"""
Discovery Service - Source Discovery Engines

Multi-provider source discovery engine for competitive intelligence.
Supports news APIs, RSS feeds, and intelligent web scraping.
"""

from .engines.news_api_client import NewsAPIClient
from .engines.rss_monitor import RSSMonitor
from .engines.web_scraper import WebScraper
from .engines.source_manager import SourceManager

__all__ = [
    'NewsAPIClient',
    'RSSMonitor', 
    'WebScraper',
    'SourceManager'
]
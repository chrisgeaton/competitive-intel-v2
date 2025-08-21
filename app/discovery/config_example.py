"""
Example configuration for the Discovery Engine system.
"""

# Example configuration for the Discovery Service engines
DISCOVERY_CONFIG = {
    # News API providers configuration
    'news_apis': {
        'newsapi_key': 'your-newsapi-key-here',
        'gnews_key': 'your-gnews-key-here',
        'bing_news_key': 'your-bing-news-key-here'
    },
    
    # RSS Monitor configuration
    'rss_monitor': {
        'enabled': True,
        'max_concurrent': 5,
        'feeds': [
            # Example RSS feeds - these would be dynamically added
            'https://feeds.reuters.com/reuters/businessNews',
            'https://feeds.bloomberg.com/business/news/index.rss',
            'https://techcrunch.com/feed/'
        ]
    },
    
    # Web Scraper configuration (use carefully and ethically)
    'web_scraper': {
        'enabled': False,  # Disabled by default due to ethical concerns
        'user_agent': 'Mozilla/5.0 (compatible; CompetitiveIntel/2.0)',
        'max_concurrent': 2,
        'respect_robots': True,
        'min_content_length': 200,
        'max_content_length': 50000
    },
    
    # Source manager configuration
    'sources': {
        'max_concurrent_engines': 3,
        'default_timeout': 30,
        'enable_fallback': True
    },
    
    # Orchestrator configuration
    'max_concurrent_requests': 5,
    'default_timeout': 60,
    'cleanup_interval': 3600  # 1 hour in seconds
}

# Rate limiting configuration (requests per hour, requests per day)
RATE_LIMITS = {
    'newsapi': {'requests_per_hour': 40, 'requests_per_day': 1000},
    'gnews': {'requests_per_hour': 4, 'requests_per_day': 100},
    'bing_news': {'requests_per_hour': 40, 'requests_per_day': 1000},
    'rss_monitor': {'requests_per_hour': 1000, 'requests_per_day': 10000},
    'web_scraper': {'requests_per_hour': 120, 'requests_per_day': 1000}
}
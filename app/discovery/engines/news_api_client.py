"""
NewsAPI client with intelligent quota management for multiple news providers.
Supports NewsAPI, GNews, and Bing News with free tier optimization.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode, quote

from .base_engine import (
    BaseDiscoveryEngine, DiscoveredItem, SourceType, ContentType, ContentExtractor
)


class NewsAPIProvider:
    """Base class for news API providers."""
    
    def __init__(self, name: str, api_key: str, base_url: str):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()


class NewsAPINewsProvider(NewsAPIProvider):
    """NewsAPI.org provider - 1000 requests/month free tier."""
    
    def __init__(self, api_key: str):
        super().__init__("newsapi", api_key, "https://newsapi.org/v2")
        self.monthly_limit = 1000
        self.requests_made = 0
        self.reset_date = datetime.now().replace(day=1) + timedelta(days=32)
        self.reset_date = self.reset_date.replace(day=1)
    
    async def search_everything(self, query: str, language: str = 'en', 
                               page_size: int = 20, sort_by: str = 'publishedAt') -> Dict[str, Any]:
        """Search all articles."""
        session = await self.get_session()
        
        params = {
            'apiKey': self.api_key,
            'q': query,
            'language': language,
            'pageSize': min(page_size, 100),  # Max 100 per request
            'sortBy': sort_by
        }
        
        url = f"{self.base_url}/everything"
        
        async with session.get(url, params=params) as response:
            if response.status == 429:
                raise Exception("NewsAPI rate limit exceeded")
            elif response.status == 401:
                raise Exception("NewsAPI authentication failed")
            elif response.status != 200:
                raise Exception(f"NewsAPI request failed: {response.status}")
            
            data = await response.json()
            self.requests_made += 1
            return data
    
    async def search_top_headlines(self, query: str = None, category: str = None,
                                  country: str = 'us', page_size: int = 20) -> Dict[str, Any]:
        """Search top headlines."""
        session = await self.get_session()
        
        params = {
            'apiKey': self.api_key,
            'pageSize': min(page_size, 100),
            'country': country
        }
        
        if query:
            params['q'] = query
        if category:
            params['category'] = category
        
        url = f"{self.base_url}/top-headlines"
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"NewsAPI headlines request failed: {response.status}")
            
            data = await response.json()
            self.requests_made += 1
            return data
    
    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota usage."""
        now = datetime.now()
        if now > self.reset_date:
            self.requests_made = 0
            self.reset_date = self.reset_date.replace(month=self.reset_date.month + 1)
            
        return {
            'provider': self.name,
            'requests_made': self.requests_made,
            'requests_remaining': max(0, self.monthly_limit - self.requests_made),
            'reset_date': self.reset_date.isoformat(),
            'limit': self.monthly_limit
        }


class GNewsProvider(NewsAPIProvider):
    """GNews provider - 100 requests/day free tier."""
    
    def __init__(self, api_key: str):
        super().__init__("gnews", api_key, "https://gnews.io/api/v4")
        self.daily_limit = 100
        self.requests_made = 0
        self.reset_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    async def search(self, query: str, language: str = 'en', country: str = 'us',
                    max_results: int = 10, sort_by: str = 'publishedAt') -> Dict[str, Any]:
        """Search articles."""
        session = await self.get_session()
        
        params = {
            'token': self.api_key,
            'q': query,
            'lang': language,
            'country': country,
            'max': min(max_results, 100),
            'sortby': sort_by
        }
        
        url = f"{self.base_url}/search"
        
        async with session.get(url, params=params) as response:
            if response.status == 429:
                raise Exception("GNews rate limit exceeded")
            elif response.status == 403:
                raise Exception("GNews authentication failed")
            elif response.status != 200:
                raise Exception(f"GNews request failed: {response.status}")
            
            data = await response.json()
            self.requests_made += 1
            return data
    
    async def top_headlines(self, category: str = None, language: str = 'en',
                           country: str = 'us', max_results: int = 10) -> Dict[str, Any]:
        """Get top headlines."""
        session = await self.get_session()
        
        params = {
            'token': self.api_key,
            'lang': language,
            'country': country,
            'max': min(max_results, 100)
        }
        
        if category:
            params['category'] = category
        
        url = f"{self.base_url}/top-headlines"
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"GNews headlines request failed: {response.status}")
            
            data = await response.json()
            self.requests_made += 1
            return data
    
    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota usage."""
        now = datetime.now()
        if now > self.reset_date:
            self.requests_made = 0
            self.reset_date = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            
        return {
            'provider': self.name,
            'requests_made': self.requests_made,
            'requests_remaining': max(0, self.daily_limit - self.requests_made),
            'reset_date': self.reset_date.isoformat(),
            'limit': self.daily_limit
        }


class BingNewsProvider(NewsAPIProvider):
    """Bing News provider - 1000 requests/month free tier."""
    
    def __init__(self, api_key: str):
        super().__init__("bing_news", api_key, "https://api.bing.microsoft.com/v7.0/news")
        self.monthly_limit = 1000
        self.requests_made = 0
        self.reset_date = datetime.now().replace(day=1) + timedelta(days=32)
        self.reset_date = self.reset_date.replace(day=1)
    
    async def search(self, query: str, count: int = 20, market: str = 'en-US',
                    sort_by: str = 'Date') -> Dict[str, Any]:
        """Search news."""
        session = await self.get_session()
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key
        }
        
        params = {
            'q': query,
            'count': min(count, 100),
            'mkt': market,
            'sortBy': sort_by
        }
        
        url = f"{self.base_url}/search"
        
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 429:
                raise Exception("Bing News rate limit exceeded")
            elif response.status == 401:
                raise Exception("Bing News authentication failed")
            elif response.status != 200:
                raise Exception(f"Bing News request failed: {response.status}")
            
            data = await response.json()
            self.requests_made += 1
            return data
    
    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota usage."""
        now = datetime.now()
        if now > self.reset_date:
            self.requests_made = 0
            self.reset_date = self.reset_date.replace(month=self.reset_date.month + 1)
            
        return {
            'provider': self.name,
            'requests_made': self.requests_made,
            'requests_remaining': max(0, self.monthly_limit - self.requests_made),
            'reset_date': self.reset_date.isoformat(),
            'limit': self.monthly_limit
        }


class NewsAPIClient(BaseDiscoveryEngine):
    """Multi-provider news API client with intelligent quota management."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("news_api_client", SourceType.NEWS_API)
        
        self.providers: List[NewsAPIProvider] = []
        self.current_provider_index = 0
        
        # Initialize providers based on config
        if config.get('newsapi_key'):
            self.providers.append(NewsAPINewsProvider(config['newsapi_key']))
            
        if config.get('gnews_key'):
            self.providers.append(GNewsProvider(config['gnews_key']))
            
        if config.get('bing_news_key'):
            self.providers.append(BingNewsProvider(config['bing_news_key']))
        
        if not self.providers:
            raise ValueError("At least one news API key must be provided")
        
        # Set up rate limiting for combined usage
        total_hourly = len(self.providers) * 10  # Conservative estimate
        self.rate_limiter.add_limit(self.name, total_hourly, total_hourly * 24)
        
        self.logger.info(f"Initialized NewsAPI client with {len(self.providers)} providers")
    
    async def discover_content(self, keywords: List[str], focus_areas: List[str] = None,
                             entities: List[str] = None, limit: int = 10) -> List[DiscoveredItem]:
        """Discover news content from multiple providers."""
        all_items = []
        
        # Build search queries
        queries = self._build_queries(keywords, focus_areas, entities)
        
        for query in queries[:3]:  # Limit to 3 queries to conserve quota
            try:
                items = await self._search_with_fallback(query, limit // len(queries))
                all_items.extend(items)
                
                if len(all_items) >= limit:
                    break
                    
            except Exception as e:
                self.logger.error(f"Search failed for query '{query}': {e}")
                continue
        
        # Remove duplicates and assess quality
        filtered_items = await self.filter_duplicates(all_items)
        
        # Assess quality for each item
        for item in filtered_items:
            item.quality_score = await self.assess_quality(item)
        
        # Sort by quality and published date
        filtered_items.sort(key=lambda x: (x.quality_score, x.published_at), reverse=True)
        
        return filtered_items[:limit]
    
    async def _search_with_fallback(self, query: str, limit: int) -> List[DiscoveredItem]:
        """Search with automatic provider fallback."""
        providers_tried = 0
        
        while providers_tried < len(self.providers):
            provider = self.providers[self.current_provider_index]
            
            # Check if provider has quota remaining
            quota = provider.get_quota_status()
            if quota['requests_remaining'] <= 0:
                self.logger.warning(f"Provider {provider.name} has no quota remaining")
                self._rotate_provider()
                providers_tried += 1
                continue
            
            try:
                items = await self._search_provider(provider, query, limit)
                return items
                
            except Exception as e:
                self.logger.error(f"Provider {provider.name} failed: {e}")
                self._rotate_provider()
                providers_tried += 1
                
                # If it's a rate limit error, wait before trying next provider
                if "rate limit" in str(e).lower():
                    await asyncio.sleep(2)
        
        raise Exception("All news providers failed or exhausted")
    
    async def _search_provider(self, provider: NewsAPIProvider, query: str, limit: int) -> List[DiscoveredItem]:
        """Search using a specific provider."""
        items = []
        
        if isinstance(provider, NewsAPINewsProvider):
            data = await provider.search_everything(query, page_size=limit)
            for article in data.get('articles', []):
                item = await self._newsapi_to_item(article, provider.name)
                items.append(item)
                
        elif isinstance(provider, GNewsProvider):
            data = await provider.search(query, max_results=limit)
            for article in data.get('articles', []):
                item = await self._gnews_to_item(article, provider.name)
                items.append(item)
                
        elif isinstance(provider, BingNewsProvider):
            data = await provider.search(query, count=limit)
            for article in data.get('value', []):
                item = await self._bing_to_item(article, provider.name)
                items.append(item)
        
        return items
    
    async def _newsapi_to_item(self, article: Dict, source_name: str) -> DiscoveredItem:
        """Convert NewsAPI article to DiscoveredItem."""
        published_str = article.get('publishedAt', '')
        published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00')) if published_str else datetime.now()
        
        content = article.get('content') or article.get('description') or ''
        content = ContentExtractor.clean_html(content)
        
        return DiscoveredItem(
            title=article.get('title', 'Untitled'),
            url=article.get('url', ''),
            content=content,
            source_name=source_name,
            source_type=SourceType.NEWS_API,
            content_type=ContentType.NEWS,
            published_at=published_at,
            author=article.get('author'),
            description=article.get('description', ''),
            keywords=ContentExtractor.extract_keywords(content),
            metadata={'source': article.get('source', {})}
        )
    
    async def _gnews_to_item(self, article: Dict, source_name: str) -> DiscoveredItem:
        """Convert GNews article to DiscoveredItem."""
        published_str = article.get('publishedAt', '')
        published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00')) if published_str else datetime.now()
        
        content = article.get('content') or article.get('description') or ''
        content = ContentExtractor.clean_html(content)
        
        return DiscoveredItem(
            title=article.get('title', 'Untitled'),
            url=article.get('url', ''),
            content=content,
            source_name=source_name,
            source_type=SourceType.NEWS_API,
            content_type=ContentType.NEWS,
            published_at=published_at,
            description=article.get('description', ''),
            keywords=ContentExtractor.extract_keywords(content),
            metadata={'source': article.get('source', {})}
        )
    
    async def _bing_to_item(self, article: Dict, source_name: str) -> DiscoveredItem:
        """Convert Bing News article to DiscoveredItem."""
        published_str = article.get('datePublished', '')
        published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00')) if published_str else datetime.now()
        
        content = article.get('description', '')
        content = ContentExtractor.clean_html(content)
        
        return DiscoveredItem(
            title=article.get('name', 'Untitled'),
            url=article.get('url', ''),
            content=content,
            source_name=source_name,
            source_type=SourceType.NEWS_API,
            content_type=ContentType.NEWS,
            published_at=published_at,
            description=article.get('description', ''),
            keywords=ContentExtractor.extract_keywords(content),
            metadata={
                'provider': article.get('provider', []),
                'category': article.get('category')
            }
        )
    
    def _build_queries(self, keywords: List[str], focus_areas: List[str] = None,
                      entities: List[str] = None) -> List[str]:
        """Build search queries from keywords, focus areas, and entities."""
        queries = []
        
        # Primary keyword query
        if keywords:
            primary_query = ' '.join(keywords[:3])  # Limit to 3 keywords
            queries.append(primary_query)
        
        # Focus area queries
        if focus_areas:
            for focus_area in focus_areas[:2]:  # Limit to 2 focus areas
                if keywords:
                    queries.append(f"{focus_area} {keywords[0]}")
                else:
                    queries.append(focus_area)
        
        # Entity queries
        if entities:
            for entity in entities[:2]:  # Limit to 2 entities
                if keywords:
                    queries.append(f'"{entity}" {keywords[0]}')
                else:
                    queries.append(f'"{entity}"')
        
        return queries[:5]  # Maximum 5 queries
    
    def _rotate_provider(self):
        """Rotate to next provider."""
        self.current_provider_index = (self.current_provider_index + 1) % len(self.providers)
    
    async def test_connection(self) -> bool:
        """Test connection to at least one provider."""
        for provider in self.providers:
            try:
                if isinstance(provider, NewsAPINewsProvider):
                    await provider.search_everything("test", page_size=1)
                elif isinstance(provider, GNewsProvider):
                    await provider.search("test", max_results=1)
                elif isinstance(provider, BingNewsProvider):
                    await provider.search("test", count=1)
                return True
            except Exception as e:
                self.logger.warning(f"Provider {provider.name} test failed: {e}")
                continue
        
        return False
    
    def get_quota_info(self) -> Dict[str, Any]:
        """Get quota information for all providers."""
        quota_info = {
            'total_providers': len(self.providers),
            'providers': []
        }
        
        for provider in self.providers:
            quota_info['providers'].append(provider.get_quota_status())
        
        return quota_info
    
    async def close(self):
        """Close all provider sessions."""
        for provider in self.providers:
            await provider.close()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'providers'):
            for provider in self.providers:
                if hasattr(provider, 'session') and provider.session and not provider.session.closed:
                    # Can't await in __del__, so we schedule it
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(provider.close())
                    except:
                        pass  # Ignore errors during cleanup
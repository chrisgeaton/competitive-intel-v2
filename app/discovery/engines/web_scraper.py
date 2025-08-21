"""
Intelligent web scraper with rate limiting and content extraction.
Respects robots.txt and implements smart content detection.
"""

import asyncio
import aiohttp
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, quote
from urllib.robotparser import RobotFileParser
import hashlib
import logging

from .base_engine import (
    BaseDiscoveryEngine, DiscoveredItem, SourceType, ContentType, ContentExtractor
)


class RobotsChecker:
    """Robots.txt compliance checker."""
    
    def __init__(self):
        self.robots_cache: Dict[str, Tuple[RobotFileParser, datetime]] = {}
        self.cache_duration = timedelta(hours=24)
    
    async def can_fetch(self, url: str, user_agent: str = '*') -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_url = urljoin(base_url, '/robots.txt')
            
            # Check cache
            if robots_url in self.robots_cache:
                rp, cached_time = self.robots_cache[robots_url]
                if datetime.now() - cached_time < self.cache_duration:
                    return rp.can_fetch(user_agent, url)
            
            # Fetch and parse robots.txt
            rp = RobotFileParser()
            rp.set_url(robots_url)
            
            try:
                # Use aiohttp to fetch robots.txt
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(robots_url) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            rp.set_robots_content(robots_content)
                        else:
                            # If no robots.txt, assume everything is allowed
                            rp.set_robots_content("")
            except:
                # If can't fetch robots.txt, assume everything is allowed
                rp.set_robots_content("")
            
            rp.read()
            self.robots_cache[robots_url] = (rp, datetime.now())
            
            return rp.can_fetch(user_agent, url)
            
        except Exception as e:
            logging.getLogger("robots_checker").debug(f"Robots.txt check failed for {url}: {e}")
            return True  # Default to allowing if check fails
    
    def get_crawl_delay(self, url: str, user_agent: str = '*') -> int:
        """Get crawl delay from robots.txt."""
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_url = urljoin(base_url, '/robots.txt')
            
            if robots_url in self.robots_cache:
                rp, _ = self.robots_cache[robots_url]
                delay = rp.crawl_delay(user_agent)
                return int(delay) if delay else 1
            
            return 1  # Default 1 second delay
            
        except:
            return 1


class ContentExtractorAdvanced(ContentExtractor):
    """Advanced content extraction with smart article detection."""
    
    @staticmethod
    def extract_article_content(html: str, url: str) -> Dict[str, Any]:
        """Extract main article content from HTML."""
        # Remove script and style elements
        html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract title
        title_match = re.search(r'<title[^>]*>([^<]*)</title>', html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        
        # Try common article selectors
        article_patterns = [
            r'<article[^>]*>(.*?)</article>',
            r'<div[^>]*class="[^"]*article[^"]*"[^>]*>(.*?)</div>',
            r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
            r'<div[^>]*class="[^"]*post[^"]*"[^>]*>(.*?)</div>',
            r'<main[^>]*>(.*?)</main>',
            r'<div[^>]*id="[^"]*content[^"]*"[^>]*>(.*?)</div>'
        ]
        
        main_content = ""
        for pattern in article_patterns:
            matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
            if matches:
                main_content = max(matches, key=len)  # Take longest match
                break
        
        # Fallback: extract all paragraph content
        if not main_content:
            paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
            main_content = ' '.join(paragraphs)
        
        # Clean content
        content = ContentExtractor.clean_html(main_content)
        
        # Extract metadata
        meta_description = ContentExtractorAdvanced._extract_meta(html, 'description')
        meta_keywords = ContentExtractorAdvanced._extract_meta(html, 'keywords')
        
        # Extract publish date
        pub_date = ContentExtractorAdvanced._extract_publish_date(html)
        
        # Extract author
        author = ContentExtractorAdvanced._extract_author(html)
        
        return {
            'title': title,
            'content': content,
            'description': meta_description,
            'keywords': meta_keywords.split(',') if meta_keywords else [],
            'published_date': pub_date,
            'author': author,
            'word_count': len(content.split()),
            'url': url
        }
    
    @staticmethod
    def _extract_meta(html: str, name: str) -> Optional[str]:
        """Extract meta tag content."""
        patterns = [
            rf'<meta[^>]*name="{name}"[^>]*content="([^"]*)"',
            rf'<meta[^>]*content="([^"]*)"[^>]*name="{name}"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    @staticmethod
    def _extract_publish_date(html: str) -> Optional[datetime]:
        """Extract publication date from HTML."""
        # Common date patterns
        date_patterns = [
            r'<meta[^>]*property="article:published_time"[^>]*content="([^"]*)"',
            r'<meta[^>]*name="pubdate"[^>]*content="([^"]*)"',
            r'<time[^>]*datetime="([^"]*)"[^>]*>',
            r'<span[^>]*class="[^"]*date[^"]*"[^>]*>([^<]*)</span>',
            r'<div[^>]*class="[^"]*date[^"]*"[^>]*>([^<]*)</div>'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                try:
                    # Try ISO format first
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except:
                    try:
                        # Try parsing common date formats
                        from dateutil import parser
                        return parser.parse(date_str)
                    except:
                        continue
        
        return None
    
    @staticmethod
    def _extract_author(html: str) -> Optional[str]:
        """Extract author information from HTML."""
        author_patterns = [
            r'<meta[^>]*name="author"[^>]*content="([^"]*)"',
            r'<meta[^>]*property="article:author"[^>]*content="([^"]*)"',
            r'<span[^>]*class="[^"]*author[^"]*"[^>]*>([^<]*)</span>',
            r'<div[^>]*class="[^"]*author[^"]*"[^>]*>([^<]*)</div>',
            r'<a[^>]*class="[^"]*author[^"]*"[^>]*>([^<]*)</a>'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None


class WebScraper(BaseDiscoveryEngine):
    """Intelligent web scraper with rate limiting and content extraction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("web_scraper", SourceType.WEB_SCRAPER)
        
        self.robots_checker = RobotsChecker()
        self.session: Optional[aiohttp.ClientSession] = None
        self.visited_urls: Set[str] = set()
        self.content_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = timedelta(hours=6)
        
        # Configuration
        config = config or {}
        self.user_agent = config.get('user_agent', 'Mozilla/5.0 (compatible; CompetitiveIntel/2.0)')
        self.max_concurrent = config.get('max_concurrent', 3)
        self.respect_robots = config.get('respect_robots', True)
        self.min_content_length = config.get('min_content_length', 200)
        self.max_content_length = config.get('max_content_length', 50000)
        
        # Set up conservative rate limiting for web scraping
        self.rate_limiter.add_limit(self.name, 120, 1000)  # 120/hour, 1000/day
        
        # Domain-specific delays (seconds between requests to same domain)
        self.domain_delays: Dict[str, float] = {}
        self.last_request_time: Dict[str, datetime] = {}
        
        self.logger.info("Initialized web scraper")
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def discover_content(self, keywords: List[str], focus_areas: List[str] = None,
                             entities: List[str] = None, limit: int = 10) -> List[DiscoveredItem]:
        """Discover content by scraping web sources."""
        all_items = []
        
        # Generate search URLs for different engines
        search_urls = self._generate_search_urls(keywords, focus_areas, entities)
        
        # Scrape search results
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._scrape_search_results_with_semaphore(semaphore, search_url, keywords, limit // len(search_urls))
            for search_url in search_urls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Search scraping failed: {result}")
        
        # Remove duplicates and assess quality
        filtered_items = await self.filter_duplicates(all_items)
        
        for item in filtered_items:
            item.quality_score = await self.assess_quality(item)
        
        # Sort by quality and recency
        filtered_items.sort(key=lambda x: (x.quality_score, x.published_at), reverse=True)
        
        return filtered_items[:limit]
    
    def _generate_search_urls(self, keywords: List[str], focus_areas: List[str] = None,
                            entities: List[str] = None) -> List[str]:
        """Generate search URLs for web scraping."""
        search_urls = []
        
        # Build search query
        query_parts = []
        if keywords:
            query_parts.extend(keywords[:3])  # Limit keywords
        if entities:
            query_parts.extend(f'"{entity}"' for entity in entities[:2])  # Add quoted entities
        
        if not query_parts:
            return []
        
        query = ' '.join(query_parts)
        encoded_query = quote(query)
        
        # Generate search engine URLs (for scraping search results, not direct API usage)
        # Note: This should be used carefully and respect terms of service
        search_engines = [
            f"https://duckduckgo.com/html/?q={encoded_query}",
            # Add other search engines that allow scraping
        ]
        
        return search_engines
    
    async def _scrape_search_results_with_semaphore(self, semaphore: asyncio.Semaphore,
                                                  search_url: str, keywords: List[str],
                                                  limit: int) -> List[DiscoveredItem]:
        """Scrape search results with semaphore for concurrency control."""
        async with semaphore:
            return await self._scrape_search_results(search_url, keywords, limit)
    
    async def _scrape_search_results(self, search_url: str, keywords: List[str],
                                   limit: int) -> List[DiscoveredItem]:
        """Scrape search results from a search engine."""
        items = []
        
        try:
            # Check if we can scrape this URL
            if self.respect_robots and not await self.robots_checker.can_fetch(search_url, self.user_agent):
                self.logger.warning(f"Robots.txt disallows scraping: {search_url}")
                return []
            
            # Apply domain-specific rate limiting
            await self._apply_domain_delay(search_url)
            
            session = await self.get_session()
            async with session.get(search_url) as response:
                if response.status != 200:
                    self.logger.warning(f"Search URL returned {response.status}: {search_url}")
                    return []
                
                html = await response.text()
                
                # Extract result URLs from search page
                result_urls = self._extract_search_result_urls(html, search_url)
                
                # Scrape individual results
                scraping_tasks = [
                    self._scrape_url(url, keywords)
                    for url in result_urls[:limit]
                    if url not in self.visited_urls
                ]
                
                scraped_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
                
                for result in scraped_results:
                    if isinstance(result, DiscoveredItem):
                        items.append(result)
                    elif isinstance(result, Exception):
                        self.logger.debug(f"URL scraping failed: {result}")
                
        except Exception as e:
            self.logger.error(f"Search results scraping failed for {search_url}: {e}")
        
        return items
    
    def _extract_search_result_urls(self, html: str, search_url: str) -> List[str]:
        """Extract result URLs from search engine page."""
        urls = []
        
        if 'duckduckgo.com' in search_url:
            # Extract DuckDuckGo results
            pattern = r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]*)"'
            matches = re.findall(pattern, html, re.IGNORECASE)
            urls.extend(matches)
        
        # Filter and normalize URLs
        filtered_urls = []
        for url in urls:
            # Skip search engine internal URLs
            if any(domain in url for domain in ['duckduckgo.com', 'google.com', 'bing.com']):
                continue
                
            # Ensure absolute URL
            if url.startswith('//'):
                url = 'https:' + url
            elif url.startswith('/'):
                base_url = f"{urlparse(search_url).scheme}://{urlparse(search_url).netloc}"
                url = urljoin(base_url, url)
            
            if url.startswith('http'):
                filtered_urls.append(url)
        
        return filtered_urls[:20]  # Limit to top 20 results
    
    async def _scrape_url(self, url: str, keywords: List[str]) -> Optional[DiscoveredItem]:
        """Scrape content from a single URL."""
        try:
            # Check cache first
            url_hash = hashlib.md5(url.encode()).hexdigest()
            if url_hash in self.content_cache:
                cached_item, cached_time = self.content_cache[url_hash]
                if datetime.now() - cached_time < self.cache_duration:
                    return cached_item
            
            # Check robots.txt
            if self.respect_robots and not await self.robots_checker.can_fetch(url, self.user_agent):
                self.logger.debug(f"Robots.txt disallows: {url}")
                return None
            
            # Apply rate limiting
            await self._apply_domain_delay(url)
            
            # Mark as visited
            self.visited_urls.add(url)
            
            session = await self.get_session()
            async with session.get(url) as response:
                if response.status != 200:
                    self.logger.debug(f"URL returned {response.status}: {url}")
                    return None
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    self.logger.debug(f"Non-HTML content: {url}")
                    return None
                
                html = await response.text()
                
                # Extract content
                extracted = ContentExtractorAdvanced.extract_article_content(html, url)
                
                # Filter by content quality and relevance
                if not self._is_quality_content(extracted, keywords):
                    return None
                
                # Create DiscoveredItem
                item = DiscoveredItem(
                    title=extracted['title'] or 'Untitled',
                    url=url,
                    content=extracted['content'],
                    source_name=urlparse(url).netloc,
                    source_type=SourceType.WEB_SCRAPER,
                    content_type=self._determine_content_type(url, extracted),
                    published_at=extracted.get('published_date') or datetime.now(),
                    author=extracted.get('author'),
                    description=extracted.get('description', ''),
                    keywords=extracted.get('keywords', []),
                    metadata={
                        'word_count': extracted.get('word_count', 0),
                        'scraped_at': datetime.now().isoformat()
                    }
                )
                
                # Cache the result
                self.content_cache[url_hash] = (item, datetime.now())
                
                return item
                
        except Exception as e:
            self.logger.debug(f"Failed to scrape {url}: {e}")
            return None
    
    async def _apply_domain_delay(self, url: str):
        """Apply rate limiting delay for domain."""
        domain = urlparse(url).netloc
        
        # Get or set delay for domain
        if domain not in self.domain_delays:
            self.domain_delays[domain] = 2.0  # Default 2 second delay
            
            # Check robots.txt for crawl delay
            if self.respect_robots:
                crawl_delay = self.robots_checker.get_crawl_delay(url, self.user_agent)
                self.domain_delays[domain] = max(crawl_delay, 1.0)
        
        # Check if we need to wait
        if domain in self.last_request_time:
            time_since_last = datetime.now() - self.last_request_time[domain]
            required_delay = timedelta(seconds=self.domain_delays[domain])
            
            if time_since_last < required_delay:
                wait_time = (required_delay - time_since_last).total_seconds()
                await asyncio.sleep(wait_time)
        
        self.last_request_time[domain] = datetime.now()
    
    def _is_quality_content(self, extracted: Dict[str, Any], keywords: List[str]) -> bool:
        """Check if extracted content meets quality criteria."""
        content = extracted.get('content', '')
        title = extracted.get('title', '')
        
        # Check minimum content length
        if len(content) < self.min_content_length:
            return False
        
        # Check maximum content length (avoid scraping entire books/documents)
        if len(content) > self.max_content_length:
            return False
        
        # Check for keyword relevance
        if keywords:
            text_to_search = f"{title} {content}".lower()
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_to_search)
            if keyword_matches == 0:
                return False
        
        # Check for spam indicators
        spam_indicators = ['click here', 'buy now', 'limited time', 'act now', 'free trial']
        content_lower = content.lower()
        spam_count = sum(1 for indicator in spam_indicators if indicator in content_lower)
        
        if spam_count > 2:  # Too many spam indicators
            return False
        
        return True
    
    def _determine_content_type(self, url: str, extracted: Dict[str, Any]) -> ContentType:
        """Determine content type based on URL and content."""
        url_lower = url.lower()
        
        if any(term in url_lower for term in ['blog', '/post/', '/article/']):
            return ContentType.BLOG
        elif any(term in url_lower for term in ['research', 'paper', 'study', 'report']):
            return ContentType.RESEARCH
        elif any(term in url_lower for term in ['news', '/press/', 'release']):
            return ContentType.NEWS
        elif any(term in url_lower for term in ['forum', 'discussion', 'thread']):
            return ContentType.FORUM
        else:
            return ContentType.OTHER
    
    async def scrape_specific_urls(self, urls: List[str], keywords: List[str]) -> List[DiscoveredItem]:
        """Scrape specific URLs directly."""
        items = []
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._scrape_url_with_semaphore(semaphore, url, keywords)
            for url in urls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, DiscoveredItem):
                items.append(result)
            elif isinstance(result, Exception):
                self.logger.debug(f"URL scraping failed: {result}")
        
        return items
    
    async def _scrape_url_with_semaphore(self, semaphore: asyncio.Semaphore,
                                       url: str, keywords: List[str]) -> Optional[DiscoveredItem]:
        """Scrape URL with semaphore for concurrency control."""
        async with semaphore:
            return await self._scrape_url(url, keywords)
    
    async def test_connection(self) -> bool:
        """Test web scraping functionality."""
        try:
            session = await self.get_session()
            # Test with a simple, scraper-friendly site
            async with session.get('https://httpbin.org/html') as response:
                return response.status == 200
        except:
            return False
    
    def get_quota_info(self) -> Dict[str, Any]:
        """Get web scraping quota information."""
        return {
            'urls_visited': len(self.visited_urls),
            'cache_size': len(self.content_cache),
            'domains_tracked': len(self.domain_delays),
            'respect_robots': self.respect_robots
        }
    
    def clear_cache(self):
        """Clear the content cache."""
        self.content_cache.clear()
        self.logger.info("Cleared web scraper cache")
    
    def clear_visited_urls(self):
        """Clear the visited URLs set."""
        self.visited_urls.clear()
        self.logger.info("Cleared visited URLs")
    
    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
            except:
                pass
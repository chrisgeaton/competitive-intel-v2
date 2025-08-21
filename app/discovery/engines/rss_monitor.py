"""
RSS feed discovery and monitoring engine with intelligent source management.
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
import hashlib
import re

from .base_engine import (
    BaseDiscoveryEngine, DiscoveredItem, SourceType, ContentType
)


class RSSFeed:
    """RSS feed representation with metadata."""
    
    def __init__(self, url: str, title: str = None, description: str = None):
        self.url = url
        self.title = title or self._extract_domain(url)
        self.description = description or ""
        self.last_updated = None
        self.last_checked = None
        self.item_count = 0
        self.error_count = 0
        self.success_rate = 1.0
        self.average_update_interval = timedelta(hours=24)  # Default
        self.is_active = True
        self.content_hash = None
        self.etag = None
        self.last_modified = None
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            domain = urlparse(url).netloc
            return domain.replace('www.', '')
        except:
            return url
    
    def update_metrics(self, success: bool, item_count: int = 0):
        """Update feed performance metrics."""
        self.last_checked = datetime.now()
        
        if success:
            if self.last_updated:
                interval = datetime.now() - self.last_updated
                # Update rolling average of update interval
                self.average_update_interval = (
                    (self.average_update_interval * 0.7) + (interval * 0.3)
                )
            self.last_updated = datetime.now()
            self.item_count += item_count
        else:
            self.error_count += 1
        
        # Calculate success rate
        total_checks = (self.item_count // max(1, item_count)) + self.error_count
        if total_checks > 0:
            self.success_rate = (total_checks - self.error_count) / total_checks
    
    def should_check(self) -> bool:
        """Determine if feed should be checked based on update patterns."""
        if not self.is_active:
            return False
            
        if not self.last_checked:
            return True
        
        time_since_check = datetime.now() - self.last_checked
        
        # Check more frequently if feed updates often
        if self.average_update_interval < timedelta(hours=6):
            check_interval = timedelta(hours=1)
        elif self.average_update_interval < timedelta(hours=24):
            check_interval = timedelta(hours=6)
        else:
            check_interval = timedelta(hours=24)
        
        # Reduce frequency if feed has errors
        if self.success_rate < 0.8:
            check_interval *= 2
        
        return time_since_check >= check_interval


class RSSParser:
    """RSS/Atom feed parser with intelligent content extraction."""
    
    @staticmethod
    def parse_feed(xml_content: str, feed_url: str) -> Dict[str, Any]:
        """Parse RSS/Atom feed XML content."""
        try:
            root = ET.fromstring(xml_content)
            
            # Detect feed format
            if root.tag == 'rss':
                return RSSParser._parse_rss(root, feed_url)
            elif root.tag == 'feed' or root.tag.endswith('}feed'):
                return RSSParser._parse_atom(root, feed_url)
            else:
                # Try to find channel or feed elements
                channel = root.find('.//channel')
                if channel is not None:
                    return RSSParser._parse_rss(root, feed_url)
                    
                feed = root.find('.//{http://www.w3.org/2005/Atom}feed')
                if feed is not None:
                    return RSSParser._parse_atom(root, feed_url)
                
                raise ValueError("Unrecognized feed format")
                
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")
    
    @staticmethod
    def _parse_rss(root: ET.Element, feed_url: str) -> Dict[str, Any]:
        """Parse RSS 2.0 format."""
        channel = root.find('.//channel')
        if channel is None:
            raise ValueError("No channel element found in RSS feed")
        
        feed_info = {
            'title': RSSParser._get_text(channel, 'title'),
            'description': RSSParser._get_text(channel, 'description'),
            'link': RSSParser._get_text(channel, 'link'),
            'last_build_date': RSSParser._get_text(channel, 'lastBuildDate'),
            'items': []
        }
        
        items = channel.findall('item')
        for item in items:
            parsed_item = {
                'title': RSSParser._get_text(item, 'title'),
                'link': RSSParser._get_text(item, 'link'),
                'description': RSSParser._get_text(item, 'description'),
                'pub_date': RSSParser._get_text(item, 'pubDate'),
                'author': RSSParser._get_text(item, 'author') or RSSParser._get_text(item, 'dc:creator'),
                'category': RSSParser._get_text(item, 'category'),
                'guid': RSSParser._get_text(item, 'guid'),
                'content': RSSParser._get_text(item, 'content:encoded') or RSSParser._get_text(item, 'description')
            }
            
            # Make URLs absolute
            if parsed_item['link'] and feed_info['link']:
                parsed_item['link'] = urljoin(feed_info['link'], parsed_item['link'])
            
            feed_info['items'].append(parsed_item)
        
        return feed_info
    
    @staticmethod
    def _parse_atom(root: ET.Element, feed_url: str) -> Dict[str, Any]:
        """Parse Atom format."""
        # Handle namespaces
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        if root.tag.endswith('}feed'):
            # Extract namespace from tag
            ns_match = re.match(r'\{([^}]+)\}', root.tag)
            if ns_match:
                ns['atom'] = ns_match.group(1)
        
        feed_info = {
            'title': RSSParser._get_text(root, 'atom:title', ns),
            'description': RSSParser._get_text(root, 'atom:subtitle', ns),
            'link': RSSParser._get_attr(root, 'atom:link', 'href', ns),
            'last_build_date': RSSParser._get_text(root, 'atom:updated', ns),
            'items': []
        }
        
        entries = root.findall('atom:entry', ns)
        for entry in entries:
            parsed_item = {
                'title': RSSParser._get_text(entry, 'atom:title', ns),
                'link': RSSParser._get_attr(entry, 'atom:link', 'href', ns),
                'description': RSSParser._get_text(entry, 'atom:summary', ns),
                'pub_date': RSSParser._get_text(entry, 'atom:published', ns) or RSSParser._get_text(entry, 'atom:updated', ns),
                'author': RSSParser._get_text(entry, 'atom:author/atom:name', ns),
                'category': RSSParser._get_attr(entry, 'atom:category', 'term', ns),
                'guid': RSSParser._get_text(entry, 'atom:id', ns),
                'content': RSSParser._get_text(entry, 'atom:content', ns) or RSSParser._get_text(entry, 'atom:summary', ns)
            }
            
            # Make URLs absolute
            if parsed_item['link'] and feed_info['link']:
                parsed_item['link'] = urljoin(feed_info['link'], parsed_item['link'])
            
            feed_info['items'].append(parsed_item)
        
        return feed_info
    
    @staticmethod
    def _get_text(element: ET.Element, path: str, namespaces: Dict[str, str] = None) -> Optional[str]:
        """Safely extract text from XML element."""
        try:
            found = element.find(path, namespaces or {})
            return found.text if found is not None else None
        except:
            return None
    
    @staticmethod
    def _get_attr(element: ET.Element, path: str, attr: str, namespaces: Dict[str, str] = None) -> Optional[str]:
        """Safely extract attribute from XML element."""
        try:
            found = element.find(path, namespaces or {})
            return found.get(attr) if found is not None else None
        except:
            return None


class RSSFeedDiscoverer:
    """Discover RSS feeds from websites."""
    
    @staticmethod
    async def discover_feeds(url: str, session: aiohttp.ClientSession) -> List[str]:
        """Discover RSS feeds from a website."""
        feeds = []
        
        try:
            # Common RSS feed locations
            common_paths = [
                '/rss',
                '/rss.xml',
                '/feed',
                '/feed.xml',
                '/feeds/all.atom.xml',
                '/atom.xml',
                '/index.xml',
                '/blog/feed',
                '/news/feed'
            ]
            
            # Try common paths first
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            for path in common_paths:
                feed_url = base_url + path
                if await RSSFeedDiscoverer._is_valid_feed(feed_url, session):
                    feeds.append(feed_url)
            
            # Parse HTML for feed links
            html_feeds = await RSSFeedDiscoverer._discover_from_html(url, session)
            feeds.extend(html_feeds)
            
        except Exception as e:
            logging.getLogger("rss_discoverer").error(f"Feed discovery failed for {url}: {e}")
        
        return list(set(feeds))  # Remove duplicates
    
    @staticmethod
    async def _is_valid_feed(url: str, session: aiohttp.ClientSession) -> bool:
        """Check if URL is a valid RSS/Atom feed."""
        try:
            async with session.head(url) as response:
                if response.status != 200:
                    return False
                
                content_type = response.headers.get('content-type', '').lower()
                return any(ft in content_type for ft in ['xml', 'rss', 'atom'])
                
        except:
            return False
    
    @staticmethod
    async def _discover_from_html(url: str, session: aiohttp.ClientSession) -> List[str]:
        """Extract feed URLs from HTML page."""
        feeds = []
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return feeds
                
                html = await response.text()
                
                # Look for feed links in HTML
                feed_patterns = [
                    r'<link[^>]*type=["\']application/rss\+xml["\'][^>]*href=["\']([^"\']+)["\']',
                    r'<link[^>]*type=["\']application/atom\+xml["\'][^>]*href=["\']([^"\']+)["\']',
                    r'<link[^>]*href=["\']([^"\']+)["\'][^>]*type=["\']application/rss\+xml["\']',
                    r'<link[^>]*href=["\']([^"\']+)["\'][^>]*type=["\']application/atom\+xml["\']'
                ]
                
                for pattern in feed_patterns:
                    matches = re.findall(pattern, html, re.IGNORECASE)
                    for match in matches:
                        feed_url = urljoin(url, match)
                        feeds.append(feed_url)
                        
        except Exception as e:
            logging.getLogger("rss_discoverer").debug(f"HTML parsing failed for {url}: {e}")
        
        return feeds


class RSSMonitor(BaseDiscoveryEngine):
    """RSS feed monitoring engine with intelligent feed management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("rss_monitor", SourceType.RSS_FEED)
        
        self.feeds: Dict[str, RSSFeed] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_concurrent_checks = config.get('max_concurrent', 5) if config else 5
        self.content_cache: Dict[str, str] = {}  # URL -> content hash
        self.discoverer = RSSFeedDiscoverer()
        
        # Set up rate limiting - RSS is generally more permissive
        self.rate_limiter.add_limit(self.name, 1000, 10000)  # 1000/hour, 10k/day
        
        self.logger.info("Initialized RSS monitor")
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; RSS Monitor/1.0)'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def add_feed(self, url: str, title: str = None, description: str = None) -> bool:
        """Add a new RSS feed for monitoring."""
        try:
            # Validate feed first
            session = await self.get_session()
            async with session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(f"Feed {url} returned status {response.status}")
                    return False
                
                content = await response.text()
                feed_data = RSSParser.parse_feed(content, url)
                
                # Create feed object
                feed = RSSFeed(
                    url=url,
                    title=title or feed_data.get('title', ''),
                    description=description or feed_data.get('description', '')
                )
                
                feed.content_hash = hashlib.md5(content.encode()).hexdigest()
                feed.etag = response.headers.get('etag')
                feed.last_modified = response.headers.get('last-modified')
                feed.update_metrics(True, len(feed_data.get('items', [])))
                
                self.feeds[url] = feed
                self.logger.info(f"Added RSS feed: {feed.title} ({url})")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add feed {url}: {e}")
            return False
    
    async def discover_and_add_feeds(self, website_url: str) -> List[str]:
        """Discover and add RSS feeds from a website."""
        session = await self.get_session()
        discovered_feeds = await self.discoverer.discover_feeds(website_url, session)
        
        added_feeds = []
        for feed_url in discovered_feeds:
            if feed_url not in self.feeds:
                if await self.add_feed(feed_url):
                    added_feeds.append(feed_url)
        
        self.logger.info(f"Discovered and added {len(added_feeds)} feeds from {website_url}")
        return added_feeds
    
    async def discover_content(self, keywords: List[str], focus_areas: List[str] = None,
                             entities: List[str] = None, limit: int = 10) -> List[DiscoveredItem]:
        """Discover content from RSS feeds."""
        all_items = []
        
        # Get feeds that should be checked
        feeds_to_check = [feed for feed in self.feeds.values() if feed.should_check()]
        
        if not feeds_to_check:
            # If no feeds need checking, check most recently updated ones
            feeds_to_check = sorted(
                self.feeds.values(), 
                key=lambda f: f.last_checked or datetime.min,
                reverse=False
            )[:5]
        
        # Check feeds concurrently
        semaphore = asyncio.Semaphore(self.max_concurrent_checks)
        tasks = [
            self._check_feed_with_semaphore(semaphore, feed, keywords, focus_areas, entities)
            for feed in feeds_to_check
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Feed check failed: {result}")
        
        # Remove duplicates and assess quality
        filtered_items = await self.filter_duplicates(all_items)
        
        for item in filtered_items:
            item.quality_score = await self.assess_quality(item)
        
        # Sort by relevance and recency
        filtered_items.sort(key=lambda x: (x.quality_score, x.published_at), reverse=True)
        
        return filtered_items[:limit]
    
    async def _check_feed_with_semaphore(self, semaphore: asyncio.Semaphore, feed: RSSFeed,
                                       keywords: List[str], focus_areas: List[str] = None,
                                       entities: List[str] = None) -> List[DiscoveredItem]:
        """Check a single feed with semaphore for concurrency control."""
        async with semaphore:
            return await self._check_feed(feed, keywords, focus_areas, entities)
    
    async def _check_feed(self, feed: RSSFeed, keywords: List[str],
                         focus_areas: List[str] = None, entities: List[str] = None) -> List[DiscoveredItem]:
        """Check a single RSS feed for new content."""
        try:
            session = await self.get_session()
            headers = {}
            
            # Use ETags and Last-Modified for efficient checking
            if feed.etag:
                headers['If-None-Match'] = feed.etag
            if feed.last_modified:
                headers['If-Modified-Since'] = feed.last_modified
            
            async with session.get(feed.url, headers=headers) as response:
                if response.status == 304:
                    # Content not modified
                    feed.update_metrics(True, 0)
                    return []
                
                if response.status != 200:
                    feed.update_metrics(False)
                    self.logger.warning(f"Feed {feed.title} returned status {response.status}")
                    return []
                
                content = await response.text()
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                # Check if content actually changed
                if feed.content_hash == content_hash:
                    feed.update_metrics(True, 0)
                    return []
                
                # Parse feed
                feed_data = RSSParser.parse_feed(content, feed.url)
                items = []
                
                for item_data in feed_data.get('items', []):
                    # Convert to DiscoveredItem
                    item = await self._rss_item_to_discovered_item(item_data, feed)
                    
                    # Filter by relevance to keywords/focus areas
                    if self._is_relevant(item, keywords, focus_areas, entities):
                        items.append(item)
                
                # Update feed metadata
                feed.content_hash = content_hash
                feed.etag = response.headers.get('etag')
                feed.last_modified = response.headers.get('last-modified')
                feed.update_metrics(True, len(items))
                
                return items
                
        except Exception as e:
            feed.update_metrics(False)
            self.logger.error(f"Failed to check feed {feed.title}: {e}")
            return []
    
    async def _rss_item_to_discovered_item(self, item_data: Dict, feed: RSSFeed) -> DiscoveredItem:
        """Convert RSS item to DiscoveredItem."""
        # Parse publication date
        pub_date_str = item_data.get('pub_date', '')
        published_at = datetime.now()
        
        if pub_date_str:
            try:
                # Try parsing RFC 2822 format (common in RSS)
                from email.utils import parsedate_to_datetime
                published_at = parsedate_to_datetime(pub_date_str)
            except:
                try:
                    # Try ISO format
                    published_at = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                except:
                    self.logger.debug(f"Could not parse date: {pub_date_str}")
        
        # Clean and extract content
        content = item_data.get('content') or item_data.get('description') or ''
        content = ContentExtractor.clean_html(content)
        
        # Determine content type based on source
        content_type = ContentType.NEWS
        if any(term in feed.url.lower() for term in ['blog', 'research', 'paper']):
            content_type = ContentType.BLOG if 'blog' in feed.url.lower() else ContentType.RESEARCH
        
        return DiscoveredItem(
            title=item_data.get('title', 'Untitled'),
            url=item_data.get('link', ''),
            content=content,
            source_name=feed.title,
            source_type=SourceType.RSS_FEED,
            content_type=content_type,
            published_at=published_at,
            author=item_data.get('author'),
            description=item_data.get('description', ''),
            keywords=ContentExtractor.extract_keywords(content),
            metadata={
                'feed_url': feed.url,
                'guid': item_data.get('guid'),
                'category': item_data.get('category')
            }
        )
    
    def _is_relevant(self, item: DiscoveredItem, keywords: List[str],
                    focus_areas: List[str] = None, entities: List[str] = None) -> bool:
        """Check if item is relevant to search criteria."""
        text_to_search = f"{item.title} {item.content} {item.description}".lower()
        
        # Check keywords
        if keywords:
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_to_search)
            if keyword_matches == 0:
                return False
        
        # Boost relevance for focus areas and entities
        relevance_score = 0
        
        if focus_areas:
            for focus_area in focus_areas:
                if focus_area.lower() in text_to_search:
                    relevance_score += 2
        
        if entities:
            for entity in entities:
                if entity.lower() in text_to_search:
                    relevance_score += 3
        
        # Require some relevance if focus areas or entities are specified
        if (focus_areas or entities) and relevance_score == 0:
            return False
        
        return True
    
    async def test_connection(self) -> bool:
        """Test connection by checking if we can access any feeds."""
        if not self.feeds:
            return True  # No feeds to test
        
        # Test first active feed
        for feed in self.feeds.values():
            if feed.is_active:
                try:
                    session = await self.get_session()
                    async with session.head(feed.url) as response:
                        return response.status == 200
                except:
                    continue
        
        return False
    
    def get_quota_info(self) -> Dict[str, Any]:
        """Get RSS monitoring quota information."""
        return {
            'total_feeds': len(self.feeds),
            'active_feeds': sum(1 for f in self.feeds.values() if f.is_active),
            'feeds_needing_check': sum(1 for f in self.feeds.values() if f.should_check()),
            'avg_success_rate': sum(f.success_rate for f in self.feeds.values()) / len(self.feeds) if self.feeds else 0
        }
    
    def get_feed_status(self) -> List[Dict[str, Any]]:
        """Get status of all monitored feeds."""
        status = []
        for feed in self.feeds.values():
            status.append({
                'url': feed.url,
                'title': feed.title,
                'is_active': feed.is_active,
                'last_checked': feed.last_checked.isoformat() if feed.last_checked else None,
                'last_updated': feed.last_updated.isoformat() if feed.last_updated else None,
                'success_rate': feed.success_rate,
                'item_count': feed.item_count,
                'error_count': feed.error_count,
                'should_check': feed.should_check()
            })
        return status
    
    def remove_feed(self, url: str) -> bool:
        """Remove a feed from monitoring."""
        if url in self.feeds:
            del self.feeds[url]
            self.logger.info(f"Removed feed: {url}")
            return True
        return False
    
    def activate_feed(self, url: str) -> bool:
        """Activate a feed."""
        if url in self.feeds:
            self.feeds[url].is_active = True
            return True
        return False
    
    def deactivate_feed(self, url: str) -> bool:
        """Deactivate a feed."""
        if url in self.feeds:
            self.feeds[url].is_active = False
            return True
        return False
    
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
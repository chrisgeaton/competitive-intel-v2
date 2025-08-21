"""
Podcast discovery engine using PodcastIndex.org API for comprehensive podcast and episode discovery.

Provides advanced podcast search, episode discovery, and metadata extraction with intelligent
filtering, relevance scoring, and integration with existing ML content processing pipeline.
Supports PodcastIndex.org free API with proper authentication and rate limiting.
"""

import asyncio
import aiohttp
import json
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urlencode, quote
from dataclasses import dataclass, field
import re

from .base_engine import (
    BaseDiscoveryEngine, DiscoveredItem, SourceType, ContentType
)

from app.discovery.utils import (
    AsyncSessionManager, UnifiedErrorHandler, EngineException, EngineErrorType,
    ContentUtils, CacheManager, cache_manager, AsyncBatchProcessor,
    DiscoveryConfig, get_config
)


@dataclass
class PodcastShow:
    """Podcast show metadata from PodcastIndex.org."""
    id: int
    title: str
    url: str
    original_url: str
    link: str
    description: str
    author: str
    owner_name: str
    image: str
    artwork: str
    last_update_time: int
    last_crawl_time: int
    last_parse_time: int
    last_good_http_status_time: int
    last_http_status: int
    content_type: str
    itunesId: Optional[int] = None
    language: str = "en"
    categories: Dict[str, str] = field(default_factory=dict)
    explicit: bool = False
    episode_count: int = 0
    crawl_errors: int = 0
    parse_errors: int = 0


@dataclass
class PodcastEpisode:
    """Podcast episode metadata from PodcastIndex.org."""
    id: int
    title: str
    link: str
    description: str
    guid: str
    date_published: int
    date_published_pretty: str
    date_crawled: int
    enclosure_url: str
    enclosure_type: str
    enclosure_length: int
    duration: int
    explicit: int
    episode: Optional[int] = None
    episode_type: Optional[str] = None
    season: Optional[int] = None
    image: Optional[str] = None
    feed_itunesId: Optional[int] = None
    feed_image: Optional[str] = None
    feed_id: int = 0
    feed_title: str = ""
    feed_language: str = "en"
    chaptersUrl: Optional[str] = None
    transcriptUrl: Optional[str] = None


class PodcastIndexAuth:
    """Authentication handler for PodcastIndex.org API."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
    
    def generate_headers(self) -> Dict[str, str]:
        """Generate authentication headers for PodcastIndex.org API."""
        api_header_time = str(int(time.time()))
        data_to_hash = self.api_key + self.api_secret + api_header_time
        sha_1 = hashlib.sha1(data_to_hash.encode()).hexdigest()
        
        return {
            "X-Auth-Date": api_header_time,
            "X-Auth-Key": self.api_key,
            "Authorization": sha_1,
            "User-Agent": "CompetitiveIntelligence/2.0"
        }


class PodcastIndexClient:
    """Client for PodcastIndex.org API with rate limiting and error handling."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.auth = PodcastIndexAuth(api_key, api_secret)
        self.base_url = "https://api.podcastindex.org/api/1.0"
        self.session_manager = AsyncSessionManager(name="podcast_index")
        self.error_handler = UnifiedErrorHandler()
        
        # Rate limiting - PodcastIndex.org is generous but we want to be respectful
        self.requests_per_minute = 60
        self.last_request_time = 0.0
        self.request_count = 0
        self.request_window_start = time.time()
        
    async def _enforce_rate_limit(self):
        """Enforce rate limiting for API requests."""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check if we're at the limit
        if self.request_count >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_window_start)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self.request_count = 0
                self.request_window_start = time.time()
        
        # Ensure minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < 1.0:  # 1 second minimum between requests
            await asyncio.sleep(1.0 - time_since_last)
        
        self.request_count += 1
        self.last_request_time = time.time()
    
    async def search_podcasts(self, query: str, limit: int = 10, clean: bool = True) -> List[PodcastShow]:
        """Search for podcasts by query."""
        await self._enforce_rate_limit()
        
        params = {
            'q': query,
            'max': min(limit, 40),  # API max is 40
            'clean': clean,
            'pretty': True
        }
        
        url = f"{self.base_url}/search/byterm"
        headers = self.auth.generate_headers()
        
        try:
            session = await self.session_manager.get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return [PodcastShow(**show) for show in data.get('feeds', [])]
                else:
                    raise EngineException(
                        f"PodcastIndex search failed: {response.status}",
                        EngineErrorType.API_ERROR,
                        {"status": response.status, "query": query}
                    )
        
        except Exception as e:
            await self.error_handler.handle_error(e, {
                'operation': 'podcast_search',
                'query': query,
                'limit': limit
            })
            raise
    
    async def search_episodes_by_podcast(self, feed_id: int, limit: int = 10, 
                                       since: Optional[int] = None) -> List[PodcastEpisode]:
        """Get episodes for a specific podcast feed."""
        await self._enforce_rate_limit()
        
        params = {
            'id': feed_id,
            'max': min(limit, 200),  # API max is 200
            'pretty': True
        }
        
        if since:
            params['since'] = since
        
        url = f"{self.base_url}/episodes/byfeedid"
        headers = self.auth.generate_headers()
        
        try:
            session = await self.session_manager.get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    episodes = []
                    for episode_data in data.get('items', []):
                        # Clean up and validate episode data
                        episode_data.setdefault('feed_id', feed_id)
                        episode_data.setdefault('feed_title', '')
                        episode_data.setdefault('feed_language', 'en')
                        episodes.append(PodcastEpisode(**episode_data))
                    return episodes
                else:
                    raise EngineException(
                        f"PodcastIndex episodes search failed: {response.status}",
                        EngineErrorType.API_ERROR,
                        {"status": response.status, "feed_id": feed_id}
                    )
        
        except Exception as e:
            await self.error_handler.handle_error(e, {
                'operation': 'episodes_search',
                'feed_id': feed_id,
                'limit': limit
            })
            raise
    
    async def search_episodes_by_title(self, query: str, limit: int = 10) -> List[PodcastEpisode]:
        """Search for episodes by title/content."""
        await self._enforce_rate_limit()
        
        params = {
            'q': query,
            'max': min(limit, 40),  # API max is 40
            'pretty': True
        }
        
        url = f"{self.base_url}/search/byterm"
        headers = self.auth.generate_headers()
        
        try:
            session = await self.session_manager.get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    episodes = []
                    for episode_data in data.get('items', []):
                        # Set defaults for missing fields
                        episode_data.setdefault('feed_title', episode_data.get('feedTitle', ''))
                        episode_data.setdefault('feed_language', 'en')
                        episodes.append(PodcastEpisode(**episode_data))
                    return episodes
                else:
                    raise EngineException(
                        f"PodcastIndex episode search failed: {response.status}",
                        EngineErrorType.API_ERROR,
                        {"status": response.status, "query": query}
                    )
        
        except Exception as e:
            await self.error_handler.handle_error(e, {
                'operation': 'episode_title_search',
                'query': query,
                'limit': limit
            })
            raise
    
    async def get_trending_podcasts(self, limit: int = 10, language: str = "en") -> List[PodcastShow]:
        """Get trending podcasts."""
        await self._enforce_rate_limit()
        
        params = {
            'max': min(limit, 40),
            'lang': language,
            'pretty': True
        }
        
        url = f"{self.base_url}/podcasts/trending"
        headers = self.auth.generate_headers()
        
        try:
            session = await self.session_manager.get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return [PodcastShow(**show) for show in data.get('feeds', [])]
                else:
                    raise EngineException(
                        f"PodcastIndex trending failed: {response.status}",
                        EngineErrorType.API_ERROR,
                        {"status": response.status}
                    )
        
        except Exception as e:
            await self.error_handler.handle_error(e, {
                'operation': 'trending_podcasts',
                'limit': limit,
                'language': language
            })
            raise
    
    async def close(self):
        """Close the session manager."""
        await self.session_manager.close()


class PodcastDiscoveryEngine(BaseDiscoveryEngine):
    """Advanced podcast discovery engine using PodcastIndex.org API."""
    
    def __init__(self, api_key: str, api_secret: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("PodcastIndex", SourceType.PODCAST)
        
        self.client = PodcastIndexClient(api_key, api_secret)
        self.content_utils = ContentUtils()
        self.batch_processor = AsyncBatchProcessor(max_concurrent=5)
        
        # Configuration
        self.config = config or {}
        self.max_episodes_per_show = self.config.get('max_episodes_per_show', 5)
        self.max_shows_per_query = self.config.get('max_shows_per_query', 10)
        self.recency_threshold_days = self.config.get('recency_threshold_days', 30)
        self.min_episode_duration = self.config.get('min_episode_duration', 300)  # 5 minutes
        self.max_episode_duration = self.config.get('max_episode_duration', 7200)  # 2 hours
        self.language_filter = self.config.get('language_filter', ['en'])
        self.exclude_explicit = self.config.get('exclude_explicit', False)
        
        # Use inherited cache from BaseDiscoveryEngine
        # Additional caching keys for podcast-specific data
        self.podcast_cache_key = "podcast_shows"
        self.episode_cache_key = "podcast_episodes"
        
        # Set engine_name for compatibility
        self.engine_name = self.name
        
        self.logger.info(f"Podcast discovery engine initialized with PodcastIndex.org API")
    
    async def discover_content(self, keywords: List[str], limit: int = 20,
                             user_context: Optional[Dict[str, Any]] = None) -> List[DiscoveredItem]:
        """Discover podcast content based on keywords and user context."""
        
        if not keywords:
            self.logger.warning("No keywords provided for podcast discovery")
            return []
        
        try:
            # Extract user preferences
            focus_areas = user_context.get('focus_areas', []) if user_context else []
            tracked_entities = user_context.get('tracked_entities', []) if user_context else []
            user_language = user_context.get('language', 'en') if user_context else 'en'
            
            # Combine search terms intelligently
            search_terms = self._build_search_terms(keywords, focus_areas, tracked_entities)
            
            # Discover podcasts and episodes
            all_items = []
            
            # Process search terms in batches
            search_tasks = []
            for term in search_terms[:5]:  # Limit to top 5 search terms
                search_tasks.append(self._discover_for_term(term, limit // len(search_terms) + 2))
            
            if search_tasks:
                results = await self.batch_processor.process_batch(search_tasks)
                for term_results in results:
                    if isinstance(term_results, list):
                        all_items.extend(term_results)
            
            # Apply filtering and scoring
            filtered_items = await self._filter_and_score_items(all_items, keywords, user_context)
            
            # Sort by relevance and recency, limit results
            filtered_items.sort(key=lambda x: (x.relevance_score, x.quality_score), reverse=True)
            final_items = filtered_items[:limit]
            
            self.logger.info(f"Discovered {len(final_items)} podcast items from {len(search_terms)} search terms")
            return final_items
            
        except Exception as e:
            await self.error_handler.handle_error(e, {
                'keywords': keywords,
                'limit': limit,
                'user_context_keys': list(user_context.keys()) if user_context else []
            })
            return []
    
    def _build_search_terms(self, keywords: List[str], focus_areas: List[str], 
                           tracked_entities: List[str]) -> List[str]:
        """Build intelligent search terms from user context."""
        search_terms = set()
        
        # Primary keywords (direct user input)
        search_terms.update(keywords[:10])  # Limit primary keywords
        
        # Focus areas (strategic interests)
        for focus_area in focus_areas[:5]:
            # Clean and add focus area
            clean_area = re.sub(r'[^\w\s]', '', focus_area).strip()
            if clean_area:
                search_terms.add(clean_area)
        
        # Tracked entities (companies, people, technologies)
        for entity in tracked_entities[:10]:
            # Clean entity name
            clean_entity = re.sub(r'[^\w\s]', '', entity).strip()
            if clean_entity:
                search_terms.add(clean_entity)
                
                # Add variations for company names
                if 'inc' not in clean_entity.lower() and 'corp' not in clean_entity.lower():
                    search_terms.add(f"{clean_entity} company")
        
        # Combine terms for more specific searches
        combined_terms = []
        if keywords and focus_areas:
            for keyword in keywords[:3]:
                for area in focus_areas[:2]:
                    combined_terms.append(f"{keyword} {area}")
        
        search_terms.update(combined_terms[:5])
        
        return list(search_terms)
    
    async def _discover_for_term(self, search_term: str, limit: int) -> List[DiscoveredItem]:
        """Discover content for a specific search term."""
        items = []
        
        try:
            # Search for podcasts matching the term
            podcasts = await self.client.search_podcasts(search_term, limit=min(limit, 10))
            
            # For each relevant podcast, get recent episodes
            for podcast in podcasts:
                if not self._is_podcast_relevant(podcast):
                    continue
                
                # Get recent episodes
                since_timestamp = int((datetime.now() - timedelta(days=self.recency_threshold_days)).timestamp())
                episodes = await self.client.search_episodes_by_podcast(
                    podcast.id, 
                    limit=self.max_episodes_per_show,
                    since=since_timestamp
                )
                
                # Convert episodes to DiscoveredItems
                for episode in episodes:
                    if self._is_episode_relevant(episode, podcast):
                        item = self._convert_episode_to_item(episode, podcast, search_term)
                        items.append(item)
            
            # Also search for episodes directly by term
            direct_episodes = await self.client.search_episodes_by_title(search_term, limit=5)
            for episode in direct_episodes:
                if self._is_episode_relevant(episode):
                    # Create a minimal podcast show object for context
                    podcast = PodcastShow(
                        id=episode.feed_id,
                        title=episode.feed_title,
                        url="",
                        original_url="",
                        link="",
                        description="",
                        author="",
                        owner_name="",
                        image=episode.feed_image or "",
                        artwork=episode.feed_image or "",
                        last_update_time=0,
                        last_crawl_time=0,
                        last_parse_time=0,
                        last_good_http_status_time=0,
                        last_http_status=200,
                        content_type="audio/mpeg",
                        language=episode.feed_language
                    )
                    
                    item = self._convert_episode_to_item(episode, podcast, search_term)
                    items.append(item)
            
        except Exception as e:
            self.logger.error(f"Failed to discover content for term '{search_term}': {e}")
        
        return items
    
    def _is_podcast_relevant(self, podcast: PodcastShow) -> bool:
        """Check if a podcast is relevant based on our criteria."""
        # Language filter
        if self.language_filter and podcast.language not in self.language_filter:
            return False
        
        # Explicit content filter
        if self.exclude_explicit and podcast.explicit:
            return False
        
        # Check if podcast is active (updated recently)
        if podcast.last_update_time:
            last_update = datetime.fromtimestamp(podcast.last_update_time)
            days_since_update = (datetime.now() - last_update).days
            if days_since_update > 90:  # Not updated in 3 months
                return False
        
        # Check for minimum episode count (avoid inactive podcasts)
        if podcast.episode_count > 0 and podcast.episode_count < 3:
            return False
        
        return True
    
    def _is_episode_relevant(self, episode: PodcastEpisode, podcast: Optional[PodcastShow] = None) -> bool:
        """Check if an episode is relevant based on our criteria."""
        # Duration filter
        if episode.duration:
            if episode.duration < self.min_episode_duration or episode.duration > self.max_episode_duration:
                return False
        
        # Explicit content filter
        if self.exclude_explicit and episode.explicit:
            return False
        
        # Recency filter
        if episode.date_published:
            published_date = datetime.fromtimestamp(episode.date_published)
            days_since_published = (datetime.now() - published_date).days
            if days_since_published > self.recency_threshold_days:
                return False
        
        # Quality filters
        if not episode.title or len(episode.title.strip()) < 5:
            return False
        
        if not episode.description or len(episode.description.strip()) < 20:
            return False
        
        return True
    
    def _convert_episode_to_item(self, episode: PodcastEpisode, podcast: PodcastShow, 
                                search_term: str) -> DiscoveredItem:
        """Convert a podcast episode to a DiscoveredItem."""
        
        # Format content for processing
        content = self._format_episode_content(episode, podcast)
        
        # Extract keywords from title and description
        keywords = self.content_utils.extract_keywords(f"{episode.title} {episode.description}")
        keywords.append(search_term)  # Add the search term that found this
        
        # Add podcast-specific keywords
        if podcast.categories:
            keywords.extend(list(podcast.categories.keys())[:3])
        
        # Calculate duration string
        duration_str = self._format_duration(episode.duration)
        
        # Create metadata with podcast-specific information
        metadata = {
            'podcast_id': podcast.id,
            'podcast_title': podcast.title,
            'podcast_author': podcast.author,
            'podcast_owner': podcast.owner_name,
            'episode_id': episode.id,
            'episode_number': episode.episode,
            'season_number': episode.season,
            'episode_type': episode.episode_type,
            'duration_seconds': episode.duration,
            'duration_formatted': duration_str,
            'enclosure_url': episode.enclosure_url,
            'enclosure_type': episode.enclosure_type,
            'enclosure_length': episode.enclosure_length,
            'podcast_image': podcast.artwork or podcast.image,
            'episode_image': episode.image,
            'podcast_categories': podcast.categories,
            'language': getattr(podcast, 'language', 'en'),
            'explicit': bool(episode.explicit),
            'transcript_url': episode.transcriptUrl,
            'chapters_url': episode.chaptersUrl,
            'search_term': search_term,
            'discovery_source': 'podcastindex',
            'ready_for_transcription': bool(episode.transcriptUrl or episode.enclosure_url)
        }
        
        return DiscoveredItem(
            title=episode.title,
            url=episode.link or episode.enclosure_url,
            content=content,
            source_name=f"PodcastIndex: {podcast.title}",
            source_type=SourceType.PODCAST,
            content_type=ContentType.PODCAST,
            published_at=datetime.fromtimestamp(episode.date_published) if episode.date_published else datetime.now(),
            author=podcast.author or podcast.owner_name,
            description=episode.description,
            keywords=list(set(keywords)),  # Remove duplicates
            metadata=metadata,
            extracted_at=datetime.now()
        )
    
    def _format_episode_content(self, episode: PodcastEpisode, podcast: PodcastShow) -> str:
        """Format episode content for processing."""
        content_parts = [
            f"Podcast: {podcast.title}",
            f"Episode: {episode.title}",
            f"Description: {episode.description}",
        ]
        
        if episode.duration:
            content_parts.append(f"Duration: {self._format_duration(episode.duration)}")
        
        if podcast.author:
            content_parts.append(f"Host: {podcast.author}")
        
        if podcast.categories:
            categories_str = ", ".join(list(podcast.categories.keys())[:3])
            content_parts.append(f"Categories: {categories_str}")
        
        return " | ".join(content_parts)
    
    def _format_duration(self, seconds: int) -> str:
        """Format duration in seconds to readable string."""
        if not seconds:
            return "Unknown"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    async def _filter_and_score_items(self, items: List[DiscoveredItem], 
                                     original_keywords: List[str],
                                     user_context: Optional[Dict[str, Any]] = None) -> List[DiscoveredItem]:
        """Apply advanced filtering and scoring to discovered items."""
        filtered_items = []
        
        for item in items:
            # Remove duplicates based on episode ID
            episode_id = item.metadata.get('episode_id')
            if episode_id and any(existing.metadata.get('episode_id') == episode_id for existing in filtered_items):
                continue
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(item, original_keywords, user_context)
            item.relevance_score = relevance_score
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(item)
            item.quality_score = quality_score
            
            # Only include items above minimum thresholds
            if relevance_score > 0.3 and quality_score > 0.4:
                filtered_items.append(item)
        
        return filtered_items
    
    def _calculate_relevance_score(self, item: DiscoveredItem, keywords: List[str],
                                  user_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate relevance score for a podcast episode."""
        score = 0.0
        
        # Text to analyze
        text_content = f"{item.title} {item.description}".lower()
        
        # Keyword matching (40% of score)
        keyword_matches = 0
        for keyword in keywords:
            if keyword.lower() in text_content:
                keyword_matches += 1
        
        if keywords:
            score += (keyword_matches / len(keywords)) * 0.4
        
        # User context matching (30% of score)
        if user_context:
            context_score = 0.0
            
            # Focus areas
            focus_areas = user_context.get('focus_areas', [])
            for area in focus_areas:
                if area.lower() in text_content:
                    context_score += 0.1
            
            # Tracked entities
            tracked_entities = user_context.get('tracked_entities', [])
            for entity in tracked_entities:
                if entity.lower() in text_content:
                    context_score += 0.15
            
            score += min(context_score, 0.3)  # Cap at 30%
        
        # Recency bonus (15% of score)
        if item.published_at:
            days_old = (datetime.now() - item.published_at.replace(tzinfo=None)).days
            if days_old <= 7:
                score += 0.15
            elif days_old <= 30:
                score += 0.10
            elif days_old <= 90:
                score += 0.05
        
        # Duration bonus (15% of score) - prefer moderate length content
        duration = item.metadata.get('duration_seconds', 0)
        if 600 <= duration <= 3600:  # 10 minutes to 1 hour is ideal
            score += 0.15
        elif 300 <= duration < 600 or 3600 < duration <= 5400:  # 5-10 min or 1-1.5 hours
            score += 0.10
        elif duration > 0:  # Any valid duration
            score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_quality_score(self, item: DiscoveredItem) -> float:
        """Calculate quality score for a podcast episode."""
        score = 0.0
        
        # Title quality (25% of score)
        title_length = len(item.title)
        if 20 <= title_length <= 100:
            score += 0.25
        elif 10 <= title_length < 20 or 100 < title_length <= 150:
            score += 0.15
        elif title_length >= 10:
            score += 0.10
        
        # Description quality (25% of score)
        desc_length = len(item.description)
        if 100 <= desc_length <= 1000:
            score += 0.25
        elif 50 <= desc_length < 100 or 1000 < desc_length <= 2000:
            score += 0.15
        elif desc_length >= 50:
            score += 0.10
        
        # Podcast metadata completeness (25% of score)
        metadata_score = 0.0
        if item.metadata.get('podcast_author'):
            metadata_score += 0.05
        if item.metadata.get('podcast_categories'):
            metadata_score += 0.05
        if item.metadata.get('podcast_image'):
            metadata_score += 0.05
        if item.metadata.get('enclosure_url'):
            metadata_score += 0.05
        if item.metadata.get('duration_seconds'):
            metadata_score += 0.05
        
        score += metadata_score
        
        # Transcription readiness (25% of score)
        if item.metadata.get('transcript_url'):
            score += 0.25
        elif item.metadata.get('ready_for_transcription'):
            score += 0.15
        elif item.metadata.get('enclosure_url'):
            score += 0.10
        
        return min(score, 1.0)
    
    async def test_connection(self) -> bool:
        """Test if PodcastIndex.org API is accessible and configured properly."""
        try:
            # Try to get trending podcasts (lightweight test)
            trending = await self.client.get_trending_podcasts(limit=1)
            return len(trending) >= 0  # Even empty result means API is working
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_quota_info(self) -> Dict[str, Any]:
        """Get current quota usage and limits for PodcastIndex.org API."""
        return {
            "service": "PodcastIndex.org",
            "quota_type": "rate_limited",
            "requests_per_minute": self.client.requests_per_minute,
            "requests_made_this_window": self.client.request_count,
            "window_start": self.client.request_window_start,
            "rate_limit_remaining": max(0, self.client.requests_per_minute - self.client.request_count),
            "quota_reset_time": None,  # PodcastIndex.org doesn't have daily quotas
            "is_unlimited": False,
            "cost_per_request": 0.0,  # Free API
            "monthly_limit": None,  # No monthly limit published
            "requests_made_total": getattr(self, 'total_requests_made', 0)
        }
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and metrics."""
        return {
            "engine_name": self.name,
            "source_type": self.source_type.value,
            "is_available": True,
            "last_used": getattr(self, 'last_request_time', None),
            "requests_made": getattr(self, 'requests_made', 0),
            "success_rate": getattr(self, 'success_rate', 0.0),
            "avg_response_time": getattr(self, 'avg_response_time', 0.0),
            "configuration": {
                "max_episodes_per_show": self.max_episodes_per_show,
                "max_shows_per_query": self.max_shows_per_query,
                "recency_threshold_days": self.recency_threshold_days,
                "language_filter": self.language_filter,
                "exclude_explicit": self.exclude_explicit
            },
            "cache_stats": {
                "cache_size": len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
                "cache_ttl": getattr(self.cache, 'ttl_seconds', 0)
            }
        }
    
    async def close(self):
        """Clean up resources."""
        await self.client.close()
        # Note: BaseDiscoveryEngine doesn't have a close method, so we just clean up our own resources
        self.logger.info("Podcast discovery engine closed")


# Factory function for easy initialization
def create_podcast_engine(api_key: str, api_secret: str, **config) -> PodcastDiscoveryEngine:
    """Create a configured podcast discovery engine."""
    return PodcastDiscoveryEngine(api_key, api_secret, config)
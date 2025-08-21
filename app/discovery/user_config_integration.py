"""
User Config Service integration for targeted discovery based on user profiles.
Pulls strategic profiles, focus areas, and tracked entities for personalized discovery.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..models.user import User
from ..models.strategic_profile import UserStrategicProfile, UserFocusArea
from ..models.tracking import UserEntityTracking
from ..models.delivery import UserDeliveryPreferences
from ..database import get_db_session
from .engines.base_engine import DiscoveredItem
from .ml_integration import DiscoveryMLIntegrator


@dataclass
class UserDiscoveryProfile:
    """Comprehensive user profile for discovery targeting."""
    user_id: int
    strategic_profile: Optional[UserStrategicProfile]
    focus_areas: List[UserFocusArea]
    tracked_entities: List[UserEntityTracking]
    delivery_preferences: Optional[UserDeliveryPreferences]
    discovery_keywords: List[str]
    priority_entities: List[str]
    preferred_sources: List[str]
    content_preferences: Dict[str, Any]


@dataclass
class DiscoveryPreferences:
    """User preferences for content discovery."""
    preferred_content_types: List[str]
    preferred_sources: List[str]
    quality_threshold: float
    recency_preference: str  # 'latest', 'balanced', 'comprehensive'
    max_items_per_source: int
    language_preference: str
    geographic_focus: Optional[str]


class UserConfigIntegrator:
    """Integrates with User Config Service for targeted discovery."""
    
    def __init__(self, ml_integrator: DiscoveryMLIntegrator):
        self.ml_integrator = ml_integrator
        self.logger = logging.getLogger("discovery.user_config_integrator")
        
        # Cache user profiles to reduce database queries
        self.profile_cache: Dict[int, Tuple[UserDiscoveryProfile, datetime]] = {}
        self.cache_duration = timedelta(minutes=15)  # 15-minute cache
    
    async def get_user_discovery_profile(self, user_id: int) -> Optional[UserDiscoveryProfile]:
        """Get comprehensive user discovery profile."""
        now = datetime.now()
        
        # Check cache first
        if user_id in self.profile_cache:
            profile, cached_time = self.profile_cache[user_id]
            if now - cached_time < self.cache_duration:
                return profile
        
        try:
            async for db_session in get_db_session():
                profile = await self._fetch_user_profile(db_session, user_id)
                
                if profile:
                    # Cache the profile
                    self.profile_cache[user_id] = (profile, now)
                
                return profile
                
        except Exception as e:
            self.logger.error(f"Failed to get user discovery profile for user {user_id}: {e}")
            return None
    
    async def _fetch_user_profile(self, db_session: AsyncSession, 
                                 user_id: int) -> Optional[UserDiscoveryProfile]:
        """Fetch user profile from database."""
        try:
            # Get user with all related data
            user_query = select(User).where(User.id == user_id).options(
                selectinload(User.strategic_profile),
                selectinload(User.focus_areas),
                selectinload(User.entity_tracking),
                selectinload(User.delivery_preferences)
            )
            
            result = await db_session.execute(user_query)
            user = result.scalar_one_or_none()
            
            if not user:
                self.logger.warning(f"User {user_id} not found")
                return None
            
            # Build discovery keywords from profile
            keywords = await self._extract_discovery_keywords(user)
            
            # Get priority entities
            priority_entities = await self._get_priority_entities(user)
            
            # Get preferred sources
            preferred_sources = await self._get_preferred_sources(user)
            
            # Build content preferences
            content_preferences = await self._build_content_preferences(user)
            
            profile = UserDiscoveryProfile(
                user_id=user_id,
                strategic_profile=user.strategic_profile,
                focus_areas=user.focus_areas or [],
                tracked_entities=user.entity_tracking or [],
                delivery_preferences=user.delivery_preferences,
                discovery_keywords=keywords,
                priority_entities=priority_entities,
                preferred_sources=preferred_sources,
                content_preferences=content_preferences
            )
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Database query failed for user {user_id}: {e}")
            return None
    
    async def _extract_discovery_keywords(self, user: User) -> List[str]:
        """Extract discovery keywords from user profile."""
        keywords = set()
        
        # Strategic profile keywords
        if user.strategic_profile:
            if user.strategic_profile.industry:
                keywords.update(user.strategic_profile.industry.split())
            
            if user.strategic_profile.role:
                keywords.update(user.strategic_profile.role.split())
            
            if user.strategic_profile.strategic_goals:
                for goal in user.strategic_profile.strategic_goals:
                    keywords.update(goal.split())
        
        # Focus area keywords
        if user.focus_areas:
            for focus_area in user.focus_areas:
                keywords.update(focus_area.focus_area.split())
                if focus_area.keywords:
                    keywords.update(focus_area.keywords)
        
        # Entity keywords
        if user.entity_tracking:
            for entity in user.entity_tracking:
                keywords.add(entity.entity_name)
                if entity.keywords:
                    keywords.update(entity.keywords)
        
        # Clean and filter keywords
        cleaned_keywords = [
            keyword.lower().strip()
            for keyword in keywords
            if len(keyword.strip()) > 2  # Filter out very short words
        ]
        
        return list(set(cleaned_keywords))[:20]  # Limit to top 20 keywords
    
    async def _get_priority_entities(self, user: User) -> List[str]:
        """Get high-priority entities for focused discovery."""
        if not user.entity_tracking:
            return []
        
        # Sort by priority and get top entities
        priority_entities = sorted(
            user.entity_tracking,
            key=lambda e: (e.priority, e.entity_name),
            reverse=True
        )
        
        return [entity.entity_name for entity in priority_entities[:10]]
    
    async def _get_preferred_sources(self, user: User) -> List[str]:
        """Get user's preferred content sources."""
        preferred_sources = []
        
        # Industry-specific sources
        if user.strategic_profile and user.strategic_profile.industry:
            industry = user.strategic_profile.industry.lower()
            
            if 'technology' in industry or 'tech' in industry:
                preferred_sources.extend([
                    'techcrunch.com', 'wired.com', 'arstechnica.com',
                    'venturebeat.com', 'theverge.com'
                ])
            elif 'finance' in industry or 'banking' in industry:
                preferred_sources.extend([
                    'bloomberg.com', 'reuters.com', 'wsj.com',
                    'ft.com', 'marketwatch.com'
                ])
            elif 'healthcare' in industry or 'medical' in industry:
                preferred_sources.extend([
                    'nature.com', 'nejm.org', 'bmj.com',
                    'healthline.com', 'medscape.com'
                ])
            elif 'energy' in industry:
                preferred_sources.extend([
                    'energycentral.com', 'oilprice.com',
                    'renewableenergyworld.com'
                ])
        
        # Add general authoritative sources
        preferred_sources.extend([
            'reuters.com', 'ap.org', 'bbc.com', 'nytimes.com'
        ])
        
        return list(set(preferred_sources))
    
    async def _build_content_preferences(self, user: User) -> Dict[str, Any]:
        """Build content preferences based on user profile."""
        preferences = {
            'quality_threshold': 0.6,  # Default quality threshold
            'recency_preference': 'balanced',
            'content_types': ['news', 'blog', 'research'],
            'max_items_per_source': 5,
            'language': 'en'
        }
        
        # Adjust based on role
        if user.strategic_profile and user.strategic_profile.role:
            role = user.strategic_profile.role.lower()
            
            if any(term in role for term in ['executive', 'ceo', 'president', 'director']):
                preferences['quality_threshold'] = 0.7
                preferences['recency_preference'] = 'latest'
                preferences['content_types'] = ['news', 'research', 'press_release']
                
            elif any(term in role for term in ['analyst', 'researcher', 'scientist']):
                preferences['quality_threshold'] = 0.8
                preferences['recency_preference'] = 'comprehensive'
                preferences['content_types'] = ['research', 'blog', 'news']
                preferences['max_items_per_source'] = 10
                
            elif any(term in role for term in ['manager', 'lead', 'supervisor']):
                preferences['quality_threshold'] = 0.6
                preferences['recency_preference'] = 'balanced'
                preferences['max_items_per_source'] = 7
        
        # Adjust based on delivery preferences
        if user.delivery_preferences:
            if hasattr(user.delivery_preferences, 'content_format'):
                if user.delivery_preferences.content_format == 'summary':
                    preferences['max_items_per_source'] = 3
                elif user.delivery_preferences.content_format == 'detailed':
                    preferences['max_items_per_source'] = 8
        
        return preferences
    
    async def get_discovery_preferences(self, user_id: int) -> Optional[DiscoveryPreferences]:
        """Get discovery preferences for a user."""
        profile = await self.get_user_discovery_profile(user_id)
        if not profile:
            return None
        
        content_prefs = profile.content_preferences
        
        return DiscoveryPreferences(
            preferred_content_types=content_prefs.get('content_types', ['news', 'blog']),
            preferred_sources=profile.preferred_sources,
            quality_threshold=content_prefs.get('quality_threshold', 0.6),
            recency_preference=content_prefs.get('recency_preference', 'balanced'),
            max_items_per_source=content_prefs.get('max_items_per_source', 5),
            language_preference=content_prefs.get('language', 'en'),
            geographic_focus=self._get_geographic_focus(profile)
        )
    
    def _get_geographic_focus(self, profile: UserDiscoveryProfile) -> Optional[str]:
        """Determine geographic focus from user profile."""
        # This could be enhanced to extract geographic preferences
        # from strategic profile or explicit user settings
        return None  # Default to no geographic focus
    
    async def personalize_discovered_items(self, items: List[DiscoveredItem],
                                         user_id: int) -> List[DiscoveredItem]:
        """Personalize discovered items based on user profile."""
        if not items:
            return items
        
        try:
            # Get user preferences
            preferences = await self.get_discovery_preferences(user_id)
            if not preferences:
                return items
            
            # Filter by quality threshold
            quality_filtered = [
                item for item in items
                if (item.quality_score or 0.0) >= preferences.quality_threshold
            ]
            
            # Apply ML scoring
            ml_scored = await self.ml_integrator.score_discovered_items(quality_filtered, user_id)
            
            # Sort by personalized relevance
            personalized_items = await self._apply_personalization(ml_scored, preferences)
            
            return personalized_items
            
        except Exception as e:
            self.logger.error(f"Personalization failed for user {user_id}: {e}")
            return items
    
    async def _apply_personalization(self, items: List[DiscoveredItem],
                                   preferences: DiscoveryPreferences) -> List[DiscoveredItem]:
        """Apply personalization logic to discovered items."""
        # Group by source
        items_by_source = {}
        for item in items:
            source = item.source_name
            if source not in items_by_source:
                items_by_source[source] = []
            items_by_source[source].append(item)
        
        # Apply per-source limits
        limited_items = []
        for source, source_items in items_by_source.items():
            # Sort source items by relevance
            source_items.sort(key=lambda x: x.relevance_score or 0.0, reverse=True)
            
            # Apply source preference boost
            if source in preferences.preferred_sources:
                for item in source_items:
                    item.relevance_score = (item.relevance_score or 0.0) * 1.2
            
            # Limit items per source
            limited_items.extend(source_items[:preferences.max_items_per_source])
        
        # Apply recency preferences
        if preferences.recency_preference == 'latest':
            # Boost recent content
            for item in limited_items:
                if item.published_at:
                    hours_old = (datetime.now() - item.published_at).total_seconds() / 3600
                    if hours_old < 24:
                        item.relevance_score = (item.relevance_score or 0.0) * 1.3
        elif preferences.recency_preference == 'comprehensive':
            # More balanced approach, slight preference for variety
            pass  # No specific adjustment needed
        
        # Final sort by relevance score
        limited_items.sort(key=lambda x: x.relevance_score or 0.0, reverse=True)
        
        return limited_items
    
    async def get_user_discovery_keywords(self, user_id: int) -> List[str]:
        """Get discovery keywords for a user."""
        profile = await self.get_user_discovery_profile(user_id)
        return profile.discovery_keywords if profile else []
    
    async def get_user_focus_areas(self, user_id: int) -> List[Dict[str, Any]]:
        """Get focus areas for a user."""
        profile = await self.get_user_discovery_profile(user_id)
        if not profile or not profile.focus_areas:
            return []
        
        return [
            {
                'focus_area': fa.focus_area,
                'keywords': fa.keywords or [],
                'priority': fa.priority,
                'priority_label': fa.priority_label
            }
            for fa in profile.focus_areas
        ]
    
    async def get_user_tracked_entities(self, user_id: int) -> List[Dict[str, Any]]:
        """Get tracked entities for a user."""
        profile = await self.get_user_discovery_profile(user_id)
        if not profile or not profile.tracked_entities:
            return []
        
        return [
            {
                'entity_name': entity.entity_name,
                'entity_type': entity.entity_type,
                'keywords': entity.keywords or [],
                'priority': entity.priority
            }
            for entity in profile.tracked_entities
        ]
    
    async def update_user_preferences_from_engagement(self, user_id: int,
                                                    engagement_data: Dict[str, Any]):
        """Update user preferences based on engagement patterns."""
        try:
            # This could analyze engagement patterns and adjust user preferences
            # For now, just invalidate cache to ensure fresh data
            self.invalidate_user_cache(user_id)
            
            # Update ML models
            await self.ml_integrator.update_ml_models_from_engagement(user_id)
            
            self.logger.info(f"Updated preferences from engagement for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update preferences from engagement: {e}")
    
    def invalidate_user_cache(self, user_id: int):
        """Invalidate cached user profile."""
        if user_id in self.profile_cache:
            del self.profile_cache[user_id]
        
        # Also invalidate ML integrator cache
        self.ml_integrator.invalidate_user_context_cache(user_id)
    
    def invalidate_all_caches(self):
        """Invalidate all cached user profiles."""
        self.profile_cache.clear()
        self.logger.info("Cleared all user profile caches")
    
    async def get_bulk_discovery_profiles(self, user_ids: List[int]) -> Dict[int, UserDiscoveryProfile]:
        """Get discovery profiles for multiple users efficiently."""
        profiles = {}
        uncached_users = []
        
        # Check cache first
        now = datetime.now()
        for user_id in user_ids:
            if user_id in self.profile_cache:
                profile, cached_time = self.profile_cache[user_id]
                if now - cached_time < self.cache_duration:
                    profiles[user_id] = profile
                else:
                    uncached_users.append(user_id)
            else:
                uncached_users.append(user_id)
        
        # Fetch uncached profiles
        if uncached_users:
            try:
                async for db_session in get_db_session():
                    for user_id in uncached_users:
                        profile = await self._fetch_user_profile(db_session, user_id)
                        if profile:
                            profiles[user_id] = profile
                            self.profile_cache[user_id] = (profile, now)
                        
            except Exception as e:
                self.logger.error(f"Bulk profile fetch failed: {e}")
        
        return profiles
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_profiles': len(self.profile_cache),
            'cache_duration_minutes': self.cache_duration.total_seconds() / 60,
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder implementation)."""
        # This would need proper tracking in a real implementation
        return 0.85  # Placeholder value
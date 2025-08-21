"""
ML integration layer for discovery engines with the Discovery Service ML models.
Provides intelligent relevance scoring and content quality assessment.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..services.discovery_service import DiscoveryService, UserContext
from ..models.discovery import DiscoveredContent, ContentEngagement
from .engines.base_engine import DiscoveredItem, ContentType, SourceType


@dataclass
class MLScoringResult:
    """ML scoring result for discovered content."""
    relevance_score: float
    credibility_score: float
    freshness_score: float
    overall_score: float
    engagement_prediction: float
    ml_confidence: float
    scoring_factors: Dict[str, float]


class DiscoveryMLIntegrator:
    """Integrates discovery engines with ML models from Discovery Service."""
    
    def __init__(self, discovery_service: DiscoveryService):
        self.discovery_service = discovery_service
        self.logger = logging.getLogger("discovery.ml_integrator")
        
        # Cache for user contexts to avoid repeated database queries
        self.user_context_cache: Dict[int, Tuple[UserContext, datetime]] = {}
        self.context_cache_duration = 300  # 5 minutes
    
    async def score_discovered_items(self, items: List[DiscoveredItem], 
                                   user_id: int) -> List[DiscoveredItem]:
        """Score discovered items using ML models and update their scores."""
        if not items:
            return items
        
        try:
            # Get user context
            user_context = await self._get_user_context(user_id)
            if not user_context:
                self.logger.warning(f"Could not get user context for user {user_id}")
                return items
            
            # Score each item
            scored_items = []
            for item in items:
                try:
                    ml_result = await self._score_single_item(item, user_context)
                    
                    # Update item scores
                    item.relevance_score = ml_result.relevance_score
                    item.quality_score = max(item.quality_score, ml_result.overall_score)
                    
                    # Add ML metadata
                    item.metadata.update({
                        'ml_credibility_score': ml_result.credibility_score,
                        'ml_freshness_score': ml_result.freshness_score,
                        'ml_engagement_prediction': ml_result.engagement_prediction,
                        'ml_confidence': ml_result.ml_confidence,
                        'ml_scoring_factors': ml_result.scoring_factors,
                        'scored_at': datetime.now().isoformat()
                    })
                    
                    scored_items.append(item)
                    
                except Exception as e:
                    self.logger.error(f"Failed to score item {item.url}: {e}")
                    # Keep original item if scoring fails
                    scored_items.append(item)
            
            return scored_items
            
        except Exception as e:
            self.logger.error(f"ML scoring failed for user {user_id}: {e}")
            return items
    
    async def _get_user_context(self, user_id: int) -> Optional[UserContext]:
        """Get user context with caching."""
        now = datetime.now()
        
        # Check cache first
        if user_id in self.user_context_cache:
            cached_context, cached_time = self.user_context_cache[user_id]
            if (now - cached_time).total_seconds() < self.context_cache_duration:
                return cached_context
        
        try:
            # Fetch fresh context
            user_context = await self.discovery_service._get_user_context(user_id)
            
            # Cache the result
            self.user_context_cache[user_id] = (user_context, now)
            
            return user_context
            
        except Exception as e:
            self.logger.error(f"Failed to get user context for user {user_id}: {e}")
            return None
    
    async def _score_single_item(self, item: DiscoveredItem, 
                               user_context: UserContext) -> MLScoringResult:
        """Score a single discovered item using ML models."""
        try:
            # Convert DiscoveredItem to the format expected by ML models
            content_data = self._item_to_content_data(item)
            
            # Calculate relevance score
            relevance_score = await self.discovery_service._calculate_relevance_score(
                content_data, user_context
            )
            
            # Calculate credibility score
            credibility_score = await self.discovery_service._assess_content_credibility(
                content_data
            )
            
            # Calculate freshness score
            freshness_score = await self.discovery_service._calculate_freshness_score(
                content_data
            )
            
            # Calculate engagement prediction
            engagement_prediction = await self.discovery_service._predict_engagement_score(
                content_data, user_context
            )
            
            # Calculate overall score
            overall_score = await self.discovery_service._calculate_overall_content_score(
                relevance_score, credibility_score, freshness_score, engagement_prediction
            )
            
            # Calculate ML confidence
            ml_confidence = await self.discovery_service._calculate_ml_confidence(
                content_data, user_context
            )
            
            # Detailed scoring factors for transparency
            scoring_factors = {
                'strategic_profile_match': await self._calculate_strategic_match(content_data, user_context),
                'focus_area_relevance': await self._calculate_focus_area_relevance(content_data, user_context),
                'entity_relevance': await self._calculate_entity_relevance(content_data, user_context),
                'content_quality': await self._assess_content_quality(content_data),
                'source_authority': await self._assess_source_authority(content_data),
                'recency_bonus': await self._calculate_recency_bonus(content_data)
            }
            
            return MLScoringResult(
                relevance_score=relevance_score,
                credibility_score=credibility_score,
                freshness_score=freshness_score,
                overall_score=overall_score,
                engagement_prediction=engagement_prediction,
                ml_confidence=ml_confidence,
                scoring_factors=scoring_factors
            )
            
        except Exception as e:
            self.logger.error(f"ML scoring failed for item {item.url}: {e}")
            # Return default scores if ML scoring fails
            return MLScoringResult(
                relevance_score=0.5,
                credibility_score=0.5,
                freshness_score=0.5,
                overall_score=0.5,
                engagement_prediction=0.5,
                ml_confidence=0.1,  # Low confidence for fallback
                scoring_factors={}
            )
    
    def _item_to_content_data(self, item: DiscoveredItem) -> Dict[str, Any]:
        """Convert DiscoveredItem to content data format expected by ML models."""
        return {
            'title': item.title,
            'content': item.content,
            'url': item.url,
            'source_name': item.source_name,
            'source_type': item.source_type.value,
            'content_type': item.content_type.value,
            'published_at': item.published_at,
            'author': item.author,
            'description': item.description,
            'keywords': item.keywords,
            'metadata': item.metadata
        }
    
    async def _calculate_strategic_match(self, content_data: Dict[str, Any], 
                                       user_context: UserContext) -> float:
        """Calculate how well content matches user's strategic profile."""
        if not user_context.strategic_profile:
            return 0.0
        
        text_content = f"{content_data.get('title', '')} {content_data.get('content', '')}".lower()
        
        score = 0.0
        
        # Industry match
        if user_context.strategic_profile.industry:
            industry_terms = user_context.strategic_profile.industry.lower().split()
            for term in industry_terms:
                if term in text_content:
                    score += 0.3
        
        # Role relevance
        if user_context.strategic_profile.role:
            role_terms = user_context.strategic_profile.role.lower().split()
            for term in role_terms:
                if term in text_content:
                    score += 0.2
        
        # Strategic goals alignment
        if user_context.strategic_profile.strategic_goals:
            for goal in user_context.strategic_profile.strategic_goals:
                goal_terms = goal.lower().split()
                for term in goal_terms:
                    if term in text_content:
                        score += 0.1
        
        return min(score, 1.0)
    
    async def _calculate_focus_area_relevance(self, content_data: Dict[str, Any],
                                            user_context: UserContext) -> float:
        """Calculate relevance to user's focus areas."""
        if not user_context.focus_areas:
            return 0.0
        
        text_content = f"{content_data.get('title', '')} {content_data.get('content', '')}".lower()
        
        max_relevance = 0.0
        
        for focus_area in user_context.focus_areas:
            area_score = 0.0
            
            # Focus area name match
            if focus_area.focus_area.lower() in text_content:
                area_score += 0.5
            
            # Keywords match
            if focus_area.keywords:
                keyword_matches = sum(
                    1 for keyword in focus_area.keywords
                    if keyword.lower() in text_content
                )
                area_score += min(keyword_matches * 0.2, 0.4)
            
            # Weight by priority
            priority_weight = focus_area.priority / 4.0  # Normalize priority (1-4 scale)
            weighted_score = area_score * priority_weight
            
            max_relevance = max(max_relevance, weighted_score)
        
        return min(max_relevance, 1.0)
    
    async def _calculate_entity_relevance(self, content_data: Dict[str, Any],
                                        user_context: UserContext) -> float:
        """Calculate relevance to tracked entities."""
        if not user_context.tracked_entities:
            return 0.0
        
        text_content = f"{content_data.get('title', '')} {content_data.get('content', '')}".lower()
        
        max_relevance = 0.0
        
        for entity in user_context.tracked_entities:
            entity_score = 0.0
            
            # Entity name match
            if entity.entity_name.lower() in text_content:
                entity_score += 0.6
            
            # Entity keywords match
            if entity.keywords:
                keyword_matches = sum(
                    1 for keyword in entity.keywords
                    if keyword.lower() in text_content
                )
                entity_score += min(keyword_matches * 0.1, 0.3)
            
            # Weight by priority
            priority_weight = entity.priority / 4.0
            weighted_score = entity_score * priority_weight
            
            max_relevance = max(max_relevance, weighted_score)
        
        return min(max_relevance, 1.0)
    
    async def _assess_content_quality(self, content_data: Dict[str, Any]) -> float:
        """Assess content quality based on various factors."""
        score = 0.0
        
        title = content_data.get('title', '')
        content = content_data.get('content', '')
        
        # Title quality
        if title and len(title) > 10:
            score += 0.2
            if not any(spam in title.lower() for spam in ['click', 'amazing', 'shocking']):
                score += 0.1
        
        # Content length and quality
        if len(content) > 200:
            score += 0.3
        if len(content) > 1000:
            score += 0.2
        
        # Author presence
        if content_data.get('author'):
            score += 0.1
        
        # Metadata richness
        if content_data.get('description'):
            score += 0.1
        if content_data.get('keywords'):
            score += 0.1
        
        return min(score, 1.0)
    
    async def _assess_source_authority(self, content_data: Dict[str, Any]) -> float:
        """Assess source authority and credibility."""
        source_name = content_data.get('source_name', '').lower()
        url = content_data.get('url', '').lower()
        
        # Authoritative domains get higher scores
        authoritative_domains = [
            'reuters.com', 'ap.org', 'bbc.com', 'nytimes.com', 'wsj.com',
            'bloomberg.com', 'ft.com', 'economist.com', 'nature.com',
            'science.org', 'ieee.org', 'harvard.edu', 'mit.edu'
        ]
        
        for domain in authoritative_domains:
            if domain in url or domain in source_name:
                return 0.9
        
        # News sources generally get medium authority
        if any(indicator in url for indicator in ['news', 'press', 'media']):
            return 0.6
        
        # Academic and research sources
        if any(indicator in url for indicator in ['.edu', '.org', 'research', 'study']):
            return 0.7
        
        # Default authority
        return 0.5
    
    async def _calculate_recency_bonus(self, content_data: Dict[str, Any]) -> float:
        """Calculate bonus for recent content."""
        published_at = content_data.get('published_at')
        if not published_at:
            return 0.0
        
        if isinstance(published_at, str):
            try:
                published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            except:
                return 0.0
        
        hours_old = (datetime.now() - published_at).total_seconds() / 3600
        
        if hours_old < 1:
            return 0.3  # Very fresh content
        elif hours_old < 6:
            return 0.2  # Recent content
        elif hours_old < 24:
            return 0.1  # Today's content
        elif hours_old < 72:
            return 0.05  # This week's content
        else:
            return 0.0  # Older content
    
    async def record_user_engagement(self, user_id: int, item: DiscoveredItem,
                                   engagement_type: str, engagement_value: float):
        """Record user engagement for ML learning."""
        try:
            # Find or create content record
            content_record = await self.discovery_service._get_or_create_content_record(
                item, user_id
            )
            
            if content_record:
                # Record engagement
                await self.discovery_service._record_content_engagement(
                    user_id=user_id,
                    content_id=content_record.id,
                    engagement_type=engagement_type,
                    engagement_value=engagement_value
                )
                
                self.logger.info(f"Recorded engagement for user {user_id}: {engagement_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to record engagement: {e}")
    
    async def update_ml_models_from_engagement(self, user_id: int):
        """Update ML models based on recent user engagement."""
        try:
            await self.discovery_service._update_user_preferences_from_engagement(user_id)
            self.logger.info(f"Updated ML models from engagement for user {user_id}")
        except Exception as e:
            self.logger.error(f"Failed to update ML models: {e}")
    
    def invalidate_user_context_cache(self, user_id: int):
        """Invalidate cached user context."""
        if user_id in self.user_context_cache:
            del self.user_context_cache[user_id]
    
    def get_ml_performance_metrics(self) -> Dict[str, Any]:
        """Get ML integration performance metrics."""
        return {
            'cached_user_contexts': len(self.user_context_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'average_scoring_time': self._get_average_scoring_time(),
            'ml_service_status': 'active'
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder implementation)."""
        # This would need proper tracking in a real implementation
        return 0.75  # Placeholder value
    
    def _get_average_scoring_time(self) -> float:
        """Get average ML scoring time (placeholder implementation)."""
        # This would need proper timing tracking in a real implementation
        return 0.15  # Placeholder value in seconds
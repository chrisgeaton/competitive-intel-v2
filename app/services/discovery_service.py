"""
Discovery Service with ML learning algorithms for competitive intelligence v2.

Advanced ML-driven content discovery, relevance scoring, and user behavior learning
with SendGrid engagement integration and continuous algorithm improvement.
"""

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from urllib.parse import urlparse, urljoin
# import numpy as np  # Not needed for basic operations
from dataclasses import dataclass
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, and_, or_, desc
from sqlalchemy.orm import selectinload

from app.models.discovery import (
    DiscoveredSource, DiscoveredContent, ContentEngagement, 
    DiscoveryJob, MLModelMetrics
)
from app.models.user import User
from app.models.strategic_profile import UserStrategicProfile, UserFocusArea
from app.models.tracking import UserEntityTracking
from app.models.delivery import UserDeliveryPreferences
from app.utils.router_base import BaseRouterOperations
from app.utils.exceptions import errors

# Import shared discovery utilities
from app.discovery.utils import (
    ContentUtils, get_user_context_cache, get_content_processing_cache,
    get_ml_scoring_cache, DiscoveryConfig, get_config,
    AsyncBatchProcessor, batch_processor
)


@dataclass
class UserContext:
    """User context for ML-driven discovery optimization."""
    user_id: int
    strategic_profile: Optional[Dict[str, Any]]
    focus_areas: List[Dict[str, Any]]
    tracked_entities: List[Dict[str, Any]]
    delivery_preferences: Optional[Dict[str, Any]]
    engagement_history: Dict[str, float]
    ml_preferences: Dict[str, float]


@dataclass
class ContentSimilarity:
    """Content similarity metrics for deduplication."""
    content_id: int
    similarity_score: float
    duplicate_type: str  # 'exact', 'near_duplicate', 'similar'
    matching_features: List[str]


@dataclass
class MLScores:
    """ML-generated scores for content evaluation."""
    relevance_score: float
    credibility_score: float
    freshness_score: float
    engagement_prediction: float
    overall_score: float
    confidence_level: float
    model_version: str


class DiscoveryService(BaseRouterOperations):
    """
    ML-driven competitive intelligence discovery service.
    
    Features:
    - Intelligent source discovery based on user context
    - ML relevance scoring with continuous learning
    - SendGrid engagement data integration
    - Content similarity deduplication
    - User behavior correlation and pattern learning
    """
    
    def __init__(self):
        super().__init__("discovery_service")
        
        # Use centralized configuration
        self.config = get_config()
        
        # ML Configuration from centralized config
        self.ml_model_version = self.config.ml.model_version
        self.similarity_threshold = self.config.ml.similarity_threshold
        self.relevance_threshold = self.config.ml.relevance_threshold
        
        # Enhanced engagement weights from config
        self.engagement_weights = {
            'email_open': self.config.ml.email_open_weight,
            'email_click': self.config.ml.email_click_weight,
            'time_spent': self.config.ml.time_spent_weight,
            'manual_feedback': self.config.ml.manual_feedback_weight,
            'bookmark': 5.0,
            'share': 7.0,
            'feedback_positive': 10.0,
            'feedback_negative': -5.0
        }
        
        # Use shared utilities
        self.batch_processor = AsyncBatchProcessor(
            batch_size=self.config.performance.batch_size,
            max_concurrent=self.config.performance.max_concurrent_engines,
            timeout=self.config.performance.default_timeout
        )
        
        # Get caches
        self.user_context_cache = get_user_context_cache()
        self.content_processing_cache = get_content_processing_cache()
        self.ml_scoring_cache = get_ml_scoring_cache()
    
    async def get_user_context(self, db: AsyncSession, user_id: int) -> UserContext:
        """
        Retrieve comprehensive user context for ML-driven discovery.
        
        Combines strategic profile, focus areas, tracked entities, and
        historical engagement data for personalized content discovery.
        """
        # Check cache first
        cache_key = f"user_context_{user_id}"
        cached_context = self.user_context_cache.get(cache_key)
        if cached_context:
            return cached_context
        
        # Get strategic profile
        strategic_profile_result = await db.execute(
            select(UserStrategicProfile).where(UserStrategicProfile.user_id == user_id)
        )
        strategic_profile = strategic_profile_result.scalar_one_or_none()
        
        # Get focus areas with keywords
        focus_areas_result = await db.execute(
            select(UserFocusArea).where(UserFocusArea.user_id == user_id)
        )
        focus_areas = focus_areas_result.scalars().all()
        
        # Get tracked entities
        entities_result = await db.execute(
            select(UserEntityTracking).where(UserEntityTracking.user_id == user_id)
        )
        tracked_entities = entities_result.scalars().all()
        
        # Get delivery preferences
        delivery_prefs_result = await db.execute(
            select(UserDeliveryPreferences).where(UserDeliveryPreferences.user_id == user_id)
        )
        delivery_preferences = delivery_prefs_result.scalar_one_or_none()
        
        # Calculate engagement history and ML preferences
        engagement_history = await self._calculate_engagement_history(db, user_id)
        ml_preferences = await self._calculate_ml_preferences(db, user_id)
        
        user_context = UserContext(
            user_id=user_id,
            strategic_profile=strategic_profile.__dict__ if strategic_profile else None,
            focus_areas=[fa.__dict__ for fa in focus_areas],
            tracked_entities=[te.__dict__ for te in tracked_entities],
            delivery_preferences=delivery_preferences.__dict__ if delivery_preferences else None,
            engagement_history=engagement_history,
            ml_preferences=ml_preferences
        )
        
        # Cache the user context
        self.user_context_cache.put(cache_key, user_context)
        return user_context
    
    async def _calculate_engagement_history(self, db: AsyncSession, user_id: int) -> Dict[str, float]:
        """Calculate user engagement history for ML training."""
        # Get recent engagement data (last 90 days)
        cutoff_date = datetime.utcnow() - timedelta(days=90)
        
        engagement_result = await db.execute(
            select(
                ContentEngagement.engagement_type,
                func.avg(ContentEngagement.engagement_value).label('avg_value'),
                func.count(ContentEngagement.id).label('count')
            )
            .where(
                and_(
                    ContentEngagement.user_id == user_id,
                    ContentEngagement.created_at >= cutoff_date
                )
            )
            .group_by(ContentEngagement.engagement_type)
        )
        
        engagement_data = {}
        for row in engagement_result:
            weight = self.engagement_weights.get(row.engagement_type, 1.0)
            engagement_data[row.engagement_type] = float(row.avg_value * weight * row.count)
        
        return engagement_data
    
    async def _calculate_ml_preferences(self, db: AsyncSession, user_id: int) -> Dict[str, float]:
        """Calculate ML-derived user preferences based on behavior patterns."""
        # Get content with high engagement
        high_engagement_result = await db.execute(
            select(DiscoveredContent)
            .join(ContentEngagement)
            .where(
                and_(
                    ContentEngagement.user_id == user_id,
                    ContentEngagement.engagement_value > 5.0
                )
            )
            .options(selectinload(DiscoveredContent.engagements))
        )
        
        high_engagement_content = high_engagement_result.scalars().all()
        
        # Analyze patterns in high-engagement content
        patterns = {
            'preferred_content_length': 0.0,
            'preferred_freshness': 0.0,
            'preferred_credibility': 0.0,
            'category_preferences': {},
            'source_preferences': {}
        }
        
        if high_engagement_content:
            # Calculate average preferences from high-engagement content
            total_items = len(high_engagement_content)
            
            for content in high_engagement_content:
                patterns['preferred_freshness'] += float(content.freshness_score or 0.5)
                patterns['preferred_credibility'] += float(content.credibility_score or 0.5)
                
                # Track category and source preferences
                if content.predicted_categories:
                    try:
                        categories = json.loads(content.predicted_categories)
                        for category in categories:
                            patterns['category_preferences'][category] = patterns['category_preferences'].get(category, 0) + 1
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                if content.source_id:
                    patterns['source_preferences'][content.source_id] = patterns['source_preferences'].get(content.source_id, 0) + 1
            
            # Normalize averages
            patterns['preferred_freshness'] /= total_items
            patterns['preferred_credibility'] /= total_items
        
        return patterns
    
    async def calculate_ml_relevance_score(
        self,
        db: AsyncSession,
        content: DiscoveredContent,
        user_context: UserContext
    ) -> MLScores:
        """
        Calculate ML-driven relevance scores with continuous learning integration.
        
        Uses user behavior patterns, strategic context, and historical engagement
        to predict content relevance and user engagement likelihood.
        """
        # Initialize scores
        relevance_components = []
        credibility_components = []
        freshness_components = []
        engagement_predictions = []
        
        # 1. Strategic Profile Relevance
        if user_context.strategic_profile:
            strategic_relevance = await self._calculate_strategic_relevance(
                content, user_context.strategic_profile
            )
            relevance_components.append(('strategic', strategic_relevance, 0.3))
        
        # 2. Focus Areas Matching
        focus_relevance = await self._calculate_focus_areas_relevance(
            content, user_context.focus_areas
        )
        relevance_components.append(('focus_areas', focus_relevance, 0.4))
        
        # 3. Entity Tracking Relevance
        entity_relevance = await self._calculate_entity_relevance(
            content, user_context.tracked_entities
        )
        relevance_components.append(('entities', entity_relevance, 0.3))
        
        # 4. Historical Engagement Pattern Matching
        engagement_prediction = await self._predict_user_engagement(
            db, content, user_context
        )
        engagement_predictions.append(engagement_prediction)
        
        # 5. Content Credibility Assessment
        credibility_score = await self._assess_content_credibility(db, content)
        credibility_components.append(credibility_score)
        
        # 6. Content Freshness Scoring
        freshness_score = await self._calculate_freshness_score(content)
        freshness_components.append(freshness_score)
        
        # 7. User Preference Alignment
        preference_alignment = await self._calculate_preference_alignment(
            content, user_context.ml_preferences
        )
        relevance_components.append(('preferences', preference_alignment, 0.2))
        
        # Calculate weighted scores
        relevance_score = sum(score * weight for _, score, weight in relevance_components) / sum(weight for _, _, weight in relevance_components)
        credibility_score = sum(credibility_components) / len(credibility_components) if credibility_components else 0.5
        freshness_score = sum(freshness_components) / len(freshness_components) if freshness_components else 0.5
        engagement_prediction = sum(engagement_predictions) / len(engagement_predictions) if engagement_predictions else 0.5
        
        # Calculate overall composite score with ML weighting
        overall_score = (
            relevance_score * 0.4 +
            credibility_score * 0.2 +
            freshness_score * 0.15 +
            engagement_prediction * 0.25
        )
        
        # Calculate model confidence based on data availability
        confidence_level = await self._calculate_model_confidence(
            user_context, len(relevance_components), len(engagement_predictions)
        )
        
        return MLScores(
            relevance_score=float(relevance_score),
            credibility_score=float(credibility_score),
            freshness_score=float(freshness_score),
            engagement_prediction=float(engagement_prediction),
            overall_score=float(overall_score),
            confidence_level=float(confidence_level),
            model_version=self.ml_model_version
        )
    
    async def _calculate_strategic_relevance(
        self,
        content: DiscoveredContent,
        strategic_profile: Dict[str, Any]
    ) -> float:
        """Calculate relevance based on user's strategic business profile."""
        relevance_score = 0.0
        
        # Industry relevance
        if strategic_profile.get('industry_type') and content.content_text:
            industry_keywords = self._get_industry_keywords(strategic_profile['industry_type'])
            text_lower = content.content_text.lower()
            
            industry_matches = sum(1 for keyword in industry_keywords if keyword in text_lower)
            relevance_score += min(industry_matches / len(industry_keywords), 1.0) * 0.4
        
        # Organization type relevance
        if strategic_profile.get('organization_type'):
            org_relevance = self._calculate_org_type_relevance(
                content, strategic_profile['organization_type']
            )
            relevance_score += org_relevance * 0.3
        
        # Strategic goals alignment
        if strategic_profile.get('strategic_goals'):
            goals_relevance = self._calculate_goals_relevance(
                content, strategic_profile['strategic_goals']
            )
            relevance_score += goals_relevance * 0.3
        
        return min(relevance_score, 1.0)
    
    async def _calculate_focus_areas_relevance(
        self,
        content: DiscoveredContent,
        focus_areas: List[Dict[str, Any]]
    ) -> float:
        """Calculate relevance based on user's defined focus areas."""
        if not focus_areas:
            return 0.5  # Neutral score if no focus areas defined
        
        max_relevance = 0.0
        total_weight = 0.0
        
        for focus_area in focus_areas:
            area_relevance = 0.0
            area_weight = focus_area.get('priority_level', 2) / 4.0  # Normalize priority 1-4 to 0.25-1.0
            
            # Title and description matching
            if content.title:
                title_matches = self._calculate_text_similarity(
                    content.title.lower(),
                    focus_area.get('description', '').lower()
                )
                area_relevance += title_matches * 0.5
            
            # Keywords matching
            if focus_area.get('keywords') and content.content_text:
                try:
                    keywords = json.loads(focus_area['keywords']) if isinstance(focus_area['keywords'], str) else focus_area['keywords']
                    text_lower = content.content_text.lower()
                    
                    keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
                    keyword_relevance = min(keyword_matches / len(keywords), 1.0)
                    area_relevance += keyword_relevance * 0.5
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Weight by priority and accumulate
            weighted_relevance = area_relevance * area_weight
            max_relevance = max(max_relevance, weighted_relevance)
            total_weight += area_weight
        
        return min(max_relevance, 1.0)
    
    async def _calculate_entity_relevance(
        self,
        content: DiscoveredContent,
        tracked_entities: List[Dict[str, Any]]
    ) -> float:
        """Calculate relevance based on tracked entities (competitors, technologies, etc.)."""
        if not tracked_entities:
            return 0.5  # Neutral score if no entities tracked
        
        entity_matches = 0
        high_priority_matches = 0
        total_entities = len(tracked_entities)
        
        content_text = (content.title or '') + ' ' + (content.content_text or '')
        content_text_lower = content_text.lower()
        
        for entity in tracked_entities:
            entity_name = entity.get('entity_name', '').lower()
            entity_keywords = entity.get('keywords', [])
            entity_priority = entity.get('priority_level', 2)
            
            # Direct name matching
            if entity_name and entity_name in content_text_lower:
                entity_matches += 1
                if entity_priority >= 3:  # High priority entities
                    high_priority_matches += 1
            
            # Keywords matching
            if entity_keywords:
                try:
                    keywords = json.loads(entity_keywords) if isinstance(entity_keywords, str) else entity_keywords
                    for keyword in keywords:
                        if keyword.lower() in content_text_lower:
                            entity_matches += 0.5  # Partial match for keywords
                            if entity_priority >= 3:
                                high_priority_matches += 0.5
                            break
                except (json.JSONDecodeError, TypeError):
                    continue
        
        # Calculate relevance with priority weighting
        base_relevance = min(entity_matches / total_entities, 1.0)
        priority_boost = min(high_priority_matches / total_entities, 0.3)
        
        return min(base_relevance + priority_boost, 1.0)
    
    async def _predict_user_engagement(
        self,
        db: AsyncSession,
        content: DiscoveredContent,
        user_context: UserContext
    ) -> float:
        """Predict user engagement likelihood based on historical patterns."""
        engagement_prediction = 0.5  # Default neutral prediction
        
        # Historical engagement patterns
        if user_context.engagement_history:
            # Content type preferences
            content_type_engagement = user_context.engagement_history.get(
                f"content_type_{content.content_type}", 0
            )
            engagement_prediction += min(content_type_engagement / 100.0, 0.2)
            
            # Source-based engagement
            if content.source_id:
                source_engagement = user_context.engagement_history.get(
                    f"source_{content.source_id}", 0
                )
                engagement_prediction += min(source_engagement / 100.0, 0.2)
        
        # ML preferences alignment
        if user_context.ml_preferences:
            # Freshness preference matching
            freshness_diff = abs(
                float(content.freshness_score or 0.5) - 
                user_context.ml_preferences.get('preferred_freshness', 0.5)
            )
            engagement_prediction += (1.0 - freshness_diff) * 0.1
            
            # Credibility preference matching
            credibility_diff = abs(
                float(content.credibility_score or 0.5) - 
                user_context.ml_preferences.get('preferred_credibility', 0.5)
            )
            engagement_prediction += (1.0 - credibility_diff) * 0.1
        
        # Time-based engagement patterns
        current_hour = datetime.utcnow().hour
        user_active_hours = user_context.engagement_history.get('active_hours', {})
        hour_engagement = user_active_hours.get(str(current_hour), 0.5)
        engagement_prediction += hour_engagement * 0.1
        
        return min(engagement_prediction, 1.0)
    
    async def _assess_content_credibility(self, db: AsyncSession, content: DiscoveredContent) -> float:
        """Assess content credibility based on source and content analysis."""
        credibility_score = 0.5  # Start with neutral credibility
        
        # Source credibility
        if content.source:
            source_credibility = float(content.source.credibility_score or 0.5)
            credibility_score += source_credibility * 0.4
        
        # Author credibility (if available)
        if content.author:
            # Simple heuristic: known domains and author patterns
            author_credibility = self._assess_author_credibility(content.author)
            credibility_score += author_credibility * 0.2
        
        # Content quality indicators
        if content.content_text:
            content_quality = self._assess_content_quality(content.content_text)
            credibility_score += content_quality * 0.3
        
        # URL credibility
        if content.content_url:
            url_credibility = self._assess_url_credibility(content.content_url)
            credibility_score += url_credibility * 0.1
        
        return min(credibility_score, 1.0)
    
    async def _calculate_freshness_score(self, content: DiscoveredContent) -> float:
        """Calculate content freshness score based on publication and discovery dates."""
        if not content.published_at:
            # No publication date, use discovery date
            age_hours = (datetime.utcnow() - content.discovered_at).total_seconds() / 3600
        else:
            age_hours = (datetime.utcnow() - content.published_at).total_seconds() / 3600
        
        # Freshness decay function: exponential decay with configurable half-life
        half_life_hours = 24 * 7  # 1 week half-life
        freshness_score = 2 ** (-age_hours / half_life_hours)
        
        return min(freshness_score, 1.0)
    
    async def _calculate_preference_alignment(
        self,
        content: DiscoveredContent,
        ml_preferences: Dict[str, Any]
    ) -> float:
        """Calculate how well content aligns with learned user preferences."""
        if not ml_preferences:
            return 0.5
        
        alignment_score = 0.5
        
        # Category preferences
        if content.predicted_categories and ml_preferences.get('category_preferences'):
            try:
                categories = json.loads(content.predicted_categories)
                category_prefs = ml_preferences['category_preferences']
                
                category_score = 0.0
                for category in categories:
                    if category in category_prefs:
                        category_score += category_prefs[category] / sum(category_prefs.values())
                
                alignment_score += min(category_score, 0.3)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Source preferences
        if content.source_id and ml_preferences.get('source_preferences'):
            source_prefs = ml_preferences['source_preferences']
            if content.source_id in source_prefs:
                source_score = source_prefs[content.source_id] / sum(source_prefs.values())
                alignment_score += min(source_score, 0.2)
        
        return min(alignment_score, 1.0)
    
    async def _calculate_model_confidence(
        self,
        user_context: UserContext,
        num_relevance_components: int,
        num_engagement_predictions: int
    ) -> float:
        """Calculate ML model confidence based on available data."""
        confidence = 0.5  # Base confidence
        
        # Data availability boosts confidence
        if user_context.strategic_profile:
            confidence += 0.1
        
        if user_context.focus_areas:
            confidence += 0.1
        
        if user_context.tracked_entities:
            confidence += 0.1
        
        if user_context.engagement_history:
            confidence += 0.15
        
        # Component availability
        confidence += min(num_relevance_components * 0.05, 0.15)
        confidence += min(num_engagement_predictions * 0.05, 0.1)
        
        return min(confidence, 1.0)
    
    async def detect_content_similarity(
        self,
        db: AsyncSession,
        new_content: DiscoveredContent,
        user_id: int,
        days_back: int = 30
    ) -> List[ContentSimilarity]:
        """
        Detect similar content for deduplication using multiple similarity algorithms.
        
        Uses URL similarity, content hash matching, and semantic similarity
        to identify potential duplicates and near-duplicates.
        """
        similarities = []
        
        # Get recent content for the user
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        existing_content_result = await db.execute(
            select(DiscoveredContent)
            .where(
                and_(
                    DiscoveredContent.user_id == user_id,
                    DiscoveredContent.discovered_at >= cutoff_date,
                    DiscoveredContent.id != new_content.id
                )
            )
        )
        
        existing_content = existing_content_result.scalars().all()
        
        for existing in existing_content:
            # 1. Exact URL matching
            if new_content.content_url == existing.content_url:
                similarities.append(ContentSimilarity(
                    content_id=existing.id,
                    similarity_score=1.0,
                    duplicate_type='exact',
                    matching_features=['url']
                ))
                continue
            
            # 2. URL similarity (domain + path analysis)
            url_similarity = self._calculate_url_similarity(
                new_content.content_url, existing.content_url
            )
            
            # 3. Content hash similarity
            hash_similarity = 0.0
            if new_content.content_hash and existing.content_hash:
                hash_similarity = 1.0 if new_content.content_hash == existing.content_hash else 0.0
            
            # 4. Title similarity
            title_similarity = 0.0
            if new_content.title and existing.title:
                title_similarity = self._calculate_text_similarity(
                    new_content.title, existing.title
                )
            
            # 5. Content text similarity (if available)
            content_similarity = 0.0
            if new_content.content_text and existing.content_text:
                content_similarity = self._calculate_text_similarity(
                    new_content.content_text[:1000],  # First 1000 chars for performance
                    existing.content_text[:1000]
                )
            
            # Calculate overall similarity
            similarity_components = [
                ('url', url_similarity, 0.3),
                ('hash', hash_similarity, 0.2),
                ('title', title_similarity, 0.3),
                ('content', content_similarity, 0.2)
            ]
            
            overall_similarity = sum(
                score * weight for _, score, weight in similarity_components
                if score > 0
            ) / sum(weight for _, score, weight in similarity_components if score > 0)
            
            # Determine duplicate type
            duplicate_type = 'similar'
            matching_features = [feature for feature, score, _ in similarity_components if score > 0.7]
            
            if overall_similarity >= 0.95:
                duplicate_type = 'exact'
            elif overall_similarity >= self.similarity_threshold:
                duplicate_type = 'near_duplicate'
            
            if overall_similarity >= 0.5:  # Only include reasonably similar content
                similarities.append(ContentSimilarity(
                    content_id=existing.id,
                    similarity_score=overall_similarity,
                    duplicate_type=duplicate_type,
                    matching_features=matching_features
                ))
        
        # Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return similarities
    
    async def process_sendgrid_engagement(
        self,
        db: AsyncSession,
        engagement_data: Dict[str, Any]
    ) -> ContentEngagement:
        """
        Process SendGrid engagement data for ML learning and content optimization.
        
        Transforms SendGrid webhook data into structured engagement records
        for ML model training and user behavior analysis.
        """
        # Extract core engagement information
        event_type = engagement_data.get('event')
        email = engagement_data.get('email')
        timestamp = engagement_data.get('timestamp')
        sg_event_id = engagement_data.get('sg_event_id')
        sg_message_id = engagement_data.get('sg_message_id')
        
        # Map SendGrid events to engagement types
        event_mapping = {
            'open': 'email_open',
            'click': 'email_click',
            'bounce': 'email_bounce',
            'dropped': 'email_dropped',
            'spamreport': 'email_spam',
            'unsubscribe': 'email_unsubscribe',
            'delivered': 'email_delivered'
        }
        
        engagement_type = event_mapping.get(event_type, f'email_{event_type}')
        
        # Find user by email
        user_result = await db.execute(
            select(User).where(User.email == email)
        )
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise errors.not_found(f"User not found for email: {email}")
        
        # Extract content ID from URL or custom data
        content_id = None
        if engagement_type == 'email_click':
            url = engagement_data.get('url', '')
            content_id = self._extract_content_id_from_url(url)
        
        # If no content ID found, try to find from recent deliveries
        if not content_id:
            content_id = await self._find_content_from_delivery(
                db, user.id, sg_message_id, timestamp
            )
        
        # Calculate engagement value based on event type
        engagement_value = self.engagement_weights.get(engagement_type, 1.0)
        
        # Extract additional context
        user_agent = engagement_data.get('useragent', '')
        ip_address = engagement_data.get('ip', '')
        device_type = self._extract_device_type(user_agent)
        
        # Get current user context for behavior correlation
        user_context = await self.get_user_context(db, user.id)
        
        # Create engagement record
        engagement = ContentEngagement(
            user_id=user.id,
            content_id=content_id,
            engagement_type=engagement_type,
            engagement_value=Decimal(str(engagement_value)),
            engagement_context=json.dumps({
                'sg_event': event_type,
                'device_type': device_type,
                'ip_address': ip_address,
                'user_agent': user_agent
            }),
            sendgrid_event_id=sg_event_id,
            sendgrid_message_id=sg_message_id,
            email_subject=engagement_data.get('subject', ''),
            user_agent=user_agent,
            ip_address=ip_address,
            device_type=device_type,
            user_strategic_profile_snapshot=json.dumps(user_context.strategic_profile),
            focus_areas_matched=json.dumps([fa['name'] for fa in user_context.focus_areas]),
            entities_matched=json.dumps([e['entity_name'] for e in user_context.tracked_entities]),
            engagement_timestamp=datetime.fromtimestamp(timestamp) if timestamp else datetime.utcnow()
        )
        
        # Save engagement
        db.add(engagement)
        await db.commit()
        await db.refresh(engagement)
        
        # Trigger ML learning update (async)
        asyncio.create_task(self._update_ml_models_with_engagement(db, engagement))
        
        return engagement
    
    async def _update_ml_models_with_engagement(
        self,
        db: AsyncSession,
        engagement: ContentEngagement
    ) -> None:
        """Update ML models with new engagement data for continuous learning."""
        # This would typically be a separate background task
        # For now, we'll update simple statistics
        
        if engagement.content_id:
            # Update content's actual engagement score
            content_result = await db.execute(
                select(DiscoveredContent).where(DiscoveredContent.id == engagement.content_id)
            )
            content = content_result.scalar_one_or_none()
            
            if content:
                # Calculate new actual engagement score
                engagement_result = await db.execute(
                    select(func.avg(ContentEngagement.engagement_value))
                    .where(ContentEngagement.content_id == engagement.content_id)
                )
                avg_engagement = engagement_result.scalar() or 0.0
                
                # Update content with actual engagement
                await db.execute(
                    update(DiscoveredContent)
                    .where(DiscoveredContent.id == engagement.content_id)
                    .values(actual_engagement_score=float(avg_engagement))
                )
                
                # Update source engagement score
                source_result = await db.execute(
                    select(func.avg(DiscoveredContent.actual_engagement_score))
                    .where(DiscoveredContent.source_id == content.source_id)
                )
                avg_source_engagement = source_result.scalar() or 0.0
                
                await db.execute(
                    update(DiscoveredSource)
                    .where(DiscoveredSource.id == content.source_id)
                    .values(user_engagement_score=float(avg_source_engagement))
                )
                
                await db.commit()
    
    # Utility methods
    
    def _get_industry_keywords(self, industry_type: str) -> List[str]:
        """Get relevant keywords for industry type."""
        industry_keywords = {
            'technology': ['software', 'tech', 'digital', 'innovation', 'startup', 'ai', 'machine learning'],
            'finance': ['financial', 'banking', 'investment', 'fintech', 'trading', 'cryptocurrency'],
            'healthcare': ['medical', 'healthcare', 'pharmaceutical', 'biotech', 'clinical', 'patient'],
            'retail': ['retail', 'ecommerce', 'consumer', 'shopping', 'brand', 'merchandise'],
            'manufacturing': ['manufacturing', 'industrial', 'production', 'supply chain', 'automation'],
        }
        return industry_keywords.get(industry_type.lower(), [])
    
    def _calculate_org_type_relevance(self, content: DiscoveredContent, org_type: str) -> float:
        """Calculate relevance based on organization type."""
        org_keywords = {
            'startup': ['startup', 'entrepreneur', 'venture', 'seed', 'funding', 'disruption'],
            'enterprise': ['enterprise', 'corporation', 'fortune', 'global', 'multinational'],
            'consulting': ['consulting', 'advisory', 'strategy', 'implementation', 'best practices'],
            'government': ['government', 'public sector', 'regulation', 'policy', 'compliance']
        }
        
        keywords = org_keywords.get(org_type.lower(), [])
        if not keywords or not content.content_text:
            return 0.5
        
        text_lower = content.content_text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return min(matches / len(keywords), 1.0)
    
    def _calculate_goals_relevance(self, content: DiscoveredContent, strategic_goals: str) -> float:
        """Calculate relevance based on strategic goals."""
        goals_keywords = {
            'market_expansion': ['market', 'expansion', 'growth', 'new markets', 'geographic'],
            'product_innovation': ['innovation', 'product', 'development', 'research', 'technology'],
            'cost_optimization': ['cost', 'efficiency', 'optimization', 'savings', 'lean'],
            'competitive_advantage': ['competitive', 'advantage', 'differentiation', 'unique', 'leadership']
        }
        
        keywords = goals_keywords.get(strategic_goals.lower(), [])
        if not keywords or not content.content_text:
            return 0.5
        
        text_lower = content.content_text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return min(matches / len(keywords), 1.0)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using ContentUtils."""
        return ContentUtils.calculate_text_similarity(text1, text2)
    
    def _assess_author_credibility(self, author: str) -> float:
        """Simple author credibility assessment."""
        # Simple heuristics for author credibility
        credibility = 0.5
        
        # Authors with institutional affiliations
        if any(keyword in author.lower() for keyword in ['dr.', 'prof.', 'phd', 'university', 'institute']):
            credibility += 0.3
        
        # Authors with professional titles
        if any(keyword in author.lower() for keyword in ['ceo', 'cto', 'director', 'manager', 'analyst']):
            credibility += 0.2
        
        return min(credibility, 1.0)
    
    def _assess_content_quality(self, content_text: str) -> float:
        """Assess content quality using ContentUtils."""
        return ContentUtils.assess_content_quality(
            content_text or "",
            "",  # No title available in this context
            self.config.content.min_content_length,
            self.config.content.max_content_length
        )
    
    def _assess_url_credibility(self, url: str) -> float:
        """Assess URL credibility based on domain and structure."""
        if not url:
            return 0.5
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            credibility = 0.5
            
            # Known credible domains
            credible_domains = [
                'reuters.com', 'bbc.com', 'wsj.com', 'ft.com', 'bloomberg.com',
                'techcrunch.com', 'wired.com', 'harvard.edu', 'mit.edu'
            ]
            
            if any(credible in domain for credible in credible_domains):
                credibility += 0.3
            
            # TLD credibility
            if domain.endswith(('.edu', '.gov', '.org')):
                credibility += 0.2
            elif domain.endswith('.com'):
                credibility += 0.1
            
            # HTTPS bonus
            if parsed.scheme == 'https':
                credibility += 0.1
            
            return min(credibility, 1.0)
        except Exception:
            return 0.5
    
    def _calculate_url_similarity(self, url1: str, url2: str) -> float:
        """Calculate URL similarity using ContentUtils normalization."""
        if not url1 or not url2:
            return 0.0
        
        try:
            # Normalize URLs for comparison
            normalized1 = ContentUtils.normalize_url(url1)
            normalized2 = ContentUtils.normalize_url(url2)
            
            # Exact match after normalization
            if normalized1 == normalized2:
                return 1.0
            
            # Domain and path similarity
            parsed1 = urlparse(normalized1)
            parsed2 = urlparse(normalized2)
            
            domain_similarity = 1.0 if parsed1.netloc == parsed2.netloc else 0.0
            path_similarity = ContentUtils.calculate_text_similarity(parsed1.path, parsed2.path)
            
            return domain_similarity * 0.6 + path_similarity * 0.4
            
        except Exception:
            return 0.0
    
    def _extract_content_id_from_url(self, url: str) -> Optional[int]:
        """Extract content ID from click tracking URL."""
        # This would extract content ID from your email tracking URLs
        # Implementation depends on your URL structure
        try:
            if 'content_id=' in url:
                content_id_match = re.search(r'content_id=(\d+)', url)
                if content_id_match:
                    return int(content_id_match.group(1))
        except (ValueError, AttributeError):
            pass
        return None
    
    async def _find_content_from_delivery(
        self,
        db: AsyncSession,
        user_id: int,
        sg_message_id: str,
        timestamp: int
    ) -> Optional[int]:
        """Find content ID from recent deliveries."""
        if not sg_message_id:
            return None
        
        # Look for recently delivered content
        delivery_window = datetime.fromtimestamp(timestamp) - timedelta(hours=24)
        
        content_result = await db.execute(
            select(DiscoveredContent.id)
            .where(
                and_(
                    DiscoveredContent.user_id == user_id,
                    DiscoveredContent.is_delivered == True,
                    DiscoveredContent.delivered_at >= delivery_window
                )
            )
            .order_by(DiscoveredContent.delivered_at.desc())
            .limit(1)
        )
        
        result = content_result.scalar_one_or_none()
        return result
    
    def _extract_device_type(self, user_agent: str) -> str:
        """Extract device type from user agent string."""
        if not user_agent:
            return 'unknown'
        
        user_agent_lower = user_agent.lower()
        
        if any(mobile in user_agent_lower for mobile in ['mobile', 'android', 'iphone', 'ipod']):
            return 'mobile'
        elif any(tablet in user_agent_lower for tablet in ['tablet', 'ipad']):
            return 'tablet'
        else:
            return 'desktop'
    
    async def generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        if not content:
            return ''
        
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    async def generate_similarity_hash(self, content: str) -> str:
        """Generate similarity hash using ContentUtils."""
        return ContentUtils.generate_content_hash(content)
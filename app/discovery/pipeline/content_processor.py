"""
Content Processor - ML scoring, deduplication, and quality assessment for discovered content.

Processes discovered content with advanced ML scoring algorithms, intelligent deduplication
using similarity detection, and comprehensive quality assessment for competitive intelligence.
"""

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import json
from difflib import SequenceMatcher

from sqlalchemy.ext.asyncio import AsyncSession  
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import selectinload

from app.database import get_db_session
from app.models.discovery import DiscoveredContent, ContentEngagement, MLModelMetrics
from app.models.user import User
from app.models.strategic_profile import UserStrategicProfile
from app.models.tracking import TrackingEntity

from ..utils import (
    ContentUtils,
    get_ml_scoring_cache,
    get_content_processing_cache,
    UnifiedErrorHandler,
    batch_processor
)
from ..engines.base_engine import DiscoveredItem


@dataclass
class ProcessingResult:
    """Result of content processing operations."""
    original_item: DiscoveredItem
    processed_content: Dict[str, Any]
    ml_scores: Dict[str, float]
    quality_metrics: Dict[str, Any]
    deduplication_info: Dict[str, Any]
    processing_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class MLScoringContext:
    """Context for ML scoring operations."""
    user_id: int
    strategic_profile: Optional[Dict[str, Any]] = None
    focus_areas: List[str] = field(default_factory=list)
    tracked_entities: List[str] = field(default_factory=list)
    historical_engagement: Dict[str, float] = field(default_factory=dict)
    model_version: str = "1.0"


@dataclass
class QualityMetrics:
    """Content quality assessment metrics."""
    readability_score: float = 0.0
    content_depth_score: float = 0.0
    source_credibility_score: float = 0.0
    freshness_score: float = 0.0
    relevance_score: float = 0.0
    overall_quality_score: float = 0.0
    confidence_score: float = 0.5


class ContentProcessor:
    """
    Advanced content processor for competitive intelligence pipeline.
    
    Provides ML-enabled content scoring, intelligent deduplication,
    and comprehensive quality assessment for discovered content.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.content_processor")
        self.error_handler = UnifiedErrorHandler()
        self.content_utils = ContentUtils()
        
        # Caching
        self.ml_cache = get_ml_scoring_cache()
        self.processing_cache = get_content_processing_cache()
        
        # Processing configuration
        self.similarity_threshold = 0.85  # For duplicate detection
        self.min_content_length = 100    # Minimum content length
        self.max_content_length = 50000  # Maximum content length
        
        # ML model settings
        self.current_model_version = "1.0"
        self.ml_confidence_threshold = 0.6
        
        # Performance tracking
        self.processing_stats = {
            'items_processed': 0,
            'duplicates_detected': 0,
            'ml_scores_computed': 0,
            'cache_hits': 0,
            'processing_errors': 0
        }
        
        self.logger.info("Content processor initialized")
    
    async def process_user_content(self, user_context: Dict[str, Any], 
                                 discovered_items: List[DiscoveredItem]) -> List[ProcessingResult]:
        """
        Process discovered content for a specific user.
        
        Args:
            user_context: User context including profile and preferences
            discovered_items: List of discovered content items
            
        Returns:
            List[ProcessingResult]: Processed content with scores and metrics
        """
        if not discovered_items:
            return []
        
        start_time = datetime.now()
        results = []
        
        try:
            self.logger.debug(f"Processing {len(discovered_items)} items for user {user_context.get('user_id')}")
            
            # Build ML scoring context
            ml_context = await self._build_ml_context(user_context)
            
            # Process items in batches for efficiency
            batch_size = 10
            for i in range(0, len(discovered_items), batch_size):
                batch = discovered_items[i:i + batch_size]
                batch_results = await self._process_content_batch(batch, ml_context)
                results.extend(batch_results)
            
            # Post-process for deduplication and final scoring
            results = await self._post_process_results(results, ml_context)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"Processed {len(results)} items in {processing_time:.2f}s")
            
            # Update stats
            self.processing_stats['items_processed'] += len(discovered_items)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Content processing failed: {e}")
            self.processing_stats['processing_errors'] += 1
            raise
    
    async def _build_ml_context(self, user_context: Dict[str, Any]) -> MLScoringContext:
        """Build ML scoring context from user data."""
        user_id = user_context.get('user_id')
        
        try:
            # Get user strategic profile and engagement history
            async with get_db_session() as session:
                # Load strategic profile
                strategic_result = await session.execute(
                    select(UserStrategicProfile).where(UserStrategicProfile.user_id == user_id)
                )
                strategic_profile = strategic_result.scalar_one_or_none()
                
                # Load engagement history for ML training
                engagement_result = await session.execute(
                    select(ContentEngagement)
                    .where(
                        and_(
                            ContentEngagement.user_id == user_id,
                            ContentEngagement.created_at >= func.now() - text("INTERVAL '30 days'")
                        )
                    )
                    .limit(100)
                )
                
                engagements = engagement_result.scalars().all()
                
                # Build engagement patterns
                engagement_patterns = {}
                for engagement in engagements:
                    key = f"{engagement.engagement_type}"
                    if key not in engagement_patterns:
                        engagement_patterns[key] = []
                    engagement_patterns[key].append(float(engagement.engagement_value))
                
                # Average engagement values
                historical_engagement = {}
                for key, values in engagement_patterns.items():
                    historical_engagement[key] = sum(values) / len(values) if values else 0.0
                
                # Build strategic context
                strategic_context = None
                if strategic_profile:
                    strategic_context = {
                        'industry': strategic_profile.industry,
                        'organization_type': strategic_profile.organization_type,
                        'role': strategic_profile.role,
                        'strategic_goals': strategic_profile.strategic_goals or []
                    }
                
                return MLScoringContext(
                    user_id=user_id,
                    strategic_profile=strategic_context,
                    focus_areas=user_context.get('focus_areas', []),
                    tracked_entities=user_context.get('tracked_entities', []),
                    historical_engagement=historical_engagement,
                    model_version=self.current_model_version
                )
                
        except Exception as e:
            self.logger.error(f"Failed to build ML context for user {user_id}: {e}")
            # Return minimal context
            return MLScoringContext(
                user_id=user_id,
                focus_areas=user_context.get('focus_areas', []),
                tracked_entities=user_context.get('tracked_entities', [])
            )
    
    async def _process_content_batch(self, items: List[DiscoveredItem], 
                                   ml_context: MLScoringContext) -> List[ProcessingResult]:
        """Process a batch of content items."""
        batch_results = []
        
        # Create processing tasks
        tasks = []
        for item in items:
            task = self._process_single_item(item, ml_context)
            tasks.append(task)
        
        # Execute batch processing
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process item {i}: {result}")
                # Create error result
                error_result = ProcessingResult(
                    original_item=items[i],
                    processed_content={},
                    ml_scores={},
                    quality_metrics={},
                    deduplication_info={},
                    processing_time=0.0,
                    errors=[str(result)]
                )
                batch_results.append(error_result)
            else:
                batch_results.append(result)
        
        return batch_results
    
    async def _process_single_item(self, item: DiscoveredItem, 
                                 ml_context: MLScoringContext) -> ProcessingResult:
        """Process a single content item."""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"content_process:{item.content_hash}:{ml_context.model_version}"
            cached_result = await self.processing_cache.get(cache_key)
            
            if cached_result:
                self.processing_stats['cache_hits'] += 1
                return ProcessingResult(**cached_result)
            
            # Extract and enhance content
            processed_content = await self._extract_content_features(item)
            
            # Generate ML scores
            ml_scores = await self._compute_ml_scores(item, processed_content, ml_context)
            
            # Assess content quality
            quality_metrics = await self._assess_content_quality(item, processed_content)
            
            # Detect duplicates and similarity
            dedup_info = await self._detect_duplicates(item, processed_content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ProcessingResult(
                original_item=item,
                processed_content=processed_content,
                ml_scores=ml_scores,
                quality_metrics=quality_metrics,
                deduplication_info=dedup_info,
                processing_time=processing_time
            )
            
            # Cache result
            result_dict = {
                'original_item': item,
                'processed_content': processed_content,
                'ml_scores': ml_scores,
                'quality_metrics': quality_metrics,
                'deduplication_info': dedup_info,
                'processing_time': processing_time,
                'errors': [],
                'warnings': []
            }
            await self.processing_cache.set(cache_key, result_dict, ttl=3600)  # 1 hour
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process content item: {e}")
            return ProcessingResult(
                original_item=item,
                processed_content={},
                ml_scores={},
                quality_metrics={},
                deduplication_info={},
                processing_time=(datetime.now() - start_time).total_seconds(),
                errors=[str(e)]
            )
    
    async def _extract_content_features(self, item: DiscoveredItem) -> Dict[str, Any]:
        """Extract enhanced features from content."""
        content = item.content or ""
        title = item.title or ""
        
        # Basic content analysis
        word_count = len(content.split())
        char_count = len(content)
        sentence_count = len(re.findall(r'[.!?]+', content))
        
        # Extract entities and keywords
        extracted_entities = self._extract_entities(content + " " + title)
        keywords = self._extract_keywords(content + " " + title)
        
        # Content categorization
        categories = self._categorize_content(content, title)
        
        # Sentiment analysis (basic implementation)
        sentiment = self._analyze_sentiment(content)
        
        # Generate content hash for deduplication
        content_hash = self._generate_content_hash(content)
        similarity_hash = self._generate_similarity_hash(content)
        
        return {
            'content_hash': content_hash,
            'similarity_hash': similarity_hash,
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': word_count / max(sentence_count, 1),
            'extracted_entities': extracted_entities,
            'keywords': keywords,
            'categories': categories,
            'sentiment': sentiment,
            'language': self._detect_language(content),
            'reading_time_minutes': max(1, word_count // 200),  # Assume 200 WPM
            'content_structure': self._analyze_content_structure(content)
        }
    
    async def _compute_ml_scores(self, item: DiscoveredItem, processed_content: Dict[str, Any],
                               ml_context: MLScoringContext) -> Dict[str, float]:
        """Compute ML-based relevance and engagement scores."""
        
        # Cache key for ML scores
        cache_key = f"ml_scores:{processed_content['content_hash']}:{ml_context.user_id}:{ml_context.model_version}"
        cached_scores = await self.ml_cache.get(cache_key)
        
        if cached_scores:
            self.processing_stats['cache_hits'] += 1
            return cached_scores
        
        try:
            # Relevance scoring based on user profile
            relevance_score = self._compute_relevance_score(item, processed_content, ml_context)
            
            # Engagement prediction based on historical patterns
            engagement_score = self._predict_engagement(item, processed_content, ml_context)
            
            # Source credibility scoring
            credibility_score = self._compute_credibility_score(item, processed_content)
            
            # Freshness scoring
            freshness_score = self._compute_freshness_score(item)
            
            # Competitive relevance scoring
            competitive_score = self._compute_competitive_relevance(item, processed_content, ml_context)
            
            # Overall composite score
            overall_score = self._compute_overall_score({
                'relevance': relevance_score,
                'engagement': engagement_score,
                'credibility': credibility_score,
                'freshness': freshness_score,
                'competitive': competitive_score
            })
            
            scores = {
                'relevance_score': relevance_score,
                'engagement_prediction_score': engagement_score,
                'credibility_score': credibility_score,
                'freshness_score': freshness_score,
                'competitive_relevance_score': competitive_score,
                'overall_score': overall_score,
                'ml_confidence_level': self._compute_confidence_level(processed_content)
            }
            
            # Cache scores
            await self.ml_cache.set(cache_key, scores, ttl=1800)  # 30 minutes
            self.processing_stats['ml_scores_computed'] += 1
            
            return scores
            
        except Exception as e:
            self.logger.error(f"ML scoring failed: {e}")
            # Return default scores
            return {
                'relevance_score': 0.5,
                'engagement_prediction_score': 0.5,
                'credibility_score': 0.5,
                'freshness_score': 0.5,
                'competitive_relevance_score': 0.5,
                'overall_score': 0.5,
                'ml_confidence_level': 0.3
            }
    
    def _compute_relevance_score(self, item: DiscoveredItem, processed_content: Dict[str, Any],
                               ml_context: MLScoringContext) -> float:
        """Compute content relevance score based on user profile."""
        score = 0.0
        weight_sum = 0.0
        
        content_text = (item.content or "") + " " + (item.title or "")
        content_lower = content_text.lower()
        
        # Focus area matching
        if ml_context.focus_areas:
            focus_matches = 0
            for focus_area in ml_context.focus_areas:
                if focus_area.lower() in content_lower:
                    focus_matches += 1
            
            focus_score = min(focus_matches / len(ml_context.focus_areas), 1.0)
            score += focus_score * 0.4  # 40% weight for focus areas
            weight_sum += 0.4
        
        # Entity matching
        if ml_context.tracked_entities:
            entity_matches = 0
            for entity in ml_context.tracked_entities:
                if entity.lower() in content_lower:
                    entity_matches += 1
            
            entity_score = min(entity_matches / len(ml_context.tracked_entities), 1.0)
            score += entity_score * 0.3  # 30% weight for entities
            weight_sum += 0.3
        
        # Strategic profile alignment
        if ml_context.strategic_profile:
            strategic_score = self._compute_strategic_alignment(content_text, ml_context.strategic_profile)
            score += strategic_score * 0.2  # 20% weight for strategic alignment
            weight_sum += 0.2
        
        # Content quality indicators
        quality_score = min(processed_content.get('word_count', 0) / 500, 1.0)  # Prefer longer content
        score += quality_score * 0.1  # 10% weight for content quality
        weight_sum += 0.1
        
        return score / weight_sum if weight_sum > 0 else 0.5
    
    def _predict_engagement(self, item: DiscoveredItem, processed_content: Dict[str, Any],
                          ml_context: MLScoringContext) -> float:
        """Predict user engagement based on historical patterns."""
        if not ml_context.historical_engagement:
            return 0.5  # Neutral prediction without historical data
        
        # Analyze content features that correlate with past engagement
        engagement_factors = []
        
        # Content length preference
        word_count = processed_content.get('word_count', 0)
        avg_engagement_by_length = ml_context.historical_engagement.get('time_spent', 0)
        
        # Assume preference correlation with historical time spent
        if avg_engagement_by_length > 0:
            length_factor = min(word_count / (avg_engagement_by_length * 20), 1.0)  # Rough correlation
            engagement_factors.append(length_factor)
        
        # Category preference
        categories = processed_content.get('categories', [])
        if categories:
            category_factor = 0.6  # Default if categories match historical patterns
            engagement_factors.append(category_factor)
        
        # Sentiment preference
        sentiment = processed_content.get('sentiment', {})
        if sentiment.get('polarity', 0) > 0.1:  # Positive content
            sentiment_factor = 0.7
        elif sentiment.get('polarity', 0) < -0.1:  # Negative content  
            sentiment_factor = 0.4
        else:
            sentiment_factor = 0.5  # Neutral
        engagement_factors.append(sentiment_factor)
        
        # Source type preference
        source_factor = 0.6  # Default source preference
        engagement_factors.append(source_factor)
        
        # Compute weighted average
        return sum(engagement_factors) / len(engagement_factors) if engagement_factors else 0.5
    
    def _compute_credibility_score(self, item: DiscoveredItem, processed_content: Dict[str, Any]) -> float:
        """Compute source and content credibility score."""
        score = 0.5  # Base score
        
        # Source domain analysis
        if item.url:
            domain = item.url.split('//')[1].split('/')[0] if '//' in item.url else item.url
            
            # Known credible domains (simplified)
            credible_domains = {
                'reuters.com': 0.9,
                'bloomberg.com': 0.9,
                'wsj.com': 0.9,
                'ft.com': 0.9,
                'techcrunch.com': 0.8,
                'venturebeat.com': 0.8,
                'harvard.edu': 0.95,
                'mit.edu': 0.95
            }
            
            for credible_domain, domain_score in credible_domains.items():
                if credible_domain in domain:
                    score = max(score, domain_score)
                    break
        
        # Content quality indicators
        word_count = processed_content.get('word_count', 0)
        if word_count > 300:  # Substantial content
            score += 0.1
        
        if processed_content.get('sentence_count', 0) > 5:  # Well-structured
            score += 0.05
        
        # Author presence
        if item.metadata.get('author'):
            score += 0.1
        
        # Publication date presence
        if item.metadata.get('published_date'):
            score += 0.05
        
        return min(score, 1.0)
    
    def _compute_freshness_score(self, item: DiscoveredItem) -> float:
        """Compute content freshness score based on publication date."""
        published_date = item.metadata.get('published_date')
        if not published_date:
            return 0.5  # Unknown freshness
        
        try:
            if isinstance(published_date, str):
                from dateutil import parser
                published_date = parser.parse(published_date)
            
            now = datetime.now(timezone.utc)
            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=timezone.utc)
            
            age_days = (now - published_date).days
            
            # Freshness decay function
            if age_days <= 1:
                return 1.0
            elif age_days <= 7:
                return 0.9
            elif age_days <= 30:
                return 0.7
            elif age_days <= 90:
                return 0.5
            else:
                return 0.3
                
        except Exception as e:
            self.logger.error(f"Failed to compute freshness score: {e}")
            return 0.5
    
    def _compute_competitive_relevance(self, item: DiscoveredItem, processed_content: Dict[str, Any],
                                     ml_context: MLScoringContext) -> float:
        """Compute competitive intelligence relevance score."""
        content_text = (item.content or "") + " " + (item.title or "")
        content_lower = content_text.lower()
        
        # Competitive keywords
        competitive_terms = [
            'competitor', 'competition', 'market share', 'strategy', 'acquisition',
            'merger', 'funding', 'investment', 'partnership', 'product launch',
            'innovation', 'disruption', 'trend', 'analysis', 'forecast'
        ]
        
        # Industry-specific competitive terms
        if ml_context.strategic_profile:
            industry = ml_context.strategic_profile.get('industry', '').lower()
            if 'tech' in industry:
                competitive_terms.extend(['startup', 'ai', 'software', 'platform', 'saas'])
            elif 'finance' in industry:
                competitive_terms.extend(['fintech', 'banking', 'payments', 'crypto', 'investment'])
        
        # Count competitive term matches
        matches = 0
        for term in competitive_terms:
            if term in content_lower:
                matches += 1
        
        # Calculate score
        base_score = min(matches / len(competitive_terms) * 2, 1.0)
        
        # Boost for entities mentioned
        entity_boost = 0.0
        for entity in ml_context.tracked_entities:
            if entity.lower() in content_lower:
                entity_boost += 0.2
        
        return min(base_score + entity_boost, 1.0)
    
    def _compute_strategic_alignment(self, content: str, strategic_profile: Dict[str, Any]) -> float:
        """Compute alignment with user's strategic profile."""
        content_lower = content.lower()
        alignment_score = 0.0
        
        # Industry alignment
        industry = strategic_profile.get('industry', '').lower()
        if industry and industry in content_lower:
            alignment_score += 0.3
        
        # Role alignment
        role = strategic_profile.get('role', '').lower()
        role_terms = {
            'ceo': ['leadership', 'strategy', 'vision', 'growth'],
            'cto': ['technology', 'innovation', 'development', 'engineering'],
            'cmo': ['marketing', 'brand', 'customer', 'campaign'],
            'cfo': ['financial', 'revenue', 'investment', 'budget']
        }
        
        if role in role_terms:
            for term in role_terms[role]:
                if term in content_lower:
                    alignment_score += 0.1
        
        # Strategic goals alignment
        goals = strategic_profile.get('strategic_goals', [])
        for goal in goals:
            if goal.lower() in content_lower:
                alignment_score += 0.15
        
        return min(alignment_score, 1.0)
    
    def _compute_overall_score(self, component_scores: Dict[str, float]) -> float:
        """Compute weighted overall score from component scores."""
        weights = {
            'relevance': 0.35,
            'engagement': 0.25,
            'credibility': 0.20,
            'freshness': 0.10,
            'competitive': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for component, score in component_scores.items():
            if component in weights:
                weight = weights[component]
                weighted_score += score * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _compute_confidence_level(self, processed_content: Dict[str, Any]) -> float:
        """Compute ML model confidence level."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for longer content
        word_count = processed_content.get('word_count', 0)
        if word_count > 500:
            confidence += 0.2
        elif word_count > 200:
            confidence += 0.1
        
        # Higher confidence for structured content
        if processed_content.get('sentence_count', 0) > 10:
            confidence += 0.1
        
        # Higher confidence for content with entities
        if len(processed_content.get('extracted_entities', [])) > 3:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    async def _assess_content_quality(self, item: DiscoveredItem, 
                                    processed_content: Dict[str, Any]) -> QualityMetrics:
        """Assess comprehensive content quality."""
        
        # Readability assessment
        readability = self._assess_readability(processed_content)
        
        # Content depth assessment
        depth = self._assess_content_depth(processed_content)
        
        # Source credibility (computed earlier, reuse)
        credibility = processed_content.get('credibility_score', 0.5)
        
        # Freshness (computed earlier, reuse) 
        freshness = processed_content.get('freshness_score', 0.5)
        
        # Overall relevance (computed earlier, reuse)
        relevance = processed_content.get('relevance_score', 0.5)
        
        # Compute overall quality
        overall_quality = (readability + depth + credibility + freshness + relevance) / 5
        
        return QualityMetrics(
            readability_score=readability,
            content_depth_score=depth,
            source_credibility_score=credibility,
            freshness_score=freshness,
            relevance_score=relevance,
            overall_quality_score=overall_quality,
            confidence_score=self._compute_confidence_level(processed_content)
        )
    
    def _assess_readability(self, processed_content: Dict[str, Any]) -> float:
        """Assess content readability."""
        word_count = processed_content.get('word_count', 0)
        sentence_count = processed_content.get('sentence_count', 1)
        avg_sentence_length = processed_content.get('avg_sentence_length', 0)
        
        # Simple readability score based on sentence length
        if avg_sentence_length < 15:
            return 0.9  # Easy to read
        elif avg_sentence_length < 25:
            return 0.7  # Moderate
        else:
            return 0.5  # More complex
    
    def _assess_content_depth(self, processed_content: Dict[str, Any]) -> float:
        """Assess content depth and comprehensiveness."""
        word_count = processed_content.get('word_count', 0)
        entity_count = len(processed_content.get('extracted_entities', []))
        keyword_count = len(processed_content.get('keywords', []))
        
        # Depth based on content length and richness
        depth_score = 0.0
        
        # Length contribution
        if word_count > 1000:
            depth_score += 0.4
        elif word_count > 500:
            depth_score += 0.3
        elif word_count > 200:
            depth_score += 0.2
        else:
            depth_score += 0.1
        
        # Entity richness
        if entity_count > 10:
            depth_score += 0.3
        elif entity_count > 5:
            depth_score += 0.2
        else:
            depth_score += 0.1
        
        # Keyword diversity
        if keyword_count > 15:
            depth_score += 0.3
        elif keyword_count > 8:
            depth_score += 0.2
        else:
            depth_score += 0.1
        
        return min(depth_score, 1.0)
    
    async def _detect_duplicates(self, item: DiscoveredItem, 
                               processed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Detect duplicate content and compute similarity scores."""
        content_hash = processed_content.get('content_hash')
        similarity_hash = processed_content.get('similarity_hash')
        
        if not content_hash or not similarity_hash:
            return {'is_duplicate': False, 'similarity_score': 0.0}
        
        try:
            # Check for exact duplicates
            async with get_db_session() as session:
                exact_duplicate = await session.execute(
                    select(DiscoveredContent)
                    .where(DiscoveredContent.content_hash == content_hash)
                    .limit(1)
                )
                
                if exact_duplicate.scalar_one_or_none():
                    self.processing_stats['duplicates_detected'] += 1
                    return {
                        'is_duplicate': True,
                        'duplicate_type': 'exact',
                        'similarity_score': 1.0,
                        'original_content_id': exact_duplicate.scalar_one().id
                    }
                
                # Check for similar content
                similar_content = await session.execute(
                    select(DiscoveredContent)
                    .where(DiscoveredContent.similarity_hash == similarity_hash)
                    .limit(5)
                )
                
                similar_items = similar_content.scalars().all()
                
                if similar_items:
                    # Compute detailed similarity
                    max_similarity = 0.0
                    most_similar_id = None
                    
                    for similar_item in similar_items:
                        similarity = self._compute_content_similarity(
                            item.content or "", 
                            similar_item.content_text or ""
                        )
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_id = similar_item.id
                    
                    if max_similarity > self.similarity_threshold:
                        self.processing_stats['duplicates_detected'] += 1
                        return {
                            'is_duplicate': True,
                            'duplicate_type': 'similar',
                            'similarity_score': max_similarity,
                            'original_content_id': most_similar_id
                        }
                
                return {
                    'is_duplicate': False,
                    'similarity_score': max_similarity if 'max_similarity' in locals() else 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Duplicate detection failed: {e}")
            return {'is_duplicate': False, 'similarity_score': 0.0}
    
    def _compute_content_similarity(self, content1: str, content2: str) -> float:
        """Compute similarity between two content strings."""
        if not content1 or not content2:
            return 0.0
        
        # Use sequence matcher for similarity
        matcher = SequenceMatcher(None, content1.lower(), content2.lower())
        return matcher.ratio()
    
    async def _post_process_results(self, results: List[ProcessingResult], 
                                  ml_context: MLScoringContext) -> List[ProcessingResult]:
        """Post-process results for final optimization."""
        if not results:
            return results
        
        # Filter out duplicates
        unique_results = []
        seen_hashes = set()
        
        for result in results:
            content_hash = result.processed_content.get('content_hash')
            if content_hash and content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append(result)
            elif not result.deduplication_info.get('is_duplicate', False):
                unique_results.append(result)
        
        # Sort by overall score
        unique_results.sort(key=lambda x: x.ml_scores.get('overall_score', 0.0), reverse=True)
        
        return unique_results
    
    # Utility methods for content analysis
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text (simplified implementation)."""
        # Simple entity extraction - in production, use NLP libraries like spaCy
        entities = []
        
        # Company patterns
        company_patterns = [
            r'\b[A-Z][a-zA-Z]*\s+Inc\.?\b',
            r'\b[A-Z][a-zA-Z]*\s+Corp\.?\b',
            r'\b[A-Z][a-zA-Z]*\s+LLC\b',
            r'\b[A-Z][a-zA-Z]*\s+Ltd\.?\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # Person patterns (simplified)
        person_patterns = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        persons = re.findall(person_patterns, text)
        entities.extend(persons[:5])  # Limit to prevent noise
        
        return list(set(entities))[:20]  # Deduplicate and limit
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction - in production, use TF-IDF or similar
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter common words
        stop_words = {'that', 'with', 'have', 'this', 'will', 'been', 'from', 'they', 'were', 'their'}
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:20]]
    
    def _categorize_content(self, content: str, title: str) -> List[str]:
        """Categorize content based on keywords and patterns."""
        text = (content + " " + title).lower()
        
        categories = []
        
        category_keywords = {
            'technology': ['software', 'ai', 'machine learning', 'cloud', 'digital', 'tech'],
            'business': ['strategy', 'market', 'revenue', 'growth', 'investment', 'finance'],
            'news': ['announced', 'reports', 'breaking', 'update', 'latest'],
            'research': ['study', 'analysis', 'research', 'report', 'findings', 'data'],
            'competitive': ['competitor', 'competition', 'vs', 'versus', 'compared to']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['general']
    
    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze content sentiment (simplified implementation)."""
        # Simple sentiment analysis - in production, use libraries like TextBlob or VADER
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'growth', 'innovation']
        negative_words = ['bad', 'poor', 'negative', 'decline', 'loss', 'failure', 'problem']
        
        content_lower = content.lower()
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        total_words = len(content.split())
        
        if total_words == 0:
            return {'polarity': 0.0, 'subjectivity': 0.5}
        
        polarity = (positive_count - negative_count) / max(total_words / 100, 1)
        subjectivity = (positive_count + negative_count) / max(total_words / 50, 1)
        
        return {
            'polarity': max(-1.0, min(1.0, polarity)),
            'subjectivity': max(0.0, min(1.0, subjectivity))
        }
    
    def _detect_language(self, content: str) -> str:
        """Detect content language (simplified)."""
        # Simple language detection - in production, use libraries like langdetect
        if not content:
            return 'unknown'
        
        # Count English words
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on']
        english_count = sum(1 for word in english_words if word in content.lower())
        
        return 'en' if english_count > 3 else 'unknown'
    
    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure and formatting."""
        lines = content.split('\n')
        paragraphs = [line.strip() for line in lines if line.strip()]
        
        return {
            'paragraph_count': len(paragraphs),
            'avg_paragraph_length': sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1),
            'has_headings': any(line.isupper() or line.startswith('#') for line in lines),
            'has_lists': any(line.strip().startswith(('-', '*', '•')) for line in lines),
            'line_count': len(lines)
        }
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for exact duplicate detection."""
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.strip().lower())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _generate_similarity_hash(self, content: str) -> str:
        """Generate hash for similarity detection."""
        # Remove punctuation and normalize for similarity
        normalized = re.sub(r'[^\w\s]', '', content.lower())
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        # Create hash of first 1000 characters for similarity grouping
        similarity_content = normalized[:1000]
        return hashlib.md5(similarity_content.encode()).hexdigest()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get content processing statistics."""
        return {
            'items_processed': self.processing_stats['items_processed'],
            'duplicates_detected': self.processing_stats['duplicates_detected'],
            'ml_scores_computed': self.processing_stats['ml_scores_computed'],
            'cache_hits': self.processing_stats['cache_hits'],
            'processing_errors': self.processing_stats['processing_errors'],
            'cache_hit_rate': (
                self.processing_stats['cache_hits'] / 
                max(self.processing_stats['items_processed'], 1)
            ),
            'duplicate_rate': (
                self.processing_stats['duplicates_detected'] / 
                max(self.processing_stats['items_processed'], 1)
            )
        }
    
    async def clear_cache(self):
        """Clear processing caches."""
        await self.processing_cache.clear()
        await self.ml_cache.clear()
        self.logger.info("Content processor caches cleared")
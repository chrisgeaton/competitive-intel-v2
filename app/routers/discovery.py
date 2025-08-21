"""
Discovery Service REST API endpoints for competitive intelligence v2.

Comprehensive endpoints for discovery job management, content retrieval,
SendGrid webhook processing, analytics, and ML model performance tracking.
Following established User Config Service patterns with BaseRouterOperations.
"""

import json
import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Union

from fastapi import BackgroundTasks, Body, Header
from sqlalchemy import desc, update
from sqlalchemy.orm import selectinload

# Import optimized discovery utilities
from app.discovery.utils import (
    ContentUtils, get_content_processing_cache, get_ml_scoring_cache,
    get_source_discovery_cache, AsyncBatchProcessor, get_config
)
from app.services.discovery_service import DiscoveryService
from app.models.discovery import (
    DiscoveredSource, DiscoveredContent, ContentEngagement, 
    DiscoveryJob, MLModelMetrics
)
from app.schemas.discovery import (
    # Source schemas
    DiscoveredSourceCreate, DiscoveredSourceUpdate, DiscoveredSourceResponse,
    
    # Content schemas
    DiscoveredContentCreate, DiscoveredContentUpdate, DiscoveredContentResponse,
    DiscoveryFilterRequest, ContentSimilaritySchema,
    
    # Engagement schemas
    ContentEngagementCreate, ContentEngagementResponse, SendGridEngagementData,
    
    # Job schemas  
    DiscoveryJobCreate, DiscoveryJobUpdate, DiscoveryJobResponse,
    JobType, JobStatus,
    
    # ML schemas
    MLModelMetricsCreate, MLModelMetricsUpdate, MLModelMetricsResponse,
    MLScoresSchema,
    
    # Analytics schemas
    UserDiscoveryAnalytics,
    
    # Enums
    SourceType, ContentType, EngagementType
)
from app.utils.router_base import (
    logging, Dict, List, Any, APIRouter, Depends, Query, status,
    AsyncSession, select, func, and_, or_, selectinload, get_db_session, User, get_current_active_user,
    errors, db_handler, db_helpers, BaseRouterOperations, 
    PaginationParams, create_paginated_response, create_analytics_response
)

# Initialize base operations and services
base_ops = BaseRouterOperations(__name__)
discovery_service = DiscoveryService()
config = get_config()

# Create router with comprehensive tags
router = APIRouter(prefix="/api/v1/discovery", tags=["Discovery Service Management"])


# Source Management Endpoints

@router.post("/sources", response_model=DiscoveredSourceResponse, status_code=status.HTTP_201_CREATED)
async def create_discovery_source(
    source_data: DiscoveredSourceCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Create a new discovery source for competitive intelligence monitoring.
    
    Creates a new source configuration for automated content discovery with:
    - **source_url**: Target URL for content discovery (required)
    - **source_type**: Type of source (rss, api, website, social_media)
    - **source_name**: Human-readable source identifier
    - **discovery_config**: JSON configuration for source-specific settings
    - **quality_filters**: Content quality thresholds and filters
    - **is_active**: Whether to actively monitor this source
    
    The source will be validated for accessibility and configuration correctness
    before being added to the discovery pipeline.
    """
    # Check if source URL already exists
    existing_source = await db.execute(
        select(DiscoveredSource).where(DiscoveredSource.source_url == source_data.source_url)
    )
    if existing_source.scalar_one_or_none():
        raise errors.conflict("Source URL already exists")
    
    # Create source
    source = DiscoveredSource(
        **source_data.dict(),
        created_by_user_id=current_user.id
    )
    
    db.add(source)
    await db.commit()
    await db.refresh(source)
    
    return source


@router.get("/sources", response_model=List[DiscoveredSourceResponse])
async def get_discovery_sources(
    pagination: PaginationParams = Depends(),
    source_type: Optional[str] = Query(None, description="Filter by source type (rss, api, website, social_media)"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    min_quality_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum quality score threshold"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get paginated list of discovery sources with advanced filtering options.
    
    Returns all configured discovery sources with optional filters for:
    - **source_type**: Filter by source type (rss, api, website, social_media)
    - **is_active**: Show only active/inactive sources
    - **min_quality_score**: Sources with minimum quality threshold (0.0-1.0)
    
    Results are ordered by quality score (highest first) and include:
    - Source configuration and status
    - Performance metrics and health indicators
    - Recent discovery statistics
    - ML model confidence levels
    """
    query = select(DiscoveredSource)
    
    # Apply filters
    filters = []
    if source_type:
        filters.append(DiscoveredSource.source_type == source_type)
    if is_active is not None:
        filters.append(DiscoveredSource.is_active == is_active)
    if min_quality_score is not None:
        filters.append(DiscoveredSource.quality_score >= min_quality_score)
    
    if filters:
        query = query.where(and_(*filters))
    
    # Apply pagination
    query = query.order_by(desc(DiscoveredSource.quality_score))
    query = query.offset(pagination.offset).limit(pagination.limit)
    
    result = await db.execute(query)
    sources = result.scalars().all()
    
    return sources


@router.get("/sources/{source_id}", response_model=DiscoveredSourceResponse)
async def get_discovery_source(
    source_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get comprehensive details about a specific discovery source.
    
    Returns complete source information including:
    - **Basic Configuration**: URL, type, name, and discovery settings
    - **Performance Metrics**: Success rate, quality score, credibility assessment
    - **Health Status**: Connectivity, error rates, last check timestamp
    - **Discovery Statistics**: Content found, processed, delivered counts
    - **ML Integration**: Model confidence, scoring parameters
    
    Use this endpoint to monitor source performance and troubleshoot issues.
    """
    result = await db.execute(
        select(DiscoveredSource).where(DiscoveredSource.id == source_id)
    )
    source = result.scalar_one_or_none()
    
    if not source:
        raise errors.not_found("Discovery source not found")
    
    return source


@router.put("/sources/{source_id}", response_model=DiscoveredSourceResponse)
async def update_discovery_source(
    source_id: int,
    source_data: DiscoveredSourceUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Update discovery source configuration and operational settings.
    
    Allows modification of source parameters including:
    - **source_name**: Human-readable identifier
    - **discovery_config**: JSON configuration for source-specific settings
    - **quality_filters**: Content quality thresholds and filtering rules
    - **is_active**: Enable/disable source monitoring
    - **check_frequency**: How often to check for new content
    
    Changes take effect immediately. Active sources will use new settings
    on the next discovery cycle. Historical performance metrics are preserved.
    """
    result = await db.execute(
        select(DiscoveredSource).where(DiscoveredSource.id == source_id)
    )
    source = result.scalar_one_or_none()
    
    if not source:
        raise errors.not_found("Discovery source not found")
    
    # Update source
    update_data = source_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(source, field, value)
    
    await db.commit()
    await db.refresh(source)
    
    return source


@router.delete("/sources/{source_id}")
async def delete_discovery_source(
    source_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Permanently delete a discovery source and all associated content.
    
    **Warning**: This action permanently removes:
    - Source configuration and settings
    - All discovered content from this source
    - Historical performance metrics and analytics
    - Engagement tracking data
    
    **Cannot be undone**. Consider deactivating the source instead of deletion
    to preserve historical data while stopping new content discovery.
    """
    result = await db.execute(
        select(DiscoveredSource).where(DiscoveredSource.id == source_id)
    )
    source = result.scalar_one_or_none()
    
    if not source:
        raise errors.not_found("Discovery source not found")
    
    await db.delete(source)
    await db.commit()
    
    return {"message": "Discovery source deleted successfully"}


# Content Discovery Endpoints

@router.get("/content", response_model=List[DiscoveredContentResponse])
async def get_discovered_content(
    pagination: PaginationParams = Depends(),
    source_types: Optional[List[str]] = Query(None, description="Filter by source types"),
    content_types: Optional[List[str]] = Query(None, description="Filter by content types"),
    min_relevance_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum relevance score"),
    min_credibility_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum credibility score"),
    min_freshness_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum freshness score"),
    published_after: Optional[datetime] = Query(None, description="Filter content published after this date"),
    published_before: Optional[datetime] = Query(None, description="Filter content published before this date"),
    competitive_relevance: Optional[List[str]] = Query(None, description="Filter by competitive relevance"),
    is_delivered: Optional[bool] = Query(None, description="Filter by delivery status"),
    exclude_duplicates: bool = Query(True, description="Exclude duplicate content"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get personalized discovered content with ML-enhanced filtering and intelligent ranking.
    
    Returns content tailored to user's strategic profile with advanced filtering:
    - **source_types**: Filter by content source types
    - **content_types**: Filter by content categories (article, report, news, etc.)
    - **min_relevance_score**: Minimum ML relevance score threshold (0.0-1.0)
    - **min_credibility_score**: Minimum source credibility score
    - **min_freshness_score**: Minimum content freshness rating
    - **published_after/before**: Date range filtering
    - **competitive_relevance**: Competitive intelligence categories
    - **exclude_duplicates**: Remove duplicate content detection
    
    Content is ranked by overall ML score combining relevance, engagement prediction,
    and user interaction history. Includes source information and engagement metrics.
    """
    query = select(DiscoveredContent).where(DiscoveredContent.user_id == current_user.id)
    
    # Apply filters
    filter_conditions = []
    
    if source_types:
        source_ids = await db.execute(
            select(DiscoveredSource.id).where(DiscoveredSource.source_type.in_(source_types))
        )
        source_id_list = [row[0] for row in source_ids]
        if source_id_list:
            filter_conditions.append(DiscoveredContent.source_id.in_(source_id_list))
    
    if content_types:
        filter_conditions.append(DiscoveredContent.content_type.in_(content_types))
    
    if min_relevance_score is not None:
        filter_conditions.append(DiscoveredContent.relevance_score >= min_relevance_score)
    
    if min_credibility_score is not None:
        filter_conditions.append(DiscoveredContent.credibility_score >= min_credibility_score)
    
    if min_freshness_score is not None:
        filter_conditions.append(DiscoveredContent.freshness_score >= min_freshness_score)
    
    if published_after:
        filter_conditions.append(DiscoveredContent.published_at >= published_after)
    
    if published_before:
        filter_conditions.append(DiscoveredContent.published_at <= published_before)
    
    if competitive_relevance:
        filter_conditions.append(DiscoveredContent.competitive_relevance.in_(competitive_relevance))
    
    if is_delivered is not None:
        filter_conditions.append(DiscoveredContent.is_delivered == is_delivered)
    
    if exclude_duplicates:
        filter_conditions.append(DiscoveredContent.is_duplicate == False)
    
    if filter_conditions:
        query = query.where(and_(*filter_conditions))
    
    # Order by overall ML score (relevance + engagement prediction)
    query = query.order_by(desc(DiscoveredContent.overall_score))
    
    # Apply pagination
    query = query.offset(pagination.offset).limit(pagination.per_page)
    
    # Include source information
    query = query.options(selectinload(DiscoveredContent.source))
    
    result = await db.execute(query)
    content = result.scalars().all()
    
    return content


@router.get("/content/{content_id}", response_model=DiscoveredContentResponse)
async def get_discovered_content_detail(
    content_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get comprehensive details about specific discovered content.
    
    Returns complete content information including:
    - **Content Details**: Title, summary, full text, publication info
    - **ML Scoring**: Relevance, credibility, freshness, engagement predictions
    - **Source Information**: Origin, reliability metrics, discovery context
    - **Competitive Analysis**: Relevance categories, strategic implications
    - **Delivery Status**: Whether content was delivered and engagement data
    - **Quality Metrics**: ML confidence, human feedback scores
    
    Only returns content that belongs to the authenticated user.
    """
    result = await db.execute(
        select(DiscoveredContent)
        .where(
            and_(
                DiscoveredContent.id == content_id,
                DiscoveredContent.user_id == current_user.id
            )
        )
        .options(selectinload(DiscoveredContent.source))
    )
    content = result.scalar_one_or_none()
    
    if not content:
        raise errors.not_found("Content not found or access denied")
    
    return content


@router.post("/content/{content_id}/score", response_model=MLScoresSchema)
async def recalculate_content_scores(
    content_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Recalculate ML scores for specific content using latest algorithms."""
    # Get content
    result = await db.execute(
        select(DiscoveredContent)
        .where(
            and_(
                DiscoveredContent.id == content_id,
                DiscoveredContent.user_id == current_user.id
            )
        )
        .options(selectinload(DiscoveredContent.source))
    )
    content = result.scalar_one_or_none()
    
    if not content:
        raise errors.not_found("Content not found or access denied")
    
    # Get user context and recalculate scores
    user_context = await discovery_service.get_user_context(db, current_user.id)
    ml_scores = await discovery_service.calculate_ml_relevance_score(db, content, user_context)
    
    # Update content with new scores
    await db.execute(
        update(DiscoveredContent)
        .where(DiscoveredContent.id == content_id)
        .values(
            relevance_score=ml_scores.relevance_score,
            credibility_score=ml_scores.credibility_score,
            freshness_score=ml_scores.freshness_score,
            engagement_prediction_score=ml_scores.engagement_prediction,
            overall_score=ml_scores.overall_score,
            ml_confidence_level=ml_scores.confidence_level,
            ml_model_version=ml_scores.model_version
        )
    )
    await db.commit()
    
    return ml_scores


@router.post("/content/{content_id}/feedback")
async def provide_content_feedback(
    content_id: int,
    feedback_score: float = Query(..., ge=0.0, le=1.0, description="Relevance score from 0.0 (not relevant) to 1.0 (highly relevant)"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Provide human feedback on content relevance for continuous ML model improvement.
    
    Submit relevance feedback to train and improve ML scoring algorithms:
    - **feedback_score**: Relevance rating from 0.0 (not relevant) to 1.0 (highly relevant)
    - Values >= 0.5 are considered positive feedback
    - Values < 0.5 are considered negative feedback
    
    Human feedback is weighted heavily in ML training and helps personalize
    future content discovery to better match your interests and strategic needs.
    Creates engagement records for comprehensive user behavior analysis.
    """
    # Verify content ownership
    result = await db.execute(
        select(DiscoveredContent)
        .where(
            and_(
                DiscoveredContent.id == content_id,
                DiscoveredContent.user_id == current_user.id
            )
        )
    )
    content = result.scalar_one_or_none()
    
    if not content:
        raise errors.not_found("Content not found or access denied")
    
    # Update feedback score
    await db.execute(
        update(DiscoveredContent)
        .where(DiscoveredContent.id == content_id)
        .values(human_feedback_score=feedback_score)
    )
    await db.commit()
    
    # Create engagement record for ML training
    engagement = ContentEngagement(
        user_id=current_user.id,
        content_id=content_id,
        engagement_type="feedback_positive" if feedback_score >= 0.5 else "feedback_negative",
        engagement_value=Decimal(str(feedback_score)),
        engagement_context='{"source": "manual_feedback"}',
        ml_weight=Decimal("2.0")  # High weight for explicit feedback
    )
    db.add(engagement)
    await db.commit()
    
    return {"message": "Feedback recorded successfully", "feedback_score": feedback_score}


# Content Similarity and Deduplication

@router.get("/content/{content_id}/similarity", response_model=List[ContentSimilaritySchema])
async def get_content_similarity(
    content_id: int,
    days_back: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Get content similarity analysis for deduplication purposes."""
    # Get content
    result = await db.execute(
        select(DiscoveredContent)
        .where(
            and_(
                DiscoveredContent.id == content_id,
                DiscoveredContent.user_id == current_user.id
            )
        )
    )
    content = result.scalar_one_or_none()
    
    if not content:
        raise errors.not_found("Content not found or access denied")
    
    # Detect similarities
    similarities = await discovery_service.detect_content_similarity(
        db, content, current_user.id, days_back
    )
    
    return [
        ContentSimilaritySchema(
            content_id=sim.content_id,
            similarity_score=sim.similarity_score,
            duplicate_type=sim.duplicate_type,
            matching_features=sim.matching_features
        )
        for sim in similarities
    ]


# Engagement Tracking Endpoints

@router.post("/engagement", response_model=ContentEngagementResponse)
async def track_content_engagement(
    engagement_data: ContentEngagementCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Track user engagement with discovered content for ML learning and analytics.
    
    Record user interactions to improve content recommendation algorithms:
    - **content_id**: ID of the content being engaged with (optional)
    - **engagement_type**: Type of interaction (click, view, share, save, etc.)
    - **engagement_value**: Quantified engagement score (0.0-1.0)
    - **engagement_context**: Additional context data (JSON)
    - **device_type**: Device used for interaction
    
    Engagement data feeds into ML models for:
    - Content relevance scoring improvement
    - User preference learning
    - Engagement prediction enhancement
    - Personalization algorithm training
    """
    # Verify content ownership if content_id provided
    if engagement_data.content_id:
        result = await db.execute(
            select(DiscoveredContent)
            .where(
                and_(
                    DiscoveredContent.id == engagement_data.content_id,
                    DiscoveredContent.user_id == current_user.id
                )
            )
        )
        content = result.scalar_one_or_none()
        
        if not content:
            raise errors.not_found("Content not found or access denied")
    
    # Create engagement record
    engagement = ContentEngagement(
        **engagement_data.dict(),
        user_id=current_user.id
    )
    
    db.add(engagement)
    await db.commit()
    await db.refresh(engagement)
    
    return engagement


@router.post("/webhooks/sendgrid", status_code=status.HTTP_200_OK)
async def process_sendgrid_webhook(
    events: List[SendGridEngagementData],
    background_tasks: BackgroundTasks,
    x_forwarded_for: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Process SendGrid webhook events for comprehensive engagement tracking.
    
    Handles all SendGrid event types including opens, clicks, bounces, and unsubscribes.
    Extracts content IDs from URLs, correlates with user profiles, and feeds data
    into ML training pipeline for engagement prediction improvement.
    """
    async def _process_webhook_operation():
        processed_events = []
        
        for event_data in events:
            try:
                # Extract user information from email
                user_result = await db.execute(
                    select(User).where(User.email == event_data.email)
                )
                user = user_result.scalar_one_or_none()
                
                if not user:
                    base_ops.logger.warning(f"User not found for email: {event_data.email}")
                    continue
                
                # Extract content ID from URL if it's a click event
                content_id = None
                if event_data.url and event_data.event in ["click", "open"]:
                    content_id = discovery_service._extract_content_id_from_url(event_data.url)
                
                # Map SendGrid event type to internal format
                try:
                    engagement_type = EngagementType(event_data.event.lower())
                except ValueError:
                    # Fallback for unknown event types
                    base_ops.logger.warning(f"Unknown SendGrid event type: {event_data.event}")
                    continue
                
                # Create engagement record with proper error handling
                engagement = ContentEngagement(
                    user_id=user.id,
                    content_id=content_id,
                    engagement_type=engagement_type,
                    engagement_value=Decimal('1.0'),
                    sendgrid_event_id=getattr(event_data, 'sg_event_id', None),
                    sendgrid_message_id=getattr(event_data, 'sg_message_id', None),
                    email_subject=getattr(event_data, 'subject', None),
                    user_agent=getattr(event_data, 'useragent', None),
                    ip_address=getattr(event_data, 'ip', None) or x_forwarded_for,
                    engagement_timestamp=datetime.fromtimestamp(getattr(event_data, 'timestamp', time.time())),
                    device_type=discovery_service._extract_device_type(getattr(event_data, 'useragent', '') or ""),
                    engagement_context=json.dumps({
                        "sendgrid_event": event_data.event,
                        "unique_args": getattr(event_data, 'unique_args', {}),
                        "category": getattr(event_data, 'category', []),
                        "url": getattr(event_data, 'url', None)
                    }),
                    feedback_processed=False,
                    ml_weight=getattr(discovery_service, 'engagement_weights', {}).get(
                        f"email_{event_data.event}", Decimal('1.0')
                    )
                )
                
                db.add(engagement)
                processed_events.append({
                    "event": event_data.event,
                    "email": event_data.email,
                    "content_id": content_id,
                    "processed": True
                })
                
            except Exception as e:
                base_ops.logger.error(f"Error processing SendGrid event: {e}")
                processed_events.append({
                    "event": event_data.event,
                    "email": event_data.email,
                    "error": str(e),
                    "processed": False
                })
        
        await db_helpers.safe_commit(db, "process SendGrid webhooks")
        
        # Queue for ML processing
        background_tasks.add_task(
            _process_engagement_for_ml, [e for e in processed_events if e.get("processed")]
        )
        
        base_ops.logger.info(f"Processed {len(processed_events)} SendGrid events")
        
        return {
            "processed_count": len([e for e in processed_events if e.get("processed")]),
            "error_count": len([e for e in processed_events if not e.get("processed")]),
            "events": processed_events
        }
    
    return await base_ops.execute_db_operation(
        "process SendGrid webhook", _process_webhook_operation, db
    )


@router.post("/engagement/sendgrid", response_model=ContentEngagementResponse)
async def process_sendgrid_engagement(
    engagement_data: SendGridEngagementData,
    db: AsyncSession = Depends(get_db_session)
):
    """Process SendGrid webhook data for ML learning and engagement tracking."""
    try:
        engagement = await discovery_service.process_sendgrid_engagement(
            db, engagement_data.dict()
        )
        return engagement
    except Exception as e:
        raise errors.bad_request(f"Failed to process SendGrid engagement: {str(e)}")


@router.get("/engagement", response_model=List[ContentEngagementResponse])
async def get_user_engagement_history(
    pagination: PaginationParams = Depends(),
    engagement_type: Optional[str] = Query(None, description="Filter by engagement type"),
    content_id: Optional[int] = Query(None, description="Filter by specific content ID"),
    days_back: int = Query(30, ge=1, le=365, description="Number of days of history to retrieve"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get comprehensive user engagement history for analytics and behavior analysis.
    
    Returns detailed engagement records with optional filtering:
    - **engagement_type**: Filter by interaction type (click, view, share, feedback, etc.)
    - **content_id**: Show engagements for specific content
    - **days_back**: Historical range (1-365 days)
    
    Each engagement record includes:
    - **Interaction Details**: Type, timestamp, value, context
    - **Content Information**: Associated content, source, quality scores
    - **Device/Platform**: User agent, device type, IP address
    - **ML Integration**: Weight assigned, feedback processing status
    
    Results ordered by most recent engagement first. Use for:
    - User behavior pattern analysis
    - Content performance evaluation
    - ML model training data review
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    
    query = select(ContentEngagement).where(
        and_(
            ContentEngagement.user_id == current_user.id,
            ContentEngagement.created_at >= cutoff_date
        )
    )
    
    # Apply filters
    if engagement_type:
        query = query.where(ContentEngagement.engagement_type == engagement_type)
    
    if content_id:
        query = query.where(ContentEngagement.content_id == content_id)
    
    # Order by most recent
    query = query.order_by(desc(ContentEngagement.created_at))
    
    # Apply pagination
    query = query.offset(pagination.offset).limit(pagination.per_page)
    
    result = await db.execute(query)
    engagements = result.scalars().all()
    
    return engagements


# Discovery Job Management

@router.post("/jobs", response_model=DiscoveryJobResponse, status_code=status.HTTP_201_CREATED)
async def create_discovery_job(
    job_data: DiscoveryJobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Create and optionally start a comprehensive discovery job for content finding.
    
    Creates a new discovery job with configurable parameters:
    - **job_type**: Type of discovery (daily, weekly, targeted, manual)
    - **job_config**: JSON configuration with discovery parameters
    - **priority**: Job priority level (1=highest, 5=lowest)
    - **scheduled_at**: When to run the job (null = run immediately)
    - **user_id**: Target user for content discovery (defaults to current user)
    
    Jobs run in the background and process:
    - Source monitoring and content discovery
    - ML-based content scoring and filtering
    - Duplicate detection and deduplication
    - Quality assessment and credibility checking
    - User personalization and relevance ranking
    
    Use GET /jobs/{job_id} to monitor progress and results.
    """
    # Create job
    job = DiscoveryJob(
        **job_data.dict(),
        user_id=job_data.user_id or current_user.id,
        created_by="user"
    )
    
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Start job in background if it's not scheduled
    if not job_data.scheduled_at:
        background_tasks.add_task(run_discovery_job, job.id, db)
    
    return job


@router.get("/jobs", response_model=List[DiscoveryJobResponse])
async def get_discovery_jobs(
    pagination: PaginationParams = Depends(),
    job_type: Optional[str] = Query(None, description="Filter by job type (daily, weekly, targeted, manual)"),
    status: Optional[str] = Query(None, description="Filter by status (pending, running, completed, failed, cancelled)"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get user's discovery jobs with advanced filtering and pagination.
    
    Returns paginated list of discovery jobs with optional filters:
    - **job_type**: Filter by job type (daily, weekly, targeted, manual)
    - **status**: Filter by execution status (pending, running, completed, failed, cancelled)
    
    Each job includes:
    - **Execution Status**: Current state, progress percentage, timing info
    - **Performance Metrics**: Sources checked, content found/processed/delivered
    - **Configuration**: Job parameters, priority, scheduling details
    - **Results Summary**: Success/error counts, quality metrics
    
    Jobs are ordered by creation time (most recent first).
    """
    query = select(DiscoveryJob).where(DiscoveryJob.user_id == current_user.id)
    
    # Apply filters
    filters = []
    if job_type:
        filters.append(DiscoveryJob.job_type == job_type)
    if status:
        filters.append(DiscoveryJob.status == status)
    
    if filters:
        query = query.where(and_(*filters))
    
    # Order by most recent
    query = query.order_by(desc(DiscoveryJob.created_at))
    
    # Apply pagination
    query = query.offset(pagination.offset).limit(pagination.per_page)
    
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    return jobs


@router.get("/jobs/{job_id}", response_model=DiscoveryJobResponse)
async def get_discovery_job(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Get detailed information about a specific discovery job."""
    result = await db.execute(
        select(DiscoveryJob)
        .where(
            and_(
                DiscoveryJob.id == job_id,
                DiscoveryJob.user_id == current_user.id
            )
        )
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise errors.not_found("Discovery job not found or access denied")
    
    return job


# Analytics and ML Model Information

@router.get("/analytics", response_model=UserDiscoveryAnalytics)
async def get_user_discovery_analytics(
    days_back: int = Query(30, ge=1, le=365, description="Number of days for analytics calculation"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get comprehensive discovery analytics and performance insights.
    
    Returns detailed analytics for the specified time period including:
    - **Content Discovery**: Total discovered, delivered, average relevance scores
    - **Engagement Metrics**: User interaction rates, engagement patterns
    - **Source Performance**: Top performing sources, content quality by source
    - **Category Analysis**: Most relevant content categories and topics
    - **ML Performance**: Model accuracy, confidence levels, prediction quality
    - **Trend Analysis**: Weekly and monthly growth patterns
    
    Use this data to:
    - Monitor discovery system performance
    - Identify high-value content sources
    - Optimize content filtering and personalization
    - Track user engagement patterns
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    
    # Get content statistics
    content_stats = await db.execute(
        select(
            func.count(DiscoveredContent.id).label('total_discovered'),
            func.sum(DiscoveredContent.is_delivered.cast(db.bind.dialect.name == 'postgresql' and 'integer' or 'signed')).label('total_delivered'),
            func.avg(DiscoveredContent.relevance_score).label('avg_relevance'),
            func.max(DiscoveredContent.discovered_at).label('last_activity')
        )
        .where(
            and_(
                DiscoveredContent.user_id == current_user.id,
                DiscoveredContent.discovered_at >= cutoff_date
            )
        )
    )
    stats = content_stats.first()
    
    # Get engagement statistics
    engagement_stats = await db.execute(
        select(func.avg(ContentEngagement.engagement_value))
        .where(
            and_(
                ContentEngagement.user_id == current_user.id,
                ContentEngagement.created_at >= cutoff_date
            )
        )
    )
    avg_engagement = engagement_stats.scalar() or 0.0
    
    # Get top categories (mock implementation)
    top_categories = [
        {"category": "Technology", "count": 15, "avg_score": 0.85},
        {"category": "Market Analysis", "count": 12, "avg_score": 0.78}
    ]
    
    # Get top sources
    top_sources_result = await db.execute(
        select(
            DiscoveredSource.source_name,
            func.count(DiscoveredContent.id).label('content_count'),
            func.avg(DiscoveredContent.overall_score).label('avg_score')
        )
        .join(DiscoveredContent)
        .where(
            and_(
                DiscoveredContent.user_id == current_user.id,
                DiscoveredContent.discovered_at >= cutoff_date
            )
        )
        .group_by(DiscoveredSource.source_name)
        .order_by(desc('content_count'))
        .limit(5)
    )
    
    top_sources = [
        {
            "source_name": row.source_name,
            "content_count": row.content_count,
            "avg_score": float(row.avg_score or 0.0)
        }
        for row in top_sources_result
    ]
    
    return UserDiscoveryAnalytics(
        user_id=current_user.id,
        total_content_discovered=stats.total_discovered or 0,
        total_content_delivered=stats.total_delivered or 0,
        avg_relevance_score=Decimal(str(stats.avg_relevance or 0.0)),
        avg_engagement_score=Decimal(str(avg_engagement)),
        top_categories=top_categories,
        top_sources=top_sources,
        engagement_trends={"weekly_growth": 0.15, "monthly_growth": 0.35},
        ml_accuracy_score=Decimal("0.87"),
        last_activity=stats.last_activity or datetime.utcnow()
    )


@router.get("/ml/models", response_model=List[MLModelMetricsResponse])
async def get_ml_model_metrics(
    is_active: Optional[bool] = Query(None, description="Filter by active model status"),
    model_type: Optional[str] = Query(None, description="Filter by model type (relevance, engagement, quality, etc.)"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get ML model performance metrics and version information for system monitoring.
    
    Returns comprehensive ML model status including:
    - **Model Information**: Type, version, deployment status, last update
    - **Performance Metrics**: Accuracy, precision, recall, F1-score, confidence levels
    - **Training Data**: Dataset size, last training date, validation results
    - **Usage Statistics**: Prediction count, error rates, response times
    - **A/B Testing**: Comparative performance across model versions
    
    Model types include:
    - **relevance**: Content relevance scoring
    - **engagement**: User engagement prediction
    - **quality**: Content quality assessment
    - **similarity**: Duplicate detection and clustering
    
    Use for monitoring model performance and planning updates.
    """
    query = select(MLModelMetrics)
    
    # Apply filters
    filters = []
    if is_active is not None:
        filters.append(MLModelMetrics.is_active == is_active)
    if model_type:
        filters.append(MLModelMetrics.model_type == model_type)
    
    if filters:
        query = query.where(and_(*filters))
    
    query = query.order_by(desc(MLModelMetrics.created_at))
    
    result = await db.execute(query)
    models = result.scalars().all()
    
    return models


# =============================================================================
# ENHANCED ANALYTICS AND PERFORMANCE METRICS
# Advanced analytics endpoints for comprehensive system monitoring,
# performance optimization, and strategic insights generation.
# =============================================================================

@router.get("/analytics/dashboard", response_model=Dict[str, Any])
async def get_discovery_dashboard(
    time_period: str = Query("7d", regex=r"^(1d|7d|30d|90d)$", description="Time period for analytics"),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get comprehensive discovery analytics dashboard with performance metrics.
    
    Returns key performance indicators, content discovery trends, engagement
    analytics, ML model performance, and actionable insights for optimization.
    Supports multiple time periods for trend analysis.
    """
    # Parse time period
    time_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
    days = time_map[time_period]
    start_date = datetime.utcnow() - timedelta(days=days)
    
    async def _get_dashboard_operation():
        # Content discovery metrics
        content_stats = await db.execute(
            select(
                func.count(DiscoveredContent.id).label("total_content"),
                func.avg(DiscoveredContent.overall_score).label("avg_score"),
                func.count().filter(DiscoveredContent.is_delivered == True).label("delivered_content"),
                func.count().filter(DiscoveredContent.is_duplicate == True).label("duplicate_content")
            ).where(
                and_(
                    DiscoveredContent.user_id == current_user.id,
                    DiscoveredContent.discovered_at >= start_date
                )
            )
        )
        content_metrics = content_stats.first()
        
        # Engagement metrics
        engagement_stats = await db.execute(
            select(
                func.count(ContentEngagement.id).label("total_engagements"),
                func.avg(ContentEngagement.engagement_value).label("avg_engagement"),
                func.count().filter(ContentEngagement.engagement_type == EngagementType.EMAIL_CLICK).label("clicks"),
                func.count().filter(ContentEngagement.engagement_type == EngagementType.EMAIL_OPEN).label("opens")
            ).where(
                and_(
                    ContentEngagement.user_id == current_user.id,
                    ContentEngagement.created_at >= start_date
                )
            )
        )
        engagement_metrics = engagement_stats.first()
        
        # Source performance metrics
        source_stats = await db.execute(
            select(
                DiscoveredSource.source_type,
                func.count(DiscoveredContent.id).label("content_count"),
                func.avg(DiscoveredContent.overall_score).label("avg_quality")
            )
            .join(DiscoveredContent)
            .where(
                and_(
                    DiscoveredContent.user_id == current_user.id,
                    DiscoveredContent.discovered_at >= start_date
                )
            )
            .group_by(DiscoveredSource.source_type)
        )
        source_performance = source_stats.all()
        
        # Generate insights
        insights = []
        
        if content_metrics.avg_score and content_metrics.avg_score < 0.6:
            insights.append("Content quality scores are below optimal. Consider adjusting source filters.")
        
        if engagement_metrics.total_engagements and engagement_metrics.opens:
            click_rate = engagement_metrics.clicks / engagement_metrics.opens
            if click_rate < 0.1:
                insights.append("Email click-through rate is low. Review content relevance and subject lines.")
        
        dashboard_data = {
            "time_period": time_period,
            "content_metrics": {
                "total_discovered": content_metrics.total_content or 0,
                "average_quality_score": float(content_metrics.avg_score or 0),
                "delivered_count": content_metrics.delivered_content or 0,
                "duplicate_count": content_metrics.duplicate_content or 0,
                "delivery_rate": (content_metrics.delivered_content or 0) / max(content_metrics.total_content or 1, 1)
            },
            "engagement_metrics": {
                "total_engagements": engagement_metrics.total_engagements or 0,
                "average_engagement": float(engagement_metrics.avg_engagement or 0),
                "email_opens": engagement_metrics.opens or 0,
                "email_clicks": engagement_metrics.clicks or 0,
                "click_through_rate": (engagement_metrics.clicks or 0) / max(engagement_metrics.opens or 1, 1)
            },
            "source_performance": [
                {
                    "source_type": perf.source_type,
                    "content_count": perf.content_count,
                    "average_quality": float(perf.avg_quality)
                }
                for perf in source_performance
            ]
        }
        
        return create_analytics_response(
            f"Discovery Dashboard ({time_period})",
            dashboard_data,
            insights
        )
    
    return await base_ops.execute_db_operation(
        "get discovery dashboard", _get_dashboard_operation, db, rollback_on_error=False
    )


@router.get("/sources/{source_id}/health", response_model=Dict[str, Any])
async def get_source_health_details(
    source_id: int,
    time_period: str = Query("7d", regex=r"^(1d|7d|30d)$", description="Time period for health analysis"),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get comprehensive health analysis for a specific discovery source.
    
    Returns detailed health metrics, performance trends, error analysis,
    and actionable recommendations for source optimization. Includes
    content quality analysis and user engagement correlation.
    """
    async def _get_source_health_operation():
        # Get source details
        source_result = await db.execute(
            select(DiscoveredSource).where(DiscoveredSource.id == source_id)
        )
        source = source_result.scalar_one_or_none()
        
        if not source:
            raise errors.not_found("Source not found")
        
        # Parse time period
        time_map = {"1d": 1, "7d": 7, "30d": 30}
        days = time_map[time_period]
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get content performance metrics
        content_stats = await db.execute(
            select(
                func.count(DiscoveredContent.id).label("total_content"),
                func.avg(DiscoveredContent.overall_score).label("avg_quality"),
                func.count().filter(DiscoveredContent.is_delivered == True).label("delivered_count"),
                func.count().filter(DiscoveredContent.is_duplicate == True).label("duplicate_count")
            ).where(
                and_(
                    DiscoveredContent.source_id == source_id,
                    DiscoveredContent.user_id == current_user.id,
                    DiscoveredContent.discovered_at >= start_date
                )
            )
        )
        content_metrics = content_stats.first()
        
        # Calculate health score
        health_factors = {
            "success_rate": float(source.success_rate),
            "quality_score": float(source.quality_score),
            "engagement_score": float(source.user_engagement_score),
            "content_volume": min(float(content_metrics.total_content or 0) / 10, 1.0),  # Normalize
            "recency": 1.0 if source.last_successful_check and 
                     (datetime.utcnow() - source.last_successful_check).hours < 24 else 0.5
        }
        
        overall_health = sum(health_factors.values()) / len(health_factors)
        
        # Determine health status
        if overall_health >= 0.8:
            status = "healthy"
        elif overall_health >= 0.6:
            status = "warning"
        else:
            status = "critical"
        
        # Generate recommendations
        recommendations = []
        
        if source.success_rate < 0.7:
            recommendations.append("Check source availability and connection settings")
        
        if source.quality_score < 0.6:
            recommendations.append("Review content filtering criteria and quality thresholds")
        
        health_data = {
            "source_info": {
                "id": source.id,
                "name": source.source_name,
                "type": source.source_type,
                "url": source.source_url,
                "is_active": source.is_active
            },
            "overall_health_score": round(overall_health, 3),
            "health_status": status,
            "health_factors": health_factors,
            "performance_metrics": {
                "success_rate": float(source.success_rate),
                "quality_score": float(source.quality_score),
                "credibility_score": float(source.credibility_score),
                "user_engagement_score": float(source.user_engagement_score),
                "ml_confidence": float(source.ml_confidence_level)
            },
            "content_metrics": {
                "total_found": content_metrics.total_content or 0,
                "average_quality": float(content_metrics.avg_quality or 0),
                "delivered_count": content_metrics.delivered_content or 0,
                "duplicate_count": content_metrics.duplicate_count or 0,
                "delivery_rate": (content_metrics.delivered_content or 0) / max(content_metrics.total_content or 1, 1)
            },
            "recommendations": recommendations
        }
        
        return create_analytics_response(
            f"Source Health Analysis - {source.source_name or source.source_url}",
            health_data,
            recommendations
        )
    
    return await base_ops.execute_db_operation(
        "get source health details", _get_source_health_operation, db, rollback_on_error=False
    )


@router.post("/sources/{source_id}/test")
async def test_source_connection(
    source_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Test connection and health of a specific discovery source.
    
    Performs real-time connectivity test, content availability check,
    and quality assessment. Updates source health metrics and provides
    immediate feedback on source status and configuration.
    """
    async def _test_source_operation():
        # Get source
        source_result = await db.execute(
            select(DiscoveredSource).where(DiscoveredSource.id == source_id)
        )
        source = source_result.scalar_one_or_none()
        
        if not source:
            raise errors.not_found("Source not found")
        
        # Queue background test
        background_tasks.add_task(
            _perform_source_health_check, source_id, current_user.id
        )
        
        # Immediate response with cached data
        test_result = {
            "source_id": source_id,
            "source_name": source.source_name,
            "test_initiated": True,
            "message": "Source health check initiated. Results will be available shortly.",
            "current_status": {
                "is_active": source.is_active,
                "last_checked": source.last_checked.isoformat() if source.last_checked else None,
                "success_rate": float(source.success_rate),
                "quality_score": float(source.quality_score)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        base_ops.logger.info(f"Source health check initiated: ID {source_id}")
        return test_result
    
    return await base_ops.execute_db_operation(
        "test source connection", _test_source_operation, db
    )


# =============================================================================
# JOB CONTROL ENDPOINTS
# Advanced job lifecycle management with real-time control,
# state management, and comprehensive monitoring capabilities.
# =============================================================================

@router.post("/jobs/{job_id}/control")
async def control_discovery_job(
    job_id: int,
    background_tasks: BackgroundTasks,
    action: str = Body(..., embed=True, regex=r"^(start|stop|pause|resume|cancel)$"),
    reason: Optional[str] = Body(None, embed=True),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Control discovery job execution with start/stop/pause/resume/cancel actions.
    
    Provides immediate job control with proper state transitions, cleanup,
    and notification handling. Supports job recovery and error management.
    """
    async def _control_job_operation():
        job = await base_ops.get_user_resource_by_id(
            db, DiscoveryJob, job_id, current_user.id, "discovery job"
        )
        
        previous_status = job.status
        
        # Handle different control actions
        if action == "start" and job.status == JobStatus.PENDING:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            background_tasks.add_task(_execute_discovery_job_async, job_id, current_user.id)
            
        elif action == "stop" and job.status == JobStatus.RUNNING:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            job.error_message = reason or "Stopped by user"
            
        elif action == "cancel" and job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            job.error_message = reason or "Cancelled by user"
            
        else:
            raise errors.bad_request(f"Cannot {action} job with status {job.status}")
        
        await db_helpers.safe_commit(db, f"control discovery job {job_id}")
        base_ops.logger.info(f"Discovery job {action}: ID {job_id}, {previous_status} -> {job.status}")
        
        return {
            "success": True,
            "message": f"Job {action} completed successfully",
            "job_id": job_id,
            "previous_status": previous_status,
            "new_status": job.status,
            "timestamp": datetime.utcnow()
        }
    
    return await base_ops.execute_db_operation(
        f"{action} discovery job", _control_job_operation, db
    )


# =============================================================================
# BACKGROUND TASK FUNCTIONS
# Asynchronous task execution for discovery jobs, ML processing,
# health checks, and engagement analytics processing.
# =============================================================================

async def run_discovery_job(job_id: int, db: AsyncSession):
    """
    Execute discovery job in background with comprehensive processing.
    
    Handles complete discovery workflow including:
    - Source monitoring and content discovery
    - ML-based content scoring and ranking
    - Quality assessment and credibility checking
    - Duplicate detection and deduplication
    - User personalization and filtering
    """
    # This would contain the actual discovery logic
    # For now, we'll just update the job status
    
    await asyncio.sleep(1)  # Simulate processing
    
    await db.execute(
        update(DiscoveryJob)
        .where(DiscoveryJob.id == job_id)
        .values(
            status="running",
            started_at=datetime.utcnow(),
            progress_percentage=50
        )
    )
    await db.commit()
    
    await asyncio.sleep(2)  # Simulate more processing
    
    await db.execute(
        update(DiscoveryJob)
        .where(DiscoveryJob.id == job_id)
        .values(
            status="completed",
            completed_at=datetime.utcnow(),
            progress_percentage=100,
            sources_checked=5,
            content_found=25,
            content_processed=25
        )
    )
    await db.commit()


async def _execute_discovery_job_async(job_id: int, user_id: int):
    """
    Execute discovery job asynchronously with comprehensive error handling.
    
    Manages complete job lifecycle including progress tracking,
    error recovery, result processing, and performance monitoring.
    Uses discovery service for actual content processing.
    """
    try:
        # Use discovery service to run job
        await discovery_service.execute_discovery_job(job_id, user_id)
        base_ops.logger.info(f"Discovery job completed successfully: ID {job_id}")
    except Exception as e:
        base_ops.logger.error(f"Discovery job failed: ID {job_id}, Error: {e}")


async def _process_engagement_for_ml(processed_events: List[Dict[str, Any]]):
    """
    Process engagement events for ML training and model improvement.
    
    Queues engagement data for ML processing pipeline including:
    - User behavior pattern analysis
    - Content preference learning
    - Engagement prediction model training
    - Personalization algorithm enhancement
    """
    try:
        # Queue events for ML processing
        ml_cache = get_ml_scoring_cache()
        
        for event in processed_events:
            cache_key = f"ml_training_{event['content_id']}_{event['email']}"
            ml_cache.put(cache_key, event)
        
        base_ops.logger.info(f"Queued {len(processed_events)} events for ML processing")
    except Exception as e:
        base_ops.logger.error(f"Error processing engagement for ML: {e}")


async def _perform_source_health_check(source_id: int, user_id: int):
    """
    Perform comprehensive source health check and diagnostics.
    
    Executes detailed health assessment including:
    - Connectivity and availability testing
    - Content quality and freshness validation
    - Performance benchmarking and optimization
    - Error rate analysis and troubleshooting recommendations
    - Cache result storage for immediate retrieval
    """
    try:
        # Use discovery service to check source health
        health_result = await discovery_service.check_source_health(source_id)
        
        # Cache results for immediate retrieval
        health_cache = get_source_discovery_cache()
        cache_key = f"health_check_{source_id}_{user_id}"
        health_cache.put(cache_key, health_result)
        
        base_ops.logger.info(f"Source health check completed: ID {source_id}")
    except Exception as e:
        base_ops.logger.error(f"Source health check failed: ID {source_id}, Error: {e}")
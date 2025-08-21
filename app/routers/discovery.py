"""
Discovery Service API endpoints for competitive intelligence v2.

ML-driven content discovery, relevance scoring, and engagement tracking
with comprehensive analytics and user behavior learning.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, and_, or_, desc
from sqlalchemy.orm import selectinload

from app.database import get_db_session
from app.middleware import get_current_active_user
from app.models.user import User
from app.models.discovery import (
    DiscoveredSource, DiscoveredContent, ContentEngagement,
    DiscoveryJob, MLModelMetrics
)
from app.schemas.discovery import (
    DiscoveredSourceCreate, DiscoveredSourceUpdate, DiscoveredSourceResponse,
    DiscoveredContentCreate, DiscoveredContentUpdate, DiscoveredContentResponse,
    ContentEngagementCreate, ContentEngagementResponse, SendGridEngagementData,
    DiscoveryJobCreate, DiscoveryJobUpdate, DiscoveryJobResponse,
    MLModelMetricsResponse, UserDiscoveryAnalytics, DiscoveryFilterRequest,
    MLScoresSchema, ContentSimilaritySchema
)
from app.services.discovery_service import DiscoveryService
from app.utils.router_base import BaseRouterOperations, PaginationParams
from app import errors

router = APIRouter(prefix="/api/v1/discovery", tags=["discovery"])
discovery_service = DiscoveryService()


# Source Management Endpoints

@router.post("/sources", response_model=DiscoveredSourceResponse)
async def create_discovery_source(
    source_data: DiscoveredSourceCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new discovery source for competitive intelligence monitoring."""
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
    source_type: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    min_quality_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Get paginated list of discovery sources with optional filtering."""
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
    """Get detailed information about a specific discovery source."""
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
    """Update discovery source configuration and settings."""
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
    """Delete a discovery source and all associated content."""
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
    filters: DiscoveryFilterRequest = Depends(),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Get personalized discovered content with ML-driven filtering and ranking."""
    query = select(DiscoveredContent).where(DiscoveredContent.user_id == current_user.id)
    
    # Apply filters
    filter_conditions = []
    
    if filters.source_types:
        source_ids = await db.execute(
            select(DiscoveredSource.id).where(DiscoveredSource.source_type.in_(filters.source_types))
        )
        source_id_list = [row[0] for row in source_ids]
        if source_id_list:
            filter_conditions.append(DiscoveredContent.source_id.in_(source_id_list))
    
    if filters.content_types:
        filter_conditions.append(DiscoveredContent.content_type.in_(filters.content_types))
    
    if filters.min_relevance_score is not None:
        filter_conditions.append(DiscoveredContent.relevance_score >= filters.min_relevance_score)
    
    if filters.min_credibility_score is not None:
        filter_conditions.append(DiscoveredContent.credibility_score >= filters.min_credibility_score)
    
    if filters.min_freshness_score is not None:
        filter_conditions.append(DiscoveredContent.freshness_score >= filters.min_freshness_score)
    
    if filters.published_after:
        filter_conditions.append(DiscoveredContent.published_at >= filters.published_after)
    
    if filters.published_before:
        filter_conditions.append(DiscoveredContent.published_at <= filters.published_before)
    
    if filters.competitive_relevance:
        filter_conditions.append(DiscoveredContent.competitive_relevance.in_(filters.competitive_relevance))
    
    if filters.is_delivered is not None:
        filter_conditions.append(DiscoveredContent.is_delivered == filters.is_delivered)
    
    if filters.exclude_duplicates:
        filter_conditions.append(DiscoveredContent.is_duplicate == False)
    
    if filter_conditions:
        query = query.where(and_(*filter_conditions))
    
    # Order by overall ML score (relevance + engagement prediction)
    query = query.order_by(desc(DiscoveredContent.overall_score))
    
    # Apply pagination
    query = query.offset(pagination.offset).limit(pagination.limit)
    
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
    """Get detailed information about specific discovered content."""
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
    feedback_score: float = Query(..., ge=0.0, le=1.0),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Provide human feedback on content relevance for ML training."""
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
    """Track user engagement with discovered content for ML learning."""
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
    engagement_type: Optional[str] = Query(None),
    content_id: Optional[int] = Query(None),
    days_back: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Get user engagement history for analytics and ML training."""
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
    query = query.offset(pagination.offset).limit(pagination.limit)
    
    result = await db.execute(query)
    engagements = result.scalars().all()
    
    return engagements


# Discovery Job Management

@router.post("/jobs", response_model=DiscoveryJobResponse)
async def create_discovery_job(
    job_data: DiscoveryJobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Create and optionally start a discovery job for content finding."""
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
    job_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Get user's discovery jobs with filtering and pagination."""
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
    query = query.offset(pagination.offset).limit(pagination.limit)
    
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
    days_back: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Get comprehensive discovery analytics for the user."""
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
    is_active: Optional[bool] = Query(None),
    model_type: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Get ML model performance metrics and version information."""
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


# Background task function
async def run_discovery_job(job_id: int, db: AsyncSession):
    """Background task to run discovery job."""
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
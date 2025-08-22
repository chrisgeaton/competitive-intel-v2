"""
Analysis Service REST API endpoints for competitive intelligence v2.

Comprehensive endpoints for AI-powered content analysis, strategic insights generation,
batch processing, and analysis result management. Following established patterns 
with BaseRouterOperations integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc, select, and_, func
from sqlalchemy.orm import selectinload

from app.database import get_db_session
from app.services.analysis_service import AnalysisService
from app.services.ai_service import AIProvider
from app.models.analysis import AnalysisResult, StrategicInsight, AnalysisJob
from app.models.user import User
from app.models.discovery import DiscoveredContent
from app.analysis.core import AnalysisStage, ContentPriority, AnalysisContext
from app.middleware import get_current_user
from app.utils.exceptions import errors


# Initialize router and service
router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
analysis_service = AnalysisService()
logger = logging.getLogger(__name__)


# === Pydantic Schemas ===

from pydantic import BaseModel, Field, validator
from enum import Enum


class AnalysisStageEnum(str, Enum):
    """Analysis stages for API."""
    FILTERING = "filtering"
    RELEVANCE_ANALYSIS = "relevance_analysis"
    INSIGHT_EXTRACTION = "insight_extraction"
    SUMMARY_GENERATION = "summary_generation"


class AnalysisBatchCreate(BaseModel):
    """Request to create analysis batch."""
    max_items: Optional[int] = Field(default=10, ge=1, le=100)
    priority_filter: Optional[List[str]] = Field(default=None)
    stages: List[AnalysisStageEnum] = Field(default=[AnalysisStageEnum.RELEVANCE_ANALYSIS, AnalysisStageEnum.INSIGHT_EXTRACTION])
    analysis_depth: Optional[str] = Field(default="standard")  # quick, standard, deep


class AnalysisBatchResponse(BaseModel):
    """Analysis batch information."""
    batch_id: str
    user_id: int
    total_items: int
    filtered_items: int
    estimated_cost_cents: int
    priority: str
    stages: List[str]
    created_at: datetime
    status: str = "created"


class AnalysisResultResponse(BaseModel):
    """Analysis result with insights."""
    id: int
    content_id: int
    batch_id: str
    user_id: int
    
    # Filter stage results
    filter_passed: bool
    filter_score: float
    filter_priority: str
    matched_keywords: Optional[List[str]]
    matched_entities: Optional[List[str]]
    
    # Relevance analysis results
    relevance_score: Optional[float]
    strategic_alignment: Optional[float]
    competitive_impact: Optional[float]
    urgency_score: Optional[float]
    
    # Insight extraction results
    key_insights: Optional[List[str]]
    action_items: Optional[List[str]]
    strategic_implications: Optional[List[str]]
    risk_assessment: Optional[Dict[str, Any]]
    opportunity_assessment: Optional[Dict[str, Any]]
    
    # Summary results
    executive_summary: Optional[str]
    detailed_analysis: Optional[str]
    confidence_reasoning: Optional[str]
    
    # Metadata
    stage_completed: str
    confidence_level: float
    ai_cost_cents: int
    processing_time_ms: int
    analysis_timestamp: datetime
    created_at: datetime


class StrategicInsightResponse(BaseModel):
    """Strategic insight details."""
    id: int
    analysis_result_id: int
    insight_type: str
    insight_category: str
    insight_priority: str
    insight_title: str
    insight_description: str
    insight_implications: Optional[str]
    suggested_actions: Optional[List[str]]
    timeline_relevance: Optional[str]
    estimated_impact: Optional[str]
    relevance_score: float
    novelty_score: float
    actionability_score: float
    user_rating: Optional[int]
    marked_as_actionable: bool
    action_taken: bool
    extracted_at: datetime


class AnalysisStatsResponse(BaseModel):
    """Analysis statistics."""
    total_analyzed: int
    total_cost_cents: int
    average_relevance: float
    insights_generated: int
    stage_1_savings_percent: int
    top_insight_types: List[Dict[str, Any]]
    processing_performance: Dict[str, Any]
    cost_breakdown: Dict[str, Any]


class SingleContentAnalysisRequest(BaseModel):
    """Request for single content analysis."""
    content_id: int
    stages: List[AnalysisStageEnum] = Field(default=[AnalysisStageEnum.RELEVANCE_ANALYSIS, AnalysisStageEnum.INSIGHT_EXTRACTION])
    ai_provider: Optional[str] = Field(default=None)  # openai, anthropic, mock
    force_reanalysis: bool = Field(default=False)


# === Analysis Endpoints ===

@router.post("/batches", response_model=AnalysisBatchResponse)
async def create_analysis_batch(
    batch_request: AnalysisBatchCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Create analysis batch for pending content.
    
    Creates a batch of content items for multi-stage AI analysis,
    starting with Stage 1 filtering for cost optimization.
    """
    try:
        # Create analysis batch
        batch = await analysis_service.create_analysis_batch(
            db=db,
            user_id=current_user.id,
            max_items=batch_request.max_items
        )
        
        if not batch:
            raise HTTPException(
                status_code=404,
                detail="No pending content available for analysis"
            )
        
        # Convert stages
        stages = [AnalysisStage(stage.value) for stage in batch_request.stages]
        
        # Estimate cost
        cost_estimate = await analysis_service.estimate_batch_cost(batch, stages)
        
        # Schedule background analysis
        background_tasks.add_task(
            process_analysis_batch,
            batch,
            stages,
            db
        )
        
        return AnalysisBatchResponse(
            batch_id=batch.batch_id,
            user_id=batch.user_id,
            total_items=len(batch.content_items),
            filtered_items=len(batch.content_items),  # Already filtered in batch creation
            estimated_cost_cents=cost_estimate["estimated_cost"],
            priority=batch.priority.value,
            stages=[stage.value for stage in stages],
            created_at=datetime.now(),
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"Failed to create analysis batch for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content/{content_id}/analyze", response_model=AnalysisResultResponse)
async def analyze_single_content(
    content_id: int = Path(..., description="Content ID to analyze"),
    analysis_request: SingleContentAnalysisRequest = Body(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Analyze single content item with AI.
    
    Performs complete multi-stage analysis on a specific content item
    with real-time results.
    """
    try:
        # Verify content exists and belongs to user's context
        content_query = select(DiscoveredContent).where(
            and_(
                DiscoveredContent.id == content_id,
                DiscoveredContent.user_id == current_user.id
            )
        )
        result = await db.execute(content_query)
        content = result.scalar_one_or_none()
        
        if not content:
            raise HTTPException(
                status_code=404,
                detail=f"Content {content_id} not found or not accessible"
            )
        
        # Check for existing analysis if not forcing reanalysis
        if not analysis_request.force_reanalysis:
            existing_query = select(AnalysisResult).where(
                and_(
                    AnalysisResult.content_id == content_id,
                    AnalysisResult.user_id == current_user.id
                )
            ).order_by(desc(AnalysisResult.created_at))
            
            existing_result = await db.execute(existing_query)
            existing = existing_result.scalar_one_or_none()
            
            if existing:
                return _convert_to_response(existing)
        
        # Create single-item batch
        batch = await analysis_service.create_analysis_batch(
            db=db,
            user_id=current_user.id,
            max_items=1
        )
        
        if not batch or not batch.content_items:
            raise HTTPException(
                status_code=400,
                detail="Content did not pass initial filtering"
            )
        
        # Filter to specific content
        batch.content_items = [
            item for item in batch.content_items 
            if item["id"] == content_id
        ]
        
        if not batch.content_items:
            raise HTTPException(
                status_code=400,
                detail="Specified content did not pass filtering criteria"
            )
        
        # Convert stages
        stages = [AnalysisStage(stage.value) for stage in analysis_request.stages]
        
        # Perform analysis
        analysis_results = await analysis_service.perform_deep_analysis(
            db=db,
            batch=batch,
            stages=stages
        )
        
        if not analysis_results:
            raise HTTPException(
                status_code=500,
                detail="Analysis failed to generate results"
            )
        
        # Save results
        saved_ids = await analysis_service.save_analysis_results(db, analysis_results)
        
        if not saved_ids:
            raise HTTPException(
                status_code=500,
                detail="Failed to save analysis results"
            )
        
        # Return latest result
        result_query = select(AnalysisResult).where(
            AnalysisResult.id == saved_ids[0]
        )
        db_result = await db.execute(result_query)
        analysis_result = db_result.scalar_one()
        
        return _convert_to_response(analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single content analysis failed for content {content_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results", response_model=List[AnalysisResultResponse])
async def get_analysis_results(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    batch_id: Optional[str] = Query(default=None),
    min_relevance: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    priority_filter: Optional[List[str]] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get analysis results for user.
    
    Retrieves paginated analysis results with optional filtering
    by batch, relevance score, and priority.
    """
    try:
        # Build query
        query = select(AnalysisResult).where(
            AnalysisResult.user_id == current_user.id
        )
        
        # Apply filters
        if batch_id:
            query = query.where(AnalysisResult.analysis_batch_id == batch_id)
        
        if min_relevance is not None:
            query = query.where(AnalysisResult.relevance_score >= min_relevance)
        
        if priority_filter:
            query = query.where(AnalysisResult.filter_priority.in_(priority_filter))
        
        # Apply pagination and ordering
        query = query.order_by(desc(AnalysisResult.analysis_timestamp)) \
                    .offset(offset) \
                    .limit(limit)
        
        # Execute query
        result = await db.execute(query)
        analysis_results = result.scalars().all()
        
        # Convert to response models
        return [_convert_to_response(result) for result in analysis_results]
        
    except Exception as e:
        logger.error(f"Failed to get analysis results for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights", response_model=List[StrategicInsightResponse])
async def get_strategic_insights(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    insight_type: Optional[str] = Query(default=None),
    priority_filter: Optional[List[str]] = Query(default=None),
    actionable_only: bool = Query(default=False),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get strategic insights for user.
    
    Retrieves paginated strategic insights with filtering options
    for type, priority, and actionable status.
    """
    try:
        # Build query
        query = select(StrategicInsight).where(
            StrategicInsight.user_id == current_user.id
        )
        
        # Apply filters
        if insight_type:
            query = query.where(StrategicInsight.insight_type == insight_type)
        
        if priority_filter:
            query = query.where(StrategicInsight.insight_priority.in_(priority_filter))
        
        if actionable_only:
            query = query.where(StrategicInsight.marked_as_actionable == True)
        
        # Apply pagination and ordering
        query = query.order_by(desc(StrategicInsight.extracted_at)) \
                    .offset(offset) \
                    .limit(limit)
        
        # Execute query
        result = await db.execute(query)
        insights = result.scalars().all()
        
        # Convert to response models
        return [_convert_insight_to_response(insight) for insight in insights]
        
    except Exception as e:
        logger.error(f"Failed to get insights for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/insights/{insight_id}/actionable")
async def mark_insight_actionable(
    insight_id: int = Path(..., description="Insight ID"),
    actionable: bool = Body(..., embed=True),
    notes: Optional[str] = Body(default=None, embed=True),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """Mark strategic insight as actionable or not actionable."""
    try:
        # Find insight
        query = select(StrategicInsight).where(
            and_(
                StrategicInsight.id == insight_id,
                StrategicInsight.user_id == current_user.id
            )
        )
        result = await db.execute(query)
        insight = result.scalar_one_or_none()
        
        if not insight:
            raise HTTPException(
                status_code=404,
                detail=f"Insight {insight_id} not found"
            )
        
        # Update insight
        insight.marked_as_actionable = actionable
        if notes:
            insight.action_notes = notes
        
        await db.commit()
        
        return {"message": f"Insight marked as {'actionable' if actionable else 'not actionable'}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update insight {insight_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=AnalysisStatsResponse)
async def get_analysis_stats(
    days: int = Query(default=30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get analysis statistics for user.
    
    Provides comprehensive analytics on analysis performance,
    costs, insights generated, and processing metrics.
    """
    try:
        # Date range
        start_date = datetime.now() - timedelta(days=days)
        
        # Get analysis results count and metrics
        analysis_query = select(
            func.count(AnalysisResult.id).label("total_count"),
            func.sum(AnalysisResult.ai_cost_cents).label("total_cost"),
            func.avg(AnalysisResult.relevance_score).label("avg_relevance"),
            func.sum(AnalysisResult.processing_time_ms).label("total_processing_time")
        ).where(
            and_(
                AnalysisResult.user_id == current_user.id,
                AnalysisResult.created_at >= start_date
            )
        )
        
        analysis_result = await db.execute(analysis_query)
        analysis_stats = analysis_result.first()
        
        # Get insights count and types
        insights_query = select(
            func.count(StrategicInsight.id).label("insight_count"),
            StrategicInsight.insight_type
        ).where(
            and_(
                StrategicInsight.user_id == current_user.id,
                StrategicInsight.extracted_at >= start_date
            )
        ).group_by(StrategicInsight.insight_type)
        
        insights_result = await db.execute(insights_query)
        insight_types = insights_result.all()
        
        # Calculate performance metrics
        total_analyzed = analysis_stats.total_count or 0
        total_cost = analysis_stats.total_cost or 0
        avg_relevance = float(analysis_stats.avg_relevance or 0.0)
        total_insights = sum(row.insight_count for row in insight_types)
        avg_processing_time = (analysis_stats.total_processing_time or 0) / max(1, total_analyzed)
        
        return AnalysisStatsResponse(
            total_analyzed=total_analyzed,
            total_cost_cents=total_cost,
            average_relevance=avg_relevance,
            insights_generated=total_insights,
            stage_1_savings_percent=70,  # From Stage 1 filtering
            top_insight_types=[
                {"type": row.insight_type, "count": row.insight_count}
                for row in insight_types
            ],
            processing_performance={
                "avg_processing_time_ms": avg_processing_time,
                "items_per_second": 1000.0 / max(1, avg_processing_time),
                "cost_per_item_cents": total_cost / max(1, total_analyzed)
            },
            cost_breakdown={
                "filtering_stage": 0,  # Stage 1 is free (rule-based)
                "relevance_analysis": int(total_cost * 0.4),
                "insight_extraction": int(total_cost * 0.6)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get analysis stats for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Background Task Functions ===

async def process_analysis_batch(
    batch,
    stages: List[AnalysisStage],
    db: AsyncSession
):
    """Background task to process analysis batch."""
    try:
        # Perform deep analysis
        analysis_results = await analysis_service.perform_deep_analysis(
            db=db,
            batch=batch,
            stages=stages
        )
        
        # Save results
        if analysis_results:
            await analysis_service.save_analysis_results(db, analysis_results)
            
        logger.info(f"Completed background analysis for batch {batch.batch_id}")
        
    except Exception as e:
        logger.error(f"Background analysis failed for batch {batch.batch_id}: {e}")


# === Helper Functions ===

def _convert_to_response(analysis_result: AnalysisResult) -> AnalysisResultResponse:
    """Convert database AnalysisResult to response model."""
    return AnalysisResultResponse(
        id=analysis_result.id,
        content_id=analysis_result.content_id,
        batch_id=analysis_result.analysis_batch_id or "",
        user_id=analysis_result.user_id,
        filter_passed=analysis_result.filter_passed,
        filter_score=float(analysis_result.filter_score or 0.0),
        filter_priority=analysis_result.filter_priority,
        matched_keywords=analysis_result.filter_matched_keywords,
        matched_entities=analysis_result.filter_matched_entities,
        relevance_score=float(analysis_result.relevance_score) if analysis_result.relevance_score else None,
        strategic_alignment=float(analysis_result.strategic_alignment) if analysis_result.strategic_alignment else None,
        competitive_impact=float(analysis_result.competitive_impact) if analysis_result.competitive_impact else None,
        urgency_score=float(analysis_result.urgency_score) if analysis_result.urgency_score else None,
        key_insights=analysis_result.key_insights,
        action_items=analysis_result.action_items,
        strategic_implications=analysis_result.strategic_implications,
        risk_assessment=analysis_result.risk_assessment,
        opportunity_assessment=analysis_result.opportunity_assessment,
        executive_summary=analysis_result.executive_summary,
        detailed_analysis=analysis_result.detailed_analysis,
        confidence_reasoning=analysis_result.confidence_reasoning,
        stage_completed=analysis_result.stage_completed,
        confidence_level=float(analysis_result.confidence_level or 0.0),
        ai_cost_cents=analysis_result.ai_cost_cents or 0,
        processing_time_ms=analysis_result.processing_time_ms or 0,
        analysis_timestamp=analysis_result.analysis_timestamp,
        created_at=analysis_result.created_at
    )


def _convert_insight_to_response(insight: StrategicInsight) -> StrategicInsightResponse:
    """Convert database StrategicInsight to response model."""
    return StrategicInsightResponse(
        id=insight.id,
        analysis_result_id=insight.analysis_result_id,
        insight_type=insight.insight_type,
        insight_category=insight.insight_category,
        insight_priority=insight.insight_priority,
        insight_title=insight.insight_title,
        insight_description=insight.insight_description,
        insight_implications=insight.insight_implications,
        suggested_actions=insight.suggested_actions,
        timeline_relevance=insight.timeline_relevance,
        estimated_impact=insight.estimated_impact,
        relevance_score=float(insight.relevance_score or 0.0),
        novelty_score=float(insight.novelty_score or 0.0),
        actionability_score=float(insight.actionability_score or 0.0),
        user_rating=insight.user_rating,
        marked_as_actionable=insight.marked_as_actionable or False,
        action_taken=insight.action_taken or False,
        extracted_at=insight.extracted_at
    )
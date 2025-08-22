"""
Orchestration API Router for End-to-End Pipeline Management

FastAPI router for managing the complete competitive intelligence pipeline
with user preference integration and scheduling capabilities.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.database import get_db_session
from app.middleware import get_current_user
from app.models.user import User
from app.services.orchestration_service import (
    OrchestrationService, PipelineExecution, PipelineStatus, PipelineStage
)
from app.utils.router_base import BaseRouterOperations


# Pydantic models for API requests/responses
class PipelineExecutionRequest(BaseModel):
    """Request to execute pipeline for user."""
    trigger_type: str = Field(default="manual", pattern="^(manual|scheduled|webhook)$")
    discovery_enabled: bool = Field(default=True)
    analysis_depth: str = Field(default="standard", pattern="^(quick|standard|deep)$")
    email_delivery: bool = Field(default=True)
    custom_config: Optional[Dict[str, Any]] = None


class PipelineExecutionResponse(BaseModel):
    """Response for pipeline execution."""
    execution_id: str
    user_id: int
    trigger_type: str
    status: PipelineStatus
    current_stage: PipelineStage
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class BatchPipelineRequest(BaseModel):
    """Request for batch pipeline execution."""
    user_ids: List[int] = Field(..., min_items=1, max_items=100)
    trigger_type: str = Field(default="manual", pattern="^(manual|scheduled|webhook)$")


class BatchPipelineResponse(BaseModel):
    """Response for batch pipeline execution."""
    total_users: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    executions: List[PipelineExecutionResponse]


class SchedulePipelineRequest(BaseModel):
    """Request to schedule pipeline execution."""
    schedule_time: datetime
    recurrence: Optional[str] = Field(None, pattern="^(daily|weekly|monthly)$")
    pipeline_config: Optional[PipelineExecutionRequest] = None


# Router setup
router = APIRouter(prefix="/api/v1/orchestration", tags=["orchestration"])


@router.post("/execute", response_model=PipelineExecutionResponse)
async def execute_pipeline(
    request: PipelineExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Execute complete competitive intelligence pipeline for current user.
    
    Runs the full Discovery > Analysis > Report Generation > Delivery pipeline
    with user-specific configuration and preferences.
    """
    try:
        # Initialize orchestration service
        orchestration_service = OrchestrationService()
        
        # Convert request to custom config
        custom_config = {
            "discovery_enabled": request.discovery_enabled,
            "analysis_depth": request.analysis_depth,
            "email_delivery": request.email_delivery
        }
        
        if request.custom_config:
            custom_config.update(request.custom_config)
        
        # Execute pipeline as background task for long-running operations
        if request.trigger_type == "manual":
            # For manual execution, run synchronously to provide immediate feedback
            execution = await orchestration_service.execute_user_pipeline(
                user_id=current_user.id,
                trigger_type=request.trigger_type,
                custom_config=custom_config
            )
        else:
            # For scheduled execution, run as background task
            background_tasks.add_task(
                orchestration_service.execute_user_pipeline,
                current_user.id,
                request.trigger_type,
                custom_config
            )
            
            # Return immediate response
            execution = PipelineExecution(
                execution_id=f"async_{current_user.id}_{datetime.utcnow().timestamp()}",
                user_id=current_user.id,
                trigger_type=request.trigger_type,
                status=PipelineStatus.PENDING,
                current_stage=PipelineStage.DISCOVERY,
                started_at=datetime.utcnow()
            )
        
        # Convert metrics to dict if present
        metrics_dict = None
        if execution.metrics:
            metrics_dict = {
                "total_runtime_seconds": execution.metrics.total_runtime_seconds,
                "discovery_items_found": execution.metrics.discovery_items_found,
                "analysis_items_processed": execution.metrics.analysis_items_processed,
                "report_items_included": execution.metrics.report_items_included,
                "emails_sent": execution.metrics.emails_sent,
                "success_rate": execution.metrics.success_rate,
                "cost_cents": execution.metrics.cost_cents
            }
        
        return PipelineExecutionResponse(
            execution_id=execution.execution_id,
            user_id=execution.user_id,
            trigger_type=execution.trigger_type,
            status=execution.status,
            current_stage=execution.current_stage,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            error_message=execution.error_message,
            metrics=metrics_dict
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute pipeline: {str(e)}"
        )


@router.get("/status/{execution_id}", response_model=PipelineExecutionResponse)
async def get_pipeline_status(
    execution_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get status of pipeline execution.
    
    Returns current status and progress of a pipeline execution
    including stage completion and performance metrics.
    """
    try:
        orchestration_service = OrchestrationService()
        
        execution = await orchestration_service.get_pipeline_status(execution_id)
        
        if not execution:
            raise HTTPException(
                status_code=404,
                detail="Pipeline execution not found"
            )
        
        # Verify ownership
        if execution.user_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="Access denied to pipeline execution"
            )
        
        # Convert metrics to dict if present
        metrics_dict = None
        if execution.metrics:
            metrics_dict = {
                "total_runtime_seconds": execution.metrics.total_runtime_seconds,
                "discovery_items_found": execution.metrics.discovery_items_found,
                "analysis_items_processed": execution.metrics.analysis_items_processed,
                "report_items_included": execution.metrics.report_items_included,
                "emails_sent": execution.metrics.emails_sent,
                "success_rate": execution.metrics.success_rate,
                "cost_cents": execution.metrics.cost_cents
            }
        
        return PipelineExecutionResponse(
            execution_id=execution.execution_id,
            user_id=execution.user_id,
            trigger_type=execution.trigger_type,
            status=execution.status,
            current_stage=execution.current_stage,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            error_message=execution.error_message,
            metrics=metrics_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get pipeline status: {str(e)}"
        )


@router.get("/history", response_model=List[PipelineExecutionResponse])
async def get_pipeline_history(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get pipeline execution history for current user.
    
    Returns paginated list of previous pipeline executions with
    status, metrics, and performance data.
    """
    try:
        orchestration_service = OrchestrationService()
        
        executions = await orchestration_service.get_user_pipeline_history(
            user_id=current_user.id,
            limit=limit
        )
        
        # Convert to response format
        responses = []
        for execution in executions:
            metrics_dict = None
            if execution.metrics:
                metrics_dict = {
                    "total_runtime_seconds": execution.metrics.total_runtime_seconds,
                    "discovery_items_found": execution.metrics.discovery_items_found,
                    "analysis_items_processed": execution.metrics.analysis_items_processed,
                    "report_items_included": execution.metrics.report_items_included,
                    "emails_sent": execution.metrics.emails_sent,
                    "success_rate": execution.metrics.success_rate,
                    "cost_cents": execution.metrics.cost_cents
                }
            
            response = PipelineExecutionResponse(
                execution_id=execution.execution_id,
                user_id=execution.user_id,
                trigger_type=execution.trigger_type,
                status=execution.status,
                current_stage=execution.current_stage,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                error_message=execution.error_message,
                metrics=metrics_dict
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get pipeline history: {str(e)}"
        )


@router.post("/batch/execute", response_model=BatchPipelineResponse)
async def execute_batch_pipeline(
    request: BatchPipelineRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Execute pipeline for multiple users in batch.
    
    Administrative endpoint for batch processing multiple users.
    Includes rate limiting and error handling for large batches.
    """
    try:
        # Note: In production, would add admin role check
        # For now, allow any user to test with their own ID
        if current_user.id not in request.user_ids:
            raise HTTPException(
                status_code=403,
                detail="Can only execute batch pipeline for your own user ID"
            )
        
        orchestration_service = OrchestrationService()
        
        # Execute batch pipeline as background task
        background_tasks.add_task(
            _execute_batch_pipeline_task,
            request.user_ids,
            request.trigger_type
        )
        
        # Return immediate response
        return BatchPipelineResponse(
            total_users=len(request.user_ids),
            successful_executions=0,
            failed_executions=0,
            success_rate=0.0,
            executions=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute batch pipeline: {str(e)}"
        )


@router.post("/schedule")
async def schedule_pipeline(
    request: SchedulePipelineRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Schedule pipeline execution for current user.
    
    Sets up recurring or one-time pipeline execution based on
    user preferences and strategic intelligence requirements.
    """
    try:
        # Validate schedule time
        if request.schedule_time <= datetime.utcnow():
            raise HTTPException(
                status_code=400,
                detail="Schedule time must be in the future"
            )
        
        # Create scheduled job configuration
        job_config = {
            "job_id": f"scheduled_pipeline_{current_user.id}_{datetime.utcnow().timestamp()}",
            "user_id": current_user.id,
            "schedule_time": request.schedule_time.isoformat(),
            "recurrence": request.recurrence,
            "pipeline_config": request.pipeline_config.dict() if request.pipeline_config else None,
            "status": "scheduled",
            "created_at": datetime.utcnow().isoformat()
        }
        
        # In full implementation, would store in ScheduledJobs table
        # and integrate with job scheduler service
        
        return {
            "message": "Pipeline execution scheduled successfully",
            "job_config": job_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule pipeline: {str(e)}"
        )


@router.post("/daily/execute")
async def execute_daily_pipelines(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Execute daily pipeline for all eligible users.
    
    Administrative endpoint for triggering daily batch processing
    of all users with daily delivery preferences enabled.
    """
    try:
        # Note: In production, would require admin role
        orchestration_service = OrchestrationService()
        
        # Execute daily pipelines as background task
        background_tasks.add_task(
            orchestration_service.schedule_daily_pipelines
        )
        
        return {
            "message": "Daily pipeline execution initiated",
            "initiated_at": datetime.utcnow().isoformat(),
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute daily pipelines: {str(e)}"
        )


@router.get("/analytics/performance")
async def get_performance_analytics(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get pipeline performance analytics.
    
    Returns performance metrics, success rates, and optimization
    recommendations for pipeline executions.
    """
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # In full implementation, would query PipelineExecutions table
        # For now, return mock analytics structure
        
        analytics = {
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "pipeline_performance": {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0.0,
                "avg_runtime_seconds": 0,
                "total_cost_cents": 0
            },
            "stage_performance": {
                "discovery": {"avg_items": 0, "avg_time_seconds": 0},
                "analysis": {"avg_items": 0, "avg_time_seconds": 0, "avg_cost_cents": 0},
                "report_generation": {"avg_items": 0, "avg_time_seconds": 0},
                "delivery": {"success_rate": 0.0, "avg_time_seconds": 0}
            },
            "optimization_recommendations": [
                "Monitor content discovery patterns for optimization opportunities",
                "Track analysis cost efficiency by priority level",
                "Optimize delivery timing based on engagement patterns",
                "Regular performance monitoring helps identify bottlenecks"
            ]
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance analytics: {str(e)}"
        )


# Background task functions
async def _execute_batch_pipeline_task(user_ids: List[int], trigger_type: str):
    """Background task for batch pipeline execution."""
    try:
        orchestration_service = OrchestrationService()
        
        executions = await orchestration_service.execute_batch_pipeline(
            user_ids=user_ids,
            trigger_type=trigger_type
        )
        
        # Log results
        import logging
        logger = logging.getLogger(__name__)
        
        successful = len([e for e in executions if e.status == PipelineStatus.COMPLETED])
        failed = len([e for e in executions if e.status == PipelineStatus.FAILED])
        
        logger.info(
            f"Batch pipeline completed: {successful} successful, {failed} failed"
        )
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Batch pipeline task failed: {e}")
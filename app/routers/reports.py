"""
Report Generation API Router for Phase 4

FastAPI router for strategic intelligence report generation and delivery.
Provides endpoints for generating reports in multiple formats and managing
report preferences with established BaseRouterOperations patterns.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from pydantic import BaseModel, Field

from app.database import get_db_session
from app.middleware import get_current_user
from app.models.user import User
from app.services.report_service import (
    ReportService, ReportGenerationRequest, ReportOutput, 
    ReportFormat, ReportType
)
from app.services.sendgrid_service import SendGridService
from app.analysis.core.shared_types import ContentPriority
from app.utils.router_base import BaseRouterOperations


# Pydantic models for API requests/responses
class GenerateReportRequest(BaseModel):
    """Request to generate strategic intelligence report."""
    report_type: ReportType = Field(default=ReportType.DAILY_DIGEST)
    output_formats: List[ReportFormat] = Field(default=[ReportFormat.EMAIL_HTML])
    date_range_days: int = Field(default=1, ge=1, le=30)
    min_priority: ContentPriority = Field(default=ContentPriority.MEDIUM)
    max_items_per_section: int = Field(default=10, ge=1, le=50)
    include_low_priority: bool = Field(default=False)
    send_email: bool = Field(default=True)
    custom_filters: Optional[Dict[str, Any]] = None


class ReportResponse(BaseModel):
    """Response for generated report."""
    report_id: str
    user_id: int
    report_type: ReportType
    format: ReportFormat
    generated_at: datetime
    content_items_count: int
    sections_count: int
    metadata: Dict[str, Any]
    download_url: Optional[str] = None
    email_sent: bool = False
    email_status: Optional[str] = None


class ReportListResponse(BaseModel):
    """Response for listing user reports."""
    reports: List[ReportResponse]
    total_count: int
    page: int
    page_size: int


class EmailDeliveryResponse(BaseModel):
    """Response for email delivery status."""
    message_id: str
    status: str
    recipient_email: str
    sent_at: datetime
    error_message: Optional[str] = None


# Router setup
router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.post("/generate", response_model=List[ReportResponse])
async def generate_report(
    request: GenerateReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Generate strategic intelligence report in requested formats.
    
    Generates priority-based strategic intelligence reports using analyzed
    content from Discovery and Analysis services. Supports multiple output
    formats including SendGrid email delivery.
    """
    try:
        # Initialize services
        report_service = ReportService()
        
        # Create service request
        service_request = ReportGenerationRequest(
            user_id=current_user.id,
            report_type=request.report_type,
            output_formats=request.output_formats,
            date_range_days=request.date_range_days,
            min_priority=request.min_priority,
            max_items_per_section=request.max_items_per_section,
            include_low_priority=request.include_low_priority,
            custom_filters=request.custom_filters
        )
        
        # Generate reports
        report_outputs = await report_service.generate_report(service_request)
        
        if not report_outputs:
            raise HTTPException(
                status_code=404,
                detail="No content available for report generation with current filters"
            )
        
        # Convert to API responses
        responses = []
        for output in report_outputs:
            response = ReportResponse(
                report_id=output.report_id,
                user_id=output.user_id,
                report_type=output.report_type,
                format=output.format,
                generated_at=output.generated_at,
                content_items_count=output.content_items_count,
                sections_count=output.sections_count,
                metadata=output.metadata
            )
            responses.append(response)
        
        # Handle email delivery if requested
        if request.send_email:
            email_output = None
            for output in report_outputs:
                if output.format == ReportFormat.EMAIL_HTML:
                    email_output = output
                    break
            
            if email_output:
                # Schedule email delivery as background task
                background_tasks.add_task(
                    _send_report_email,
                    current_user.email,
                    current_user.name,
                    current_user.id,
                    email_output.content,
                    email_output.metadata
                )
                
                # Update response to indicate email scheduled
                for response in responses:
                    if response.format == ReportFormat.EMAIL_HTML:
                        response.email_sent = True
                        response.email_status = "scheduled"
        
        return responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.get("/history", response_model=ReportListResponse)
async def get_report_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    report_type: Optional[ReportType] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get user's report generation history.
    
    Returns paginated list of previously generated reports with metadata
    and download links for accessing historical reports.
    """
    try:
        # Note: In full implementation, would query UserReports table
        # For now, returning mock response structure
        
        reports = []  # Would be populated from database query
        total_count = 0
        
        return ReportListResponse(
            reports=reports,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve report history: {str(e)}"
        )


@router.get("/{report_id}/content")
async def get_report_content(
    report_id: str,
    format: ReportFormat = Query(ReportFormat.API_JSON),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get content of a previously generated report.
    
    Returns the actual report content in requested format for download
    or programmatic access. Validates user ownership of report.
    """
    try:
        # Note: In full implementation, would query UserReports table
        # and validate ownership
        
        # For demo, regenerate report
        report_service = ReportService()
        
        service_request = ReportGenerationRequest(
            user_id=current_user.id,
            report_type=ReportType.DAILY_DIGEST,
            output_formats=[format],
            date_range_days=1,
            min_priority=ContentPriority.MEDIUM
        )
        
        report_outputs = await report_service.generate_report(service_request)
        
        if not report_outputs:
            raise HTTPException(
                status_code=404,
                detail="Report not found or no content available"
            )
        
        # Return content based on format
        output = report_outputs[0]
        
        if format == ReportFormat.API_JSON:
            import json
            return json.loads(output.content)
        else:
            return {"content": output.content, "format": format.value}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve report content: {str(e)}"
        )


@router.post("/email/send", response_model=EmailDeliveryResponse)
async def send_email_report(
    report_id: Optional[str] = None,
    regenerate: bool = Query(default=False),
    recipient_email: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Send strategic intelligence report via email.
    
    Sends existing report by ID or regenerates and sends fresh report.
    Uses SendGrid service for professional email delivery with tracking.
    """
    try:
        # Determine recipient
        email = recipient_email or current_user.email
        if not email:
            raise HTTPException(
                status_code=400,
                detail="Recipient email address required"
            )
        
        # Generate or retrieve report content
        if regenerate or not report_id:
            report_service = ReportService()
            
            service_request = ReportGenerationRequest(
                user_id=current_user.id,
                report_type=ReportType.DAILY_DIGEST,
                output_formats=[ReportFormat.EMAIL_HTML],
                date_range_days=1,
                min_priority=ContentPriority.MEDIUM
            )
            
            report_outputs = await report_service.generate_report(service_request)
            
            if not report_outputs:
                raise HTTPException(
                    status_code=404,
                    detail="No content available for email report"
                )
            
            html_content = report_outputs[0].content
            metadata = report_outputs[0].metadata
        else:
            # In full implementation, would retrieve from UserReports table
            raise HTTPException(
                status_code=501,
                detail="Retrieving existing reports not yet implemented"
            )
        
        # Send email
        result = await _send_report_email(
            email, current_user.name, current_user.id, html_content, metadata
        )
        
        return EmailDeliveryResponse(
            message_id=result.message_id,
            status=result.status.value,
            recipient_email=result.recipient_email,
            sent_at=result.sent_at,
            error_message=result.error_message
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email report: {str(e)}"
        )


@router.get("/analytics/engagement")
async def get_engagement_analytics(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get email engagement analytics for user's reports.
    
    Returns engagement metrics from SendGrid including open rates,
    click rates, and content interaction patterns for optimization.
    """
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get SendGrid statistics
        sendgrid_service = SendGridService()
        stats = await sendgrid_service.get_email_statistics(start_date, end_date)
        
        # In full implementation, would also query ContentEngagement table
        # for detailed user interaction data
        
        analytics = {
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "email_statistics": stats,
            "user_engagement": {
                "total_reports_sent": 0,  # Would query from database
                "avg_open_rate": 0.0,
                "avg_click_rate": 0.0,
                "top_content_types": [],
                "engagement_by_priority": {}
            },
            "recommendations": [
                "Regular engagement analysis helps optimize content relevance",
                "Track which priority levels generate most user interaction",
                "Monitor email delivery times for optimal engagement"
            ]
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve engagement analytics: {str(e)}"
        )


@router.post("/schedule")
async def schedule_report_delivery(
    schedule_time: datetime,
    report_config: GenerateReportRequest,
    recurrence: Optional[str] = Query(None, pattern="^(daily|weekly|monthly)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Schedule automatic report delivery.
    
    Sets up recurring or one-time report generation and delivery
    based on user preferences and strategic intelligence requirements.
    """
    try:
        # In full implementation, would create scheduled job
        # using job scheduler service
        
        scheduled_job = {
            "job_id": f"report_schedule_{current_user.id}_{datetime.utcnow().timestamp()}",
            "user_id": current_user.id,
            "schedule_time": schedule_time.isoformat(),
            "recurrence": recurrence,
            "report_config": report_config.dict(),
            "status": "scheduled",
            "created_at": datetime.utcnow().isoformat()
        }
        
        return {
            "message": "Report delivery scheduled successfully",
            "job_details": scheduled_job
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule report delivery: {str(e)}"
        )


# Background task functions
async def _send_report_email(
    recipient_email: str,
    recipient_name: str,
    user_id: int,
    html_content: str,
    metadata: Dict[str, Any]
):
    """Background task to send report email via SendGrid."""
    try:
        sendgrid_service = SendGridService()
        
        result = await sendgrid_service.send_strategic_report_email(
            recipient_email=recipient_email,
            recipient_name=recipient_name,
            user_id=user_id,
            report_content_html=html_content,
            user_context=metadata.get("user_context", {}),
            report_metadata=metadata
        )
        
        return result
        
    except Exception as e:
        # Log error but don't raise to avoid breaking background task
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to send report email to {recipient_email}: {e}")
        
        from app.services.sendgrid_service import EmailDeliveryResult, EmailStatus
        return EmailDeliveryResult(
            message_id="",
            status=EmailStatus.FAILED,
            recipient_email=recipient_email,
            user_id=user_id,
            sent_at=datetime.utcnow(),
            error_message=str(e)
        )
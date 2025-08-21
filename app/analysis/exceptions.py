"""
Analysis Service specific exceptions and error handling.

Provides comprehensive error taxonomy for analysis operations
with proper error codes and recovery strategies.
"""

from typing import Optional, Dict, Any
from enum import Enum


class AnalysisErrorCode(Enum):
    """Error codes for analysis operations."""
    
    # Configuration errors
    INVALID_USER_CONTEXT = "ANALYSIS_001"
    MISSING_STRATEGIC_PROFILE = "ANALYSIS_002"
    INVALID_ANALYSIS_STAGES = "ANALYSIS_003"
    INSUFFICIENT_USER_CONFIG = "ANALYSIS_004"
    
    # Content errors
    CONTENT_NOT_FOUND = "ANALYSIS_101"
    CONTENT_ALREADY_ANALYZED = "ANALYSIS_102"
    CONTENT_TOO_SHORT = "ANALYSIS_103"
    CONTENT_INVALID_FORMAT = "ANALYSIS_104"
    NO_PENDING_CONTENT = "ANALYSIS_105"
    
    # AI Service errors
    AI_PROVIDER_UNAVAILABLE = "ANALYSIS_201"
    AI_API_QUOTA_EXCEEDED = "ANALYSIS_202"
    AI_REQUEST_TIMEOUT = "ANALYSIS_203"
    AI_INVALID_RESPONSE = "ANALYSIS_204"
    AI_MODEL_OVERLOADED = "ANALYSIS_205"
    
    # Processing errors
    BATCH_PROCESSING_FAILED = "ANALYSIS_301"
    STAGE_PROCESSING_FAILED = "ANALYSIS_302"
    FILTER_PROCESSING_FAILED = "ANALYSIS_303"
    INSIGHT_EXTRACTION_FAILED = "ANALYSIS_304"
    DATABASE_SAVE_FAILED = "ANALYSIS_305"
    
    # Cost and limits
    COST_LIMIT_EXCEEDED = "ANALYSIS_401"
    RATE_LIMIT_EXCEEDED = "ANALYSIS_402"
    QUOTA_EXHAUSTED = "ANALYSIS_403"
    BUDGET_INSUFFICIENT = "ANALYSIS_404"
    
    # System errors
    CACHE_ERROR = "ANALYSIS_501"
    CONCURRENCY_ERROR = "ANALYSIS_502"
    RESOURCE_UNAVAILABLE = "ANALYSIS_503"
    CONFIGURATION_ERROR = "ANALYSIS_504"


class AnalysisException(Exception):
    """Base exception for analysis operations."""
    
    def __init__(
        self,
        message: str,
        error_code: AnalysisErrorCode,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": {
                "message": self.message,
                "code": self.error_code.value,
                "details": self.details,
                "recoverable": self.recoverable,
                "retry_after": self.retry_after
            }
        }


class ConfigurationError(AnalysisException):
    """Error in analysis configuration or user context."""
    
    def __init__(self, message: str, missing_fields: Optional[List[str]] = None):
        super().__init__(
            message=message,
            error_code=AnalysisErrorCode.INVALID_USER_CONTEXT,
            details={"missing_fields": missing_fields or []},
            recoverable=False
        )


class ContentError(AnalysisException):
    """Error related to content processing."""
    
    def __init__(
        self,
        message: str,
        content_id: Optional[int] = None,
        error_code: AnalysisErrorCode = AnalysisErrorCode.CONTENT_NOT_FOUND
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details={"content_id": content_id},
            recoverable=error_code != AnalysisErrorCode.CONTENT_NOT_FOUND
        )


class AIServiceError(AnalysisException):
    """Error from AI service operations."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        error_code: AnalysisErrorCode = AnalysisErrorCode.AI_PROVIDER_UNAVAILABLE,
        retry_after: Optional[int] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details={"provider": provider, "model": model},
            recoverable=True,
            retry_after=retry_after
        )


class ProcessingError(AnalysisException):
    """Error during analysis processing."""
    
    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        batch_id: Optional[str] = None,
        error_code: AnalysisErrorCode = AnalysisErrorCode.BATCH_PROCESSING_FAILED
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details={"stage": stage, "batch_id": batch_id},
            recoverable=True
        )


class CostLimitError(AnalysisException):
    """Error when cost limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        current_cost: Optional[float] = None,
        limit: Optional[float] = None
    ):
        super().__init__(
            message=message,
            error_code=AnalysisErrorCode.COST_LIMIT_EXCEEDED,
            details={"current_cost": current_cost, "limit": limit},
            recoverable=False
        )


class RateLimitError(AnalysisException):
    """Error when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: int = 60,
        limit_type: str = "requests"
    ):
        super().__init__(
            message=message,
            error_code=AnalysisErrorCode.RATE_LIMIT_EXCEEDED,
            details={"limit_type": limit_type},
            recoverable=True,
            retry_after=retry_after
        )


class ValidationError(AnalysisException):
    """Error in data validation."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=AnalysisErrorCode.CONTENT_INVALID_FORMAT,
            details={
                "field": field,
                "value": str(value) if value is not None else None,
                "expected": expected
            },
            recoverable=False
        )


def handle_ai_service_error(error: Exception) -> AnalysisException:
    """
    Convert AI service errors to analysis exceptions.
    
    Args:
        error: Original exception from AI service
        
    Returns:
        Appropriate AnalysisException
    """
    error_str = str(error).lower()
    
    if "rate limit" in error_str or "quota" in error_str:
        return AIServiceError(
            message=f"AI service rate limit exceeded: {error}",
            error_code=AnalysisErrorCode.AI_API_QUOTA_EXCEEDED,
            retry_after=300  # 5 minutes
        )
    elif "timeout" in error_str:
        return AIServiceError(
            message=f"AI service timeout: {error}",
            error_code=AnalysisErrorCode.AI_REQUEST_TIMEOUT,
            retry_after=60
        )
    elif "overloaded" in error_str or "busy" in error_str:
        return AIServiceError(
            message=f"AI service overloaded: {error}",
            error_code=AnalysisErrorCode.AI_MODEL_OVERLOADED,
            retry_after=120
        )
    elif "unauthorized" in error_str or "forbidden" in error_str:
        return AIServiceError(
            message=f"AI service authentication failed: {error}",
            error_code=AnalysisErrorCode.AI_PROVIDER_UNAVAILABLE,
            retry_after=None
        )
    else:
        return AIServiceError(
            message=f"AI service error: {error}",
            error_code=AnalysisErrorCode.AI_PROVIDER_UNAVAILABLE,
            retry_after=60
        )


def handle_database_error(error: Exception) -> AnalysisException:
    """
    Convert database errors to analysis exceptions.
    
    Args:
        error: Original database exception
        
    Returns:
        Appropriate AnalysisException
    """
    error_str = str(error).lower()
    
    if "connection" in error_str or "network" in error_str:
        return ProcessingError(
            message=f"Database connection error: {error}",
            error_code=AnalysisErrorCode.RESOURCE_UNAVAILABLE
        )
    elif "constraint" in error_str or "integrity" in error_str:
        return ProcessingError(
            message=f"Database integrity error: {error}",
            error_code=AnalysisErrorCode.DATABASE_SAVE_FAILED
        )
    elif "timeout" in error_str:
        return ProcessingError(
            message=f"Database timeout: {error}",
            error_code=AnalysisErrorCode.DATABASE_SAVE_FAILED
        )
    else:
        return ProcessingError(
            message=f"Database error: {error}",
            error_code=AnalysisErrorCode.DATABASE_SAVE_FAILED
        )


def validate_analysis_context(context) -> None:
    """
    Validate analysis context for completeness.
    
    Args:
        context: AnalysisContext object
        
    Raises:
        ValidationError: If context is invalid
    """
    missing_fields = []
    
    if not context.user_id:
        missing_fields.append("user_id")
    
    if not context.strategic_profile:
        missing_fields.append("strategic_profile")
    
    if not context.focus_areas:
        missing_fields.append("focus_areas")
    
    if missing_fields:
        raise ConfigurationError(
            f"Analysis context missing required fields: {', '.join(missing_fields)}",
            missing_fields=missing_fields
        )
    
    # Validate strategic profile content
    if context.strategic_profile:
        required_profile_fields = ["industry", "role", "strategic_goals"]
        missing_profile_fields = [
            field for field in required_profile_fields
            if not context.strategic_profile.get(field)
        ]
        
        if missing_profile_fields:
            raise ConfigurationError(
                f"Strategic profile missing required fields: {', '.join(missing_profile_fields)}",
                missing_fields=missing_profile_fields
            )


def validate_content_for_analysis(content: Dict[str, Any]) -> None:
    """
    Validate content for analysis requirements.
    
    Args:
        content: Content dictionary
        
    Raises:
        ValidationError: If content is invalid
    """
    if not content.get("id"):
        raise ValidationError(
            "Content missing required ID field",
            field="id"
        )
    
    content_text = content.get("content_text", "")
    title = content.get("title", "")
    
    if not content_text and not title:
        raise ContentError(
            "Content has no text or title for analysis",
            content_id=content.get("id"),
            error_code=AnalysisErrorCode.CONTENT_TOO_SHORT
        )
    
    total_text = f"{title} {content_text}".strip()
    if len(total_text) < 50:
        raise ContentError(
            f"Content too short for analysis (minimum 50 characters, got {len(total_text)})",
            content_id=content.get("id"),
            error_code=AnalysisErrorCode.CONTENT_TOO_SHORT
        )


def validate_analysis_stages(stages: List[str]) -> None:
    """
    Validate analysis stages are valid and in correct order.
    
    Args:
        stages: List of stage names
        
    Raises:
        ValidationError: If stages are invalid
    """
    from app.analysis.utils.common_types import AnalysisStage
    
    valid_stages = [stage.value for stage in AnalysisStage]
    invalid_stages = [stage for stage in stages if stage not in valid_stages]
    
    if invalid_stages:
        raise ValidationError(
            f"Invalid analysis stages: {', '.join(invalid_stages)}",
            field="stages",
            value=invalid_stages,
            expected=f"One of: {', '.join(valid_stages)}"
        )
    
    if not stages:
        raise ValidationError(
            "At least one analysis stage must be specified",
            field="stages",
            value=stages,
            expected="Non-empty list of stage names"
        )
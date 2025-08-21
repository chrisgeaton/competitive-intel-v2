"""
Discovery Service Pydantic schemas for competitive intelligence v2.

Request/response models for ML-driven content discovery, scoring,
and engagement tracking with comprehensive validation.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class SourceType(str, Enum):
    """Source types for content discovery."""
    WEB_SCRAPING = "web_scraping"
    RSS_FEEDS = "rss_feeds"
    NEWS_APIS = "news_apis"
    SOCIAL_APIS = "social_apis"
    SEARCH_APIS = "search_apis"
    CUSTOM_APIS = "custom_apis"
    DATABASE_FEEDS = "database_feeds"
    RESEARCH_APIS = "research_apis"


class ContentType(str, Enum):
    """Content types for categorization."""
    ARTICLE = "article"
    REPORT = "report"
    NEWS = "news"
    BLOG = "blog"
    SOCIAL = "social"
    RESEARCH = "research"
    WHITEPAPER = "whitepaper"
    PRESS_RELEASE = "press_release"


class EngagementType(str, Enum):
    """User engagement types for ML learning."""
    EMAIL_OPEN = "email_open"
    EMAIL_CLICK = "email_click"
    TIME_SPENT = "time_spent"
    BOOKMARK = "bookmark"
    SHARE = "share"
    FEEDBACK_POSITIVE = "feedback_positive"
    FEEDBACK_NEGATIVE = "feedback_negative"
    EMAIL_BOUNCE = "email_bounce"
    EMAIL_DROPPED = "email_dropped"
    EMAIL_SPAM = "email_spam"
    EMAIL_UNSUBSCRIBE = "email_unsubscribe"
    EMAIL_DELIVERED = "email_delivered"


class JobType(str, Enum):
    """Discovery job types."""
    SCHEDULED_DISCOVERY = "scheduled_discovery"
    MANUAL_DISCOVERY = "manual_discovery"
    ML_TRAINING = "ml_training"
    SOURCE_CHECK = "source_check"
    ENGAGEMENT_PROCESSING = "engagement_processing"


class JobStatus(str, Enum):
    """Discovery job status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Source Management Schemas

class DiscoveredSourceCreate(BaseModel):
    """Schema for creating a new discovery source."""
    source_type: SourceType
    source_url: str = Field(..., max_length=2000)
    source_name: Optional[str] = Field(None, max_length=200)
    source_description: Optional[str] = None
    check_frequency_minutes: int = Field(60, ge=5, le=10080)  # 5 minutes to 1 week
    created_by_user_id: Optional[int] = None
    
    @validator('source_url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://', 'feed://', 'rss://')):
            raise ValueError('URL must start with http://, https://, feed://, or rss://')
        return v


class DiscoveredSourceUpdate(BaseModel):
    """Schema for updating a discovery source."""
    source_name: Optional[str] = Field(None, max_length=200)
    source_description: Optional[str] = None
    is_active: Optional[bool] = None
    check_frequency_minutes: Optional[int] = Field(None, ge=5, le=10080)


class DiscoveredSourceResponse(BaseModel):
    """Schema for discovery source responses."""
    id: int
    source_type: SourceType
    source_url: str
    source_name: Optional[str]
    source_description: Optional[str]
    is_active: bool
    last_checked: Optional[datetime]
    last_successful_check: Optional[datetime]
    check_frequency_minutes: int
    success_rate: Decimal
    quality_score: Decimal
    relevance_score: Decimal
    credibility_score: Decimal
    user_engagement_score: Decimal
    total_content_found: int
    total_content_delivered: int
    total_user_engagements: int
    ml_confidence_level: Decimal
    created_at: datetime
    updated_at: datetime
    created_by_user_id: Optional[int]
    
    class Config:
        from_attributes = True
        json_encoders = {
            Decimal: lambda v: float(v) if v is not None else None
        }


# Content Discovery Schemas

class DiscoveredContentCreate(BaseModel):
    """Schema for creating discovered content."""
    title: str = Field(..., max_length=500)
    content_url: str = Field(..., max_length=2000)
    content_text: Optional[str] = None
    content_summary: Optional[str] = None
    author: Optional[str] = Field(None, max_length=200)
    published_at: Optional[datetime] = None
    content_language: str = Field("en", max_length=10)
    content_type: ContentType = ContentType.ARTICLE
    source_id: int
    user_id: int
    predicted_categories: Optional[str] = None  # JSON string
    detected_entities: Optional[str] = None  # JSON string
    sentiment_score: Optional[Decimal] = Field(None, ge=-1.0, le=1.0)
    competitive_relevance: Optional[str] = Field(None, pattern="^(high|medium|low|unknown)$")
    
    @validator('content_url')
    def validate_content_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Content URL must start with http:// or https://')
        return v


class DiscoveredContentUpdate(BaseModel):
    """Schema for updating discovered content."""
    title: Optional[str] = Field(None, max_length=500)
    content_text: Optional[str] = None
    content_summary: Optional[str] = None
    author: Optional[str] = Field(None, max_length=200)
    published_at: Optional[datetime] = None
    content_type: Optional[ContentType] = None
    predicted_categories: Optional[str] = None
    detected_entities: Optional[str] = None
    sentiment_score: Optional[Decimal] = Field(None, ge=-1.0, le=1.0)
    competitive_relevance: Optional[str] = Field(None, pattern="^(high|medium|low|unknown)$")
    human_feedback_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)


class MLScoresSchema(BaseModel):
    """Schema for ML-generated content scores."""
    relevance_score: Decimal = Field(..., ge=0.0, le=1.0)
    credibility_score: Decimal = Field(..., ge=0.0, le=1.0)
    freshness_score: Decimal = Field(..., ge=0.0, le=1.0)
    engagement_prediction: Decimal = Field(..., ge=0.0, le=1.0)
    overall_score: Decimal = Field(..., ge=0.0, le=1.0)
    confidence_level: Decimal = Field(..., ge=0.0, le=1.0)
    model_version: str


class DiscoveredContentResponse(BaseModel):
    """Schema for discovered content responses."""
    id: int
    title: str
    content_url: str
    content_text: Optional[str]
    content_summary: Optional[str]
    content_hash: Optional[str]
    similarity_hash: Optional[str]
    author: Optional[str]
    published_at: Optional[datetime]
    discovered_at: datetime
    content_language: str
    content_type: ContentType
    source_id: int
    user_id: int
    relevance_score: Decimal
    credibility_score: Decimal
    freshness_score: Decimal
    engagement_prediction_score: Decimal
    overall_score: Decimal
    ml_model_version: str
    ml_confidence_level: Decimal
    human_feedback_score: Optional[Decimal]
    actual_engagement_score: Optional[Decimal]
    predicted_categories: Optional[str]
    detected_entities: Optional[str]
    sentiment_score: Optional[Decimal]
    competitive_relevance: Optional[str]
    is_delivered: bool
    delivered_at: Optional[datetime]
    delivery_method: Optional[str]
    delivery_status: str
    is_duplicate: bool
    duplicate_of_content_id: Optional[int]
    similarity_score: Optional[Decimal]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
        json_encoders = {
            Decimal: lambda v: float(v) if v is not None else None
        }


class ContentSimilaritySchema(BaseModel):
    """Schema for content similarity analysis."""
    content_id: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    duplicate_type: str = Field(..., pattern="^(exact|near_duplicate|similar)$")
    matching_features: List[str]


# Engagement Tracking Schemas

class ContentEngagementCreate(BaseModel):
    """Schema for creating content engagement records."""
    user_id: int
    content_id: Optional[int] = None
    engagement_type: EngagementType
    engagement_value: Decimal = Field(1.0, ge=0.0)
    engagement_context: Optional[str] = None
    sendgrid_event_id: Optional[str] = Field(None, max_length=100)
    sendgrid_message_id: Optional[str] = Field(None, max_length=100)
    email_subject: Optional[str] = Field(None, max_length=200)
    user_agent: Optional[str] = Field(None, max_length=500)
    ip_address: Optional[str] = Field(None, max_length=45)
    device_type: Optional[str] = Field(None, pattern="^(desktop|mobile|tablet|unknown)$")
    session_duration: Optional[int] = Field(None, ge=0)
    click_sequence: Optional[int] = Field(None, ge=1)
    time_to_click: Optional[int] = Field(None, ge=0)
    engagement_timestamp: Optional[datetime] = None


class SendGridEngagementData(BaseModel):
    """Schema for processing SendGrid webhook data."""
    event: str
    email: str
    timestamp: int
    sg_event_id: Optional[str] = None
    sg_message_id: Optional[str] = None
    subject: Optional[str] = None
    url: Optional[str] = None
    useragent: Optional[str] = None
    ip: Optional[str] = None
    category: Optional[List[str]] = None
    unique_args: Optional[Dict[str, Any]] = None
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v <= 0:
            raise ValueError('Timestamp must be positive')
        return v


class ContentEngagementResponse(BaseModel):
    """Schema for content engagement responses."""
    id: int
    user_id: int
    content_id: Optional[int]
    engagement_type: EngagementType
    engagement_value: Decimal
    engagement_context: Optional[str]
    sendgrid_event_id: Optional[str]
    sendgrid_message_id: Optional[str]
    email_subject: Optional[str]
    user_agent: Optional[str]
    ip_address: Optional[str]
    device_type: Optional[str]
    session_duration: Optional[int]
    click_sequence: Optional[int]
    time_to_click: Optional[int]
    user_strategic_profile_snapshot: Optional[str]
    focus_areas_matched: Optional[str]
    entities_matched: Optional[str]
    predicted_engagement: Optional[Decimal]
    prediction_accuracy: Optional[Decimal]
    feedback_processed: bool
    ml_weight: Decimal
    created_at: datetime
    engagement_timestamp: Optional[datetime]
    content_age_at_engagement: Optional[int]
    
    class Config:
        from_attributes = True
        json_encoders = {
            Decimal: lambda v: float(v) if v is not None else None
        }


# Discovery Job Schemas

class DiscoveryJobCreate(BaseModel):
    """Schema for creating discovery jobs."""
    user_id: Optional[int] = None
    job_type: JobType
    job_subtype: Optional[str] = Field(None, max_length=50)
    scheduled_at: Optional[datetime] = None
    job_parameters: Optional[str] = None  # JSON string
    source_filters: Optional[str] = None  # JSON string
    quality_threshold: Decimal = Field(0.7, ge=0.0, le=1.0)
    max_retries: int = Field(3, ge=0, le=10)


class DiscoveryJobUpdate(BaseModel):
    """Schema for updating discovery jobs."""
    status: Optional[JobStatus] = None
    progress_percentage: Optional[int] = Field(None, ge=0, le=100)
    sources_checked: Optional[int] = Field(None, ge=0)
    sources_successful: Optional[int] = Field(None, ge=0)
    sources_failed: Optional[int] = Field(None, ge=0)
    content_found: Optional[int] = Field(None, ge=0)
    content_processed: Optional[int] = Field(None, ge=0)
    content_delivered: Optional[int] = Field(None, ge=0)
    duplicates_detected: Optional[int] = Field(None, ge=0)
    ml_feedback_processed: Optional[int] = Field(None, ge=0)
    ml_model_updated: Optional[bool] = None
    ml_accuracy_improvement: Optional[Decimal] = Field(None, ge=-1.0, le=1.0)
    new_patterns_discovered: Optional[int] = Field(None, ge=0)
    avg_relevance_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    avg_engagement_prediction: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    processing_time_seconds: Optional[int] = Field(None, ge=0)
    api_calls_made: Optional[int] = Field(None, ge=0)
    error_message: Optional[str] = None
    error_details: Optional[str] = None


class DiscoveryJobResponse(BaseModel):
    """Schema for discovery job responses."""
    id: int
    user_id: Optional[int]
    job_type: JobType
    job_subtype: Optional[str]
    status: JobStatus
    progress_percentage: int
    sources_checked: int
    sources_successful: int
    sources_failed: int
    content_found: int
    content_processed: int
    content_delivered: int
    duplicates_detected: int
    ml_feedback_processed: int
    ml_model_updated: bool
    ml_accuracy_improvement: Optional[Decimal]
    new_patterns_discovered: int
    avg_relevance_score: Optional[Decimal]
    avg_engagement_prediction: Optional[Decimal]
    processing_time_seconds: Optional[int]
    api_calls_made: int
    scheduled_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    next_run_at: Optional[datetime]
    error_message: Optional[str]
    error_details: Optional[str]
    retry_count: int
    max_retries: int
    job_parameters: Optional[str]
    ml_model_version: Optional[str]
    source_filters: Optional[str]
    quality_threshold: Decimal
    created_at: datetime
    updated_at: datetime
    created_by: str
    
    class Config:
        from_attributes = True
        json_encoders = {
            Decimal: lambda v: float(v) if v is not None else None
        }


# ML Model Metrics Schemas

class MLModelMetricsCreate(BaseModel):
    """Schema for creating ML model metrics."""
    model_version: str = Field(..., max_length=20)
    model_type: str = Field(..., max_length=50)
    model_name: str = Field(..., max_length=100)
    training_data_size: int = Field(..., ge=1)
    training_duration_seconds: int = Field(..., ge=1)
    training_accuracy: Decimal = Field(..., ge=0.0, le=1.0)
    validation_accuracy: Decimal = Field(..., ge=0.0, le=1.0)
    cross_validation_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    precision_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    recall_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    roc_auc_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    mean_squared_error: Optional[Decimal] = Field(None, ge=0.0)
    feature_importance: Optional[str] = None  # JSON string
    model_parameters: Optional[str] = None  # JSON string
    training_features: Optional[str] = None  # JSON string
    created_by: str = Field("system", max_length=50)


class MLModelMetricsUpdate(BaseModel):
    """Schema for updating ML model metrics."""
    production_accuracy: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    user_satisfaction_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    engagement_prediction_accuracy: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    is_active: Optional[bool] = None
    deployed_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    rollback_reason: Optional[str] = Field(None, max_length=200)


class MLModelMetricsResponse(BaseModel):
    """Schema for ML model metrics responses."""
    id: int
    model_version: str
    model_type: str
    model_name: str
    training_data_size: int
    training_duration_seconds: int
    training_accuracy: Decimal
    validation_accuracy: Decimal
    cross_validation_score: Optional[Decimal]
    precision_score: Optional[Decimal]
    recall_score: Optional[Decimal]
    f1_score: Optional[Decimal]
    roc_auc_score: Optional[Decimal]
    mean_squared_error: Optional[Decimal]
    production_accuracy: Optional[Decimal]
    user_satisfaction_score: Optional[Decimal]
    engagement_prediction_accuracy: Optional[Decimal]
    is_active: bool
    deployed_at: Optional[datetime]
    deprecated_at: Optional[datetime]
    rollback_reason: Optional[str]
    feature_importance: Optional[str]
    model_parameters: Optional[str]
    training_features: Optional[str]
    created_at: datetime
    updated_at: datetime
    created_by: str
    
    class Config:
        from_attributes = True
        json_encoders = {
            Decimal: lambda v: float(v) if v is not None else None
        }


# Analytics and Reporting Schemas

class UserDiscoveryAnalytics(BaseModel):
    """Schema for user discovery analytics."""
    user_id: int
    total_content_discovered: int
    total_content_delivered: int
    avg_relevance_score: Decimal
    avg_engagement_score: Decimal
    top_categories: List[Dict[str, Any]]
    top_sources: List[Dict[str, Any]]
    engagement_trends: Dict[str, Any]
    ml_accuracy_score: Decimal
    last_activity: datetime
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v) if v is not None else None
        }


class DiscoveryFilterRequest(BaseModel):
    """Schema for filtering discovery requests."""
    source_types: Optional[List[SourceType]] = None
    content_types: Optional[List[ContentType]] = None
    min_relevance_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    min_credibility_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    min_freshness_score: Optional[Decimal] = Field(None, ge=0.0, le=1.0)
    published_after: Optional[datetime] = None
    published_before: Optional[datetime] = None
    categories: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    competitive_relevance: Optional[List[str]] = None
    is_delivered: Optional[bool] = None
    exclude_duplicates: bool = True
    
    @validator('competitive_relevance')
    def validate_competitive_relevance(cls, v):
        if v is not None:
            valid_values = {'high', 'medium', 'low', 'unknown'}
            for item in v:
                if item not in valid_values:
                    raise ValueError(f'Invalid competitive relevance: {item}')
        return v
"""
Discovery Service database models for competitive intelligence v2.

ML-enabled models for intelligent source discovery and content scoring
with user behavior learning and engagement optimization.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Index, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional

from app.database import Base


class DiscoveredSource(Base):
    """
    Sources for competitive intelligence discovery.
    
    Tracks source performance, quality metrics, and ML learning data
    for continuous improvement of source selection algorithms.
    """
    __tablename__ = "discovered_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(50), nullable=False, index=True)  # 'rss', 'web_scraping', 'news_api', 'social_api'
    source_url = Column(String(2000), nullable=False, unique=True)
    source_name = Column(String(200), nullable=True)
    source_description = Column(Text, nullable=True)
    
    # Source status and health metrics
    is_active = Column(Boolean, default=True, index=True)
    last_checked = Column(DateTime, nullable=True)
    last_successful_check = Column(DateTime, nullable=True)
    check_frequency_minutes = Column(Integer, default=60)  # How often to check this source
    
    # ML performance metrics
    success_rate = Column(Numeric(5, 4), default=0.0000)  # Success rate for content extraction
    quality_score = Column(Numeric(5, 4), default=0.5000)  # ML-computed overall quality score
    relevance_score = Column(Numeric(5, 4), default=0.5000)  # Average relevance of content from this source
    credibility_score = Column(Numeric(5, 4), default=0.5000)  # Source credibility assessment
    user_engagement_score = Column(Numeric(5, 4), default=0.0000)  # User engagement with content from this source
    
    # ML learning metrics
    total_content_found = Column(Integer, default=0)
    total_content_delivered = Column(Integer, default=0)
    total_user_engagements = Column(Integer, default=0)
    ml_confidence_level = Column(Numeric(5, 4), default=0.5000)  # ML model confidence in scores
    
    # Administrative fields
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    created_by_user_id = Column(Integer, nullable=True)  # User who added this source
    
    # Relationships
    discovered_content = relationship("DiscoveredContent", back_populates="source", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DiscoveredSource(id={self.id}, name='{self.source_name}', type='{self.source_type}', quality={self.quality_score})>"


class DiscoveredContent(Base):
    """
    Content discovered from competitive intelligence sources.
    
    ML-enabled content scoring with user engagement tracking
    for continuous improvement of relevance algorithms.
    """
    __tablename__ = "discovered_content"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Content identification
    title = Column(String(500), nullable=False, index=True)
    content_url = Column(String(2000), nullable=False, index=True)
    content_text = Column(Text, nullable=True)
    content_summary = Column(Text, nullable=True)  # AI-generated summary
    content_hash = Column(String(64), nullable=True, index=True)  # For deduplication
    similarity_hash = Column(String(64), nullable=True, index=True)  # For content similarity detection
    
    # Content metadata
    author = Column(String(200), nullable=True)
    published_at = Column(DateTime, nullable=True, index=True)
    discovered_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    content_language = Column(String(10), default='en')
    content_type = Column(String(50), default='article')  # 'article', 'report', 'news', 'blog', 'social'
    
    # Relationships
    source_id = Column(Integer, ForeignKey("discovered_sources.id"), nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)  # Target user for this content
    
    # ML scoring system
    relevance_score = Column(Numeric(5, 4), default=0.0000, index=True)  # Relevance to user's focus areas
    credibility_score = Column(Numeric(5, 4), default=0.0000)  # Source and content credibility
    freshness_score = Column(Numeric(5, 4), default=0.0000)  # Content recency and timeliness
    engagement_prediction_score = Column(Numeric(5, 4), default=0.0000)  # ML prediction of user engagement
    overall_score = Column(Numeric(5, 4), default=0.0000, index=True)  # Composite ML score
    
    # ML learning and feedback
    ml_model_version = Column(String(20), default='1.0')  # Version of ML model used for scoring
    ml_confidence_level = Column(Numeric(5, 4), default=0.5000)  # Model confidence in scores
    human_feedback_score = Column(Numeric(5, 4), nullable=True)  # Human validation of ML scores
    actual_engagement_score = Column(Numeric(5, 4), nullable=True)  # Actual user engagement for ML training
    
    # Content categorization (ML-generated)
    predicted_categories = Column(Text, nullable=True)  # JSON array of predicted categories
    detected_entities = Column(Text, nullable=True)  # JSON array of detected entities (companies, people, etc.)
    sentiment_score = Column(Numeric(5, 4), nullable=True)  # Content sentiment analysis
    competitive_relevance = Column(String(20), nullable=True)  # 'high', 'medium', 'low', 'unknown'
    
    # Delivery tracking
    is_delivered = Column(Boolean, default=False, index=True)
    delivered_at = Column(DateTime, nullable=True)
    delivery_method = Column(String(50), nullable=True)  # 'email', 'api', 'web_notification'
    delivery_status = Column(String(20), default='pending')  # 'pending', 'sent', 'failed', 'bounced'
    
    # Deduplication tracking
    is_duplicate = Column(Boolean, default=False, index=True)
    duplicate_of_content_id = Column(Integer, ForeignKey("discovered_content.id"), nullable=True)
    similarity_score = Column(Numeric(5, 4), nullable=True)  # Similarity to other content
    
    # Administrative fields
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    source = relationship("DiscoveredSource", back_populates="discovered_content")
    analysis_results = relationship("AnalysisResult", back_populates="content")
    strategic_insights = relationship("StrategicInsight", back_populates="content", cascade="all, delete-orphan")
    engagements = relationship("ContentEngagement", back_populates="content", cascade="all, delete-orphan")
    duplicate_parent = relationship("DiscoveredContent", remote_side=[id], backref="duplicates")
    
    def __repr__(self):
        return f"<DiscoveredContent(id={self.id}, title='{self.title[:50]}...', score={self.overall_score})>"


class ContentEngagement(Base):
    """
    User engagement tracking for ML learning and content optimization.
    
    Tracks SendGrid email engagement, user behavior, and interaction patterns
    for continuous improvement of ML relevance algorithms.
    """
    __tablename__ = "content_engagement"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Core engagement tracking
    user_id = Column(Integer, nullable=False, index=True)
    content_id = Column(Integer, ForeignKey("discovered_content.id"), nullable=False, index=True)
    
    # Engagement types and metrics
    engagement_type = Column(String(50), nullable=False, index=True)  # 'email_open', 'email_click', 'time_spent', 'bookmark', 'share', 'feedback'
    engagement_value = Column(Numeric(10, 4), default=1.0000)  # Numeric value (time in seconds, rating, etc.)
    engagement_context = Column(Text, nullable=True)  # Additional context (device, location, etc.)
    
    # SendGrid specific engagement data
    sendgrid_event_id = Column(String(100), nullable=True, unique=True)  # SendGrid event ID for deduplication
    sendgrid_message_id = Column(String(100), nullable=True)  # SendGrid message ID
    email_subject = Column(String(200), nullable=True)  # Email subject for correlation
    user_agent = Column(String(500), nullable=True)  # User agent from email click
    ip_address = Column(String(45), nullable=True)  # IP address for geographic analysis
    
    # ML training data
    session_duration = Column(Integer, nullable=True)  # Session duration in seconds
    click_sequence = Column(Integer, nullable=True)  # Order of clicks in email
    time_to_click = Column(Integer, nullable=True)  # Time from email open to click (seconds)
    device_type = Column(String(20), nullable=True)  # 'desktop', 'mobile', 'tablet'
    
    # Behavioral correlation data
    user_strategic_profile_snapshot = Column(Text, nullable=True)  # JSON snapshot of user profile at time of engagement
    focus_areas_matched = Column(Text, nullable=True)  # JSON array of matched focus areas
    entities_matched = Column(Text, nullable=True)  # JSON array of matched tracked entities
    
    # ML feedback and learning
    predicted_engagement = Column(Numeric(5, 4), nullable=True)  # ML prediction before actual engagement
    prediction_accuracy = Column(Numeric(5, 4), nullable=True)  # How accurate the prediction was
    feedback_processed = Column(Boolean, default=False, index=True)  # Whether this engagement has been processed for ML training
    ml_weight = Column(Numeric(5, 4), default=1.0000)  # Weight for ML training (high for reliable engagements)
    
    # Temporal and contextual data
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    engagement_timestamp = Column(DateTime, nullable=True)  # When the actual engagement occurred
    content_age_at_engagement = Column(Integer, nullable=True)  # Age of content when engaged (hours)
    
    # Relationships
    content = relationship("DiscoveredContent", back_populates="engagements")
    
    def __repr__(self):
        return f"<ContentEngagement(id={self.id}, type='{self.engagement_type}', value={self.engagement_value})>"


class DiscoveryJob(Base):
    """
    Discovery job tracking and ML learning coordination.
    
    Tracks discovery operations, ML model training runs, and performance metrics
    for continuous system optimization.
    """
    __tablename__ = "discovery_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Job identification and ownership
    user_id = Column(Integer, nullable=True, index=True)  # Null for system-wide jobs
    job_type = Column(String(50), nullable=False, index=True)  # 'scheduled_discovery', 'manual_discovery', 'ml_training', 'source_check'
    job_subtype = Column(String(50), nullable=True)  # 'full_refresh', 'incremental', 'priority_sources'
    
    # Job status and progress
    status = Column(String(20), default='pending', index=True)  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    progress_percentage = Column(Integer, default=0)  # 0-100
    
    # Discovery metrics
    sources_checked = Column(Integer, default=0)
    sources_successful = Column(Integer, default=0)
    sources_failed = Column(Integer, default=0)
    content_found = Column(Integer, default=0)
    content_processed = Column(Integer, default=0)
    content_delivered = Column(Integer, default=0)
    duplicates_detected = Column(Integer, default=0)
    
    # ML and learning metrics
    ml_feedback_processed = Column(Integer, default=0)  # Number of engagement feedbacks processed
    ml_model_updated = Column(Boolean, default=False)  # Whether ML model was updated in this job
    ml_accuracy_improvement = Column(Numeric(5, 4), nullable=True)  # Improvement in ML accuracy
    new_patterns_discovered = Column(Integer, default=0)  # New user behavior patterns discovered
    
    # Performance metrics
    avg_relevance_score = Column(Numeric(5, 4), nullable=True)  # Average relevance score of discovered content
    avg_engagement_prediction = Column(Numeric(5, 4), nullable=True)  # Average predicted engagement
    processing_time_seconds = Column(Integer, nullable=True)  # Total processing time
    api_calls_made = Column(Integer, default=0)  # Number of external API calls
    
    # Job timing
    scheduled_at = Column(DateTime, nullable=True)  # When job was scheduled
    started_at = Column(DateTime, nullable=True, index=True)  # When job actually started
    completed_at = Column(DateTime, nullable=True, index=True)  # When job completed
    next_run_at = Column(DateTime, nullable=True)  # When next similar job should run
    
    # Error handling and debugging
    error_message = Column(Text, nullable=True)
    error_details = Column(Text, nullable=True)  # JSON with detailed error information
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Job configuration and parameters
    job_parameters = Column(Text, nullable=True)  # JSON with job-specific parameters
    ml_model_version = Column(String(20), nullable=True)  # ML model version used
    source_filters = Column(Text, nullable=True)  # JSON with source filtering criteria
    quality_threshold = Column(Numeric(5, 4), default=0.7000)  # Minimum quality threshold for content
    
    # Administrative fields
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(50), default='system')  # 'system', 'user', 'scheduler'
    
    def __repr__(self):
        return f"<DiscoveryJob(id={self.id}, type='{self.job_type}', status='{self.status}', content_found={self.content_found})>"


class MLModelMetrics(Base):
    """
    ML model performance tracking and versioning.
    
    Tracks ML model performance, training metrics, and version history
    for continuous model improvement and A/B testing.
    """
    __tablename__ = "ml_model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model identification
    model_version = Column(String(20), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # 'relevance_scorer', 'engagement_predictor', 'content_classifier'
    model_name = Column(String(100), nullable=False)
    
    # Training metrics
    training_data_size = Column(Integer, nullable=False)
    training_duration_seconds = Column(Integer, nullable=False)
    training_accuracy = Column(Numeric(5, 4), nullable=False)
    validation_accuracy = Column(Numeric(5, 4), nullable=False)
    cross_validation_score = Column(Numeric(5, 4), nullable=True)
    
    # Performance metrics
    precision_score = Column(Numeric(5, 4), nullable=True)
    recall_score = Column(Numeric(5, 4), nullable=True)
    f1_score = Column(Numeric(5, 4), nullable=True)
    roc_auc_score = Column(Numeric(5, 4), nullable=True)
    mean_squared_error = Column(Numeric(10, 6), nullable=True)
    
    # Production performance
    production_accuracy = Column(Numeric(5, 4), nullable=True)  # Actual accuracy in production
    user_satisfaction_score = Column(Numeric(5, 4), nullable=True)  # User feedback on model performance
    engagement_prediction_accuracy = Column(Numeric(5, 4), nullable=True)  # How well model predicts engagement
    
    # Model deployment
    is_active = Column(Boolean, default=False, index=True)
    deployed_at = Column(DateTime, nullable=True)
    deprecated_at = Column(DateTime, nullable=True)
    rollback_reason = Column(String(200), nullable=True)
    
    # Feature importance and explainability
    feature_importance = Column(Text, nullable=True)  # JSON with feature importance scores
    model_parameters = Column(Text, nullable=True)  # JSON with model hyperparameters
    training_features = Column(Text, nullable=True)  # JSON array of features used in training
    
    # Administrative fields
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(50), default='system')
    
    def __repr__(self):
        return f"<MLModelMetrics(id={self.id}, version='{self.model_version}', type='{self.model_type}', accuracy={self.training_accuracy})>"


# Performance indexes for optimized queries
Index("idx_discovered_content_user_score", DiscoveredContent.user_id, DiscoveredContent.overall_score.desc())
Index("idx_discovered_content_published", DiscoveredContent.published_at.desc())
Index("idx_discovered_content_delivered", DiscoveredContent.is_delivered, DiscoveredContent.delivered_at)
Index("idx_discovered_content_hash", DiscoveredContent.content_hash)
Index("idx_discovered_content_similarity", DiscoveredContent.similarity_hash)

Index("idx_discovered_sources_active_type", DiscoveredSource.is_active, DiscoveredSource.source_type)
Index("idx_discovered_sources_quality", DiscoveredSource.quality_score.desc())
Index("idx_discovered_sources_last_checked", DiscoveredSource.last_checked)

Index("idx_content_engagement_user_content", ContentEngagement.user_id, ContentEngagement.content_id)
Index("idx_content_engagement_type_time", ContentEngagement.engagement_type, ContentEngagement.created_at.desc())
Index("idx_content_engagement_feedback", ContentEngagement.feedback_processed, ContentEngagement.created_at)

Index("idx_discovery_jobs_user_status", DiscoveryJob.user_id, DiscoveryJob.status, DiscoveryJob.created_at.desc())
Index("idx_discovery_jobs_type_status", DiscoveryJob.job_type, DiscoveryJob.status)
Index("idx_discovery_jobs_next_run", DiscoveryJob.next_run_at)

Index("idx_ml_model_metrics_active", MLModelMetrics.is_active, MLModelMetrics.model_type)
Index("idx_ml_model_metrics_version", MLModelMetrics.model_version, MLModelMetrics.model_type)
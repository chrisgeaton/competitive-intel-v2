"""
Analysis Service database models for competitive intelligence v2.

Models for storing AI analysis results, strategic insights, and analysis metadata
with cost tracking and performance monitoring.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Index, Numeric, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any

from app.database import Base


class AnalysisResult(Base):
    """
    Storage for AI analysis results with cost tracking and performance metrics.
    
    Stores results from multi-stage analysis pipeline including filtering,
    relevance scoring, insight extraction, and summary generation.
    """
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Content and user references
    content_id = Column(Integer, ForeignKey("discovered_content.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Analysis metadata
    analysis_batch_id = Column(String(50), nullable=True, index=True)  # For batch processing tracking
    pipeline_id = Column(String(50), nullable=True, index=True)  # Pipeline execution ID
    
    # Stage completion tracking
    stage_completed = Column(String(50), nullable=False, index=True)  # filtering, relevance, insight, summary
    analysis_depth = Column(String(20), default="standard")  # quick, standard, deep
    
    # Stage 1: Filtering results
    filter_passed = Column(Boolean, nullable=False, index=True)
    filter_score = Column(Numeric(5, 4), nullable=False)
    filter_priority = Column(String(20), nullable=False)  # critical, high, medium, low
    filter_matched_keywords = Column(JSON, nullable=True)  # Array of matched keywords
    filter_matched_entities = Column(JSON, nullable=True)  # Array of matched entities
    filter_reason = Column(Text, nullable=True)  # Reason if failed
    
    # Stage 2: Relevance analysis (if passed filtering)
    relevance_score = Column(Numeric(5, 4), nullable=True)
    strategic_alignment = Column(Numeric(5, 4), nullable=True)  # Alignment with user strategic goals
    competitive_impact = Column(Numeric(5, 4), nullable=True)  # Potential competitive impact
    urgency_score = Column(Numeric(5, 4), nullable=True)  # Time sensitivity
    
    # Stage 3: AI-generated insights (if relevant)
    key_insights = Column(JSON, nullable=True)  # Array of key insights
    action_items = Column(JSON, nullable=True)  # Array of suggested actions
    strategic_implications = Column(JSON, nullable=True)  # Array of strategic implications
    risk_assessment = Column(JSON, nullable=True)  # Risk analysis if applicable
    opportunity_assessment = Column(JSON, nullable=True)  # Opportunity analysis if applicable
    
    # Stage 4: Summary generation
    executive_summary = Column(Text, nullable=True)  # Brief executive summary
    detailed_analysis = Column(Text, nullable=True)  # Detailed analysis text
    confidence_reasoning = Column(Text, nullable=True)  # Why AI is confident/uncertain
    
    # AI provider and cost tracking
    ai_provider = Column(String(50), nullable=True)  # openai, anthropic, etc.
    model_used = Column(String(100), nullable=True)  # gpt-4, claude-3, etc.
    total_tokens_used = Column(Integer, default=0)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    ai_cost_cents = Column(Integer, default=0)  # Total cost in cents
    
    # Performance and quality metrics
    processing_time_ms = Column(Integer, default=0)
    confidence_level = Column(Numeric(5, 4), default=0.0)  # Overall confidence (0.0-1.0)
    quality_score = Column(Numeric(5, 4), default=0.0)  # Analysis quality assessment
    
    # Metadata and tracking
    analysis_timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    prompt_template_version = Column(String(20), nullable=True)  # Template version used
    analysis_metadata = Column(JSON, nullable=True)  # Additional metadata
    
    # Administrative fields
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    content = relationship("DiscoveredContent", back_populates="analysis_results")
    user = relationship("User", back_populates="analysis_results")
    strategic_insights = relationship("StrategicInsight", back_populates="analysis_result", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_analysis_user_timestamp', 'user_id', 'analysis_timestamp'),
        Index('idx_analysis_stage_score', 'stage_completed', 'relevance_score'),
        Index('idx_analysis_batch_filter', 'analysis_batch_id', 'filter_passed'),
        Index('idx_analysis_cost_tracking', 'ai_provider', 'ai_cost_cents'),
    )
    
    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, user_id={self.user_id}, stage='{self.stage_completed}', relevance={self.relevance_score})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "content_id": self.content_id,
            "user_id": self.user_id,
            "stage_completed": self.stage_completed,
            "filter_passed": self.filter_passed,
            "filter_score": float(self.filter_score) if self.filter_score else None,
            "filter_priority": self.filter_priority,
            "relevance_score": float(self.relevance_score) if self.relevance_score else None,
            "strategic_alignment": float(self.strategic_alignment) if self.strategic_alignment else None,
            "competitive_impact": float(self.competitive_impact) if self.competitive_impact else None,
            "key_insights": self.key_insights,
            "action_items": self.action_items,
            "executive_summary": self.executive_summary,
            "confidence_level": float(self.confidence_level) if self.confidence_level else None,
            "ai_cost_cents": self.ai_cost_cents,
            "processing_time_ms": self.processing_time_ms,
            "analysis_timestamp": self.analysis_timestamp.isoformat() if self.analysis_timestamp else None
        }


class StrategicInsight(Base):
    """
    Extracted strategic insights with categorization and tracking.
    
    Stores high-level strategic insights extracted from content analysis
    with categorization, priority scoring, and action tracking.
    """
    __tablename__ = "strategic_insights"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # References
    analysis_result_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    content_id = Column(Integer, ForeignKey("discovered_content.id"), nullable=False, index=True)
    
    # Insight classification
    insight_type = Column(String(50), nullable=False, index=True)  # competitive, market, regulatory, opportunity, risk
    insight_category = Column(String(100), nullable=False)  # More specific categorization
    insight_priority = Column(String(20), nullable=False, index=True)  # critical, high, medium, low
    
    # Insight content
    insight_title = Column(String(500), nullable=False)  # Brief title/summary
    insight_description = Column(Text, nullable=False)  # Detailed description
    insight_implications = Column(Text, nullable=True)  # Strategic implications
    
    # Supporting evidence
    supporting_evidence = Column(JSON, nullable=True)  # Key quotes, data points
    confidence_factors = Column(JSON, nullable=True)  # Why AI is confident
    uncertainty_factors = Column(JSON, nullable=True)  # Areas of uncertainty
    
    # Action and timing
    suggested_actions = Column(JSON, nullable=True)  # Array of suggested actions
    timeline_relevance = Column(String(50), nullable=True)  # immediate, short_term, medium_term, long_term
    estimated_impact = Column(String(50), nullable=True)  # low, medium, high, critical
    
    # Strategic context alignment
    aligned_focus_areas = Column(JSON, nullable=True)  # Which user focus areas this relates to
    aligned_entities = Column(JSON, nullable=True)  # Which tracked entities this affects
    strategic_goal_alignment = Column(JSON, nullable=True)  # Which strategic goals this supports
    
    # Scoring and metrics
    relevance_score = Column(Numeric(5, 4), nullable=False)
    novelty_score = Column(Numeric(5, 4), default=0.0)  # How new/unique this insight is
    actionability_score = Column(Numeric(5, 4), default=0.0)  # How actionable this insight is
    
    # User interaction tracking
    user_rating = Column(Integer, nullable=True)  # 1-5 user rating
    user_feedback = Column(Text, nullable=True)  # User feedback text
    marked_as_actionable = Column(Boolean, default=False)  # User marked for action
    action_taken = Column(Boolean, default=False)  # User took action
    action_notes = Column(Text, nullable=True)  # Notes on actions taken
    
    # Metadata
    extracted_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    ai_extraction_metadata = Column(JSON, nullable=True)  # AI extraction details
    
    # Administrative fields
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    analysis_result = relationship("AnalysisResult", back_populates="strategic_insights")
    user = relationship("User", back_populates="strategic_insights")
    content = relationship("DiscoveredContent", back_populates="strategic_insights")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_insights_user_priority', 'user_id', 'insight_priority'),
        Index('idx_insights_type_category', 'insight_type', 'insight_category'),
        Index('idx_insights_timeline_impact', 'timeline_relevance', 'estimated_impact'),
        Index('idx_insights_user_actionable', 'user_id', 'marked_as_actionable'),
        Index('idx_insights_user_extracted', 'user_id', 'extracted_at'),
    )
    
    def __repr__(self):
        return f"<StrategicInsight(id={self.id}, type='{self.insight_type}', priority='{self.insight_priority}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "analysis_result_id": self.analysis_result_id,
            "insight_type": self.insight_type,
            "insight_category": self.insight_category,
            "insight_priority": self.insight_priority,
            "insight_title": self.insight_title,
            "insight_description": self.insight_description,
            "insight_implications": self.insight_implications,
            "suggested_actions": self.suggested_actions,
            "timeline_relevance": self.timeline_relevance,
            "estimated_impact": self.estimated_impact,
            "relevance_score": float(self.relevance_score) if self.relevance_score else None,
            "novelty_score": float(self.novelty_score) if self.novelty_score else None,
            "actionability_score": float(self.actionability_score) if self.actionability_score else None,
            "user_rating": self.user_rating,
            "marked_as_actionable": self.marked_as_actionable,
            "action_taken": self.action_taken,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None
        }


class AnalysisJob(Base):
    """
    Tracking for analysis job execution and batch processing.
    
    Manages analysis job queue, batch processing, and job status tracking
    for monitoring and performance optimization.
    """
    __tablename__ = "analysis_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Job identification
    job_id = Column(String(100), nullable=False, unique=True, index=True)
    batch_id = Column(String(100), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Job configuration
    job_type = Column(String(50), nullable=False)  # single_content, batch_analysis, user_full_analysis
    job_priority = Column(String(20), nullable=False, index=True)  # critical, high, medium, low
    analysis_stages = Column(JSON, nullable=False)  # Which stages to run
    
    # Content references
    content_ids = Column(JSON, nullable=False)  # Array of content IDs to analyze
    total_content_count = Column(Integer, nullable=False)
    
    # Job status and progress
    status = Column(String(50), nullable=False, index=True)  # queued, in_progress, completed, failed, cancelled
    progress_percentage = Column(Integer, default=0)
    processed_count = Column(Integer, default=0)
    successful_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    
    # Cost and performance tracking
    estimated_cost_cents = Column(Integer, nullable=True)
    actual_cost_cents = Column(Integer, default=0)
    total_processing_time_ms = Column(Integer, default=0)
    
    # Timing
    queued_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Job configuration and context
    job_config = Column(JSON, nullable=True)  # Job-specific configuration
    user_context = Column(JSON, nullable=True)  # User strategic context snapshot
    
    # Administrative fields
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="analysis_jobs")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_jobs_user_status', 'user_id', 'status'),
        Index('idx_jobs_priority_queued', 'job_priority', 'queued_at'),
        Index('idx_jobs_batch_status', 'batch_id', 'status'),
        Index('idx_jobs_type_status', 'job_type', 'status'),
    )
    
    def __repr__(self):
        return f"<AnalysisJob(id='{self.job_id}', status='{self.status}', progress={self.progress_percentage}%)>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "batch_id": self.batch_id,
            "user_id": self.user_id,
            "job_type": self.job_type,
            "job_priority": self.job_priority,
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "total_content_count": self.total_content_count,
            "processed_count": self.processed_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "estimated_cost_cents": self.estimated_cost_cents,
            "actual_cost_cents": self.actual_cost_cents,
            "queued_at": self.queued_at.isoformat() if self.queued_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message
        }
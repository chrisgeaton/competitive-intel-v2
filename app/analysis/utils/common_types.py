"""
Common types and data structures for Analysis Service.

Consolidated type definitions to eliminate duplication.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any


class AnalysisStage(Enum):
    """Analysis pipeline stages."""
    FILTERING = "filtering"      # Stage 1: Content filtering (70% cost savings)
    RELEVANCE = "relevance"      # Stage 2: Relevance scoring
    INSIGHT = "insight"          # Stage 3: Insight extraction
    SUMMARY = "summary"          # Stage 4: Summary generation
    PREPROCESSING = "preprocessing"
    ENRICHMENT = "enrichment"
    ANALYSIS = "analysis"
    SCORING = "scoring"
    INSIGHT_EXTRACTION = "insight_extraction"
    SUMMARIZATION = "summarization"
    POSTPROCESSING = "postprocessing"


# Alias for pipeline compatibility
PipelineStage = AnalysisStage


class ContentPriority(Enum):
    """Content priority levels for analysis."""
    CRITICAL = "critical"        # Immediate analysis required
    HIGH = "high"               # Priority analysis
    MEDIUM = "medium"           # Standard analysis
    LOW = "low"                # Batch analysis


class FilterStrategy(Enum):
    """Filtering strategies for content evaluation."""
    STRICT = "strict"           # All conditions must match
    BALANCED = "balanced"       # Most conditions should match
    LENIENT = "lenient"         # Some conditions should match
    CUSTOM = "custom"           # Custom logic


class ProcessingStatus(Enum):
    """Processing status for pipeline stages."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FilterMatch:
    """Details of a filter match."""
    filter_type: str
    matched_value: str
    match_location: str          # 'title', 'content', 'metadata'
    match_strength: float        # 0.0 to 1.0
    context: Optional[str] = None  # Surrounding text


@dataclass
class FilterResult:
    """Result from content filtering."""
    passed: bool
    score: float                 # 0.0 to 1.0
    matches: List[FilterMatch] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0
    processing_time_ms: int = 0
    
    def add_match(self, match: FilterMatch):
        """Add a match to the result."""
        self.matches.append(match)
        
    def add_failure_reason(self, reason: str):
        """Add a failure reason."""
        self.failure_reasons.append(reason)


@dataclass
class AnalysisContext:
    """Context for content analysis including user preferences and strategic goals."""
    user_id: int
    strategic_profile: Optional[Dict[str, Any]] = None
    focus_areas: List[Dict[str, Any]] = field(default_factory=list)
    tracked_entities: List[Dict[str, Any]] = field(default_factory=list)
    delivery_preferences: Optional[Dict[str, Any]] = None
    analysis_depth: str = "standard"  # 'quick', 'standard', 'deep'
    cost_limit: Optional[Decimal] = None
    
    # Enhanced properties from User Config Service integration
    @property
    def industry(self) -> str:
        """Get user's industry from strategic profile."""
        if self.strategic_profile:
            return self.strategic_profile.get("industry", "generic")
        return "generic"
    
    @property
    def role(self) -> str:
        """Get user's role from strategic profile."""
        if self.strategic_profile:
            return self.strategic_profile.get("role", "generic")
        return "generic"
    
    @property
    def organization_type(self) -> str:
        """Get organization type from strategic profile."""
        if self.strategic_profile:
            return self.strategic_profile.get("organization_type", "generic")
        return "generic"
    
    @property
    def strategic_goals(self) -> List[str]:
        """Get strategic goals from strategic profile."""
        if self.strategic_profile:
            goals = self.strategic_profile.get("strategic_goals", [])
            if isinstance(goals, list):
                return goals
            elif isinstance(goals, str):
                return [g.strip() for g in goals.split(",") if g.strip()]
        return []
    
    @property
    def priority(self) -> str:
        """Determine overall priority based on user context."""
        # High priority for enterprise users with critical goals
        if self.organization_type in ["enterprise", "large_enterprise"]:
            return "high"
        elif self.organization_type in ["startup", "small_business"]:
            return "medium"
        else:
            return "medium"
    
    @property
    def focus_area_keywords(self) -> List[str]:
        """Get all keywords from focus areas."""
        keywords = []
        for area in self.focus_areas:
            area_keywords = area.get("keywords", "")
            if isinstance(area_keywords, str):
                keywords.extend([k.strip() for k in area_keywords.split(",") if k.strip()])
            elif isinstance(area_keywords, list):
                keywords.extend(area_keywords)
        return list(set(keywords))  # Remove duplicates
    
    @property
    def entity_keywords(self) -> List[str]:
        """Get all keywords from tracked entities."""
        keywords = []
        for entity in self.tracked_entities:
            entity_keywords = entity.get("keywords", "")
            if isinstance(entity_keywords, str):
                keywords.extend([k.strip() for k in entity_keywords.split(",") if k.strip()])
            elif isinstance(entity_keywords, list):
                keywords.extend(entity_keywords)
        return list(set(keywords))
    
    @property
    def all_keywords(self) -> List[str]:
        """Get all keywords from focus areas and entities."""
        return self.focus_area_keywords + self.entity_keywords
    
    @property
    def entity_names(self) -> List[str]:
        """Get names of all tracked entities."""
        names = []
        for entity in self.tracked_entities:
            name = entity.get("entity_name") or entity.get("name")
            if name:
                names.append(name)
        return names
    
    @property
    def high_priority_entities(self) -> List[str]:
        """Get names of high priority tracked entities."""
        high_priority = []
        for entity in self.tracked_entities:
            if entity.get("priority", 3) >= 3:  # Priority 3 or 4
                name = entity.get("entity_name") or entity.get("name")
                if name:
                    high_priority.append(name)
        return high_priority
    
    @property
    def delivery_frequency(self) -> str:
        """Get delivery frequency preference."""
        if self.delivery_preferences:
            return self.delivery_preferences.get("frequency", "daily")
        return "daily"
    
    @property
    def min_significance_level(self) -> str:
        """Get minimum significance level for content."""
        if self.delivery_preferences:
            return self.delivery_preferences.get("min_significance_level", "medium")
        return "medium"
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis context for AI prompts."""
        return {
            "user_id": self.user_id,
            "industry": self.industry,
            "role": self.role,
            "organization_type": self.organization_type,
            "strategic_goals": self.strategic_goals,
            "focus_areas_count": len(self.focus_areas),
            "tracked_entities_count": len(self.tracked_entities),
            "high_priority_entities": self.high_priority_entities,
            "analysis_depth": self.analysis_depth,
            "priority": self.priority,
            "min_significance": self.min_significance_level
        }


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: AnalysisStage
    status: ProcessingStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_time_ms: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    cost: Decimal = Decimal("0.00")
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete result from pipeline processing."""
    pipeline_id: str
    content_id: int
    user_id: int
    stages_completed: List[StageResult] = field(default_factory=list)
    total_processing_time_ms: int = 0
    total_cost: Decimal = Decimal("0.00")
    final_score: float = 0.0
    final_status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now())
    completed_at: Optional[datetime] = None
    
    def add_stage_result(self, result: StageResult):
        """Add a stage result to the pipeline."""
        self.stages_completed.append(result)
        self.total_processing_time_ms += result.processing_time_ms
        self.total_cost += result.cost
        
    def get_stage_result(self, stage: AnalysisStage) -> Optional[StageResult]:
        """Get result for a specific stage."""
        for result in self.stages_completed:
            if result.stage == stage:
                return result
        return None
        
    def is_stage_completed(self, stage: AnalysisStage) -> bool:
        """Check if a stage was completed successfully."""
        result = self.get_stage_result(stage)
        return result and result.status == ProcessingStatus.COMPLETED


@dataclass
class AnalysisResult:
    """Complete analysis result for content."""
    content_id: int
    user_id: int
    stage_completed: AnalysisStage
    
    # Stage 1: Filtering results
    filter_passed: bool
    filter_score: float
    filter_priority: ContentPriority
    
    # Stage 2: Relevance analysis (if passed filtering)
    relevance_score: Optional[float] = None
    strategic_alignment: Optional[float] = None
    competitive_impact: Optional[float] = None
    
    # Stage 3: Insight extraction (if relevant)
    key_insights: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    strategic_implications: List[str] = field(default_factory=list)
    
    # Stage 4: Summary generation
    executive_summary: Optional[str] = None
    detailed_analysis: Optional[str] = None
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=lambda: datetime.now())
    processing_time_ms: int = 0
    ai_cost: Decimal = Decimal("0.00")
    model_used: str = "gpt-4o-mini"
    confidence_level: float = 0.0


@dataclass
class AnalysisBatch:
    """Batch of content for analysis."""
    batch_id: str
    user_id: int
    content_items: List[Dict[str, Any]]
    context: AnalysisContext
    priority: ContentPriority
    created_at: datetime = field(default_factory=lambda: datetime.now())
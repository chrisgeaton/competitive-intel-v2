"""
Shared types and data structures for Analysis Service.

Consolidates all common types, enums, and dataclasses to eliminate
duplication across the Analysis Service codebase.
"""

import json
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod


# === Core Enums (Consolidated) ===

class AnalysisStage(Enum):
    """Analysis pipeline stages - consolidated from multiple files."""
    FILTERING = "filtering"
    RELEVANCE_ANALYSIS = "relevance_analysis"
    INSIGHT_EXTRACTION = "insight_extraction"
    SUMMARY_GENERATION = "summary_generation"
    
    # Legacy aliases for compatibility
    RELEVANCE = "relevance_analysis"
    INSIGHT = "insight_extraction"
    SUMMARY = "summary_generation"


class ContentPriority(Enum):
    """Content priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    @classmethod
    def from_score(cls, score: float) -> 'ContentPriority':
        """Convert relevance score to priority."""
        if score >= 0.8:
            return cls.CRITICAL
        elif score >= 0.6:
            return cls.HIGH
        elif score >= 0.4:
            return cls.MEDIUM
        else:
            return cls.LOW


class AIProvider(Enum):
    """Supported AI providers - consolidated."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"
    
    @property
    def is_mock(self) -> bool:
        """Check if provider is mock."""
        return self == self.MOCK
    
    @property
    def requires_api_key(self) -> bool:
        """Check if provider requires API key."""
        return self in (self.OPENAI, self.ANTHROPIC)


class IndustryType(Enum):
    """Industry classifications for specialized analysis."""
    HEALTHCARE = "healthcare"
    FINTECH = "fintech"
    NONPROFIT = "nonprofit"
    TECHNOLOGY = "technology"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    EDUCATION = "education"
    GOVERNMENT = "government"
    ENERGY = "energy"
    GENERIC = "generic"
    
    @classmethod
    def from_string(cls, industry: str) -> 'IndustryType':
        """Convert string to IndustryType with fallback."""
        industry_lower = industry.lower()
        
        mapping = {
            "health": cls.HEALTHCARE,
            "medical": cls.HEALTHCARE,
            "fintech": cls.FINTECH,
            "financial": cls.FINTECH,
            "nonprofit": cls.NONPROFIT,
            "non-profit": cls.NONPROFIT,
            "tech": cls.TECHNOLOGY,
            "software": cls.TECHNOLOGY,
            "manufacturing": cls.MANUFACTURING,
            "retail": cls.RETAIL,
            "ecommerce": cls.RETAIL,
            "education": cls.EDUCATION,
            "government": cls.GOVERNMENT,
            "public": cls.GOVERNMENT,
            "energy": cls.ENERGY
        }
        
        for key, value in mapping.items():
            if key in industry_lower:
                return value
        
        return cls.GENERIC


class RoleType(Enum):
    """Role classifications for analysis context."""
    CEO = "ceo"
    CTO = "cto"
    PRODUCT_MANAGER = "product_manager"
    STRATEGY_ANALYST = "strategy_analyst"
    BUSINESS_ANALYST = "business_analyst"
    MARKETING_MANAGER = "marketing_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    RESEARCH_DIRECTOR = "research_director"
    GENERIC = "generic"
    
    @classmethod
    def from_string(cls, role: str) -> 'RoleType':
        """Convert string to RoleType with fallback."""
        role_lower = role.lower()
        
        mapping = {
            "ceo": cls.CEO,
            "chief executive": cls.CEO,
            "cto": cls.CTO,
            "chief technology": cls.CTO,
            "product manager": cls.PRODUCT_MANAGER,
            "strategy analyst": cls.STRATEGY_ANALYST,
            "business analyst": cls.BUSINESS_ANALYST,
            "marketing manager": cls.MARKETING_MANAGER,
            "compliance": cls.COMPLIANCE_OFFICER,
            "research director": cls.RESEARCH_DIRECTOR
        }
        
        for key, value in mapping.items():
            if key in role_lower:
                return value
        
        return cls.GENERIC


# === Enhanced Data Classes ===

@dataclass
class ServiceConfig:
    """Consolidated service configuration."""
    batch_size: int = 10
    max_concurrent_analyses: int = 5
    filter_threshold: float = 0.3
    relevance_threshold: float = 0.5
    cost_limit_cents: Optional[int] = None
    cache_ttl_seconds: int = 3600
    retry_attempts: int = 3
    timeout_seconds: int = 300
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.batch_size <= 0:
            self.batch_size = 10
        if self.filter_threshold < 0 or self.filter_threshold > 1:
            self.filter_threshold = 0.3
        if self.relevance_threshold < 0 or self.relevance_threshold > 1:
            self.relevance_threshold = 0.5


@dataclass
class AnalysisContext:
    """Enhanced analysis context with optimized property access."""
    user_id: int
    strategic_profile: Optional[Dict[str, Any]] = None
    focus_areas: List[Dict[str, Any]] = field(default_factory=list)
    tracked_entities: List[Dict[str, Any]] = field(default_factory=list)
    delivery_preferences: Optional[Dict[str, Any]] = None
    analysis_depth: str = "standard"
    cost_limit: Optional[Decimal] = None
    
    # Cached properties for performance
    _industry_cache: Optional[IndustryType] = field(default=None, init=False)
    _role_cache: Optional[RoleType] = field(default=None, init=False)
    _keywords_cache: Optional[List[str]] = field(default=None, init=False)
    
    @property
    def industry(self) -> IndustryType:
        """Get user's industry with caching."""
        if self._industry_cache is None:
            if self.strategic_profile:
                industry_str = self.strategic_profile.get("industry", "generic")
                self._industry_cache = IndustryType.from_string(industry_str)
            else:
                self._industry_cache = IndustryType.GENERIC
        return self._industry_cache
    
    @property
    def role(self) -> RoleType:
        """Get user's role with caching."""
        if self._role_cache is None:
            if self.strategic_profile:
                role_str = self.strategic_profile.get("role", "generic")
                self._role_cache = RoleType.from_string(role_str)
            else:
                self._role_cache = RoleType.GENERIC
        return self._role_cache
    
    @property
    def strategic_goals(self) -> List[str]:
        """Get strategic goals list."""
        if self.strategic_profile:
            goals = self.strategic_profile.get("strategic_goals", [])
            if isinstance(goals, list):
                return goals
            elif isinstance(goals, str):
                return [g.strip() for g in goals.split(",") if g.strip()]
        return []
    
    @property
    def all_keywords(self) -> List[str]:
        """Get all keywords with caching."""
        if self._keywords_cache is None:
            keywords = []
            
            # Focus area keywords
            for area in self.focus_areas:
                area_keywords = area.get("keywords", "")
                if isinstance(area_keywords, str):
                    keywords.extend([k.strip() for k in area_keywords.split(",") if k.strip()])
                elif isinstance(area_keywords, list):
                    keywords.extend(area_keywords)
            
            # Entity keywords
            for entity in self.tracked_entities:
                entity_keywords = entity.get("keywords", "")
                if isinstance(entity_keywords, str):
                    keywords.extend([k.strip() for k in entity_keywords.split(",") if k.strip()])
                elif isinstance(entity_keywords, list):
                    keywords.extend(entity_keywords)
            
            self._keywords_cache = list(set(keywords))  # Remove duplicates
        
        return self._keywords_cache
    
    @property
    def priority(self) -> str:
        """Determine analysis priority based on context."""
        org_type = ""
        if self.strategic_profile:
            org_type = self.strategic_profile.get("organization_type", "").lower()
        
        if "enterprise" in org_type or "large" in org_type:
            return "high"
        elif "startup" in org_type or "small" in org_type:
            return "medium"
        else:
            return "medium"
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get optimized context summary."""
        return {
            "user_id": self.user_id,
            "industry": self.industry.value,
            "role": self.role.value,
            "strategic_goals": self.strategic_goals,
            "keyword_count": len(self.all_keywords),
            "entity_count": len(self.tracked_entities),
            "priority": self.priority,
            "analysis_depth": self.analysis_depth
        }


@dataclass
class AIModelConfig:
    """AI model configuration with optimization."""
    provider: AIProvider
    model_name: str
    max_tokens: int
    temperature: float
    cost_per_1k_input: Decimal
    cost_per_1k_output: Decimal
    context_window: int
    supports_json: bool = True
    
    @property
    def cost_efficiency_score(self) -> float:
        """Calculate cost efficiency score for model selection."""
        avg_cost = (self.cost_per_1k_input + self.cost_per_1k_output) / 2
        return float(self.context_window / max(1, avg_cost * 1000))
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> int:
        """Estimate cost in cents."""
        input_cost = (input_tokens / 1000) * float(self.cost_per_1k_input)
        output_cost = (output_tokens / 1000) * float(self.cost_per_1k_output)
        return int((input_cost + output_cost) * 100)


@dataclass
class AIResponse:
    """Standardized AI response with enhanced metadata."""
    content: str
    usage: Dict[str, int]
    model: str
    provider: AIProvider
    cost_cents: int
    processing_time_ms: int
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate processing speed."""
        if self.processing_time_ms > 0:
            total_tokens = self.usage.get("total_tokens", 0)
            return (total_tokens * 1000) / self.processing_time_ms
        return 0.0
    
    @property
    def cost_per_token(self) -> float:
        """Calculate cost efficiency."""
        total_tokens = self.usage.get("total_tokens", 0)
        if total_tokens > 0:
            return self.cost_cents / total_tokens
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "usage": self.usage,
            "model": self.model,
            "provider": self.provider.value,
            "cost_cents": self.cost_cents,
            "processing_time_ms": self.processing_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "cost_per_token": self.cost_per_token,
            "metadata": self.metadata or {}
        }


@dataclass
class PromptTemplate:
    """Enhanced prompt template with optimization."""
    stage: AnalysisStage
    industry: IndustryType
    role: RoleType
    system_prompt: str
    user_prompt_template: str
    response_format: Dict[str, Any]
    keywords: List[str]
    focus_areas: List[str]
    
    # Performance optimization fields
    _compiled_template: Optional[str] = field(default=None, init=False)
    _template_hash: Optional[str] = field(default=None, init=False)
    
    @property
    def template_id(self) -> str:
        """Generate unique template identifier."""
        if self._template_hash is None:
            template_str = f"{self.stage.value}_{self.industry.value}_{self.role.value}"
            import hashlib
            self._template_hash = hashlib.md5(template_str.encode()).hexdigest()[:8]
        return self._template_hash
    
    def build_prompt(self, context: AnalysisContext, content: str, **kwargs) -> str:
        """Build optimized prompt with caching."""
        template_vars = {
            "role": context.role.value,
            "industry": context.industry.value,
            "strategic_goals": ", ".join(context.strategic_goals),
            "focus_areas": ", ".join([fa.get("focus_area", "") for fa in context.focus_areas]),
            "content": content,
            "priority": context.priority,
            **kwargs
        }
        
        # Use cached template if available
        if self._compiled_template is None:
            self._compiled_template = f"{self.system_prompt}\n\n{self.user_prompt_template}"
        
        return self._compiled_template.format(**template_vars)


# === Filter and Analysis Results ===

@dataclass
class FilterResult:
    """Enhanced filter result with performance optimization."""
    content_id: int
    passed: bool
    relevance_score: float
    matched_keywords: List[str]
    matched_entities: List[str]
    priority: ContentPriority
    filter_reason: Optional[str] = None
    confidence: float = 0.0
    processing_time_ms: int = 0
    
    @property
    def match_quality_score(self) -> float:
        """Calculate match quality based on keywords and entities."""
        keyword_score = min(1.0, len(self.matched_keywords) / 5.0)
        entity_score = min(1.0, len(self.matched_entities) / 3.0)
        return (keyword_score + entity_score) / 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "content_id": self.content_id,
            "passed": self.passed,
            "relevance_score": self.relevance_score,
            "matched_keywords": self.matched_keywords,
            "matched_entities": self.matched_entities,
            "priority": self.priority.value,
            "filter_reason": self.filter_reason,
            "confidence": self.confidence,
            "match_quality_score": self.match_quality_score,
            "processing_time_ms": self.processing_time_ms
        }


@dataclass
class AnalysisBatch:
    """Optimized analysis batch with enhanced tracking."""
    batch_id: str
    user_id: int
    content_items: List[Dict[str, Any]]
    context: AnalysisContext
    priority: ContentPriority
    created_at: datetime = field(default_factory=datetime.now)
    
    # Performance tracking
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_items(self) -> int:
        """Get total number of items in batch."""
        return len(self.content_items)
    
    @property
    def high_priority_items(self) -> int:
        """Count high priority items in batch."""
        return sum(1 for item in self.content_items 
                  if item.get("filter_result", {}).get("priority") in ["critical", "high"])
    
    def add_processing_stat(self, stage: str, stat_name: str, value: Any):
        """Add processing statistics."""
        if stage not in self.processing_stats:
            self.processing_stats[stage] = {}
        self.processing_stats[stage][stat_name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "batch_id": self.batch_id,
            "user_id": self.user_id,
            "total_items": self.total_items,
            "high_priority_items": self.high_priority_items,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "processing_stats": self.processing_stats
        }


# === Validation Utilities ===

def validate_analysis_context(context: AnalysisContext) -> None:
    """Optimized context validation."""
    if not context.user_id or context.user_id <= 0:
        raise ValueError("Invalid user_id")
    
    if not context.strategic_profile:
        raise ValueError("Strategic profile is required")
    
    required_fields = ["industry", "role", "strategic_goals"]
    missing_fields = []
    
    for field in required_fields:
        if not context.strategic_profile.get(field):
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Strategic profile missing: {', '.join(missing_fields)}")


def validate_content_for_analysis(content: Dict[str, Any]) -> None:
    """Optimized content validation."""
    if not content.get("id"):
        raise ValueError("Content missing ID")
    
    title = content.get("title", "")
    text = content.get("content_text", "")
    total_text = f"{title} {text}".strip()
    
    if len(total_text) < 50:
        raise ValueError(f"Content too short: {len(total_text)} chars (minimum 50)")


# === Common Constants ===

DEFAULT_CONFIG = ServiceConfig()

STAGE_WEIGHTS = {
    AnalysisStage.FILTERING: 0.1,
    AnalysisStage.RELEVANCE_ANALYSIS: 0.3,
    AnalysisStage.INSIGHT_EXTRACTION: 0.4,
    AnalysisStage.SUMMARY_GENERATION: 0.2
}

INDUSTRY_KEYWORDS = {
    IndustryType.HEALTHCARE: ["health", "medical", "patient", "clinical", "FDA", "HIPAA"],
    IndustryType.FINTECH: ["finance", "payment", "blockchain", "banking", "regulatory"],
    IndustryType.NONPROFIT: ["nonprofit", "charity", "grant", "funding", "social"],
    IndustryType.TECHNOLOGY: ["tech", "software", "AI", "platform", "innovation"]
}
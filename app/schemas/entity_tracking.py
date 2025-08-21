"""
Entity tracking schemas with validation for competitive intelligence.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class EntityType(str, Enum):
    """Types of entities that can be tracked."""
    COMPETITOR = "competitor"
    ORGANIZATION = "organization"
    TOPIC = "topic"
    PERSON = "person"
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    MARKET_SEGMENT = "market_segment"
    REGULATORY_BODY = "regulatory_body"


class TrackingPriority(int, Enum):
    """Priority levels for entity tracking."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TrackingEntityBase(BaseModel):
    """Base tracking entity schema."""
    name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="Name of the entity to track"
    )
    entity_type: EntityType = Field(
        ...,
        description="Type of entity"
    )
    domain: Optional[str] = Field(
        None,
        max_length=255,
        description="Domain/website of the entity"
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Description of the entity"
    )
    industry: Optional[str] = Field(
        None,
        max_length=100,
        description="Industry sector of the entity"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about the entity"
    )

    @validator('name')
    def validate_name(cls, v):
        """Validate entity name."""
        if not v or v.isspace():
            raise ValueError("Entity name cannot be empty")
        return v.strip()

    @validator('domain')
    def validate_domain(cls, v):
        """Validate domain format."""
        if v:
            v = v.strip().lower()
            # Remove common prefixes
            if v.startswith('http://'):
                v = v[7:]
            if v.startswith('https://'):
                v = v[8:]
            if v.startswith('www.'):
                v = v[4:]
            # Remove trailing slash
            v = v.rstrip('/')
            return v
        return v

    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata size."""
        if v:
            # Limit metadata size (approximate check)
            import json
            metadata_str = json.dumps(v)
            if len(metadata_str) > 5000:
                raise ValueError("Metadata too large (max 5KB)")
        return v


class TrackingEntityCreate(TrackingEntityBase):
    """Schema for creating a tracking entity."""
    
    class Config:
        schema_extra = {
            "example": {
                "name": "OpenAI",
                "entity_type": "competitor",
                "domain": "openai.com",
                "description": "AI research company",
                "industry": "Artificial Intelligence",
                "metadata": {
                    "founded": "2015",
                    "headquarters": "San Francisco",
                    "key_products": ["GPT", "DALL-E", "Codex"]
                }
            }
        }


class TrackingEntityUpdate(BaseModel):
    """Schema for updating a tracking entity."""
    name: Optional[str] = Field(None, min_length=2, max_length=255)
    entity_type: Optional[EntityType] = None
    domain: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    industry: Optional[str] = Field(None, max_length=100)
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "description": "Leading AI research company",
                "metadata": {
                    "valuation": "$100B",
                    "employees": "500+"
                }
            }
        }


class TrackingEntityResponse(BaseModel):
    """Schema for tracking entity responses."""
    id: int
    name: str
    entity_type: str
    domain: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, alias="metadata_json")
    created_at: datetime

    class Config:
        from_attributes = True
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "id": 1,
                "name": "OpenAI",
                "entity_type": "competitor",
                "domain": "openai.com",
                "description": "AI research company",
                "industry": "Artificial Intelligence",
                "metadata": {"founded": "2015"},
                "created_at": "2025-01-01T00:00:00Z"
            }
        }


class UserEntityTrackingBase(BaseModel):
    """Base user entity tracking schema."""
    entity_id: int = Field(
        ...,
        description="ID of the entity to track"
    )
    priority: Optional[TrackingPriority] = Field(
        default=TrackingPriority.HIGH,
        description="Tracking priority"
    )
    custom_keywords: Optional[List[str]] = Field(
        default_factory=list,
        max_items=10,
        description="Custom keywords for this entity"
    )
    tracking_enabled: Optional[bool] = Field(
        default=True,
        description="Whether tracking is enabled"
    )

    @validator('custom_keywords', pre=True)
    def validate_keywords(cls, v):
        """Validate and clean keywords."""
        if v is None:
            return []
        if isinstance(v, list):
            # Clean and deduplicate
            seen = set()
            unique_keywords = []
            for keyword in v:
                cleaned = str(keyword).strip()
                if cleaned and len(cleaned) >= 2 and cleaned not in seen:
                    seen.add(cleaned)
                    unique_keywords.append(cleaned)
            return unique_keywords[:10]
        return v


class UserEntityTrackingCreate(UserEntityTrackingBase):
    """Schema for creating user entity tracking."""
    
    class Config:
        schema_extra = {
            "example": {
                "entity_id": 1,
                "priority": 3,
                "custom_keywords": ["GPT-4", "ChatGPT", "AI safety"],
                "tracking_enabled": True
            }
        }


class UserEntityTrackingUpdate(BaseModel):
    """Schema for updating user entity tracking."""
    priority: Optional[TrackingPriority] = None
    custom_keywords: Optional[List[str]] = Field(None, max_items=10)
    tracking_enabled: Optional[bool] = None

    class Config:
        schema_extra = {
            "example": {
                "priority": 4,
                "custom_keywords": ["GPT-5", "AGI"],
                "tracking_enabled": True
            }
        }


class UserEntityTrackingResponse(BaseModel):
    """Schema for user entity tracking responses."""
    id: int
    user_id: int
    entity_id: int
    entity: TrackingEntityResponse
    priority: TrackingPriority
    priority_label: str
    custom_keywords: List[str]
    tracking_enabled: bool
    created_at: datetime

    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "user_id": 123,
                "entity_id": 1,
                "entity": {
                    "id": 1,
                    "name": "OpenAI",
                    "entity_type": "competitor",
                    "domain": "openai.com",
                    "description": "AI research company",
                    "industry": "Artificial Intelligence",
                    "metadata": {},
                    "created_at": "2025-01-01T00:00:00Z"
                },
                "priority": 3,
                "priority_label": "high",
                "custom_keywords": ["GPT-4", "ChatGPT"],
                "tracking_enabled": True,
                "created_at": "2025-01-01T00:00:00Z"
            }
        }


class EntityTrackingListResponse(BaseModel):
    """Schema for listing tracked entities with pagination."""
    items: List[UserEntityTrackingResponse]
    total: int
    page: int
    per_page: int
    pages: int

    class Config:
        schema_extra = {
            "example": {
                "items": [],
                "total": 10,
                "page": 1,
                "per_page": 10,
                "pages": 1
            }
        }


class EntityTrackingAnalytics(BaseModel):
    """Analytics for entity tracking."""
    total_tracked_entities: int
    entities_by_type: Dict[str, int]
    priority_distribution: Dict[str, int]
    enabled_count: int
    disabled_count: int
    top_industries: List[str]
    keyword_cloud: List[str]

    class Config:
        schema_extra = {
            "example": {
                "total_tracked_entities": 15,
                "entities_by_type": {
                    "competitor": 5,
                    "technology": 4,
                    "organization": 3,
                    "topic": 3
                },
                "priority_distribution": {
                    "critical": 2,
                    "high": 8,
                    "medium": 4,
                    "low": 1
                },
                "enabled_count": 13,
                "disabled_count": 2,
                "top_industries": ["Technology", "Finance", "Healthcare"],
                "keyword_cloud": ["AI", "blockchain", "cloud", "security"]
            }
        }


class EntitySearchRequest(BaseModel):
    """Schema for searching entities."""
    query: Optional[str] = Field(None, description="Search query")
    entity_types: Optional[List[EntityType]] = Field(None, description="Filter by entity types")
    industries: Optional[List[str]] = Field(None, description="Filter by industries")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "AI",
                "entity_types": ["competitor", "technology"],
                "industries": ["Artificial Intelligence", "Technology"]
            }
        }
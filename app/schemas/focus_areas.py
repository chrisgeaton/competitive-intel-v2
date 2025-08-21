"""
Focus areas schemas with validation for user intelligence priorities.
"""

from datetime import datetime
from typing import Optional, List
from enum import IntEnum
from pydantic import BaseModel, Field, validator


class PriorityLevel(IntEnum):
    """Priority levels for focus areas."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class FocusAreaBase(BaseModel):
    """Base focus area schema."""
    focus_area: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="The area of focus for intelligence gathering"
    )
    keywords: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Keywords associated with this focus area"
    )
    priority: Optional[PriorityLevel] = Field(
        default=PriorityLevel.HIGH,
        description="Priority level (1=low, 2=medium, 3=high, 4=critical)"
    )

    @validator('focus_area')
    def validate_focus_area(cls, v):
        """Validate focus area format."""
        if not v or v.isspace():
            raise ValueError("Focus area cannot be empty")
        return v.strip()

    @validator('keywords', pre=True)
    def validate_keywords(cls, v):
        """Validate and clean keywords."""
        if v is None:
            return []
        if isinstance(v, str):
            # Handle single string input
            return [v.strip()]
        if isinstance(v, list):
            # Clean and remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for keyword in v:
                cleaned = str(keyword).strip()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    unique_keywords.append(cleaned)
            return unique_keywords[:20]  # Limit to 20 keywords
        return v

    @validator('keywords')
    def validate_keywords_length(cls, v):
        """Validate individual keyword lengths."""
        if v:
            for keyword in v:
                if len(keyword) > 100:
                    raise ValueError(f"Keyword too long (max 100 chars): {keyword[:50]}...")
                if len(keyword) < 2:
                    raise ValueError(f"Keyword too short (min 2 chars): {keyword}")
        return v


class FocusAreaCreate(FocusAreaBase):
    """Schema for creating a focus area."""
    
    class Config:
        schema_extra = {
            "example": {
                "focus_area": "AI and Machine Learning",
                "keywords": [
                    "artificial intelligence",
                    "machine learning",
                    "deep learning",
                    "neural networks",
                    "GPT"
                ],
                "priority": 3
            }
        }


class FocusAreaUpdate(BaseModel):
    """Schema for updating a focus area."""
    focus_area: Optional[str] = Field(
        None,
        min_length=2,
        max_length=255,
        description="Updated focus area name"
    )
    keywords: Optional[List[str]] = Field(
        None,
        max_items=20,
        description="Updated keywords list"
    )
    priority: Optional[PriorityLevel] = Field(
        None,
        description="Updated priority level"
    )

    @validator('focus_area')
    def validate_focus_area(cls, v):
        """Validate focus area format."""
        if v is not None:
            if not v or v.isspace():
                raise ValueError("Focus area cannot be empty")
            return v.strip()
        return v

    @validator('keywords', pre=True)
    def validate_keywords(cls, v):
        """Validate and clean keywords."""
        if v is None:
            return None
        if isinstance(v, list):
            # Clean and remove duplicates
            seen = set()
            unique_keywords = []
            for keyword in v:
                cleaned = str(keyword).strip()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    unique_keywords.append(cleaned)
            return unique_keywords[:20]
        return v

    class Config:
        schema_extra = {
            "example": {
                "priority": 4,
                "keywords": [
                    "artificial intelligence",
                    "machine learning",
                    "AGI"
                ]
            }
        }


class FocusAreaResponse(FocusAreaBase):
    """Schema for focus area responses."""
    id: int
    user_id: int
    priority_label: str = Field(description="Human-readable priority label")
    created_at: datetime

    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "user_id": 123,
                "focus_area": "AI and Machine Learning",
                "keywords": [
                    "artificial intelligence",
                    "machine learning",
                    "deep learning"
                ],
                "priority": 3,
                "priority_label": "high",
                "created_at": "2025-01-01T00:00:00Z"
            }
        }


class FocusAreaBulkCreate(BaseModel):
    """Schema for bulk creating focus areas."""
    focus_areas: List[FocusAreaCreate] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of focus areas to create"
    )

    @validator('focus_areas')
    def validate_unique_areas(cls, v):
        """Ensure focus areas are unique."""
        areas = [area.focus_area.lower() for area in v]
        if len(areas) != len(set(areas)):
            raise ValueError("Duplicate focus areas not allowed in bulk create")
        return v

    class Config:
        schema_extra = {
            "example": {
                "focus_areas": [
                    {
                        "focus_area": "AI and Machine Learning",
                        "keywords": ["AI", "ML", "deep learning"],
                        "priority": 3
                    },
                    {
                        "focus_area": "Cybersecurity",
                        "keywords": ["security", "threats", "vulnerabilities"],
                        "priority": 4
                    }
                ]
            }
        }


class FocusAreaListResponse(BaseModel):
    """Schema for listing focus areas with pagination."""
    items: List[FocusAreaResponse]
    total: int
    page: int
    per_page: int
    pages: int

    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {
                        "id": 1,
                        "user_id": 123,
                        "focus_area": "AI and Machine Learning",
                        "keywords": ["AI", "ML"],
                        "priority": 3,
                        "priority_label": "high",
                        "created_at": "2025-01-01T00:00:00Z"
                    }
                ],
                "total": 5,
                "page": 1,
                "per_page": 10,
                "pages": 1
            }
        }


class FocusAreaAnalytics(BaseModel):
    """Analytics for user's focus areas."""
    total_focus_areas: int
    priority_distribution: dict
    keyword_count: int
    most_common_keywords: List[str]
    coverage_score: float = Field(
        description="Score indicating how comprehensive the focus areas are (0-100)"
    )
    recommendations: List[str] = Field(
        description="Recommended focus areas based on profile"
    )

    class Config:
        schema_extra = {
            "example": {
                "total_focus_areas": 5,
                "priority_distribution": {
                    "critical": 1,
                    "high": 2,
                    "medium": 1,
                    "low": 1
                },
                "keyword_count": 25,
                "most_common_keywords": ["AI", "security", "cloud"],
                "coverage_score": 75.0,
                "recommendations": [
                    "Supply Chain Management",
                    "Sustainability Initiatives"
                ]
            }
        }
"""
User profile and preference schemas for request/response validation.
"""

from datetime import datetime, time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field, field_validator


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    name: Optional[str] = Field(None, min_length=2, max_length=255)
    email: Optional[EmailStr] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Jane Doe",
                "email": "jane@example.com"
            }
        }


class UserProfile(BaseModel):
    """Complete user profile response."""
    id: int
    email: EmailStr
    name: str
    is_active: bool
    subscription_status: str
    created_at: datetime
    last_login: Optional[datetime] = None
    strategic_profile: Optional["StrategicProfileResponse"] = None
    focus_areas: List["FocusAreaResponse"] = []
    delivery_preferences: Optional["DeliveryPreferencesResponse"] = None
    
    class Config:
        from_attributes = True


class StrategicProfileCreate(BaseModel):
    """Schema for creating strategic profile."""
    industry: Optional[str] = Field(None, max_length=100)
    organization_type: Optional[str] = Field(None, max_length=100)
    role: Optional[str] = Field(None, max_length=100)
    strategic_goals: List[str] = Field(default_factory=list)
    organization_size: Optional[str] = Field(None, pattern="^(small|medium|large|enterprise)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "industry": "Healthcare",
                "organization_type": "Enterprise",
                "role": "Product Manager",
                "strategic_goals": ["AI Integration", "Market Expansion", "Regulatory Compliance"],
                "organization_size": "large"
            }
        }


class StrategicProfileUpdate(BaseModel):
    """Schema for updating strategic profile."""
    industry: Optional[str] = Field(None, max_length=100)
    organization_type: Optional[str] = Field(None, max_length=100)
    role: Optional[str] = Field(None, max_length=100)
    strategic_goals: Optional[List[str]] = None
    organization_size: Optional[str] = Field(None, pattern="^(small|medium|large|enterprise)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "industry": "Fintech",
                "role": "CEO",
                "strategic_goals": ["Digital Transformation", "Customer Acquisition"]
            }
        }


class StrategicProfileResponse(BaseModel):
    """Strategic profile response."""
    id: int
    user_id: int
    industry: Optional[str]
    organization_type: Optional[str]
    role: Optional[str]
    strategic_goals: List[str]
    organization_size: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class FocusAreaCreate(BaseModel):
    """Schema for creating focus area."""
    focus_area: str = Field(..., min_length=1, max_length=255)
    keywords: List[str] = Field(default_factory=list)
    priority: int = Field(default=3, ge=1, le=4)
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v: int) -> int:
        if v not in [1, 2, 3, 4]:
            raise ValueError('Priority must be 1 (low), 2 (medium), 3 (high), or 4 (critical)')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "focus_area": "AI in Healthcare",
                "keywords": ["machine learning", "medical AI", "diagnostic AI"],
                "priority": 4
            }
        }


class FocusAreaUpdate(BaseModel):
    """Schema for updating focus area."""
    focus_area: Optional[str] = Field(None, min_length=1, max_length=255)
    keywords: Optional[List[str]] = None
    priority: Optional[int] = Field(None, ge=1, le=4)
    
    class Config:
        json_schema_extra = {
            "example": {
                "keywords": ["deep learning", "neural networks"],
                "priority": 3
            }
        }


class FocusAreaResponse(BaseModel):
    """Focus area response."""
    id: int
    user_id: int
    focus_area: str
    keywords: List[str]
    priority: int
    priority_label: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class DeliverySchedule(BaseModel):
    """Delivery schedule configuration."""
    frequency: str = Field(default="daily", pattern="^(real_time|hourly|daily|weekly|monthly)$")
    delivery_time: time = Field(default=time(8, 0))
    timezone: str = Field(default="UTC", max_length=50)
    weekend_delivery: bool = Field(default=False)


class ContentPreferences(BaseModel):
    """Content preferences configuration."""
    max_articles_per_report: int = Field(default=10, ge=1, le=100)
    min_significance_level: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    content_format: str = Field(
        default="executive_summary",
        pattern="^(full|executive_summary|summary|bullet_points|headlines_only)$"
    )


class NotificationPreferences(BaseModel):
    """Notification preferences configuration."""
    email_enabled: bool = Field(default=True)
    urgent_alerts_enabled: bool = Field(default=True)
    digest_mode: bool = Field(default=True)


class DeliveryPreferencesUpdate(BaseModel):
    """Schema for updating delivery preferences."""
    delivery_schedule: Optional[DeliverySchedule] = None
    content_preferences: Optional[ContentPreferences] = None
    notification_preferences: Optional[NotificationPreferences] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "delivery_schedule": {
                    "frequency": "daily",
                    "delivery_time": "09:00:00",
                    "timezone": "America/New_York",
                    "weekend_delivery": False
                },
                "content_preferences": {
                    "max_articles_per_report": 15,
                    "min_significance_level": "medium",
                    "content_format": "executive_summary"
                },
                "notification_preferences": {
                    "email_enabled": True,
                    "urgent_alerts_enabled": True,
                    "digest_mode": True
                }
            }
        }


class DeliveryPreferencesResponse(BaseModel):
    """Delivery preferences response."""
    id: int
    user_id: int
    delivery_schedule: DeliverySchedule
    content_preferences: ContentPreferences
    notification_preferences: NotificationPreferences
    created_at: datetime
    updated_at: datetime
    
    @classmethod
    def from_orm_model(cls, model):
        """Create response from ORM model."""
        return cls(
            id=model.id,
            user_id=model.user_id,
            delivery_schedule=DeliverySchedule(
                frequency=model.frequency,
                delivery_time=model.delivery_time,
                timezone=model.timezone,
                weekend_delivery=model.weekend_delivery
            ),
            content_preferences=ContentPreferences(
                max_articles_per_report=model.max_articles_per_report,
                min_significance_level=model.min_significance_level,
                content_format=model.content_format
            ),
            notification_preferences=NotificationPreferences(
                email_enabled=model.email_enabled,
                urgent_alerts_enabled=model.urgent_alerts_enabled,
                digest_mode=model.digest_mode
            ),
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    class Config:
        from_attributes = True
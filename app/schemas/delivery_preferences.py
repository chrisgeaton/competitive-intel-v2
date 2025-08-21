"""
Delivery preferences schemas with validation for report delivery settings.
"""

from datetime import datetime, time
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, validator


class FrequencyType(str, Enum):
    """Report delivery frequency options."""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class SignificanceLevel(str, Enum):
    """Minimum significance level for content inclusion."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentFormat(str, Enum):
    """Content format options for reports."""
    FULL = "full"
    EXECUTIVE_SUMMARY = "executive_summary"
    SUMMARY = "summary"
    BULLET_POINTS = "bullet_points"
    HEADLINES_ONLY = "headlines_only"


class DeliveryPreferencesBase(BaseModel):
    """Base delivery preferences schema."""
    frequency: Optional[FrequencyType] = Field(
        default=FrequencyType.DAILY,
        description="How often to deliver reports"
    )
    delivery_time: Optional[str] = Field(
        default="08:00",
        description="Time of day to deliver reports (HH:MM format)"
    )
    timezone: Optional[str] = Field(
        default="UTC",
        max_length=50,
        description="Timezone for delivery scheduling"
    )
    weekend_delivery: Optional[bool] = Field(
        default=False,
        description="Whether to deliver reports on weekends"
    )
    max_articles_per_report: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of articles per report"
    )
    min_significance_level: Optional[SignificanceLevel] = Field(
        default=SignificanceLevel.MEDIUM,
        description="Minimum significance level for content inclusion"
    )
    content_format: Optional[ContentFormat] = Field(
        default=ContentFormat.EXECUTIVE_SUMMARY,
        description="Format for report content"
    )
    email_enabled: Optional[bool] = Field(
        default=True,
        description="Whether to send email notifications"
    )
    urgent_alerts_enabled: Optional[bool] = Field(
        default=True,
        description="Whether to send urgent alert notifications"
    )
    digest_mode: Optional[bool] = Field(
        default=True,
        description="Whether to use digest mode for multiple articles"
    )

    @validator('delivery_time')
    def validate_delivery_time(cls, v):
        """Validate delivery time format."""
        if v is None:
            return v
        
        try:
            # Parse HH:MM format
            time_parts = v.split(':')
            if len(time_parts) != 2:
                raise ValueError("Time must be in HH:MM format")
            
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            
            if not (0 <= hour <= 23):
                raise ValueError("Hour must be between 00 and 23")
            if not (0 <= minute <= 59):
                raise ValueError("Minute must be between 00 and 59")
            
            # Ensure zero-padding
            return f"{hour:02d}:{minute:02d}"
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid time format: {str(e)}")

    @validator('timezone')
    def validate_timezone(cls, v):
        """Validate timezone format."""
        if v is None:
            return v
        
        # Common timezone validation
        valid_timezones = [
            'UTC', 'GMT', 'EST', 'PST', 'CST', 'MST',
            'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
            'Europe/London', 'Europe/Paris', 'Europe/Berlin',
            'Asia/Tokyo', 'Asia/Shanghai', 'Asia/Kolkata',
            'Australia/Sydney', 'Australia/Melbourne'
        ]
        
        # Allow standard timezone formats
        if v in valid_timezones or v.startswith(('America/', 'Europe/', 'Asia/', 'Australia/', 'Africa/')):
            return v
        
        # Allow UTC offset format like UTC+5, UTC-8
        if v.startswith('UTC') and len(v) > 3:
            offset_part = v[3:]
            if offset_part[0] in ['+', '-'] and offset_part[1:].isdigit():
                offset = int(offset_part[1:])
                if 0 <= offset <= 12:
                    return v
        
        raise ValueError(f"Invalid timezone: {v}")


class DeliveryPreferencesCreate(DeliveryPreferencesBase):
    """Schema for creating delivery preferences."""
    
    class Config:
        schema_extra = {
            "example": {
                "frequency": "daily",
                "delivery_time": "08:00",
                "timezone": "America/New_York",
                "weekend_delivery": False,
                "max_articles_per_report": 15,
                "min_significance_level": "medium",
                "content_format": "executive_summary",
                "email_enabled": True,
                "urgent_alerts_enabled": True,
                "digest_mode": True
            }
        }


class DeliveryPreferencesUpdate(BaseModel):
    """Schema for updating delivery preferences."""
    frequency: Optional[FrequencyType] = None
    delivery_time: Optional[str] = None
    timezone: Optional[str] = None
    weekend_delivery: Optional[bool] = None
    max_articles_per_report: Optional[int] = Field(None, ge=1, le=50)
    min_significance_level: Optional[SignificanceLevel] = None
    content_format: Optional[ContentFormat] = None
    email_enabled: Optional[bool] = None
    urgent_alerts_enabled: Optional[bool] = None
    digest_mode: Optional[bool] = None

    @validator('delivery_time')
    def validate_delivery_time(cls, v):
        """Validate delivery time format."""
        if v is None:
            return v
        
        try:
            time_parts = v.split(':')
            if len(time_parts) != 2:
                raise ValueError("Time must be in HH:MM format")
            
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            
            if not (0 <= hour <= 23):
                raise ValueError("Hour must be between 00 and 23")
            if not (0 <= minute <= 59):
                raise ValueError("Minute must be between 00 and 59")
            
            return f"{hour:02d}:{minute:02d}"
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid time format: {str(e)}")

    @validator('timezone')
    def validate_timezone(cls, v):
        """Validate timezone format."""
        if v is None:
            return v
        
        valid_timezones = [
            'UTC', 'GMT', 'EST', 'PST', 'CST', 'MST',
            'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
            'Europe/London', 'Europe/Paris', 'Europe/Berlin',
            'Asia/Tokyo', 'Asia/Shanghai', 'Asia/Kolkata',
            'Australia/Sydney', 'Australia/Melbourne'
        ]
        
        if v in valid_timezones or v.startswith(('America/', 'Europe/', 'Asia/', 'Australia/', 'Africa/')):
            return v
        
        if v.startswith('UTC') and len(v) > 3:
            offset_part = v[3:]
            if offset_part[0] in ['+', '-'] and offset_part[1:].isdigit():
                offset = int(offset_part[1:])
                if 0 <= offset <= 12:
                    return v
        
        raise ValueError(f"Invalid timezone: {v}")

    class Config:
        schema_extra = {
            "example": {
                "frequency": "weekly",
                "delivery_time": "09:30",
                "weekend_delivery": True,
                "content_format": "bullet_points"
            }
        }


class DeliveryPreferencesResponse(BaseModel):
    """Schema for delivery preferences responses."""
    id: int
    user_id: int
    frequency: FrequencyType
    delivery_time: str = Field(description="Time in HH:MM format")
    timezone: str
    weekend_delivery: bool
    max_articles_per_report: int
    min_significance_level: SignificanceLevel
    content_format: ContentFormat
    email_enabled: bool
    urgent_alerts_enabled: bool
    digest_mode: bool
    created_at: datetime
    updated_at: datetime

    @validator('delivery_time', pre=True)
    def format_delivery_time(cls, v):
        """Convert time object to HH:MM string format."""
        if hasattr(v, 'strftime'):
            return v.strftime('%H:%M')
        elif isinstance(v, str):
            return v
        return str(v)

    @classmethod
    def from_orm_model(cls, model):
        """Create response from ORM model with proper type conversion."""
        # Convert delivery_time to string format before construction
        delivery_time_str = model.delivery_time.strftime('%H:%M') if hasattr(model.delivery_time, 'strftime') else str(model.delivery_time)
        
        # Create instance bypassing all validation
        instance = cls.__new__(cls)
        instance.id = model.id
        instance.user_id = model.user_id
        instance.frequency = model.frequency
        instance.delivery_time = delivery_time_str
        instance.timezone = model.timezone
        instance.weekend_delivery = model.weekend_delivery
        instance.max_articles_per_report = model.max_articles_per_report
        instance.min_significance_level = model.min_significance_level
        instance.content_format = model.content_format
        instance.email_enabled = model.email_enabled
        instance.urgent_alerts_enabled = model.urgent_alerts_enabled
        instance.digest_mode = model.digest_mode
        instance.created_at = model.created_at
        instance.updated_at = model.updated_at
        
        return instance

    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "user_id": 123,
                "frequency": "daily",
                "delivery_time": "08:00",
                "timezone": "America/New_York",
                "weekend_delivery": False,
                "max_articles_per_report": 15,
                "min_significance_level": "medium",
                "content_format": "executive_summary",
                "email_enabled": True,
                "urgent_alerts_enabled": True,
                "digest_mode": True,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z"
            }
        }


class DeliveryPreferencesAnalytics(BaseModel):
    """Analytics for delivery preferences."""
    delivery_schedule: str = Field(description="Next delivery schedule description")
    articles_this_week: int = Field(description="Articles delivered this week")
    avg_articles_per_report: float = Field(description="Average articles per report")
    urgent_alerts_count: int = Field(description="Urgent alerts sent this week")
    email_open_rate: Optional[float] = Field(description="Email open rate percentage")
    last_delivery: Optional[datetime] = Field(description="Last report delivery time")
    next_delivery: Optional[datetime] = Field(description="Next scheduled delivery")
    frequency_recommendations: List[str] = Field(description="Frequency optimization suggestions")

    class Config:
        schema_extra = {
            "example": {
                "delivery_schedule": "Daily at 8:00 AM EST (weekdays only)",
                "articles_this_week": 47,
                "avg_articles_per_report": 12.3,
                "urgent_alerts_count": 3,
                "email_open_rate": 78.5,
                "last_delivery": "2025-01-01T13:00:00Z",
                "next_delivery": "2025-01-02T13:00:00Z",
                "frequency_recommendations": [
                    "Consider weekly delivery to reduce email volume",
                    "Enable weekend delivery for breaking news"
                ]
            }
        }


class DeliveryPreferencesDefaults(BaseModel):
    """Default preferences based on user profile."""
    recommended_frequency: FrequencyType
    recommended_time: str
    recommended_timezone: str
    recommended_format: ContentFormat
    reasoning: List[str] = Field(description="Explanation for recommendations")

    class Config:
        schema_extra = {
            "example": {
                "recommended_frequency": "daily",
                "recommended_time": "08:00",
                "recommended_timezone": "America/New_York",
                "recommended_format": "executive_summary",
                "reasoning": [
                    "Daily frequency recommended for active professionals",
                    "Morning delivery for start-of-day briefing",
                    "Executive summary format for time-efficient reading"
                ]
            }
        }
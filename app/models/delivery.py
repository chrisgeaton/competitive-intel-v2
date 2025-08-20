"""
Delivery preferences model for the Competitive Intelligence v2 system.
"""

from datetime import datetime, time
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Time,
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base


class UserDeliveryPreferences(Base):
    """User's preferences for content delivery and notifications."""
    __tablename__ = 'user_delivery_preferences'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        unique=True,
        index=True
    )
    
    # Delivery schedule
    frequency: Mapped[str] = mapped_column(
        String(50),
        default='daily',
        nullable=False
    )
    delivery_time: Mapped[time] = mapped_column(
        Time,
        default=time(8, 0),
        nullable=False
    )
    timezone: Mapped[str] = mapped_column(
        String(50),
        default='UTC',
        nullable=False
    )
    weekend_delivery: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    
    # Content preferences
    max_articles_per_report: Mapped[int] = mapped_column(
        Integer,
        default=10,
        nullable=False
    )
    min_significance_level: Mapped[str] = mapped_column(
        String(50),
        default='medium',
        nullable=False
    )
    content_format: Mapped[str] = mapped_column(
        String(50),
        default='executive_summary',
        nullable=False
    )
    
    # Notification preferences
    email_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    urgent_alerts_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    digest_mode: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    
    user: Mapped["User"] = relationship(
        "User",
        back_populates="delivery_preferences",
        lazy="selectin"
    )
    
    VALID_FREQUENCIES = ['real_time', 'hourly', 'daily', 'weekly', 'monthly']
    VALID_SIGNIFICANCE_LEVELS = ['low', 'medium', 'high', 'critical']
    VALID_CONTENT_FORMATS = ['full', 'executive_summary', 'summary', 'bullet_points', 'headlines_only']
    
    def __repr__(self) -> str:
        return f"<UserDeliveryPreferences(id={self.id}, user_id={self.user_id}, frequency={self.frequency})>"
    
    
    def should_deliver_today(self, current_date: datetime) -> bool:
        """Check if content should be delivered on the given date."""
        if not self.weekend_delivery:
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() in [5, 6]:
                return False
        
        if self.frequency == 'real_time':
            return True
        elif self.frequency == 'hourly':
            return True
        elif self.frequency == 'daily':
            return True
        elif self.frequency == 'weekly':
            # Deliver on Mondays (weekday=0)
            return current_date.weekday() == 0
        elif self.frequency == 'monthly':
            # Deliver on the 1st of each month
            return current_date.day == 1
        
        return False
    
    def get_next_delivery_time(self, from_date: datetime) -> Optional[datetime]:
        """Calculate the next delivery time from the given date."""
        if self.frequency == 'real_time':
            return from_date
        
        # Start with the delivery time on the from_date
        next_delivery = datetime.combine(
            from_date.date(),
            self.delivery_time
        )
        
        # If we've already passed today's delivery time, move to next period
        if next_delivery <= from_date:
            if self.frequency == 'hourly':
                next_delivery = from_date.replace(
                    minute=self.delivery_time.minute,
                    second=0,
                    microsecond=0
                ) + datetime.timedelta(hours=1)
            elif self.frequency == 'daily':
                next_delivery += datetime.timedelta(days=1)
            elif self.frequency == 'weekly':
                next_delivery += datetime.timedelta(weeks=1)
            elif self.frequency == 'monthly':
                # Move to next month
                if next_delivery.month == 12:
                    next_delivery = next_delivery.replace(
                        year=next_delivery.year + 1,
                        month=1
                    )
                else:
                    next_delivery = next_delivery.replace(
                        month=next_delivery.month + 1
                    )
        
        # Skip weekends if necessary
        while not self.weekend_delivery and next_delivery.weekday() in [5, 6]:
            next_delivery += datetime.timedelta(days=1)
        
        return next_delivery
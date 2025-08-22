"""
User and authentication models for the Competitive Intelligence v2 system.
"""

from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, 
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    """User account model."""
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    subscription_status: Mapped[str] = mapped_column(
        String(50), 
        default='trial',
        nullable=False
    )
    
    sessions: Mapped[List["UserSession"]] = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    strategic_profile: Mapped[Optional["UserStrategicProfile"]] = relationship(
        "UserStrategicProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    focus_areas: Mapped[List["UserFocusArea"]] = relationship(
        "UserFocusArea",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    entity_tracking: Mapped[List["UserEntityTracking"]] = relationship(
        "UserEntityTracking",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    delivery_preferences: Mapped[Optional["UserDeliveryPreferences"]] = relationship(
        "UserDeliveryPreferences",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        "AnalysisResult",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    strategic_insights: Mapped[List["StrategicInsight"]] = relationship(
        "StrategicInsight",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    analysis_jobs: Mapped[List["AnalysisJob"]] = relationship(
        "AnalysisJob",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, name={self.name})>"
    
    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.is_active
    
    @property
    def is_anonymous(self) -> bool:
        """Check if user is anonymous."""
        return False
    
    def update_last_login(self):
        """Update the last login timestamp."""
        self.last_login = datetime.now(timezone.utc)


class UserSession(Base):
    """User session model for authentication tokens."""
    __tablename__ = 'user_sessions'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False
    )
    token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    
    user: Mapped["User"] = relationship(
        "User",
        back_populates="sessions",
        lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<UserSession(id={self.id}, user_id={self.user_id}, expires_at={self.expires_at})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if the session is valid."""
        return not self.is_expired and self.user.is_active
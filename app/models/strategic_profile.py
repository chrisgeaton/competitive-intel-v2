"""
Strategic profile and focus area models for the Competitive Intelligence v2 system.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey,
    ARRAY, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from app.database import Base


class UserStrategicProfile(Base):
    """User's strategic profile that drives intelligence personalization."""
    __tablename__ = 'user_strategic_profile'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        unique=True
    )
    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    organization_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    role: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    strategic_goals: Mapped[Optional[List[str]]] = mapped_column(
        PG_ARRAY(Text),
        nullable=True,
        default=list
    )
    organization_size: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
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
        back_populates="strategic_profile",
        lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<UserStrategicProfile(id={self.id}, user_id={self.user_id}, industry={self.industry}, role={self.role})>"
    


class UserFocusArea(Base):
    """Key focus areas that user wants intelligence on."""
    __tablename__ = 'user_focus_areas'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    focus_area: Mapped[str] = mapped_column(String(255), nullable=False)
    keywords: Mapped[Optional[List[str]]] = mapped_column(
        PG_ARRAY(Text),
        nullable=True,
        default=list
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    
    user: Mapped["User"] = relationship(
        "User",
        back_populates="focus_areas",
        lazy="selectin"
    )
    
    __table_args__ = (
        UniqueConstraint('user_id', 'focus_area', name='uq_user_focus_area'),
        Index('idx_user_focus_areas_priority', 'priority'),
    )
    
    def __repr__(self) -> str:
        return f"<UserFocusArea(id={self.id}, user_id={self.user_id}, focus_area={self.focus_area}, priority={self.priority})>"
    
    @property
    def priority_label(self) -> str:
        """Get priority as string."""
        priority_map = {1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}
        return priority_map.get(self.priority, 'medium')
    

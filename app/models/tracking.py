"""
Entity tracking models for the Competitive Intelligence v2 system.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, 
    ForeignKey, UniqueConstraint, Index, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB, ARRAY as PG_ARRAY
from app.database import Base


class TrackingEntity(Base):
    """Entities to track (competitors, organizations, topics, people, technologies)."""
    __tablename__ = 'tracking_entities'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    domain: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    
    user_tracking: Mapped[List["UserEntityTracking"]] = relationship(
        "UserEntityTracking",
        back_populates="entity",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    __table_args__ = (
        UniqueConstraint('name', 'entity_type', name='uq_entity_name_type'),
        Index('idx_tracking_entities_type', 'entity_type'),
        Index('idx_tracking_entities_industry', 'industry'),
    )
    
    VALID_ENTITY_TYPES = [
        'competitor',
        'organization',
        'topic',
        'person',
        'technology',
        'product',
        'market_segment',
        'regulatory_body'
    ]
    
    def __repr__(self) -> str:
        return f"<TrackingEntity(id={self.id}, name={self.name}, type={self.entity_type})>"
    


class UserEntityTracking(Base):
    """User's specific tracking preferences for entities."""
    __tablename__ = 'user_entity_tracking'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    entity_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('tracking_entities.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False
    )
    custom_keywords: Mapped[Optional[List[str]]] = mapped_column(
        PG_ARRAY(String),
        nullable=True,
        default=list
    )
    tracking_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    
    user: Mapped["User"] = relationship(
        "User",
        back_populates="entity_tracking",
        lazy="selectin"
    )
    
    entity: Mapped["TrackingEntity"] = relationship(
        "TrackingEntity",
        back_populates="user_tracking",
        lazy="selectin"
    )
    
    __table_args__ = (
        UniqueConstraint('user_id', 'entity_id', name='uq_user_entity'),
        Index('idx_user_entity_tracking_priority', 'priority'),
        Index('idx_user_entity_tracking_enabled', 'tracking_enabled'),
    )
    
    def __repr__(self) -> str:
        return f"<UserEntityTracking(id={self.id}, user_id={self.user_id}, entity_id={self.entity_id}, priority={self.priority})>"
    
    @property
    def priority_label(self) -> str:
        """Get priority as string."""
        priority_map = {1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}
        return priority_map.get(self.priority, 'medium')
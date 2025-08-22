"""
Base service class for Phase 4 services to eliminate code duplication.

Provides common functionality used across Report and Orchestration services
following established Phase 1-3 patterns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func

from app.database import get_db_session
from app.models.user import User
from app.models.strategic_profile import UserStrategicProfile, UserFocusArea
from app.models.tracking import UserEntityTracking  
from app.models.delivery import UserDeliveryPreferences
from app.utils.router_base import BaseRouterOperations


def cached(ttl: int = 300, key_prefix: str = "default"):
    """Simple cache decorator for base service (avoid circular import)."""
    def decorator(func):
        # For now, just pass through - full caching would be implemented with external cache
        return func
    return decorator


class BaseIntelligenceService:
    """
    Base class for intelligence services providing common functionality.
    
    Eliminates code duplication across Report and Orchestration services
    by providing shared methods for user context, database operations,
    and common intelligence processing patterns.
    """
    
    def __init__(self, service_name: str):
        self.base_ops = BaseRouterOperations(service_name)
        self.logger = logging.getLogger(service_name)
    
    @cached(ttl=600, key_prefix="user_context")
    async def get_user_strategic_context(
        self, 
        db: AsyncSession, 
        user_id: int
    ) -> Dict[str, Any]:
        """
        Get comprehensive user strategic context for personalization.
        
        Consolidated method used by both Report and Orchestration services
        to retrieve user profile, focus areas, and delivery preferences.
        """
        try:
            # Get user basic info
            user_query = select(User.name, User.email).where(User.id == user_id)
            user_result = await db.execute(user_query)
            user_row = user_result.fetchone()
            
            # Get strategic profile
            profile_query = select(UserStrategicProfile).where(
                UserStrategicProfile.user_id == user_id
            )
            profile_result = await db.execute(profile_query)
            profile = profile_result.scalar_one_or_none()
            
            # Get focus areas
            focus_query = select(UserFocusArea).where(
                UserFocusArea.user_id == user_id
            ).order_by(desc(UserFocusArea.priority))
            focus_result = await db.execute(focus_query)
            focus_areas = focus_result.scalars().all()
            
            # Get entity tracking
            entity_query = select(UserEntityTracking).where(
                UserEntityTracking.user_id == user_id
            ).order_by(desc(UserEntityTracking.priority))
            entity_result = await db.execute(entity_query)
            entities = entity_result.scalars().all()
            
            return {
                "user_name": user_row.name if user_row else "User",
                "user_email": user_row.email if user_row else "",
                "industry": profile.industry if profile else "Unknown",
                "role": profile.role if profile else "Unknown",
                "organization_type": profile.organization_type if profile else "Unknown",
                "strategic_goals": profile.strategic_goals if profile else [],
                "focus_areas": [
                    {
                        "focus_area": fa.focus_area,
                        "priority": fa.priority,
                        "keywords": fa.keywords
                    }
                    for fa in focus_areas
                ],
                "tracked_entities": [
                    {
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "priority": entity.priority,
                        "keywords": entity.keywords
                    }
                    for entity in entities
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user context for user {user_id}: {e}")
            return {
                "user_name": "User",
                "user_email": "",
                "industry": "Unknown",
                "role": "Unknown", 
                "organization_type": "Unknown",
                "strategic_goals": [],
                "focus_areas": [],
                "tracked_entities": []
            }
    
    async def get_user_delivery_preferences(
        self, 
        db: AsyncSession, 
        user_id: int
    ) -> Optional[UserDeliveryPreferences]:
        """
        Get user delivery preferences.
        
        Common method for retrieving user delivery configuration
        used by both services for email and scheduling decisions.
        """
        try:
            query = select(UserDeliveryPreferences).where(
                UserDeliveryPreferences.user_id == user_id
            )
            result = await db.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting delivery preferences for user {user_id}: {e}")
            return None
    
    def calculate_content_score(
        self, 
        strategic_alignment: float, 
        competitive_impact: float, 
        urgency_score: float,
        base_score: float = 0.0
    ) -> float:
        """
        Calculate composite content score using established scoring weights.
        
        Standardized scoring calculation used across services
        for consistent content prioritization.
        """
        # Weight factors based on Phase 3 Analysis Service patterns
        weights = {
            "strategic_alignment": 0.4,
            "competitive_impact": 0.3,
            "urgency": 0.2,
            "base": 0.1
        }
        
        composite_score = (
            (strategic_alignment * weights["strategic_alignment"]) +
            (competitive_impact * weights["competitive_impact"]) +
            (urgency_score * weights["urgency"]) +
            (base_score * weights["base"])
        )
        
        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, composite_score))
    
    def create_relevance_explanation(
        self, 
        matched_entities: List[str],
        matched_focus_areas: List[str],
        strategic_alignment: float,
        competitive_impact: float,
        urgency_score: float
    ) -> str:
        """
        Create standardized relevance explanation for content items.
        
        Unified explanation generation used by both Report and Orchestration
        services for consistent "why relevant" messaging.
        """
        explanations = []
        
        # Strategic alignment explanation
        if strategic_alignment >= 0.8:
            explanations.append("Excellent strategic alignment")
        elif strategic_alignment >= 0.6:
            explanations.append("Strong strategic alignment")
        elif strategic_alignment >= 0.4:
            explanations.append("Moderate strategic alignment")
        
        # Entity matching explanation
        if matched_entities:
            top_entities = matched_entities[:3]  # Top 3 entities
            explanations.append(f"Matches tracked entities: {', '.join(top_entities)}")
        
        # Focus area explanation
        if matched_focus_areas:
            top_areas = matched_focus_areas[:2]  # Top 2 areas
            explanations.append(f"Relevant to focus areas: {', '.join(top_areas)}")
        
        # Impact explanation
        if competitive_impact >= 0.7:
            explanations.append("High competitive impact")
        elif competitive_impact >= 0.5:
            explanations.append("Moderate competitive impact")
        
        # Urgency explanation
        if urgency_score >= 0.8:
            explanations.append("Time-sensitive information")
        elif urgency_score >= 0.6:
            explanations.append("Moderately time-sensitive")
        
        # Return formatted explanation or default
        if explanations:
            return "; ".join(explanations)
        else:
            return "Relevant to your strategic intelligence requirements"
    
    async def execute_with_error_handling(
        self,
        operation_name: str,
        operation_func,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with standardized error handling and logging.
        
        Common error handling pattern used across all intelligence services
        for consistent error reporting and recovery.
        """
        try:
            self.logger.info(f"Starting {operation_name}")
            start_time = datetime.utcnow()
            
            result = await operation_func(*args, **kwargs)
            
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"{operation_name} completed successfully in {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"{operation_name} failed: {str(e)}")
            raise
    
    def validate_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime,
        max_days: int = 30
    ) -> tuple[datetime, datetime]:
        """
        Validate and normalize date range for content queries.
        
        Common validation used by both services for consistent
        date range handling and performance optimization.
        """
        # Ensure start_date is before end_date
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        # Limit date range for performance
        days_diff = (end_date - start_date).days
        if days_diff > max_days:
            self.logger.warning(f"Date range limited to {max_days} days")
            start_date = end_date - timedelta(days=max_days)
        
        return start_date, end_date
    
    def create_performance_metrics(
        self,
        start_time: datetime,
        items_processed: int = 0,
        items_successful: int = 0,
        cost_cents: int = 0
    ) -> Dict[str, Any]:
        """
        Create standardized performance metrics.
        
        Common metrics structure used across services for 
        consistent performance tracking and reporting.
        """
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        return {
            "execution_time_seconds": round(total_time, 2),
            "items_processed": items_processed,
            "items_successful": items_successful,
            "success_rate": (items_successful / items_processed * 100) if items_processed > 0 else 0.0,
            "cost_cents": cost_cents,
            "throughput_per_second": round(items_processed / total_time, 2) if total_time > 0 else 0.0,
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat()
        }
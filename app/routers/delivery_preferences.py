"""
Delivery preferences management routes for the Competitive Intelligence v2 API.
"""

from datetime import datetime, time
from typing import Optional

from app.models.delivery import UserDeliveryPreferences
from app.schemas.delivery_preferences import (
    DeliveryPreferencesCreate,
    DeliveryPreferencesUpdate,
    DeliveryPreferencesResponse,
    DeliveryPreferencesAnalytics,
    DeliveryPreferencesDefaults,
    FrequencyType,
    ContentFormat
)
from app.utils.router_base import (
    logging, Dict, List, Any, APIRouter, Depends, status,
    AsyncSession, select, get_db_session, User, get_current_active_user,
    errors, db_handler, db_helpers, BaseRouterOperations, create_analytics_response
)

base_ops = BaseRouterOperations(__name__)

router = APIRouter(prefix="/api/v1/users/delivery-preferences", tags=["Delivery Preferences Management"])


def _convert_time_string_to_time(time_str: str) -> time:
    """Convert HH:MM string to time object."""
    hour, minute = map(int, time_str.split(':'))
    return time(hour, minute)


def _generate_analytics(preferences: UserDeliveryPreferences, user: User) -> DeliveryPreferencesAnalytics:
    """Generate analytics for delivery preferences."""
    # Calculate next delivery time
    next_delivery = None
    if preferences:
        next_delivery = preferences.get_next_delivery_time(datetime.utcnow())
    
    # Generate schedule description
    schedule_desc = "Not configured"
    if preferences:
        freq_desc = preferences.frequency.replace('_', ' ').title()
        time_desc = preferences.delivery_time.strftime('%I:%M %p')
        timezone_desc = preferences.timezone
        
        if preferences.frequency == 'real_time':
            schedule_desc = "Real-time notifications"
        elif preferences.frequency == 'hourly':
            schedule_desc = f"Hourly at :{preferences.delivery_time.minute:02d} minutes ({timezone_desc})"
        else:
            schedule_desc = f"{freq_desc} at {time_desc} {timezone_desc}"
            
        if not preferences.weekend_delivery and preferences.frequency not in ['real_time', 'hourly']:
            schedule_desc += " (weekdays only)"
    
    # Generate frequency recommendations
    recommendations = []
    if preferences:
        if preferences.frequency == 'real_time':
            recommendations.append("Consider daily digest to reduce notification volume")
        elif preferences.max_articles_per_report > 20:
            recommendations.append("Reduce max articles per report for better readability")
        elif preferences.content_format == 'full':
            recommendations.append("Try executive summary format for quicker reading")
        
        if not preferences.weekend_delivery:
            recommendations.append("Enable weekend delivery for breaking news coverage")
    else:
        recommendations.append("Set up delivery preferences to receive personalized reports")
    
    return DeliveryPreferencesAnalytics(
        delivery_schedule=schedule_desc,
        articles_this_week=0,  # Would be calculated from actual reports
        avg_articles_per_report=preferences.max_articles_per_report if preferences else 10,
        urgent_alerts_count=0,  # Would be calculated from actual alerts
        email_open_rate=None,  # Would be calculated from engagement data
        last_delivery=None,  # Would be from last report
        next_delivery=next_delivery,
        frequency_recommendations=recommendations
    )


def _generate_defaults(user: User) -> DeliveryPreferencesDefaults:
    """Generate default preferences based on user profile."""
    # Default recommendations
    freq = FrequencyType.DAILY
    delivery_time = "08:00"
    timezone = "UTC"
    content_format = ContentFormat.EXECUTIVE_SUMMARY
    reasoning = [
        "Daily frequency recommended for regular intelligence updates",
        "Morning delivery for start-of-day briefing",
        "Executive summary format for time-efficient reading"
    ]
    
    # Customize based on user profile if available
    if hasattr(user, 'strategic_profile') and user.strategic_profile:
        if user.strategic_profile.role in ['ceo', 'executive', 'president']:
            content_format = ContentFormat.EXECUTIVE_SUMMARY
            reasoning.append("Executive summary recommended for leadership roles")
        elif user.strategic_profile.role in ['analyst', 'researcher']:
            content_format = ContentFormat.FULL
            freq = FrequencyType.WEEKLY
            reasoning = [
                "Weekly frequency for detailed analysis roles",
                "Full content format for comprehensive research"
            ]
        elif user.strategic_profile.role in ['product_manager', 'marketing_manager']:
            freq = FrequencyType.DAILY
            reasoning.append("Daily updates recommended for fast-moving product roles")
    
    return DeliveryPreferencesDefaults(
        recommended_frequency=freq,
        recommended_time=delivery_time,
        recommended_timezone=timezone,
        recommended_format=content_format,
        reasoning=reasoning
    )


@router.get("/", response_model=DeliveryPreferencesResponse)
async def get_delivery_preferences(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Get user's delivery preferences."""
    async def _get_preferences_operation():
        # Get existing preferences
        existing = await db.scalar(
            select(UserDeliveryPreferences).where(
                UserDeliveryPreferences.user_id == current_user.id
            )
        )
        
        if not existing:
            raise errors.not_found("Delivery preferences not configured")
        
        # Return raw dict to bypass FastAPI response validation
        return {
            "id": existing.id,
            "user_id": existing.user_id,
            "frequency": existing.frequency,
            "delivery_time": existing.delivery_time.strftime('%H:%M') if hasattr(existing.delivery_time, 'strftime') else str(existing.delivery_time),
            "timezone": existing.timezone,
            "weekend_delivery": existing.weekend_delivery,
            "max_articles_per_report": existing.max_articles_per_report,
            "min_significance_level": existing.min_significance_level,
            "content_format": existing.content_format,
            "email_enabled": existing.email_enabled,
            "urgent_alerts_enabled": existing.urgent_alerts_enabled,
            "digest_mode": existing.digest_mode,
            "created_at": existing.created_at,
            "updated_at": existing.updated_at
        }
    
    return await db_handler.handle_db_operation(
        "get delivery preferences", _get_preferences_operation, db, rollback_on_error=False
    )


@router.put("/", response_model=DeliveryPreferencesResponse)
async def create_or_update_delivery_preferences(
    preferences_data: DeliveryPreferencesUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Create or update user's delivery preferences."""
    async def _update_preferences_operation():
        # Get existing preferences
        existing = await db.scalar(
            select(UserDeliveryPreferences).where(
                UserDeliveryPreferences.user_id == current_user.id
            )
        )
        
        if existing:
            # Update existing preferences
            update_dict = preferences_data.model_dump(exclude_unset=True)
            for field, value in update_dict.items():
                if value is not None:
                    if field == 'delivery_time' and isinstance(value, str):
                        # Convert time string to time object
                        value = _convert_time_string_to_time(value)
                    setattr(existing, field, value)
            
            await db_helpers.safe_commit(db, "delivery preferences update")
            await db.refresh(existing)
            
            logging.getLogger(__name__).info(f"Delivery preferences updated for user {current_user.email}")
            # Return raw dict to bypass FastAPI response validation
            return {
                "id": existing.id,
                "user_id": existing.user_id,
                "frequency": existing.frequency,
                "delivery_time": existing.delivery_time.strftime('%H:%M') if hasattr(existing.delivery_time, 'strftime') else str(existing.delivery_time),
                "timezone": existing.timezone,
                "weekend_delivery": existing.weekend_delivery,
                "max_articles_per_report": existing.max_articles_per_report,
                "min_significance_level": existing.min_significance_level,
                "content_format": existing.content_format,
                "email_enabled": existing.email_enabled,
                "urgent_alerts_enabled": existing.urgent_alerts_enabled,
                "digest_mode": existing.digest_mode,
                "created_at": existing.created_at,
                "updated_at": existing.updated_at
            }
        
        else:
            # Create new preferences with provided values
            create_data = DeliveryPreferencesCreate(**preferences_data.model_dump(exclude_unset=True))
            
            new_preferences = UserDeliveryPreferences(
                user_id=current_user.id,
                frequency=create_data.frequency,
                delivery_time=_convert_time_string_to_time(create_data.delivery_time),
                timezone=create_data.timezone,
                weekend_delivery=create_data.weekend_delivery,
                max_articles_per_report=create_data.max_articles_per_report,
                min_significance_level=create_data.min_significance_level,
                content_format=create_data.content_format,
                email_enabled=create_data.email_enabled,
                urgent_alerts_enabled=create_data.urgent_alerts_enabled,
                digest_mode=create_data.digest_mode
            )
            
            db.add(new_preferences)
            await db_helpers.safe_commit(db, "delivery preferences creation")
            await db.refresh(new_preferences)
            
            logging.getLogger(__name__).info(f"Delivery preferences created for user {current_user.email}")
            # Return raw dict to bypass FastAPI response validation
            return {
                "id": new_preferences.id,
                "user_id": new_preferences.user_id,
                "frequency": new_preferences.frequency,
                "delivery_time": new_preferences.delivery_time.strftime('%H:%M') if hasattr(new_preferences.delivery_time, 'strftime') else str(new_preferences.delivery_time),
                "timezone": new_preferences.timezone,
                "weekend_delivery": new_preferences.weekend_delivery,
                "max_articles_per_report": new_preferences.max_articles_per_report,
                "min_significance_level": new_preferences.min_significance_level,
                "content_format": new_preferences.content_format,
                "email_enabled": new_preferences.email_enabled,
                "urgent_alerts_enabled": new_preferences.urgent_alerts_enabled,
                "digest_mode": new_preferences.digest_mode,
                "created_at": new_preferences.created_at,
                "updated_at": new_preferences.updated_at
            }
    
    return await db_handler.handle_db_operation(
        "create or update delivery preferences", _update_preferences_operation, db
    )


@router.delete("/")
async def reset_delivery_preferences(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Reset delivery preferences to defaults."""
    async def _reset_preferences_operation():
        # Get existing preferences
        existing = await db.scalar(
            select(UserDeliveryPreferences).where(
                UserDeliveryPreferences.user_id == current_user.id
            )
        )
        
        if existing:
            await db_helpers.safe_delete(db, existing, "reset delivery preferences")
            logging.getLogger(__name__).info(f"Delivery preferences reset for user {current_user.email}")
        
        return {"message": "Delivery preferences reset to defaults"}
    
    return await db_handler.handle_db_operation(
        "reset delivery preferences", _reset_preferences_operation, db
    )


@router.get("/analytics", response_model=DeliveryPreferencesAnalytics)
async def get_delivery_analytics(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Get analytics about delivery preferences and usage."""
    async def _get_analytics_operation():
        # Get existing preferences
        preferences = await db.scalar(
            select(UserDeliveryPreferences).where(
                UserDeliveryPreferences.user_id == current_user.id
            )
        )
        
        return _generate_analytics(preferences, current_user)
    
    return await db_handler.handle_db_operation(
        "get delivery analytics", _get_analytics_operation, db, rollback_on_error=False
    )


@router.get("/defaults", response_model=DeliveryPreferencesDefaults)
async def get_recommended_defaults(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Get recommended default preferences based on user profile."""
    async def _get_defaults_operation():
        # Load user with strategic profile
        user = await db_helpers.get_user_by_id(
            db, current_user.id, load_relations=True, validate_exists=True
        )
        
        return _generate_defaults(user)
    
    return await db_handler.handle_db_operation(
        "get recommended defaults", _get_defaults_operation, db, rollback_on_error=False
    )


@router.post("/test-schedule")
async def test_delivery_schedule(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Test delivery schedule calculation for debugging."""
    async def _test_schedule_operation():
        # Get existing preferences
        preferences = await db.scalar(
            select(UserDeliveryPreferences).where(
                UserDeliveryPreferences.user_id == current_user.id
            )
        )
        
        if not preferences:
            raise errors.not_found("Delivery preferences not configured")
        
        now = datetime.utcnow()
        should_deliver_today = preferences.should_deliver_today(now)
        next_delivery = preferences.get_next_delivery_time(now)
        
        return {
            "current_time": now.isoformat(),
            "should_deliver_today": should_deliver_today,
            "next_delivery_time": next_delivery.isoformat() if next_delivery else None,
            "frequency": preferences.frequency,
            "delivery_time": preferences.delivery_time.strftime('%H:%M'),
            "timezone": preferences.timezone,
            "weekend_delivery": preferences.weekend_delivery
        }
    
    return await db_handler.handle_db_operation(
        "test delivery schedule", _test_schedule_operation, db, rollback_on_error=False
    )
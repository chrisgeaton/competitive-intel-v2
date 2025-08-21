"""
Router modules for the Competitive Intelligence v2 API endpoints.
"""

from .auth import router as auth_router
from .users import router as users_router
from .strategic_profile import router as strategic_profile_router
from .focus_areas import router as focus_areas_router
from .entity_tracking import router as entity_tracking_router
from .delivery_preferences import router as delivery_preferences_router

__all__ = ['auth_router', 'users_router', 'strategic_profile_router', 'focus_areas_router', 'entity_tracking_router', 'delivery_preferences_router']
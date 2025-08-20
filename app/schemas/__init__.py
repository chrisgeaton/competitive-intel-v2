"""
Pydantic schemas for request/response validation in the Competitive Intelligence v2 system.
"""

from .auth import (
    UserRegister,
    UserLogin,
    Token,
    TokenData,
    UserResponse,
    PasswordChange,
    SessionResponse
)
from .user import (
    UserUpdate,
    UserProfile,
    StrategicProfileCreate,
    StrategicProfileUpdate,
    StrategicProfileResponse,
    FocusAreaCreate,
    FocusAreaUpdate,
    FocusAreaResponse,
    DeliveryPreferencesUpdate,
    DeliveryPreferencesResponse
)

__all__ = [
    # Auth schemas
    'UserRegister',
    'UserLogin',
    'Token',
    'TokenData',
    'UserResponse',
    'PasswordChange',
    'SessionResponse',
    # User schemas
    'UserUpdate',
    'UserProfile',
    'StrategicProfileCreate',
    'StrategicProfileUpdate',
    'StrategicProfileResponse',
    'FocusAreaCreate',
    'FocusAreaUpdate',
    'FocusAreaResponse',
    'DeliveryPreferencesUpdate',
    'DeliveryPreferencesResponse'
]
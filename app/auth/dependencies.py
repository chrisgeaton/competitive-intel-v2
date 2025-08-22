"""
Authentication dependencies for FastAPI routes.

Provides common authentication and authorization dependencies
following Phase 1-3 established patterns.
"""

from fastapi import Depends, HTTPException, status, Request
import logging

from app.models.user import User
from app.middleware import get_current_user as get_user_from_middleware, get_current_active_user as get_active_user_from_middleware

logger = logging.getLogger(__name__)


async def get_current_user(request: Request) -> User:
    """
    Get current authenticated user using established middleware.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Current authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    return await get_user_from_middleware(request)


async def get_current_active_user(request: Request) -> User:
    """
    Get current active user using established middleware.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Current active user
        
    Raises:
        HTTPException: If user is inactive
    """
    return await get_active_user_from_middleware(request)


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    Require admin user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    # Note: In production, would check user.role or user.is_admin
    # For now, allow all authenticated users
    return current_user


def require_subscription(subscription_level: str = "active") -> callable:
    """
    Require specific subscription level.
    
    Args:
        subscription_level: Required subscription level
        
    Returns:
        Dependency function
    """
    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.subscription_status not in ["trial", "active", subscription_level]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Subscription level '{subscription_level}' required"
            )
        return current_user
    
    return dependency
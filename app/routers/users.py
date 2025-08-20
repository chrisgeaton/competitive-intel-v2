"""
User management routes for the Competitive Intelligence v2 API.
"""

import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import get_db_session
from app.auth import auth_service
from app.models.user import User
from app.models.strategic_profile import UserStrategicProfile, UserFocusArea
from app.models.delivery import UserDeliveryPreferences
from app.schemas.auth import UserResponse, PasswordChange
from app.schemas.user import (
    UserUpdate, UserProfile,
    StrategicProfileCreate, StrategicProfileUpdate, StrategicProfileResponse,
    FocusAreaCreate, FocusAreaUpdate, FocusAreaResponse,
    DeliveryPreferencesUpdate, DeliveryPreferencesResponse
)
from app.middleware import get_current_user, get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/users", tags=["User Management"])


@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user's complete profile information.
    
    Returns comprehensive user profile including:
    - Basic user information
    - Strategic profile and focus areas
    - Delivery preferences
    - Account status and subscription information
    """
    try:
        # Load user with all relationships
        result = await db.execute(
            select(User)
            .where(User.id == current_user.id)
            .options(
                # Load all related data
                selectinload(User.strategic_profile),
                selectinload(User.focus_areas),
                selectinload(User.delivery_preferences)
            )
        )
        user_with_relations = result.scalar_one()
        
        # Build profile response
        profile_data = {
            "id": user_with_relations.id,
            "email": user_with_relations.email,
            "name": user_with_relations.name,
            "is_active": user_with_relations.is_active,
            "subscription_status": user_with_relations.subscription_status,
            "created_at": user_with_relations.created_at,
            "last_login": user_with_relations.last_login,
            "strategic_profile": None,
            "focus_areas": [],
            "delivery_preferences": None
        }
        
        # Add strategic profile if exists
        if user_with_relations.strategic_profile:
            profile = user_with_relations.strategic_profile
            profile_data["strategic_profile"] = StrategicProfileResponse(
                id=profile.id,
                user_id=profile.user_id,
                industry=profile.industry,
                organization_type=profile.organization_type,
                role=profile.role,
                strategic_goals=profile.strategic_goals or [],
                organization_size=profile.organization_size,
                created_at=profile.created_at,
                updated_at=profile.updated_at
            )
        
        # Add focus areas
        for focus_area in user_with_relations.focus_areas:
            profile_data["focus_areas"].append(FocusAreaResponse(
                id=focus_area.id,
                user_id=focus_area.user_id,
                focus_area=focus_area.focus_area,
                keywords=focus_area.keywords or [],
                priority=focus_area.priority,
                priority_label=focus_area.priority_label,
                created_at=focus_area.created_at
            ))
        
        # Add delivery preferences if exists
        if user_with_relations.delivery_preferences:
            prefs = user_with_relations.delivery_preferences
            profile_data["delivery_preferences"] = DeliveryPreferencesResponse.from_orm_model(prefs)
        
        return UserProfile(**profile_data)
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )


@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update current user's basic profile information.
    
    Allows updating:
    - **name**: User's display name (2-255 characters)
    - **email**: Email address (must be unique)
    
    Note: Email changes may require verification in production systems.
    """
    try:
        # Check if email is being changed and already exists
        if user_data.email and user_data.email != current_user.email:
            result = await db.execute(
                select(User).where(User.email == user_data.email)
            )
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email address already in use"
                )
            
            current_user.email = user_data.email
        
        # Update name if provided
        if user_data.name:
            current_user.name = user_data.name
        
        await db.commit()
        await db.refresh(current_user)
        
        logger.info(f"Profile updated for user: {current_user.email}")
        
        return UserResponse.model_validate(current_user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Change user's password.
    
    Security features:
    - Validates current password before change
    - Enforces strong password requirements
    - Revokes all other sessions for security
    
    - **current_password**: User's current password for verification
    - **new_password**: New strong password meeting security requirements
    """
    try:
        # Verify current password
        if not auth_service.verify_password(password_data.current_password, current_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        new_hashed_password = auth_service.hash_password(password_data.new_password)
        
        # Update user password
        current_user.password_hash = new_hashed_password
        await db.commit()
        
        # Revoke all other sessions for security
        await auth_service.revoke_all_user_sessions(db, current_user.id)
        
        logger.info(f"Password changed for user: {current_user.email}")
        
        return {"message": "Password changed successfully. Please log in again on other devices."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.post("/strategic-profile", response_model=StrategicProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_strategic_profile(
    profile_data: StrategicProfileCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create user's strategic profile.
    
    Defines the user's business context for personalized intelligence:
    - **industry**: Business industry (e.g., "Healthcare", "Fintech")
    - **organization_type**: Type of organization (e.g., "Enterprise", "Startup")
    - **role**: User's role (e.g., "Product Manager", "CEO")
    - **strategic_goals**: List of key strategic objectives
    - **organization_size**: Size category (small, medium, large, enterprise)
    """
    try:
        # Check if profile already exists
        result = await db.execute(
            select(UserStrategicProfile).where(UserStrategicProfile.user_id == current_user.id)
        )
        existing_profile = result.scalar_one_or_none()
        
        if existing_profile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Strategic profile already exists. Use PUT to update."
            )
        
        # Create new strategic profile
        new_profile = UserStrategicProfile(
            user_id=current_user.id,
            industry=profile_data.industry,
            organization_type=profile_data.organization_type,
            role=profile_data.role,
            strategic_goals=profile_data.strategic_goals,
            organization_size=profile_data.organization_size
        )
        
        db.add(new_profile)
        await db.commit()
        await db.refresh(new_profile)
        
        logger.info(f"Strategic profile created for user: {current_user.email}")
        
        return StrategicProfileResponse.model_validate(new_profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating strategic profile: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create strategic profile"
        )


@router.put("/strategic-profile", response_model=StrategicProfileResponse)
async def update_strategic_profile(
    profile_data: StrategicProfileUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update user's strategic profile.
    
    Updates existing strategic profile with new business context information.
    Only provided fields will be updated.
    """
    try:
        # Get existing profile
        result = await db.execute(
            select(UserStrategicProfile).where(UserStrategicProfile.user_id == current_user.id)
        )
        profile = result.scalar_one_or_none()
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategic profile not found. Create one first."
            )
        
        # Update fields if provided
        if profile_data.industry is not None:
            profile.industry = profile_data.industry
        if profile_data.organization_type is not None:
            profile.organization_type = profile_data.organization_type
        if profile_data.role is not None:
            profile.role = profile_data.role
        if profile_data.strategic_goals is not None:
            profile.strategic_goals = profile_data.strategic_goals
        if profile_data.organization_size is not None:
            profile.organization_size = profile_data.organization_size
        
        profile.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(profile)
        
        logger.info(f"Strategic profile updated for user: {current_user.email}")
        
        return StrategicProfileResponse.model_validate(profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategic profile: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update strategic profile"
        )


@router.get("/strategic-profile", response_model=StrategicProfileResponse)
async def get_strategic_profile(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get user's strategic profile.
    
    Returns the user's business context and strategic objectives.
    """
    try:
        result = await db.execute(
            select(UserStrategicProfile).where(UserStrategicProfile.user_id == current_user.id)
        )
        profile = result.scalar_one_or_none()
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategic profile not found"
            )
        
        return StrategicProfileResponse.model_validate(profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategic profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get strategic profile"
        )


@router.delete("/account")
async def delete_user_account(
    password_confirmation: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete user account permanently.
    
    **WARNING**: This action is irreversible and will:
    - Delete all user data including profiles and preferences
    - Revoke all sessions immediately
    - Remove all associated records
    
    - **password_confirmation**: Current password for security verification
    """
    try:
        # Verify password for security
        if not auth_service.verify_password(password_confirmation, current_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password confirmation incorrect"
            )
        
        # Revoke all sessions first
        await auth_service.revoke_all_user_sessions(db, current_user.id)
        
        # Delete user (cascade will handle related records)
        await db.delete(current_user)
        await db.commit()
        
        logger.info(f"User account deleted: {current_user.email}")
        
        return {"message": "Account deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user account: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )
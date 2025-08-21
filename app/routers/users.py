"""
User management routes for the Competitive Intelligence v2 API.
"""

from datetime import datetime

from app.auth import auth_service
from app.models.strategic_profile import UserStrategicProfile
from app.schemas.auth import UserResponse, PasswordChange
from app.schemas.user import (
    UserUpdate, UserProfile,
    StrategicProfileCreate, StrategicProfileUpdate, StrategicProfileResponse,
    FocusAreaResponse, DeliveryPreferencesResponse
)
from app.utils.router_base import (
    logging, Dict, List, Any, APIRouter, Depends, status,
    AsyncSession, select, selectinload, get_db_session, User, get_current_active_user,
    errors, db_handler, db_helpers, validators, BaseRouterOperations
)

base_ops = BaseRouterOperations(__name__)

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
    async def _get_profile_operation():
        # Load user with all relationships
        user_with_relations = await db_helpers.get_user_by_id(
            db, current_user.id, load_relations=True, validate_exists=True
        )
        
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
    
    return await db_handler.handle_db_operation(
        "get user profile", _get_profile_operation, db, rollback_on_error=False
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
    async def _update_profile_operation():
        # Get fresh user instance in current session
        user = await db_helpers.get_user_by_id(
            db, current_user.id, validate_exists=True
        )
        
        # Check if email is being changed and already exists
        if user_data.email and user_data.email != user.email:
            await db_helpers.check_email_unique(db, user_data.email, user.id)
            user.email = user_data.email
        
        # Update name if provided
        if user_data.name:
            user.name = user_data.name
        
        await db_helpers.safe_commit(db, "profile update")
        await db.refresh(user)
        
        logger.info(f"Profile updated for user: {user.email}")
        return UserResponse.model_validate(user)
    
    return await db_handler.handle_db_operation(
        "update user profile", _update_profile_operation, db
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
    async def _change_password_operation():
        # Verify current password
        validators.validate_password_match(
            auth_service.verify_password(password_data.current_password, current_user.password_hash),
            "Current password is incorrect"
        )
        
        # Hash new password
        new_hashed_password = auth_service.hash_password(password_data.new_password)
        
        # Update user password
        current_user.password_hash = new_hashed_password
        await db_helpers.safe_commit(db, "password change")
        
        # Revoke all other sessions for security
        await auth_service.revoke_all_user_sessions(db, current_user.id)
        
        logger.info(f"Password changed for user: {current_user.email}")
        return {"message": "Password changed successfully. Please log in again on other devices."}
    
    return await db_handler.handle_db_operation(
        "change password", _change_password_operation, db
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
    async def _create_profile_operation():
        # Check if profile already exists
        existing_profile = await db_helpers.get_model_by_field(
            db, UserStrategicProfile, "user_id", current_user.id
        )
        
        if existing_profile:
            raise errors.conflict("Strategic profile already exists. Use PUT to update.")
        
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
        await db_helpers.safe_commit(db, "strategic profile creation")
        await db.refresh(new_profile)
        
        logger.info(f"Strategic profile created for user: {current_user.email}")
        return StrategicProfileResponse.model_validate(new_profile)
    
    return await db_handler.handle_db_operation(
        "create strategic profile", _create_profile_operation, db
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
    async def _update_profile_operation():
        # Get existing profile
        profile = await db_helpers.get_model_by_field(
            db, UserStrategicProfile, "user_id", current_user.id,
            validate_exists=True, resource_name="Strategic profile"
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
        
        await db_helpers.safe_commit(db, "strategic profile update")
        await db.refresh(profile)
        
        logger.info(f"Strategic profile updated for user: {current_user.email}")
        return StrategicProfileResponse.model_validate(profile)
    
    return await db_handler.handle_db_operation(
        "update strategic profile", _update_profile_operation, db
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
    async def _get_strategic_profile_operation():
        profile = await db_helpers.get_model_by_field(
            db, UserStrategicProfile, "user_id", current_user.id,
            validate_exists=True, resource_name="Strategic profile"
        )
        
        return StrategicProfileResponse.model_validate(profile)
    
    return await db_handler.handle_db_operation(
        "get strategic profile", _get_strategic_profile_operation, db, rollback_on_error=False
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
    async def _delete_account_operation():
        # Verify password for security
        validators.validate_password_match(
            auth_service.verify_password(password_confirmation, current_user.password_hash),
            "Password confirmation incorrect"
        )
        
        # Revoke all sessions first
        await auth_service.revoke_all_user_sessions(db, current_user.id)
        
        # Delete user (cascade will handle related records)
        await db_helpers.safe_delete(db, current_user, "delete user account")
        
        logger.info(f"User account deleted: {current_user.email}")
        return {"message": "Account deleted successfully"}
    
    return await db_handler.handle_db_operation(
        "delete user account", _delete_account_operation, db
    )
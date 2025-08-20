"""
Common database operation utilities.
"""

import logging
from typing import Optional, Type, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import User
from app.utils.exceptions import validators, errors

logger = logging.getLogger(__name__)


class DatabaseHelpers:
    """Common database operation helpers."""
    
    @staticmethod
    async def get_user_by_id(
        db: AsyncSession, 
        user_id: int, 
        load_relations: bool = False,
        validate_exists: bool = True,
        validate_active: bool = False
    ) -> Optional[User]:
        """
        Get user by ID with optional validation and relation loading.
        
        Args:
            db: Database session
            user_id: User ID to fetch
            load_relations: Whether to load user relationships
            validate_exists: Whether to raise exception if user not found
            validate_active: Whether to validate user is active
            
        Returns:
            User instance or None
            
        Raises:
            HTTPException: If validation fails
        """
        query = select(User).where(User.id == user_id)
        
        if load_relations:
            query = query.options(
                selectinload(User.strategic_profile),
                selectinload(User.focus_areas),
                selectinload(User.delivery_preferences)
            )
        
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if validate_exists:
            validators.validate_user_exists(user)
        
        if validate_active and user:
            validators.validate_user_active(user)
        
        return user
    
    @staticmethod
    async def get_user_by_email(
        db: AsyncSession, 
        email: str,
        validate_exists: bool = False,
        validate_active: bool = False
    ) -> Optional[User]:
        """
        Get user by email with optional validation.
        
        Args:
            db: Database session
            email: Email to search for
            validate_exists: Whether to raise exception if user not found
            validate_active: Whether to validate user is active
            
        Returns:
            User instance or None
            
        Raises:
            HTTPException: If validation fails
        """
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if validate_exists:
            validators.validate_user_exists(user, "User with this email")
        
        if validate_active and user:
            validators.validate_user_active(user)
        
        return user
    
    @staticmethod
    async def check_email_unique(db: AsyncSession, email: str, exclude_user_id: Optional[int] = None) -> None:
        """
        Check if email is unique (not already taken by another user).
        
        Args:
            db: Database session
            email: Email to check
            exclude_user_id: User ID to exclude from check (for updates)
            
        Raises:
            HTTPException: If email is already taken
        """
        query = select(User).where(User.email == email)
        
        if exclude_user_id:
            query = query.where(User.id != exclude_user_id)
        
        result = await db.execute(query)
        existing_user = result.scalar_one_or_none()
        
        validators.validate_unique_email(existing_user, email)
    
    @staticmethod
    async def get_model_by_field(
        db: AsyncSession,
        model_class: Type,
        field_name: str,
        field_value: Any,
        validate_exists: bool = False,
        resource_name: str = "Resource"
    ) -> Optional[Any]:
        """
        Generic function to get model by any field.
        
        Args:
            db: Database session
            model_class: SQLAlchemy model class
            field_name: Field name to filter by
            field_value: Value to filter by
            validate_exists: Whether to raise exception if not found
            resource_name: Name for error messages
            
        Returns:
            Model instance or None
            
        Raises:
            HTTPException: If validation fails
        """
        field = getattr(model_class, field_name)
        result = await db.execute(select(model_class).where(field == field_value))
        instance = result.scalar_one_or_none()
        
        if validate_exists and not instance:
            raise errors.not_found(resource_name)
        
        return instance
    
    @staticmethod
    async def safe_commit(db: AsyncSession, operation_name: str = "operation"):
        """
        Safely commit database transaction with error handling.
        
        Args:
            db: Database session
            operation_name: Name of operation for logging
            
        Raises:
            HTTPException: On commit failure
        """
        try:
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to commit {operation_name}: {e}")
            await db.rollback()
            raise errors.internal_error(f"Failed to save {operation_name}")
    
    @staticmethod
    async def safe_delete(db: AsyncSession, instance, operation_name: str = "delete"):
        """
        Safely delete instance with error handling.
        
        Args:
            db: Database session
            instance: Model instance to delete
            operation_name: Name of operation for logging
            
        Raises:
            HTTPException: On delete failure
        """
        try:
            await db.delete(instance)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to {operation_name}: {e}")
            await db.rollback()
            raise errors.internal_error(f"Failed to {operation_name}")


# Convenience instance
db_helpers = DatabaseHelpers()
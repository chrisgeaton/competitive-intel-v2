"""
Common exception utilities for consistent error handling.
"""

import logging
from typing import Optional, Any
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class ErrorHandlers:
    """Centralized error handling utilities."""
    
    @staticmethod
    def unauthorized(detail: str = "Not authenticated") -> HTTPException:
        """Create unauthorized exception."""
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail
        )
    
    @staticmethod
    def forbidden(detail: str = "Access forbidden") -> HTTPException:
        """Create forbidden exception."""
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )
    
    @staticmethod
    def not_found(resource: str = "Resource") -> HTTPException:
        """Create not found exception."""
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} not found"
        )
    
    @staticmethod
    def bad_request(detail: str = "Bad request") -> HTTPException:
        """Create bad request exception."""
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )
    
    @staticmethod
    def conflict(detail: str = "Resource already exists") -> HTTPException:
        """Create conflict exception."""
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail
        )
    
    @staticmethod
    def internal_error(detail: str = "Internal server error") -> HTTPException:
        """Create internal server error exception."""
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )
    
    @staticmethod
    def too_many_requests(detail: str = "Too many requests", retry_after: Optional[int] = None) -> HTTPException:
        """Create rate limit exception."""
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers=headers
        )
    
    @staticmethod
    def validation_error(detail: str = "Validation failed") -> HTTPException:
        """Create validation error exception."""
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )


class DatabaseErrorHandler:
    """Database operation error handling utilities."""
    
    @staticmethod
    async def handle_db_operation(
        operation_name: str,
        operation_func,
        db_session,
        rollback_on_error: bool = True,
        log_errors: bool = True
    ) -> Any:
        """
        Handle database operations with consistent error handling.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Async function to execute
            db_session: Database session
            rollback_on_error: Whether to rollback on error
            log_errors: Whether to log errors
            
        Returns:
            Result of operation_func
            
        Raises:
            HTTPException: On database errors
        """
        try:
            result = await operation_func()
            return result
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            if rollback_on_error and db_session:
                await db_session.rollback()
            raise
        except Exception as e:
            if log_errors:
                logger.error(f"Error in {operation_name}: {e}")
            if rollback_on_error and db_session:
                await db_session.rollback()
            raise ErrorHandlers.internal_error(f"Failed to {operation_name}")


class ValidationHelpers:
    """Common validation helper functions."""
    
    @staticmethod
    def validate_user_exists(user, user_identifier: str = "User"):
        """Validate that user exists."""
        if not user:
            raise ErrorHandlers.not_found(user_identifier)
        return user
    
    @staticmethod
    def validate_user_active(user):
        """Validate that user is active."""
        if not user.is_active:
            raise ErrorHandlers.bad_request("User account is inactive")
        return user
    
    @staticmethod
    def validate_unique_email(existing_user, new_email: str):
        """Validate email uniqueness."""
        if existing_user:
            raise ErrorHandlers.conflict("Email address already in use")
    
    @staticmethod
    def validate_password_match(is_valid: bool, detail: str = "Invalid credentials"):
        """Validate password match."""
        if not is_valid:
            raise ErrorHandlers.unauthorized(detail)


# Convenience instances
errors = ErrorHandlers()
db_handler = DatabaseErrorHandler()
validators = ValidationHelpers()
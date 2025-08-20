"""
Authentication routes for the User Config Service.
"""

import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db_session
from app.auth import auth_service
from app.models.user import User, UserSession
from app.schemas.auth import (
    UserRegister, UserLogin, Token, UserResponse,
    PasswordChange, SessionResponse
)
from app.middleware import get_current_user, get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Register a new user account.
    
    - **email**: Valid email address (will be used as username)
    - **name**: User's full name
    - **password**: Strong password (min 8 chars, uppercase, lowercase, digit, special char)
    """
    try:
        # Check if user already exists
        result = await db.execute(
            select(User).where(User.email == user_data.email)
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Hash password and create user
        hashed_password = auth_service.hash_password(user_data.password)
        
        new_user = User(
            email=user_data.email,
            name=user_data.name,
            password_hash=hashed_password,
            is_active=True,
            subscription_status="trial"
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        logger.info(f"New user registered: {new_user.email}")
        
        return UserResponse.model_validate(new_user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )


@router.post("/login", response_model=Token)
async def login_user(
    user_credentials: UserLogin,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Authenticate user and return access tokens.
    
    - **email**: User's email address
    - **password**: User's password
    - **remember_me**: Extend session duration (optional)
    """
    try:
        # Authenticate user
        user = await auth_service.authenticate_user(
            db,
            user_credentials.email,
            user_credentials.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Create session
        session = await auth_service.create_user_session(
            db, user, remember_me=user_credentials.remember_me
        )
        
        # Create JWT tokens
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "session_id": session.id,
            "scopes": ["user:read", "user:write"]
        }
        
        access_token = auth_service.create_access_token(token_data)
        refresh_token = auth_service.create_refresh_token(token_data)
        
        logger.info(f"User logged in: {user.email}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=3600,  # 1 hour
            refresh_token=refresh_token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/logout")
async def logout_user(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Logout user and revoke current session.
    """
    try:
        # Get session token from authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            token_data = auth_service.decode_token(token)
            
            if token_data and token_data.session_id:
                # Find and revoke the session
                result = await db.execute(
                    select(UserSession).where(UserSession.id == token_data.session_id)
                )
                session = result.scalar_one_or_none()
                
                if session:
                    await db.delete(session)
                    await db.commit()
        
        logger.info(f"User logged out: {current_user.email}")
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/logout-all")
async def logout_all_sessions(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Logout user from all devices/sessions.
    """
    try:
        # Get current session to exclude from revocation
        current_session_id = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            token_data = auth_service.decode_token(token)
            if token_data:
                current_session_id = token_data.session_id
        
        # Revoke all sessions except current
        current_session_token = None
        if current_session_id:
            result = await db.execute(
                select(UserSession.token).where(UserSession.id == current_session_id)
            )
            current_session_token = result.scalar_one_or_none()
        
        revoked_count = await auth_service.revoke_all_user_sessions(
            db, current_user.id, except_current=current_session_token
        )
        
        logger.info(f"User {current_user.email} logged out from {revoked_count} sessions")
        
        return {"message": f"Successfully logged out from {revoked_count} other sessions"}
        
    except Exception as e:
        logger.error(f"Error during logout-all: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout all failed"
        )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Change user's password.
    
    - **current_password**: User's current password
    - **new_password**: New strong password
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


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user's profile information.
    """
    return UserResponse.model_validate(current_user)


@router.get("/sessions", response_model=list[SessionResponse])
async def get_user_sessions(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get all active sessions for the current user.
    """
    try:
        result = await db.execute(
            select(UserSession)
            .where(UserSession.user_id == current_user.id)
            .order_by(UserSession.created_at.desc())
        )
        sessions = result.scalars().all()
        
        # Filter valid sessions and add validity info
        valid_sessions = []
        for session in sessions:
            if session.is_valid:
                session_response = SessionResponse.model_validate(session)
                session_response.is_valid = session.is_valid
                valid_sessions.append(session_response)
        
        return valid_sessions
        
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get sessions"
        )


@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: int,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Revoke a specific session by ID.
    """
    try:
        # Find the session
        result = await db.execute(
            select(UserSession).where(
                UserSession.id == session_id,
                UserSession.user_id == current_user.id
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Revoke the session
        await db.delete(session)
        await db.commit()
        
        logger.info(f"Session {session_id} revoked for user: {current_user.email}")
        
        return {"message": "Session revoked successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking session: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke session"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Refresh access token using refresh token.
    
    - **refresh_token**: Valid refresh token
    """
    try:
        # Decode refresh token
        token_data = auth_service.decode_token(refresh_token)
        
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Verify user exists and is active
        result = await db.execute(
            select(User).where(User.id == token_data.user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        new_token_data = {
            "sub": str(user.id),
            "email": user.email,
            "session_id": token_data.session_id,
            "scopes": token_data.scopes
        }
        
        new_access_token = auth_service.create_access_token(new_token_data)
        
        return Token(
            access_token=new_access_token,
            token_type="bearer",
            expires_in=3600,
            refresh_token=refresh_token  # Keep same refresh token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token"
        )
"""
Authentication routes for the Competitive Intelligence v2 API.
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

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Register a new user account.
    
    Creates a new user with email verification and strong password requirements.
    
    - **email**: Valid email address (will be used as username)
    - **name**: User's full name (2-255 characters)
    - **password**: Strong password (min 8 chars, uppercase, lowercase, digit, special char)
    
    Returns the created user profile without sensitive information.
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
    Authenticate user and return JWT access tokens.
    
    Validates credentials and creates secure session with JWT tokens.
    
    - **email**: User's registered email address
    - **password**: User's password
    - **remember_me**: Extend session duration to 30 days (optional, default: false)
    
    Returns access token, refresh token, and token metadata.
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
    
    Invalidates the current session token to prevent further use.
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


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Refresh access token using refresh token.
    
    Extends session by generating a new access token from a valid refresh token.
    
    - **refresh_token**: Valid refresh token from login response
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
"""
Authentication schemas for request/response validation.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, field_validator
import re


def validate_password(password: str) -> str:
    """Validate password meets security requirements."""
    if len(password) < 8:
        raise ValueError('Password must be at least 8 characters long')
    if not re.search(r'[A-Z]', password):
        raise ValueError('Password must contain at least one uppercase letter')
    if not re.search(r'[a-z]', password):
        raise ValueError('Password must contain at least one lowercase letter')
    if not re.search(r'\d', password):
        raise ValueError('Password must contain at least one digit')
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', password):
        raise ValueError('Password must contain at least one special character')
    return password


class UserRegister(BaseModel):
    """Schema for user registration request."""
    email: EmailStr = Field(..., description="User's email address")
    name: str = Field(..., min_length=2, max_length=255, description="User's full name")
    password: str = Field(..., min_length=8, max_length=100, description="User's password")
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        return validate_password(v)
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "name": "John Doe",
                "password": "SecureP@ss123!"
            }
        }


class UserLogin(BaseModel):
    """Schema for user login request."""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")
    remember_me: bool = Field(default=False, description="Extend session duration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecureP@ss123!",
                "remember_me": False
            }
        }


class Token(BaseModel):
    """Schema for authentication token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token for extending session")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class TokenData(BaseModel):
    """Schema for JWT token payload data."""
    user_id: int
    email: str
    session_id: Optional[int] = None
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None
    scopes: List[str] = Field(default_factory=list)


class UserResponse(BaseModel):
    """Schema for user information response."""
    id: int
    email: EmailStr
    name: str
    is_active: bool
    subscription_status: str
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "email": "user@example.com",
                "name": "John Doe",
                "is_active": True,
                "subscription_status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "last_login": "2024-01-15T10:30:00Z"
            }
        }


class PasswordChange(BaseModel):
    """Schema for password change request."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    
    @field_validator('new_password')
    @classmethod
    def validate_new_password_strength(cls, v: str) -> str:
        return validate_password(v)
    
    class Config:
        json_schema_extra = {
            "example": {
                "current_password": "OldP@ss123!",
                "new_password": "NewSecureP@ss456!"
            }
        }


class SessionResponse(BaseModel):
    """Schema for session information response."""
    id: int
    user_id: int
    token: str
    expires_at: datetime
    created_at: datetime
    is_valid: bool
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "user_id": 1,
                "token": "session_token_here",
                "expires_at": "2024-01-02T00:00:00Z",
                "created_at": "2024-01-01T00:00:00Z",
                "is_valid": True
            }
        }


class PasswordResetRequest(BaseModel):
    """Schema for password reset request."""
    email: EmailStr = Field(..., description="Email address to send reset link")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }


class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation."""
    reset_token: str = Field(..., description="Password reset token from email")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    
    @field_validator('new_password')
    @classmethod
    def validate_reset_password_strength(cls, v: str) -> str:
        return validate_password(v)
    
    class Config:
        json_schema_extra = {
            "example": {
                "reset_token": "reset_token_from_email",
                "new_password": "NewSecureP@ss789!"
            }
        }
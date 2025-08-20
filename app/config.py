"""
Configuration settings for the Competitive Intelligence v2 system.
"""

import os
import secrets
import warnings
from datetime import timedelta
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application settings
    APP_NAME: str = "Competitive Intelligence v2"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False)
    
    # Database settings
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://admin:yourpassword@localhost:5432/competitive_intelligence"
    )
    
    # JWT settings
    SECRET_KEY: str = Field(default="")
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)  # 1 hour
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)  # 7 days
    
    # Security settings
    BCRYPT_ROUNDS: int = Field(default=12)
    PASSWORD_MIN_LENGTH: int = Field(default=8)
    PASSWORD_MAX_LENGTH: int = Field(default=100)
    
    # Rate limiting settings
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=60)
    RATE_LIMIT_LOGIN_ATTEMPTS: int = Field(default=5)
    RATE_LIMIT_LOGIN_WINDOW_MINUTES: int = Field(default=15)
    
    # Session settings
    SESSION_EXPIRE_HOURS: int = Field(default=24)  # Default session
    SESSION_REMEMBER_ME_DAYS: int = Field(default=30)  # Extended session
    MAX_SESSIONS_PER_USER: int = Field(default=5)
    
    # CORS settings
    CORS_ORIGINS: list = Field(default=["http://localhost:3000", "http://localhost:8000"])
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: list = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    CORS_ALLOW_HEADERS: list = Field(default=["*"])
    
    # Email settings (for password reset, notifications)
    SMTP_HOST: Optional[str] = Field(default=None)
    SMTP_PORT: int = Field(default=587)
    SMTP_USERNAME: Optional[str] = Field(default=None)
    SMTP_PASSWORD: Optional[str] = Field(default=None)
    SMTP_FROM_EMAIL: str = Field(default="noreply@competitive-intel.com")
    SMTP_FROM_NAME: str = Field(default="Competitive Intelligence")
    
    # OpenAI settings (for future AI analysis)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4-turbo-preview")
    OPENAI_MAX_TOKENS: int = Field(default=2000)
    OPENAI_TEMPERATURE: float = Field(default=0.7)
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    @validator('SECRET_KEY')
    def validate_secret_key(cls, v):
        # If no SECRET_KEY provided, generate a secure one for development
        if not v or v == "your-secret-key-here-change-this-in-production":
            # Check if we're in production
            if os.getenv('ENVIRONMENT') == 'production':
                raise ValueError(
                    "SECRET_KEY must be set in production environment. "
                    "Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
                )
            
            # For development, generate a secure random key
            secure_key = secrets.token_urlsafe(64)
            warnings.warn(
                f"No SECRET_KEY provided. Using generated key for development: {secure_key[:32]}..."
                "\nFor production, set SECRET_KEY environment variable with a secure random key.",
                UserWarning
            )
            return secure_key
        
        # Validate key strength
        if len(v) < 32:
            raise ValueError(
                "SECRET_KEY must be at least 32 characters long for security. "
                "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
            )
        
        return v
    
    @validator('CORS_ORIGINS', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @property
    def access_token_expire_timedelta(self) -> timedelta:
        """Get access token expiration as timedelta."""
        return timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    @property
    def refresh_token_expire_timedelta(self) -> timedelta:
        """Get refresh token expiration as timedelta."""
        return timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)
    
    @property
    def session_expire_timedelta(self) -> timedelta:
        """Get default session expiration as timedelta."""
        return timedelta(hours=self.SESSION_EXPIRE_HOURS)
    
    @property
    def session_remember_me_timedelta(self) -> timedelta:
        """Get extended session expiration as timedelta."""
        return timedelta(days=self.SESSION_REMEMBER_ME_DAYS)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()


# Security headers configuration
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
}
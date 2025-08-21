"""
Middleware for authentication, security, and rate limiting.
"""

import time
import logging
from typing import Optional, Dict, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta

from fastapi import Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import select

from app.config import settings, SECURITY_HEADERS
from app.auth import auth_service
from app.models.user import User
from app.utils.exceptions import errors

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.login_attempts: Dict[str, deque] = defaultdict(deque)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (for proxy/load balancer setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _clean_old_requests(self, request_queue: deque, window_seconds: int):
        """Remove requests older than the time window."""
        cutoff_time = time.time() - window_seconds
        while request_queue and request_queue[0] < cutoff_time:
            request_queue.popleft()
    
    def _is_rate_limited(self, client_ip: str, endpoint: str) -> bool:
        """Check if client is rate limited for general requests."""
        if not settings.RATE_LIMIT_ENABLED:
            return False
        
        key = f"{client_ip}:{endpoint}"
        request_queue = self.requests[key]
        
        # Clean old requests
        self._clean_old_requests(request_queue, 60)  # 1 minute window
        
        # Check if limit exceeded
        if len(request_queue) >= settings.RATE_LIMIT_REQUESTS_PER_MINUTE:
            return True
        
        # Add current request
        request_queue.append(time.time())
        return False
    
    def _is_login_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited for login attempts."""
        if not settings.RATE_LIMIT_ENABLED:
            return False
        
        request_queue = self.login_attempts[client_ip]
        
        # Clean old attempts
        window_seconds = settings.RATE_LIMIT_LOGIN_WINDOW_MINUTES * 60
        self._clean_old_requests(request_queue, window_seconds)
        
        # Check if limit exceeded
        if len(request_queue) >= settings.RATE_LIMIT_LOGIN_ATTEMPTS:
            return True
        
        # Add current attempt
        request_queue.append(time.time())
        return False
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        endpoint = str(request.url.path)
        
        # Special handling for login endpoints
        if "/auth/login" in endpoint:
            if self._is_login_rate_limited(client_ip):
                logger.warning(f"Login rate limit exceeded for IP: {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Too many login attempts. Try again in {settings.RATE_LIMIT_LOGIN_WINDOW_MINUTES} minutes."
                    },
                    headers={"Retry-After": str(settings.RATE_LIMIT_LOGIN_WINDOW_MINUTES * 60)}
                )
        
        # General rate limiting
        elif self._is_rate_limited(client_ip, endpoint):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}, endpoint: {endpoint}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests. Please try again later."},
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)


class JWTBearer(HTTPBearer):
    """Custom JWT Bearer authentication."""
    
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        
        if credentials:
            if not credentials.scheme == "Bearer":
                raise errors.unauthorized("Invalid authentication scheme")
            
            token_data = auth_service.decode_token(credentials.credentials)
            if not token_data:
                raise errors.unauthorized("Invalid token or expired token")
            
            # Add token data to request state for use in endpoints
            request.state.token_data = token_data
            return credentials
        
        return None


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle authentication for protected routes."""
    
    # Routes that don't require authentication
    PUBLIC_ROUTES = {
        "/",
        "/docs",
        "/openapi.json",
        "/health",
        "/auth/register",
        "/auth/login",
        "/auth/reset-password",
        "/auth/confirm-reset",
        "/api/v1/strategic-profile/enums/industries",
        "/api/v1/strategic-profile/enums/organization-types",
        "/api/v1/strategic-profile/enums/roles",
        "/api/v1/strategic-profile/enums/strategic-goals",
        "/api/v1/strategic-profile/enums/organization-sizes"
    }
    
    # Routes that require authentication
    PROTECTED_PREFIXES = [
        "/api/v1/users",
        "/api/v1/profile",
        "/api/v1/strategic-profile",
        "/api/v1/focus-areas",
        "/api/v1/entities",
        "/api/v1/preferences"
    ]
    
    # Specific protected auth routes  
    PROTECTED_AUTH_ROUTES = {
        "/api/v1/auth/logout"
    }
    
    def _is_protected_route(self, path: str) -> bool:
        """Check if route requires authentication."""
        if path in self.PUBLIC_ROUTES:
            return False
        
        # Check specific protected auth routes
        if path in self.PROTECTED_AUTH_ROUTES:
            return True
        
        for prefix in self.PROTECTED_PREFIXES:
            if path.startswith(prefix):
                return True
        
        return False
    
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        if self._is_protected_route(path):
            # Extract token from Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Missing or invalid authorization header"}
                )
            
            token = auth_header.split(" ")[1]
            token_data = auth_service.decode_token(token)
            
            if not token_data:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or expired token"}
                )
            
            # Verify user exists and is active
            from app.database import db_manager
            async with db_manager.get_session() as db:
                try:
                    result = await db.execute(
                        select(User).where(User.id == token_data.user_id)
                    )
                    user = result.scalar_one_or_none()
                    
                    if not user or not user.is_active:
                        return JSONResponse(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            content={"detail": "User not found or inactive"}
                        )
                    
                    # Add user and token data to request state
                    request.state.current_user = user
                    request.state.token_data = token_data
                    
                except Exception as e:
                    logger.error(f"Error verifying user in auth middleware: {e}")
                    return JSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"detail": "Authentication service error"}
                    )
        
        return await call_next(request)


# Dependency functions for FastAPI endpoints
jwt_bearer = JWTBearer()


async def get_current_user(request: Request) -> User:
    """
    Dependency to get the current authenticated user.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If user not authenticated
    """
    if not hasattr(request.state, "current_user"):
        raise errors.unauthorized("Not authenticated")
    
    return request.state.current_user


async def get_current_active_user(request: Request) -> User:
    """
    Dependency to get the current authenticated and active user.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Current active user
        
    Raises:
        HTTPException: If user not authenticated or inactive
    """
    user = await get_current_user(request)
    
    if not user.is_active:
        raise errors.bad_request("Inactive user")
    
    return user


def get_token_data(request: Request) -> Optional[Any]:
    """
    Get token data from request state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Token data if available
    """
    return getattr(request.state, "token_data", None)
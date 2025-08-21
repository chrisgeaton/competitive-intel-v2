"""
Main FastAPI application for the User Config Service.
"""

import logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.database import init_db, close_db
from app.middleware import (
    SecurityHeadersMiddleware, 
    RateLimitMiddleware, 
    AuthenticationMiddleware
)
from app.routers import auth_router, users_router, strategic_profile_router, focus_areas_router, entity_tracking_router, delivery_preferences_router
from app.routers.discovery import router as discovery_router
from app.routers.analysis import router as analysis_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting User Config Service...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down User Config Service...")
    await close_db()
    logger.info("Database connections closed")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## Competitive Intelligence v2 - Complete Platform

A comprehensive competitive intelligence platform with ML-driven content discovery, user management, and personalized intelligence delivery.

### Features
- **Secure Authentication**: JWT-based authentication with refresh tokens
- **User Management**: Complete profile management and preferences
- **Strategic Profiles**: Business context for personalized intelligence
- **Discovery Service**: ML-driven content discovery with learning algorithms
- **Analysis Service**: Multi-stage AI analysis with strategic insights
- **Engagement Tracking**: SendGrid integration for ML learning
- **Content Scoring**: AI-powered relevance and credibility assessment
- **Deduplication**: Advanced content similarity detection
- **Session Management**: Multi-device session handling
- **Security**: Enterprise-grade security with rate limiting and validation

### Authentication
All protected endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

Get your token by using the `/api/v1/auth/login` endpoint.
    """,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    contact={
        "name": "Competitive Intelligence v2 Support",
        "email": "support@competitive-intel.com"
    },
    license_info={
        "name": "MIT License",
        "identifier": "MIT"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Add custom middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE)
app.add_middleware(AuthenticationMiddleware)

# Include routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(strategic_profile_router)
app.include_router(focus_areas_router)
app.include_router(entity_tracking_router)
app.include_router(delivery_preferences_router)
app.include_router(discovery_router)
app.include_router(analysis_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.database import db_manager
    
    try:
        db_healthy = await db_manager.health_check()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "database": "connected" if db_healthy else "disconnected",
            "timestamp": str(datetime.utcnow())
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "error",
                "error": str(e),
                "timestamp": str(datetime.utcnow())
            }
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "type": "validation_error"
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={
                "detail": "The requested resource was not found",
                "path": str(request.url.path),
                "type": "not_found"
            }
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "type": "http_error"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle internal server errors."""
    logger.error(f"Internal server error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "server_error",
            "timestamp": str(datetime.utcnow())
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
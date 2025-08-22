"""
Main FastAPI application for the User Config Service.
"""

# import logging
# from datetime import datetime
# from contextlib import asynccontextmanager
from fastapi import FastAPI  # , Request, status
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.exceptions import RequestValidationError
# from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
# from app.database import init_db, close_db
from app.middleware import (
    SecurityHeadersMiddleware, 
    RateLimitMiddleware, 
    AuthenticationMiddleware
)
from app.routers import auth_router, users_router, strategic_profile_router, focus_areas_router, entity_tracking_router, delivery_preferences_router
# Phase 2-4 routers
from app.routers.discovery import router as discovery_router
from app.routers.analysis import router as analysis_router
from app.routers.reports import router as reports_router
from app.routers.orchestration import router as orchestration_router

# Configure logging - COMMENTED OUT
# logging.basicConfig(
#     level=getattr(logging, settings.LOG_LEVEL),
#     format=settings.LOG_FORMAT
# )
# logger = logging.getLogger(__name__)


# LIFESPAN COMMENTED OUT FOR TESTING
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     pass


# Create FastAPI application
app = FastAPI(
    title="Test",  # settings.APP_NAME,
    version="1.0",  # settings.APP_VERSION,
    description="""
## Competitive Intelligence v2 - Complete Platform

A comprehensive competitive intelligence platform with ML-driven content discovery, user management, and personalized intelligence delivery.

### Features
- **Secure Authentication**: JWT-based authentication with refresh tokens
- **User Management**: Complete profile management and preferences
- **Strategic Profiles**: Business context for personalized intelligence
- **Discovery Service**: ML-driven content discovery with learning algorithms
- **Analysis Service**: Multi-stage AI analysis with strategic insights
- **Report Generation**: Priority-based strategic intelligence reports
- **Multi-Format Delivery**: SendGrid Email, API JSON, and Dashboard formats
- **Content Curation**: Deduplication with quality scoring preferences
- **Engagement Tracking**: SendGrid integration for ML learning
- **End-to-End Orchestration**: Complete Discovery → Analysis → Reports → Delivery pipeline
- **Session Management**: Multi-device session handling
- **Security**: Enterprise-grade security with rate limiting and validation

### Authentication
All protected endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

Get your token by using the `/api/v1/auth/login` endpoint.
    """,
    # lifespan=lifespan,
    docs_url="/docs",  # Always enable for development
    redoc_url="/redoc",  # Always enable for development
    contact={
        "name": "Competitive Intelligence v2 Support",
        "email": "support@competitive-intel.com"
    },
    license_info={
        "name": "MIT License",
        "identifier": "MIT"
    }
)

# Test 1: CORS only - no custom middleware
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE)
app.add_middleware(SecurityHeadersMiddleware)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(strategic_profile_router)
app.include_router(focus_areas_router)
app.include_router(entity_tracking_router)
app.include_router(delivery_preferences_router)
# Phase 2-4 routers
app.include_router(discovery_router)
app.include_router(analysis_router)
app.include_router(reports_router)
app.include_router(orchestration_router)


@app.get("/")
async def root():
    return {"message": "Competitive Intelligence v2 API", "status": "operational"}

@app.get("/test-cors")
async def test_cors_endpoint():
    return {"message": "CORS test successful", "timestamp": "2025-08-22T12:07:00Z"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}


# ALL EXCEPTION HANDLERS COMMENTED OUT FOR TESTING
# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     pass

# @app.exception_handler(StarletteHTTPException)
# async def http_exception_handler(request: Request, exc: StarletteHTTPException):
#     pass

# @app.exception_handler(500)
# async def internal_error_handler(request: Request, exc):
#     pass


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
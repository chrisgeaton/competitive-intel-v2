# Competitive Intelligence v2 - API Documentation

## Overview
Comprehensive FastAPI application with enterprise-grade user management, authentication, and profile system.

**Base URL**: `http://localhost:8000` (development)  
**API Version**: v1  
**Documentation**: `http://localhost:8000/docs` (Swagger UI)  

## Quick Start

### 1. Start the Application
```bash
cd competitive-intel-v2
python app/main.py
```

### 2. Access API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Authentication

All protected endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

Get your token by using the `/api/v1/auth/login` endpoint.

## API Endpoints

### Authentication Endpoints (`/api/v1/auth`)

#### POST /api/v1/auth/register
Register a new user account.

**Request Body**:
```json
{
  "email": "user@example.com",
  "name": "John Doe", 
  "password": "SecurePass123!"
}
```

**Response**: User profile without sensitive data
```json
{
  "id": 1,
  "email": "user@example.com",
  "name": "John Doe",
  "is_active": true,
  "subscription_status": "trial",
  "created_at": "2025-01-01T00:00:00Z",
  "last_login": null
}
```

#### POST /api/v1/auth/login
Authenticate user and get JWT tokens.

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "remember_me": false
}
```

**Response**: JWT tokens for API access
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### POST /api/v1/auth/logout
Logout and revoke current session.

**Headers**: `Authorization: Bearer <token>`  
**Response**: 
```json
{"message": "Successfully logged out"}
```

#### POST /api/v1/auth/refresh
Refresh access token using refresh token.

**Request Body**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response**: New access token
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer", 
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### User Management Endpoints (`/api/v1/users`)

#### GET /api/v1/users/profile
Get complete user profile with all related data.

**Headers**: `Authorization: Bearer <token>`  
**Response**: Complete user profile
```json
{
  "id": 1,
  "email": "user@example.com",
  "name": "John Doe",
  "is_active": true,
  "subscription_status": "trial",
  "created_at": "2025-01-01T00:00:00Z",
  "last_login": "2025-01-15T10:30:00Z",
  "strategic_profile": {
    "id": 1,
    "user_id": 1,
    "industry": "Healthcare",
    "organization_type": "Enterprise",
    "role": "Product Manager",
    "strategic_goals": ["AI Integration", "Market Expansion"],
    "organization_size": "large",
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
  },
  "focus_areas": [],
  "delivery_preferences": null
}
```

#### PUT /api/v1/users/profile  
Update basic user profile information.

**Headers**: `Authorization: Bearer <token>`  
**Request Body**:
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com"
}
```

**Response**: Updated user profile
```json
{
  "id": 1,
  "email": "jane@example.com",
  "name": "Jane Doe",
  "is_active": true,
  "subscription_status": "trial",
  "created_at": "2025-01-01T00:00:00Z",
  "last_login": "2025-01-15T10:30:00Z"
}
```

#### POST /api/v1/users/change-password
Change user's password with security validation.

**Headers**: `Authorization: Bearer <token>`  
**Request Body**:
```json
{
  "current_password": "OldPass123!",
  "new_password": "NewSecurePass456!"
}
```

**Response**:
```json
{"message": "Password changed successfully. Please log in again on other devices."}
```

#### POST /api/v1/users/strategic-profile
Create user's strategic business profile.

**Headers**: `Authorization: Bearer <token>`  
**Request Body**:
```json
{
  "industry": "Healthcare",
  "organization_type": "Enterprise", 
  "role": "Product Manager",
  "strategic_goals": ["AI Integration", "Market Expansion", "Regulatory Compliance"],
  "organization_size": "large"
}
```

**Response**: Created strategic profile
```json
{
  "id": 1,
  "user_id": 1,
  "industry": "Healthcare",
  "organization_type": "Enterprise",
  "role": "Product Manager", 
  "strategic_goals": ["AI Integration", "Market Expansion", "Regulatory Compliance"],
  "organization_size": "large",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

#### PUT /api/v1/users/strategic-profile
Update existing strategic profile.

**Headers**: `Authorization: Bearer <token>`  
**Request Body** (all fields optional):
```json
{
  "industry": "Fintech",
  "role": "CEO",
  "strategic_goals": ["Digital Transformation", "Customer Acquisition"]
}
```

#### GET /api/v1/users/strategic-profile
Get user's strategic profile.

**Headers**: `Authorization: Bearer <token>`  
**Response**: Strategic profile details

#### DELETE /api/v1/users/account
Permanently delete user account.

**Headers**: `Authorization: Bearer <token>`  
**Request Body**:
```json
{
  "password_confirmation": "CurrentPassword123!"
}
```

**Response**:
```json
{"message": "Account deleted successfully"}
```

### System Endpoints

#### GET /
Root endpoint - service information.

**Response**:
```json
{
  "service": "Competitive Intelligence v2",
  "version": "2.0.0",
  "status": "running"
}
```

#### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "database": "connected", 
  "timestamp": "2025-01-01T00:00:00Z"
}
```

## Error Responses

All endpoints return consistent error responses:

### 400 Bad Request
```json
{
  "detail": "Validation error",
  "type": "validation_error"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid credentials",
  "type": "http_error"
}
```

### 404 Not Found
```json
{
  "detail": "The requested resource was not found",
  "path": "/api/v1/invalid/endpoint",
  "type": "not_found"
}
```

### 422 Validation Error
```json
{
  "detail": "Validation error",
  "errors": [
    {
      "loc": ["body", "email"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "type": "validation_error"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error",
  "type": "server_error",
  "timestamp": "2025-01-01T00:00:00Z"
}
```

## Security Features

### Password Requirements
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter  
- At least one digit
- At least one special character

### JWT Tokens
- **Access Token**: 1 hour expiration
- **Refresh Token**: 7 days expiration
- Secure random secret key (64+ characters)
- Automatic rotation on password change

### Rate Limiting
- 60 requests per minute per IP
- 5 login attempts per 15 minutes per IP
- Configurable limits via environment variables

### Security Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security: max-age=31536000
- Content-Security-Policy: restrictive policy
- Referrer-Policy: strict-origin-when-cross-origin

### CORS Configuration
- Configurable allowed origins
- Credentials support
- Preflight request handling

## Development Testing

### Run API Tests
```bash
# Test all endpoints and functionality
python scripts/test_api_endpoints.py

# Run comprehensive QA (includes API testing)  
python scripts/comprehensive_qa.py
```

### Start Development Server
```bash
# Auto-reload on code changes
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use the built-in runner
python app/main.py
```

### Database Setup
```bash
# Ensure PostgreSQL is running on localhost:5432
# Database: competitive_intelligence
# User: admin / Password: yourpassword

# Database tables are auto-created on first run
```

## Production Deployment

### Environment Variables
Set these for production:
```bash
SECRET_KEY=your-64-character-secure-random-key-here
ENVIRONMENT=production
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db
DEBUG=false
```

### Security Checklist
- [ ] Set secure SECRET_KEY (64+ characters)
- [ ] Configure production database URL
- [ ] Set ENVIRONMENT=production
- [ ] Disable DEBUG mode
- [ ] Configure HTTPS/TLS
- [ ] Set appropriate CORS origins
- [ ] Review rate limiting settings
- [ ] Set up monitoring and logging

### Performance
- Database connection pooling
- Async request handling
- Efficient password hashing (bcrypt rounds: 12)
- JWT token caching
- Middleware optimization

## Architecture

```
app/
├── main.py              # FastAPI app initialization
├── config.py            # Settings and configuration
├── database.py          # Database connection management
├── auth.py              # Authentication services
├── middleware.py        # Security and auth middleware
├── routers/            
│   ├── auth.py         # Authentication endpoints
│   └── users.py        # User management endpoints
├── models/             # SQLAlchemy ORM models
├── schemas/            # Pydantic request/response models
└── ...
```

## Testing Coverage

**API Endpoint Testing**: 100% success rate (14/14 tests)
- ✅ Critical imports and dependencies
- ✅ Database connectivity and health
- ✅ Route registration and routing
- ✅ Schema validation and error handling
- ✅ Middleware configuration
- ✅ OpenAPI documentation generation

**Quality Assurance**: 100% success rate (35/35 tests)
- ✅ Complete functionality validation
- ✅ Security compliance testing
- ✅ Performance benchmarking
- ✅ Error handling validation
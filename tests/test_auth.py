"""
Authentication endpoint tests.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from tests.conftest import TestDataFactory


@pytest.mark.auth
class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    async def test_register_user_success(self, client: AsyncClient, test_factory: TestDataFactory):
        """Test successful user registration."""
        user_data = test_factory.create_user_data("newuser@example.com", "New User")
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["name"] == "New User"
        assert data["is_active"] == True
        assert data["subscription_status"] == "trial"
        assert "id" in data
        assert "created_at" in data
    
    async def test_register_user_duplicate_email(self, client: AsyncClient, test_user: User, test_factory: TestDataFactory):
        """Test registration with duplicate email."""
        user_data = test_factory.create_user_data(test_user.email, "Duplicate User")
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 409
        data = response.json()
        assert "already registered" in data["detail"].lower()
    
    async def test_register_user_invalid_email(self, client: AsyncClient, test_factory: TestDataFactory):
        """Test registration with invalid email."""
        user_data = test_factory.create_user_data("invalid-email", "Test User")
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "validation error" in data["detail"].lower()
    
    async def test_register_user_weak_password(self, client: AsyncClient, test_factory: TestDataFactory):
        """Test registration with weak password."""
        user_data = test_factory.create_user_data("test@example.com", "Test User")
        user_data["password"] = "weak"
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 422
    
    async def test_login_success(self, client: AsyncClient, test_user: User, test_factory: TestDataFactory):
        """Test successful login."""
        login_data = test_factory.create_login_data(test_user.email)
        
        response = await client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 3600
    
    async def test_login_invalid_email(self, client: AsyncClient, test_factory: TestDataFactory):
        """Test login with invalid email."""
        login_data = test_factory.create_login_data("nonexistent@example.com")
        
        response = await client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "invalid credentials" in data["detail"].lower()
    
    async def test_login_invalid_password(self, client: AsyncClient, test_user: User, test_factory: TestDataFactory):
        """Test login with invalid password."""
        login_data = test_factory.create_login_data(test_user.email, "wrongpassword")
        
        response = await client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "invalid credentials" in data["detail"].lower()
    
    async def test_refresh_token_success(self, client: AsyncClient, test_user: User, test_factory: TestDataFactory):
        """Test successful token refresh."""
        # First login to get tokens
        login_data = test_factory.create_login_data(test_user.email)
        login_response = await client.post("/api/v1/auth/login", json=login_data)
        
        assert login_response.status_code == 200
        login_data = login_response.json()
        refresh_token = login_data["refresh_token"]
        
        # Use refresh token
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    async def test_refresh_token_invalid(self, client: AsyncClient):
        """Test refresh with invalid token."""
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid_token"}
        )
        
        assert response.status_code == 401
    
    async def test_logout_success(self, client: AsyncClient, auth_headers: dict):
        """Test successful logout."""
        response = await client.post("/api/v1/auth/logout", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "logged out" in data["message"].lower()
    
    async def test_logout_without_auth(self, client: AsyncClient):
        """Test logout without authentication."""
        response = await client.post("/api/v1/auth/logout")
        
        assert response.status_code == 401
    
    async def test_me_endpoint_success(self, client: AsyncClient, auth_headers: dict, test_user: User):
        """Test /me endpoint with valid token."""
        response = await client.get("/api/v1/auth/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["name"] == test_user.name
        assert data["is_active"] == test_user.is_active
    
    async def test_me_endpoint_without_auth(self, client: AsyncClient):
        """Test /me endpoint without authentication."""
        response = await client.get("/api/v1/auth/me")
        
        assert response.status_code == 401
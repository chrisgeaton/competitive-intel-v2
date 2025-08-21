"""
User management endpoint tests.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


@pytest.mark.auth
class TestUserEndpoints:
    """Test user management endpoints."""
    
    async def test_get_profile_success(self, client: AsyncClient, auth_headers: dict, test_user: User):
        """Test getting user profile."""
        response = await client.get("/api/v1/users/profile", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["name"] == test_user.name
        assert data["is_active"] == test_user.is_active
        assert data["subscription_status"] == test_user.subscription_status
        assert "id" in data
        assert "created_at" in data
    
    async def test_get_profile_without_auth(self, client: AsyncClient):
        """Test getting profile without authentication."""
        response = await client.get("/api/v1/users/profile")
        
        assert response.status_code == 401
    
    async def test_update_profile_success(self, client: AsyncClient, auth_headers: dict, test_user: User):
        """Test updating user profile."""
        update_data = {
            "name": "Chris Eaton Updated",
            "subscription_status": "active"
        }
        
        response = await client.put("/api/v1/users/profile", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Chris Eaton Updated"
        assert data["subscription_status"] == "active"
        assert data["email"] == test_user.email
    
    async def test_update_profile_partial(self, client: AsyncClient, auth_headers: dict, test_user: User):
        """Test partial profile update."""
        update_data = {"name": "Chris E."}
        
        response = await client.put("/api/v1/users/profile", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Chris E."
        assert data["email"] == test_user.email
    
    async def test_update_profile_invalid_email(self, client: AsyncClient, auth_headers: dict):
        """Test updating profile with invalid email."""
        update_data = {"email": "invalid-email"}
        
        response = await client.put("/api/v1/users/profile", json=update_data, headers=auth_headers)
        
        assert response.status_code == 422
    
    async def test_update_profile_duplicate_email(self, client: AsyncClient, auth_headers: dict, second_test_user: User):
        """Test updating profile with duplicate email."""
        update_data = {"email": second_test_user.email}
        
        response = await client.put("/api/v1/users/profile", json=update_data, headers=auth_headers)
        
        assert response.status_code == 409
        data = response.json()
        assert "already in use" in data["detail"].lower()
    
    async def test_update_profile_without_auth(self, client: AsyncClient):
        """Test updating profile without authentication."""
        update_data = {"name": "Unauthorized User"}
        
        response = await client.put("/api/v1/users/profile", json=update_data)
        
        assert response.status_code == 401
    
    async def test_change_password_success(self, client: AsyncClient, auth_headers: dict):
        """Test successful password change."""
        password_data = {
            "current_password": "TestPassword123!",
            "new_password": "NewPassword123!",
            "confirm_password": "NewPassword123!"
        }
        
        response = await client.post("/api/v1/users/change-password", json=password_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "password changed" in data["message"].lower()
    
    async def test_change_password_invalid_current(self, client: AsyncClient, auth_headers: dict):
        """Test password change with invalid current password."""
        password_data = {
            "current_password": "WrongPassword123!",
            "new_password": "NewPassword123!",
            "confirm_password": "NewPassword123!"
        }
        
        response = await client.post("/api/v1/users/change-password", json=password_data, headers=auth_headers)
        
        assert response.status_code == 400
        data = response.json()
        assert "current password" in data["detail"].lower()
    
    async def test_change_password_mismatch(self, client: AsyncClient, auth_headers: dict):
        """Test password change with mismatched confirmation."""
        password_data = {
            "current_password": "TestPassword123!",
            "new_password": "NewPassword123!",
            "confirm_password": "DifferentPassword123!"
        }
        
        response = await client.post("/api/v1/users/change-password", json=password_data, headers=auth_headers)
        
        assert response.status_code == 400
        data = response.json()
        assert "do not match" in data["detail"].lower()
    
    async def test_change_password_weak_new(self, client: AsyncClient, auth_headers: dict):
        """Test password change with weak new password."""
        password_data = {
            "current_password": "TestPassword123!",
            "new_password": "weak",
            "confirm_password": "weak"
        }
        
        response = await client.post("/api/v1/users/change-password", json=password_data, headers=auth_headers)
        
        assert response.status_code == 422
    
    async def test_change_password_without_auth(self, client: AsyncClient):
        """Test password change without authentication."""
        password_data = {
            "current_password": "TestPassword123!",
            "new_password": "NewPassword123!",
            "confirm_password": "NewPassword123!"
        }
        
        response = await client.post("/api/v1/users/change-password", json=password_data)
        
        assert response.status_code == 401
    
    async def test_deactivate_account_success(self, client: AsyncClient, auth_headers: dict):
        """Test successful account deactivation."""
        response = await client.post("/api/v1/users/deactivate", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "deactivated" in data["message"].lower()
    
    async def test_deactivate_account_without_auth(self, client: AsyncClient):
        """Test account deactivation without authentication."""
        response = await client.post("/api/v1/users/deactivate")
        
        assert response.status_code == 401
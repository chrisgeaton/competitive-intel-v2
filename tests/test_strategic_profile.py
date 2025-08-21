"""
Strategic profile endpoint tests.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


@pytest.mark.auth
class TestStrategicProfileEndpoints:
    """Test strategic profile management endpoints."""
    
    async def test_create_strategic_profile_success(self, client: AsyncClient, auth_headers: dict, sample_strategic_profile_data: dict):
        """Test creating strategic profile."""
        response = await client.post("/api/v1/strategic-profile/", json=sample_strategic_profile_data, headers=auth_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["industry"] == sample_strategic_profile_data["industry"]
        assert data["organization_type"] == sample_strategic_profile_data["organization_type"]
        assert data["role"] == sample_strategic_profile_data["role"]
        assert data["strategic_goals"] == sample_strategic_profile_data["strategic_goals"]
        assert data["organization_size"] == sample_strategic_profile_data["organization_size"]
        assert "user_id" in data
        assert "created_at" in data
    
    async def test_create_strategic_profile_duplicate(self, client: AsyncClient, auth_headers: dict, test_user_with_profile: User, sample_strategic_profile_data: dict):
        """Test creating duplicate strategic profile."""
        response = await client.post("/api/v1/strategic-profile/", json=sample_strategic_profile_data, headers=auth_headers)
        
        assert response.status_code == 409
        data = response.json()
        assert "already exists" in data["detail"].lower()
    
    async def test_create_strategic_profile_invalid_industry(self, client: AsyncClient, auth_headers: dict, sample_strategic_profile_data: dict):
        """Test creating profile with invalid industry."""
        sample_strategic_profile_data["industry"] = "invalid_industry"
        
        response = await client.post("/api/v1/strategic-profile/", json=sample_strategic_profile_data, headers=auth_headers)
        
        assert response.status_code == 422
    
    async def test_get_strategic_profile_success(self, client: AsyncClient, auth_headers: dict, test_user_with_profile: User):
        """Test getting strategic profile."""
        response = await client.get("/api/v1/strategic-profile/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["industry"] == "technology"
        assert data["role"] == "ceo"
        assert "strategic_goals" in data
        assert "user_id" in data
    
    async def test_get_strategic_profile_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test getting non-existent strategic profile."""
        response = await client.get("/api/v1/strategic-profile/", headers=auth_headers)
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    async def test_update_strategic_profile_success(self, client: AsyncClient, auth_headers: dict, test_user_with_profile: User):
        """Test updating strategic profile."""
        update_data = {
            "industry": "healthcare",
            "role": "product_manager",
            "strategic_goals": ["ai_integration", "compliance"]
        }
        
        response = await client.put("/api/v1/strategic-profile/", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["industry"] == "healthcare"
        assert data["role"] == "product_manager"
        assert data["strategic_goals"] == ["ai_integration", "compliance"]
    
    async def test_update_strategic_profile_partial(self, client: AsyncClient, auth_headers: dict, test_user_with_profile: User):
        """Test partial strategic profile update."""
        update_data = {"role": "cto"}
        
        response = await client.put("/api/v1/strategic-profile/", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "cto"
        assert data["industry"] == "technology"  # Should remain unchanged
    
    async def test_delete_strategic_profile_success(self, client: AsyncClient, auth_headers: dict, test_user_with_profile: User):
        """Test deleting strategic profile."""
        response = await client.delete("/api/v1/strategic-profile/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "deleted" in data["message"].lower()
    
    async def test_get_industries_enum(self, client: AsyncClient):
        """Test getting industries enum (public endpoint)."""
        response = await client.get("/api/v1/strategic-profile/enums/industries")
        
        assert response.status_code == 200
        data = response.json()
        assert "industries" in data
        assert "healthcare" in data["industries"]
        assert "technology" in data["industries"]
    
    async def test_get_roles_enum(self, client: AsyncClient):
        """Test getting roles enum (public endpoint)."""
        response = await client.get("/api/v1/strategic-profile/enums/roles")
        
        assert response.status_code == 200
        data = response.json()
        assert "roles" in data
        assert "ceo" in data["roles"]
        assert "product_manager" in data["roles"]
    
    async def test_get_strategic_goals_enum(self, client: AsyncClient):
        """Test getting strategic goals enum (public endpoint)."""
        response = await client.get("/api/v1/strategic-profile/enums/strategic-goals")
        
        assert response.status_code == 200
        data = response.json()
        assert "strategic_goals" in data
        assert len(data["strategic_goals"]) > 0
    
    async def test_get_organization_types_enum(self, client: AsyncClient):
        """Test getting organization types enum (public endpoint)."""
        response = await client.get("/api/v1/strategic-profile/enums/organization-types")
        
        assert response.status_code == 200
        data = response.json()
        assert "organization_types" in data
        assert "startup" in data["organization_types"]
        assert "enterprise" in data["organization_types"]
    
    async def test_get_organization_sizes_enum(self, client: AsyncClient):
        """Test getting organization sizes enum (public endpoint)."""
        response = await client.get("/api/v1/strategic-profile/enums/organization-sizes")
        
        assert response.status_code == 200
        data = response.json()
        assert "organization_sizes" in data
        assert "small" in data["organization_sizes"]
        assert "large" in data["organization_sizes"]
    
    async def test_get_analytics_with_profile(self, client: AsyncClient, auth_headers: dict, test_user_with_profile: User):
        """Test getting analytics with existing profile."""
        response = await client.get("/api/v1/strategic-profile/analytics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "profile_completeness" in data
        assert "industry_insights" in data
        assert "role_recommendations" in data
        assert data["profile_completeness"] > 0
    
    async def test_get_stats_with_profile(self, client: AsyncClient, auth_headers: dict, test_user_with_profile: User):
        """Test getting stats with existing profile."""
        response = await client.get("/api/v1/strategic-profile/stats", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "total_users_with_profiles" in data
        assert "industry_distribution" in data
        assert "role_distribution" in data
        assert "popular_goals" in data
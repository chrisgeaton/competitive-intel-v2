"""
Focus areas endpoint tests.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


@pytest.mark.auth
class TestFocusAreasEndpoints:
    """Test focus areas management endpoints."""
    
    async def test_create_focus_area_success(self, client: AsyncClient, auth_headers: dict, sample_focus_area_data: dict):
        """Test creating focus area."""
        response = await client.post("/api/v1/users/focus-areas/", json=sample_focus_area_data, headers=auth_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["focus_area"] == sample_focus_area_data["focus_area"]
        assert data["keywords"] == sample_focus_area_data["keywords"]
        assert data["priority"] == sample_focus_area_data["priority"]
        assert data["priority_label"] == "high"
        assert "id" in data
        assert "user_id" in data
        assert "created_at" in data
    
    async def test_create_focus_area_duplicate(self, client: AsyncClient, auth_headers: dict, sample_focus_area_data: dict):
        """Test creating duplicate focus area."""
        # Create first focus area
        await client.post("/api/v1/users/focus-areas/", json=sample_focus_area_data, headers=auth_headers)
        
        # Try to create duplicate
        response = await client.post("/api/v1/users/focus-areas/", json=sample_focus_area_data, headers=auth_headers)
        
        assert response.status_code == 409
        data = response.json()
        assert "already exists" in data["detail"].lower()
    
    async def test_create_focus_area_invalid_priority(self, client: AsyncClient, auth_headers: dict, sample_focus_area_data: dict):
        """Test creating focus area with invalid priority."""
        sample_focus_area_data["priority"] = 5  # Invalid priority (max is 4)
        
        response = await client.post("/api/v1/users/focus-areas/", json=sample_focus_area_data, headers=auth_headers)
        
        assert response.status_code == 422
    
    async def test_create_focus_area_too_many_keywords(self, client: AsyncClient, auth_headers: dict, sample_focus_area_data: dict):
        """Test creating focus area with too many keywords."""
        sample_focus_area_data["keywords"] = [f"keyword{i}" for i in range(25)]  # Max is 20
        
        response = await client.post("/api/v1/users/focus-areas/", json=sample_focus_area_data, headers=auth_headers)
        
        assert response.status_code == 422
    
    async def test_get_focus_areas_empty(self, client: AsyncClient, auth_headers: dict):
        """Test getting focus areas when none exist."""
        response = await client.get("/api/v1/users/focus-areas/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["per_page"] == 10
        assert data["pages"] == 0
    
    async def test_get_focus_areas_with_data(self, client: AsyncClient, auth_headers: dict, sample_focus_area_data: dict):
        """Test getting focus areas with data."""
        # Create a focus area first
        await client.post("/api/v1/users/focus-areas/", json=sample_focus_area_data, headers=auth_headers)
        
        response = await client.get("/api/v1/users/focus-areas/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 1
        assert data["items"][0]["focus_area"] == sample_focus_area_data["focus_area"]
    
    async def test_get_focus_areas_pagination(self, client: AsyncClient, auth_headers: dict):
        """Test focus areas pagination."""
        # Create multiple focus areas
        for i in range(15):
            focus_area_data = {
                "focus_area": f"Focus Area {i}",
                "keywords": [f"keyword{i}"],
                "priority": (i % 4) + 1
            }
            await client.post("/api/v1/users/focus-areas/", json=focus_area_data, headers=auth_headers)
        
        # Test first page
        response = await client.get("/api/v1/users/focus-areas/?page=1&per_page=10", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 10
        assert data["total"] == 15
        assert data["pages"] == 2
        
        # Test second page
        response = await client.get("/api/v1/users/focus-areas/?page=2&per_page=10", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 5
    
    async def test_get_focus_areas_filter_by_priority(self, client: AsyncClient, auth_headers: dict):
        """Test filtering focus areas by priority."""
        # Create focus areas with different priorities
        for priority in [1, 2, 3, 4]:
            focus_area_data = {
                "focus_area": f"Priority {priority} Area",
                "keywords": [f"priority{priority}"],
                "priority": priority
            }
            await client.post("/api/v1/users/focus-areas/", json=focus_area_data, headers=auth_headers)
        
        # Filter by high priority (3)
        response = await client.get("/api/v1/users/focus-areas/?priority=3", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["priority"] == 3
    
    async def test_get_focus_areas_search(self, client: AsyncClient, auth_headers: dict):
        """Test searching focus areas."""
        # Create focus areas
        focus_areas = [
            {"focus_area": "Machine Learning Research", "keywords": ["ml"], "priority": 3},
            {"focus_area": "Cybersecurity Threats", "keywords": ["security"], "priority": 4},
            {"focus_area": "Cloud Computing", "keywords": ["cloud"], "priority": 2}
        ]
        
        for area in focus_areas:
            await client.post("/api/v1/users/focus-areas/", json=area, headers=auth_headers)
        
        # Search for "machine"
        response = await client.get("/api/v1/users/focus-areas/?search=machine", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert "machine" in data["items"][0]["focus_area"].lower()
    
    async def test_get_focus_area_by_id_success(self, client: AsyncClient, auth_headers: dict, sample_focus_area_data: dict):
        """Test getting specific focus area by ID."""
        # Create focus area
        create_response = await client.post("/api/v1/users/focus-areas/", json=sample_focus_area_data, headers=auth_headers)
        created_area = create_response.json()
        
        # Get by ID
        response = await client.get(f"/api/v1/users/focus-areas/{created_area['id']}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == created_area["id"]
        assert data["focus_area"] == sample_focus_area_data["focus_area"]
    
    async def test_get_focus_area_by_id_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test getting non-existent focus area by ID."""
        response = await client.get("/api/v1/users/focus-areas/999", headers=auth_headers)
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    async def test_update_focus_area_success(self, client: AsyncClient, auth_headers: dict, sample_focus_area_data: dict):
        """Test updating focus area."""
        # Create focus area
        create_response = await client.post("/api/v1/users/focus-areas/", json=sample_focus_area_data, headers=auth_headers)
        created_area = create_response.json()
        
        # Update focus area
        update_data = {
            "focus_area": "Updated AI Research",
            "priority": 4,
            "keywords": ["ai", "research", "updated"]
        }
        
        response = await client.put(f"/api/v1/users/focus-areas/{created_area['id']}", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["focus_area"] == "Updated AI Research"
        assert data["priority"] == 4
        assert data["keywords"] == ["ai", "research", "updated"]
    
    async def test_update_focus_area_partial(self, client: AsyncClient, auth_headers: dict, sample_focus_area_data: dict):
        """Test partial focus area update."""
        # Create focus area
        create_response = await client.post("/api/v1/users/focus-areas/", json=sample_focus_area_data, headers=auth_headers)
        created_area = create_response.json()
        
        # Partial update
        update_data = {"priority": 1}
        
        response = await client.put(f"/api/v1/users/focus-areas/{created_area['id']}", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["priority"] == 1
        assert data["focus_area"] == sample_focus_area_data["focus_area"]  # Should remain unchanged
    
    async def test_delete_focus_area_success(self, client: AsyncClient, auth_headers: dict, sample_focus_area_data: dict):
        """Test deleting focus area."""
        # Create focus area
        create_response = await client.post("/api/v1/users/focus-areas/", json=sample_focus_area_data, headers=auth_headers)
        created_area = create_response.json()
        
        # Delete focus area
        response = await client.delete(f"/api/v1/users/focus-areas/{created_area['id']}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "deleted" in data["message"].lower()
    
    async def test_delete_focus_area_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test deleting non-existent focus area."""
        response = await client.delete("/api/v1/users/focus-areas/999", headers=auth_headers)
        
        assert response.status_code == 404
    
    async def test_bulk_create_focus_areas_success(self, client: AsyncClient, auth_headers: dict):
        """Test bulk creating focus areas."""
        bulk_data = {
            "focus_areas": [
                {"focus_area": "Bulk Area 1", "keywords": ["bulk1"], "priority": 2},
                {"focus_area": "Bulk Area 2", "keywords": ["bulk2"], "priority": 3},
                {"focus_area": "Bulk Area 3", "keywords": ["bulk3"], "priority": 1}
            ]
        }
        
        response = await client.post("/api/v1/users/focus-areas/bulk", json=bulk_data, headers=auth_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert len(data) == 3
        assert all(area["focus_area"].startswith("Bulk Area") for area in data)
    
    async def test_bulk_create_focus_areas_with_duplicates(self, client: AsyncClient, auth_headers: dict):
        """Test bulk creating focus areas with some duplicates."""
        # Create one focus area first
        await client.post("/api/v1/users/focus-areas/", json={"focus_area": "Existing Area", "priority": 2}, headers=auth_headers)
        
        bulk_data = {
            "focus_areas": [
                {"focus_area": "Existing Area", "keywords": ["existing"], "priority": 2},  # Duplicate
                {"focus_area": "New Bulk Area", "keywords": ["new"], "priority": 3}
            ]
        }
        
        response = await client.post("/api/v1/users/focus-areas/bulk", json=bulk_data, headers=auth_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert len(data) == 1  # Only the new one should be created
        assert data[0]["focus_area"] == "New Bulk Area"
    
    async def test_delete_all_focus_areas_success(self, client: AsyncClient, auth_headers: dict):
        """Test deleting all focus areas."""
        # Create multiple focus areas
        for i in range(3):
            focus_area_data = {"focus_area": f"Area {i}", "priority": 2}
            await client.post("/api/v1/users/focus-areas/", json=focus_area_data, headers=auth_headers)
        
        # Delete all
        response = await client.delete("/api/v1/users/focus-areas/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "3" in data["message"]
        assert "deleted" in data["message"].lower()
    
    async def test_delete_all_focus_areas_empty(self, client: AsyncClient, auth_headers: dict):
        """Test deleting all focus areas when none exist."""
        response = await client.delete("/api/v1/users/focus-areas/", headers=auth_headers)
        
        assert response.status_code == 404
        data = response.json()
        assert "no focus areas" in data["detail"].lower()
    
    async def test_get_focus_area_analytics_success(self, client: AsyncClient, auth_headers: dict):
        """Test getting focus area analytics."""
        # Create focus areas with different priorities
        focus_areas = [
            {"focus_area": "High Priority Area", "keywords": ["ai", "machine"], "priority": 3},
            {"focus_area": "Critical Area", "keywords": ["security", "threat"], "priority": 4},
            {"focus_area": "Medium Area", "keywords": ["cloud", "computing"], "priority": 2}
        ]
        
        for area in focus_areas:
            await client.post("/api/v1/users/focus-areas/", json=area, headers=auth_headers)
        
        response = await client.get("/api/v1/users/focus-areas/analytics/summary", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_focus_areas"] == 3
        assert "priority_distribution" in data
        assert data["priority_distribution"]["high"] == 1
        assert data["priority_distribution"]["critical"] == 1
        assert data["priority_distribution"]["medium"] == 1
        assert data["keyword_count"] == 6  # Total unique keywords
        assert "most_common_keywords" in data
        assert "coverage_score" in data
        assert "recommendations" in data
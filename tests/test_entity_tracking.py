"""
Entity tracking endpoint tests.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


@pytest.mark.auth
class TestEntityTrackingEndpoints:
    """Test entity tracking management endpoints."""
    
    async def test_create_tracking_entity_success(self, client: AsyncClient, auth_headers: dict, sample_entity_data: dict):
        """Test creating tracking entity."""
        response = await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_entity_data["name"]
        assert data["entity_type"] == sample_entity_data["entity_type"]
        assert data["domain"] == sample_entity_data["domain"]
        assert data["description"] == sample_entity_data["description"]
        assert data["industry"] == sample_entity_data["industry"]
        assert "id" in data
        assert "created_at" in data
    
    async def test_create_tracking_entity_duplicate(self, client: AsyncClient, auth_headers: dict, sample_entity_data: dict):
        """Test creating duplicate tracking entity."""
        # Create first entity
        await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        
        # Try to create duplicate
        response = await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        
        assert response.status_code == 409
        data = response.json()
        assert "already exists" in data["detail"].lower()
    
    async def test_create_tracking_entity_invalid_type(self, client: AsyncClient, auth_headers: dict, sample_entity_data: dict):
        """Test creating entity with invalid type."""
        sample_entity_data["entity_type"] = "invalid_type"
        
        response = await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        
        assert response.status_code == 422
    
    async def test_get_available_entities_empty(self, client: AsyncClient):
        """Test getting available entities when none exist (public endpoint)."""
        response = await client.get("/api/v1/users/entity-tracking/entities")
        
        assert response.status_code == 200
        data = response.json()
        assert data == []
    
    async def test_get_available_entities_with_data(self, client: AsyncClient, auth_headers: dict, sample_entity_data: dict):
        """Test getting available entities with data."""
        # Create an entity first
        await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        
        response = await client.get("/api/v1/users/entity-tracking/entities")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == sample_entity_data["name"]
    
    async def test_get_available_entities_filter_by_type(self, client: AsyncClient, auth_headers: dict):
        """Test filtering available entities by type."""
        # Create entities of different types
        entities = [
            {"name": "Competitor Corp", "entity_type": "competitor", "industry": "Tech"},
            {"name": "Tech Topic", "entity_type": "topic", "industry": "Tech"},
            {"name": "Important Person", "entity_type": "person", "industry": "Tech"}
        ]
        
        for entity in entities:
            await client.post("/api/v1/users/entity-tracking/entities", json=entity, headers=auth_headers)
        
        # Filter by competitor type
        response = await client.get("/api/v1/users/entity-tracking/entities?entity_type=competitor")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["entity_type"] == "competitor"
    
    async def test_get_available_entities_filter_by_industry(self, client: AsyncClient, auth_headers: dict):
        """Test filtering available entities by industry."""
        # Create entities in different industries
        entities = [
            {"name": "Health Corp", "entity_type": "competitor", "industry": "Healthcare"},
            {"name": "Tech Corp", "entity_type": "competitor", "industry": "Technology"}
        ]
        
        for entity in entities:
            await client.post("/api/v1/users/entity-tracking/entities", json=entity, headers=auth_headers)
        
        # Filter by healthcare industry
        response = await client.get("/api/v1/users/entity-tracking/entities?industry=Healthcare")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["industry"] == "Healthcare"
    
    async def test_get_available_entities_search(self, client: AsyncClient, auth_headers: dict):
        """Test searching available entities."""
        # Create entities
        entities = [
            {"name": "OpenAI Research", "entity_type": "competitor", "description": "AI research company"},
            {"name": "Microsoft", "entity_type": "competitor", "description": "Technology company"},
            {"name": "AI Ethics", "entity_type": "topic", "description": "Artificial intelligence ethics"}
        ]
        
        for entity in entities:
            await client.post("/api/v1/users/entity-tracking/entities", json=entity, headers=auth_headers)
        
        # Search for "AI"
        response = await client.get("/api/v1/users/entity-tracking/entities?search=AI")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2  # OpenAI and AI Ethics should match
    
    async def test_start_tracking_entity_success(self, client: AsyncClient, auth_headers: dict, sample_entity_data: dict):
        """Test starting to track an entity."""
        # Create entity first
        entity_response = await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        entity = entity_response.json()
        
        # Start tracking
        tracking_data = {
            "entity_id": entity["id"],
            "priority": 3,
            "custom_keywords": ["openai", "gpt", "ai"],
            "tracking_enabled": True
        }
        
        response = await client.post("/api/v1/users/entity-tracking/", json=tracking_data, headers=auth_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["entity_id"] == entity["id"]
        assert data["priority"] == 3
        assert data["priority_label"] == "high"
        assert data["custom_keywords"] == ["openai", "gpt", "ai"]
        assert data["tracking_enabled"] == True
        assert "entity" in data
        assert data["entity"]["name"] == sample_entity_data["name"]
    
    async def test_start_tracking_entity_duplicate(self, client: AsyncClient, auth_headers: dict, sample_entity_data: dict):
        """Test starting to track an entity that's already being tracked."""
        # Create entity and start tracking
        entity_response = await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        entity = entity_response.json()
        
        tracking_data = {"entity_id": entity["id"], "priority": 3}
        await client.post("/api/v1/users/entity-tracking/", json=tracking_data, headers=auth_headers)
        
        # Try to track again
        response = await client.post("/api/v1/users/entity-tracking/", json=tracking_data, headers=auth_headers)
        
        assert response.status_code == 409
        data = response.json()
        assert "already tracking" in data["detail"].lower()
    
    async def test_start_tracking_nonexistent_entity(self, client: AsyncClient, auth_headers: dict):
        """Test tracking non-existent entity."""
        tracking_data = {"entity_id": 999, "priority": 3}
        
        response = await client.post("/api/v1/users/entity-tracking/", json=tracking_data, headers=auth_headers)
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    async def test_get_user_tracked_entities_empty(self, client: AsyncClient, auth_headers: dict):
        """Test getting user's tracked entities when none exist."""
        response = await client.get("/api/v1/users/entity-tracking/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["per_page"] == 10
        assert data["pages"] == 0
    
    async def test_get_user_tracked_entities_with_data(self, client: AsyncClient, auth_headers: dict, sample_entity_data: dict):
        """Test getting user's tracked entities with data."""
        # Create entity and start tracking
        entity_response = await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        entity = entity_response.json()
        
        tracking_data = {"entity_id": entity["id"], "priority": 4, "custom_keywords": ["test"]}
        await client.post("/api/v1/users/entity-tracking/", json=tracking_data, headers=auth_headers)
        
        response = await client.get("/api/v1/users/entity-tracking/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 1
        assert data["items"][0]["priority"] == 4
        assert data["items"][0]["entity"]["name"] == sample_entity_data["name"]
    
    async def test_get_user_tracked_entities_filter_by_priority(self, client: AsyncClient, auth_headers: dict):
        """Test filtering tracked entities by priority."""
        # Create entities and track them with different priorities
        for i, priority in enumerate([1, 2, 3, 4], 1):
            entity_data = {"name": f"Entity {i}", "entity_type": "competitor"}
            entity_response = await client.post("/api/v1/users/entity-tracking/entities", json=entity_data, headers=auth_headers)
            entity = entity_response.json()
            
            tracking_data = {"entity_id": entity["id"], "priority": priority}
            await client.post("/api/v1/users/entity-tracking/", json=tracking_data, headers=auth_headers)
        
        # Filter by critical priority (4)
        response = await client.get("/api/v1/users/entity-tracking/?priority=4", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["priority"] == 4
    
    async def test_get_user_tracked_entities_filter_by_type(self, client: AsyncClient, auth_headers: dict):
        """Test filtering tracked entities by entity type."""
        # Create entities of different types
        entities = [
            {"name": "Competitor", "entity_type": "competitor"},
            {"name": "Topic", "entity_type": "topic"}
        ]
        
        for entity_data in entities:
            entity_response = await client.post("/api/v1/users/entity-tracking/entities", json=entity_data, headers=auth_headers)
            entity = entity_response.json()
            
            tracking_data = {"entity_id": entity["id"], "priority": 3}
            await client.post("/api/v1/users/entity-tracking/", json=tracking_data, headers=auth_headers)
        
        # Filter by competitor type
        response = await client.get("/api/v1/users/entity-tracking/?entity_type=competitor", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["entity"]["entity_type"] == "competitor"
    
    async def test_update_entity_tracking_success(self, client: AsyncClient, auth_headers: dict, sample_entity_data: dict):
        """Test updating entity tracking settings."""
        # Create entity and start tracking
        entity_response = await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        entity = entity_response.json()
        
        tracking_response = await client.post("/api/v1/users/entity-tracking/", json={"entity_id": entity["id"], "priority": 2}, headers=auth_headers)
        tracking = tracking_response.json()
        
        # Update tracking settings
        update_data = {
            "priority": 4,
            "custom_keywords": ["updated", "keywords"],
            "tracking_enabled": False
        }
        
        response = await client.put(f"/api/v1/users/entity-tracking/{tracking['id']}", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["priority"] == 4
        assert data["custom_keywords"] == ["updated", "keywords"]
        assert data["tracking_enabled"] == False
    
    async def test_update_entity_tracking_partial(self, client: AsyncClient, auth_headers: dict, sample_entity_data: dict):
        """Test partial update of entity tracking."""
        # Create entity and start tracking
        entity_response = await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        entity = entity_response.json()
        
        tracking_response = await client.post("/api/v1/users/entity-tracking/", json={"entity_id": entity["id"], "priority": 2}, headers=auth_headers)
        tracking = tracking_response.json()
        
        # Partial update
        update_data = {"priority": 1}
        
        response = await client.put(f"/api/v1/users/entity-tracking/{tracking['id']}", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["priority"] == 1
        assert data["tracking_enabled"] == True  # Should remain unchanged
    
    async def test_update_entity_tracking_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test updating non-existent tracking record."""
        update_data = {"priority": 3}
        
        response = await client.put("/api/v1/users/entity-tracking/999", json=update_data, headers=auth_headers)
        
        assert response.status_code == 404
    
    async def test_stop_tracking_entity_success(self, client: AsyncClient, auth_headers: dict, sample_entity_data: dict):
        """Test stopping entity tracking."""
        # Create entity and start tracking
        entity_response = await client.post("/api/v1/users/entity-tracking/entities", json=sample_entity_data, headers=auth_headers)
        entity = entity_response.json()
        
        tracking_response = await client.post("/api/v1/users/entity-tracking/", json={"entity_id": entity["id"], "priority": 3}, headers=auth_headers)
        tracking = tracking_response.json()
        
        # Stop tracking
        response = await client.delete(f"/api/v1/users/entity-tracking/{tracking['id']}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "stopped tracking" in data["message"].lower()
        assert sample_entity_data["name"] in data["message"]
    
    async def test_stop_tracking_entity_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test stopping tracking for non-existent record."""
        response = await client.delete("/api/v1/users/entity-tracking/999", headers=auth_headers)
        
        assert response.status_code == 404
    
    async def test_search_entities_success(self, client: AsyncClient, auth_headers: dict):
        """Test searching entities."""
        # Create entities
        entities = [
            {"name": "OpenAI", "entity_type": "competitor", "industry": "AI"},
            {"name": "Microsoft AI", "entity_type": "organization", "industry": "Technology"},
            {"name": "Machine Learning", "entity_type": "topic", "industry": "AI"}
        ]
        
        for entity in entities:
            await client.post("/api/v1/users/entity-tracking/entities", json=entity, headers=auth_headers)
        
        # Search with query and filters
        search_data = {
            "query": "AI",
            "entity_types": ["competitor", "topic"],
            "industries": ["AI"]
        }
        
        response = await client.post("/api/v1/users/entity-tracking/search", json=search_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2  # OpenAI and Machine Learning should match
    
    async def test_search_entities_empty_query(self, client: AsyncClient):
        """Test searching entities with empty query."""
        search_data = {"query": ""}
        
        response = await client.post("/api/v1/users/entity-tracking/search", json=search_data)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    async def test_get_tracking_analytics_success(self, client: AsyncClient, auth_headers: dict):
        """Test getting tracking analytics."""
        # Create entities and track them
        entities = [
            {"name": "Competitor 1", "entity_type": "competitor", "industry": "Tech"},
            {"name": "Topic 1", "entity_type": "topic", "industry": "AI"},
            {"name": "Organization 1", "entity_type": "organization", "industry": "Tech"}
        ]
        
        for i, entity_data in enumerate(entities, 1):
            entity_response = await client.post("/api/v1/users/entity-tracking/entities", json=entity_data, headers=auth_headers)
            entity = entity_response.json()
            
            tracking_data = {
                "entity_id": entity["id"],
                "priority": i + 1,  # Different priorities
                "custom_keywords": [f"keyword{i}"]
            }
            await client.post("/api/v1/users/entity-tracking/", json=tracking_data, headers=auth_headers)
        
        response = await client.get("/api/v1/users/entity-tracking/analytics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_tracked_entities"] == 3
        assert "entities_by_type" in data
        assert data["entities_by_type"]["competitor"] == 1
        assert data["entities_by_type"]["topic"] == 1
        assert data["entities_by_type"]["organization"] == 1
        assert "priority_distribution" in data
        assert data["enabled_count"] == 3
        assert data["disabled_count"] == 0
        assert "top_industries" in data
        assert "keyword_cloud" in data
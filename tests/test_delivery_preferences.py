"""
Delivery preferences endpoint tests.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.delivery import UserDeliveryPreferences


@pytest.mark.auth
class TestDeliveryPreferencesEndpoints:
    """Test delivery preferences management endpoints."""
    
    async def test_get_delivery_preferences_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test getting delivery preferences when none exist."""
        response = await client.get("/api/v1/users/delivery-preferences/", headers=auth_headers)
        
        assert response.status_code == 404
        data = response.json()
        assert "not configured" in data["detail"].lower()
    
    async def test_get_delivery_preferences_success(self, client: AsyncClient, auth_headers: dict, test_delivery_preferences: UserDeliveryPreferences):
        """Test getting existing delivery preferences."""
        response = await client.get("/api/v1/users/delivery-preferences/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["frequency"] == test_delivery_preferences.frequency
        assert data["delivery_time"] == "08:00"
        assert data["timezone"] == test_delivery_preferences.timezone
        assert data["weekend_delivery"] == test_delivery_preferences.weekend_delivery
        assert data["max_articles_per_report"] == test_delivery_preferences.max_articles_per_report
        assert data["min_significance_level"] == test_delivery_preferences.min_significance_level
        assert data["content_format"] == test_delivery_preferences.content_format
        assert "id" in data
        assert "user_id" in data
        assert "created_at" in data
        assert "updated_at" in data
    
    async def test_create_delivery_preferences_success(self, client: AsyncClient, auth_headers: dict, sample_delivery_preferences_data: dict):
        """Test creating delivery preferences."""
        response = await client.put("/api/v1/users/delivery-preferences/", json=sample_delivery_preferences_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["frequency"] == sample_delivery_preferences_data["frequency"]
        assert data["delivery_time"] == sample_delivery_preferences_data["delivery_time"]
        assert data["timezone"] == sample_delivery_preferences_data["timezone"]
        assert data["weekend_delivery"] == sample_delivery_preferences_data["weekend_delivery"]
        assert data["max_articles_per_report"] == sample_delivery_preferences_data["max_articles_per_report"]
        assert data["min_significance_level"] == sample_delivery_preferences_data["min_significance_level"]
        assert data["content_format"] == sample_delivery_preferences_data["content_format"]
        assert data["email_enabled"] == sample_delivery_preferences_data["email_enabled"]
        assert data["urgent_alerts_enabled"] == sample_delivery_preferences_data["urgent_alerts_enabled"]
        assert data["digest_mode"] == sample_delivery_preferences_data["digest_mode"]
        assert "id" in data
        assert "user_id" in data
    
    async def test_create_delivery_preferences_invalid_frequency(self, client: AsyncClient, auth_headers: dict, sample_delivery_preferences_data: dict):
        """Test creating preferences with invalid frequency."""
        sample_delivery_preferences_data["frequency"] = "invalid_frequency"
        
        response = await client.put("/api/v1/users/delivery-preferences/", json=sample_delivery_preferences_data, headers=auth_headers)
        
        assert response.status_code == 422
    
    async def test_create_delivery_preferences_invalid_time_format(self, client: AsyncClient, auth_headers: dict, sample_delivery_preferences_data: dict):
        """Test creating preferences with invalid time format."""
        sample_delivery_preferences_data["delivery_time"] = "25:00"  # Invalid hour
        
        response = await client.put("/api/v1/users/delivery-preferences/", json=sample_delivery_preferences_data, headers=auth_headers)
        
        assert response.status_code == 422
    
    async def test_create_delivery_preferences_invalid_timezone(self, client: AsyncClient, auth_headers: dict, sample_delivery_preferences_data: dict):
        """Test creating preferences with invalid timezone."""
        sample_delivery_preferences_data["timezone"] = "Invalid/Timezone"
        
        response = await client.put("/api/v1/users/delivery-preferences/", json=sample_delivery_preferences_data, headers=auth_headers)
        
        assert response.status_code == 422
    
    async def test_create_delivery_preferences_invalid_max_articles(self, client: AsyncClient, auth_headers: dict, sample_delivery_preferences_data: dict):
        """Test creating preferences with invalid max articles."""
        sample_delivery_preferences_data["max_articles_per_report"] = 100  # Too high (max is 50)
        
        response = await client.put("/api/v1/users/delivery-preferences/", json=sample_delivery_preferences_data, headers=auth_headers)
        
        assert response.status_code == 422
    
    async def test_update_delivery_preferences_success(self, client: AsyncClient, auth_headers: dict, test_delivery_preferences: UserDeliveryPreferences):
        """Test updating existing delivery preferences."""
        update_data = {
            "frequency": "weekly",
            "delivery_time": "18:30",
            "weekend_delivery": True,
            "content_format": "bullet_points"
        }
        
        response = await client.put("/api/v1/users/delivery-preferences/", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["frequency"] == "weekly"
        assert data["delivery_time"] == "18:30"
        assert data["weekend_delivery"] == True
        assert data["content_format"] == "bullet_points"
        # Other fields should remain unchanged
        assert data["timezone"] == test_delivery_preferences.timezone
        assert data["max_articles_per_report"] == test_delivery_preferences.max_articles_per_report
    
    async def test_update_delivery_preferences_partial(self, client: AsyncClient, auth_headers: dict, test_delivery_preferences: UserDeliveryPreferences):
        """Test partial update of delivery preferences."""
        update_data = {"frequency": "hourly"}
        
        response = await client.put("/api/v1/users/delivery-preferences/", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["frequency"] == "hourly"
        # All other fields should remain unchanged
        assert data["delivery_time"] == "08:00"
        assert data["timezone"] == test_delivery_preferences.timezone
    
    async def test_update_delivery_preferences_all_frequencies(self, client: AsyncClient, auth_headers: dict):
        """Test all valid frequency options."""
        frequencies = ["real_time", "hourly", "daily", "weekly", "monthly"]
        
        for frequency in frequencies:
            update_data = {"frequency": frequency}
            response = await client.put("/api/v1/users/delivery-preferences/", json=update_data, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["frequency"] == frequency
    
    async def test_update_delivery_preferences_all_significance_levels(self, client: AsyncClient, auth_headers: dict):
        """Test all valid significance levels."""
        levels = ["low", "medium", "high", "critical"]
        
        for level in levels:
            update_data = {"min_significance_level": level}
            response = await client.put("/api/v1/users/delivery-preferences/", json=update_data, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["min_significance_level"] == level
    
    async def test_update_delivery_preferences_all_content_formats(self, client: AsyncClient, auth_headers: dict):
        """Test all valid content formats."""
        formats = ["full", "executive_summary", "summary", "bullet_points", "headlines_only"]
        
        for format_type in formats:
            update_data = {"content_format": format_type}
            response = await client.put("/api/v1/users/delivery-preferences/", json=update_data, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["content_format"] == format_type
    
    async def test_update_delivery_preferences_time_validation(self, client: AsyncClient, auth_headers: dict):
        """Test time format validation."""
        valid_times = ["00:00", "12:30", "23:59", "09:05"]
        invalid_times = ["24:00", "12:60", "9:5", "invalid"]
        
        # Test valid times
        for time_str in valid_times:
            update_data = {"delivery_time": time_str}
            response = await client.put("/api/v1/users/delivery-preferences/", json=update_data, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["delivery_time"] == time_str
        
        # Test invalid times
        for time_str in invalid_times:
            update_data = {"delivery_time": time_str}
            response = await client.put("/api/v1/users/delivery-preferences/", json=update_data, headers=auth_headers)
            
            assert response.status_code == 422
    
    async def test_update_delivery_preferences_timezone_validation(self, client: AsyncClient, auth_headers: dict):
        """Test timezone validation."""
        valid_timezones = [
            "UTC", "GMT", "EST", "PST",
            "America/New_York", "Europe/London", "Asia/Tokyo",
            "UTC+5", "UTC-8"
        ]
        
        for timezone in valid_timezones:
            update_data = {"timezone": timezone}
            response = await client.put("/api/v1/users/delivery-preferences/", json=update_data, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["timezone"] == timezone
    
    async def test_reset_delivery_preferences_success(self, client: AsyncClient, auth_headers: dict, test_delivery_preferences: UserDeliveryPreferences):
        """Test resetting delivery preferences."""
        response = await client.delete("/api/v1/users/delivery-preferences/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "reset" in data["message"].lower()
    
    async def test_reset_delivery_preferences_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test resetting preferences when none exist."""
        response = await client.delete("/api/v1/users/delivery-preferences/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "reset" in data["message"].lower()
    
    async def test_get_delivery_analytics_with_preferences(self, client: AsyncClient, auth_headers: dict, test_delivery_preferences: UserDeliveryPreferences):
        """Test getting delivery analytics with existing preferences."""
        response = await client.get("/api/v1/users/delivery-preferences/analytics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "delivery_schedule" in data
        assert "Daily at 08:00 AM" in data["delivery_schedule"]
        assert "weekdays only" in data["delivery_schedule"]
        assert data["articles_this_week"] >= 0
        assert data["avg_articles_per_report"] == test_delivery_preferences.max_articles_per_report
        assert data["urgent_alerts_count"] >= 0
        assert data["next_delivery"] is not None
        assert "frequency_recommendations" in data
        assert isinstance(data["frequency_recommendations"], list)
    
    async def test_get_delivery_analytics_without_preferences(self, client: AsyncClient, auth_headers: dict):
        """Test getting delivery analytics without preferences."""
        response = await client.get("/api/v1/users/delivery-preferences/analytics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["delivery_schedule"] == "Not configured"
        assert "frequency_recommendations" in data
        assert "Set up delivery preferences" in data["frequency_recommendations"][0]
    
    async def test_get_recommended_defaults_success(self, client: AsyncClient, auth_headers: dict):
        """Test getting recommended default preferences."""
        response = await client.get("/api/v1/users/delivery-preferences/defaults", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["recommended_frequency"] in ["real_time", "hourly", "daily", "weekly", "monthly"]
        assert ":" in data["recommended_time"]  # Should be HH:MM format
        assert data["recommended_timezone"] != ""
        assert data["recommended_format"] in ["full", "executive_summary", "summary", "bullet_points", "headlines_only"]
        assert "reasoning" in data
        assert isinstance(data["reasoning"], list)
        assert len(data["reasoning"]) > 0
    
    async def test_get_recommended_defaults_with_profile(self, client: AsyncClient, auth_headers: dict, test_user_with_profile: User):
        """Test getting defaults with user profile context."""
        response = await client.get("/api/v1/users/delivery-preferences/defaults", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "reasoning" in data
        # Should have profile-specific recommendations
        reasoning_text = " ".join(data["reasoning"]).lower()
        assert any(keyword in reasoning_text for keyword in ["ceo", "executive", "leadership", "daily"])
    
    async def test_test_delivery_schedule_success(self, client: AsyncClient, auth_headers: dict, test_delivery_preferences: UserDeliveryPreferences):
        """Test delivery schedule calculation."""
        response = await client.post("/api/v1/users/delivery-preferences/test-schedule", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "current_time" in data
        assert "should_deliver_today" in data
        assert "next_delivery_time" in data
        assert data["frequency"] == test_delivery_preferences.frequency
        assert data["delivery_time"] == "08:00"
        assert data["timezone"] == test_delivery_preferences.timezone
        assert data["weekend_delivery"] == test_delivery_preferences.weekend_delivery
        assert isinstance(data["should_deliver_today"], bool)
    
    async def test_test_delivery_schedule_not_found(self, client: AsyncClient, auth_headers: dict):
        """Test schedule calculation without preferences."""
        response = await client.post("/api/v1/users/delivery-preferences/test-schedule", headers=auth_headers)
        
        assert response.status_code == 404
        data = response.json()
        assert "not configured" in data["detail"].lower()
    
    async def test_delivery_preferences_without_auth(self, client: AsyncClient, sample_delivery_preferences_data: dict):
        """Test all endpoints without authentication."""
        endpoints = [
            ("GET", "/api/v1/users/delivery-preferences/"),
            ("PUT", "/api/v1/users/delivery-preferences/"),
            ("DELETE", "/api/v1/users/delivery-preferences/"),
            ("GET", "/api/v1/users/delivery-preferences/analytics"),
            ("GET", "/api/v1/users/delivery-preferences/defaults"),
            ("POST", "/api/v1/users/delivery-preferences/test-schedule")
        ]
        
        for method, endpoint in endpoints:
            if method == "GET":
                response = await client.get(endpoint)
            elif method == "PUT":
                response = await client.put(endpoint, json=sample_delivery_preferences_data)
            elif method == "DELETE":
                response = await client.delete(endpoint)
            elif method == "POST":
                response = await client.post(endpoint)
            
            assert response.status_code == 401
    
    async def test_delivery_preferences_edge_cases(self, client: AsyncClient, auth_headers: dict):
        """Test edge cases for delivery preferences."""
        # Test minimum values
        min_data = {
            "max_articles_per_report": 1,
            "delivery_time": "00:00"
        }
        
        response = await client.put("/api/v1/users/delivery-preferences/", json=min_data, headers=auth_headers)
        assert response.status_code == 200
        
        # Test maximum values
        max_data = {
            "max_articles_per_report": 50,
            "delivery_time": "23:59"
        }
        
        response = await client.put("/api/v1/users/delivery-preferences/", json=max_data, headers=auth_headers)
        assert response.status_code == 200
        
        # Test boolean combinations
        bool_combinations = [
            {"email_enabled": True, "urgent_alerts_enabled": True, "digest_mode": True, "weekend_delivery": True},
            {"email_enabled": False, "urgent_alerts_enabled": False, "digest_mode": False, "weekend_delivery": False},
            {"email_enabled": True, "urgent_alerts_enabled": False, "digest_mode": True, "weekend_delivery": False}
        ]
        
        for combo in bool_combinations:
            response = await client.put("/api/v1/users/delivery-preferences/", json=combo, headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            for key, value in combo.items():
                assert data[key] == value
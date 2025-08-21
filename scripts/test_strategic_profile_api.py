"""
Strategic Profile API Testing Script
Test comprehensive strategic profile management functionality.
"""

import asyncio
import httpx
import json
from typing import Optional


class StrategicProfileAPITester:
    """Test strategic profile API endpoints."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8002"):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.test_user_email = f"strategictest_{int(asyncio.get_event_loop().time())}@example.com"
    
    async def test_all_endpoints(self):
        """Run comprehensive test of all strategic profile endpoints."""
        print("STRATEGIC PROFILE API TESTING")
        print("=" * 50)
        
        try:
            async with httpx.AsyncClient() as client:
                # 1. Register and login to get access token
                print("\n=== Authentication ===")
                await self._register_and_login(client)
                
                # 2. Test enum endpoints (no auth required)
                print("\n=== Testing Enum Endpoints ===")
                await self._test_enum_endpoints(client)
                
                # 3. Test strategic profile CRUD operations
                print("\n=== Testing Strategic Profile CRUD ===")
                await self._test_create_profile(client)
                await self._test_get_profile(client)
                await self._test_update_profile(client)
                
                # 4. Test analytics and insights
                print("\n=== Testing Analytics ===")
                await self._test_analytics(client)
                await self._test_statistics(client)
                
                # 5. Test error scenarios
                print("\n=== Testing Error Handling ===")
                await self._test_error_scenarios(client)
                
                # 6. Clean up
                print("\n=== Cleanup ===")
                await self._test_delete_profile(client)
                
                print("\n" + "=" * 50)
                print("[PASS] ALL STRATEGIC PROFILE API TESTS COMPLETED SUCCESSFULLY")
                print("=" * 50)
                
        except Exception as e:
            print(f"\n[FAIL] Test failed: {e}")
            raise
    
    async def _register_and_login(self, client: httpx.AsyncClient):
        """Register test user and get access token."""
        # Register
        register_data = {
            "email": self.test_user_email,
            "name": "Strategic Test User",
            "password": "TestPass123!"
        }
        
        response = await client.post(f"{self.base_url}/api/v1/auth/register", json=register_data)
        if response.status_code == 201:
            print("[PASS] User registered successfully")
        else:
            print(f"[FAIL] Registration failed: {response.text}")
            raise Exception(f"Registration failed: {response.status_code}")
        
        # Login
        login_data = {
            "email": self.test_user_email,
            "password": "TestPass123!"
        }
        
        response = await client.post(f"{self.base_url}/api/v1/auth/login", json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data["access_token"]
            print("[PASS] Login successful, access token obtained")
        else:
            print(f"[FAIL] Login failed: {response.text}")
            raise Exception(f"Login failed: {response.status_code}")
    
    def _get_auth_headers(self):
        """Get authorization headers."""
        return {"Authorization": f"Bearer {self.access_token}"}
    
    async def _test_enum_endpoints(self, client: httpx.AsyncClient):
        """Test all enum endpoints."""
        enum_endpoints = [
            "industries",
            "organization-types", 
            "roles",
            "strategic-goals",
            "organization-sizes"
        ]
        
        for endpoint in enum_endpoints:
            response = await client.get(f"{self.base_url}/api/v1/strategic-profile/enums/{endpoint}")
            if response.status_code == 200:
                data = response.json()
                print(f"[PASS] {endpoint}: {len(list(data.values())[0])} options available")
            else:
                print(f"[FAIL] {endpoint} failed: {response.status_code}")
    
    async def _test_create_profile(self, client: httpx.AsyncClient):
        """Test creating a strategic profile."""
        profile_data = {
            "industry": "technology",
            "organization_type": "startup",
            "role": "ceo",
            "strategic_goals": [
                "market_expansion",
                "product_development",
                "competitive_intelligence"
            ],
            "organization_size": "small"
        }
        
        response = await client.post(
            f"{self.base_url}/api/v1/strategic-profile/",
            json=profile_data,
            headers=self._get_auth_headers()
        )
        
        if response.status_code == 201:
            profile = response.json()
            print(f"[PASS] Strategic profile created: ID {profile['id']}")
            print(f"   Industry: {profile['industry']}")
            print(f"   Role: {profile['role']}")
            print(f"   Goals: {', '.join(profile['strategic_goals'])}")
        else:
            print(f"[FAIL] Profile creation failed: {response.text}")
            raise Exception(f"Profile creation failed: {response.status_code}")
    
    async def _test_get_profile(self, client: httpx.AsyncClient):
        """Test getting a strategic profile."""
        response = await client.get(
            f"{self.base_url}/api/v1/strategic-profile/",
            headers=self._get_auth_headers()
        )
        
        if response.status_code == 200:
            profile = response.json()
            print(f"[PASS] Profile retrieved: {profile['industry']} - {profile['role']}")
        else:
            print(f"[FAIL] Profile retrieval failed: {response.text}")
            raise Exception(f"Profile retrieval failed: {response.status_code}")
    
    async def _test_update_profile(self, client: httpx.AsyncClient):
        """Test updating a strategic profile."""
        update_data = {
            "industry": "healthcare",
            "strategic_goals": [
                "digital_transformation",
                "compliance",
                "operational_efficiency",
                "data_analytics"
            ]
        }
        
        response = await client.put(
            f"{self.base_url}/api/v1/strategic-profile/",
            json=update_data,
            headers=self._get_auth_headers()
        )
        
        if response.status_code == 200:
            profile = response.json()
            print(f"[PASS] Profile updated: Industry changed to {profile['industry']}")
            print(f"   New goals: {', '.join(profile['strategic_goals'])}")
        else:
            print(f"[FAIL] Profile update failed: {response.text}")
            raise Exception(f"Profile update failed: {response.status_code}")
    
    async def _test_analytics(self, client: httpx.AsyncClient):
        """Test profile analytics endpoint."""
        response = await client.get(
            f"{self.base_url}/api/v1/strategic-profile/analytics",
            headers=self._get_auth_headers()
        )
        
        if response.status_code == 200:
            analytics = response.json()
            print(f"[PASS] Analytics retrieved:")
            print(f"   Completeness: {analytics['profile_completeness']:.1f}%")
            print(f"   Missing fields: {', '.join(analytics['missing_fields']) if analytics['missing_fields'] else 'None'}")
            print(f"   Recommended goals: {', '.join(analytics['recommended_goals'][:3])}...")
        else:
            print(f"[FAIL] Analytics failed: {response.text}")
    
    async def _test_statistics(self, client: httpx.AsyncClient):
        """Test profile statistics endpoint."""
        response = await client.get(
            f"{self.base_url}/api/v1/strategic-profile/stats",
            headers=self._get_auth_headers()
        )
        
        if response.status_code == 200:
            stats = response.json()
            print(f"[PASS] Statistics retrieved:")
            print(f"   Total profiles: {stats['total_profiles']}")
            if stats['industry_distribution']:
                print(f"   Top industry: {stats['industry_distribution'][0]['industry']}")
        else:
            print(f"[FAIL] Statistics failed: {response.text}")
    
    async def _test_error_scenarios(self, client: httpx.AsyncClient):
        """Test error handling scenarios."""
        # Test invalid industry
        invalid_data = {
            "industry": "invalid_industry",
            "role": "ceo"
        }
        
        response = await client.put(
            f"{self.base_url}/api/v1/strategic-profile/",
            json=invalid_data,
            headers=self._get_auth_headers()
        )
        
        if response.status_code == 422:
            print("[PASS] Invalid industry correctly rejected (HTTP 422)")
        else:
            print(f"[FAIL] Invalid industry should be rejected: {response.status_code}")
        
        # Test unauthorized access
        response = await client.get(f"{self.base_url}/api/v1/strategic-profile/")
        
        if response.status_code == 401:
            print("[PASS] Unauthorized access correctly rejected (HTTP 401)")
        else:
            print(f"[FAIL] Unauthorized access should be rejected: {response.status_code}")
    
    async def _test_delete_profile(self, client: httpx.AsyncClient):
        """Test deleting a strategic profile."""
        response = await client.delete(
            f"{self.base_url}/api/v1/strategic-profile/",
            headers=self._get_auth_headers()
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[PASS] Profile deleted: {result['message']}")
        else:
            print(f"[FAIL] Profile deletion failed: {response.text}")


async def main():
    """Run strategic profile API tests."""
    tester = StrategicProfileAPITester()
    
    try:
        await tester.test_all_endpoints()
        print("\n[SUCCESS] All strategic profile API tests passed!")
        return 0
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Testing stopped by user")
        return 1
    except Exception as e:
        print(f"\n\n[ERROR] Testing failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
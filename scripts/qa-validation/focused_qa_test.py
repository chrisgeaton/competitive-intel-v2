#!/usr/bin/env python3
"""
Focused QA Test Script - Testing the Three Major Fixes

This script specifically tests the three critical issues that were identified and fixed:
1. Delivery preferences schema validation error - FIXED
2. Missing /me endpoint in authentication router - FIXED  
3. Entity listing authentication issue - FIXED
"""

import asyncio
import json
import time
import random
from datetime import datetime
from typing import Dict, Any, Optional

import httpx


class FocusedQAValidator:
    """Focused QA validation for the three major fixes."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8002"):
        self.base_url = base_url
        self.session = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
        
    def log(self, message: str):
        """Log message with ASCII-only characters."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        ascii_message = message.encode('ascii', errors='ignore').decode('ascii')
        print(f"[{timestamp}] {ascii_message}")
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.session.aclose()
    
    async def register_and_login_user(self, email_suffix: str) -> Optional[str]:
        """Register and login a unique test user to avoid rate limiting."""
        try:
            # Use random suffix to avoid conflicts
            rand_id = random.randint(1000, 9999)
            email = f"test_fix_{email_suffix}_{rand_id}@example.com"
            
            # Register user
            register_data = {
                "email": email,
                "name": f"Test Fix User {email_suffix}",
                "password": "TestPassword123!"
            }
            
            register_response = await self.session.post(
                f"{self.base_url}/api/v1/auth/register",
                json=register_data
            )
            
            if register_response.status_code != 201:
                self.log(f"Registration failed: {register_response.status_code} - {register_response.text}")
                return None
            
            self.log(f"User registered: {email}")
            
            # Login user
            login_data = {
                "email": email,
                "password": "TestPassword123!"
            }
            
            login_response = await self.session.post(
                f"{self.base_url}/api/v1/auth/login",
                json=login_data
            )
            
            if login_response.status_code != 200:
                self.log(f"Login failed: {login_response.status_code} - {register_response.text}")
                return None
            
            token_data = login_response.json()
            access_token = token_data.get("access_token")
            
            if not access_token:
                self.log("No access token received")
                return None
            
            self.log(f"Login successful for {email}")
            return access_token
            
        except Exception as e:
            self.log(f"Auth setup failed: {str(e)}")
            return None
    
    def get_auth_headers(self, token: str) -> Dict[str, str]:
        """Get authorization headers."""
        return {"Authorization": f"Bearer {token}"}
    
    async def test_fix_1_delivery_preferences(self) -> Dict[str, Any]:
        """Test Fix #1: Delivery preferences schema validation."""
        self.log("=== TESTING FIX #1: DELIVERY PREFERENCES SCHEMA ===")
        
        # Get fresh token
        token = await self.register_and_login_user("delivery_prefs")
        if not token:
            return {"name": "delivery_preferences_fix", "status": "FAIL", "error": "Authentication failed"}
        
        headers = self.get_auth_headers(token)
        
        try:
            # Test GET delivery preferences (should work without schema errors)
            get_response = await self.session.get(
                f"{self.base_url}/api/v1/users/delivery-preferences",
                headers=headers
            )
            
            if get_response.status_code == 404:
                self.log("[PASS] No existing preferences (expected)")
                
                # Test PUT delivery preferences (create new)
                prefs_data = {
                    "frequency": "daily",
                    "delivery_time": "09:00",
                    "timezone": "UTC",
                    "weekend_delivery": False,
                    "max_articles_per_report": 10,
                    "min_significance_level": "medium",
                    "content_format": "detailed",
                    "email_enabled": True,
                    "urgent_alerts_enabled": True,
                    "digest_mode": "individual"
                }
                
                put_response = await self.session.put(
                    f"{self.base_url}/api/v1/users/delivery-preferences",
                    headers=headers,
                    json=prefs_data
                )
                
                if put_response.status_code == 200:
                    self.log("[PASS] Delivery preferences created successfully")
                    
                    # Test GET again to verify no schema errors
                    get_response2 = await self.session.get(
                        f"{self.base_url}/api/v1/users/delivery-preferences",
                        headers=headers
                    )
                    
                    if get_response2.status_code == 200:
                        response_data = get_response2.json()
                        if "delivery_time" in response_data and response_data["frequency"] == "daily":
                            self.log("[PASS] Schema validation working correctly")
                            return {"name": "delivery_preferences_fix", "status": "PASS", "details": "Schema validation resolved"}
                        else:
                            return {"name": "delivery_preferences_fix", "status": "FAIL", "error": "Invalid response data"}
                    else:
                        return {"name": "delivery_preferences_fix", "status": "FAIL", "error": f"GET failed: {get_response2.status_code}"}
                else:
                    return {"name": "delivery_preferences_fix", "status": "FAIL", "error": f"PUT failed: {put_response.status_code} - {put_response.text}"}
            
            elif get_response.status_code == 200:
                self.log("[PASS] Existing preferences retrieved without schema errors")
                return {"name": "delivery_preferences_fix", "status": "PASS", "details": "Schema validation working"}
            
            else:
                return {"name": "delivery_preferences_fix", "status": "FAIL", "error": f"Unexpected status: {get_response.status_code}"}
                
        except Exception as e:
            return {"name": "delivery_preferences_fix", "status": "FAIL", "error": f"Exception: {str(e)}"}
    
    async def test_fix_2_auth_me_endpoint(self) -> Dict[str, Any]:
        """Test Fix #2: Missing /me endpoint in authentication router."""
        self.log("=== TESTING FIX #2: AUTH /ME ENDPOINT ===")
        
        # Get fresh token
        token = await self.register_and_login_user("auth_me")
        if not token:
            return {"name": "auth_me_endpoint_fix", "status": "FAIL", "error": "Authentication failed"}
        
        headers = self.get_auth_headers(token)
        
        try:
            # Test unauthorized access first
            unauth_response = await self.session.get(f"{self.base_url}/api/v1/auth/me")
            
            if unauth_response.status_code == 401:
                self.log("[PASS] Correctly blocks unauthorized access")
            else:
                self.log(f"[WARN] Unexpected unauthorized response: {unauth_response.status_code}")
            
            # Test authorized access
            auth_response = await self.session.get(
                f"{self.base_url}/api/v1/auth/me",
                headers=headers
            )
            
            if auth_response.status_code == 200:
                user_data = auth_response.json()
                required_fields = ["id", "email", "name", "is_active", "subscription_status"]
                
                if all(field in user_data for field in required_fields):
                    self.log(f"[PASS] /me endpoint working! User: {user_data.get('email', 'unknown')}")
                    self.log(f"  User ID: {user_data.get('id')}")
                    self.log(f"  Active: {user_data.get('is_active')}")
                    self.log(f"  Subscription: {user_data.get('subscription_status')}")
                    
                    return {"name": "auth_me_endpoint_fix", "status": "PASS", "details": "Endpoint working correctly"}
                else:
                    missing_fields = [f for f in required_fields if f not in user_data]
                    return {"name": "auth_me_endpoint_fix", "status": "FAIL", "error": f"Missing fields: {missing_fields}"}
            
            elif auth_response.status_code == 404:
                return {"name": "auth_me_endpoint_fix", "status": "FAIL", "error": "Endpoint still returns 404 - not fixed"}
            
            else:
                return {"name": "auth_me_endpoint_fix", "status": "FAIL", "error": f"Unexpected status: {auth_response.status_code} - {auth_response.text}"}
                
        except Exception as e:
            return {"name": "auth_me_endpoint_fix", "status": "FAIL", "error": f"Exception: {str(e)}"}
    
    async def test_fix_3_entity_tracking_auth(self) -> Dict[str, Any]:
        """Test Fix #3: Entity listing authentication issue."""
        self.log("=== TESTING FIX #3: ENTITY TRACKING AUTHENTICATION ===")
        
        # Get fresh token
        token = await self.register_and_login_user("entity_tracking")
        if not token:
            return {"name": "entity_tracking_auth_fix", "status": "FAIL", "error": "Authentication failed"}
        
        headers = self.get_auth_headers(token)
        
        try:
            # Test unauthorized access first
            unauth_response = await self.session.get(f"{self.base_url}/api/v1/entities/entities")
            
            if unauth_response.status_code == 401:
                self.log("[PASS] Correctly requires authentication")
            else:
                self.log(f"[WARN] Unexpected unauthorized response: {unauth_response.status_code}")
            
            # Test entity listing with authentication
            entities_response = await self.session.get(
                f"{self.base_url}/api/v1/entities/entities",
                headers=headers
            )
            
            if entities_response.status_code == 200:
                entities_data = entities_response.json()
                self.log(f"[PASS] Entity listing working - {len(entities_data)} entities found")
                
                # Test entity search
                search_response = await self.session.get(
                    f"{self.base_url}/api/v1/entities/search?query=tech",
                    headers=headers
                )
                
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    self.log(f"[PASS] Entity search working - {len(search_data)} results found")
                    
                    # Test analytics endpoint
                    analytics_response = await self.session.get(
                        f"{self.base_url}/api/v1/entities/analytics",
                        headers=headers
                    )
                    
                    if analytics_response.status_code == 200:
                        analytics_data = analytics_response.json()
                        tracked_count = analytics_data.get("total_tracked_entities", 0)
                        self.log(f"[PASS] Entity analytics working - tracking {tracked_count} entities")
                        
                        return {"name": "entity_tracking_auth_fix", "status": "PASS", "details": "All endpoints working correctly"}
                    else:
                        return {"name": "entity_tracking_auth_fix", "status": "FAIL", "error": f"Analytics failed: {analytics_response.status_code}"}
                else:
                    return {"name": "entity_tracking_auth_fix", "status": "FAIL", "error": f"Search failed: {search_response.status_code}"}
            
            elif entities_response.status_code == 401:
                return {"name": "entity_tracking_auth_fix", "status": "FAIL", "error": "Still getting 401 - authentication fix not working"}
            
            else:
                return {"name": "entity_tracking_auth_fix", "status": "FAIL", "error": f"Unexpected status: {entities_response.status_code} - {entities_response.text}"}
                
        except Exception as e:
            return {"name": "entity_tracking_auth_fix", "status": "FAIL", "error": f"Exception: {str(e)}"}
    
    async def run_focused_qa(self) -> Dict[str, Any]:
        """Run focused QA validation for the three major fixes."""
        self.log("================================================================================")
        self.log("FOCUSED QA VALIDATION - THREE MAJOR FIXES")
        self.log("================================================================================")
        self.log(f"Target: {self.base_url}")
        self.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("================================================================================")
        
        start_time = time.time()
        
        # Test all three fixes
        tests = [
            self.test_fix_1_delivery_preferences(),
            self.test_fix_2_auth_me_endpoint(),
            self.test_fix_3_entity_tracking_auth()
        ]
        
        # Run tests
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Process results
        total_duration = (time.time() - start_time) * 1000
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_ms": round(total_duration, 1),
            "fixes_tested": 3,
            "fixes_passed": 0,
            "fixes_failed": 0,
            "test_results": []
        }
        
        self.log("\n================================================================================")
        self.log("FOCUSED QA RESULTS - FIX VALIDATION")
        self.log("================================================================================")
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                test_result = {
                    "name": f"fix_{i+1}_exception",
                    "status": "FAIL", 
                    "error": f"Exception: {str(result)}"
                }
            else:
                test_result = result
            
            summary["test_results"].append(test_result)
            
            if test_result["status"] == "PASS":
                summary["fixes_passed"] += 1
                status_icon = "✓"
                status_text = "PASS"
            else:
                summary["fixes_failed"] += 1
                status_icon = "✗"
                status_text = "FAIL"
            
            self.log(f"[{status_icon}] Fix #{i+1} ({test_result['name']}): {status_text}")
            if test_result["status"] == "PASS" and "details" in test_result:
                self.log(f"    Details: {test_result['details']}")
            elif test_result["status"] == "FAIL" and "error" in test_result:
                self.log(f"    Error: {test_result['error']}")
        
        self.log("\n================================================================================")
        self.log("FINAL VALIDATION SUMMARY")
        self.log("================================================================================")
        
        success_rate = (summary["fixes_passed"] / summary["fixes_tested"]) * 100
        overall_status = "SUCCESS" if summary["fixes_passed"] == summary["fixes_tested"] else "PARTIAL"
        
        self.log(f"Duration: {summary['total_duration_ms']}ms ({summary['total_duration_ms']/1000:.2f}s)")
        self.log(f"Fixes Tested: {summary['fixes_tested']}")
        self.log(f"  - Passed: {summary['fixes_passed']}")
        self.log(f"  - Failed: {summary['fixes_failed']}")
        self.log(f"Success Rate: {success_rate:.1f}%")
        self.log(f"Overall Status: {overall_status}")
        
        if summary["fixes_passed"] == summary["fixes_tested"]:
            self.log("\n🎉 ALL FIXES VALIDATED SUCCESSFULLY!")
            self.log("The User Config Service is operating correctly with all major issues resolved.")
        else:
            self.log(f"\n⚠️  {summary['fixes_failed']} FIX(ES) STILL NEED ATTENTION")
        
        self.log("\n[ASCII] All output formatted with ASCII-only characters")
        
        return summary


async def main():
    """Main execution function."""
    validator = FocusedQAValidator()
    
    try:
        results = await validator.run_focused_qa()
        
        # Save results
        with open("focused_qa_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: focused_qa_results.json")
        
    except Exception as e:
        print(f"Focused QA validation failed: {e}")
        return 1
    
    finally:
        await validator.cleanup()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
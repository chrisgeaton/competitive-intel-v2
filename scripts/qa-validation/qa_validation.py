"""
Comprehensive QA Validation Script for User Config Service
Validates all 6 service modules with detailed testing and ASCII-only reporting
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import httpx


class QAValidator:
    """Comprehensive QA validator for the User Config Service."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8002"):
        self.base_url = base_url
        self.results = {}
        self.auth_token = None
        self.auth_headers = {}
        self.test_user_id = None
        self.performance_metrics = {}
        
    def log(self, message: str):
        """Log message with ASCII-only formatting."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def format_result(self, status: str, message: str = "") -> str:
        """Format test result with ASCII-only characters."""
        status_map = {
            "PASS": "[PASS]",
            "FAIL": "[FAIL]", 
            "WARN": "[WARN]",
            "INFO": "[INFO]",
            "ERROR": "[ERROR]"
        }
        return f"{status_map.get(status, '[UNKNOWN]')} {message}"
    
    async def measure_performance(self, func, *args, **kwargs):
        """Measure performance of async function."""
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, duration
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> tuple[httpx.Response, float]:
        """Make HTTP request with performance measurement."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, timeout=30.0, **kwargs)
            
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        return response, duration
    
    # === AUTHENTICATION TESTS ===
    
    async def test_authentication_module(self) -> Dict[str, Any]:
        """Test authentication endpoints."""
        self.log("Testing Authentication Module...")
        results = {"module": "authentication", "tests": [], "status": "PASS"}
        
        # Test 1: User Registration
        self.log("  Testing user registration...")
        user_data = {
            "name": "QA Test User",
            "email": f"qa_test_{int(time.time())}@example.com",
            "password": "QATestPass123!"
        }
        
        response, duration = await self.make_request("POST", "/api/v1/auth/register", json=user_data)
        
        if response.status_code == 201:
            data = response.json()
            self.test_user_id = data.get("id")
            results["tests"].append({
                "name": "user_registration",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"User registered with ID {self.test_user_id}"
            })
            self.log(self.format_result("PASS", f"User registration successful ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "user_registration", 
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}, Response: {response.text}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"User registration failed: {response.status_code}"))
        
        # Test 2: User Login
        self.log("  Testing user login...")
        login_data = {
            "email": user_data["email"],
            "password": user_data["password"]
        }
        
        response, duration = await self.make_request("POST", "/api/v1/auth/login", json=login_data)
        
        if response.status_code == 200:
            data = response.json()
            self.auth_token = data.get("access_token")
            self.auth_headers = {"Authorization": f"Bearer {self.auth_token}"}
            results["tests"].append({
                "name": "user_login",
                "status": "PASS", 
                "duration_ms": duration,
                "details": "Login successful, token obtained"
            })
            self.log(self.format_result("PASS", f"User login successful ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "user_login",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"User login failed: {response.status_code}"))
        
        # Test 3: Token Validation (/me endpoint)
        if self.auth_token:
            self.log("  Testing token validation...")
            response, duration = await self.make_request("GET", "/api/v1/auth/me", headers=self.auth_headers)
            
            if response.status_code == 200:
                results["tests"].append({
                    "name": "token_validation",
                    "status": "PASS",
                    "duration_ms": duration,
                    "details": "Token validation successful"
                })
                self.log(self.format_result("PASS", f"Token validation successful ({duration:.1f}ms)"))
            else:
                results["tests"].append({
                    "name": "token_validation",
                    "status": "FAIL", 
                    "duration_ms": duration,
                    "details": f"Status: {response.status_code}"
                })
                results["status"] = "FAIL"
                self.log(self.format_result("FAIL", f"Token validation failed: {response.status_code}"))
        
        # Test 4: Unauthorized access
        self.log("  Testing unauthorized access protection...")
        response, duration = await self.make_request("GET", "/api/v1/users/profile")
        
        if response.status_code == 401:
            results["tests"].append({
                "name": "unauthorized_protection",
                "status": "PASS",
                "duration_ms": duration,
                "details": "Properly blocks unauthorized access"
            })
            self.log(self.format_result("PASS", f"Unauthorized access blocked ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "unauthorized_protection",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Expected 401, got {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Unauthorized access not blocked: {response.status_code}"))
        
        return results
    
    # === USER MANAGEMENT TESTS ===
    
    async def test_user_management_module(self) -> Dict[str, Any]:
        """Test user management endpoints."""
        self.log("Testing User Management Module...")
        results = {"module": "user_management", "tests": [], "status": "PASS"}
        
        if not self.auth_token:
            results["status"] = "SKIP"
            results["tests"].append({
                "name": "module_skipped",
                "status": "SKIP", 
                "details": "No auth token available"
            })
            return results
        
        # Test 1: Get Profile
        self.log("  Testing get user profile...")
        response, duration = await self.make_request("GET", "/api/v1/users/profile", headers=self.auth_headers)
        
        if response.status_code == 200:
            profile_data = response.json()
            results["tests"].append({
                "name": "get_profile",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"Profile retrieved for user ID {profile_data.get('id')}"
            })
            self.log(self.format_result("PASS", f"Profile retrieval successful ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "get_profile",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Profile retrieval failed: {response.status_code}"))
        
        # Test 2: Update Profile
        self.log("  Testing update user profile...")
        update_data = {"name": "QA Test User Updated"}
        response, duration = await self.make_request("PUT", "/api/v1/users/profile", 
                                                    headers=self.auth_headers, json=update_data)
        
        if response.status_code == 200:
            results["tests"].append({
                "name": "update_profile",
                "status": "PASS",
                "duration_ms": duration,
                "details": "Profile update successful"
            })
            self.log(self.format_result("PASS", f"Profile update successful ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "update_profile",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Profile update failed: {response.status_code}"))
        
        return results
    
    # === STRATEGIC PROFILE TESTS ===
    
    async def test_strategic_profile_module(self) -> Dict[str, Any]:
        """Test strategic profile endpoints."""
        self.log("Testing Strategic Profile Module...")
        results = {"module": "strategic_profile", "tests": [], "status": "PASS"}
        
        if not self.auth_token:
            results["status"] = "SKIP"
            return results
        
        # Test 1: Get Industries Enum (public endpoint)
        self.log("  Testing industries enum endpoint...")
        response, duration = await self.make_request("GET", "/api/v1/strategic-profile/enums/industries")
        
        if response.status_code == 200:
            data = response.json()
            industry_count = len(data.get("industries", []))
            results["tests"].append({
                "name": "industries_enum",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"Retrieved {industry_count} industries"
            })
            self.log(self.format_result("PASS", f"Industries enum retrieved ({industry_count} items, {duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "industries_enum",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Industries enum failed: {response.status_code}"))
        
        # Test 2: Create Strategic Profile
        self.log("  Testing strategic profile creation...")
        profile_data = {
            "industry": "technology",
            "organization_type": "startup",
            "role": "ceo",
            "strategic_goals": ["market_expansion", "product_development"],
            "organization_size": "medium"
        }
        
        response, duration = await self.make_request("POST", "/api/v1/strategic-profile/",
                                                    headers=self.auth_headers, json=profile_data)
        
        if response.status_code == 201:
            results["tests"].append({
                "name": "create_strategic_profile",
                "status": "PASS",
                "duration_ms": duration,
                "details": "Strategic profile created successfully"
            })
            self.log(self.format_result("PASS", f"Strategic profile created ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "create_strategic_profile",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}, Response: {response.text}"
            })
            if response.status_code != 409:  # 409 is acceptable (already exists)
                results["status"] = "FAIL"
            self.log(self.format_result("WARN" if response.status_code == 409 else "FAIL", 
                                      f"Strategic profile creation: {response.status_code}"))
        
        # Test 3: Get Strategic Profile Analytics
        self.log("  Testing strategic profile analytics...")
        response, duration = await self.make_request("GET", "/api/v1/strategic-profile/analytics", 
                                                    headers=self.auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            completeness = data.get("profile_completeness", 0)
            results["tests"].append({
                "name": "strategic_analytics",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"Analytics retrieved, completeness: {completeness}%"
            })
            self.log(self.format_result("PASS", f"Strategic analytics retrieved ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "strategic_analytics",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Strategic analytics failed: {response.status_code}"))
        
        return results
    
    # === FOCUS AREAS TESTS ===
    
    async def test_focus_areas_module(self) -> Dict[str, Any]:
        """Test focus areas endpoints."""
        self.log("Testing Focus Areas Module...")
        results = {"module": "focus_areas", "tests": [], "status": "PASS"}
        
        if not self.auth_token:
            results["status"] = "SKIP"
            return results
        
        # Test 1: Create Focus Area
        self.log("  Testing focus area creation...")
        focus_area_data = {
            "focus_area": "QA Test AI Research",
            "keywords": ["artificial intelligence", "machine learning", "qa testing"],
            "priority": 3
        }
        
        response, duration = await self.make_request("POST", "/api/v1/users/focus-areas/",
                                                    headers=self.auth_headers, json=focus_area_data)
        
        focus_area_id = None
        if response.status_code == 201:
            data = response.json()
            focus_area_id = data.get("id")
            results["tests"].append({
                "name": "create_focus_area",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"Focus area created with ID {focus_area_id}"
            })
            self.log(self.format_result("PASS", f"Focus area created ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "create_focus_area",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Focus area creation failed: {response.status_code}"))
        
        # Test 2: List Focus Areas
        self.log("  Testing focus areas listing...")
        response, duration = await self.make_request("GET", "/api/v1/users/focus-areas/",
                                                    headers=self.auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            total_areas = data.get("total", 0)
            results["tests"].append({
                "name": "list_focus_areas",
                "status": "PASS", 
                "duration_ms": duration,
                "details": f"Retrieved {total_areas} focus areas"
            })
            self.log(self.format_result("PASS", f"Focus areas listed ({total_areas} items, {duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "list_focus_areas",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Focus areas listing failed: {response.status_code}"))
        
        # Test 3: Focus Areas Analytics
        self.log("  Testing focus areas analytics...")
        response, duration = await self.make_request("GET", "/api/v1/users/focus-areas/analytics/summary",
                                                    headers=self.auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            total_areas = data.get("total_focus_areas", 0)
            coverage_score = data.get("coverage_score", 0)
            results["tests"].append({
                "name": "focus_areas_analytics",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"Analytics: {total_areas} areas, {coverage_score}% coverage"
            })
            self.log(self.format_result("PASS", f"Focus areas analytics retrieved ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "focus_areas_analytics",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Focus areas analytics failed: {response.status_code}"))
        
        return results
    
    # === ENTITY TRACKING TESTS ===
    
    async def test_entity_tracking_module(self) -> Dict[str, Any]:
        """Test entity tracking endpoints."""
        self.log("Testing Entity Tracking Module...")
        results = {"module": "entity_tracking", "tests": [], "status": "PASS"}
        
        if not self.auth_token:
            results["status"] = "SKIP"
            return results
        
        # Test 1: Create Tracking Entity
        self.log("  Testing tracking entity creation...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        entity_data = {
            "name": f"QA Test Company {timestamp}",
            "entity_type": "competitor",
            "domain": f"qatest{timestamp}.com",
            "description": "QA testing company",
            "industry": "Technology"
        }
        
        response, duration = await self.make_request("POST", "/api/v1/users/entity-tracking/entities",
                                                    headers=self.auth_headers, json=entity_data)
        
        entity_id = None
        if response.status_code == 201:
            data = response.json()
            entity_id = data.get("id")
            results["tests"].append({
                "name": "create_tracking_entity",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"Entity created with ID {entity_id}"
            })
            self.log(self.format_result("PASS", f"Tracking entity created ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "create_tracking_entity",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Tracking entity creation failed: {response.status_code}"))
        
        # Test 2: Start Tracking Entity
        if entity_id:
            self.log("  Testing entity tracking activation...")
            tracking_data = {
                "entity_id": entity_id,
                "priority": 3,
                "custom_keywords": ["qa", "testing", "validation"],
                "tracking_enabled": True
            }
            
            response, duration = await self.make_request("POST", "/api/v1/users/entity-tracking/",
                                                        headers=self.auth_headers, json=tracking_data)
            
            if response.status_code == 201:
                results["tests"].append({
                    "name": "start_entity_tracking",
                    "status": "PASS",
                    "duration_ms": duration,
                    "details": "Entity tracking activated successfully"
                })
                self.log(self.format_result("PASS", f"Entity tracking started ({duration:.1f}ms)"))
            else:
                results["tests"].append({
                    "name": "start_entity_tracking",
                    "status": "FAIL",
                    "duration_ms": duration,
                    "details": f"Status: {response.status_code}"
                })
                results["status"] = "FAIL"
                self.log(self.format_result("FAIL", f"Entity tracking failed: {response.status_code}"))
        
        # Test 3: Get Available Entities (requires authentication)
        self.log("  Testing available entities listing...")
        response, duration = await self.make_request("GET", "/api/v1/users/entity-tracking/entities",
                                                    headers=self.auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            entity_count = len(data)
            results["tests"].append({
                "name": "list_available_entities",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"Retrieved {entity_count} available entities"
            })
            self.log(self.format_result("PASS", f"Available entities listed ({entity_count} items, {duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "list_available_entities",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Available entities listing failed: {response.status_code}"))
        
        # Test 4: Entity Tracking Analytics
        self.log("  Testing entity tracking analytics...")
        response, duration = await self.make_request("GET", "/api/v1/users/entity-tracking/analytics",
                                                    headers=self.auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            total_tracked = data.get("total_tracked_entities", 0)
            enabled_count = data.get("enabled_count", 0)
            results["tests"].append({
                "name": "entity_tracking_analytics",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"Analytics: {total_tracked} tracked, {enabled_count} enabled"
            })
            self.log(self.format_result("PASS", f"Entity tracking analytics retrieved ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "entity_tracking_analytics",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Entity tracking analytics failed: {response.status_code}"))
        
        return results
    
    # === DELIVERY PREFERENCES TESTS ===
    
    async def test_delivery_preferences_module(self) -> Dict[str, Any]:
        """Test delivery preferences endpoints."""
        self.log("Testing Delivery Preferences Module...")
        results = {"module": "delivery_preferences", "tests": [], "status": "PASS"}
        
        if not self.auth_token:
            results["status"] = "SKIP"
            return results
        
        # Test 1: Get Recommended Defaults
        self.log("  Testing delivery preferences defaults...")
        response, duration = await self.make_request("GET", "/api/v1/users/delivery-preferences/defaults",
                                                    headers=self.auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            recommended_frequency = data.get("recommended_frequency")
            results["tests"].append({
                "name": "delivery_defaults",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"Defaults retrieved, recommended frequency: {recommended_frequency}"
            })
            self.log(self.format_result("PASS", f"Delivery defaults retrieved ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "delivery_defaults",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Delivery defaults failed: {response.status_code}"))
        
        # Test 2: Create/Update Delivery Preferences
        self.log("  Testing delivery preferences configuration...")
        preferences_data = {
            "frequency": "daily",
            "delivery_time": "09:00",
            "timezone": "America/New_York",
            "weekend_delivery": False,
            "max_articles_per_report": 15,
            "min_significance_level": "medium",
            "content_format": "executive_summary",
            "email_enabled": True,
            "urgent_alerts_enabled": True,
            "digest_mode": True
        }
        
        response, duration = await self.make_request("PUT", "/api/v1/users/delivery-preferences/",
                                                    headers=self.auth_headers, json=preferences_data)
        
        if response.status_code == 200:
            results["tests"].append({
                "name": "configure_delivery_preferences",
                "status": "PASS",
                "duration_ms": duration,
                "details": "Delivery preferences configured successfully"
            })
            self.log(self.format_result("PASS", f"Delivery preferences configured ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "configure_delivery_preferences", 
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}, Response: {response.text}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Delivery preferences failed: {response.status_code}"))
        
        # Test 3: Get Delivery Analytics
        self.log("  Testing delivery preferences analytics...")
        response, duration = await self.make_request("GET", "/api/v1/users/delivery-preferences/analytics",
                                                    headers=self.auth_headers)
        
        if response.status_code == 200:
            data = response.json()
            schedule = data.get("delivery_schedule", "Not configured")
            results["tests"].append({
                "name": "delivery_analytics",
                "status": "PASS",
                "duration_ms": duration,
                "details": f"Analytics retrieved, schedule: {schedule}"
            })
            self.log(self.format_result("PASS", f"Delivery analytics retrieved ({duration:.1f}ms)"))
        else:
            results["tests"].append({
                "name": "delivery_analytics",
                "status": "FAIL",
                "duration_ms": duration,
                "details": f"Status: {response.status_code}"
            })
            results["status"] = "FAIL"
            self.log(self.format_result("FAIL", f"Delivery analytics failed: {response.status_code}"))
        
        return results
    
    # === COMPREHENSIVE QA EXECUTION ===
    
    async def run_comprehensive_qa(self) -> Dict[str, Any]:
        """Run comprehensive QA validation across all modules."""
        self.log("=== STARTING COMPREHENSIVE QA VALIDATION ===")
        start_time = time.time()
        
        qa_results = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_ms": 0,
            "modules": [],
            "summary": {
                "total_modules": 6,
                "modules_passed": 0,
                "modules_failed": 0,
                "modules_skipped": 0,
                "total_tests": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0
            }
        }
        
        # Run all module tests
        modules_to_test = [
            ("Authentication", self.test_authentication_module),
            ("User Management", self.test_user_management_module),
            ("Strategic Profile", self.test_strategic_profile_module),
            ("Focus Areas", self.test_focus_areas_module),
            ("Entity Tracking", self.test_entity_tracking_module),
            ("Delivery Preferences", self.test_delivery_preferences_module)
        ]
        
        for module_name, test_func in modules_to_test:
            try:
                module_results = await test_func()
                qa_results["modules"].append(module_results)
                
                # Update summary
                if module_results["status"] == "PASS":
                    qa_results["summary"]["modules_passed"] += 1
                elif module_results["status"] == "FAIL":
                    qa_results["summary"]["modules_failed"] += 1
                elif module_results["status"] == "SKIP":
                    qa_results["summary"]["modules_skipped"] += 1
                
                # Count tests
                for test in module_results["tests"]:
                    qa_results["summary"]["total_tests"] += 1
                    if test["status"] == "PASS":
                        qa_results["summary"]["tests_passed"] += 1
                    elif test["status"] == "FAIL":
                        qa_results["summary"]["tests_failed"] += 1
                    elif test["status"] == "SKIP":
                        qa_results["summary"]["tests_skipped"] += 1
                        
            except Exception as e:
                self.log(self.format_result("ERROR", f"Module {module_name} failed with exception: {str(e)}"))
                qa_results["modules"].append({
                    "module": module_name.lower().replace(" ", "_"),
                    "status": "ERROR",
                    "error": str(e),
                    "tests": []
                })
                qa_results["summary"]["modules_failed"] += 1
        
        end_time = time.time()
        qa_results["total_duration_ms"] = (end_time - start_time) * 1000
        
        self.log("=== QA VALIDATION COMPLETED ===")
        return qa_results


async def main():
    """Main QA validation execution."""
    validator = QAValidator()
    
    print("=" * 80)
    print("COMPREHENSIVE QA VALIDATION - USER CONFIG SERVICE")
    print("=" * 80)
    print(f"Target: {validator.base_url}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run comprehensive QA
    results = await validator.run_comprehensive_qa()
    
    # Generate final report
    print("\n" + "=" * 80)
    print("QA VALIDATION SUMMARY REPORT")
    print("=" * 80)
    
    summary = results["summary"]
    total_duration = results["total_duration_ms"]
    
    print(f"Total Duration: {total_duration:.1f}ms ({total_duration/1000:.2f}s)")
    print(f"Modules Tested: {summary['total_modules']}")
    print(f"  - Passed: {summary['modules_passed']}")
    print(f"  - Failed: {summary['modules_failed']}")
    print(f"  - Skipped: {summary['modules_skipped']}")
    print(f"Tests Executed: {summary['total_tests']}")
    print(f"  - Passed: {summary['tests_passed']}")
    print(f"  - Failed: {summary['tests_failed']}")
    print(f"  - Skipped: {summary['tests_skipped']}")
    
    # Module-by-module results
    print("\n" + "-" * 80)
    print("MODULE-BY-MODULE RESULTS")
    print("-" * 80)
    
    for module in results["modules"]:
        module_name = module["module"].replace("_", " ").title()
        status = module["status"]
        test_count = len(module["tests"])
        
        status_symbol = {
            "PASS": "[PASS]",
            "FAIL": "[FAIL]", 
            "SKIP": "[SKIP]",
            "ERROR": "[ERROR]"
        }.get(status, "[UNKNOWN]")
        
        print(f"{status_symbol} {module_name} ({test_count} tests)")
        
        for test in module["tests"]:
            test_status = {
                "PASS": "[PASS]",
                "FAIL": "[FAIL]",
                "SKIP": "[SKIP]"
            }.get(test["status"], "[UNKNOWN]")
            
            duration = test.get("duration_ms", 0)
            details = test.get("details", "")
            print(f"    {test_status} {test['name']} ({duration:.1f}ms) - {details}")
    
    # Overall result
    print("\n" + "=" * 80)
    overall_status = "PASS" if summary["modules_failed"] == 0 else "FAIL"
    success_rate = (summary["tests_passed"] / max(summary["total_tests"], 1)) * 100
    
    print(f"OVERALL QA STATUS: {overall_status}")
    print(f"SUCCESS RATE: {success_rate:.1f}% ({summary['tests_passed']}/{summary['total_tests']})")
    print("=" * 80)
    
    # Save detailed results
    with open("qa_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: qa_validation_results.json")
    
    return overall_status == "PASS"


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
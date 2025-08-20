#!/usr/bin/env python3
"""
Comprehensive API Testing Suite
Tests all FastAPI endpoints with actual HTTP requests and full integration validation
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import httpx

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ComprehensiveAPITester:
    """Complete API testing with HTTP requests."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8002"):
        self.base_url = base_url
        self.results = []
        self.start_time = time.time()
        self.access_token = None
        self.user_id = None
        self.test_user_email = f"apitest_{int(time.time())}@example.com"
        self.test_password = "TestPass123!"
    
    def log_test(self, name: str, success: bool, message: str, details: str = "", response_time: float = 0):
        """Log test result."""
        self.results.append({
            'name': name,
            'success': success,
            'message': message,
            'details': details,
            'response_time': response_time,
            'timestamp': datetime.now()
        })
        status = "PASS" if success else "FAIL"
        timing = f" ({response_time:.3f}s)" if response_time > 0 else ""
        print(f"[{status}] {name}: {message}{timing}")
        if details and not success:
            print(f"    Details: {details}")
    
    async def start_test_server(self):
        """Check if test server is running, if not provide guidance."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    self.log_test("Server Connection", True, "FastAPI server is running and accessible")
                    return True
                else:
                    self.log_test("Server Connection", False, f"Server responded with status {response.status_code}")
                    return False
        except Exception as e:
            self.log_test("Server Connection", False, "FastAPI server not accessible", str(e))
            print("\nTo run these tests, start the FastAPI server first:")
            print("python app/main.py")
            print("Then run this test suite again.")
            return False
    
    async def test_system_endpoints(self):
        """Test system endpoints."""
        print("\n=== Testing System Endpoints ===")
        
        async with httpx.AsyncClient() as client:
            # Test root endpoint
            start_time = time.time()
            try:
                response = await client.get(f"{self.base_url}/")
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    if "service" in data and "version" in data:
                        self.log_test("Root Endpoint", True, f"Service: {data.get('service')}, Version: {data.get('version')}", response_time=response_time)
                    else:
                        self.log_test("Root Endpoint", False, "Missing required fields in response", json.dumps(data))
                else:
                    self.log_test("Root Endpoint", False, f"HTTP {response.status_code}")
            except Exception as e:
                self.log_test("Root Endpoint", False, "Request failed", str(e))
            
            # Test health endpoint
            start_time = time.time()
            try:
                response = await client.get(f"{self.base_url}/health")
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        self.log_test("Health Check Endpoint", True, f"Database: {data.get('database')}", response_time=response_time)
                    else:
                        self.log_test("Health Check Endpoint", False, f"Unhealthy status: {data.get('status')}")
                else:
                    self.log_test("Health Check Endpoint", False, f"HTTP {response.status_code}")
            except Exception as e:
                self.log_test("Health Check Endpoint", False, "Request failed", str(e))
    
    async def test_authentication_flow(self):
        """Test complete authentication flow."""
        print("\n=== Testing Authentication Flow ===")
        
        async with httpx.AsyncClient() as client:
            # Test user registration
            start_time = time.time()
            try:
                register_data = {
                    "email": self.test_user_email,
                    "name": "API Test User",
                    "password": self.test_password
                }
                response = await client.post(f"{self.base_url}/api/v1/auth/register", json=register_data)
                response_time = time.time() - start_time
                
                if response.status_code == 201:
                    user_data = response.json()
                    self.user_id = user_data.get("id")
                    self.log_test("User Registration", True, f"User created with ID: {self.user_id}", response_time=response_time)
                elif response.status_code == 400 and "already exists" in response.text:
                    self.log_test("User Registration", True, "User already exists (acceptable for testing)", response_time=response_time)
                else:
                    self.log_test("User Registration", False, f"HTTP {response.status_code}", response.text)
                    return False
            except Exception as e:
                self.log_test("User Registration", False, "Request failed", str(e))
                return False
            
            # Test user login
            start_time = time.time()
            try:
                login_data = {
                    "email": self.test_user_email,
                    "password": self.test_password,
                    "remember_me": False
                }
                response = await client.post(f"{self.base_url}/api/v1/auth/login", json=login_data)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.access_token = token_data.get("access_token")
                    if self.access_token:
                        self.log_test("User Login", True, f"JWT token received (expires in {token_data.get('expires_in')}s)", response_time=response_time)
                    else:
                        self.log_test("User Login", False, "No access token in response", json.dumps(token_data))
                        return False
                else:
                    self.log_test("User Login", False, f"HTTP {response.status_code}", response.text)
                    return False
            except Exception as e:
                self.log_test("User Login", False, "Request failed", str(e))
                return False
            
            # Test invalid login
            start_time = time.time()
            try:
                invalid_login = {
                    "email": self.test_user_email,
                    "password": "WrongPassword123!",
                    "remember_me": False
                }
                response = await client.post(f"{self.base_url}/api/v1/auth/login", json=invalid_login)
                response_time = time.time() - start_time
                
                if response.status_code == 401:
                    self.log_test("Invalid Login Rejection", True, "Wrong password correctly rejected", response_time=response_time)
                else:
                    self.log_test("Invalid Login Rejection", False, f"Wrong password accepted - HTTP {response.status_code}")
            except Exception as e:
                self.log_test("Invalid Login Rejection", False, "Request failed", str(e))
            
            return True
    
    async def test_protected_endpoints(self):
        """Test protected endpoints requiring authentication."""
        print("\n=== Testing Protected Endpoints ===")
        
        if not self.access_token:
            self.log_test("Protected Endpoints", False, "No access token available for testing")
            return
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with httpx.AsyncClient() as client:
            # Test get user profile
            start_time = time.time()
            try:
                response = await client.get(f"{self.base_url}/api/v1/users/profile", headers=headers)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    profile_data = response.json()
                    if "id" in profile_data and "email" in profile_data:
                        self.log_test("Get User Profile", True, f"Profile retrieved for user {profile_data.get('email')}", response_time=response_time)
                    else:
                        self.log_test("Get User Profile", False, "Invalid profile response format", json.dumps(profile_data))
                else:
                    self.log_test("Get User Profile", False, f"HTTP {response.status_code}", response.text)
            except Exception as e:
                self.log_test("Get User Profile", False, "Request failed", str(e))
            
            # Test update user profile
            start_time = time.time()
            try:
                update_data = {"name": "Updated API Test User"}
                response = await client.put(f"{self.base_url}/api/v1/users/profile", json=update_data, headers=headers)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    updated_data = response.json()
                    if updated_data.get("name") == "Updated API Test User":
                        self.log_test("Update User Profile", True, "Profile updated successfully", response_time=response_time)
                    else:
                        self.log_test("Update User Profile", False, "Profile not updated correctly", json.dumps(updated_data))
                else:
                    self.log_test("Update User Profile", False, f"HTTP {response.status_code}", response.text)
            except Exception as e:
                self.log_test("Update User Profile", False, "Request failed", str(e))
            
            # Test create strategic profile
            start_time = time.time()
            try:
                strategic_data = {
                    "industry": "Technology",
                    "organization_type": "Startup",
                    "role": "API Tester",
                    "strategic_goals": ["API Testing", "Quality Assurance"],
                    "organization_size": "small"
                }
                response = await client.post(f"{self.base_url}/api/v1/users/strategic-profile", json=strategic_data, headers=headers)
                response_time = time.time() - start_time
                
                if response.status_code == 201:
                    strategic_profile = response.json()
                    if strategic_profile.get("industry") == "Technology":
                        self.log_test("Create Strategic Profile", True, f"Strategic profile created for {strategic_profile.get('role')}", response_time=response_time)
                    else:
                        self.log_test("Create Strategic Profile", False, "Strategic profile not created correctly", json.dumps(strategic_profile))
                elif response.status_code == 400 and "already exists" in response.text:
                    self.log_test("Create Strategic Profile", True, "Strategic profile already exists (acceptable)", response_time=response_time)
                else:
                    self.log_test("Create Strategic Profile", False, f"HTTP {response.status_code}", response.text)
            except Exception as e:
                self.log_test("Create Strategic Profile", False, "Request failed", str(e))
            
            # Test get strategic profile
            start_time = time.time()
            try:
                response = await client.get(f"{self.base_url}/api/v1/users/strategic-profile", headers=headers)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    profile = response.json()
                    if "industry" in profile:
                        self.log_test("Get Strategic Profile", True, f"Retrieved profile for {profile.get('industry')} industry", response_time=response_time)
                    else:
                        self.log_test("Get Strategic Profile", False, "Invalid strategic profile format", json.dumps(profile))
                elif response.status_code == 404:
                    self.log_test("Get Strategic Profile", True, "No strategic profile found (acceptable)", response_time=response_time)
                else:
                    self.log_test("Get Strategic Profile", False, f"HTTP {response.status_code}", response.text)
            except Exception as e:
                self.log_test("Get Strategic Profile", False, "Request failed", str(e))
    
    async def test_unauthorized_access(self):
        """Test that protected endpoints reject unauthorized requests."""
        print("\n=== Testing Unauthorized Access Protection ===")
        
        async with httpx.AsyncClient() as client:
            protected_endpoints = [
                ("GET", "/api/v1/users/profile"),
                ("PUT", "/api/v1/users/profile"),
                ("GET", "/api/v1/users/strategic-profile"),
                ("POST", "/api/v1/users/strategic-profile"),
                ("DELETE", "/api/v1/users/account")
            ]
            
            for method, endpoint in protected_endpoints:
                start_time = time.time()
                try:
                    if method == "GET":
                        response = await client.get(f"{self.base_url}{endpoint}")
                    elif method == "PUT":
                        response = await client.put(f"{self.base_url}{endpoint}", json={})
                    elif method == "POST":
                        response = await client.post(f"{self.base_url}{endpoint}", json={})
                    elif method == "DELETE":
                        response = await client.delete(f"{self.base_url}{endpoint}")
                    
                    response_time = time.time() - start_time
                    
                    if response.status_code == 401:
                        self.log_test(f"Unauthorized Access {method} {endpoint}", True, "Correctly rejected unauthorized request", response_time=response_time)
                    else:
                        self.log_test(f"Unauthorized Access {method} {endpoint}", False, f"Should reject but returned HTTP {response.status_code}")
                except Exception as e:
                    self.log_test(f"Unauthorized Access {method} {endpoint}", False, "Request failed", str(e))
    
    async def test_input_validation(self):
        """Test input validation across endpoints."""
        print("\n=== Testing Input Validation ===")
        
        async with httpx.AsyncClient() as client:
            # Test invalid registration data
            invalid_registrations = [
                ({"email": "invalid-email", "name": "Test", "password": "Test123!"}, "Invalid email format"),
                ({"email": "test@example.com", "name": "", "password": "Test123!"}, "Empty name"),
                ({"email": "test@example.com", "name": "Test", "password": "weak"}, "Weak password"),
                ({"name": "Test", "password": "Test123!"}, "Missing email")
            ]
            
            for invalid_data, description in invalid_registrations:
                start_time = time.time()
                try:
                    response = await client.post(f"{self.base_url}/api/v1/auth/register", json=invalid_data)
                    response_time = time.time() - start_time
                    
                    if response.status_code in [400, 422]:
                        self.log_test(f"Input Validation - {description}", True, f"Invalid data correctly rejected (HTTP {response.status_code})", response_time=response_time)
                    else:
                        self.log_test(f"Input Validation - {description}", False, f"Invalid data accepted - HTTP {response.status_code}")
                except Exception as e:
                    self.log_test(f"Input Validation - {description}", False, "Request failed", str(e))
    
    async def test_error_handling(self):
        """Test error handling for various scenarios."""
        print("\n=== Testing Error Handling ===")
        
        async with httpx.AsyncClient() as client:
            # Test 404 for non-existent endpoint
            start_time = time.time()
            try:
                response = await client.get(f"{self.base_url}/api/v1/nonexistent")
                response_time = time.time() - start_time
                
                if response.status_code == 404:
                    error_data = response.json()
                    if "detail" in error_data and "type" in error_data:
                        self.log_test("404 Error Handling", True, "Non-existent endpoint correctly returns 404 with proper error format", response_time=response_time)
                    else:
                        self.log_test("404 Error Handling", False, "404 response missing proper error format", json.dumps(error_data))
                else:
                    self.log_test("404 Error Handling", False, f"Expected 404 but got HTTP {response.status_code}")
            except Exception as e:
                self.log_test("404 Error Handling", False, "Request failed", str(e))
            
            # Test invalid JSON handling
            start_time = time.time()
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/register",
                    content="invalid json",
                    headers={"Content-Type": "application/json"}
                )
                response_time = time.time() - start_time
                
                if response.status_code in [400, 422]:
                    self.log_test("Invalid JSON Handling", True, f"Invalid JSON correctly rejected (HTTP {response.status_code})", response_time=response_time)
                else:
                    self.log_test("Invalid JSON Handling", False, f"Invalid JSON accepted - HTTP {response.status_code}")
            except Exception as e:
                self.log_test("Invalid JSON Handling", False, "Request failed", str(e))
    
    async def test_performance_benchmarks(self):
        """Test API performance benchmarks."""
        print("\n=== Testing Performance Benchmarks ===")
        
        performance_tests = []
        
        async with httpx.AsyncClient() as client:
            # Benchmark multiple requests
            for i in range(5):
                start_time = time.time()
                try:
                    response = await client.get(f"{self.base_url}/health")
                    response_time = time.time() - start_time
                    performance_tests.append(response_time)
                except Exception:
                    pass
            
            if performance_tests:
                avg_response_time = sum(performance_tests) / len(performance_tests)
                max_response_time = max(performance_tests)
                min_response_time = min(performance_tests)
                
                if avg_response_time < 1.0:  # Less than 1 second average
                    self.log_test("API Performance", True, 
                                f"Average: {avg_response_time:.3f}s, Min: {min_response_time:.3f}s, Max: {max_response_time:.3f}s",
                                response_time=avg_response_time)
                else:
                    self.log_test("API Performance", False, 
                                f"Slow response times - Average: {avg_response_time:.3f}s")
            else:
                self.log_test("API Performance", False, "Could not measure performance")
    
    async def cleanup_test_data(self):
        """Clean up test data if possible."""
        if self.access_token:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            async with httpx.AsyncClient() as client:
                try:
                    # Attempt to logout
                    await client.post(f"{self.base_url}/api/v1/auth/logout", headers=headers)
                    self.log_test("Test Cleanup", True, "Test session logged out successfully")
                except Exception:
                    self.log_test("Test Cleanup", True, "Test cleanup attempted (logout may have failed)")
    
    def print_comprehensive_report(self):
        """Print detailed test report."""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("COMPETITIVE INTELLIGENCE V2 - COMPREHENSIVE API TEST REPORT")
        print("="*80)
        print(f"Test Duration: {total_time:.2f} seconds")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result['success'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total API Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        # Performance statistics
        response_times = [r['response_time'] for r in self.results if r['response_time'] > 0]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            print(f"  Average Response Time: {avg_time:.3f}s")
        
        # Determine status
        if failed_tests == 0:
            status = "[HEALTHY] ALL API ENDPOINTS FUNCTIONAL"
        elif failed_tests <= 2:
            status = "[MINOR ISSUES] MOSTLY FUNCTIONAL"
        else:
            status = "[CRITICAL] SIGNIFICANT API ISSUES"
        
        print(f"\nAPI STATUS: {status}")
        
        # Show failed tests
        if failed_tests > 0:
            print(f"\nFAILED TESTS:")
            for result in self.results:
                if not result['success']:
                    print(f"  ! {result['name']}: {result['message']}")
                    if result['details']:
                        print(f"    {result['details']}")
        
        print("\n" + "="*80)
        
        return failed_tests == 0
    
    async def run_comprehensive_tests(self):
        """Run all comprehensive API tests."""
        print("COMPETITIVE INTELLIGENCE V2 - COMPREHENSIVE API TESTING")
        print("="*70)
        
        # Check server connectivity first
        if not await self.start_test_server():
            return False
        
        test_methods = [
            self.test_system_endpoints,
            self.test_authentication_flow,
            self.test_protected_endpoints,
            self.test_unauthorized_access,
            self.test_input_validation,
            self.test_error_handling,
            self.test_performance_benchmarks
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                self.log_test(test_method.__name__, False, f"Test suite failed: {e}")
        
        # Cleanup
        await self.cleanup_test_data()
        
        return self.print_comprehensive_report()


async def main():
    """Run comprehensive API tests."""
    tester = ComprehensiveAPITester()
    
    try:
        success = await tester.run_comprehensive_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] API testing stopped by user")
        return 1
    except Exception as e:
        print(f"\n\n[ERROR] API testing system failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
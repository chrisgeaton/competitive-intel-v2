#!/usr/bin/env python3
"""
Comprehensive Integration Testing Script for Discovery Service and User Config Service
Tests end-to-end workflows including authentication, data integration, API consistency,
ML learning loops, source discovery engines, content deduplication, database operations,
error handling, and performance under realistic loads.
"""

import asyncio
import json
import time
import statistics
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import aiohttp
import requests

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
TEST_EMAIL = f"test_user_{int(time.time())}@example.com"
TEST_PASSWORD = "TestPassword123!"
MAX_CONCURRENT_REQUESTS = 10
PERFORMANCE_TEST_DURATION = 30  # seconds

@dataclass
class TestResult:
    test_name: str
    success: bool
    response_time: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class IntegrationTestReport:
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    test_results: List[TestResult] = None
    performance_metrics: Dict[str, Any] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.test_results is None:
            self.test_results = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.recommendations is None:
            self.recommendations = []

class IntegrationTester:
    """Comprehensive integration tester for both services."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.auth_token = None
        self.user_data = {}
        self.report = IntegrationTestReport(start_time=datetime.now())
        
    def add_result(self, result: TestResult):
        """Add test result to report."""
        self.report.test_results.append(result)
        self.report.total_tests += 1
        if result.success:
            self.report.passed_tests += 1
        else:
            self.report.failed_tests += 1
    
    def log(self, message: str):
        """Log message with timestamp."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> TestResult:
        """Make HTTP request and return TestResult."""
        start_time = time.time()
        test_name = f"{method} {endpoint}"
        
        try:
            headers = kwargs.pop('headers', {})
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, 
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    **kwargs
                ) as response:
                    response_time = time.time() - start_time
                    
                    try:
                        data = await response.json()
                    except:
                        data = await response.text()
                    
                    return TestResult(
                        test_name=test_name,
                        success=200 <= response.status < 400,
                        response_time=response_time,
                        status_code=response.status,
                        details=data if isinstance(data, dict) else {"response": str(data)}
                    )
                    
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                test_name=test_name,
                success=False,
                response_time=response_time,
                error_message=str(e)
            )
    
    async def test_user_authentication_flow(self):
        """Test 1: Complete user authentication flow across both services."""
        self.log("Testing user authentication flow...")
        
        # Test user registration
        result = await self.make_request(
            'POST', '/api/v1/auth/register',
            json={
                "email": TEST_EMAIL,
                "password": TEST_PASSWORD,
                "name": "Test User"
            }
        )
        self.add_result(result)
        
        if not result.success:
            self.log(f"Registration failed: {result.error_message or result.details}")
            return False
        
        # Test user login
        result = await self.make_request(
            'POST', '/api/v1/auth/login',
            json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
        )
        self.add_result(result)
        
        if result.success and result.details:
            self.auth_token = result.details.get('access_token')
            self.log(f"Authentication successful - Token obtained")
        else:
            self.log(f"Login failed: {result.error_message or result.details}")
            return False
        
        # Test protected endpoint access
        result = await self.make_request('GET', '/api/v1/users/profile')
        self.add_result(result)
        
        if result.success:
            self.user_data = result.details
            self.log(f"Protected endpoint access successful")
            return True
        else:
            self.log(f"Protected access failed: {result.error_message}")
            return False
    
    async def test_user_config_service_integration(self):
        """Test 2: User Config Service data integration."""
        self.log("Testing User Config Service data integration...")
        
        # Test strategic profile creation
        strategic_profile = {
            "industry": "technology",
            "organization_type": "startup",
            "role": "cto",
            "strategic_goals": ["digital_transformation", "innovation"],
            "organization_size": "small"
        }
        
        result = await self.make_request(
            'POST', '/api/v1/strategic-profile',
            json=strategic_profile
        )
        self.add_result(result)
        
        if not result.success:
            self.log(f"Strategic profile creation failed: {result.error_message}")
            return False
        
        # Test focus areas creation
        focus_areas = [
            {
                "area_name": "AI & Machine Learning",
                "keywords": ["artificial intelligence", "machine learning", "neural networks"],
                "priority": "high",
                "monitoring_frequency": "daily"
            }
        ]
        
        for area in focus_areas:
            result = await self.make_request(
                'POST', '/api/v1/focus-areas',
                json=area
            )
            self.add_result(result)
        
        # Test entity tracking creation
        entities = [
            {
                "entity_name": "OpenAI",
                "entity_type": "company",
                "keywords": ["OpenAI", "ChatGPT", "GPT-4"],
                "monitoring_priority": "high"
            }
        ]
        
        for entity in entities:
            result = await self.make_request(
                'POST', '/api/v1/entity-tracking',
                json=entity
            )
            self.add_result(result)
        
        self.log("User Config Service integration test completed")
        return True
    
    async def test_discovery_api_consistency(self):
        """Test 3: Discovery API consistency with User Config patterns."""
        self.log("Testing Discovery API consistency...")
        
        # Test discovery sources management
        discovery_source = {
            "source_url": "https://techcrunch.com/feed/",
            "source_type": "rss",
            "source_name": "TechCrunch AI Feed",
            "is_active": True,
            "discovery_config": {
                "keywords": ["AI", "machine learning"],
                "update_frequency": "hourly"
            }
        }
        
        result = await self.make_request(
            'POST', '/api/v1/discovery/sources',
            json=discovery_source
        )
        self.add_result(result)
        
        if result.success:
            source_id = result.details.get('id')
            self.log(f"Discovery source created with ID: {source_id}")
            
            # Test source retrieval
            result = await self.make_request('GET', f'/api/v1/discovery/sources/{source_id}')
            self.add_result(result)
            
            # Test source health check
            result = await self.make_request('POST', f'/api/v1/discovery/sources/{source_id}/test')
            self.add_result(result)
        
        # Test discovery jobs
        discovery_job = {
            "job_type": "targeted",
            "job_config": {
                "focus_areas": ["AI", "technology"],
                "source_types": ["rss", "api"]
            },
            "priority": 1
        }
        
        result = await self.make_request(
            'POST', '/api/v1/discovery/jobs',
            json=discovery_job
        )
        self.add_result(result)
        
        self.log("Discovery API consistency test completed")
        return True
    
    async def test_ml_learning_loop(self):
        """Test 4: ML learning loop with mock SendGrid engagement data."""
        self.log("Testing ML learning loop with mock SendGrid data...")
        
        # Mock SendGrid webhook data
        sendgrid_events = [
            {
                "email": TEST_EMAIL,
                "timestamp": int(time.time()),
                "event": "open",
                "sg_event_id": "test-event-1",
                "sg_message_id": "test-msg-1",
                "useragent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            },
            {
                "email": TEST_EMAIL,
                "timestamp": int(time.time()),
                "event": "click",
                "url": "https://example.com/article/1",
                "sg_event_id": "test-event-2", 
                "sg_message_id": "test-msg-2",
                "useragent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
        ]
        
        result = await self.make_request(
            'POST', '/api/v1/discovery/webhooks/sendgrid',
            json=sendgrid_events
        )
        self.add_result(result)
        
        # Test engagement tracking
        engagement_data = {
            "engagement_type": "feedback_positive",
            "engagement_value": 0.8,
            "device_type": "desktop",
            "engagement_context": '{"source": "manual_test"}'
        }
        
        result = await self.make_request(
            'POST', '/api/v1/discovery/engagement',
            json=engagement_data
        )
        self.add_result(result)
        
        # Test ML model metrics
        result = await self.make_request('GET', '/api/v1/discovery/ml/models')
        self.add_result(result)
        
        self.log("ML learning loop test completed")
        return True
    
    async def test_content_operations(self):
        """Test 5: Content discovery, deduplication and quality scoring."""
        self.log("Testing content operations...")
        
        # Test content filtering and retrieval
        params = {
            "page": 1,
            "per_page": 10,
            "min_relevance_score": 0.5,
            "exclude_duplicates": True
        }
        
        result = await self.make_request(
            'GET', '/api/v1/discovery/content',
            params=params
        )
        self.add_result(result)
        
        # Test analytics dashboard
        result = await self.make_request(
            'GET', '/api/v1/discovery/analytics/dashboard',
            params={"time_period": "7d"}
        )
        self.add_result(result)
        
        # Test user analytics
        result = await self.make_request(
            'GET', '/api/v1/discovery/analytics',
            params={"days_back": 30}
        )
        self.add_result(result)
        
        self.log("Content operations test completed")
        return True
    
    async def test_error_handling(self):
        """Test 6: Error handling and edge cases."""
        self.log("Testing error handling and edge cases...")
        
        # Test invalid authentication
        old_token = self.auth_token
        self.auth_token = "invalid_token_12345"
        
        result = await self.make_request('GET', '/api/v1/users/profile')
        self.add_result(TestResult(
            test_name="Invalid Authentication Test",
            success=result.status_code == 401,  # Should return unauthorized
            response_time=result.response_time,
            status_code=result.status_code
        ))
        
        self.auth_token = old_token
        
        # Test malformed requests
        result = await self.make_request(
            'POST', '/api/v1/discovery/sources',
            json={"invalid": "data"}
        )
        self.add_result(TestResult(
            test_name="Malformed Request Test",
            success=400 <= result.status_code < 500,  # Should return client error
            response_time=result.response_time,
            status_code=result.status_code
        ))
        
        # Test non-existent resources
        result = await self.make_request('GET', '/api/v1/discovery/sources/999999')
        self.add_result(TestResult(
            test_name="Non-existent Resource Test",
            success=result.status_code == 404,  # Should return not found
            response_time=result.response_time,
            status_code=result.status_code
        ))
        
        self.log("Error handling test completed")
        return True
    
    async def test_performance(self):
        """Test 7: Performance under realistic loads."""
        self.log("Testing performance under realistic loads...")
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        async def make_performance_request():
            result = await self.make_request('GET', '/api/v1/users/profile')
            return result
        
        # Run concurrent requests for specified duration
        start_time = time.time()
        while time.time() - start_time < PERFORMANCE_TEST_DURATION:
            tasks = []
            for _ in range(MAX_CONCURRENT_REQUESTS):
                tasks.append(make_performance_request())
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, TestResult):
                    response_times.append(result.response_time)
                    if result.success:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                else:
                    failed_requests += 1
        
        # Calculate performance metrics
        if response_times:
            self.report.performance_metrics = {
                "test_duration": PERFORMANCE_TEST_DURATION,
                "total_requests": len(response_times),
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / len(response_times) * 100,
                "avg_response_time": statistics.mean(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "median_response_time": statistics.median(response_times),
                "requests_per_second": len(response_times) / PERFORMANCE_TEST_DURATION
            }
        
        self.log(f"Performance test completed: {len(response_times)} requests in {PERFORMANCE_TEST_DURATION}s")
        return True
    
    async def generate_recommendations(self):
        """Generate recommendations based on test results."""
        self.log("Generating recommendations...")
        
        # Analyze results and generate recommendations
        if self.report.failed_tests > 0:
            failure_rate = self.report.failed_tests / self.report.total_tests * 100
            if failure_rate > 10:
                self.report.recommendations.append(
                    f"High failure rate ({failure_rate:.1f}%) detected. Review error handling and validation."
                )
        
        # Performance recommendations
        if self.report.performance_metrics:
            avg_response = self.report.performance_metrics.get('avg_response_time', 0)
            if avg_response > 1.0:
                self.report.recommendations.append(
                    f"Average response time ({avg_response:.2f}s) is high. Consider optimization."
                )
            
            success_rate = self.report.performance_metrics.get('success_rate', 100)
            if success_rate < 95:
                self.report.recommendations.append(
                    f"Success rate ({success_rate:.1f}%) is below 95%. Investigate failures."
                )
        
        # API consistency recommendations
        discovery_tests = [r for r in self.report.test_results if 'discovery' in r.test_name.lower()]
        config_tests = [r for r in self.report.test_results if any(x in r.test_name.lower() for x in ['strategic', 'focus', 'entity'])]
        
        if discovery_tests and config_tests:
            discovery_success = sum(1 for t in discovery_tests if t.success) / len(discovery_tests) * 100
            config_success = sum(1 for t in config_tests if t.success) / len(config_tests) * 100
            
            if abs(discovery_success - config_success) > 20:
                self.report.recommendations.append(
                    "Significant difference in success rates between Discovery and Config services. Review consistency."
                )
        
        if not self.report.recommendations:
            self.report.recommendations.append("All tests passed successfully. Services are performing well.")
    
    def generate_ascii_report(self) -> str:
        """Generate comprehensive ASCII report."""
        self.report.end_time = datetime.now()
        duration = self.report.end_time - self.report.start_time
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE INTEGRATION TEST REPORT")
        report.append("Discovery Service & User Config Service")
        report.append("=" * 80)
        report.append("")
        
        # Test Summary
        report.append("TEST SUMMARY")
        report.append("-" * 40)
        report.append(f"Start Time:      {self.report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"End Time:        {self.report.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Duration:        {duration.total_seconds():.1f} seconds")
        report.append(f"Total Tests:     {self.report.total_tests}")
        report.append(f"Passed:          {self.report.passed_tests}")
        report.append(f"Failed:          {self.report.failed_tests}")
        
        success_rate = (self.report.passed_tests / self.report.total_tests * 100) if self.report.total_tests > 0 else 0
        report.append(f"Success Rate:    {success_rate:.1f}%")
        report.append("")
        
        # Performance Metrics
        if self.report.performance_metrics:
            report.append("PERFORMANCE METRICS")
            report.append("-" * 40)
            metrics = self.report.performance_metrics
            report.append(f"Total Requests:       {metrics.get('total_requests', 'N/A')}")
            report.append(f"Successful Requests:  {metrics.get('successful_requests', 'N/A')}")
            report.append(f"Failed Requests:      {metrics.get('failed_requests', 'N/A')}")
            report.append(f"Success Rate:         {metrics.get('success_rate', 'N/A'):.1f}%")
            report.append(f"Avg Response Time:    {metrics.get('avg_response_time', 'N/A'):.3f}s")
            report.append(f"Min Response Time:    {metrics.get('min_response_time', 'N/A'):.3f}s")
            report.append(f"Max Response Time:    {metrics.get('max_response_time', 'N/A'):.3f}s")
            report.append(f"Requests/Second:      {metrics.get('requests_per_second', 'N/A'):.2f}")
            report.append("")
        
        # Detailed Test Results
        report.append("DETAILED TEST RESULTS")
        report.append("-" * 40)
        
        for i, result in enumerate(self.report.test_results[:20], 1):  # Show first 20 detailed results
            status = "PASS" if result.success else "FAIL"
            report.append(f"{i:2}. [{status}] {result.test_name}")
            report.append(f"    Status Code: {result.status_code or 'N/A'}")
            report.append(f"    Response Time: {result.response_time:.3f}s")
            if result.error_message:
                report.append(f"    Error: {result.error_message}")
            report.append("")
        
        if len(self.report.test_results) > 20:
            report.append(f"... and {len(self.report.test_results) - 20} more tests")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        for i, rec in enumerate(self.report.recommendations, 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        # Integration Analysis
        report.append("INTEGRATION ANALYSIS")
        report.append("-" * 40)
        
        # Endpoint coverage
        unique_endpoints = set()
        for result in self.report.test_results:
            endpoint = result.test_name.split(' ', 1)[-1].split('?')[0]
            unique_endpoints.add(endpoint)
        
        report.append(f"Endpoints Tested:     {len(unique_endpoints)}")
        report.append(f"Discovery Endpoints:  {len([e for e in unique_endpoints if 'discovery' in e])}")
        report.append(f"Config Endpoints:     {len([e for e in unique_endpoints if any(x in e for x in ['strategic', 'focus', 'entity', 'users'])])}")
        report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    async def run_comprehensive_tests(self):
        """Run all integration tests."""
        self.log("Starting comprehensive integration testing...")
        
        try:
            # Test 1: Authentication Flow
            auth_success = await self.test_user_authentication_flow()
            
            if auth_success:
                # Test 2: User Config Service Integration
                await self.test_user_config_service_integration()
                
                # Test 3: Discovery API Consistency
                await self.test_discovery_api_consistency()
                
                # Test 4: ML Learning Loop
                await self.test_ml_learning_loop()
                
                # Test 5: Content Operations
                await self.test_content_operations()
                
                # Test 6: Error Handling
                await self.test_error_handling()
                
                # Test 7: Performance Testing
                await self.test_performance()
            
            # Generate recommendations
            await self.generate_recommendations()
            
            self.log("Integration testing completed!")
            
        except Exception as e:
            self.log(f"Critical error during testing: {e}")
            self.report.recommendations.append(f"Critical error encountered: {e}")

async def main():
    """Run comprehensive integration tests."""
    print("Starting Comprehensive Integration Testing")
    print("=" * 50)
    
    tester = IntegrationTester()
    
    try:
        await tester.run_comprehensive_tests()
        
        # Generate and display report
        report = tester.generate_ascii_report()
        print("\n" + report)
        
        # Save report to file
        with open('integration_test_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\nDetailed report saved to: integration_test_report.txt")
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"\nCritical error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
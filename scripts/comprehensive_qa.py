"""
COMPREHENSIVE QA VALIDATION SYSTEM
for Competitive Intelligence v2 - User Config Service

This script performs complete validation of:
- Code functionality and imports
- Database operations and integrity
- Authentication security
- Performance characteristics
- Error handling robustness
- Security vulnerabilities
- Production readiness

Output: Clear PASS/FAIL results with detailed explanations
"""

import asyncio
import sys
import time
import traceback
import importlib
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

class ComprehensiveQA:
    """Complete QA validation system."""
    
    def __init__(self):
        self.results = {
            'imports': {'passed': 0, 'failed': 0, 'details': []},
            'database': {'passed': 0, 'failed': 0, 'details': []},
            'authentication': {'passed': 0, 'failed': 0, 'details': []},
            'security': {'passed': 0, 'failed': 0, 'details': []},
            'performance': {'passed': 0, 'failed': 0, 'details': []},
            'error_handling': {'passed': 0, 'failed': 0, 'details': []},
        }
        self.critical_failures = []
        self.warnings = []
        self.start_time = time.time()
    
    def log_result(self, category: str, test_name: str, passed: bool, message: str, is_critical: bool = False):
        """Log a test result."""
        status = "PASS" if passed else "FAIL"
        self.results[category]['details'].append(f"[{status}] {test_name}: {message}")
        
        if passed:
            self.results[category]['passed'] += 1
        else:
            self.results[category]['failed'] += 1
            if is_critical:
                self.critical_failures.append(f"{category.upper()}: {test_name} - {message}")
    
    def log_warning(self, message: str):
        """Log a warning."""
        self.warnings.append(message)
    
    async def test_imports(self):
        """Test all critical imports."""
        print("  Testing module imports...")
        
        critical_modules = [
            'app.config', 'app.database', 'app.auth',
            'app.models.user', 'app.models.strategic_profile', 
            'app.models.tracking', 'app.models.delivery',
            'app.schemas.auth', 'app.schemas.user',
            'app.middleware', 'app.routes.auth', 'app.main'
        ]
        
        for module in critical_modules:
            try:
                importlib.import_module(module)
                self.log_result('imports', f'Import {module}', True, 'Module imported successfully')
            except Exception as e:
                self.log_result('imports', f'Import {module}', False, f'Import failed: {e}', is_critical=True)
    
    async def test_database(self):
        """Test database functionality."""
        print("  Testing database operations...")
        
        try:
            from app.database import db_manager
            from app.models.user import User
            from sqlalchemy import text, select
            
            # Test connection
            await db_manager.initialize()
            self.log_result('database', 'Database Connection', True, 'Connected successfully')
            
            # Test health check
            is_healthy = await db_manager.health_check()
            self.log_result('database', 'Health Check', is_healthy, 
                          'Database is healthy' if is_healthy else 'Database health check failed', 
                          is_critical=not is_healthy)
            
            # Test basic query
            async with db_manager.get_session() as session:
                result = await session.execute(text("SELECT 1 + 1 as test"))
                test_value = result.scalar()
                query_success = test_value == 2
                self.log_result('database', 'Basic Query', query_success, 
                              'Query executed correctly' if query_success else f'Query failed: got {test_value}')
            
            # Test model operations
            async with db_manager.get_session() as session:
                test_user = User(
                    email=f"qatest_{int(time.time())}@example.com",
                    name="QA Test User",
                    password_hash="test_hash",
                    is_active=True
                )
                session.add(test_user)
                await session.commit()
                await session.refresh(test_user)
                
                # Verify user was created
                result = await session.execute(
                    select(User).where(User.id == test_user.id)
                )
                found_user = result.scalar_one_or_none()
                
                model_success = found_user is not None
                self.log_result('database', 'Model Operations', model_success,
                              'User model CRUD operations working' if model_success else 'Model operations failed')
                
                # Clean up
                if found_user:
                    await session.delete(found_user)
                    await session.commit()
            
            await db_manager.close()
            
        except Exception as e:
            self.log_result('database', 'Database Test', False, f'Database test failed: {e}', is_critical=True)
    
    async def test_authentication(self):
        """Test authentication system."""
        print("  Testing authentication...")
        
        try:
            from app.auth import auth_service
            from app.database import db_manager
            from app.models.user import User
            
            # Test password hashing
            password = "TestP@ssword123!"
            hashed = auth_service.hash_password(password)
            hash_valid = hashed.startswith('$2b$')
            self.log_result('authentication', 'Password Hashing', hash_valid,
                          'Password hashed with bcrypt' if hash_valid else 'Password hash format invalid')
            
            # Test password verification
            verify_correct = auth_service.verify_password(password, hashed)
            verify_wrong = not auth_service.verify_password("WrongPassword", hashed)
            
            self.log_result('authentication', 'Password Verification', verify_correct,
                          'Correct password verified' if verify_correct else 'Password verification failed')
            
            self.log_result('authentication', 'Password Rejection', verify_wrong,
                          'Wrong password rejected' if verify_wrong else 'Wrong password accepted - SECURITY ISSUE',
                          is_critical=not verify_wrong)
            
            # Test JWT tokens
            token_data = {"sub": "123", "email": "test@example.com"}
            token = auth_service.create_access_token(token_data)
            token_valid = len(token.split('.')) == 3
            
            self.log_result('authentication', 'JWT Creation', token_valid,
                          'JWT token created with proper format' if token_valid else 'JWT token format invalid')
            
            decoded = auth_service.decode_token(token)
            decode_success = decoded is not None and decoded.user_id == 123
            
            self.log_result('authentication', 'JWT Validation', decode_success,
                          'JWT token decoded correctly' if decode_success else 'JWT token validation failed')
            
            # Test invalid token rejection
            invalid_rejected = auth_service.decode_token("invalid.token.here") is None
            self.log_result('authentication', 'Invalid Token Rejection', invalid_rejected,
                          'Invalid tokens rejected' if invalid_rejected else 'Invalid token accepted - SECURITY ISSUE',
                          is_critical=not invalid_rejected)
            
            # Test full authentication flow
            await db_manager.initialize()
            async with db_manager.get_session() as session:
                test_user = User(
                    email=f"authtest_{int(time.time())}@example.com",
                    name="Auth Test",
                    password_hash=auth_service.hash_password(password),
                    is_active=True
                )
                session.add(test_user)
                await session.commit()
                await session.refresh(test_user)
                
                # Test authentication
                auth_user = await auth_service.authenticate_user(session, test_user.email, password)
                auth_success = auth_user is not None
                
                self.log_result('authentication', 'User Authentication', auth_success,
                              'User authentication working' if auth_success else 'User authentication failed')
                
                # Test wrong password rejection
                wrong_auth = await auth_service.authenticate_user(session, test_user.email, "WrongPassword")
                wrong_rejected = wrong_auth is None
                
                self.log_result('authentication', 'Wrong Password Auth', wrong_rejected,
                              'Wrong password auth rejected' if wrong_rejected else 'Wrong password accepted - SECURITY ISSUE',
                              is_critical=not wrong_rejected)
                
                # Clean up
                await session.delete(test_user)
                await session.commit()
            
            await db_manager.close()
            
        except Exception as e:
            self.log_result('authentication', 'Authentication Test', False, f'Auth test failed: {e}', is_critical=True)
    
    async def test_security(self):
        """Test security measures."""
        print("  Testing security measures...")
        
        try:
            from app.config import settings, SECURITY_HEADERS
            
            # Test security headers
            required_headers = ['X-Content-Type-Options', 'X-Frame-Options', 'X-XSS-Protection']
            missing_headers = [h for h in required_headers if h not in SECURITY_HEADERS]
            
            headers_ok = len(missing_headers) == 0
            self.log_result('security', 'Security Headers', headers_ok,
                          f'{len(SECURITY_HEADERS)} security headers configured' if headers_ok 
                          else f'Missing headers: {missing_headers}',
                          is_critical=not headers_ok)
            
            # Test secret key with comprehensive validation
            from app.auth import validate_jwt_security
            secret_secure, secret_message = validate_jwt_security()
            self.log_result('security', 'JWT Secret Key', secret_secure,
                          secret_message,
                          is_critical=not secret_secure)
            
            # Test bcrypt rounds
            bcrypt_secure = settings.BCRYPT_ROUNDS >= 10
            self.log_result('security', 'Password Hash Strength', bcrypt_secure,
                          f'bcrypt rounds: {settings.BCRYPT_ROUNDS}' if bcrypt_secure 
                          else f'bcrypt rounds too low: {settings.BCRYPT_ROUNDS}',
                          is_critical=not bcrypt_secure)
            
            # Test rate limiting
            rate_limit_on = settings.RATE_LIMIT_ENABLED
            self.log_result('security', 'Rate Limiting', rate_limit_on,
                          f'Rate limiting enabled: {settings.RATE_LIMIT_REQUESTS_PER_MINUTE}/min' if rate_limit_on
                          else 'Rate limiting disabled',
                          is_critical=not rate_limit_on)
            
            # Test input validation
            from app.schemas.auth import UserRegister
            from pydantic import ValidationError
            
            validation_working = True
            try:
                UserRegister(email="invalid", name="Test", password="weak")
                validation_working = False
            except ValidationError:
                pass  # Expected
            
            self.log_result('security', 'Input Validation', validation_working,
                          'Input validation rejecting invalid data' if validation_working
                          else 'Input validation not working - SECURITY RISK',
                          is_critical=not validation_working)
            
        except Exception as e:
            self.log_result('security', 'Security Test', False, f'Security test failed: {e}', is_critical=True)
    
    async def test_performance(self):
        """Test performance characteristics."""
        print("  Testing performance...")
        
        try:
            from app.database import db_manager
            from app.auth import auth_service
            
            # Test database connection time
            start = time.time()
            await db_manager.initialize()
            await db_manager.health_check()
            db_time = time.time() - start
            
            db_fast = db_time < 10.0  # 10 seconds max
            self.log_result('performance', 'Database Connection Speed', db_fast,
                          f'Database connection: {db_time:.2f}s' if db_fast 
                          else f'Database connection too slow: {db_time:.2f}s')
            
            # Test password hashing time
            start = time.time()
            auth_service.hash_password("TestPassword123!")
            hash_time = time.time() - start
            
            hash_reasonable = 0.1 <= hash_time <= 2.0  # Should be slow but not too slow
            self.log_result('performance', 'Password Hash Timing', hash_reasonable,
                          f'Password hashing: {hash_time:.3f}s' if hash_reasonable
                          else f'Password hashing time unusual: {hash_time:.3f}s')
            
            if hash_time < 0.1:
                self.log_warning("Password hashing very fast - may be insecure")
            
            # Test JWT creation time
            start = time.time()
            token_data = {"sub": "123", "email": "test@example.com"}
            auth_service.create_access_token(token_data)
            jwt_time = time.time() - start
            
            jwt_fast = jwt_time < 0.1  # 100ms max
            self.log_result('performance', 'JWT Creation Speed', jwt_fast,
                          f'JWT creation: {jwt_time:.3f}s' if jwt_fast
                          else f'JWT creation too slow: {jwt_time:.3f}s')
            
            await db_manager.close()
            
        except Exception as e:
            self.log_result('performance', 'Performance Test', False, f'Performance test failed: {e}')
    
    async def test_error_handling(self):
        """Test error handling robustness."""
        print("  Testing error handling...")
        
        try:
            from app.auth import auth_service
            from app.database import db_manager
            from app.models.user import User
            
            # Test invalid JWT handling
            invalid_result = auth_service.decode_token("clearly.invalid.token")
            jwt_error_handled = invalid_result is None
            
            self.log_result('error_handling', 'Invalid JWT Handling', jwt_error_handled,
                          'Invalid JWT tokens properly rejected' if jwt_error_handled
                          else 'Invalid JWT tokens not handled properly',
                          is_critical=not jwt_error_handled)
            
            # Test invalid password hash handling
            password_error_handled = True
            try:
                result = auth_service.verify_password("test", "not_a_real_hash")
                if result:  # Should be False, not an exception
                    password_error_handled = False
            except Exception:
                pass  # Exception is also acceptable
            
            self.log_result('error_handling', 'Invalid Hash Handling', password_error_handled,
                          'Invalid password hashes handled gracefully' if password_error_handled
                          else 'Invalid password hashes not handled properly')
            
            # Test database constraint handling
            await db_manager.initialize()
            try:
                async with db_manager.get_session() as session:
                    # Try to create duplicate users
                    email = f"duplicate_{int(time.time())}@example.com"
                    
                    user1 = User(email=email, name="User 1", password_hash="hash1")
                    session.add(user1)
                    await session.commit()
                    
                    try:
                        user2 = User(email=email, name="User 2", password_hash="hash2")
                        session.add(user2)
                        await session.commit()
                        
                        # If we get here, constraint wasn't enforced
                        self.log_result('error_handling', 'Database Constraints', False,
                                      'Duplicate email constraint not enforced', is_critical=True)
                    except Exception:
                        # Expected - constraint violation
                        self.log_result('error_handling', 'Database Constraints', True,
                                      'Database constraints properly enforced')
                        await session.rollback()
            
            except Exception as e:
                self.log_result('error_handling', 'Database Error Test', False, f'Database error test failed: {e}')
            
            await db_manager.close()
            
        except Exception as e:
            self.log_result('error_handling', 'Error Handling Test', False, f'Error handling test failed: {e}')
    
    def print_comprehensive_report(self):
        """Print detailed QA report."""
        total_time = time.time() - self.start_time
        
        print("\\n" + "="*80)
        print("COMPETITIVE INTELLIGENCE V2 - COMPREHENSIVE QA REPORT")
        print("="*80)
        print(f"Validation completed in {total_time:.2f} seconds")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate totals
        total_tests = sum(cat['passed'] + cat['failed'] for cat in self.results.values())
        total_passed = sum(cat['passed'] for cat in self.results.values())
        total_failed = sum(cat['failed'] for cat in self.results.values())
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\\nOVERALL SUMMARY:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_passed}")
        print(f"  Failed: {total_failed}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Critical Failures: {len(self.critical_failures)}")
        print(f"  Warnings: {len(self.warnings)}")
        
        # Determine overall status
        if len(self.critical_failures) == 0 and total_failed == 0:
            status = "[HEALTHY] SYSTEM READY FOR PRODUCTION"
        elif len(self.critical_failures) == 0:
            status = "[CAUTION] SYSTEM FUNCTIONAL WITH MINOR ISSUES"
        else:
            status = "[CRITICAL] SYSTEM HAS SERIOUS ISSUES"
        
        print(f"\\nSTATUS: {status}")
        
        # Print category details
        for category, data in self.results.items():
            total_cat = data['passed'] + data['failed']
            if total_cat > 0:
                print(f"\\n{category.upper().replace('_', ' ')} ({data['passed']}/{total_cat} passed)")
                print("-" * 40)
                for detail in data['details']:
                    print(f"  {detail}")
        
        # Print critical failures
        if self.critical_failures:
            print(f"\\n[CRITICAL FAILURES] - MUST BE ADDRESSED:")
            for failure in self.critical_failures:
                print(f"  ! {failure}")
        
        # Print warnings
        if self.warnings:
            print(f"\\n[WARNINGS] - SHOULD BE REVIEWED:")
            for warning in self.warnings:
                print(f"  * {warning}")
        
        # Print recommendations
        print(f"\\nRECOMMENDATIONS:")
        if len(self.critical_failures) == 0 and total_failed == 0:
            print("  [APPROVED] System passes all quality checks")
            print("  [DEPLOY] Ready for production deployment")
            print("  [MONITOR] Continue monitoring in production")
        elif len(self.critical_failures) == 0:
            print("  [REVIEW] Address minor issues if possible")
            print("  [DEPLOY] System is functional for deployment")
        else:
            print("  [BLOCK] Do not deploy until critical issues are resolved")
            print("  [FIX] Address all critical failures before proceeding")
            print("  [RETEST] Run QA validation again after fixes")
        
        print("\\n" + "="*80)
        
        return len(self.critical_failures) == 0 and total_failed == 0
    
    async def run_comprehensive_validation(self):
        """Run all validation tests."""
        print("COMPREHENSIVE QA VALIDATION STARTING")
        print("="*50)
        print("This will validate all system components...")
        print()
        
        test_categories = [
            ("IMPORTS & CONFIGURATION", self.test_imports),
            ("DATABASE OPERATIONS", self.test_database),
            ("AUTHENTICATION SYSTEM", self.test_authentication),
            ("SECURITY MEASURES", self.test_security),
            ("PERFORMANCE CHARACTERISTICS", self.test_performance),
            ("ERROR HANDLING", self.test_error_handling),
        ]
        
        for category_name, test_func in test_categories:
            print(f"[{category_name}]")
            try:
                await test_func()
            except Exception as e:
                self.critical_failures.append(f"{category_name}: Unexpected test failure - {e}")
                print(f"  [ERROR] Test category failed: {e}")
            print()
        
        return self.print_comprehensive_report()


async def main():
    """Run comprehensive QA validation."""
    qa = ComprehensiveQA()
    
    try:
        all_passed = await qa.run_comprehensive_validation()
        return 0 if all_passed else 1
    except KeyboardInterrupt:
        print("\\n\\n[INTERRUPTED] QA validation stopped by user")
        return 1
    except Exception as e:
        print(f"\\n\\n[SYSTEM ERROR] QA validation system failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
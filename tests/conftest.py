"""
Test configuration and fixtures for the Competitive Intelligence v2 API.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, Any
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.main import app
from app.database import Base, get_db_session
from app.models.user import User, UserSession
from app.models.strategic_profile import UserStrategicProfile
from app.models.delivery import UserDeliveryPreferences
from app.auth import AuthService

# Test database URL - using in-memory SQLite for fast tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Test engine and session
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

TestAsyncSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_db():
    """Set up test database."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    async with TestAsyncSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def override_get_db_session(db_session: AsyncSession):
    """Override the get_db_session dependency."""
    async def _override_get_db_session():
        yield db_session
    
    app.dependency_overrides[get_db_session] = _override_get_db_session
    yield
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(override_get_db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user (Chris Eaton profile)."""
    
    user = User(
        email="chris.eaton@example.com",
        name="Chris Eaton",
        password_hash=AuthService.hash_password("TestPassword123!"),
        is_active=True,
        subscription_status="active"
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user


@pytest_asyncio.fixture
async def test_user_with_profile(db_session: AsyncSession, test_user: User) -> User:
    """Create a test user with strategic profile."""
    profile = UserStrategicProfile(
        user_id=test_user.id,
        industry="technology",
        organization_type="startup",
        role="ceo",
        strategic_goals=["market_expansion", "product_development", "competitive_positioning"],
        organization_size="medium"
    )
    
    db_session.add(profile)
    await db_session.commit()
    await db_session.refresh(profile)
    
    # Refresh user to load profile
    await db_session.refresh(test_user)
    
    return test_user


@pytest_asyncio.fixture
async def auth_token(test_user: User) -> str:
    """Create an authentication token for the test user."""
    token_data = {
        "user_id": test_user.id,
        "email": test_user.email,
        "session_id": 1,
        "scopes": ["user:read", "user:write"]
    }
    
    return AuthService.create_access_token(data=token_data)


@pytest_asyncio.fixture
async def auth_headers(auth_token: str) -> Dict[str, str]:
    """Create authentication headers."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest_asyncio.fixture
async def test_user_session(db_session: AsyncSession, test_user: User, auth_token: str) -> UserSession:
    """Create a test user session."""
    session = UserSession(
        user_id=test_user.id,
        token=auth_token,
        expires_at=None,  # No expiration for tests
        is_active=True
    )
    
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    
    return session


@pytest_asyncio.fixture
async def second_test_user(db_session: AsyncSession) -> User:
    """Create a second test user for multi-user tests."""
    
    user = User(
        email="jane.doe@example.com",
        name="Jane Doe",
        password_hash=AuthService.hash_password("AnotherPassword123!"),
        is_active=True,
        subscription_status="trial"
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user


@pytest_asyncio.fixture
async def test_delivery_preferences(db_session: AsyncSession, test_user: User) -> UserDeliveryPreferences:
    """Create test delivery preferences."""
    from datetime import time
    
    preferences = UserDeliveryPreferences(
        user_id=test_user.id,
        frequency="daily",
        delivery_time=time(8, 0),
        timezone="America/New_York",
        weekend_delivery=False,
        max_articles_per_report=15,
        min_significance_level="medium",
        content_format="executive_summary",
        email_enabled=True,
        urgent_alerts_enabled=True,
        digest_mode=True
    )
    
    db_session.add(preferences)
    await db_session.commit()
    await db_session.refresh(preferences)
    
    return preferences


# Common test data
@pytest.fixture
def sample_focus_area_data() -> Dict[str, Any]:
    """Sample focus area data for testing."""
    return {
        "focus_area": "AI and Machine Learning",
        "keywords": ["artificial intelligence", "machine learning", "deep learning"],
        "priority": 3
    }


@pytest.fixture
def sample_entity_data() -> Dict[str, Any]:
    """Sample entity tracking data for testing."""
    return {
        "name": "OpenAI",
        "entity_type": "competitor",
        "domain": "openai.com",
        "description": "AI research company",
        "industry": "Artificial Intelligence"
    }


@pytest.fixture
def sample_delivery_preferences_data() -> Dict[str, Any]:
    """Sample delivery preferences data for testing."""
    return {
        "frequency": "daily",
        "delivery_time": "09:00",
        "timezone": "America/New_York",
        "weekend_delivery": False,
        "max_articles_per_report": 20,
        "min_significance_level": "high",
        "content_format": "executive_summary",
        "email_enabled": True,
        "urgent_alerts_enabled": True,
        "digest_mode": True
    }


@pytest.fixture
def sample_strategic_profile_data() -> Dict[str, Any]:
    """Sample strategic profile data for testing."""
    return {
        "industry": "healthcare",
        "organization_type": "enterprise",
        "role": "product_manager",
        "strategic_goals": ["ai_integration", "regulatory_compliance"],
        "organization_size": "large"
    }


# Test utilities
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_user_data(email: str = None, name: str = None) -> Dict[str, Any]:
        """Create user registration data."""
        return {
            "name": name or "Test User",
            "email": email or "test@example.com",
            "password": "TestPassword123!"
        }
    
    @staticmethod
    def create_login_data(email: str = None, password: str = None) -> Dict[str, Any]:
        """Create login data."""
        return {
            "email": email or "chris.eaton@example.com",
            "password": password or "TestPassword123!"
        }


@pytest.fixture
def test_factory() -> TestDataFactory:
    """Provide test data factory."""
    return TestDataFactory()


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "auth: marks tests that require authentication"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
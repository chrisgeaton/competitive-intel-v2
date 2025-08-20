"""Database configuration and connection management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import text
from asyncpg.exceptions import PostgresError

logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql+asyncpg://admin:yourpassword@localhost:5432/competitive_intelligence"

Base = declarative_base()

class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self._engine = None
        self._sessionmaker = None
    
    async def initialize(self):
        """Initialize the database engine and session factory."""
        try:
            self._engine = create_async_engine(
                self.database_url,
                poolclass=NullPool,
                pool_pre_ping=True,
                connect_args={
                    "server_settings": {
                        "application_name": "competitive_intel_v2",
                        "jit": "off"
                    },
                    "command_timeout": 60,
                }
            )
            
            self._sessionmaker = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False
            )
            
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close all database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session with automatic transaction management."""
        if not self._sessionmaker:
            await self.initialize()
        
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except PostgresError as e:
                await session.rollback()
                logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                await session.rollback()
                logger.error(f"Unexpected error in database session: {e}")
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

db_manager = DatabaseManager()

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency function for FastAPI to get database sessions."""
    async with db_manager.get_session() as session:
        yield session

async def init_db():
    """Initialize database tables if they don't exist."""
    try:
        await db_manager.initialize()
        
        async with db_manager._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise

async def close_db():
    """Close database connections on application shutdown."""
    await db_manager.close()
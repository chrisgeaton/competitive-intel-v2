"""
Database migration script to create Discovery Service tables.

Creates all tables required for ML-driven competitive intelligence
discovery with content scoring and engagement tracking.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from app.config import settings
from app.models.discovery import DiscoveredSource, DiscoveredContent, ContentEngagement, DiscoveryJob, MLModelMetrics
from app.database import Base


async def create_discovery_tables():
    """Create all Discovery Service tables in the database."""
    
    print("Creating Discovery Service database tables...")
    
    # Create async engine
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    
    try:
        # Create all discovery tables
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            from app.models import user, strategic_profile, tracking, delivery, discovery
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
            print("Discovery Service tables created successfully!")
            
            # Verify tables were created
            result = await conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE '%discover%'")
            )
            tables = result.fetchall()
            
            print("\nDiscovery tables created:")
            for table in tables:
                print(f"  - {table[0]}")
            
            # Create some initial ML model metrics record
            await conn.execute(
                text("""
                INSERT INTO ml_model_metrics (
                    model_version, model_type, model_name, training_data_size, 
                    training_duration_seconds, training_accuracy, validation_accuracy,
                    is_active, created_by, created_at, updated_at
                ) VALUES (
                    '2.0', 'relevance_scorer', 'Discovery Relevance Model v2.0', 1000,
                    120, 0.8500, 0.8200, true, 'system', NOW(), NOW()
                )
                ON CONFLICT DO NOTHING
                """)
            )
            
            print("Initial ML model metrics created!")
            
    except Exception as e:
        print(f"Error creating discovery tables: {e}")
        raise
    finally:
        await engine.dispose()


async def verify_discovery_setup():
    """Verify the Discovery Service database setup."""
    
    print("\nVerifying Discovery Service setup...")
    
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    
    try:
        async with engine.begin() as conn:
            # Check all discovery tables exist
            required_tables = [
                'discovered_sources',
                'discovered_content', 
                'content_engagement',
                'discovery_jobs',
                'ml_model_metrics'
            ]
            
            for table in required_tables:
                result = await conn.execute(
                    text(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table}'")
                )
                count = result.scalar()
                
                if count > 0:
                    print(f"  OK: {table}")
                else:
                    print(f"  MISSING: {table}")
                    return False
            
            # Check indexes exist
            result = await conn.execute(
                text("""
                SELECT indexname FROM pg_indexes 
                WHERE schemaname = 'public' 
                AND (tablename LIKE '%discover%' OR tablename LIKE '%content%' OR tablename LIKE '%ml%')
                ORDER BY indexname
                """)
            )
            indexes = result.fetchall()
            
            print(f"\nDiscovery indexes created: {len(indexes)}")
            for index in indexes[:10]:  # Show first 10 indexes
                print(f"  - {index[0]}")
            
            if len(indexes) > 10:
                print(f"  ... and {len(indexes) - 10} more")
            
            # Check ML model record
            result = await conn.execute(
                text("SELECT COUNT(*) FROM ml_model_metrics WHERE model_version = '2.0'")
            )
            ml_count = result.scalar()
            
            if ml_count > 0:
                print("  OK: ML model metrics")
            else:
                print("  MISSING: ML model metrics")
            
            print("\nDiscovery Service database setup verification complete!")
            return True
            
    except Exception as e:
        print(f"Error verifying discovery setup: {e}")
        return False
    finally:
        await engine.dispose()


async def main():
    """Main function to create and verify Discovery Service tables."""
    
    print("Discovery Service Database Setup")
    print("=" * 50)
    
    try:
        # Create tables
        await create_discovery_tables()
        
        # Verify setup
        success = await verify_discovery_setup()
        
        if success:
            print("\nDiscovery Service database setup completed successfully!")
            print("\nNext steps:")
            print("  1. Start the application: python app/main.py")
            print("  2. Visit http://localhost:8000/docs to see Discovery API endpoints")
            print("  3. Test discovery endpoints with authentication")
            print("  4. Configure discovery sources and run discovery jobs")
        else:
            print("\nDiscovery Service setup incomplete. Please check errors above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nFatal error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
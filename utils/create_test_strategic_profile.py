#!/usr/bin/env python3
"""
Create test user with strategic profile for end-to-end testing.
"""

import asyncio
import sys
from datetime import datetime, timezone
from sqlalchemy import select, delete
from app.database import db_manager
from app.models.user import User
from app.models.strategic_profile import UserStrategicProfile, UserFocusArea
from app.models.tracking import TrackingEntity, UserEntityTracking
from app.models.delivery import UserDeliveryPreferences
from app.auth import auth_service

async def create_test_user():
    """Create test user with complete strategic profile."""
    
    async with db_manager.get_session() as db:
        try:
            # Clean up existing test user if present
            existing_user = await db.execute(
                select(User).where(User.email == "ceaton@livedata.com")
            )
            existing = existing_user.scalar_one_or_none()
            
            if existing:
                print(f"Found existing user {existing.email}, cleaning up...")
                
                # Delete related data
                await db.execute(delete(UserFocusArea).where(UserFocusArea.user_id == existing.id))
                await db.execute(delete(UserEntityTracking).where(UserEntityTracking.user_id == existing.id))
                await db.execute(delete(UserDeliveryPreferences).where(UserDeliveryPreferences.user_id == existing.id))
                await db.execute(delete(UserStrategicProfile).where(UserStrategicProfile.user_id == existing.id))
                await db.delete(existing)
                await db.commit()
                print("Cleanup complete")
            
            # Create new test user
            print("\nCreating test user...")
            user = User(
                name="Chris Eaton (Test User)",
                email="ceaton@livedata.com",
                password_hash=auth_service.hash_password("TestPassword123!"),
                is_active=True,
                created_at=datetime.now(timezone.utc)
            )
            db.add(user)
            await db.flush()
            
            # Create strategic profile
            print("Creating strategic profile...")
            profile = UserStrategicProfile(
                user_id=user.id,
                industry="Technology",
                organization_type="Corporation",
                role="CEO",
                strategic_goals=["Monitor AI competitive landscape", "Track AI regulation", "Identify market trends"],
                organization_size="Large",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            db.add(profile)
            
            # Create focus areas
            print("Creating focus areas...")
            focus_areas_data = [
                ("artificial intelligence", 5, "Core technology focus"),
                ("machine learning", 4, "Technical implementation"),
                ("AI regulation", 5, "Compliance and policy tracking"),
                ("competitive intelligence", 3, "Market analysis")
            ]
            
            for keyword, priority, description in focus_areas_data:
                # Create user focus area
                user_focus = UserFocusArea(
                    user_id=user.id,
                    focus_area=keyword,
                    keywords=[keyword, description],
                    priority=priority,
                    created_at=datetime.now(timezone.utc)
                )
                db.add(user_focus)
            
            # Create tracked entities
            print("Creating tracked entities...")
            entities_data = [
                ("OpenAI", "organization", 5, {"industry": "AI", "type": "competitor"}),
                ("Anthropic", "organization", 4, {"industry": "AI", "type": "competitor"}),
                ("Google AI", "organization", 4, {"industry": "AI", "type": "competitor"}),
                ("Microsoft AI", "organization", 3, {"industry": "AI", "type": "competitor"}),
                ("Meta AI", "organization", 3, {"industry": "AI", "type": "competitor"})
            ]
            
            for name, entity_type, priority, metadata in entities_data:
                # First create or get the tracking entity
                existing_entity = await db.execute(
                    select(TrackingEntity).where(
                        TrackingEntity.name == name,
                        TrackingEntity.entity_type == entity_type
                    )
                )
                entity = existing_entity.scalar_one_or_none()
                
                if not entity:
                    entity = TrackingEntity(
                        name=name,
                        entity_type=entity_type,
                        description=f"{name} - AI company",
                        industry="Technology",
                        metadata_json=metadata,
                        created_at=datetime.now(timezone.utc)
                    )
                    db.add(entity)
                    await db.flush()
                
                # Create user tracking relationship
                user_tracking = UserEntityTracking(
                    user_id=user.id,
                    entity_id=entity.id,
                    priority=priority,
                    custom_keywords=[name.lower()],
                    tracking_enabled=True,
                    created_at=datetime.now(timezone.utc)
                )
                db.add(user_tracking)
            
            # Create delivery preferences
            print("Creating delivery preferences...")
            from datetime import time
            delivery_pref = UserDeliveryPreferences(
                user_id=user.id,
                frequency="daily",
                delivery_time=time(9, 0),  # 9:00 AM
                timezone="America/New_York",
                weekend_delivery=False,
                max_articles_per_report=10,
                min_significance_level="medium",
                content_format="executive_summary",
                email_enabled=True,
                urgent_alerts_enabled=True,
                digest_mode=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            db.add(delivery_pref)
            
            await db.commit()
            
            # Generate access token for testing
            token_data = {"user_id": user.id, "email": user.email}
            access_token = auth_service.create_access_token(data=token_data)
            
            print("\nTest user created successfully!")
            print(f"Email: {user.email}")
            print(f"User ID: {user.id}")
            print(f"Strategic Profile ID: {profile.id}")
            print(f"Focus Areas: {len(focus_areas_data)}")
            print(f"Tracked Entities: {len(entities_data)}")
            print(f"\nAccess Token (for testing):")
            print(f"{access_token}\n")
            
            return user.id, access_token
            
        except Exception as e:
            await db.rollback()
            print(f"Error creating test user: {e}")
            raise

if __name__ == "__main__":
    user_id, token = asyncio.run(create_test_user())
    print(f"Test user ready for Phase 5 testing!")
    print(f"User ID: {user_id}")
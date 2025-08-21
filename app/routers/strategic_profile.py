"""
Strategic profile management routes for the Competitive Intelligence v2 API.
"""

from typing import Optional

from app.models.strategic_profile import UserStrategicProfile
from app.schemas.strategic_profile import (
    StrategicProfileCreate,
    StrategicProfileUpdate,
    StrategicProfileResponse,
    StrategicProfileAnalytics,
    IndustryType,
    OrganizationType,
    StrategicGoalCategory,
    UserRole,
    OrganizationSize
)
from app.utils.router_base import (
    logging, Dict, List, Any, APIRouter, Depends, Query, status,
    AsyncSession, select, func, get_db_session, User, get_current_active_user,
    errors, db_handler, db_helpers, BaseRouterOperations, create_analytics_response
)

base_ops = BaseRouterOperations(__name__)

router = APIRouter(prefix="/api/v1/strategic-profile", tags=["Strategic Profile Management"])


@router.get("/", response_model=StrategicProfileResponse)
async def get_strategic_profile(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get user's strategic profile.
    
    Returns the user's complete business context including industry, role,
    organization details, and strategic objectives.
    """
    async def _get_profile_operation():
        profile = await db_helpers.get_model_by_field(
            db, UserStrategicProfile, "user_id", current_user.id,
            validate_exists=True, resource_name="Strategic profile"
        )
        
        return StrategicProfileResponse.model_validate(profile)
    
    return await db_handler.handle_db_operation(
        "get strategic profile", _get_profile_operation, db, rollback_on_error=False
    )


@router.put("/", response_model=StrategicProfileResponse)
async def update_strategic_profile(
    profile_data: StrategicProfileUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update user's strategic profile.
    
    Updates the user's business context with enhanced validation for:
    - **industry**: Industry sector from predefined list
    - **organization_type**: Type of organization
    - **role**: User's role with validation
    - **strategic_goals**: List of strategic objectives (max 10)
    - **organization_size**: Organization size category
    
    Only provided fields will be updated. All fields are optional.
    """
    async def _update_profile_operation():
        # Get existing profile
        profile = await db_helpers.get_model_by_field(
            db, UserStrategicProfile, "user_id", current_user.id,
            validate_exists=True, resource_name="Strategic profile"
        )
        
        # Update fields if provided
        update_data = profile_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if value is not None:
                setattr(profile, field, value)
        
        # Update timestamp
        from datetime import datetime
        profile.updated_at = datetime.utcnow()
        
        await db_helpers.safe_commit(db, "strategic profile update")
        await db.refresh(profile)
        
        logger.info(f"Strategic profile updated for user: {current_user.email}")
        return StrategicProfileResponse.model_validate(profile)
    
    return await db_handler.handle_db_operation(
        "update strategic profile", _update_profile_operation, db
    )


@router.post("/", response_model=StrategicProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_strategic_profile(
    profile_data: StrategicProfileCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create user's strategic profile.
    
    Creates a new strategic profile with enhanced validation for:
    - **industry**: Industry sector (healthcare, technology, finance, etc.)
    - **organization_type**: Organization type (startup, enterprise, nonprofit, etc.)
    - **role**: User's role (ceo, manager, analyst, etc.)
    - **strategic_goals**: List of strategic objectives (max 10 goals)
    - **organization_size**: Size category (micro, small, medium, large, enterprise)
    
    All fields are optional but recommended for better intelligence personalization.
    """
    async def _create_profile_operation():
        # Check if profile already exists
        existing_profile = await db_helpers.get_model_by_field(
            db, UserStrategicProfile, "user_id", current_user.id
        )
        
        if existing_profile:
            raise errors.conflict("Strategic profile already exists. Use PUT to update.")
        
        # Create new strategic profile
        profile_dict = profile_data.model_dump(exclude_unset=True)
        new_profile = UserStrategicProfile(
            user_id=current_user.id,
            **profile_dict
        )
        
        db.add(new_profile)
        await db_helpers.safe_commit(db, "strategic profile creation")
        await db.refresh(new_profile)
        
        logger.info(f"Strategic profile created for user: {current_user.email}")
        return StrategicProfileResponse.model_validate(new_profile)
    
    return await db_handler.handle_db_operation(
        "create strategic profile", _create_profile_operation, db
    )


@router.delete("/")
async def delete_strategic_profile(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete user's strategic profile.
    
    **Warning**: This permanently removes the user's strategic profile
    and all associated business context. This action cannot be undone.
    """
    async def _delete_profile_operation():
        profile = await db_helpers.get_model_by_field(
            db, UserStrategicProfile, "user_id", current_user.id,
            validate_exists=True, resource_name="Strategic profile"
        )
        
        await db_helpers.safe_delete(db, profile, "delete strategic profile")
        
        logger.info(f"Strategic profile deleted for user: {current_user.email}")
        return {"message": "Strategic profile deleted successfully"}
    
    return await db_handler.handle_db_operation(
        "delete strategic profile", _delete_profile_operation, db
    )


@router.get("/analytics", response_model=StrategicProfileAnalytics)
async def get_profile_analytics(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get strategic profile analytics and insights.
    
    Returns profile completeness analysis, missing fields, and
    industry-specific recommendations for strategic goals.
    """
    async def _get_analytics_operation():
        profile = await db_helpers.get_model_by_field(
            db, UserStrategicProfile, "user_id", current_user.id,
            validate_exists=True, resource_name="Strategic profile"
        )
        
        # Calculate profile completeness
        total_fields = 5  # industry, organization_type, role, strategic_goals, organization_size
        completed_fields = 0
        missing_fields = []
        
        if profile.industry:
            completed_fields += 1
        else:
            missing_fields.append("industry")
            
        if profile.organization_type:
            completed_fields += 1
        else:
            missing_fields.append("organization_type")
            
        if profile.role:
            completed_fields += 1
        else:
            missing_fields.append("role")
            
        if profile.strategic_goals and len(profile.strategic_goals) > 0:
            completed_fields += 1
        else:
            missing_fields.append("strategic_goals")
            
        if profile.organization_size:
            completed_fields += 1
        else:
            missing_fields.append("organization_size")
        
        completeness = (completed_fields / total_fields) * 100
        
        # Generate recommendations based on industry and role
        recommended_goals = _get_recommended_goals(profile.industry, profile.role)
        
        # Get industry trends (simplified example)
        industry_trends = _get_industry_trends(profile.industry) if profile.industry else None
        
        return StrategicProfileAnalytics(
            profile_completeness=completeness,
            missing_fields=missing_fields,
            recommended_goals=recommended_goals,
            industry_trends=industry_trends
        )
    
    return await db_handler.handle_db_operation(
        "get profile analytics", _get_analytics_operation, db, rollback_on_error=False
    )


@router.get("/enums/industries")
async def get_industry_types():
    """Get list of supported industry types."""
    return {
        "industries": [
            {"value": industry.value, "label": industry.value.replace("_", " ").title()}
            for industry in IndustryType
        ]
    }


@router.get("/enums/organization-types")
async def get_organization_types():
    """Get list of supported organization types."""
    return {
        "organization_types": [
            {"value": org_type.value, "label": org_type.value.replace("_", " ").title()}
            for org_type in OrganizationType
        ]
    }


@router.get("/enums/roles")
async def get_user_roles():
    """Get list of supported user roles."""
    return {
        "roles": [
            {"value": role.value, "label": role.value.replace("_", " ").title()}
            for role in UserRole
        ]
    }


@router.get("/enums/strategic-goals")
async def get_strategic_goal_categories():
    """Get list of supported strategic goal categories."""
    return {
        "strategic_goals": [
            {"value": goal.value, "label": goal.value.replace("_", " ").title()}
            for goal in StrategicGoalCategory
        ]
    }


@router.get("/enums/organization-sizes")
async def get_organization_sizes():
    """Get list of supported organization sizes."""
    return {
        "organization_sizes": [
            {"value": size.value, "label": size.value.replace("_", " ").title()}
            for size in OrganizationSize
        ]
    }


@router.get("/stats")
async def get_profile_statistics(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get strategic profile statistics and trends.
    
    Returns aggregate statistics about strategic profiles in the system
    for benchmarking and insights.
    """
    async def _get_stats_operation():
        # Get total profiles count
        total_profiles = await db.scalar(
            select(func.count(UserStrategicProfile.id))
        )
        
        # Get industry distribution
        industry_stats = await db.execute(
            select(
                UserStrategicProfile.industry,
                func.count(UserStrategicProfile.id).label('count')
            )
            .where(UserStrategicProfile.industry.isnot(None))
            .group_by(UserStrategicProfile.industry)
            .order_by(func.count(UserStrategicProfile.id).desc())
        )
        
        # Get role distribution
        role_stats = await db.execute(
            select(
                UserStrategicProfile.role,
                func.count(UserStrategicProfile.id).label('count')
            )
            .where(UserStrategicProfile.role.isnot(None))
            .group_by(UserStrategicProfile.role)
            .order_by(func.count(UserStrategicProfile.id).desc())
        )
        
        return {
            "total_profiles": total_profiles or 0,
            "industry_distribution": [
                {"industry": row.industry, "count": row.count}
                for row in industry_stats.fetchall()
            ],
            "role_distribution": [
                {"role": row.role, "count": row.count}
                for row in role_stats.fetchall()
            ]
        }
    
    return await db_handler.handle_db_operation(
        "get profile statistics", _get_stats_operation, db, rollback_on_error=False
    )


def _get_recommended_goals(industry: Optional[str], role: Optional[str]) -> List[StrategicGoalCategory]:
    """Generate recommended strategic goals based on industry and role."""
    recommendations = []
    
    # Industry-based recommendations
    if industry:
        industry_recommendations = {
            IndustryType.TECHNOLOGY: [
                StrategicGoalCategory.INNOVATION,
                StrategicGoalCategory.DIGITAL_TRANSFORMATION,
                StrategicGoalCategory.PRODUCT_DEVELOPMENT
            ],
            IndustryType.HEALTHCARE: [
                StrategicGoalCategory.COMPLIANCE,
                StrategicGoalCategory.DIGITAL_TRANSFORMATION,
                StrategicGoalCategory.OPERATIONAL_EFFICIENCY
            ],
            IndustryType.FINANCE: [
                StrategicGoalCategory.COMPLIANCE,
                StrategicGoalCategory.RISK_MANAGEMENT,
                StrategicGoalCategory.DIGITAL_TRANSFORMATION
            ],
            IndustryType.EDUCATION: [
                StrategicGoalCategory.DIGITAL_TRANSFORMATION,
                StrategicGoalCategory.OPERATIONAL_EFFICIENCY,
                StrategicGoalCategory.TECHNOLOGY_ADOPTION
            ],
            IndustryType.NONPROFIT: [
                StrategicGoalCategory.OPERATIONAL_EFFICIENCY,
                StrategicGoalCategory.SUSTAINABILITY,
                StrategicGoalCategory.BRAND_AWARENESS
            ]
        }
        
        if industry in industry_recommendations:
            recommendations.extend(industry_recommendations[industry])
    
    # Role-based recommendations
    if role:
        role_recommendations = {
            UserRole.CEO: [
                StrategicGoalCategory.MARKET_EXPANSION,
                StrategicGoalCategory.REVENUE_GROWTH,
                StrategicGoalCategory.COMPETITIVE_INTELLIGENCE
            ],
            UserRole.CTO: [
                StrategicGoalCategory.DIGITAL_TRANSFORMATION,
                StrategicGoalCategory.TECHNOLOGY_ADOPTION,
                StrategicGoalCategory.INNOVATION
            ],
            UserRole.CMO: [
                StrategicGoalCategory.BRAND_AWARENESS,
                StrategicGoalCategory.CUSTOMER_ACQUISITION,
                StrategicGoalCategory.MARKET_EXPANSION
            ]
        }
        
        if role in role_recommendations:
            recommendations.extend(role_recommendations[role])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for goal in recommendations:
        if goal not in seen:
            seen.add(goal)
            unique_recommendations.append(goal)
    
    return unique_recommendations[:5]  # Return top 5 recommendations


def _get_industry_trends(industry: str) -> Optional[dict]:
    """Get industry-specific trends and insights."""
    # Simplified example - in production this would come from data analysis
    trends = {
        IndustryType.TECHNOLOGY: {
            "top_goals": ["innovation", "digital_transformation", "product_development"],
            "emerging_trends": ["ai_adoption", "cloud_migration", "cybersecurity"]
        },
        IndustryType.HEALTHCARE: {
            "top_goals": ["compliance", "digital_transformation", "operational_efficiency"],
            "emerging_trends": ["telemedicine", "ai_diagnostics", "patient_experience"]
        },
        IndustryType.FINANCE: {
            "top_goals": ["compliance", "risk_management", "digital_transformation"],
            "emerging_trends": ["fintech_integration", "blockchain", "regulatory_tech"]
        }
    }
    
    return trends.get(industry)
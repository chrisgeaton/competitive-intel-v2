"""
Focus areas management routes for the Competitive Intelligence v2 API.
"""

from typing import Optional
from fastapi import HTTPException

from app.models.strategic_profile import UserFocusArea
from app.schemas.focus_areas import (
    FocusAreaCreate,
    FocusAreaUpdate,
    FocusAreaResponse,
    FocusAreaBulkCreate,
    FocusAreaListResponse,
    FocusAreaAnalytics,
    PriorityLevel
)
from app.utils.router_base import (
    logging, math, Dict, List, Any, APIRouter, Depends, Query, status,
    AsyncSession, select, func, and_, get_db_session, User, get_current_active_user,
    errors, db_handler, db_helpers, BaseRouterOperations, PaginationParams, 
    create_paginated_response, create_analytics_response
)

base_ops = BaseRouterOperations(__name__)

router = APIRouter(prefix="/api/v1/users/focus-areas", tags=["Focus Areas Management"])


@router.get("/", response_model=FocusAreaListResponse)
async def get_focus_areas(
    pagination: PaginationParams = Depends(),
    priority: Optional[PriorityLevel] = Query(None, description="Filter by priority"),
    search: Optional[str] = Query(None, description="Search in focus area names"),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get user's focus areas with pagination and filtering.
    
    - **page**: Page number (starts at 1)
    - **per_page**: Number of items per page (max 100)
    - **priority**: Filter by priority level (1-4)
    - **search**: Search term for focus area names
    """
    async def _get_focus_areas_operation():
        # Prepare filters for basic cases
        filters = {}
        if priority:
            filters['priority'] = priority
        
        # Handle search separately as it requires custom query logic
        if search:
            query = select(UserFocusArea).where(
                and_(
                    UserFocusArea.user_id == current_user.id,
                    UserFocusArea.focus_area.ilike(f"%{search}%")
                )
            )
            if priority:
                query = query.where(UserFocusArea.priority == priority)
            
            # Get total count with search
            count_query = select(func.count()).select_from(query.subquery())
            total_items = await db.scalar(count_query) or 0
            
            # Apply pagination and ordering
            query = query.offset(pagination.offset).limit(pagination.per_page).order_by(
                UserFocusArea.priority.desc(),
                UserFocusArea.created_at.desc()
            )
            result = await db.execute(query)
            focus_areas = result.scalars().all()
        else:
            # Use optimized base operation for simple filtering
            focus_areas, total_items = await base_ops.get_user_resources_paginated(
                db, UserFocusArea, current_user.id, pagination, filters
            )
        
        # Convert to response models with priority labels
        items = []
        for fa in focus_areas:
            response = FocusAreaResponse.model_validate(fa)
            response.priority_label = fa.priority_label
            items.append(response)
        
        return FocusAreaListResponse(
            items=items,
            pagination={
                "page": pagination.page,
                "per_page": pagination.per_page,
                "total_items": total_items,
                "total_pages": math.ceil(total_items / pagination.per_page) if total_items > 0 else 1,
                "has_next": pagination.page < (math.ceil(total_items / pagination.per_page) if total_items > 0 else 1),
                "has_prev": pagination.page > 1
            }
        )
    
    return await base_ops.execute_db_operation(
        "get focus areas", _get_focus_areas_operation, db, rollback_on_error=False
    )


@router.get("/{focus_area_id}", response_model=FocusAreaResponse)
async def get_focus_area(
    focus_area_id: int,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific focus area by ID."""
    async def _get_focus_area_operation():
        focus_area = await db_helpers.get_model_by_field(
            db, UserFocusArea, "id", focus_area_id,
            validate_exists=True, resource_name="Focus area"
        )
        
        # Verify ownership
        if focus_area.user_id != current_user.id:
            raise errors.forbidden("You don't have access to this focus area")
        
        response = FocusAreaResponse.model_validate(focus_area)
        response.priority_label = focus_area.priority_label
        return response
    
    return await db_handler.handle_db_operation(
        "get focus area", _get_focus_area_operation, db, rollback_on_error=False
    )


@router.post("/", response_model=FocusAreaResponse, status_code=status.HTTP_201_CREATED)
async def create_focus_area(
    focus_area_data: FocusAreaCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new focus area.
    
    - **focus_area**: Name of the focus area (2-255 characters)
    - **keywords**: List of keywords (max 20, each 2-100 characters)
    - **priority**: Priority level (1=low, 2=medium, 3=high, 4=critical)
    """
    async def _create_focus_area_operation():
        # Check if focus area already exists for this user
        existing = await db.scalar(
            select(UserFocusArea).where(
                and_(
                    UserFocusArea.user_id == current_user.id,
                    UserFocusArea.focus_area == focus_area_data.focus_area
                )
            )
        )
        
        if existing:
            raise errors.conflict(f"Focus area '{focus_area_data.focus_area}' already exists")
        
        # Create new focus area
        new_focus_area = UserFocusArea(
            user_id=current_user.id,
            focus_area=focus_area_data.focus_area,
            keywords=focus_area_data.keywords,
            priority=focus_area_data.priority
        )
        
        db.add(new_focus_area)
        await db_helpers.safe_commit(db, "focus area creation")
        await db.refresh(new_focus_area)
        
        logger.info(f"Focus area created for user {current_user.email}: {new_focus_area.focus_area}")
        
        response = FocusAreaResponse.model_validate(new_focus_area)
        response.priority_label = new_focus_area.priority_label
        return response
    
    return await db_handler.handle_db_operation(
        "create focus area", _create_focus_area_operation, db
    )


@router.post("/bulk", response_model=List[FocusAreaResponse], status_code=status.HTTP_201_CREATED)
async def create_bulk_focus_areas(
    bulk_data: FocusAreaBulkCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create multiple focus areas at once.
    
    Maximum 10 focus areas per request.
    """
    async def _bulk_create_operation():
        created_areas = []
        
        for focus_area_data in bulk_data.focus_areas:
            # Check if already exists
            existing = await db.scalar(
                select(UserFocusArea).where(
                    and_(
                        UserFocusArea.user_id == current_user.id,
                        UserFocusArea.focus_area == focus_area_data.focus_area
                    )
                )
            )
            
            if not existing:
                new_focus_area = UserFocusArea(
                    user_id=current_user.id,
                    focus_area=focus_area_data.focus_area,
                    keywords=focus_area_data.keywords,
                    priority=focus_area_data.priority
                )
                db.add(new_focus_area)
                created_areas.append(new_focus_area)
        
        if created_areas:
            await db_helpers.safe_commit(db, "bulk focus area creation")
            
            # Refresh all created areas
            for area in created_areas:
                await db.refresh(area)
        
        logger.info(f"Bulk created {len(created_areas)} focus areas for user {current_user.email}")
        
        # Convert to response models
        responses = []
        for area in created_areas:
            response = FocusAreaResponse.model_validate(area)
            response.priority_label = area.priority_label
            responses.append(response)
        
        return responses
    
    return await db_handler.handle_db_operation(
        "bulk create focus areas", _bulk_create_operation, db
    )


@router.put("/{focus_area_id}", response_model=FocusAreaResponse)
async def update_focus_area(
    focus_area_id: int,
    update_data: FocusAreaUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update a focus area.
    
    Only provided fields will be updated.
    """
    async def _update_focus_area_operation():
        # Get existing focus area
        focus_area = await db_helpers.get_model_by_field(
            db, UserFocusArea, "id", focus_area_id,
            validate_exists=True, resource_name="Focus area"
        )
        
        # Verify ownership
        if focus_area.user_id != current_user.id:
            raise errors.forbidden("You don't have access to this focus area")
        
        # Check if new name conflicts with existing
        if update_data.focus_area and update_data.focus_area != focus_area.focus_area:
            existing = await db.scalar(
                select(UserFocusArea).where(
                    and_(
                        UserFocusArea.user_id == current_user.id,
                        UserFocusArea.focus_area == update_data.focus_area,
                        UserFocusArea.id != focus_area_id
                    )
                )
            )
            
            if existing:
                raise errors.conflict(f"Focus area '{update_data.focus_area}' already exists")
        
        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            if value is not None:
                setattr(focus_area, field, value)
        
        await db_helpers.safe_commit(db, "focus area update")
        await db.refresh(focus_area)
        
        logger.info(f"Focus area {focus_area_id} updated for user {current_user.email}")
        
        response = FocusAreaResponse.model_validate(focus_area)
        response.priority_label = focus_area.priority_label
        return response
    
    return await db_handler.handle_db_operation(
        "update focus area", _update_focus_area_operation, db
    )


@router.delete("/{focus_area_id}")
async def delete_focus_area(
    focus_area_id: int,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a focus area."""
    return await base_ops.delete_user_resource(
        db, UserFocusArea, focus_area_id, current_user.id, "focus area"
    )


@router.delete("/")
async def delete_all_focus_areas(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete all focus areas for the current user.
    
    **Warning**: This action cannot be undone.
    """
    async def _delete_all_operation():
        # Get all user's focus areas
        result = await db.execute(
            select(UserFocusArea).where(UserFocusArea.user_id == current_user.id)
        )
        focus_areas = result.scalars().all()
        
        if not focus_areas:
            raise errors.not_found("No focus areas to delete")
        
        count = len(focus_areas)
        
        # Delete all
        for area in focus_areas:
            await db.delete(area)
        
        await db_helpers.safe_commit(db, "delete all focus areas")
        
        logger.info(f"Deleted {count} focus areas for user {current_user.email}")
        
        return {"message": f"Successfully deleted {count} focus areas"}
    
    return await db_handler.handle_db_operation(
        "delete all focus areas", _delete_all_operation, db
    )


@router.get("/analytics/summary", response_model=FocusAreaAnalytics)
async def get_focus_area_analytics(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get analytics and insights about user's focus areas.
    
    Returns priority distribution, keyword analysis, and recommendations.
    """
    async def _get_analytics_operation():
        # Get all user's focus areas
        result = await db.execute(
            select(UserFocusArea).where(UserFocusArea.user_id == current_user.id)
        )
        focus_areas = result.scalars().all()
        
        # Calculate analytics
        total = len(focus_areas)
        priority_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        all_keywords = []
        
        for area in focus_areas:
            priority_dist[area.priority_label] += 1
            if area.keywords:
                all_keywords.extend(area.keywords)
        
        # Get most common keywords
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        most_common = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        most_common_keywords = [k for k, _ in most_common]
        
        # Calculate coverage score (0-100)
        coverage_score = min(100, (total * 20))  # 5 areas = 100% coverage
        
        # Generate recommendations based on user's profile
        recommendations = _generate_focus_area_recommendations(current_user, focus_areas)
        
        return FocusAreaAnalytics(
            total_focus_areas=total,
            priority_distribution=priority_dist,
            keyword_count=len(set(all_keywords)),
            most_common_keywords=most_common_keywords,
            coverage_score=float(coverage_score),
            recommendations=recommendations
        )
    
    return await db_handler.handle_db_operation(
        "get focus area analytics", _get_analytics_operation, db, rollback_on_error=False
    )


def _generate_focus_area_recommendations(user: User, existing_areas: List[UserFocusArea]) -> List[str]:
    """Generate focus area recommendations based on user profile."""
    existing_names = {area.focus_area.lower() for area in existing_areas}
    
    # Common recommended focus areas
    all_recommendations = [
        "Artificial Intelligence and Machine Learning",
        "Cybersecurity Threats and Solutions",
        "Digital Transformation Strategies",
        "Supply Chain Management",
        "Sustainability and ESG",
        "Remote Work Technologies",
        "Customer Experience Innovation",
        "Data Privacy and Compliance",
        "Blockchain and Cryptocurrency",
        "Cloud Computing Services",
        "Market Trends and Analysis",
        "Competitive Landscape",
        "Regulatory Changes",
        "Emerging Technologies",
        "Partnership Opportunities"
    ]
    
    # Filter out already existing areas
    recommendations = []
    for rec in all_recommendations:
        if rec.lower() not in existing_names:
            recommendations.append(rec)
            if len(recommendations) >= 5:
                break
    
    return recommendations
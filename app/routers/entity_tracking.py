"""
Entity tracking management routes for the Competitive Intelligence v2 API.
"""

from typing import Optional

from app.models.tracking import TrackingEntity, UserEntityTracking
from app.schemas.entity_tracking import (
    TrackingEntityCreate,
    TrackingEntityUpdate,
    TrackingEntityResponse,
    UserEntityTrackingCreate,
    UserEntityTrackingUpdate,
    UserEntityTrackingResponse,
    EntityTrackingListResponse,
    EntityTrackingAnalytics,
    EntitySearchRequest,
    EntityType,
    TrackingPriority
)
from app.utils.router_base import (
    logging, math, Dict, List, Any, APIRouter, Depends, Query, status,
    AsyncSession, select, func, and_, or_, selectinload, get_db_session, User, get_current_active_user,
    errors, db_handler, db_helpers, BaseRouterOperations, PaginationParams, 
    create_paginated_response, create_analytics_response
)

base_ops = BaseRouterOperations(__name__)

router = APIRouter(prefix="/api/v1/users/entity-tracking", tags=["Entity Tracking Management"])


def _create_entity_response(entity) -> TrackingEntityResponse:
    """Helper to create TrackingEntityResponse with proper field mapping."""
    metadata_value = entity.metadata_json
    if metadata_value is None:
        metadata_value = {}
    elif not isinstance(metadata_value, dict):
        metadata_value = {}
        
    return TrackingEntityResponse(
        id=entity.id,
        name=entity.name,
        entity_type=entity.entity_type,
        domain=entity.domain,
        description=entity.description,
        industry=entity.industry,
        metadata=metadata_value,
        created_at=entity.created_at
    )


@router.get("/entities", response_model=List[TrackingEntityResponse])
async def get_available_entities(
    entity_type: Optional[EntityType] = Query(None, description="Filter by entity type"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    search: Optional[str] = Query(None, description="Search in entity names"),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get available entities to track.
    
    - **entity_type**: Filter by entity type
    - **industry**: Filter by industry
    - **search**: Search term for entity names
    """
    async def _get_entities_operation():
        query = select(TrackingEntity)
        
        # Apply filters
        if entity_type:
            query = query.where(TrackingEntity.entity_type == entity_type)
        
        if industry:
            query = query.where(TrackingEntity.industry == industry)
        
        if search:
            query = query.where(
                or_(
                    TrackingEntity.name.ilike(f"%{search}%"),
                    TrackingEntity.description.ilike(f"%{search}%")
                )
            )
        
        query = query.order_by(TrackingEntity.name)
        
        result = await db.execute(query)
        entities = result.scalars().all()
        
        return [_create_entity_response(entity) for entity in entities]
    
    return await db_handler.handle_db_operation(
        "get available entities", _get_entities_operation, db, rollback_on_error=False
    )


@router.post("/entities", response_model=TrackingEntityResponse, status_code=status.HTTP_201_CREATED)
async def create_tracking_entity(
    entity_data: TrackingEntityCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new tracking entity.
    
    Creates a new entity that can be tracked by users.
    """
    async def _create_entity_operation():
        # Check if entity already exists
        existing = await db.scalar(
            select(TrackingEntity).where(
                and_(
                    TrackingEntity.name == entity_data.name,
                    TrackingEntity.entity_type == entity_data.entity_type
                )
            )
        )
        
        if existing:
            raise errors.conflict(f"Entity '{entity_data.name}' of type '{entity_data.entity_type}' already exists")
        
        # Create new entity
        new_entity = TrackingEntity(
            name=entity_data.name,
            entity_type=entity_data.entity_type,
            domain=entity_data.domain,
            description=entity_data.description,
            industry=entity_data.industry,
            metadata_json=entity_data.metadata
        )
        
        db.add(new_entity)
        await db_helpers.safe_commit(db, "entity creation")
        await db.refresh(new_entity)
        
        logger.info(f"Entity created: {new_entity.name} ({new_entity.entity_type})")
        logger.info(f"Metadata type: {type(new_entity.metadata_json)}, value: {new_entity.metadata_json}")
        
        # Create response with proper field mapping
        metadata_value = new_entity.metadata_json
        if metadata_value is None:
            metadata_value = {}
        elif not isinstance(metadata_value, dict):
            metadata_value = {}
            
        return TrackingEntityResponse(
            id=new_entity.id,
            name=new_entity.name,
            entity_type=new_entity.entity_type,
            domain=new_entity.domain,
            description=new_entity.description,
            industry=new_entity.industry,
            metadata=metadata_value,
            created_at=new_entity.created_at
        )
    
    return await db_handler.handle_db_operation(
        "create tracking entity", _create_entity_operation, db
    )


@router.get("/", response_model=EntityTrackingListResponse)
async def get_user_tracked_entities(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    priority: Optional[TrackingPriority] = Query(None, description="Filter by priority"),
    entity_type: Optional[EntityType] = Query(None, description="Filter by entity type"),
    enabled_only: bool = Query(True, description="Show only enabled tracking"),
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get user's tracked entities with pagination.
    
    - **page**: Page number (starts at 1)
    - **per_page**: Number of items per page (max 100)
    - **priority**: Filter by tracking priority
    - **entity_type**: Filter by entity type
    - **enabled_only**: Show only enabled tracking (default: true)
    """
    async def _get_tracked_entities_operation():
        # Build query with join
        query = select(UserEntityTracking).where(
            UserEntityTracking.user_id == current_user.id
        ).options(selectinload(UserEntityTracking.entity))
        
        # Apply filters
        if priority:
            query = query.where(UserEntityTracking.priority == priority)
        
        if enabled_only:
            query = query.where(UserEntityTracking.tracking_enabled == True)
        
        if entity_type:
            query = query.join(TrackingEntity).where(
                TrackingEntity.entity_type == entity_type
            )
        
        # Get total count
        count_query = select(func.count()).select_from(UserEntityTracking).where(
            UserEntityTracking.user_id == current_user.id
        )
        if priority:
            count_query = count_query.where(UserEntityTracking.priority == priority)
        if enabled_only:
            count_query = count_query.where(UserEntityTracking.tracking_enabled == True)
        if entity_type:
            count_query = count_query.join(TrackingEntity).where(
                TrackingEntity.entity_type == entity_type
            )
        
        total = await db.scalar(count_query) or 0
        
        # Apply pagination
        offset = (page - 1) * per_page
        query = query.offset(offset).limit(per_page).order_by(
            UserEntityTracking.priority.desc(),
            UserEntityTracking.created_at.desc()
        )
        
        # Execute query
        result = await db.execute(query)
        tracked_entities = result.scalars().all()
        
        # Calculate pages
        pages = math.ceil(total / per_page) if total > 0 else 0
        
        # Convert to response models
        items = []
        for tracking in tracked_entities:
            response = UserEntityTrackingResponse(
                id=tracking.id,
                user_id=tracking.user_id,
                entity_id=tracking.entity_id,
                entity=_create_entity_response(tracking.entity),
                priority=tracking.priority,
                priority_label=tracking.priority_label,
                custom_keywords=tracking.custom_keywords or [],
                tracking_enabled=tracking.tracking_enabled,
                created_at=tracking.created_at
            )
            items.append(response)
        
        return EntityTrackingListResponse(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            pages=pages
        )
    
    return await db_handler.handle_db_operation(
        "get tracked entities", _get_tracked_entities_operation, db, rollback_on_error=False
    )


@router.post("/", response_model=UserEntityTrackingResponse, status_code=status.HTTP_201_CREATED)
async def track_entity(
    tracking_data: UserEntityTrackingCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Start tracking an entity.
    
    - **entity_id**: ID of the entity to track
    - **priority**: Tracking priority (1-4)
    - **custom_keywords**: Custom keywords for this entity (max 10)
    - **tracking_enabled**: Whether tracking is enabled
    """
    async def _track_entity_operation():
        # Verify entity exists
        entity = await db_helpers.get_model_by_field(
            db, TrackingEntity, "id", tracking_data.entity_id,
            validate_exists=True, resource_name="Tracking entity"
        )
        
        # Check if already tracking
        existing = await db.scalar(
            select(UserEntityTracking).where(
                and_(
                    UserEntityTracking.user_id == current_user.id,
                    UserEntityTracking.entity_id == tracking_data.entity_id
                )
            )
        )
        
        if existing:
            raise errors.conflict(f"Already tracking entity '{entity.name}'")
        
        # Create tracking record
        new_tracking = UserEntityTracking(
            user_id=current_user.id,
            entity_id=tracking_data.entity_id,
            priority=tracking_data.priority,
            custom_keywords=tracking_data.custom_keywords,
            tracking_enabled=tracking_data.tracking_enabled
        )
        
        db.add(new_tracking)
        await db_helpers.safe_commit(db, "entity tracking creation")
        await db.refresh(new_tracking)
        
        # Load entity relationship
        await db.refresh(new_tracking, ["entity"])
        
        logger.info(f"User {current_user.email} started tracking entity: {entity.name}")
        
        return UserEntityTrackingResponse(
            id=new_tracking.id,
            user_id=new_tracking.user_id,
            entity_id=new_tracking.entity_id,
            entity=_create_entity_response(new_tracking.entity),
            priority=new_tracking.priority,
            priority_label=new_tracking.priority_label,
            custom_keywords=new_tracking.custom_keywords or [],
            tracking_enabled=new_tracking.tracking_enabled,
            created_at=new_tracking.created_at
        )
    
    return await db_handler.handle_db_operation(
        "track entity", _track_entity_operation, db
    )


@router.put("/{tracking_id}", response_model=UserEntityTrackingResponse)
async def update_entity_tracking(
    tracking_id: int,
    update_data: UserEntityTrackingUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update entity tracking settings.
    
    Only provided fields will be updated.
    """
    async def _update_tracking_operation():
        # Get tracking record
        tracking = await db_helpers.get_model_by_field(
            db, UserEntityTracking, "id", tracking_id,
            validate_exists=True, resource_name="Entity tracking"
        )
        
        # Verify ownership
        if tracking.user_id != current_user.id:
            raise errors.forbidden("You don't have access to this tracking record")
        
        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            if value is not None:
                setattr(tracking, field, value)
        
        await db_helpers.safe_commit(db, "entity tracking update")
        await db.refresh(tracking)
        
        # Load entity relationship
        await db.refresh(tracking, ["entity"])
        
        logger.info(f"Entity tracking {tracking_id} updated for user {current_user.email}")
        
        return UserEntityTrackingResponse(
            id=tracking.id,
            user_id=tracking.user_id,
            entity_id=tracking.entity_id,
            entity=_create_entity_response(tracking.entity),
            priority=tracking.priority,
            priority_label=tracking.priority_label,
            custom_keywords=tracking.custom_keywords or [],
            tracking_enabled=tracking.tracking_enabled,
            created_at=tracking.created_at
        )
    
    return await db_handler.handle_db_operation(
        "update entity tracking", _update_tracking_operation, db
    )


@router.delete("/{tracking_id}")
async def stop_tracking_entity(
    tracking_id: int,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Stop tracking an entity."""
    async def _stop_tracking_operation():
        # Get tracking record
        tracking = await db_helpers.get_model_by_field(
            db, UserEntityTracking, "id", tracking_id,
            validate_exists=True, resource_name="Entity tracking"
        )
        
        # Verify ownership
        if tracking.user_id != current_user.id:
            raise errors.forbidden("You don't have access to this tracking record")
        
        # Load entity for response
        await db.refresh(tracking, ["entity"])
        entity_name = tracking.entity.name
        
        await db_helpers.safe_delete(db, tracking, "stop entity tracking")
        
        logger.info(f"User {current_user.email} stopped tracking entity: {entity_name}")
        
        return {"message": f"Stopped tracking entity '{entity_name}'"}
    
    return await db_handler.handle_db_operation(
        "stop tracking entity", _stop_tracking_operation, db
    )


@router.post("/search", response_model=List[TrackingEntityResponse])
async def search_entities(
    search_request: EntitySearchRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Search for entities to track.
    
    Search across entity names, descriptions, and filter by type or industry.
    """
    async def _search_entities_operation():
        query = select(TrackingEntity)
        
        # Apply search filters
        if search_request.query:
            query = query.where(
                or_(
                    TrackingEntity.name.ilike(f"%{search_request.query}%"),
                    TrackingEntity.description.ilike(f"%{search_request.query}%")
                )
            )
        
        if search_request.entity_types:
            query = query.where(TrackingEntity.entity_type.in_(search_request.entity_types))
        
        if search_request.industries:
            query = query.where(TrackingEntity.industry.in_(search_request.industries))
        
        query = query.order_by(TrackingEntity.name).limit(50)
        
        result = await db.execute(query)
        entities = result.scalars().all()
        
        return [_create_entity_response(entity) for entity in entities]
    
    return await db_handler.handle_db_operation(
        "search entities", _search_entities_operation, db, rollback_on_error=False
    )


@router.get("/analytics", response_model=EntityTrackingAnalytics)
async def get_tracking_analytics(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get analytics about user's entity tracking.
    
    Returns statistics about tracked entities, priorities, and keywords.
    """
    async def _get_analytics_operation():
        # Get all user's tracking records with entities
        result = await db.execute(
            select(UserEntityTracking)
            .where(UserEntityTracking.user_id == current_user.id)
            .options(selectinload(UserEntityTracking.entity))
        )
        tracking_records = result.scalars().all()
        
        # Calculate analytics
        total = len(tracking_records)
        entities_by_type = {}
        priority_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        enabled_count = 0
        disabled_count = 0
        industries = {}
        all_keywords = []
        
        for record in tracking_records:
            # Count by entity type
            entity_type = record.entity.entity_type
            entities_by_type[entity_type] = entities_by_type.get(entity_type, 0) + 1
            
            # Priority distribution
            priority_dist[record.priority_label] += 1
            
            # Enabled/disabled count
            if record.tracking_enabled:
                enabled_count += 1
            else:
                disabled_count += 1
            
            # Industries
            if record.entity.industry:
                industries[record.entity.industry] = industries.get(record.entity.industry, 0) + 1
            
            # Keywords
            if record.custom_keywords:
                all_keywords.extend(record.custom_keywords)
        
        # Get top industries
        top_industries = sorted(industries.items(), key=lambda x: x[1], reverse=True)[:5]
        top_industries_list = [industry for industry, _ in top_industries]
        
        # Get keyword cloud (unique keywords)
        keyword_cloud = list(set(all_keywords))[:20]
        
        return EntityTrackingAnalytics(
            total_tracked_entities=total,
            entities_by_type=entities_by_type,
            priority_distribution=priority_dist,
            enabled_count=enabled_count,
            disabled_count=disabled_count,
            top_industries=top_industries_list,
            keyword_cloud=keyword_cloud
        )
    
    return await db_handler.handle_db_operation(
        "get tracking analytics", _get_analytics_operation, db, rollback_on_error=False
    )
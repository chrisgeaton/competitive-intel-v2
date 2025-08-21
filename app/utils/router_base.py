"""
Common base utilities for router operations to reduce code duplication.
"""

import logging
import math
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Awaitable
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from app.database import get_db_session
from app.models.user import User
from app.middleware import get_current_active_user
from app.utils.exceptions import errors, db_handler, validators
from app.utils.database import db_helpers

# Common imports that all routers need
__all__ = [
    'logging', 'math', 'Dict', 'List', 'Optional', 'Any',
    'APIRouter', 'Depends', 'Query', 'status',
    'AsyncSession', 'select', 'func', 'and_', 'or_', 'selectinload',
    'get_db_session', 'User', 'get_current_active_user',
    'errors', 'db_handler', 'db_helpers', 'validators',
    'BaseRouterOperations', 'PaginationParams', 'create_paginated_response', 'create_analytics_response'
]

T = TypeVar('T')

class PaginationParams:
    """Standard pagination parameters."""
    
    def __init__(
        self,
        page: int = Query(1, ge=1, description="Page number"),
        per_page: int = Query(10, ge=1, le=100, description="Items per page")
    ):
        self.page = page
        self.per_page = per_page
        self.offset = (page - 1) * per_page

class BaseRouterOperations:
    """Base class for common router operations."""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
    
    async def execute_db_operation(
        self,
        operation_name: str,
        operation_func: Callable[[], Awaitable[T]],
        db: AsyncSession,
        rollback_on_error: bool = True
    ) -> T:
        """Execute database operation with standard error handling."""
        return await db_handler.handle_db_operation(
            operation_name, operation_func, db, rollback_on_error
        )
    
    async def get_user_resource_by_id(
        self,
        db: AsyncSession,
        model_class: type,
        resource_id: int,
        user_id: int,
        resource_name: str = "resource",
        load_relations: bool = False
    ):
        """Get user-owned resource by ID with validation."""
        async def _get_resource_operation():
            # Build query
            query = select(model_class).where(
                and_(
                    model_class.id == resource_id,
                    model_class.user_id == user_id
                )
            )
            
            # Add relationship loading if requested
            if load_relations and hasattr(model_class, '__mapper__'):
                for relationship in model_class.__mapper__.relationships:
                    query = query.options(selectinload(getattr(model_class, relationship.key)))
            
            result = await db.execute(query)
            resource = result.scalar_one_or_none()
            
            if not resource:
                raise errors.not_found(f"{resource_name.title()} not found")
            
            return resource
        
        return await self.execute_db_operation(
            f"get {resource_name}", _get_resource_operation, db, False
        )
    
    async def get_user_resources_paginated(
        self,
        db: AsyncSession,
        model_class: type,
        user_id: int,
        pagination: PaginationParams,
        filters: Optional[Dict[str, Any]] = None,
        load_relations: bool = False
    ):
        """Get paginated user resources with optional filters."""
        async def _get_resources_operation():
            # Base query
            query = select(model_class).where(model_class.user_id == user_id)
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if value is not None and hasattr(model_class, field):
                        query = query.where(getattr(model_class, field) == value)
            
            # Add relationship loading if requested
            if load_relations and hasattr(model_class, '__mapper__'):
                for relationship in model_class.__mapper__.relationships:
                    query = query.options(selectinload(getattr(model_class, relationship.key)))
            
            # Get total count
            count_query = select(func.count()).select_from(
                query.subquery()
            )
            total_result = await db.execute(count_query)
            total_items = total_result.scalar()
            
            # Apply pagination
            paginated_query = query.offset(pagination.offset).limit(pagination.per_page)
            result = await db.execute(paginated_query)
            items = result.scalars().all()
            
            return items, total_items
        
        return await self.execute_db_operation(
            "get paginated resources", _get_resources_operation, db, False
        )
    
    async def delete_user_resource(
        self,
        db: AsyncSession,
        model_class: type,
        resource_id: int,
        user_id: int,
        resource_name: str = "resource"
    ):
        """Delete user-owned resource with validation."""
        async def _delete_resource_operation():
            # Get resource first to validate ownership
            resource = await self.get_user_resource_by_id(
                db, model_class, resource_id, user_id, resource_name
            )
            
            # Delete the resource
            await db_helpers.safe_delete(db, resource, f"delete {resource_name}")
            
            self.logger.info(f"{resource_name.title()} deleted: ID {resource_id} for user {user_id}")
            return {"message": f"{resource_name.title()} deleted successfully"}
        
        return await self.execute_db_operation(
            f"delete {resource_name}", _delete_resource_operation, db
        )

def create_paginated_response(
    items: List[T],
    total_items: int,
    pagination: PaginationParams,
    response_class: type
) -> Dict[str, Any]:
    """Create standardized paginated response."""
    total_pages = math.ceil(total_items / pagination.per_page) if total_items > 0 else 1
    
    return {
        "items": [response_class.model_validate(item) for item in items],
        "pagination": {
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": pagination.page < total_pages,
            "has_prev": pagination.page > 1
        }
    }

def create_analytics_response(
    title: str,
    metrics: Dict[str, Any],
    insights: List[str] = None
) -> Dict[str, Any]:
    """Create standardized analytics response."""
    return {
        "title": title,
        "metrics": metrics,
        "insights": insights or [],
        "generated_at": db_helpers.get_utc_now().isoformat()
    }
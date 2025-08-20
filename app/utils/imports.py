"""
Common imports for consistent import patterns across modules.
"""

# Standard library imports
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

# FastAPI imports
from fastapi import APIRouter, Depends, Request, status
from fastapi.security import HTTPBearer

# SQLAlchemy imports
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

# Application imports
from app.database import get_db_session
from app.middleware import get_current_user, get_current_active_user
from app.utils.exceptions import errors, db_handler, validators
from app.utils.database import db_helpers

# Commonly used logger setup
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name."""
    return logging.getLogger(name)

# Common dependencies
CommonDeps = {
    'db': Depends(get_db_session),
    'current_user': Depends(get_current_user),
    'current_active_user': Depends(get_current_active_user)
}

# Common status codes
StatusCodes = {
    'CREATED': status.HTTP_201_CREATED,
    'OK': status.HTTP_200_OK,
    'UNAUTHORIZED': status.HTTP_401_UNAUTHORIZED,
    'NOT_FOUND': status.HTTP_404_NOT_FOUND,
    'BAD_REQUEST': status.HTTP_400_BAD_REQUEST,
    'INTERNAL_ERROR': status.HTTP_500_INTERNAL_SERVER_ERROR
}
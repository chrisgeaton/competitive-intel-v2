"""
API routes for the Competitive Intelligence v2 system.
"""

from .auth import router as auth_router

__all__ = ["auth_router"]
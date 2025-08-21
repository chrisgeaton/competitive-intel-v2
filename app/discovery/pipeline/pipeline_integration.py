"""
Pipeline Integration - Integration layer for authentication, discovery engines, and API endpoints.

Provides seamless integration between the daily processing pipeline and existing
authentication systems, discovery engines, ML models, and API endpoints with
proper security, error handling, and resource management.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.database import get_db_session
from app.models.user import User
from app.models.strategic_profile import UserStrategicProfile
from app.models.tracking import TrackingEntity
from app.models.delivery import UserDeliveryPreferences
# from app.auth import verify_token, get_current_user  # TODO: Fix auth imports
from app.services.discovery_service import DiscoveryService

from ..orchestrator import DiscoveryOrchestrator, DiscoveryRequest, DiscoveryResponse
from ..engines.source_manager import SourceManager
from ..ml_integration import DiscoveryMLIntegrator
from ..user_config_integration import UserConfigIntegrator
from ..utils import (
    get_config,
    UnifiedErrorHandler,
    AsyncSessionManager,
    get_user_context_cache
)

from .daily_discovery_pipeline import DailyDiscoveryPipeline, UserDiscoveryContext
from .content_processor import ContentProcessor
from .ml_training_pipeline import MLTrainingPipeline
from .job_scheduler import JobScheduler, JobType
from .monitoring_pipeline import MonitoringPipeline


@dataclass
class PipelineContext:
    """Context for pipeline operations with authentication and permissions."""
    user_id: Optional[int] = None
    is_authenticated: bool = False
    is_admin: bool = False
    permissions: List[str] = None
    request_id: str = ""
    client_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.client_info is None:
            self.client_info = {}


class PipelineAuthenticationManager:
    """Manages authentication and authorization for pipeline operations."""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.auth")
        self.error_handler = UnifiedErrorHandler()
        
        # Permission definitions
        self.pipeline_permissions = {
            'pipeline.run': 'Can execute pipeline operations',
            'pipeline.admin': 'Can manage pipeline configuration',
            'pipeline.monitor': 'Can access monitoring data',
            'pipeline.schedule': 'Can schedule jobs',
            'pipeline.ml_training': 'Can trigger ML training',
            'pipeline.user_data': 'Can access user data for processing'
        }
        
        # Admin user IDs (in production, would be in database)
        self.admin_users = set()  # Will be populated from database
    
    async def authenticate_request(self, token: Optional[str], 
                                 required_permission: Optional[str] = None) -> PipelineContext:
        """Authenticate request and build pipeline context."""
        context = PipelineContext()
        
        try:
            if not token:
                return context  # Unauthenticated context
            
            # Verify token using existing auth system
            payload = verify_token(token)
            if not payload:
                return context
            
            user_id = payload.get('sub')
            if not user_id:
                return context
            
            # Get user from database
            async with get_db_session() as session:
                result = await session.execute(
                    select(User).where(User.id == int(user_id))
                )
                user = result.scalar_one_or_none()
                
                if not user or not user.is_active:
                    return context
                
                # Build authenticated context
                context.user_id = user.id
                context.is_authenticated = True
                context.is_admin = user.id in self.admin_users or user.email.endswith('@admin.local')
                
                # Determine permissions
                context.permissions = await self._get_user_permissions(user)
                
                # Check required permission
                if required_permission and not self._has_permission(context, required_permission):
                    context.is_authenticated = False
                    self.logger.warning(f"User {user_id} lacks required permission: {required_permission}")
                
                return context
                
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return PipelineContext()  # Return unauthenticated context
    
    async def _get_user_permissions(self, user: User) -> List[str]:
        """Get user permissions based on role and configuration."""
        permissions = ['pipeline.monitor']  # Basic permission for all users
        
        # Admin users get all permissions
        if user.id in self.admin_users or user.email.endswith('@admin.local'):
            permissions.extend([
                'pipeline.run',
                'pipeline.admin', 
                'pipeline.schedule',
                'pipeline.ml_training',
                'pipeline.user_data'
            ])
        else:
            # Regular users can access their own data
            permissions.extend([
                'pipeline.user_data'  # Only for their own data
            ])
        
        return permissions
    
    def _has_permission(self, context: PipelineContext, permission: str) -> bool:
        """Check if context has required permission."""
        return context.is_authenticated and (
            context.is_admin or permission in context.permissions
        )
    
    async def authorize_user_data_access(self, context: PipelineContext, 
                                       target_user_id: int) -> bool:
        """Check if context can access specific user's data."""
        if not context.is_authenticated:
            return False
        
        # Admin users can access any user's data
        if context.is_admin:
            return True
        
        # Users can access their own data
        if context.user_id == target_user_id:
            return True
        
        return False


class PipelineServiceIntegrator:
    """Integrates pipeline with existing services and engines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("pipeline.service_integrator")
        self.error_handler = UnifiedErrorHandler()
        
        # Initialize service connections
        self.discovery_service = DiscoveryService()
        self.source_manager = None  # Will be initialized lazily
        self.ml_integrator = None   # Will be initialized lazily
        self.user_config_integrator = None  # Will be initialized lazily
        
        # Cache for service instances
        self.service_cache: Dict[str, Any] = {}
        
        self.logger.info("Pipeline Service Integrator initialized")
    
    async def get_discovery_orchestrator(self) -> DiscoveryOrchestrator:
        """Get configured discovery orchestrator instance."""
        if 'orchestrator' not in self.service_cache:
            orchestrator = DiscoveryOrchestrator(self.config)
            self.service_cache['orchestrator'] = orchestrator
        
        return self.service_cache['orchestrator']
    
    async def get_source_manager(self) -> SourceManager:
        """Get configured source manager instance."""
        if 'source_manager' not in self.service_cache:
            source_config = self.config.get('sources', {})
            source_manager = SourceManager(source_config)
            self.service_cache['source_manager'] = source_manager
        
        return self.service_cache['source_manager']
    
    async def get_ml_integrator(self) -> DiscoveryMLIntegrator:
        """Get configured ML integrator instance."""
        if 'ml_integrator' not in self.service_cache:
            ml_integrator = DiscoveryMLIntegrator(self.discovery_service)
            self.service_cache['ml_integrator'] = ml_integrator
        
        return self.service_cache['ml_integrator']
    
    async def get_user_config_integrator(self) -> UserConfigIntegrator:
        """Get configured user config integrator instance."""
        if 'user_config_integrator' not in self.service_cache:
            ml_integrator = await self.get_ml_integrator()
            user_config_integrator = UserConfigIntegrator(ml_integrator)
            self.service_cache['user_config_integrator'] = user_config_integrator
        
        return self.service_cache['user_config_integrator']
    
    async def execute_user_discovery(self, context: PipelineContext, 
                                   user_id: int, 
                                   discovery_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute discovery for a specific user with proper authentication."""
        
        # Check authorization
        auth_manager = PipelineAuthenticationManager()
        if not await auth_manager.authorize_user_data_access(context, user_id):
            raise PermissionError(f"Unauthorized access to user {user_id} data")
        
        try:
            # Get user context
            user_context = await self._build_user_discovery_context(user_id)
            if not user_context:
                raise ValueError(f"Could not build context for user {user_id}")
            
            # Create discovery request
            request_params = discovery_params or {}
            discovery_request = DiscoveryRequest(
                user_id=user_id,
                keywords=request_params.get('keywords', user_context.keywords),
                focus_areas=request_params.get('focus_areas', user_context.focus_areas),
                entities=request_params.get('entities', user_context.tracked_entities),
                limit=request_params.get('limit', 50),
                quality_threshold=request_params.get('quality_threshold', 0.5),
                enable_ml_scoring=request_params.get('enable_ml_scoring', True),
                enable_content_processing=request_params.get('enable_content_processing', True)
            )
            
            # Execute discovery
            orchestrator = await self.get_discovery_orchestrator()
            discovery_response = await orchestrator.discover_content(discovery_request)
            
            # Process results
            content_processor = ContentProcessor()
            processed_results = await content_processor.process_user_content(
                user_context.__dict__, discovery_response.items
            )
            
            return {
                'user_id': user_id,
                'request_id': discovery_response.request_id,
                'items_found': len(discovery_response.items),
                'items_processed': len(processed_results),
                'processing_time': discovery_response.processing_time,
                'engines_used': discovery_response.engines_used,
                'quality_distribution': discovery_response.quality_distribution,
                'results': [
                    {
                        'title': result.processed_content.get('title', ''),
                        'url': result.original_item.url,
                        'score': result.ml_scores.get('overall_score', 0.0),
                        'relevance': result.ml_scores.get('relevance_score', 0.0),
                        'source': result.original_item.source_name,
                        'published_date': result.original_item.metadata.get('published_date'),
                        'categories': result.processed_content.get('categories', [])
                    }
                    for result in processed_results[:20]  # Return top 20 results
                ]
            }
            
        except Exception as e:
            self.logger.error(f"User discovery failed for user {user_id}: {e}")
            raise
    
    async def _build_user_discovery_context(self, user_id: int) -> Optional[UserDiscoveryContext]:
        """Build user discovery context from database."""
        try:
            async with get_db_session() as session:
                # Get user with related data
                result = await session.execute(
                    select(User)
                    .where(User.id == user_id)
                    .options(
                        User.focus_areas.selectin_load(),
                        User.tracked_entities.selectin_load()
                    )
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    return None
                
                # Get strategic profile
                strategic_result = await session.execute(
                    select(UserStrategicProfile).where(UserStrategicProfile.user_id == user_id)
                )
                strategic_profile = strategic_result.scalar_one_or_none()
                
                # Get delivery preferences
                delivery_result = await session.execute(
                    select(UserDeliveryPreferences).where(UserDeliveryPreferences.user_id == user_id)
                )
                delivery_prefs = delivery_result.scalar_one_or_none()
                
                # Build context
                focus_areas = [fa.focus_area for fa in user.focus_areas] if user.focus_areas else []
                tracked_entities = [entity.entity_name for entity in user.tracked_entities] if user.tracked_entities else []
                keywords = focus_areas[:5] + tracked_entities[:5]  # Combine and limit
                
                strategic_context = {}
                if strategic_profile:
                    strategic_context = {
                        'industry': strategic_profile.industry,
                        'organization_type': strategic_profile.organization_type,
                        'role': strategic_profile.role
                    }
                
                delivery_config = {}
                if delivery_prefs:
                    delivery_config = {
                        'frequency': delivery_prefs.email_frequency,
                        'max_items': delivery_prefs.max_items_per_digest or 20,
                        'content_types': delivery_prefs.content_types or []
                    }
                
                return UserDiscoveryContext(
                    user_id=user_id,
                    email=user.email,
                    focus_areas=focus_areas,
                    tracked_entities=tracked_entities,
                    keywords=keywords,
                    delivery_preferences=delivery_config,
                    strategic_context=strategic_context,
                    priority_score=delivery_prefs.priority_score if delivery_prefs else 1.0,
                    last_processed=delivery_prefs.last_delivery if delivery_prefs else None
                )
                
        except Exception as e:
            self.logger.error(f"Failed to build user context for {user_id}: {e}")
            return None
    
    async def close_connections(self):
        """Close all service connections."""
        try:
            # Close cached services
            for service_name, service in self.service_cache.items():
                if hasattr(service, 'close'):
                    await service.close()
            
            self.service_cache.clear()
            self.logger.info("Closed all service connections")
            
        except Exception as e:
            self.logger.error(f"Error closing service connections: {e}")


class PipelineAPIIntegration:
    """Provides API integration layer for pipeline operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("pipeline.api_integration")
        
        # Initialize components
        self.auth_manager = PipelineAuthenticationManager()
        self.service_integrator = PipelineServiceIntegrator(config)
        
        # Initialize pipeline components
        self.daily_pipeline = DailyDiscoveryPipeline(config)
        self.ml_training_pipeline = MLTrainingPipeline(config)
        self.job_scheduler = JobScheduler(config)
        self.monitoring_pipeline = MonitoringPipeline()
        
        # Cache for user contexts
        self.user_context_cache = get_user_context_cache()
        
        self.logger.info("Pipeline API Integration initialized")
    
    async def authenticate_and_execute(self, token: Optional[str], 
                                     operation: str,
                                     operation_func,
                                     *args, **kwargs) -> Dict[str, Any]:
        """Authenticate request and execute operation with proper error handling."""
        
        # Determine required permission
        permission_map = {
            'run_pipeline': 'pipeline.run',
            'schedule_job': 'pipeline.schedule', 
            'trigger_ml_training': 'pipeline.ml_training',
            'get_monitoring_data': 'pipeline.monitor',
            'user_discovery': 'pipeline.user_data',
            'pipeline_admin': 'pipeline.admin'
        }
        
        required_permission = permission_map.get(operation)
        
        try:
            # Authenticate request
            context = await self.auth_manager.authenticate_request(token, required_permission)
            
            if required_permission and not context.is_authenticated:
                return {
                    'success': False,
                    'error': 'Authentication required',
                    'error_type': 'authentication_error'
                }
            
            # Execute operation with context
            result = await operation_func(context, *args, **kwargs)
            
            return {
                'success': True,
                'data': result,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except PermissionError as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': 'permission_error'
            }
        except ValueError as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': 'validation_error'
            }
        except Exception as e:
            self.logger.error(f"Operation {operation} failed: {e}")
            return {
                'success': False,
                'error': 'Internal server error',
                'error_type': 'internal_error'
            }
    
    # Pipeline operation endpoints
    
    async def run_daily_pipeline(self, context: PipelineContext, 
                               parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute daily discovery pipeline."""
        
        if parameters:
            # Update pipeline configuration
            for key, value in parameters.items():
                if hasattr(self.daily_pipeline, key):
                    setattr(self.daily_pipeline, key, value)
        
        # Execute pipeline
        metrics = await self.daily_pipeline.run_daily_discovery()
        
        return {
            'pipeline_execution': {
                'users_processed': metrics.users_processed,
                'users_successful': metrics.users_successful,
                'users_failed': metrics.users_failed,
                'content_discovered': metrics.total_content_discovered,
                'processing_time_seconds': metrics.duration_seconds,
                'success_rate': metrics.success_rate,
                'errors': metrics.processing_errors
            }
        }
    
    async def trigger_ml_training(self, context: PipelineContext,
                                parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Trigger ML model training."""
        
        if parameters:
            # Update training parameters
            for key, value in parameters.items():
                if hasattr(self.ml_training_pipeline, key):
                    setattr(self.ml_training_pipeline, key, value)
        
        # Execute training
        training_results = await self.ml_training_pipeline.run_training_cycle()
        
        return {
            'training_results': [
                {
                    'model_type': result.model_type,
                    'model_version': result.model_version,
                    'training_samples': result.training_samples,
                    'validation_accuracy': result.validation_accuracy,
                    'improvement': result.improvement_over_previous,
                    'training_duration': result.training_duration,
                    'errors': result.training_errors
                }
                for result in training_results
            ]
        }
    
    async def schedule_pipeline_job(self, context: PipelineContext,
                                  job_type: str, 
                                  scheduled_time: Optional[str] = None,
                                  parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Schedule a pipeline job."""
        
        try:
            # Parse job type
            job_type_enum = JobType(job_type)
        except ValueError:
            raise ValueError(f"Invalid job type: {job_type}")
        
        # Parse scheduled time
        scheduled_datetime = None
        if scheduled_time:
            try:
                scheduled_datetime = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid scheduled time format: {scheduled_time}")
        
        # Schedule job
        job_id = await self.job_scheduler.schedule_job(
            job_type=job_type_enum,
            scheduled_time=scheduled_datetime,
            parameters=parameters or {}
        )
        
        return {
            'job_id': job_id,
            'job_type': job_type,
            'scheduled_time': scheduled_datetime.isoformat() if scheduled_datetime else None,
            'status': 'scheduled'
        }
    
    async def get_job_status(self, context: PipelineContext, job_id: int) -> Dict[str, Any]:
        """Get status of a scheduled job."""
        
        job_status = await self.job_scheduler.get_job_status(job_id)
        
        if not job_status:
            raise ValueError(f"Job {job_id} not found")
        
        return job_status
    
    async def cancel_job(self, context: PipelineContext, job_id: int) -> Dict[str, Any]:
        """Cancel a scheduled job."""
        
        success = await self.job_scheduler.cancel_job(job_id)
        
        return {
            'job_id': job_id,
            'cancelled': success,
            'message': 'Job cancelled successfully' if success else 'Failed to cancel job'
        }
    
    async def execute_user_discovery(self, context: PipelineContext,
                                   user_id: int,
                                   discovery_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute discovery for a specific user."""
        
        return await self.service_integrator.execute_user_discovery(
            context, user_id, discovery_params
        )
    
    async def get_system_status(self, context: PipelineContext) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return await self.monitoring_pipeline.get_system_status()
    
    async def get_performance_report(self, context: PipelineContext,
                                   operation_type: Optional[str] = None,
                                   hours: int = 24) -> Dict[str, Any]:
        """Get performance report."""
        
        return await self.monitoring_pipeline.get_performance_report(operation_type, hours)
    
    async def get_engagement_report(self, context: PipelineContext,
                                  days: int = 7) -> Dict[str, Any]:
        """Get user engagement report."""
        
        return await self.monitoring_pipeline.get_engagement_report(days)
    
    async def get_user_engagement_analysis(self, context: PipelineContext,
                                         user_id: int,
                                         days: int = 30) -> Dict[str, Any]:
        """Get user-specific engagement analysis."""
        
        # Check authorization for user data
        if not await self.auth_manager.authorize_user_data_access(context, user_id):
            raise PermissionError(f"Unauthorized access to user {user_id} engagement data")
        
        return await self.monitoring_pipeline.get_user_engagement_analysis(user_id, days)
    
    async def get_pipeline_configuration(self, context: PipelineContext) -> Dict[str, Any]:
        """Get pipeline configuration and status."""
        
        return {
            'daily_pipeline': await self.daily_pipeline.get_pipeline_status(),
            'job_scheduler': await self.job_scheduler.get_scheduler_status(),
            'monitoring': self.monitoring_pipeline.get_monitoring_configuration(),
            'ml_training': await self.ml_training_pipeline.get_training_status()
        }
    
    async def update_pipeline_configuration(self, context: PipelineContext,
                                          config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update pipeline configuration (admin only)."""
        
        if not context.is_admin:
            raise PermissionError("Admin access required for configuration updates")
        
        # Apply configuration updates (simplified - in production would validate)
        updated_configs = {}
        
        if 'daily_pipeline' in config_updates:
            daily_config = config_updates['daily_pipeline']
            for key, value in daily_config.items():
                if hasattr(self.daily_pipeline, key):
                    setattr(self.daily_pipeline, key, value)
                    updated_configs[f'daily_pipeline.{key}'] = value
        
        if 'ml_training' in config_updates:
            ml_config = config_updates['ml_training']
            for key, value in ml_config.items():
                if hasattr(self.ml_training_pipeline, key):
                    setattr(self.ml_training_pipeline, key, value)
                    updated_configs[f'ml_training.{key}'] = value
        
        return {
            'updated_configs': updated_configs,
            'message': f'Updated {len(updated_configs)} configuration values'
        }
    
    # Lifecycle management
    
    async def start_services(self):
        """Start all pipeline services."""
        try:
            # Start job scheduler
            await self.job_scheduler.start()
            
            # Start monitoring
            await self.monitoring_pipeline.start_pipeline_monitoring()
            
            self.logger.info("All pipeline services started")
            
        except Exception as e:
            self.logger.error(f"Failed to start services: {e}")
            raise
    
    async def stop_services(self):
        """Stop all pipeline services."""
        try:
            # Stop job scheduler
            await self.job_scheduler.stop()
            
            # Stop monitoring
            await self.monitoring_pipeline.stop_pipeline_monitoring()
            
            # Close service connections
            await self.service_integrator.close_connections()
            
            self.logger.info("All pipeline services stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping services: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        try:
            # Get system status
            system_status = await self.monitoring_pipeline.get_system_status()
            
            # Check service states
            services_status = {
                'job_scheduler': {
                    'running': self.job_scheduler.is_running,
                    'active_jobs': len(self.job_scheduler.active_jobs)
                },
                'monitoring': {
                    'running': self.monitoring_pipeline.is_monitoring
                },
                'daily_pipeline': await self.daily_pipeline.get_pipeline_status()
            }
            
            return {
                'overall_health': system_status.get('system_health', {}).get('overall_status', 'unknown'),
                'services': services_status,
                'system_metrics': system_status,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'overall_health': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
"""
Daily Discovery Pipeline - Main orchestrator for automated competitive intelligence.

Orchestrates daily content discovery for all users using optimized source engines,
intelligent resource management, and comprehensive error recovery. Designed to
handle thousands of users efficiently with progress tracking and monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload

from app.database import get_db_session
from app.models.user import User
from app.models.discovery import DiscoveryJob, DiscoveredSource, DiscoveredContent
from app.models.strategic_profile import UserStrategicProfile
from app.models.tracking import TrackingEntity
from app.models.delivery import UserDeliveryPreferences

from ..utils import (
    get_config, 
    UnifiedErrorHandler, 
    batch_processor, 
    get_user_context_cache,
    get_source_discovery_cache,
    AsyncSessionManager
)
from ..orchestrator import DiscoveryOrchestrator, DiscoveryRequest
from ..engines.source_manager import SourceManager
from .content_processor import ContentProcessor
from .monitoring_pipeline import MonitoringPipeline


@dataclass
class UserDiscoveryContext:
    """User context for discovery operations."""
    user_id: int
    email: str
    focus_areas: List[str] = field(default_factory=list)
    tracked_entities: List[str] = field(default_factory=list) 
    keywords: List[str] = field(default_factory=list)
    delivery_preferences: Dict[str, Any] = field(default_factory=dict)
    strategic_context: Dict[str, Any] = field(default_factory=dict)
    priority_score: float = 1.0
    last_processed: Optional[datetime] = None


@dataclass 
class PipelineMetrics:
    """Pipeline execution metrics."""
    start_time: datetime
    end_time: Optional[datetime] = None
    users_processed: int = 0
    users_successful: int = 0
    users_failed: int = 0
    total_content_discovered: int = 0
    total_content_delivered: int = 0
    processing_errors: List[str] = field(default_factory=list)
    performance_stats: Dict[str, float] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        if not self.end_time:
            return (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        if self.users_processed == 0:
            return 0.0
        return self.users_successful / self.users_processed


class DailyDiscoveryPipeline:
    """
    Main orchestrator for daily competitive intelligence discovery pipeline.
    
    Handles automated content discovery, processing, and delivery for thousands
    of users with intelligent resource management and comprehensive monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("pipeline.daily_discovery")
        
        # Initialize components
        self.discovery_config = get_config()
        self.error_handler = UnifiedErrorHandler()
        self.session_manager = AsyncSessionManager(name="daily_discovery_pipeline")
        self.content_processor = ContentProcessor()
        self.monitoring = MonitoringPipeline()
        
        # Pipeline configuration
        self.batch_size = config.get('batch_size', 50)
        self.max_concurrent_users = config.get('max_concurrent_users', 20)
        self.content_limit_per_user = config.get('content_limit_per_user', 100)
        self.quality_threshold = config.get('quality_threshold', 0.5)
        self.max_processing_time = config.get('max_processing_time_hours', 4) * 3600
        
        # Resource management
        self.user_semaphore = asyncio.Semaphore(self.max_concurrent_users)
        self.discovery_semaphore = asyncio.Semaphore(10)  # Limit concurrent discovery operations
        
        # Caching
        self.user_context_cache = get_user_context_cache()
        self.source_cache = get_source_discovery_cache()
        
        # State tracking
        self.current_job_id: Optional[int] = None
        self.active_user_tasks: Set[int] = set()
        self.pipeline_start_time = datetime.now(timezone.utc)
        
        self.logger.info(f"Daily discovery pipeline initialized with batch_size={self.batch_size}, max_concurrent={self.max_concurrent_users}")
    
    async def run_daily_discovery(self) -> PipelineMetrics:
        """
        Execute the complete daily discovery pipeline.
        
        Returns:
            PipelineMetrics: Comprehensive execution metrics
        """
        metrics = PipelineMetrics(start_time=datetime.now(timezone.utc))
        job_id = None
        
        try:
            self.logger.info("Starting daily discovery pipeline execution")
            
            # Create job record
            job_id = await self._create_pipeline_job()
            self.current_job_id = job_id
            
            # Start monitoring
            await self.monitoring.start_pipeline_monitoring(job_id)
            
            # Execute pipeline stages
            await self._execute_pipeline_stages(metrics)
            
            # Finalize execution
            await self._finalize_pipeline_execution(job_id, metrics)
            
            self.logger.info(f"Daily discovery pipeline completed successfully in {metrics.duration_seconds:.1f}s")
            
        except Exception as e:
            self.logger.error(f"Daily discovery pipeline failed: {e}")
            metrics.processing_errors.append(f"Pipeline failure: {str(e)}")
            
            if job_id:
                await self._mark_job_failed(job_id, str(e))
        
        finally:
            metrics.end_time = datetime.now(timezone.utc)
            await self.monitoring.stop_pipeline_monitoring()
            
            # Cleanup resources
            await self._cleanup_pipeline_resources()
        
        return metrics
    
    async def _execute_pipeline_stages(self, metrics: PipelineMetrics):
        """Execute the main pipeline stages."""
        
        # Stage 1: User discovery and context loading
        self.logger.info("Stage 1: Loading user contexts")
        users = await self._load_active_users()
        self.logger.info(f"Loaded {len(users)} active users for processing")
        
        # Stage 2: Batch processing of users
        self.logger.info("Stage 2: Processing users in batches")
        await self._process_users_in_batches(users, metrics)
        
        # Stage 3: System optimization and cleanup
        self.logger.info("Stage 3: System optimization and cleanup")
        await self._optimize_and_cleanup()
        
    async def _load_active_users(self) -> List[UserDiscoveryContext]:
        """Load active users who need content discovery."""
        try:
            async with get_db_session() as session:
                # Get users with delivery preferences (active users)
                result = await session.execute(
                    select(User, UserDeliveryPreferences, UserStrategicProfile)
                    .join(UserDeliveryPreferences, User.id == UserDeliveryPreferences.user_id)
                    .outerjoin(UserStrategicProfile, User.id == UserStrategicProfile.user_id) 
                    .options(
                        selectinload(User.focus_areas),
                        selectinload(User.tracked_entities)
                    )
                    .where(
                        and_(
                            User.is_active == True,
                            UserDeliveryPreferences.is_active == True,
                            or_(
                                UserDeliveryPreferences.last_delivery.is_(None),
                                UserDeliveryPreferences.last_delivery < datetime.now(timezone.utc) - timedelta(hours=20)
                            )
                        )
                    )
                    .order_by(
                        UserDeliveryPreferences.priority_score.desc(),
                        UserDeliveryPreferences.last_delivery.asc().nullsfirst()
                    )
                )
                
                user_contexts = []
                for user, delivery_prefs, strategic_profile in result:
                    context = await self._build_user_context(user, delivery_prefs, strategic_profile)
                    user_contexts.append(context)
                
                return user_contexts
                
        except Exception as e:
            self.logger.error(f"Failed to load active users: {e}")
            raise
    
    async def _build_user_context(self, user: User, delivery_prefs: UserDeliveryPreferences, 
                                strategic_profile: Optional[UserStrategicProfile]) -> UserDiscoveryContext:
        """Build comprehensive user context for discovery."""
        
        # Extract focus areas and entities
        focus_areas = [fa.focus_area for fa in user.focus_areas] if user.focus_areas else []
        tracked_entities = [entity.entity_name for entity in user.tracked_entities] if user.tracked_entities else []
        
        # Build keywords from various sources
        keywords = []
        if focus_areas:
            keywords.extend(focus_areas[:5])  # Top 5 focus areas
        if tracked_entities:
            keywords.extend(tracked_entities[:5])  # Top 5 entities
        
        # Strategic context
        strategic_context = {}
        if strategic_profile:
            strategic_context = {
                'industry': strategic_profile.industry,
                'organization_type': strategic_profile.organization_type,
                'role': strategic_profile.role,
                'strategic_goals': strategic_profile.strategic_goals or []
            }
        
        # Delivery preferences
        delivery_config = {
            'frequency': delivery_prefs.email_frequency,
            'content_types': delivery_prefs.content_types or [],
            'digest_format': delivery_prefs.digest_format,
            'max_items': delivery_prefs.max_items_per_digest or 20,
            'timezone': str(delivery_prefs.timezone) if delivery_prefs.timezone else 'UTC'
        }
        
        return UserDiscoveryContext(
            user_id=user.id,
            email=user.email,
            focus_areas=focus_areas,
            tracked_entities=tracked_entities,
            keywords=keywords,
            delivery_preferences=delivery_config,
            strategic_context=strategic_context,
            priority_score=delivery_prefs.priority_score or 1.0,
            last_processed=delivery_prefs.last_delivery
        )
    
    async def _process_users_in_batches(self, users: List[UserDiscoveryContext], metrics: PipelineMetrics):
        """Process users in optimized batches with concurrency control."""
        
        total_users = len(users)
        processed_count = 0
        
        # Process in batches
        for batch_start in range(0, total_users, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_users)
            batch_users = users[batch_start:batch_end]
            
            self.logger.info(f"Processing batch {batch_start//self.batch_size + 1}: users {batch_start+1}-{batch_end} of {total_users}")
            
            # Process batch with concurrency control
            batch_tasks = []
            for user_context in batch_users:
                task = self._process_single_user_with_semaphore(user_context, metrics)
                batch_tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for i, result in enumerate(batch_results):
                user_context = batch_users[i]
                
                if isinstance(result, Exception):
                    self.logger.error(f"User {user_context.user_id} processing failed: {result}")
                    metrics.users_failed += 1
                    metrics.processing_errors.append(f"User {user_context.user_id}: {str(result)}")
                else:
                    metrics.users_successful += 1
                    if isinstance(result, dict) and 'content_count' in result:
                        metrics.total_content_discovered += result['content_count']
                
                metrics.users_processed += 1
                processed_count += 1
            
            # Update job progress
            if self.current_job_id:
                await self._update_job_progress(processed_count, total_users)
            
            # Brief pause between batches to prevent resource exhaustion
            if batch_end < total_users:
                await asyncio.sleep(1)
        
        self.logger.info(f"Batch processing completed: {metrics.users_successful}/{metrics.users_processed} successful")
    
    async def _process_single_user_with_semaphore(self, user_context: UserDiscoveryContext, 
                                                metrics: PipelineMetrics) -> Dict[str, Any]:
        """Process single user with semaphore-controlled concurrency."""
        async with self.user_semaphore:
            return await self._process_single_user(user_context, metrics)
    
    async def _process_single_user(self, user_context: UserDiscoveryContext, 
                                 metrics: PipelineMetrics) -> Dict[str, Any]:
        """Process discovery for a single user."""
        user_id = user_context.user_id
        start_time = datetime.now(timezone.utc)
        
        try:
            self.active_user_tasks.add(user_id)
            
            # Check cache first
            cache_key = f"user_discovery:{user_id}:{datetime.now().strftime('%Y-%m-%d')}"
            cached_result = await self.user_context_cache.get(cache_key)
            if cached_result and cached_result.get('processed_today'):
                self.logger.debug(f"User {user_id} already processed today, skipping")
                return {'status': 'cached', 'content_count': 0}
            
            # Create discovery request
            discovery_request = DiscoveryRequest(
                user_id=user_id,
                keywords=user_context.keywords,
                focus_areas=user_context.focus_areas,
                entities=user_context.tracked_entities,
                limit=self.content_limit_per_user,
                quality_threshold=self.quality_threshold,
                enable_ml_scoring=True,
                enable_content_processing=True,
                timeout_seconds=300  # 5 minute timeout per user
            )
            
            # Execute discovery
            async with self.discovery_semaphore:
                orchestrator = DiscoveryOrchestrator(self.config)
                discovery_response = await orchestrator.discover_content(discovery_request)
            
            # Process discovered content
            if discovery_response.items:
                processed_content = await self.content_processor.process_user_content(
                    user_context, discovery_response.items
                )
                
                # Store processed content
                await self._store_discovered_content(user_id, processed_content)
                content_count = len(processed_content)
            else:
                content_count = 0
            
            # Update user's last processing time
            await self._update_user_last_processed(user_id)
            
            # Cache result
            cache_result = {
                'processed_today': True,
                'content_count': content_count,
                'processing_time': (datetime.now(timezone.utc) - start_time).total_seconds()
            }
            await self.user_context_cache.set(cache_key, cache_result, ttl=86400)  # 24 hours
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.debug(f"User {user_id} processed: {content_count} items in {processing_time:.1f}s")
            
            return {
                'status': 'success',
                'user_id': user_id,
                'content_count': content_count,
                'processing_time': processing_time
            }
            
        except asyncio.TimeoutError:
            self.logger.warning(f"User {user_id} processing timed out")
            raise Exception(f"Processing timeout for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process user {user_id}: {e}")
            raise
            
        finally:
            self.active_user_tasks.discard(user_id)
    
    async def _store_discovered_content(self, user_id: int, content_items: List[Dict[str, Any]]):
        """Store discovered content in database."""
        if not content_items:
            return
        
        try:
            async with get_db_session() as session:
                for content_data in content_items:
                    # Create DiscoveredContent record
                    discovered_content = DiscoveredContent(
                        title=content_data.get('title', '')[:500],
                        content_url=content_data.get('url', ''),
                        content_text=content_data.get('content', ''),
                        content_summary=content_data.get('summary', ''),
                        content_hash=content_data.get('content_hash'),
                        similarity_hash=content_data.get('similarity_hash'),
                        author=content_data.get('author', '')[:200] if content_data.get('author') else None,
                        published_at=content_data.get('published_at'),
                        content_type=content_data.get('content_type', 'article'),
                        user_id=user_id,
                        source_id=content_data.get('source_id', 1),  # Default source
                        relevance_score=content_data.get('relevance_score', 0.0),
                        credibility_score=content_data.get('credibility_score', 0.0),
                        freshness_score=content_data.get('freshness_score', 0.0),
                        overall_score=content_data.get('overall_score', 0.0),
                        ml_confidence_level=content_data.get('ml_confidence', 0.5),
                        predicted_categories=json.dumps(content_data.get('categories', [])),
                        detected_entities=json.dumps(content_data.get('entities', [])),
                        competitive_relevance=content_data.get('competitive_relevance', 'unknown')
                    )
                    
                    session.add(discovered_content)
                
                await session.commit()
                self.logger.debug(f"Stored {len(content_items)} content items for user {user_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to store content for user {user_id}: {e}")
            raise
    
    async def _update_user_last_processed(self, user_id: int):
        """Update user's last processing timestamp."""
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(UserDeliveryPreferences).where(UserDeliveryPreferences.user_id == user_id)
                )
                delivery_prefs = result.scalar_one_or_none()
                
                if delivery_prefs:
                    delivery_prefs.last_delivery = datetime.now(timezone.utc)
                    await session.commit()
                    
        except Exception as e:
            self.logger.error(f"Failed to update last processed time for user {user_id}: {e}")
    
    async def _optimize_and_cleanup(self):
        """Perform system optimization and cleanup."""
        try:
            self.logger.info("Starting system optimization and cleanup")
            
            # Clear old cache entries
            await self.user_context_cache.clear_expired()
            await self.source_cache.clear_expired()
            
            # Update source performance metrics
            await self._update_source_metrics()
            
            # Archive old content
            await self._archive_old_content()
            
            self.logger.info("System optimization completed")
            
        except Exception as e:
            self.logger.error(f"System optimization failed: {e}")
    
    async def _update_source_metrics(self):
        """Update source performance metrics based on recent performance."""
        try:
            async with get_db_session() as session:
                # Update source success rates based on recent discovery jobs
                await session.execute("""
                    UPDATE discovered_sources 
                    SET success_rate = (
                        SELECT COALESCE(AVG(
                            CASE WHEN status = 'completed' THEN 1.0 ELSE 0.0 END
                        ), 0.0)
                        FROM discovery_jobs 
                        WHERE created_at >= NOW() - INTERVAL '7 days'
                        AND job_type = 'scheduled_discovery'
                    ),
                    updated_at = NOW()
                    WHERE last_checked >= NOW() - INTERVAL '24 hours'
                """)
                
                await session.commit()
                self.logger.debug("Updated source performance metrics")
                
        except Exception as e:
            self.logger.error(f"Failed to update source metrics: {e}")
    
    async def _archive_old_content(self):
        """Archive content older than retention period."""
        try:
            retention_days = self.config.get('content_retention_days', 90)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            
            async with get_db_session() as session:
                # Mark old content as archived (or delete based on policy)
                result = await session.execute(
                    select(func.count(DiscoveredContent.id))
                    .where(
                        and_(
                            DiscoveredContent.created_at < cutoff_date,
                            DiscoveredContent.is_delivered == True
                        )
                    )
                )
                old_content_count = result.scalar()
                
                if old_content_count > 0:
                    # For now, just log - implement archival strategy based on requirements
                    self.logger.info(f"Found {old_content_count} content items eligible for archival")
                
        except Exception as e:
            self.logger.error(f"Content archival check failed: {e}")
    
    async def _create_pipeline_job(self) -> int:
        """Create job record for pipeline execution tracking."""
        try:
            async with get_db_session() as session:
                job = DiscoveryJob(
                    job_type='scheduled_discovery',
                    job_subtype='daily_pipeline',
                    status='running',
                    started_at=datetime.now(timezone.utc),
                    job_parameters=json.dumps({
                        'batch_size': self.batch_size,
                        'max_concurrent_users': self.max_concurrent_users,
                        'content_limit_per_user': self.content_limit_per_user,
                        'quality_threshold': self.quality_threshold
                    }),
                    created_by='system'
                )
                
                session.add(job)
                await session.commit()
                await session.refresh(job)
                
                self.logger.info(f"Created pipeline job {job.id}")
                return job.id
                
        except Exception as e:
            self.logger.error(f"Failed to create pipeline job: {e}")
            raise
    
    async def _update_job_progress(self, processed: int, total: int):
        """Update job progress."""
        if not self.current_job_id:
            return
        
        try:
            progress = int((processed / total) * 100) if total > 0 else 0
            
            async with get_db_session() as session:
                result = await session.execute(
                    select(DiscoveryJob).where(DiscoveryJob.id == self.current_job_id)
                )
                job = result.scalar_one_or_none()
                
                if job:
                    job.progress_percentage = progress
                    job.updated_at = datetime.now(timezone.utc)
                    await session.commit()
                    
        except Exception as e:
            self.logger.error(f"Failed to update job progress: {e}")
    
    async def _finalize_pipeline_execution(self, job_id: int, metrics: PipelineMetrics):
        """Finalize pipeline execution and update job record."""
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(DiscoveryJob).where(DiscoveryJob.id == job_id)
                )
                job = result.scalar_one_or_none()
                
                if job:
                    job.status = 'completed'
                    job.completed_at = datetime.now(timezone.utc)
                    job.progress_percentage = 100
                    job.sources_successful = metrics.users_successful
                    job.sources_failed = metrics.users_failed
                    job.content_found = metrics.total_content_discovered
                    job.content_delivered = metrics.total_content_delivered
                    job.processing_time_seconds = int(metrics.duration_seconds)
                    job.avg_engagement_prediction = metrics.success_rate
                    
                    await session.commit()
                    
                self.logger.info(f"Finalized pipeline job {job_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to finalize pipeline job: {e}")
    
    async def _mark_job_failed(self, job_id: int, error_message: str):
        """Mark job as failed with error details."""
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(DiscoveryJob).where(DiscoveryJob.id == job_id)
                )
                job = result.scalar_one_or_none()
                
                if job:
                    job.status = 'failed'
                    job.completed_at = datetime.now(timezone.utc)
                    job.error_message = error_message[:2000]  # Truncate if too long
                    
                    await session.commit()
                    
        except Exception as e:
            self.logger.error(f"Failed to mark job as failed: {e}")
    
    async def _cleanup_pipeline_resources(self):
        """Clean up pipeline resources."""
        try:
            # Clear active tasks
            self.active_user_tasks.clear()
            
            # Close session manager connections
            await self.session_manager.close()
            
            self.logger.debug("Pipeline resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Pipeline cleanup failed: {e}")
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline execution status."""
        return {
            'is_running': bool(self.current_job_id),
            'job_id': self.current_job_id,
            'active_user_count': len(self.active_user_tasks),
            'active_users': list(self.active_user_tasks),
            'pipeline_start_time': self.pipeline_start_time.isoformat(),
            'config': {
                'batch_size': self.batch_size,
                'max_concurrent_users': self.max_concurrent_users,
                'content_limit_per_user': self.content_limit_per_user,
                'quality_threshold': self.quality_threshold
            }
        }
    
    async def stop_pipeline(self) -> bool:
        """
        Gracefully stop the running pipeline.
        
        Returns:
            bool: True if pipeline was stopped successfully
        """
        if not self.current_job_id:
            return False
        
        try:
            self.logger.info("Stopping daily discovery pipeline...")
            
            # Mark job as cancelled
            await self._mark_job_cancelled()
            
            # Clear state
            self.current_job_id = None
            self.active_user_tasks.clear()
            
            self.logger.info("Pipeline stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop pipeline: {e}")
            return False
    
    async def _mark_job_cancelled(self):
        """Mark current job as cancelled."""
        if not self.current_job_id:
            return
        
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(DiscoveryJob).where(DiscoveryJob.id == self.current_job_id)
                )
                job = result.scalar_one_or_none()
                
                if job:
                    job.status = 'cancelled'
                    job.completed_at = datetime.now(timezone.utc)
                    await session.commit()
                    
        except Exception as e:
            self.logger.error(f"Failed to mark job as cancelled: {e}")
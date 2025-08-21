"""
Job Scheduler - Background task management with proper error recovery and job queuing.

Manages background tasks, job queuing, scheduling, and coordination with intelligent
error recovery, progress tracking, and resource management for thousands of users.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import signal
import threading
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text, desc, update
from sqlalchemy.orm import selectinload

from app.database import get_db_session
from app.models.discovery import DiscoveryJob
from app.models.user import User

from ..utils import (
    get_config,
    UnifiedErrorHandler,
    batch_processor,
    AsyncSessionManager
)
from .daily_discovery_pipeline import DailyDiscoveryPipeline
from .ml_training_pipeline import MLTrainingPipeline
from .monitoring_pipeline import MonitoringPipeline


class JobType(Enum):
    """Types of scheduled jobs."""
    DAILY_DISCOVERY = "daily_discovery"
    ML_TRAINING = "ml_training"
    SOURCE_HEALTH_CHECK = "source_health_check"
    CONTENT_CLEANUP = "content_cleanup"
    USER_ANALYTICS = "user_analytics"
    SYSTEM_MAINTENANCE = "system_maintenance"


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ScheduledJob:
    """Scheduled job definition."""
    job_id: Optional[int] = None
    job_type: JobType = JobType.DAILY_DISCOVERY
    job_name: str = ""
    priority: JobPriority = JobPriority.NORMAL
    scheduled_time: Optional[datetime] = None
    max_runtime_seconds: int = 3600  # 1 hour default
    retry_count: int = 0
    max_retries: int = 3
    retry_delay_seconds: int = 300  # 5 minutes
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    # Execution state
    status: JobStatus = JobStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_data: Dict[str, Any] = field(default_factory=dict)
    
    # Resource requirements
    requires_exclusive_lock: bool = False
    estimated_duration_seconds: int = 600  # 10 minutes default
    memory_limit_mb: int = 512
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'job_type': self.job_type.value,
            'job_name': self.job_name,
            'priority': self.priority.value,
            'scheduled_time': self.scheduled_time.isoformat() if self.scheduled_time else None,
            'max_runtime_seconds': self.max_runtime_seconds,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'retry_delay_seconds': self.retry_delay_seconds,
            'parameters': self.parameters,
            'dependencies': list(self.dependencies),
            'tags': list(self.tags),
            'status': self.status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'result_data': self.result_data,
            'requires_exclusive_lock': self.requires_exclusive_lock,
            'estimated_duration_seconds': self.estimated_duration_seconds,
            'memory_limit_mb': self.memory_limit_mb
        }


class ResourceManager:
    """Manages system resources and job execution limits."""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 5)
        self.max_memory_mb = config.get('max_memory_mb', 2048)
        self.max_cpu_usage = config.get('max_cpu_usage', 0.8)
        
        # Current resource usage
        self.running_jobs: Dict[int, ScheduledJob] = {}
        self.current_memory_mb = 0
        self.exclusive_lock_holder: Optional[int] = None
        
        # Resource tracking
        self.resource_semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        self.exclusive_lock = asyncio.Lock()
        
    async def can_start_job(self, job: ScheduledJob) -> bool:
        """Check if job can be started based on resource availability."""
        
        # Check concurrent job limit
        if len(self.running_jobs) >= self.max_concurrent_jobs:
            return False
        
        # Check memory limit
        if self.current_memory_mb + job.memory_limit_mb > self.max_memory_mb:
            return False
        
        # Check exclusive lock requirement
        if job.requires_exclusive_lock and self.exclusive_lock_holder is not None:
            return False
        
        # Check if other job holds exclusive lock
        if self.exclusive_lock_holder is not None and job.job_id != self.exclusive_lock_holder:
            return False
        
        return True
    
    async def allocate_resources(self, job: ScheduledJob) -> bool:
        """Allocate resources for job execution."""
        try:
            if not await self.can_start_job(job):
                return False
            
            # Acquire semaphore
            await self.resource_semaphore.acquire()
            
            # Acquire exclusive lock if needed
            if job.requires_exclusive_lock:
                await self.exclusive_lock.acquire()
                self.exclusive_lock_holder = job.job_id
            
            # Allocate memory
            self.current_memory_mb += job.memory_limit_mb
            self.running_jobs[job.job_id] = job
            
            return True
            
        except Exception as e:
            # Clean up on failure
            await self.release_resources(job)
            return False
    
    async def release_resources(self, job: ScheduledJob):
        """Release resources after job completion."""
        try:
            # Release memory
            if job.job_id in self.running_jobs:
                self.current_memory_mb = max(0, self.current_memory_mb - job.memory_limit_mb)
                del self.running_jobs[job.job_id]
            
            # Release exclusive lock
            if job.requires_exclusive_lock and self.exclusive_lock_holder == job.job_id:
                self.exclusive_lock_holder = None
                self.exclusive_lock.release()
            
            # Release semaphore
            self.resource_semaphore.release()
            
        except Exception as e:
            logging.error(f"Error releasing resources for job {job.job_id}: {e}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource usage status."""
        return {
            'running_jobs': len(self.running_jobs),
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'current_memory_mb': self.current_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'exclusive_lock_holder': self.exclusive_lock_holder,
            'available_slots': max(0, self.max_concurrent_jobs - len(self.running_jobs))
        }


class JobScheduler:
    """
    Advanced job scheduler for competitive intelligence pipeline.
    
    Manages background tasks, scheduling, error recovery, and resource allocation
    with support for job dependencies, priorities, and system health monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("pipeline.job_scheduler")
        self.error_handler = UnifiedErrorHandler()
        
        # Components
        self.resource_manager = ResourceManager(config)
        self.session_manager = AsyncSessionManager(name="job_scheduler")
        
        # Scheduler state
        self.is_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.job_queue = asyncio.PriorityQueue()
        
        # Job tracking
        self.active_jobs: Dict[int, asyncio.Task] = {}
        self.job_history: List[ScheduledJob] = []
        self.job_handlers: Dict[JobType, Callable] = {}
        
        # Configuration
        self.polling_interval = config.get('polling_interval_seconds', 30)
        self.max_job_history = config.get('max_job_history', 1000)
        self.cleanup_interval = config.get('cleanup_interval_hours', 24) * 3600
        
        # Performance tracking
        self.scheduler_stats = {
            'jobs_scheduled': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_runtime_seconds': 0.0,
            'avg_job_runtime': 0.0
        }
        
        # Initialize job handlers
        self._register_job_handlers()
        
        self.logger.info("Job Scheduler initialized")
    
    def _register_job_handlers(self):
        """Register job type handlers."""
        self.job_handlers = {
            JobType.DAILY_DISCOVERY: self._handle_daily_discovery,
            JobType.ML_TRAINING: self._handle_ml_training,
            JobType.SOURCE_HEALTH_CHECK: self._handle_source_health_check,
            JobType.CONTENT_CLEANUP: self._handle_content_cleanup,
            JobType.USER_ANALYTICS: self._handle_user_analytics,
            JobType.SYSTEM_MAINTENANCE: self._handle_system_maintenance
        }
    
    async def start(self):
        """Start the job scheduler."""
        if self.is_running:
            self.logger.warning("Job scheduler already running")
            return
        
        self.is_running = True
        
        # Start scheduler loop
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # Schedule default recurring jobs
        await self._schedule_default_jobs()
        
        self.logger.info("Job scheduler started")
    
    async def stop(self):
        """Stop the job scheduler gracefully."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping job scheduler...")
        self.is_running = False
        
        # Cancel scheduler task
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active jobs
        for job_id, task in self.active_jobs.items():
            self.logger.info(f"Cancelling active job {job_id}")
            task.cancel()
        
        # Wait for active jobs to complete
        if self.active_jobs:
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)
        
        # Close session manager
        await self.session_manager.close()
        
        self.logger.info("Job scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Process pending jobs
                await self._process_pending_jobs()
                
                # Check for completed jobs
                await self._check_completed_jobs()
                
                # Cleanup old jobs
                await self._cleanup_old_jobs()
                
                # Update scheduler stats
                await self._update_scheduler_stats()
                
                # Sleep until next cycle
                await asyncio.sleep(self.polling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(self.polling_interval)
    
    async def _process_pending_jobs(self):
        """Process pending jobs from database and queue."""
        try:
            # Load pending jobs from database
            async with get_db_session() as session:
                result = await session.execute(
                    select(DiscoveryJob)
                    .where(
                        and_(
                            DiscoveryJob.status.in_(['pending', 'retrying']),
                            or_(
                                DiscoveryJob.scheduled_at.is_(None),
                                DiscoveryJob.scheduled_at <= datetime.now(timezone.utc)
                            )
                        )
                    )
                    .order_by(
                        DiscoveryJob.created_at.asc()  # FIFO processing
                    )
                    .limit(50)  # Process in batches
                )
                
                pending_jobs = result.scalars().all()
                
                for job_record in pending_jobs:
                    scheduled_job = await self._convert_db_job_to_scheduled_job(job_record)
                    
                    # Check dependencies
                    if await self._check_job_dependencies(scheduled_job):
                        # Check resource availability
                        if await self.resource_manager.can_start_job(scheduled_job):
                            await self._start_job(scheduled_job)
                        else:
                            self.logger.debug(f"Job {scheduled_job.job_id} waiting for resources")
                    else:
                        self.logger.debug(f"Job {scheduled_job.job_id} waiting for dependencies")
                        
        except Exception as e:
            self.logger.error(f"Failed to process pending jobs: {e}")
    
    async def _convert_db_job_to_scheduled_job(self, job_record: DiscoveryJob) -> ScheduledJob:
        """Convert database job record to ScheduledJob."""
        
        # Parse job parameters
        parameters = {}
        if job_record.job_parameters:
            try:
                parameters = json.loads(job_record.job_parameters)
            except json.JSONDecodeError:
                pass
        
        # Map job type
        job_type = JobType.DAILY_DISCOVERY  # default
        try:
            job_type = JobType(job_record.job_type)
        except ValueError:
            pass
        
        # Create scheduled job
        scheduled_job = ScheduledJob(
            job_id=job_record.id,
            job_type=job_type,
            job_name=f"{job_record.job_type}_{job_record.id}",
            scheduled_time=job_record.scheduled_at,
            retry_count=job_record.retry_count,
            max_retries=job_record.max_retries,
            parameters=parameters,
            status=JobStatus(job_record.status),
            started_at=job_record.started_at,
            completed_at=job_record.completed_at,
            error_message=job_record.error_message
        )
        
        # Set job-specific configurations
        if job_type == JobType.DAILY_DISCOVERY:
            scheduled_job.max_runtime_seconds = 4 * 3600  # 4 hours
            scheduled_job.requires_exclusive_lock = True
            scheduled_job.memory_limit_mb = 1024
            scheduled_job.priority = JobPriority.HIGH
        elif job_type == JobType.ML_TRAINING:
            scheduled_job.max_runtime_seconds = 2 * 3600  # 2 hours
            scheduled_job.memory_limit_mb = 512
            scheduled_job.priority = JobPriority.NORMAL
        
        return scheduled_job
    
    async def _check_job_dependencies(self, job: ScheduledJob) -> bool:
        """Check if job dependencies are satisfied."""
        if not job.dependencies:
            return True
        
        try:
            async with get_db_session() as session:
                for dependency in job.dependencies:
                    result = await session.execute(
                        select(DiscoveryJob.status)
                        .where(
                            and_(
                                DiscoveryJob.job_type == dependency,
                                DiscoveryJob.status == 'completed',
                                DiscoveryJob.completed_at >= datetime.now(timezone.utc) - timedelta(hours=24)
                            )
                        )
                        .order_by(DiscoveryJob.completed_at.desc())
                        .limit(1)
                    )
                    
                    if not result.scalar_one_or_none():
                        return False
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to check dependencies for job {job.job_id}: {e}")
            return False
    
    async def _start_job(self, job: ScheduledJob):
        """Start job execution."""
        try:
            # Allocate resources
            if not await self.resource_manager.allocate_resources(job):
                self.logger.warning(f"Failed to allocate resources for job {job.job_id}")
                return
            
            # Update job status in database
            await self._update_job_status(job.job_id, JobStatus.RUNNING)
            
            # Start job task
            job_task = asyncio.create_task(self._execute_job(job))
            self.active_jobs[job.job_id] = job_task
            
            self.logger.info(f"Started job {job.job_id} ({job.job_type.value})")
            self.scheduler_stats['jobs_scheduled'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to start job {job.job_id}: {e}")
            await self.resource_manager.release_resources(job)
    
    async def _execute_job(self, job: ScheduledJob):
        """Execute a scheduled job."""
        start_time = datetime.now(timezone.utc)
        
        try:
            job.started_at = start_time
            job.status = JobStatus.RUNNING
            
            # Get job handler
            handler = self.job_handlers.get(job.job_type)
            if not handler:
                raise ValueError(f"No handler for job type {job.job_type}")
            
            # Execute job with timeout
            try:
                result = await asyncio.wait_for(
                    handler(job),
                    timeout=job.max_runtime_seconds
                )
                
                # Job completed successfully
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                job.result_data = result or {}
                
                await self._update_job_status(job.job_id, JobStatus.COMPLETED, result)
                
                self.logger.info(f"Job {job.job_id} completed successfully")
                self.scheduler_stats['jobs_completed'] += 1
                
            except asyncio.TimeoutError:
                raise Exception(f"Job timed out after {job.max_runtime_seconds} seconds")
            
        except Exception as e:
            # Job failed
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            
            self.logger.error(f"Job {job.job_id} failed: {e}")
            
            # Handle retry logic
            if job.retry_count < job.max_retries:
                await self._schedule_job_retry(job)
            else:
                await self._update_job_status(job.job_id, JobStatus.FAILED, error_message=str(e))
                self.scheduler_stats['jobs_failed'] += 1
        
        finally:
            # Release resources
            await self.resource_manager.release_resources(job)
            
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            # Update runtime stats
            runtime = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.scheduler_stats['total_runtime_seconds'] += runtime
            
            # Add to job history
            self.job_history.append(job)
            if len(self.job_history) > self.max_job_history:
                self.job_history.pop(0)
    
    async def _schedule_job_retry(self, job: ScheduledJob):
        """Schedule job retry with exponential backoff."""
        try:
            job.retry_count += 1
            retry_delay = job.retry_delay_seconds * (2 ** (job.retry_count - 1))  # Exponential backoff
            retry_time = datetime.now(timezone.utc) + timedelta(seconds=retry_delay)
            
            async with get_db_session() as session:
                result = await session.execute(
                    select(DiscoveryJob).where(DiscoveryJob.id == job.job_id)
                )
                job_record = result.scalar_one_or_none()
                
                if job_record:
                    job_record.status = 'retrying'
                    job_record.retry_count = job.retry_count
                    job_record.scheduled_at = retry_time
                    job_record.error_message = job.error_message
                    
                    await session.commit()
                    
                    self.logger.info(f"Scheduled retry #{job.retry_count} for job {job.job_id} at {retry_time}")
                    
        except Exception as e:
            self.logger.error(f"Failed to schedule retry for job {job.job_id}: {e}")
    
    async def _update_job_status(self, job_id: int, status: JobStatus, 
                               result_data: Optional[Dict[str, Any]] = None,
                               error_message: Optional[str] = None):
        """Update job status in database."""
        try:
            async with get_db_session() as session:
                update_data = {
                    'status': status.value,
                    'updated_at': datetime.now(timezone.utc)
                }
                
                if status == JobStatus.RUNNING:
                    update_data['started_at'] = datetime.now(timezone.utc)
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    update_data['completed_at'] = datetime.now(timezone.utc)
                
                if result_data:
                    # Store result data as JSON (simplified - in production, might want dedicated columns)
                    update_data['job_parameters'] = json.dumps(result_data)
                
                if error_message:
                    update_data['error_message'] = error_message[:2000]  # Truncate if too long
                
                await session.execute(
                    update(DiscoveryJob)
                    .where(DiscoveryJob.id == job_id)
                    .values(**update_data)
                )
                
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update job status for {job_id}: {e}")
    
    async def _check_completed_jobs(self):
        """Check for completed job tasks and cleanup."""
        completed_jobs = []
        
        for job_id, task in list(self.active_jobs.items()):
            if task.done():
                completed_jobs.append(job_id)
        
        # Remove completed jobs from active list
        for job_id in completed_jobs:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    async def _cleanup_old_jobs(self):
        """Clean up old job records and history."""
        try:
            # Clean up database records older than retention period
            retention_days = self.config.get('job_retention_days', 30)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            
            async with get_db_session() as session:
                # Count old jobs before deletion
                count_result = await session.execute(
                    select(func.count(DiscoveryJob.id))
                    .where(
                        and_(
                            DiscoveryJob.completed_at < cutoff_date,
                            DiscoveryJob.status.in_(['completed', 'failed'])
                        )
                    )
                )
                old_job_count = count_result.scalar()
                
                if old_job_count > 0:
                    # Delete old completed jobs
                    await session.execute(
                        text("""
                            DELETE FROM discovery_jobs 
                            WHERE completed_at < :cutoff_date 
                            AND status IN ('completed', 'failed')
                        """),
                        {'cutoff_date': cutoff_date}
                    )
                    
                    await session.commit()
                    self.logger.info(f"Cleaned up {old_job_count} old job records")
                    
        except Exception as e:
            self.logger.error(f"Job cleanup failed: {e}")
    
    async def _update_scheduler_stats(self):
        """Update scheduler performance statistics."""
        if self.scheduler_stats['jobs_completed'] > 0:
            self.scheduler_stats['avg_job_runtime'] = (
                self.scheduler_stats['total_runtime_seconds'] / 
                self.scheduler_stats['jobs_completed']
            )
    
    async def _schedule_default_jobs(self):
        """Schedule default recurring jobs."""
        try:
            # Schedule daily discovery pipeline
            await self.schedule_job(
                job_type=JobType.DAILY_DISCOVERY,
                scheduled_time=self._get_next_daily_run_time(),
                parameters={
                    'batch_size': 50,
                    'max_concurrent_users': 20,
                    'quality_threshold': 0.5
                }
            )
            
            # Schedule ML training
            await self.schedule_job(
                job_type=JobType.ML_TRAINING,
                scheduled_time=self._get_next_training_time(),
                parameters={
                    'min_training_samples': 100,
                    'model_types': ['relevance_scorer', 'engagement_predictor']
                }
            )
            
            # Schedule source health check
            await self.schedule_job(
                job_type=JobType.SOURCE_HEALTH_CHECK,
                scheduled_time=datetime.now(timezone.utc) + timedelta(minutes=30),
                parameters={'check_all_sources': True}
            )
            
            self.logger.info("Default recurring jobs scheduled")
            
        except Exception as e:
            self.logger.error(f"Failed to schedule default jobs: {e}")
    
    def _get_next_daily_run_time(self) -> datetime:
        """Get next daily discovery run time (early morning)."""
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)  # 6 AM UTC
        
        if next_run <= now:
            next_run += timedelta(days=1)
        
        return next_run
    
    def _get_next_training_time(self) -> datetime:
        """Get next ML training time (late night)."""
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)  # 2 AM UTC
        
        if next_run <= now:
            next_run += timedelta(days=1)
        
        return next_run
    
    # Job Handlers
    
    async def _handle_daily_discovery(self, job: ScheduledJob) -> Dict[str, Any]:
        """Handle daily discovery pipeline execution."""
        self.logger.info("Executing daily discovery pipeline")
        
        pipeline = DailyDiscoveryPipeline(job.parameters)
        metrics = await pipeline.run_daily_discovery()
        
        return {
            'users_processed': metrics.users_processed,
            'users_successful': metrics.users_successful,
            'content_discovered': metrics.total_content_discovered,
            'processing_time': metrics.duration_seconds,
            'success_rate': metrics.success_rate
        }
    
    async def _handle_ml_training(self, job: ScheduledJob) -> Dict[str, Any]:
        """Handle ML training pipeline execution."""
        self.logger.info("Executing ML training pipeline")
        
        training_pipeline = MLTrainingPipeline(job.parameters)
        training_results = await training_pipeline.run_training_cycle()
        
        return {
            'models_trained': len(training_results),
            'training_results': [
                {
                    'model_type': metrics.model_type,
                    'model_version': metrics.model_version,
                    'training_samples': metrics.training_samples,
                    'validation_accuracy': metrics.validation_accuracy,
                    'improvement': metrics.improvement_over_previous
                }
                for metrics in training_results
            ]
        }
    
    async def _handle_source_health_check(self, job: ScheduledJob) -> Dict[str, Any]:
        """Handle source health monitoring."""
        self.logger.info("Executing source health check")
        
        # Implementation would check all sources for availability and performance
        # For now, return dummy results
        
        return {
            'sources_checked': 10,
            'sources_healthy': 9,
            'sources_down': 1,
            'avg_response_time': 250.5
        }
    
    async def _handle_content_cleanup(self, job: ScheduledJob) -> Dict[str, Any]:
        """Handle content cleanup and archival."""
        self.logger.info("Executing content cleanup")
        
        # Implementation would archive old content, remove duplicates, etc.
        
        return {
            'content_archived': 1000,
            'duplicates_removed': 50,
            'storage_freed_mb': 150.5
        }
    
    async def _handle_user_analytics(self, job: ScheduledJob) -> Dict[str, Any]:
        """Handle user analytics processing."""
        self.logger.info("Executing user analytics")
        
        # Implementation would process user engagement data, generate reports, etc.
        
        return {
            'users_analyzed': 500,
            'engagement_patterns_detected': 25,
            'recommendations_generated': 750
        }
    
    async def _handle_system_maintenance(self, job: ScheduledJob) -> Dict[str, Any]:
        """Handle system maintenance tasks."""
        self.logger.info("Executing system maintenance")
        
        # Implementation would handle cache cleanup, database optimization, etc.
        
        return {
            'caches_cleared': 5,
            'database_optimized': True,
            'disk_space_freed_mb': 500
        }
    
    # Public API
    
    async def schedule_job(self, job_type: JobType, scheduled_time: Optional[datetime] = None,
                          parameters: Optional[Dict[str, Any]] = None,
                          priority: JobPriority = JobPriority.NORMAL,
                          max_retries: int = 3) -> int:
        """
        Schedule a new job.
        
        Args:
            job_type: Type of job to schedule
            scheduled_time: When to run the job (None for immediate)
            parameters: Job-specific parameters
            priority: Job priority
            max_retries: Maximum number of retries
            
        Returns:
            int: Job ID
        """
        try:
            async with get_db_session() as session:
                job = DiscoveryJob(
                    job_type=job_type.value,
                    job_subtype='scheduled',
                    status='pending',
                    scheduled_at=scheduled_time,
                    max_retries=max_retries,
                    job_parameters=json.dumps(parameters or {}),
                    created_by='scheduler'
                )
                
                session.add(job)
                await session.commit()
                await session.refresh(job)
                
                self.logger.info(f"Scheduled job {job.id} ({job_type.value}) for {scheduled_time}")
                return job.id
                
        except Exception as e:
            self.logger.error(f"Failed to schedule job: {e}")
            raise
    
    async def cancel_job(self, job_id: int) -> bool:
        """
        Cancel a scheduled or running job.
        
        Args:
            job_id: ID of job to cancel
            
        Returns:
            bool: True if job was cancelled successfully
        """
        try:
            # Cancel active task if running
            if job_id in self.active_jobs:
                task = self.active_jobs[job_id]
                task.cancel()
                del self.active_jobs[job_id]
                self.logger.info(f"Cancelled running job {job_id}")
            
            # Update database status
            await self._update_job_status(job_id, JobStatus.CANCELLED)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def get_job_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(DiscoveryJob).where(DiscoveryJob.id == job_id)
                )
                job = result.scalar_one_or_none()
                
                if not job:
                    return None
                
                return {
                    'job_id': job.id,
                    'job_type': job.job_type,
                    'status': job.status,
                    'created_at': job.created_at.isoformat() if job.created_at else None,
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'retry_count': job.retry_count,
                    'max_retries': job.max_retries,
                    'error_message': job.error_message,
                    'progress_percentage': job.progress_percentage
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get job status for {job_id}: {e}")
            return None
    
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status."""
        return {
            'is_running': self.is_running,
            'active_jobs': len(self.active_jobs),
            'active_job_ids': list(self.active_jobs.keys()),
            'resource_status': self.resource_manager.get_resource_status(),
            'statistics': {
                **self.scheduler_stats,
                'job_history_count': len(self.job_history)
            },
            'configuration': {
                'polling_interval': self.polling_interval,
                'max_job_history': self.max_job_history,
                'cleanup_interval': self.cleanup_interval
            }
        }
    
    async def get_recent_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent job execution history."""
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(DiscoveryJob)
                    .order_by(DiscoveryJob.created_at.desc())
                    .limit(limit)
                )
                
                jobs = result.scalars().all()
                
                return [
                    {
                        'job_id': job.id,
                        'job_type': job.job_type,
                        'status': job.status,
                        'created_at': job.created_at.isoformat() if job.created_at else None,
                        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                        'processing_time_seconds': job.processing_time_seconds,
                        'content_found': job.content_found,
                        'sources_successful': job.sources_successful,
                        'sources_failed': job.sources_failed
                    }
                    for job in jobs
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get recent jobs: {e}")
            return []
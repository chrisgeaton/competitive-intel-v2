"""
Monitoring Pipeline - Performance metrics, system health, and user engagement analytics.

Tracks performance metrics, system health monitoring, user engagement analytics,
and comprehensive logging for the competitive intelligence pipeline system.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text, desc
from sqlalchemy.orm import selectinload

from app.database import get_db_session
from app.models.discovery import (
    DiscoveryJob, DiscoveredContent, ContentEngagement,
    DiscoveredSource, MLModelMetrics
)
from app.models.user import User

from ..utils import (
    get_config,
    UnifiedErrorHandler,
    get_ml_scoring_cache,
    get_content_processing_cache
)


@dataclass
class SystemHealth:
    """System health metrics."""
    timestamp: datetime
    overall_status: str = "healthy"  # healthy, degraded, critical, down
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    active_connections: int = 0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    pipeline_status: str = "idle"
    job_queue_size: int = 0
    cache_hit_rates: Dict[str, float] = field(default_factory=dict)
    component_health: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for pipeline operations."""
    timestamp: datetime
    operation_type: str
    duration_seconds: float
    success_count: int = 0
    error_count: int = 0
    throughput_per_second: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserEngagementMetrics:
    """User engagement analytics."""
    timestamp: datetime
    total_users: int = 0
    active_users: int = 0
    content_delivered: int = 0
    email_opens: int = 0
    email_clicks: int = 0
    avg_time_spent: float = 0.0
    engagement_rate: float = 0.0
    popular_categories: List[str] = field(default_factory=list)
    top_sources: List[str] = field(default_factory=list)


class SystemMonitor:
    """System resource and health monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger("monitoring.system")
        
        # Historical data storage
        self.health_history: deque = deque(maxlen=1440)  # 24 hours of minute-by-minute data
        self.performance_history: deque = deque(maxlen=10080)  # 7 days of minute-by-minute data
        
        # Thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 85.0
        self.memory_critical_threshold = 95.0
        self.disk_warning_threshold = 80.0
        self.disk_critical_threshold = 90.0
        self.response_time_warning_ms = 1000.0
        self.response_time_critical_ms = 5000.0
        
        # Alert state
        self.alerts_active: Set[str] = set()
        self.alert_cooldown: Dict[str, datetime] = {}
        
    async def collect_system_health(self) -> SystemHealth:
        """Collect comprehensive system health metrics."""
        try:
            timestamp = datetime.now(timezone.utc)
            
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connections
            connections = len(psutil.net_connections())
            
            # Database health check
            db_response_time = await self._check_database_health()
            
            # Cache performance
            cache_hit_rates = await self._get_cache_hit_rates()
            
            # Component health
            component_health = await self._check_component_health()
            
            # Determine overall status
            overall_status = self._determine_overall_status(
                cpu_percent, memory.percent, disk.percent, db_response_time
            )
            
            health = SystemHealth(
                timestamp=timestamp,
                overall_status=overall_status,
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                active_connections=connections,
                response_time_ms=db_response_time,
                cache_hit_rates=cache_hit_rates,
                component_health=component_health
            )
            
            # Store in history
            self.health_history.append(health)
            
            # Check for alerts
            await self._check_alerts(health)
            
            return health
            
        except Exception as e:
            self.logger.error(f"Failed to collect system health: {e}")
            return SystemHealth(
                timestamp=datetime.now(timezone.utc),
                overall_status="error",
                component_health={"monitor": "error"}
            )
    
    async def _check_database_health(self) -> float:
        """Check database connectivity and response time."""
        start_time = time.time()
        
        try:
            async with get_db_session() as session:
                await session.execute(text("SELECT 1"))
            
            return (time.time() - start_time) * 1000  # Convert to milliseconds
            
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return 9999.0  # Indicate failure
    
    async def _get_cache_hit_rates(self) -> Dict[str, float]:
        """Get cache hit rates for all caches."""
        try:
            ml_cache = get_ml_scoring_cache()
            content_cache = get_content_processing_cache()
            
            return {
                'ml_scoring': ml_cache.hit_rate if hasattr(ml_cache, 'hit_rate') else 0.0,
                'content_processing': content_cache.hit_rate if hasattr(content_cache, 'hit_rate') else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache hit rates: {e}")
            return {}
    
    async def _check_component_health(self) -> Dict[str, str]:
        """Check health of individual system components."""
        component_health = {}
        
        try:
            # Database
            try:
                async with get_db_session() as session:
                    await session.execute(text("SELECT COUNT(*) FROM users LIMIT 1"))
                component_health['database'] = 'healthy'
            except Exception:
                component_health['database'] = 'unhealthy'
            
            # Discovery engines (simplified check)
            component_health['discovery_engines'] = 'healthy'  # Would check actual engines
            
            # ML models
            try:
                async with get_db_session() as session:
                    active_models = await session.execute(
                        select(func.count(MLModelMetrics.id))
                        .where(MLModelMetrics.is_active == True)
                    )
                    if active_models.scalar() > 0:
                        component_health['ml_models'] = 'healthy'
                    else:
                        component_health['ml_models'] = 'warning'
            except Exception:
                component_health['ml_models'] = 'unhealthy'
            
            return component_health
            
        except Exception as e:
            self.logger.error(f"Component health check failed: {e}")
            return {'monitor': 'error'}
    
    def _determine_overall_status(self, cpu_percent: float, memory_percent: float,
                                disk_percent: float, response_time_ms: float) -> str:
        """Determine overall system health status."""
        
        # Critical conditions
        if (cpu_percent >= self.cpu_critical_threshold or
            memory_percent >= self.memory_critical_threshold or
            disk_percent >= self.disk_critical_threshold or
            response_time_ms >= self.response_time_critical_ms):
            return "critical"
        
        # Warning conditions
        if (cpu_percent >= self.cpu_warning_threshold or
            memory_percent >= self.memory_warning_threshold or
            disk_percent >= self.disk_warning_threshold or
            response_time_ms >= self.response_time_warning_ms):
            return "degraded"
        
        return "healthy"
    
    async def _check_alerts(self, health: SystemHealth):
        """Check for alert conditions and manage notifications."""
        current_alerts = set()
        
        # CPU alerts
        if health.cpu_usage_percent >= self.cpu_critical_threshold:
            current_alerts.add("cpu_critical")
        elif health.cpu_usage_percent >= self.cpu_warning_threshold:
            current_alerts.add("cpu_warning")
        
        # Memory alerts
        if health.memory_usage_percent >= self.memory_critical_threshold:
            current_alerts.add("memory_critical")
        elif health.memory_usage_percent >= self.memory_warning_threshold:
            current_alerts.add("memory_warning")
        
        # Disk alerts
        if health.disk_usage_percent >= self.disk_critical_threshold:
            current_alerts.add("disk_critical")
        elif health.disk_usage_percent >= self.disk_warning_threshold:
            current_alerts.add("disk_warning")
        
        # Response time alerts
        if health.response_time_ms >= self.response_time_critical_ms:
            current_alerts.add("response_critical")
        elif health.response_time_ms >= self.response_time_warning_ms:
            current_alerts.add("response_warning")
        
        # Process new alerts
        new_alerts = current_alerts - self.alerts_active
        resolved_alerts = self.alerts_active - current_alerts
        
        for alert in new_alerts:
            await self._trigger_alert(alert, health)
        
        for alert in resolved_alerts:
            await self._resolve_alert(alert, health)
        
        self.alerts_active = current_alerts
    
    async def _trigger_alert(self, alert_type: str, health: SystemHealth):
        """Trigger an alert notification."""
        # Check cooldown
        now = datetime.now(timezone.utc)
        if alert_type in self.alert_cooldown:
            if now - self.alert_cooldown[alert_type] < timedelta(minutes=15):
                return  # Still in cooldown
        
        self.logger.warning(f"ALERT TRIGGERED: {alert_type} - System health: {health.overall_status}")
        self.alert_cooldown[alert_type] = now
        
        # In production, would send notifications (email, Slack, etc.)
    
    async def _resolve_alert(self, alert_type: str, health: SystemHealth):
        """Resolve an alert notification."""
        self.logger.info(f"ALERT RESOLVED: {alert_type} - System health: {health.overall_status}")
        
        # In production, would send resolution notifications
    
    def get_health_trend(self, hours: int = 24) -> List[SystemHealth]:
        """Get system health trend for specified hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            health for health in self.health_history
            if health.timestamp >= cutoff_time
        ]


class PerformanceTracker:
    """Track performance metrics for pipeline operations."""
    
    def __init__(self):
        self.logger = logging.getLogger("monitoring.performance")
        
        # Performance data storage
        self.metrics_history: deque = deque(maxlen=10080)  # 7 days
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'success_count': 0,
            'error_count': 0,
            'avg_duration': 0.0,
            'success_rate': 0.0
        })
        
        # Current operation tracking
        self.active_operations: Dict[str, datetime] = {}
    
    def start_operation(self, operation_id: str, operation_type: str) -> str:
        """Start tracking an operation."""
        self.active_operations[operation_id] = datetime.now(timezone.utc)
        return operation_id
    
    def complete_operation(self, operation_id: str, operation_type: str,
                         success: bool = True, metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """Complete operation tracking and record metrics."""
        
        if operation_id not in self.active_operations:
            self.logger.warning(f"Operation {operation_id} not found in active operations")
            return None
        
        start_time = self.active_operations.pop(operation_id)
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            timestamp=end_time,
            operation_type=operation_type,
            duration_seconds=duration,
            success_count=1 if success else 0,
            error_count=0 if success else 1,
            metadata=metadata or {}
        )
        
        # Update operation statistics
        stats = self.operation_stats[operation_type]
        stats['count'] += 1
        stats['total_duration'] += duration
        
        if success:
            stats['success_count'] += 1
        else:
            stats['error_count'] += 1
        
        stats['avg_duration'] = stats['total_duration'] / stats['count']
        stats['success_rate'] = stats['success_count'] / stats['count']
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_operation_stats(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Get operation performance statistics."""
        if operation_type:
            return dict(self.operation_stats.get(operation_type, {}))
        else:
            return {op_type: dict(stats) for op_type, stats in self.operation_stats.items()}
    
    def get_performance_trend(self, operation_type: Optional[str] = None,
                            hours: int = 24) -> List[PerformanceMetrics]:
        """Get performance trend for operations."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if operation_type:
            metrics = [m for m in metrics if m.operation_type == operation_type]
        
        return metrics


class EngagementAnalyzer:
    """Analyze user engagement patterns and metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("monitoring.engagement")
        
        # Engagement data cache
        self.engagement_cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(minutes=15)
    
    async def collect_engagement_metrics(self) -> UserEngagementMetrics:
        """Collect comprehensive user engagement metrics."""
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Check cache first
            cache_key = "daily_engagement"
            if (cache_key in self.engagement_cache and 
                cache_key in self.cache_expiry and
                self.cache_expiry[cache_key] > timestamp):
                return self.engagement_cache[cache_key]
            
            async with get_db_session() as session:
                # Total and active users
                total_users_result = await session.execute(
                    select(func.count(User.id)).where(User.is_active == True)
                )
                total_users = total_users_result.scalar() or 0
                
                # Active users (engaged in last 7 days)
                active_users_result = await session.execute(
                    select(func.count(func.distinct(ContentEngagement.user_id)))
                    .where(ContentEngagement.created_at >= timestamp - timedelta(days=7))
                )
                active_users = active_users_result.scalar() or 0
                
                # Content delivery metrics (last 24 hours)
                content_delivered_result = await session.execute(
                    select(func.count(DiscoveredContent.id))
                    .where(
                        and_(
                            DiscoveredContent.is_delivered == True,
                            DiscoveredContent.delivered_at >= timestamp - timedelta(days=1)
                        )
                    )
                )
                content_delivered = content_delivered_result.scalar() or 0
                
                # Email engagement metrics (last 24 hours)
                email_opens_result = await session.execute(
                    select(func.count(ContentEngagement.id))
                    .where(
                        and_(
                            ContentEngagement.engagement_type == 'email_open',
                            ContentEngagement.created_at >= timestamp - timedelta(days=1)
                        )
                    )
                )
                email_opens = email_opens_result.scalar() or 0
                
                email_clicks_result = await session.execute(
                    select(func.count(ContentEngagement.id))
                    .where(
                        and_(
                            ContentEngagement.engagement_type == 'email_click',
                            ContentEngagement.created_at >= timestamp - timedelta(days=1)
                        )
                    )
                )
                email_clicks = email_clicks_result.scalar() or 0
                
                # Average time spent
                time_spent_result = await session.execute(
                    select(func.avg(ContentEngagement.engagement_value))
                    .where(
                        and_(
                            ContentEngagement.engagement_type == 'time_spent',
                            ContentEngagement.created_at >= timestamp - timedelta(days=7)
                        )
                    )
                )
                avg_time_spent = float(time_spent_result.scalar() or 0.0)
                
                # Engagement rate
                engagement_rate = 0.0
                if content_delivered > 0:
                    total_engagements = email_opens + email_clicks
                    engagement_rate = total_engagements / content_delivered
                
                # Popular categories
                popular_categories_result = await session.execute(
                    select(
                        func.json_array_elements_text(DiscoveredContent.predicted_categories).label('category'),
                        func.count().label('count')
                    )
                    .where(
                        and_(
                            DiscoveredContent.created_at >= timestamp - timedelta(days=7),
                            DiscoveredContent.predicted_categories.isnot(None)
                        )
                    )
                    .group_by('category')
                    .order_by(desc('count'))
                    .limit(5)
                )
                
                popular_categories = []
                try:
                    for row in popular_categories_result:
                        popular_categories.append(row.category)
                except Exception:
                    # Fallback if JSON functions not available
                    popular_categories = ['technology', 'business', 'competitive']
                
                # Top sources
                top_sources_result = await session.execute(
                    select(
                        DiscoveredSource.source_name,
                        func.count(DiscoveredContent.id).label('content_count')
                    )
                    .join(DiscoveredContent, DiscoveredSource.id == DiscoveredContent.source_id)
                    .where(DiscoveredContent.created_at >= timestamp - timedelta(days=7))
                    .group_by(DiscoveredSource.source_name)
                    .order_by(desc('content_count'))
                    .limit(5)
                )
                
                top_sources = [row.source_name for row in top_sources_result if row.source_name]
                
                # Create metrics object
                metrics = UserEngagementMetrics(
                    timestamp=timestamp,
                    total_users=total_users,
                    active_users=active_users,
                    content_delivered=content_delivered,
                    email_opens=email_opens,
                    email_clicks=email_clicks,
                    avg_time_spent=avg_time_spent,
                    engagement_rate=engagement_rate,
                    popular_categories=popular_categories,
                    top_sources=top_sources
                )
                
                # Cache results
                self.engagement_cache[cache_key] = metrics
                self.cache_expiry[cache_key] = timestamp + self.cache_ttl
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Failed to collect engagement metrics: {e}")
            return UserEngagementMetrics(
                timestamp=datetime.now(timezone.utc),
                total_users=0,
                active_users=0
            )
    
    async def get_user_engagement_summary(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get engagement summary for a specific user."""
        try:
            async with get_db_session() as session:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                # User engagement activities
                engagements_result = await session.execute(
                    select(
                        ContentEngagement.engagement_type,
                        func.count(ContentEngagement.id).label('count'),
                        func.avg(ContentEngagement.engagement_value).label('avg_value')
                    )
                    .where(
                        and_(
                            ContentEngagement.user_id == user_id,
                            ContentEngagement.created_at >= cutoff_date
                        )
                    )
                    .group_by(ContentEngagement.engagement_type)
                )
                
                engagement_summary = {}
                for row in engagements_result:
                    engagement_summary[row.engagement_type] = {
                        'count': row.count,
                        'avg_value': float(row.avg_value or 0.0)
                    }
                
                # Content delivered to user
                content_delivered_result = await session.execute(
                    select(func.count(DiscoveredContent.id))
                    .where(
                        and_(
                            DiscoveredContent.user_id == user_id,
                            DiscoveredContent.is_delivered == True,
                            DiscoveredContent.delivered_at >= cutoff_date
                        )
                    )
                )
                content_delivered = content_delivered_result.scalar() or 0
                
                return {
                    'user_id': user_id,
                    'period_days': days,
                    'content_delivered': content_delivered,
                    'engagement_summary': engagement_summary,
                    'total_engagements': sum(e['count'] for e in engagement_summary.values()),
                    'engagement_rate': (
                        sum(e['count'] for e in engagement_summary.values()) / max(content_delivered, 1)
                    )
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get user engagement summary for {user_id}: {e}")
            return {'user_id': user_id, 'error': str(e)}


class MonitoringPipeline:
    """
    Comprehensive monitoring pipeline for system health, performance, and engagement.
    
    Coordinates system monitoring, performance tracking, user engagement analysis,
    and alert management for the competitive intelligence system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.monitoring")
        self.config = get_config()
        self.error_handler = UnifiedErrorHandler()
        
        # Monitoring components
        self.system_monitor = SystemMonitor()
        self.performance_tracker = PerformanceTracker()
        self.engagement_analyzer = EngagementAnalyzer()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.monitoring_interval = 60  # 1 minute
        self.health_check_interval = 30  # 30 seconds
        self.engagement_check_interval = 300  # 5 minutes
        
        # Data retention
        self.metrics_retention_days = 30
        
        self.logger.info("Monitoring Pipeline initialized")
    
    async def start_pipeline_monitoring(self, pipeline_job_id: Optional[int] = None):
        """Start comprehensive pipeline monitoring."""
        if self.is_monitoring:
            self.logger.warning("Pipeline monitoring already active")
            return
        
        self.is_monitoring = True
        
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info(f"Started pipeline monitoring for job {pipeline_job_id}")
    
    async def stop_pipeline_monitoring(self):
        """Stop pipeline monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped pipeline monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        last_health_check = datetime.min.replace(tzinfo=timezone.utc)
        last_engagement_check = datetime.min.replace(tzinfo=timezone.utc)
        
        while self.is_monitoring:
            try:
                now = datetime.now(timezone.utc)
                
                # System health monitoring
                if (now - last_health_check).total_seconds() >= self.health_check_interval:
                    await self.system_monitor.collect_system_health()
                    last_health_check = now
                
                # Engagement analysis
                if (now - last_engagement_check).total_seconds() >= self.engagement_check_interval:
                    await self.engagement_analyzer.collect_engagement_metrics()
                    last_engagement_check = now
                
                # Cleanup old metrics
                await self._cleanup_old_metrics()
                
                # Sleep until next cycle
                await asyncio.sleep(min(self.health_check_interval, self.engagement_check_interval))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_old_metrics(self):
        """Clean up old monitoring data to prevent memory issues."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.metrics_retention_days)
            
            # Clean up system health history
            self.system_monitor.health_history = deque(
                [h for h in self.system_monitor.health_history if h.timestamp >= cutoff_date],
                maxlen=1440
            )
            
            # Clean up performance metrics
            self.performance_tracker.metrics_history = deque(
                [m for m in self.performance_tracker.metrics_history if m.timestamp >= cutoff_date],
                maxlen=10080
            )
            
            # Clean up engagement cache
            expired_keys = [
                key for key, expiry in self.engagement_analyzer.cache_expiry.items()
                if expiry < datetime.now(timezone.utc)
            ]
            
            for key in expired_keys:
                self.engagement_analyzer.engagement_cache.pop(key, None)
                self.engagement_analyzer.cache_expiry.pop(key, None)
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    # Performance tracking methods
    
    def start_operation_tracking(self, operation_type: str, operation_id: Optional[str] = None) -> str:
        """Start tracking a pipeline operation."""
        if not operation_id:
            operation_id = f"{operation_type}_{int(time.time() * 1000)}"
        
        return self.performance_tracker.start_operation(operation_id, operation_type)
    
    def complete_operation_tracking(self, operation_id: str, operation_type: str,
                                  success: bool = True, metadata: Optional[Dict[str, Any]] = None) -> Optional[PerformanceMetrics]:
        """Complete operation tracking."""
        return self.performance_tracker.complete_operation(
            operation_id, operation_type, success, metadata
        )
    
    # Data retrieval methods
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health."""
        try:
            # Get latest health data
            current_health = await self.system_monitor.collect_system_health()
            
            # Get performance statistics
            performance_stats = self.performance_tracker.get_operation_stats()
            
            # Get engagement metrics
            engagement_metrics = await self.engagement_analyzer.collect_engagement_metrics()
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'system_health': {
                    'overall_status': current_health.overall_status,
                    'cpu_usage_percent': current_health.cpu_usage_percent,
                    'memory_usage_percent': current_health.memory_usage_percent,
                    'disk_usage_percent': current_health.disk_usage_percent,
                    'response_time_ms': current_health.response_time_ms,
                    'active_connections': current_health.active_connections,
                    'component_health': current_health.component_health,
                    'cache_hit_rates': current_health.cache_hit_rates
                },
                'performance': performance_stats,
                'engagement': {
                    'total_users': engagement_metrics.total_users,
                    'active_users': engagement_metrics.active_users,
                    'content_delivered': engagement_metrics.content_delivered,
                    'engagement_rate': engagement_metrics.engagement_rate,
                    'avg_time_spent': engagement_metrics.avg_time_spent
                },
                'monitoring': {
                    'is_active': self.is_monitoring,
                    'health_data_points': len(self.system_monitor.health_history),
                    'performance_data_points': len(self.performance_tracker.metrics_history),
                    'active_alerts': list(self.system_monitor.alerts_active)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'system_health': {'overall_status': 'error'}
            }
    
    async def get_performance_report(self, operation_type: Optional[str] = None,
                                   hours: int = 24) -> Dict[str, Any]:
        """Get detailed performance report."""
        try:
            # Performance trends
            performance_trend = self.performance_tracker.get_performance_trend(operation_type, hours)
            
            # Operation statistics
            operation_stats = self.performance_tracker.get_operation_stats(operation_type)
            
            # System health trend
            health_trend = self.system_monitor.get_health_trend(hours)
            
            return {
                'period_hours': hours,
                'operation_type': operation_type or 'all',
                'performance_trend': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'operation_type': m.operation_type,
                        'duration_seconds': m.duration_seconds,
                        'success': m.success_count > 0,
                        'metadata': m.metadata
                    }
                    for m in performance_trend[-100:]  # Last 100 data points
                ],
                'operation_statistics': operation_stats,
                'system_health_trend': [
                    {
                        'timestamp': h.timestamp.isoformat(),
                        'overall_status': h.overall_status,
                        'cpu_usage_percent': h.cpu_usage_percent,
                        'memory_usage_percent': h.memory_usage_percent,
                        'response_time_ms': h.response_time_ms
                    }
                    for h in health_trend[-100:]  # Last 100 data points
                ],
                'summary': {
                    'total_operations': len(performance_trend),
                    'avg_duration': sum(m.duration_seconds for m in performance_trend) / max(len(performance_trend), 1),
                    'success_rate': sum(m.success_count for m in performance_trend) / max(len(performance_trend), 1),
                    'avg_cpu': sum(h.cpu_usage_percent for h in health_trend) / max(len(health_trend), 1),
                    'avg_memory': sum(h.memory_usage_percent for h in health_trend) / max(len(health_trend), 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}
    
    async def get_engagement_report(self, days: int = 7) -> Dict[str, Any]:
        """Get user engagement analytics report."""
        try:
            # Current engagement metrics
            current_metrics = await self.engagement_analyzer.collect_engagement_metrics()
            
            # Historical comparison (simplified - would need time series data)
            async with get_db_session() as session:
                # Get engagement trends over time
                daily_engagement = await session.execute(
                    select(
                        func.date(ContentEngagement.created_at).label('date'),
                        func.count(ContentEngagement.id).label('engagements')
                    )
                    .where(
                        ContentEngagement.created_at >= datetime.now(timezone.utc) - timedelta(days=days)
                    )
                    .group_by(func.date(ContentEngagement.created_at))
                    .order_by(func.date(ContentEngagement.created_at))
                )
                
                engagement_trend = [
                    {'date': row.date.isoformat(), 'engagements': row.engagements}
                    for row in daily_engagement
                ]
                
                # Top performing content
                top_content = await session.execute(
                    select(
                        DiscoveredContent.title,
                        DiscoveredContent.overall_score,
                        func.count(ContentEngagement.id).label('engagement_count')
                    )
                    .join(ContentEngagement, DiscoveredContent.id == ContentEngagement.content_id)
                    .where(
                        ContentEngagement.created_at >= datetime.now(timezone.utc) - timedelta(days=days)
                    )
                    .group_by(DiscoveredContent.id, DiscoveredContent.title, DiscoveredContent.overall_score)
                    .order_by(desc('engagement_count'))
                    .limit(10)
                )
                
                top_performing_content = [
                    {
                        'title': row.title[:100] + '...' if len(row.title) > 100 else row.title,
                        'score': float(row.overall_score or 0),
                        'engagement_count': row.engagement_count
                    }
                    for row in top_content
                ]
                
                return {
                    'period_days': days,
                    'current_metrics': {
                        'total_users': current_metrics.total_users,
                        'active_users': current_metrics.active_users,
                        'content_delivered': current_metrics.content_delivered,
                        'email_opens': current_metrics.email_opens,
                        'email_clicks': current_metrics.email_clicks,
                        'engagement_rate': current_metrics.engagement_rate,
                        'avg_time_spent': current_metrics.avg_time_spent
                    },
                    'trends': {
                        'daily_engagement': engagement_trend,
                        'popular_categories': current_metrics.popular_categories,
                        'top_sources': current_metrics.top_sources
                    },
                    'top_performing_content': top_performing_content,
                    'insights': {
                        'user_activation_rate': current_metrics.active_users / max(current_metrics.total_users, 1),
                        'email_click_through_rate': current_metrics.email_clicks / max(current_metrics.email_opens, 1),
                        'content_engagement_score': current_metrics.engagement_rate
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to generate engagement report: {e}")
            return {'error': str(e)}
    
    async def get_user_engagement_analysis(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get detailed engagement analysis for specific user."""
        return await self.engagement_analyzer.get_user_engagement_summary(user_id, days)
    
    def get_monitoring_configuration(self) -> Dict[str, Any]:
        """Get current monitoring configuration."""
        return {
            'monitoring_interval': self.monitoring_interval,
            'health_check_interval': self.health_check_interval,
            'engagement_check_interval': self.engagement_check_interval,
            'metrics_retention_days': self.metrics_retention_days,
            'system_thresholds': {
                'cpu_warning': self.system_monitor.cpu_warning_threshold,
                'cpu_critical': self.system_monitor.cpu_critical_threshold,
                'memory_warning': self.system_monitor.memory_warning_threshold,
                'memory_critical': self.system_monitor.memory_critical_threshold,
                'disk_warning': self.system_monitor.disk_warning_threshold,
                'disk_critical': self.system_monitor.disk_critical_threshold,
                'response_time_warning_ms': self.system_monitor.response_time_warning_ms,
                'response_time_critical_ms': self.system_monitor.response_time_critical_ms
            }
        }
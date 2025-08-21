"""
Unified source orchestration manager for coordinating multiple discovery engines.
Handles intelligent load balancing, quota management, and result aggregation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .base_engine import BaseDiscoveryEngine, DiscoveredItem, SourceMetrics
from .news_api_client import NewsAPIClient
from .rss_monitor import RSSMonitor
from .web_scraper import WebScraper


class EngineStatus(str, Enum):
    """Engine status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMITED = "rate_limited"


@dataclass
class EngineConfig:
    """Configuration for a discovery engine."""
    name: str
    engine_class: type
    config: Dict[str, Any]
    priority: int = 5  # 1-10 scale, 10 = highest priority
    weight: float = 1.0  # Weight for result aggregation
    is_enabled: bool = True
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None


@dataclass
class DiscoveryJob:
    """Discovery job specification."""
    job_id: str
    keywords: List[str]
    focus_areas: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    limit: int = 20
    min_quality_score: float = 0.3
    engines_requested: List[str] = field(default_factory=list)  # Empty = use all available
    priority: int = 5
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DiscoveryResult:
    """Discovery job result."""
    job_id: str
    items: List[DiscoveredItem]
    engines_used: List[str]
    total_items_found: int
    processing_time: float
    quality_distribution: Dict[str, int]  # Quality score ranges
    source_distribution: Dict[str, int]  # Sources used
    errors: List[str] = field(default_factory=list)
    completed_at: datetime = field(default_factory=datetime.now)


class EngineLoadBalancer:
    """Intelligent load balancer for discovery engines."""
    
    def __init__(self):
        self.engine_performance: Dict[str, Dict[str, float]] = {}
        self.recent_performance_window = timedelta(hours=1)
    
    def record_performance(self, engine_name: str, response_time: float, 
                          success: bool, items_count: int):
        """Record engine performance metrics."""
        if engine_name not in self.engine_performance:
            self.engine_performance[engine_name] = {
                'avg_response_time': response_time,
                'success_rate': 1.0 if success else 0.0,
                'avg_items_per_request': items_count,
                'total_requests': 1,
                'successful_requests': 1 if success else 0
            }
        else:
            perf = self.engine_performance[engine_name]
            perf['total_requests'] += 1
            
            if success:
                perf['successful_requests'] += 1
                # Update rolling averages
                perf['avg_response_time'] = (
                    (perf['avg_response_time'] * (perf['successful_requests'] - 1)) + response_time
                ) / perf['successful_requests']
                perf['avg_items_per_request'] = (
                    (perf['avg_items_per_request'] * (perf['successful_requests'] - 1)) + items_count
                ) / perf['successful_requests']
            
            perf['success_rate'] = perf['successful_requests'] / perf['total_requests']
    
    def get_engine_score(self, engine_name: str, priority: int, weight: float) -> float:
        """Calculate engine score for load balancing."""
        if engine_name not in self.engine_performance:
            return priority * weight  # Default score for new engines
        
        perf = self.engine_performance[engine_name]
        
        # Base score from priority and weight
        base_score = priority * weight
        
        # Adjust based on performance
        success_factor = perf['success_rate']
        speed_factor = max(0.1, 1.0 / (perf['avg_response_time'] / 10))  # Prefer faster engines
        productivity_factor = min(2.0, perf['avg_items_per_request'] / 5)  # Prefer engines that find more
        
        performance_score = base_score * success_factor * speed_factor * productivity_factor
        
        return performance_score
    
    def select_engines(self, available_engines: List[Tuple[str, int, float]], 
                      count: int) -> List[str]:
        """Select best engines for a discovery job."""
        if not available_engines:
            return []
        
        # Calculate scores for all engines
        engine_scores = [
            (name, self.get_engine_score(name, priority, weight))
            for name, priority, weight in available_engines
        ]
        
        # Sort by score descending
        engine_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top engines
        selected = [name for name, score in engine_scores[:count]]
        
        return selected


class ResultAggregator:
    """Aggregates and deduplicates results from multiple engines."""
    
    def __init__(self):
        self.similarity_threshold = 0.85
    
    async def aggregate_results(self, results_by_engine: Dict[str, List[DiscoveredItem]],
                              engine_weights: Dict[str, float],
                              limit: int) -> List[DiscoveredItem]:
        """Aggregate results from multiple engines."""
        all_items = []
        
        # Collect all items with engine weighting
        for engine_name, items in results_by_engine.items():
            weight = engine_weights.get(engine_name, 1.0)
            
            for item in items:
                # Boost relevance score based on engine weight
                item.relevance_score = (item.relevance_score or 0.5) * weight
                item.metadata['engine'] = engine_name
                item.metadata['engine_weight'] = weight
                all_items.append(item)
        
        # Remove duplicates
        deduplicated = await self._deduplicate_items(all_items)
        
        # Sort by combined score (quality + relevance + recency)
        scored_items = []
        for item in deduplicated:
            combined_score = self._calculate_combined_score(item)
            item.metadata['combined_score'] = combined_score
            scored_items.append((item, combined_score))
        
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, score in scored_items[:limit]]
    
    async def _deduplicate_items(self, items: List[DiscoveredItem]) -> List[DiscoveredItem]:
        """Remove duplicate items based on URL and content similarity."""
        if not items:
            return []
        
        unique_items = []
        seen_urls = set()
        
        for item in items:
            # Skip if exact URL match
            if item.url in seen_urls:
                continue
            
            # Check for similar titles/content
            is_duplicate = False
            for existing_item in unique_items:
                similarity = self._calculate_similarity(item, existing_item)
                if similarity > self.similarity_threshold:
                    # Keep the higher quality item
                    if item.quality_score > existing_item.quality_score:
                        unique_items.remove(existing_item)
                        seen_urls.remove(existing_item.url)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_items.append(item)
                seen_urls.add(item.url)
        
        return unique_items
    
    def _calculate_similarity(self, item1: DiscoveredItem, item2: DiscoveredItem) -> float:
        """Calculate similarity between two items."""
        # URL similarity
        if item1.url == item2.url:
            return 1.0
        
        # Title similarity
        title1_words = set(item1.title.lower().split())
        title2_words = set(item2.title.lower().split())
        
        if title1_words and title2_words:
            title_similarity = len(title1_words & title2_words) / len(title1_words | title2_words)
        else:
            title_similarity = 0.0
        
        # Content similarity (first 200 chars)
        content1 = item1.content[:200].lower()
        content2 = item2.content[:200].lower()
        
        if content1 and content2:
            content_words1 = set(content1.split())
            content_words2 = set(content2.split())
            if content_words1 and content_words2:
                content_similarity = len(content_words1 & content_words2) / len(content_words1 | content_words2)
            else:
                content_similarity = 0.0
        else:
            content_similarity = 0.0
        
        # Combined similarity (weighted)
        combined_similarity = (title_similarity * 0.6) + (content_similarity * 0.4)
        
        return combined_similarity
    
    def _calculate_combined_score(self, item: DiscoveredItem) -> float:
        """Calculate combined scoring for item ranking."""
        quality_score = item.quality_score or 0.0
        relevance_score = item.relevance_score or 0.0
        
        # Recency bonus (up to 0.2 points for content from last 24 hours)
        recency_bonus = 0.0
        if item.published_at:
            hours_old = (datetime.now() - item.published_at).total_seconds() / 3600
            if hours_old < 24:
                recency_bonus = 0.2 * (1 - hours_old / 24)
        
        # Source type bonus
        source_bonus = {
            'news_api': 0.1,
            'rss_feed': 0.05,
            'web_scraper': 0.0
        }.get(item.source_type.value, 0.0)
        
        combined_score = (
            (quality_score * 0.4) +
            (relevance_score * 0.4) +
            recency_bonus +
            source_bonus
        )
        
        return combined_score


class SourceManager(BaseDiscoveryEngine):
    """Unified source orchestration manager."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("source_manager", None)  # Manager doesn't have specific source type
        
        self.engines: Dict[str, BaseDiscoveryEngine] = {}
        self.engine_configs: Dict[str, EngineConfig] = {}
        self.engine_statuses: Dict[str, EngineStatus] = {}
        
        self.load_balancer = EngineLoadBalancer()
        self.result_aggregator = ResultAggregator()
        
        self.active_jobs: Dict[str, DiscoveryJob] = {}
        self.job_history: List[DiscoveryResult] = []
        self.max_history_size = 1000
        
        # Configuration
        self.max_concurrent_engines = config.get('max_concurrent_engines', 3)
        self.default_timeout = config.get('default_timeout', 30)
        self.enable_fallback = config.get('enable_fallback', True)
        
        # Initialize engines based on config
        self._initialize_engines(config)
        
        self.logger.info(f"Initialized SourceManager with {len(self.engines)} engines")
    
    def _initialize_engines(self, config: Dict[str, Any]):
        """Initialize discovery engines based on configuration."""
        # News API client
        if config.get('news_apis'):
            news_config = EngineConfig(
                name="news_api",
                engine_class=NewsAPIClient,
                config=config['news_apis'],
                priority=8,
                weight=1.2
            )
            self.engine_configs['news_api'] = news_config
        
        # RSS Monitor
        if config.get('rss_monitor', {}).get('enabled', True):
            rss_config = EngineConfig(
                name="rss_monitor",
                engine_class=RSSMonitor,
                config=config.get('rss_monitor', {}),
                priority=6,
                weight=1.0
            )
            self.engine_configs['rss_monitor'] = rss_config
        
        # Web Scraper
        if config.get('web_scraper', {}).get('enabled', False):  # Disabled by default due to ethical concerns
            scraper_config = EngineConfig(
                name="web_scraper",
                engine_class=WebScraper,
                config=config.get('web_scraper', {}),
                priority=4,
                weight=0.8
            )
            self.engine_configs['web_scraper'] = scraper_config
        
        # Initialize engine instances
        for engine_name, engine_config in self.engine_configs.items():
            try:
                if engine_config.is_enabled:
                    engine_instance = engine_config.engine_class(engine_config.config)
                    self.engines[engine_name] = engine_instance
                    self.engine_statuses[engine_name] = EngineStatus.ACTIVE
                    self.logger.info(f"Initialized engine: {engine_name}")
                else:
                    self.engine_statuses[engine_name] = EngineStatus.INACTIVE
            except Exception as e:
                self.logger.error(f"Failed to initialize engine {engine_name}: {e}")
                self.engine_statuses[engine_name] = EngineStatus.ERROR
                engine_config.last_error = str(e)
    
    async def discover_content(self, keywords: List[str], focus_areas: List[str] = None,
                             entities: List[str] = None, limit: int = 20) -> List[DiscoveredItem]:
        """Discover content using multiple engines with intelligent orchestration."""
        job_id = f"discover_{datetime.now().isoformat()}"
        
        job = DiscoveryJob(
            job_id=job_id,
            keywords=keywords,
            focus_areas=focus_areas or [],
            entities=entities or [],
            limit=limit
        )
        
        return await self.execute_discovery_job(job)
    
    async def execute_discovery_job(self, job: DiscoveryJob) -> List[DiscoveredItem]:
        """Execute a discovery job using optimal engine selection."""
        start_time = datetime.now()
        self.active_jobs[job.job_id] = job
        
        try:
            # Select engines for this job
            available_engines = self._get_available_engines(job)
            selected_engines = self._select_engines_for_job(job, available_engines)
            
            if not selected_engines:
                self.logger.warning("No available engines for discovery job")
                return []
            
            # Execute discovery on selected engines
            results_by_engine = await self._execute_on_engines(
                selected_engines, job.keywords, job.focus_areas, job.entities, job.limit
            )
            
            # Aggregate results
            engine_weights = {name: self.engine_configs[name].weight for name in selected_engines}
            aggregated_items = await self.result_aggregator.aggregate_results(
                results_by_engine, engine_weights, job.limit
            )
            
            # Filter by quality threshold
            quality_filtered = [
                item for item in aggregated_items 
                if item.quality_score >= job.min_quality_score
            ]
            
            # Create result summary
            processing_time = (datetime.now() - start_time).total_seconds()
            total_found = sum(len(items) for items in results_by_engine.values())
            
            result = DiscoveryResult(
                job_id=job.job_id,
                items=quality_filtered,
                engines_used=list(selected_engines),
                total_items_found=total_found,
                processing_time=processing_time,
                quality_distribution=self._get_quality_distribution(quality_filtered),
                source_distribution=self._get_source_distribution(quality_filtered)
            )
            
            # Store result
            self.job_history.append(result)
            if len(self.job_history) > self.max_history_size:
                self.job_history = self.job_history[-self.max_history_size:]
            
            self.logger.info(f"Discovery job {job.job_id} completed: {len(quality_filtered)} items in {processing_time:.2f}s")
            
            return quality_filtered
            
        except Exception as e:
            self.logger.error(f"Discovery job {job.job_id} failed: {e}")
            return []
        finally:
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    def _get_available_engines(self, job: DiscoveryJob) -> List[str]:
        """Get list of available engines for a job."""
        available = []
        
        for engine_name, status in self.engine_statuses.items():
            # Skip if engine is requested specifically and not in the list
            if job.engines_requested and engine_name not in job.engines_requested:
                continue
            
            # Skip if engine is not active
            if status != EngineStatus.ACTIVE:
                continue
            
            # Skip if engine config is disabled
            if not self.engine_configs[engine_name].is_enabled:
                continue
            
            available.append(engine_name)
        
        return available
    
    def _select_engines_for_job(self, job: DiscoveryJob, available_engines: List[str]) -> List[str]:
        """Select optimal engines for a discovery job."""
        if job.engines_requested:
            # Use specifically requested engines if available
            return [name for name in job.engines_requested if name in available_engines]
        
        # Use load balancer to select engines
        engine_specs = [
            (name, self.engine_configs[name].priority, self.engine_configs[name].weight)
            for name in available_engines
        ]
        
        # Select based on job priority and engine availability
        max_engines = min(len(available_engines), self.max_concurrent_engines)
        if job.priority >= 8:  # High priority job
            max_engines = len(available_engines)  # Use all available engines
        elif job.priority >= 6:  # Medium priority
            max_engines = min(3, len(available_engines))
        else:  # Low priority
            max_engines = min(2, len(available_engines))
        
        selected = self.load_balancer.select_engines(engine_specs, max_engines)
        
        return selected
    
    async def _execute_on_engines(self, engine_names: List[str], keywords: List[str],
                                 focus_areas: List[str], entities: List[str],
                                 limit: int) -> Dict[str, List[DiscoveredItem]]:
        """Execute discovery on multiple engines concurrently."""
        results = {}
        
        # Calculate limit per engine
        limit_per_engine = max(5, limit // len(engine_names))
        
        # Create tasks for each engine
        tasks = []
        for engine_name in engine_names:
            if engine_name in self.engines:
                task = self._execute_on_single_engine(
                    engine_name, keywords, focus_areas, entities, limit_per_engine
                )
                tasks.append((engine_name, task))
        
        # Execute tasks concurrently
        for engine_name, task in tasks:
            try:
                start_time = datetime.now()
                items = await asyncio.wait_for(task, timeout=self.default_timeout)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                results[engine_name] = items
                
                # Record performance
                self.load_balancer.record_performance(
                    engine_name, processing_time, True, len(items)
                )
                
                # Update engine status
                self.engine_configs[engine_name].last_success = datetime.now()
                self.engine_configs[engine_name].retry_count = 0
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Engine {engine_name} timed out")
                results[engine_name] = []
                self.load_balancer.record_performance(engine_name, self.default_timeout, False, 0)
                
            except Exception as e:
                self.logger.error(f"Engine {engine_name} failed: {e}")
                results[engine_name] = []
                self.load_balancer.record_performance(engine_name, 0, False, 0)
                
                # Update engine error state
                self.engine_configs[engine_name].retry_count += 1
                self.engine_configs[engine_name].last_error = str(e)
                
                if self.engine_configs[engine_name].retry_count >= self.engine_configs[engine_name].max_retries:
                    self.engine_statuses[engine_name] = EngineStatus.ERROR
        
        return results
    
    async def _execute_on_single_engine(self, engine_name: str, keywords: List[str],
                                      focus_areas: List[str], entities: List[str],
                                      limit: int) -> List[DiscoveredItem]:
        """Execute discovery on a single engine."""
        engine = self.engines[engine_name]
        
        try:
            return await engine.discover_content(keywords, focus_areas, entities, limit)
        except Exception as e:
            self.logger.error(f"Engine {engine_name} discovery failed: {e}")
            raise
    
    def _get_quality_distribution(self, items: List[DiscoveredItem]) -> Dict[str, int]:
        """Get distribution of quality scores."""
        distribution = {
            'high (0.8-1.0)': 0,
            'medium (0.5-0.8)': 0,
            'low (0.0-0.5)': 0
        }
        
        for item in items:
            score = item.quality_score or 0.0
            if score >= 0.8:
                distribution['high (0.8-1.0)'] += 1
            elif score >= 0.5:
                distribution['medium (0.5-0.8)'] += 1
            else:
                distribution['low (0.0-0.5)'] += 1
        
        return distribution
    
    def _get_source_distribution(self, items: List[DiscoveredItem]) -> Dict[str, int]:
        """Get distribution of source types."""
        distribution = {}
        
        for item in items:
            source = item.source_type.value
            distribution[source] = distribution.get(source, 0) + 1
        
        return distribution
    
    async def test_connection(self) -> bool:
        """Test connection to all engines."""
        results = []
        
        for engine_name, engine in self.engines.items():
            try:
                result = await engine.test_connection()
                results.append(result)
                if result:
                    self.engine_statuses[engine_name] = EngineStatus.ACTIVE
                else:
                    self.engine_statuses[engine_name] = EngineStatus.ERROR
            except Exception as e:
                self.logger.error(f"Engine {engine_name} test failed: {e}")
                self.engine_statuses[engine_name] = EngineStatus.ERROR
                results.append(False)
        
        return any(results)  # Return True if at least one engine is working
    
    def get_quota_info(self) -> Dict[str, Any]:
        """Get quota information for all engines."""
        quota_info = {
            'engines': {},
            'overall_status': 'healthy'
        }
        
        active_engines = 0
        for engine_name, engine in self.engines.items():
            try:
                engine_quota = engine.get_quota_info()
                quota_info['engines'][engine_name] = {
                    'status': self.engine_statuses[engine_name].value,
                    'quota': engine_quota,
                    'config': {
                        'priority': self.engine_configs[engine_name].priority,
                        'weight': self.engine_configs[engine_name].weight,
                        'enabled': self.engine_configs[engine_name].is_enabled
                    }
                }
                
                if self.engine_statuses[engine_name] == EngineStatus.ACTIVE:
                    active_engines += 1
                    
            except Exception as e:
                quota_info['engines'][engine_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        if active_engines == 0:
            quota_info['overall_status'] = 'critical'
        elif active_engines < len(self.engines) / 2:
            quota_info['overall_status'] = 'degraded'
        
        quota_info['active_engines'] = active_engines
        quota_info['total_engines'] = len(self.engines)
        
        return quota_info
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get detailed status of all engines."""
        status = {}
        
        for engine_name, engine_config in self.engine_configs.items():
            status[engine_name] = {
                'status': self.engine_statuses.get(engine_name, EngineStatus.INACTIVE).value,
                'priority': engine_config.priority,
                'weight': engine_config.weight,
                'enabled': engine_config.is_enabled,
                'retry_count': engine_config.retry_count,
                'max_retries': engine_config.max_retries,
                'last_error': engine_config.last_error,
                'last_success': engine_config.last_success.isoformat() if engine_config.last_success else None,
                'performance': self.load_balancer.engine_performance.get(engine_name, {})
            }
        
        return status
    
    def get_job_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent job history."""
        recent_jobs = self.job_history[-limit:] if self.job_history else []
        
        return [
            {
                'job_id': result.job_id,
                'items_returned': len(result.items),
                'total_items_found': result.total_items_found,
                'engines_used': result.engines_used,
                'processing_time': result.processing_time,
                'completed_at': result.completed_at.isoformat(),
                'quality_distribution': result.quality_distribution,
                'source_distribution': result.source_distribution
            }
            for result in reversed(recent_jobs)
        ]
    
    async def enable_engine(self, engine_name: str) -> bool:
        """Enable a specific engine."""
        if engine_name in self.engine_configs:
            self.engine_configs[engine_name].is_enabled = True
            
            # Try to initialize if not already initialized
            if engine_name not in self.engines:
                try:
                    config = self.engine_configs[engine_name]
                    engine_instance = config.engine_class(config.config)
                    self.engines[engine_name] = engine_instance
                    self.engine_statuses[engine_name] = EngineStatus.ACTIVE
                    self.logger.info(f"Enabled and initialized engine: {engine_name}")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to initialize engine {engine_name}: {e}")
                    self.engine_statuses[engine_name] = EngineStatus.ERROR
                    return False
            else:
                self.engine_statuses[engine_name] = EngineStatus.ACTIVE
                self.logger.info(f"Enabled engine: {engine_name}")
                return True
        
        return False
    
    async def disable_engine(self, engine_name: str) -> bool:
        """Disable a specific engine."""
        if engine_name in self.engine_configs:
            self.engine_configs[engine_name].is_enabled = False
            self.engine_statuses[engine_name] = EngineStatus.INACTIVE
            self.logger.info(f"Disabled engine: {engine_name}")
            return True
        
        return False
    
    async def close(self):
        """Close all engine connections."""
        for engine in self.engines.values():
            try:
                if hasattr(engine, 'close'):
                    await engine.close()
            except Exception as e:
                self.logger.error(f"Error closing engine: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
        except:
            pass
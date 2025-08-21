"""
Main orchestrator for the Discovery Engine with comprehensive error handling,
async patterns, and integration management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import traceback
from contextlib import asynccontextmanager

from .engines.source_manager import SourceManager
from .ml_integration import DiscoveryMLIntegrator
from .user_config_integration import UserConfigIntegrator
from .content_processor import ContentProcessor, ProcessedContent
from .engines.base_engine import DiscoveredItem
from ..services.discovery_service import DiscoveryService
from ..utils.exceptions import errors


@dataclass
class DiscoveryRequest:
    """Discovery request specification."""
    user_id: int
    keywords: List[str] = field(default_factory=list)
    focus_areas: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    limit: int = 20
    quality_threshold: float = 0.3
    enable_ml_scoring: bool = True
    enable_content_processing: bool = True
    engines_requested: List[str] = field(default_factory=list)
    timeout_seconds: int = 60
    request_id: str = field(default_factory=lambda: f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


@dataclass
class DiscoveryResponse:
    """Discovery response with comprehensive metadata."""
    request_id: str
    user_id: int
    items: List[Union[DiscoveredItem, ProcessedContent]]
    total_found: int
    processing_time: float
    engines_used: List[str]
    ml_scoring_enabled: bool
    content_processing_enabled: bool
    quality_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)


class HealthMonitor:
    """Health monitoring for discovery system components."""
    
    def __init__(self):
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = {}
        
    async def check_component_health(self, component_name: str, 
                                   health_check_func) -> Dict[str, Any]:
        """Check health of a specific component."""
        now = datetime.now()
        
        # Skip if recently checked
        if component_name in self.last_health_check:
            last_check = self.last_health_check[component_name]
            if (now - last_check).total_seconds() < self.health_check_interval:
                return self.component_health.get(component_name, {})
        
        try:
            start_time = now
            is_healthy = await health_check_func()
            response_time = (datetime.now() - start_time).total_seconds()
            
            health_status = {
                'healthy': is_healthy,
                'last_check': now.isoformat(),
                'response_time': response_time,
                'status': 'ok' if is_healthy else 'error'
            }
            
            self.component_health[component_name] = health_status
            self.last_health_check[component_name] = now
            
            return health_status
            
        except Exception as e:
            error_status = {
                'healthy': False,
                'last_check': now.isoformat(),
                'error': str(e),
                'status': 'error'
            }
            
            self.component_health[component_name] = error_status
            self.last_health_check[component_name] = now
            
            return error_status
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.component_health:
            return {'status': 'unknown', 'components': {}}
        
        healthy_components = sum(
            1 for health in self.component_health.values() 
            if health.get('healthy', False)
        )
        total_components = len(self.component_health)
        
        if healthy_components == total_components:
            status = 'healthy'
        elif healthy_components > total_components / 2:
            status = 'degraded'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'healthy_components': healthy_components,
            'total_components': total_components,
            'components': self.component_health
        }


class ErrorHandler:
    """Centralized error handling for discovery operations."""
    
    def __init__(self):
        self.logger = logging.getLogger("discovery.error_handler")
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
    async def handle_with_retry(self, operation_name: str, operation_func, 
                               max_retries: int = 3, backoff_factor: float = 1.0):
        """Execute operation with retry logic and circuit breaker pattern."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            # Check circuit breaker
            if self._is_circuit_open(operation_name):
                raise Exception(f"Circuit breaker open for {operation_name}")
            
            try:
                result = await operation_func()
                
                # Reset error count on success
                self.error_counts[operation_name] = 0
                self._close_circuit(operation_name)
                
                return result
                
            except Exception as e:
                last_exception = e
                self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
                
                self.logger.warning(f"Attempt {attempt + 1} failed for {operation_name}: {e}")
                
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    # Open circuit breaker if too many failures
                    if self.error_counts[operation_name] >= 5:
                        self._open_circuit(operation_name)
        
        raise last_exception
    
    def _is_circuit_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open."""
        if operation_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[operation_name]
        if not breaker.get('is_open', False):
            return False
        
        # Check if circuit should be half-opened
        opened_at = breaker.get('opened_at', datetime.now())
        cooldown_period = timedelta(minutes=5)  # 5 minute cooldown
        
        if datetime.now() - opened_at > cooldown_period:
            breaker['is_open'] = False
            breaker['is_half_open'] = True
            return False
        
        return True
    
    def _open_circuit(self, operation_name: str):
        """Open circuit breaker for operation."""
        self.circuit_breakers[operation_name] = {
            'is_open': True,
            'is_half_open': False,
            'opened_at': datetime.now()
        }
        self.logger.error(f"Opened circuit breaker for {operation_name}")
    
    def _close_circuit(self, operation_name: str):
        """Close circuit breaker for operation."""
        if operation_name in self.circuit_breakers:
            self.circuit_breakers[operation_name] = {
                'is_open': False,
                'is_half_open': False,
                'closed_at': datetime.now()
            }
    
    @asynccontextmanager
    async def error_context(self, operation_name: str):
        """Context manager for error handling."""
        try:
            yield
        except Exception as e:
            self.logger.error(f"Error in {operation_name}: {e}")
            self.logger.debug(traceback.format_exc())
            raise


class DiscoveryOrchestrator:
    """Main orchestrator for the Discovery Engine system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("discovery.orchestrator")
        
        # Initialize components
        self.discovery_service = DiscoveryService()
        self.source_manager = SourceManager(config.get('sources', {}))
        self.ml_integrator = DiscoveryMLIntegrator(self.discovery_service)
        self.user_config_integrator = UserConfigIntegrator(self.ml_integrator)
        self.content_processor = ContentProcessor()
        
        # System components
        self.health_monitor = HealthMonitor()
        self.error_handler = ErrorHandler()
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.last_cleanup = datetime.now()
        
        # Configuration
        self.max_concurrent_requests = config.get('max_concurrent_requests', 5)
        self.default_timeout = config.get('default_timeout', 60)
        self.cleanup_interval = timedelta(hours=1)
        
        # Request tracking
        self.active_requests: Dict[str, DiscoveryRequest] = {}
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        self.logger.info("Discovery orchestrator initialized")
    
    async def discover_content(self, request: DiscoveryRequest) -> DiscoveryResponse:
        """Main entry point for content discovery."""
        async with self.request_semaphore:
            return await self._execute_discovery_request(request)
    
    async def _execute_discovery_request(self, request: DiscoveryRequest) -> DiscoveryResponse:
        """Execute discovery request with comprehensive error handling."""
        start_time = datetime.now()
        self.active_requests[request.request_id] = request
        
        response = DiscoveryResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            items=[],
            total_found=0,
            processing_time=0.0,
            engines_used=[],
            ml_scoring_enabled=request.enable_ml_scoring,
            content_processing_enabled=request.enable_content_processing
        )
        
        try:
            async with self.error_handler.error_context("discovery_request"):
                # Step 1: Get user profile and preferences
                await self._enrich_request_with_user_data(request)
                
                # Step 2: Execute source discovery
                discovered_items = await self._execute_source_discovery(request, response)
                
                # Step 3: Apply ML scoring if enabled
                if request.enable_ml_scoring and discovered_items:
                    discovered_items = await self._apply_ml_scoring(discovered_items, request, response)
                
                # Step 4: Apply content processing if enabled
                if request.enable_content_processing and discovered_items:
                    processed_items = await self._process_discovered_content(discovered_items, response)
                    response.items = processed_items
                else:
                    response.items = discovered_items
                
                # Step 5: Apply final filtering and ranking
                response.items = await self._apply_final_filtering(response.items, request)
                
                # Step 6: Generate analytics
                self._generate_response_analytics(response)
                
        except asyncio.TimeoutError:
            error_msg = f"Discovery request {request.request_id} timed out"
            self.logger.error(error_msg)
            response.errors.append(error_msg)
            
        except Exception as e:
            error_msg = f"Discovery request {request.request_id} failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            response.errors.append(error_msg)
        
        finally:
            # Cleanup and finalize response
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            
            response.processing_time = (datetime.now() - start_time).total_seconds()
            response.completed_at = datetime.now()
            
            # Update performance metrics
            self.request_count += 1
            self.total_processing_time += response.processing_time
            
            # Periodic cleanup
            await self._periodic_cleanup()
        
        return response
    
    async def _enrich_request_with_user_data(self, request: DiscoveryRequest):
        """Enrich request with user profile data."""
        try:
            # Get user keywords if not provided
            if not request.keywords:
                user_keywords = await self.user_config_integrator.get_user_discovery_keywords(request.user_id)
                request.keywords = user_keywords[:10]  # Limit keywords
            
            # Get focus areas if not provided
            if not request.focus_areas:
                focus_areas_data = await self.user_config_integrator.get_user_focus_areas(request.user_id)
                request.focus_areas = [fa['focus_area'] for fa in focus_areas_data[:5]]
            
            # Get tracked entities if not provided
            if not request.entities:
                entities_data = await self.user_config_integrator.get_user_tracked_entities(request.user_id)
                request.entities = [entity['entity_name'] for entity in entities_data[:5]]
            
        except Exception as e:
            self.logger.warning(f"Failed to enrich request with user data: {e}")
    
    async def _execute_source_discovery(self, request: DiscoveryRequest, 
                                      response: DiscoveryResponse) -> List[DiscoveredItem]:
        """Execute source discovery with timeout and error handling."""
        try:
            discovery_task = self.source_manager.discover_content(
                keywords=request.keywords,
                focus_areas=request.focus_areas,
                entities=request.entities,
                limit=request.limit
            )
            
            discovered_items = await asyncio.wait_for(
                discovery_task, 
                timeout=request.timeout_seconds
            )
            
            response.total_found = len(discovered_items)
            response.engines_used = self._extract_engines_used(discovered_items)
            
            return discovered_items
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Source discovery timed out for request {request.request_id}")
            response.warnings.append("Source discovery timed out")
            return []
        
        except Exception as e:
            self.logger.error(f"Source discovery failed: {e}")
            response.errors.append(f"Source discovery failed: {str(e)}")
            return []
    
    async def _apply_ml_scoring(self, items: List[DiscoveredItem], 
                              request: DiscoveryRequest,
                              response: DiscoveryResponse) -> List[DiscoveredItem]:
        """Apply ML scoring with error handling."""
        try:
            scored_items = await self.ml_integrator.score_discovered_items(items, request.user_id)
            return scored_items
            
        except Exception as e:
            self.logger.error(f"ML scoring failed: {e}")
            response.warnings.append("ML scoring failed, using basic scoring")
            return items
    
    async def _process_discovered_content(self, items: List[DiscoveredItem],
                                        response: DiscoveryResponse) -> List[ProcessedContent]:
        """Process discovered content with error handling."""
        try:
            processed_items = await self.content_processor.batch_process_content(items)
            return processed_items
            
        except Exception as e:
            self.logger.error(f"Content processing failed: {e}")
            response.warnings.append("Content processing failed, returning raw content")
            return items  # Return original items if processing fails
    
    async def _apply_final_filtering(self, items: List[Union[DiscoveredItem, ProcessedContent]],
                                   request: DiscoveryRequest) -> List[Union[DiscoveredItem, ProcessedContent]]:
        """Apply final filtering and ranking."""
        # Filter by quality threshold
        filtered_items = []
        for item in items:
            original_item = item.original_item if isinstance(item, ProcessedContent) else item
            if (original_item.quality_score or 0.0) >= request.quality_threshold:
                filtered_items.append(item)
        
        # Sort by relevance score
        filtered_items.sort(
            key=lambda x: (
                x.original_item.relevance_score if isinstance(x, ProcessedContent) 
                else x.relevance_score
            ) or 0.0,
            reverse=True
        )
        
        return filtered_items[:request.limit]
    
    def _extract_engines_used(self, items: List[DiscoveredItem]) -> List[str]:
        """Extract unique engines used from discovered items."""
        engines = set()
        for item in items:
            if 'engine' in item.metadata:
                engines.add(item.metadata['engine'])
        return list(engines)
    
    def _generate_response_analytics(self, response: DiscoveryResponse):
        """Generate analytics for the response."""
        items = response.items
        
        # Quality distribution
        quality_ranges = {'high (0.8-1.0)': 0, 'medium (0.5-0.8)': 0, 'low (0.0-0.5)': 0}
        
        # Source distribution
        source_dist = {}
        
        for item in items:
            original_item = item.original_item if isinstance(item, ProcessedContent) else item
            
            # Quality distribution
            score = original_item.quality_score or 0.0
            if score >= 0.8:
                quality_ranges['high (0.8-1.0)'] += 1
            elif score >= 0.5:
                quality_ranges['medium (0.5-0.8)'] += 1
            else:
                quality_ranges['low (0.0-0.5)'] += 1
            
            # Source distribution
            source = original_item.source_name
            source_dist[source] = source_dist.get(source, 0) + 1
        
        response.quality_distribution = quality_ranges
        response.source_distribution = source_dist
        
        # Additional metadata
        response.metadata = {
            'avg_quality_score': self._calculate_avg_quality(items),
            'content_types': self._get_content_types(items),
            'processing_enabled': response.content_processing_enabled,
            'ml_scoring_enabled': response.ml_scoring_enabled
        }
    
    def _calculate_avg_quality(self, items: List[Union[DiscoveredItem, ProcessedContent]]) -> float:
        """Calculate average quality score."""
        if not items:
            return 0.0
        
        total_score = 0.0
        for item in items:
            original_item = item.original_item if isinstance(item, ProcessedContent) else item
            total_score += original_item.quality_score or 0.0
        
        return total_score / len(items)
    
    def _get_content_types(self, items: List[Union[DiscoveredItem, ProcessedContent]]) -> Dict[str, int]:
        """Get distribution of content types."""
        content_types = {}
        for item in items:
            original_item = item.original_item if isinstance(item, ProcessedContent) else item
            content_type = original_item.content_type.value
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        return content_types
    
    async def _periodic_cleanup(self):
        """Perform periodic cleanup tasks."""
        now = datetime.now()
        if now - self.last_cleanup > self.cleanup_interval:
            try:
                # Clear caches
                self.user_config_integrator.invalidate_all_caches()
                self.content_processor.clear_cache()
                
                self.logger.info("Performed periodic cleanup")
                self.last_cleanup = now
                
            except Exception as e:
                self.logger.error(f"Periodic cleanup failed: {e}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        try:
            # Check component health
            source_manager_health = await self.health_monitor.check_component_health(
                "source_manager", self.source_manager.test_connection
            )
            
            # Get overall health
            overall_health = self.health_monitor.get_overall_health()
            
            # Add performance metrics
            avg_processing_time = (
                self.total_processing_time / max(self.request_count, 1)
            )
            
            return {
                'overall_status': overall_health['status'],
                'components': {
                    'source_manager': source_manager_health,
                    **overall_health['components']
                },
                'performance': {
                    'total_requests': self.request_count,
                    'avg_processing_time': avg_processing_time,
                    'active_requests': len(self.active_requests),
                    'max_concurrent_requests': self.max_concurrent_requests
                },
                'quotas': await self._get_quota_status()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def _get_quota_status(self) -> Dict[str, Any]:
        """Get quota status from all sources."""
        try:
            return self.source_manager.get_quota_info()
        except Exception as e:
            self.logger.error(f"Failed to get quota status: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_processing_time = (
            self.total_processing_time / max(self.request_count, 1)
        )
        
        return {
            'total_requests': self.request_count,
            'avg_processing_time': avg_processing_time,
            'active_requests': len(self.active_requests),
            'error_counts': self.error_handler.error_counts,
            'circuit_breakers': self.error_handler.circuit_breakers,
            'cache_stats': {
                'user_config': self.user_config_integrator.get_cache_stats(),
                'content_processor': self.content_processor.get_processing_stats(),
                'ml_integrator': self.ml_integrator.get_ml_performance_metrics()
            }
        }
    
    async def close(self):
        """Close all connections and cleanup resources."""
        try:
            await self.source_manager.close()
            self.logger.info("Discovery orchestrator closed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
        except:
            pass
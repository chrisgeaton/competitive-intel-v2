"""
Service Base Classes - Consolidated service patterns and mixins.

Provides reusable service patterns, mixins, and base classes following
Phase 1 & 2 optimization strategies for code consolidation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from .shared_types import (
    AnalysisContext, AnalysisStage, ContentPriority, ServiceConfig,
    FilterResult, AnalysisBatch, AIResponse, DEFAULT_CONFIG,
    validate_analysis_context, validate_content_for_analysis
)
from .ai_integration import AIProviderManager, AIProviderError

logger = logging.getLogger(__name__)


# === Base Service Classes ===

class BaseAnalysisService(ABC):
    """Base class for analysis services with common functionality."""
    
    def __init__(self, service_name: str, config: ServiceConfig = None):
        self.service_name = service_name
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger(f"{__name__}.{service_name}")
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {
            "requests_processed": 0,
            "total_processing_time_ms": 0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize service components - override in subclasses."""
        pass
    
    @abstractmethod
    async def process_content(
        self,
        content: Dict[str, Any],
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """Process content - implemented by concrete services."""
        pass
    
    async def process_batch(
        self,
        batch: AnalysisBatch,
        stages: List[AnalysisStage] = None
    ) -> List[Dict[str, Any]]:
        """Process batch of content with optimization."""
        start_time = time.time()
        
        try:
            results = []
            
            # Process items in parallel with controlled concurrency
            semaphore = asyncio.Semaphore(self.config.max_concurrent_analyses)
            
            async def process_item(content_item):
                async with semaphore:
                    return await self.process_content(content_item, batch.context)
            
            # Create tasks for parallel processing
            tasks = [process_item(item) for item in batch.content_items]
            
            # Execute with progress tracking
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Item {i} failed: {result}")
                    self.performance_metrics["errors_encountered"] += 1
                    
                    # Add error result
                    results.append({
                        "content_id": batch.content_items[i].get("id"),
                        "error": str(result),
                        "processing_failed": True
                    })
                else:
                    results.append(result)
            
            # Update performance metrics
            processing_time = int((time.time() - start_time) * 1000)
            self.performance_metrics["requests_processed"] += len(batch.content_items)
            self.performance_metrics["total_processing_time_ms"] += processing_time
            
            self.logger.info(
                f"Processed batch {batch.batch_id}: {len(results)} items "
                f"in {processing_time}ms"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed for {batch.batch_id}: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics."""
        total_requests = self.performance_metrics["requests_processed"]
        
        stats = {
            "service_name": self.service_name,
            "total_requests": total_requests,
            "total_errors": self.performance_metrics["errors_encountered"],
            "error_rate": (self.performance_metrics["errors_encountered"] / max(1, total_requests)),
            "avg_processing_time_ms": (
                self.performance_metrics["total_processing_time_ms"] / max(1, total_requests)
            ),
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] / 
                max(1, self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"])
            )
        }
        
        return stats


# === Service Mixins ===

class ValidationMixin:
    """Mixin for common validation functionality."""
    
    def validate_context(self, context: AnalysisContext) -> None:
        """Validate analysis context with enhanced error messages."""
        try:
            validate_analysis_context(context)
        except ValueError as e:
            self.logger.error(f"Context validation failed: {e}")
            raise
    
    def validate_content(self, content: Dict[str, Any]) -> None:
        """Validate content for analysis."""
        try:
            validate_content_for_analysis(content)
        except ValueError as e:
            self.logger.error(f"Content validation failed: {e}")
            raise
    
    def validate_batch(self, batch: AnalysisBatch) -> None:
        """Validate analysis batch."""
        if not batch.content_items:
            raise ValueError("Batch contains no content items")
        
        if not batch.context:
            raise ValueError("Batch missing analysis context")
        
        self.validate_context(batch.context)
        
        # Validate each content item
        for i, item in enumerate(batch.content_items):
            try:
                self.validate_content(item)
            except ValueError as e:
                raise ValueError(f"Content item {i} invalid: {e}")


class ErrorHandlingMixin:
    """Mixin for consistent error handling patterns."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[Dict[str, Any]] = []
    
    def handle_error(
        self,
        error: Exception,
        context: str = "",
        reraise: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Handle errors with logging and tracking."""
        error_type = type(error).__name__
        
        # Track error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log error with context
        self.logger.error(f"Error in {context}: {error_type}: {error}")
        
        # Track recent errors (keep last 20)
        error_info = {
            "type": error_type,
            "message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        self.recent_errors.append(error_info)
        if len(self.recent_errors) > 20:
            self.recent_errors.pop(0)
        
        # Handle specific error types
        if isinstance(error, AIProviderError):
            return self._handle_ai_error(error, context)
        elif isinstance(error, ValueError):
            return self._handle_validation_error(error, context)
        elif isinstance(error, asyncio.TimeoutError):
            return self._handle_timeout_error(error, context)
        
        if reraise:
            raise
        
        return None
    
    def _handle_ai_error(self, error: AIProviderError, context: str) -> Dict[str, Any]:
        """Handle AI provider specific errors."""
        return {
            "error_type": "ai_provider",
            "message": str(error),
            "provider": error.provider.value if error.provider else None,
            "retry_after": error.retry_after,
            "context": context
        }
    
    def _handle_validation_error(self, error: ValueError, context: str) -> Dict[str, Any]:
        """Handle validation errors."""
        return {
            "error_type": "validation",
            "message": str(error),
            "context": context,
            "recoverable": False
        }
    
    def _handle_timeout_error(self, error: asyncio.TimeoutError, context: str) -> Dict[str, Any]:
        """Handle timeout errors."""
        return {
            "error_type": "timeout",
            "message": "Operation timed out",
            "context": context,
            "retry_recommended": True
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_types": self.error_counts.copy(),
            "recent_errors": self.recent_errors[-5:],  # Last 5 errors
            "error_rate": total_errors / max(1, getattr(self, 'performance_metrics', {}).get('requests_processed', 1))
        }


class PerformanceMixin:
    """Mixin for performance monitoring and optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_history: List[Dict[str, Any]] = []
        self._operation_timers: Dict[str, float] = {}
    
    def start_timer(self, operation_name: str) -> None:
        """Start timing an operation."""
        self._operation_timers[operation_name] = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """End timing and return duration in milliseconds."""
        if operation_name not in self._operation_timers:
            return 0.0
        
        duration_ms = (time.time() - self._operation_timers[operation_name]) * 1000
        del self._operation_timers[operation_name]
        
        # Record performance data
        self._record_performance(operation_name, duration_ms)
        
        return duration_ms
    
    def _record_performance(self, operation: str, duration_ms: float) -> None:
        """Record performance data point."""
        perf_data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        self.performance_history.append(perf_data)
        
        # Keep only recent history (last 100 operations)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    async def timed_operation(
        self,
        operation_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with automatic timing."""
        self.start_timer(operation_name)
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            return result
            
        finally:
            self.end_timer(operation_name)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {"no_data": True}
        
        # Calculate statistics by operation
        operation_stats = {}
        
        for record in self.performance_history:
            operation = record["operation"]
            duration = record["duration_ms"]
            
            if operation not in operation_stats:
                operation_stats[operation] = {
                    "count": 0,
                    "total_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0
                }
            
            stats = operation_stats[operation]
            stats["count"] += 1
            stats["total_time"] += duration
            stats["min_time"] = min(stats["min_time"], duration)
            stats["max_time"] = max(stats["max_time"], duration)
        
        # Calculate averages
        for operation, stats in operation_stats.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
        
        return {
            "operation_stats": operation_stats,
            "total_operations": len(self.performance_history),
            "time_range": {
                "start": self.performance_history[0]["timestamp"],
                "end": self.performance_history[-1]["timestamp"]
            }
        }


class CachingMixin:
    """Mixin for caching functionality."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(seconds=self.config.cache_ttl_seconds if hasattr(self, 'config') else 3600)
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        import hashlib
        
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached(self, cache_key: str) -> Optional[Any]:
        """Get item from cache if valid."""
        if cache_key not in self._cache:
            return None
        
        # Check TTL
        if cache_key in self._cache_timestamps:
            age = datetime.now() - self._cache_timestamps[cache_key]
            if age > self._cache_ttl:
                # Expired
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]
                return None
        
        return self._cache[cache_key]
    
    def set_cached(self, cache_key: str, value: Any) -> None:
        """Set item in cache."""
        self._cache[cache_key] = value
        self._cache_timestamps[cache_key] = datetime.now()
        
        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest 25% of items
            sorted_items = sorted(
                self._cache_timestamps.items(),
                key=lambda x: x[1]
            )
            
            for key, _ in sorted_items[:250]:
                if key in self._cache:
                    del self._cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]
    
    async def cached_operation(
        self,
        cache_key: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with caching."""
        # Check cache first
        cached_result = self.get_cached(cache_key)
        if cached_result is not None:
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics["cache_hits"] += 1
            return cached_result
        
        # Execute operation
        if asyncio.iscoroutinefunction(operation):
            result = await operation(*args, **kwargs)
        else:
            result = operation(*args, **kwargs)
        
        # Cache result
        self.set_cached(cache_key, result)
        
        if hasattr(self, 'performance_metrics'):
            self.performance_metrics["cache_misses"] += 1
        
        return result
    
    def clear_cache(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._cache_timestamps.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = getattr(self, 'performance_metrics', {}).get('cache_hits', 0)
        total_misses = getattr(self, 'performance_metrics', {}).get('cache_misses', 0)
        total_requests = total_hits + total_misses
        
        return {
            "cache_size": len(self._cache),
            "hit_rate": total_hits / max(1, total_requests),
            "total_hits": total_hits,
            "total_misses": total_misses
        }


# === Enhanced Analysis Service ===

class EnhancedAnalysisService(
    BaseAnalysisService,
    ValidationMixin,
    ErrorHandlingMixin,
    PerformanceMixin,
    CachingMixin
):
    """Enhanced analysis service with all optimization mixins."""
    
    def __init__(self, service_name: str = "enhanced_analysis", config: ServiceConfig = None):
        super().__init__(service_name, config)
        
        # Initialize AI provider manager
        self.ai_manager = AIProviderManager(config)
    
    async def process_content(
        self,
        content: Dict[str, Any],
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """Process single content item with full optimization."""
        content_id = content.get("id", "unknown")
        
        try:
            # Validate inputs
            self.validate_content(content)
            self.validate_context(context)
            
            # Generate cache key
            cache_key = self._generate_cache_key(content_id, context.user_id)
            
            # Try cached result first
            cached_result = self.get_cached(cache_key)
            if cached_result:
                return cached_result
            
            # Process with timing
            result = await self.timed_operation(
                "content_analysis",
                self._analyze_content_internal,
                content,
                context
            )
            
            # Cache successful result
            if not result.get("error"):
                self.set_cached(cache_key, result)
            
            return result
            
        except Exception as e:
            return self.handle_error(e, f"content_analysis_{content_id}", reraise=False)
    
    async def _analyze_content_internal(
        self,
        content: Dict[str, Any],
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """Internal content analysis implementation."""
        # This would contain the actual analysis logic
        # For now, return a structured result
        return {
            "content_id": content.get("id"),
            "analysis_completed": True,
            "processing_time": datetime.now().isoformat(),
            "context_summary": context.get_context_summary()
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        return {
            "performance": self.get_performance_stats(),
            "performance_summary": self.get_performance_summary(),
            "error_summary": self.get_error_summary(),
            "cache_stats": self.get_cache_stats(),
            "ai_provider_stats": self.ai_manager.get_provider_stats()
        }


# === Factory Functions ===

def create_enhanced_analysis_service(
    service_name: str = "analysis",
    config: ServiceConfig = None
) -> EnhancedAnalysisService:
    """Factory function to create enhanced analysis service."""
    return EnhancedAnalysisService(service_name, config)


def create_service_config(**kwargs) -> ServiceConfig:
    """Factory function to create service configuration."""
    return ServiceConfig(**kwargs)
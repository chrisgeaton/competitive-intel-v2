"""
Optimization Manager - Advanced performance and resource management.

Consolidates optimization patterns from Phase 1 & 2 for enhanced
Analysis Service performance and resource utilization.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

from .shared_types import (
    AnalysisStage, ContentPriority, ServiceConfig, AnalysisContext,
    DEFAULT_CONFIG
)

logger = logging.getLogger(__name__)


# === Cache Strategy Enum ===

class CacheStrategy(Enum):
    """Cache strategy options for optimization."""
    MEMORY_ONLY = "memory_only"
    PERSISTENT = "persistent"
    HYBRID = "hybrid"
    DISABLED = "disabled"


# === Performance Monitor ===

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    execution_count: int = 0
    total_time_ms: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.execution_count == 0:
            return 1.0
        return self.success_count / self.execution_count
    
    @property
    def avg_execution_time(self) -> float:
        """Calculate average execution time."""
        if self.execution_count == 0:
            return 0.0
        return self.total_time_ms / self.execution_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "operation_name": self.operation_name,
            "execution_count": self.execution_count,
            "success_rate": self.success_rate,
            "avg_execution_time_ms": self.avg_execution_time,
            "total_time_ms": self.total_time_ms,
            "avg_memory_usage_mb": self.avg_memory_usage,
            "peak_memory_usage_mb": self.peak_memory_usage,
            "cache_hit_rate": self.cache_hit_rate
        }


class PerformanceMonitor:
    """Advanced performance monitoring with optimization recommendations."""
    
    def __init__(self, config: ServiceConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._monitoring_active = True
        self._start_time = time.time()
    
    def record_operation(
        self,
        operation_name: str,
        execution_time_ms: int,
        success: bool,
        memory_usage_mb: float = 0.0,
        cache_hit: bool = False
    ):
        """Record operation performance data."""
        if not self._monitoring_active:
            return
        
        if operation_name not in self.metrics:
            self.metrics[operation_name] = PerformanceMetrics(operation_name)
        
        metric = self.metrics[operation_name]
        metric.execution_count += 1
        metric.total_time_ms += execution_time_ms
        
        if success:
            metric.success_count += 1
        else:
            metric.error_count += 1
        
        # Update memory usage (exponential moving average)
        alpha = 0.1
        metric.avg_memory_usage = (alpha * memory_usage_mb + 
                                 (1 - alpha) * metric.avg_memory_usage)
        metric.peak_memory_usage = max(metric.peak_memory_usage, memory_usage_mb)
        
        # Update cache hit rate
        if cache_hit:
            current_hits = metric.cache_hit_rate * (metric.execution_count - 1)
            metric.cache_hit_rate = (current_hits + 1) / metric.execution_count
        else:
            current_hits = metric.cache_hit_rate * (metric.execution_count - 1)
            metric.cache_hit_rate = current_hits / metric.execution_count
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total_operations = sum(m.execution_count for m in self.metrics.values())
        total_time = sum(m.total_time_ms for m in self.metrics.values())
        
        return {
            "monitoring_duration_minutes": (time.time() - self._start_time) / 60,
            "total_operations": total_operations,
            "total_time_ms": total_time,
            "operations": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "top_performers": self._get_top_performers(),
            "optimization_recommendations": self._get_optimization_recommendations()
        }
    
    def _get_top_performers(self) -> Dict[str, Any]:
        """Identify top performing operations."""
        if not self.metrics:
            return {}
        
        sorted_by_speed = sorted(
            self.metrics.values(),
            key=lambda m: m.avg_execution_time
        )
        
        sorted_by_reliability = sorted(
            self.metrics.values(),
            key=lambda m: m.success_rate,
            reverse=True
        )
        
        return {
            "fastest_operations": [m.operation_name for m in sorted_by_speed[:3]],
            "most_reliable": [m.operation_name for m in sorted_by_reliability[:3]]
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        for metric in self.metrics.values():
            # Slow operations
            if metric.avg_execution_time > 5000:  # 5 seconds
                recommendations.append(
                    f"Consider optimization for {metric.operation_name} "
                    f"(avg: {metric.avg_execution_time:.1f}ms)"
                )
            
            # Low cache hit rates
            if metric.cache_hit_rate < 0.5 and metric.execution_count > 10:
                recommendations.append(
                    f"Improve caching for {metric.operation_name} "
                    f"(hit rate: {metric.cache_hit_rate:.1%})"
                )
            
            # High error rates
            if metric.success_rate < 0.95:
                recommendations.append(
                    f"Improve reliability for {metric.operation_name} "
                    f"(success rate: {metric.success_rate:.1%})"
                )
        
        return recommendations


# === Resource Manager ===

class ResourceManager:
    """Advanced resource management and throttling."""
    
    def __init__(self, config: ServiceConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.active_operations: Dict[str, int] = {}
        self.resource_limits = {
            "max_concurrent_analyses": self.config.max_concurrent_analyses,
            "max_memory_mb": 1024,  # 1GB default
            "max_cpu_percent": 80
        }
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_analyses)
    
    async def acquire_analysis_slot(self, operation_name: str) -> bool:
        """Acquire slot for analysis operation."""
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.timeout_seconds
            )
            
            self.active_operations[operation_name] = self.active_operations.get(operation_name, 0) + 1
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout acquiring slot for {operation_name}")
            return False
    
    def release_analysis_slot(self, operation_name: str):
        """Release analysis operation slot."""
        self._semaphore.release()
        if operation_name in self.active_operations:
            self.active_operations[operation_name] = max(0, self.active_operations[operation_name] - 1)
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource utilization status."""
        return {
            "available_slots": self._semaphore._value,
            "max_concurrent": self.resource_limits["max_concurrent_analyses"],
            "active_operations": dict(self.active_operations),
            "resource_limits": self.resource_limits
        }


# === Batch Optimizer ===

class BatchOptimizer:
    """Optimized batch processing with adaptive sizing."""
    
    def __init__(self, config: ServiceConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.performance_monitor = PerformanceMonitor(config)
        self._optimal_batch_sizes: Dict[str, int] = {}
        self._batch_performance_history: Dict[str, List[float]] = {}
    
    def get_optimal_batch_size(self, operation_type: str, content_count: int) -> int:
        """Calculate optimal batch size based on historical performance."""
        if operation_type not in self._optimal_batch_sizes:
            return min(self.config.batch_size, content_count)
        
        base_size = self._optimal_batch_sizes[operation_type]
        
        # Adjust based on content count
        if content_count < base_size:
            return content_count
        
        # Scale up for large datasets with performance consideration
        scale_factor = min(2.0, content_count / base_size)
        optimal_size = int(base_size * scale_factor)
        
        return min(optimal_size, self.config.batch_size * 2)
    
    def update_batch_performance(
        self,
        operation_type: str,
        batch_size: int,
        processing_time_ms: int,
        success_rate: float
    ):
        """Update batch performance metrics for optimization."""
        if operation_type not in self._batch_performance_history:
            self._batch_performance_history[operation_type] = []
        
        # Calculate efficiency score (items per second with success penalty)
        efficiency = (batch_size / max(1, processing_time_ms / 1000)) * success_rate
        
        history = self._batch_performance_history[operation_type]
        history.append(efficiency)
        
        # Keep only recent history
        if len(history) > 20:
            history.pop(0)
        
        # Update optimal batch size based on performance
        if len(history) >= 5:
            avg_efficiency = sum(history) / len(history)
            
            # If current performance is good, consider it optimal
            if efficiency >= avg_efficiency * 1.1:  # 10% better than average
                self._optimal_batch_sizes[operation_type] = batch_size
    
    async def process_batches(
        self,
        items: List[Any],
        processor_func: Callable,
        operation_type: str,
        **kwargs
    ) -> List[Any]:
        """Process items in optimized batches."""
        if not items:
            return []
        
        optimal_size = self.get_optimal_batch_size(operation_type, len(items))
        batches = [items[i:i + optimal_size] for i in range(0, len(items), optimal_size)]
        
        results = []
        
        for i, batch in enumerate(batches):
            start_time = time.time()
            
            try:
                batch_results = await processor_func(batch, **kwargs)
                results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
                
                # Record performance
                processing_time = int((time.time() - start_time) * 1000)
                success_rate = 1.0
                
                self.update_batch_performance(
                    operation_type, len(batch), processing_time, success_rate
                )
                
                self.performance_monitor.record_operation(
                    f"batch_{operation_type}", processing_time, True
                )
                
            except Exception as e:
                logger.error(f"Batch {i+1} failed for {operation_type}: {str(e)}")
                processing_time = int((time.time() - start_time) * 1000)
                
                self.performance_monitor.record_operation(
                    f"batch_{operation_type}", processing_time, False
                )
                
                # Continue with next batch on failure
                continue
        
        return results


# === Optimization Manager ===

class OptimizationManager:
    """Central optimization manager coordinating all performance enhancements."""
    
    def __init__(self, config: ServiceConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.performance_monitor = PerformanceMonitor(config)
        self.resource_manager = ResourceManager(config)
        self.batch_optimizer = BatchOptimizer(config)
        self.cache_strategy = CacheStrategy.HYBRID
        self._optimization_active = True
    
    async def optimize_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Optimize any operation with comprehensive monitoring."""
        if not self._optimization_active:
            return await operation_func(*args, **kwargs)
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # Acquire resource slot
        slot_acquired = await self.resource_manager.acquire_analysis_slot(operation_name)
        if not slot_acquired:
            raise RuntimeError(f"Could not acquire resource slot for {operation_name}")
        
        try:
            result = await operation_func(*args, **kwargs)
            
            # Record success metrics
            processing_time = int((time.time() - start_time) * 1000)
            memory_after = self._get_memory_usage()
            
            self.performance_monitor.record_operation(
                operation_name,
                processing_time,
                True,
                memory_after - memory_before
            )
            
            return result
            
        except Exception as e:
            # Record failure metrics
            processing_time = int((time.time() - start_time) * 1000)
            memory_after = self._get_memory_usage()
            
            self.performance_monitor.record_operation(
                operation_name,
                processing_time,
                False,
                memory_after - memory_before
            )
            
            logger.error(f"Operation {operation_name} failed: {str(e)}")
            raise
            
        finally:
            self.resource_manager.release_analysis_slot(operation_name)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        return {
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "resource_status": self.resource_manager.get_resource_status(),
            "cache_strategy": self.cache_strategy.value,
            "optimization_active": self._optimization_active,
            "config": {
                "batch_size": self.config.batch_size,
                "max_concurrent": self.config.max_concurrent_analyses,
                "timeout_seconds": self.config.timeout_seconds
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown with final report."""
        self._optimization_active = False
        
        final_report = self.get_optimization_report()
        logger.info("Optimization Manager shutdown complete")
        logger.info(f"Final performance summary: {json.dumps(final_report, indent=2)}")


# === Factory Functions ===

def create_optimization_manager(config: ServiceConfig = None) -> OptimizationManager:
    """Factory function to create optimization manager."""
    return OptimizationManager(config)


def create_performance_monitor(config: ServiceConfig = None) -> PerformanceMonitor:
    """Factory function to create performance monitor."""
    return PerformanceMonitor(config)


def create_batch_optimizer(config: ServiceConfig = None) -> BatchOptimizer:
    """Factory function to create batch optimizer."""
    return BatchOptimizer(config)


def create_resource_manager(config: ServiceConfig = None) -> ResourceManager:
    """Factory function to create resource manager."""
    return ResourceManager(config)
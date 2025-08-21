"""
AI Integration Core - Consolidated AI provider management.

Optimizes AI provider interactions, cost management, and error handling
following Phase 1 & 2 patterns for performance and maintainability.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from decimal import Decimal
from dataclasses import dataclass, field

from .shared_types import (
    AIProvider, AIModelConfig, AIResponse, AnalysisContext, 
    AnalysisStage, ServiceConfig, DEFAULT_CONFIG
)

logger = logging.getLogger(__name__)


# === Enhanced Exception Hierarchy ===

class AIProviderError(Exception):
    """Base AI provider exception with enhanced error info."""
    
    def __init__(self, message: str, provider: Optional[AIProvider] = None, 
                 error_code: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.retry_after = retry_after
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring."""
        return {
            "message": str(self),
            "provider": self.provider.value if self.provider else None,
            "error_code": self.error_code,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp
        }


class AIProviderRateLimitError(AIProviderError):
    """Rate limit exceeded with enhanced retry logic."""
    
    def __init__(self, message: str, provider: AIProvider, retry_after: int = 60):
        super().__init__(message, provider, "RATE_LIMIT", retry_after)


class AIProviderTimeoutError(AIProviderError):
    """Request timeout with provider context."""
    
    def __init__(self, message: str, provider: AIProvider, timeout_duration: int):
        super().__init__(message, provider, "TIMEOUT")
        self.timeout_duration = timeout_duration


class AIProviderQuotaError(AIProviderError):
    """Quota exceeded error."""
    
    def __init__(self, message: str, provider: AIProvider):
        super().__init__(message, provider, "QUOTA_EXCEEDED", 3600)  # 1 hour retry


# === Provider Selection Strategy ===

class ProviderSelectionStrategy:
    """Optimized provider selection with cost and performance considerations."""
    
    def __init__(self, config: ServiceConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.provider_stats: Dict[AIProvider, Dict[str, Any]] = {}
        self._initialize_stats()
    
    def _initialize_stats(self):
        """Initialize provider performance statistics."""
        for provider in AIProvider:
            self.provider_stats[provider] = {
                "success_rate": 1.0,
                "avg_response_time": 1000.0,
                "cost_efficiency": 1.0,
                "last_error": None,
                "consecutive_failures": 0
            }
    
    def select_optimal_provider(
        self,
        priority: str,
        stage: AnalysisStage,
        budget_cents: Optional[int] = None
    ) -> AIProvider:
        """Select optimal provider based on multiple factors."""
        
        # Filter available providers
        available_providers = self._get_available_providers()
        
        if not available_providers:
            return AIProvider.MOCK
        
        # Score providers based on criteria
        scored_providers = []
        
        for provider in available_providers:
            score = self._calculate_provider_score(provider, priority, stage, budget_cents)
            scored_providers.append((provider, score))
        
        # Sort by score (highest first)
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        
        selected_provider = scored_providers[0][0]
        
        logger.debug(f"Selected provider {selected_provider.value} for {stage.value} "
                    f"(priority: {priority}, score: {scored_providers[0][1]:.2f})")
        
        return selected_provider
    
    def _get_available_providers(self) -> List[AIProvider]:
        """Get list of currently available providers."""
        available = []
        
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            
            # Skip providers with consecutive failures
            if stats["consecutive_failures"] >= 3:
                continue
            
            # Skip providers with recent rate limit errors
            if (stats["last_error"] and 
                isinstance(stats["last_error"], AIProviderRateLimitError) and
                time.time() - stats["last_error"].timestamp < stats["last_error"].retry_after):
                continue
            
            available.append(provider)
        
        # Always include MOCK as fallback
        if AIProvider.MOCK not in available:
            available.append(AIProvider.MOCK)
        
        return available
    
    def _calculate_provider_score(
        self,
        provider: AIProvider,
        priority: str,
        stage: AnalysisStage,
        budget_cents: Optional[int]
    ) -> float:
        """Calculate provider score based on multiple factors."""
        stats = self.provider_stats[provider]
        base_score = 0.0
        
        # Success rate factor (40% weight)
        base_score += stats["success_rate"] * 0.4
        
        # Response time factor (20% weight)
        time_score = max(0, 1.0 - (stats["avg_response_time"] / 10000))  # Normalize to 10s
        base_score += time_score * 0.2
        
        # Cost efficiency factor (25% weight)
        base_score += stats["cost_efficiency"] * 0.25
        
        # Provider-specific bonuses (15% weight)
        provider_bonus = self._get_provider_bonus(provider, priority, stage)
        base_score += provider_bonus * 0.15
        
        # Budget penalty
        if budget_cents and provider != AIProvider.MOCK:
            estimated_cost = self._estimate_provider_cost(provider, stage)
            if estimated_cost > budget_cents:
                base_score *= 0.5  # Penalty for exceeding budget
        
        return min(1.0, max(0.0, base_score))
    
    def _get_provider_bonus(self, provider: AIProvider, priority: str, stage: AnalysisStage) -> float:
        """Get provider-specific bonus scores."""
        if provider == AIProvider.OPENAI:
            if priority in ["critical", "high"]:
                return 0.8
            elif stage == AnalysisStage.INSIGHT_EXTRACTION:
                return 0.7
            else:
                return 0.6
        elif provider == AIProvider.ANTHROPIC:
            if stage == AnalysisStage.SUMMARY_GENERATION:
                return 0.8
            else:
                return 0.6
        elif provider == AIProvider.MOCK:
            return 0.3  # Lower bonus for mock
        
        return 0.5
    
    def _estimate_provider_cost(self, provider: AIProvider, stage: AnalysisStage) -> int:
        """Estimate cost for provider and stage."""
        # Simplified cost estimation
        base_costs = {
            AIProvider.OPENAI: 50,
            AIProvider.ANTHROPIC: 60,
            AIProvider.MOCK: 1
        }
        
        stage_multipliers = {
            AnalysisStage.FILTERING: 0.5,
            AnalysisStage.RELEVANCE_ANALYSIS: 1.0,
            AnalysisStage.INSIGHT_EXTRACTION: 1.5,
            AnalysisStage.SUMMARY_GENERATION: 1.2
        }
        
        base_cost = base_costs.get(provider, 50)
        multiplier = stage_multipliers.get(stage, 1.0)
        
        return int(base_cost * multiplier)
    
    def update_provider_stats(
        self,
        provider: AIProvider,
        success: bool,
        response_time_ms: int,
        error: Optional[Exception] = None
    ):
        """Update provider performance statistics."""
        stats = self.provider_stats[provider]
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        stats["success_rate"] = (alpha * (1.0 if success else 0.0) + 
                                (1 - alpha) * stats["success_rate"])
        
        # Update response time
        if success:
            stats["avg_response_time"] = (alpha * response_time_ms + 
                                         (1 - alpha) * stats["avg_response_time"])
        
        # Update consecutive failures
        if success:
            stats["consecutive_failures"] = 0
        else:
            stats["consecutive_failures"] += 1
            stats["last_error"] = error
        
        # Update cost efficiency (inverse of response time and error rate)
        stats["cost_efficiency"] = stats["success_rate"] / max(1, stats["avg_response_time"] / 1000)


# === Enhanced Base AI Provider ===

class BaseAIProvider(ABC):
    """Enhanced base AI provider with optimization and monitoring."""
    
    def __init__(self, config: AIModelConfig, service_config: ServiceConfig = None):
        self.config = config
        self.service_config = service_config or DEFAULT_CONFIG
        self.performance_monitor = ProviderPerformanceMonitor(config.provider)
        
        # Initialize provider-specific optimizations
        self._initialize_optimizations()
    
    def _initialize_optimizations(self):
        """Initialize provider-specific optimizations."""
        self._response_cache: Dict[str, AIResponse] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    @abstractmethod
    async def _make_request(
        self,
        prompt: str,
        content: str,
        context: AnalysisContext,
        stage: AnalysisStage
    ) -> AIResponse:
        """Make actual API request - implemented by concrete providers."""
        pass
    
    async def analyze_content(
        self,
        prompt: str,
        content: str,
        context: AnalysisContext,
        stage: AnalysisStage
    ) -> AIResponse:
        """Optimized content analysis with caching and monitoring."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(prompt, content, context, stage)
            
            if cache_key in self._response_cache:
                self._cache_hits += 1
                cached_response = self._response_cache[cache_key]
                logger.debug(f"Cache hit for {self.config.provider.value} - {stage.value}")
                return cached_response
            
            self._cache_misses += 1
            
            # Make actual request
            response = await self._make_request(prompt, content, context, stage)
            
            # Cache successful responses
            if len(self._response_cache) < 100:  # Limit cache size
                self._response_cache[cache_key] = response
            
            # Update performance monitoring
            processing_time = int((time.time() - start_time) * 1000)
            self.performance_monitor.record_success(processing_time, response.cost_cents)
            
            return response
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.performance_monitor.record_failure(processing_time, str(e))
            raise
    
    def _generate_cache_key(
        self,
        prompt: str,
        content: str,
        context: AnalysisContext,
        stage: AnalysisStage
    ) -> str:
        """Generate cache key for request."""
        import hashlib
        
        key_components = [
            self.config.model_name,
            stage.value,
            context.industry.value,
            context.role.value,
            hashlib.md5(content.encode()).hexdigest()[:8]
        ]
        
        return "_".join(key_components)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self._cache_hits + self._cache_misses
        if total_requests == 0:
            return 0.0
        return self._cache_hits / total_requests
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> int:
        """Estimate cost using model configuration."""
        return self.config.estimate_cost(input_tokens, output_tokens)
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count - can be overridden by providers."""
        # Simple estimation: ~4 characters per token for English
        return max(1, len(text) // 4)


# === Performance Monitoring ===

@dataclass
class ProviderPerformanceMonitor:
    """Monitor provider performance metrics."""
    provider: AIProvider
    total_requests: int = 0
    successful_requests: int = 0
    total_cost_cents: int = 0
    total_processing_time_ms: int = 0
    recent_errors: List[str] = field(default_factory=list)
    
    def record_success(self, processing_time_ms: int, cost_cents: int):
        """Record successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_processing_time_ms += processing_time_ms
        self.total_cost_cents += cost_cents
    
    def record_failure(self, processing_time_ms: int, error_message: str):
        """Record failed request."""
        self.total_requests += 1
        self.total_processing_time_ms += processing_time_ms
        
        # Keep only recent errors
        self.recent_errors.append(error_message)
        if len(self.recent_errors) > 10:
            self.recent_errors.pop(0)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_requests
    
    @property
    def average_cost_per_request(self) -> float:
        """Calculate average cost per request."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_cost_cents / self.successful_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "provider": self.provider.value,
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "avg_processing_time_ms": self.average_processing_time,
            "avg_cost_per_request_cents": self.average_cost_per_request,
            "total_cost_cents": self.total_cost_cents,
            "recent_error_count": len(self.recent_errors)
        }


# === AI Provider Manager ===

class AIProviderManager:
    """Centralized AI provider management with optimization."""
    
    def __init__(self, config: ServiceConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.providers: Dict[AIProvider, BaseAIProvider] = {}
        self.selection_strategy = ProviderSelectionStrategy(config)
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize all available providers."""
        async with self._initialization_lock:
            if self._initialized:
                return
            
            # Initialize providers based on configuration
            await self._initialize_providers()
            self._initialized = True
    
    async def _initialize_providers(self):
        """Initialize specific providers."""
        # This would be implemented with actual provider initialization
        # For now, we'll use a placeholder
        logger.info("AI Provider Manager initialized")
    
    async def analyze_content(
        self,
        content: str,
        context: AnalysisContext,
        stage: AnalysisStage,
        provider: Optional[AIProvider] = None,
        budget_cents: Optional[int] = None
    ) -> AIResponse:
        """Analyze content using optimal provider."""
        if not self._initialized:
            await self.initialize()
        
        # Select provider if not specified
        if provider is None:
            provider = self.selection_strategy.select_optimal_provider(
                context.priority, stage, budget_cents
            )
        
        # Get provider instance
        provider_instance = self.providers.get(provider)
        if not provider_instance:
            raise AIProviderError(f"Provider {provider.value} not available")
        
        # Perform analysis
        return await provider_instance.analyze_content("", content, context, stage)
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers."""
        stats = {}
        
        for provider, instance in self.providers.items():
            if hasattr(instance, 'performance_monitor'):
                stats[provider.value] = instance.performance_monitor.get_stats()
        
        # Add selection strategy stats
        stats["selection_strategy"] = {
            "provider_stats": {
                p.value: s for p, s in self.selection_strategy.provider_stats.items()
            }
        }
        
        return stats


# === Factory Functions ===

def create_ai_provider_manager(config: ServiceConfig = None) -> AIProviderManager:
    """Factory function to create AI provider manager."""
    return AIProviderManager(config)


def create_provider_selection_strategy(config: ServiceConfig = None) -> ProviderSelectionStrategy:
    """Factory function to create provider selection strategy."""
    return ProviderSelectionStrategy(config)
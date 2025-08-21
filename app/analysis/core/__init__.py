"""
Analysis Service Core - Centralized utilities and shared components.

Consolidates common patterns, types, and utilities across Analysis Service
following Phase 1 & 2 optimization patterns for 80% code reduction.
"""

from .shared_types import *
from .ai_integration import *
from .service_base import *
from .optimization_manager import *

__all__ = [
    # Shared Types
    "AnalysisStage", "ContentPriority", "AIProvider", "IndustryType", "RoleType",
    "AnalysisContext", "AIResponse", "PromptTemplate", "ServiceConfig", "FilterResult",
    "AnalysisBatch", "AIModelConfig", "validate_analysis_context", "validate_content_for_analysis",
    
    # AI Integration
    "BaseAIProvider", "AIProviderManager", "ProviderSelectionStrategy",
    "AIProviderError", "AIProviderRateLimitError", "AIProviderTimeoutError",
    "create_ai_provider_manager", "create_provider_selection_strategy",
    
    # Service Base
    "BaseAnalysisService", "ValidationMixin", "ErrorHandlingMixin", 
    "PerformanceMixin", "CachingMixin",
    
    # Optimization Manager
    "OptimizationManager", "CacheStrategy", "BatchOptimizer",
    "PerformanceMonitor", "ResourceManager", "create_optimization_manager"
]
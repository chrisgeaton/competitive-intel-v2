"""
Analysis service utilities for code optimization and shared functionality.

Centralized utilities following Phase 1 & 2 optimization patterns.
"""

from .common_types import (
    AnalysisStage, ContentPriority, FilterStrategy,
    AnalysisContext, FilterResult, FilterMatch,
    ProcessingStatus, StageResult, PipelineResult,
    PipelineStage, AnalysisBatch, AnalysisResult
)
from .cost_optimizer import CostOptimizer, ModelSelector
from .text_processor import TextProcessor, ContentExtractor
from .cache_manager import AnalysisCacheManager
from .batch_processor import AnalysisBatchProcessor
from .validation import DataValidator, ContextValidator

# Cache instances
_cost_optimizer = None
_text_processor = None
_cache_manager = None
_batch_processor = None

def get_cost_optimizer() -> CostOptimizer:
    """Get singleton cost optimizer instance."""
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = CostOptimizer()
    return _cost_optimizer

def get_text_processor() -> TextProcessor:
    """Get singleton text processor instance."""
    global _text_processor
    if _text_processor is None:
        _text_processor = TextProcessor()
    return _text_processor

def get_cache_manager() -> AnalysisCacheManager:
    """Get singleton cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = AnalysisCacheManager()
    return _cache_manager

def get_batch_processor() -> AnalysisBatchProcessor:
    """Get singleton batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = AnalysisBatchProcessor()
    return _batch_processor

__all__ = [
    # Common types
    'AnalysisStage',
    'ContentPriority', 
    'FilterStrategy',
    'AnalysisContext',
    'FilterResult',
    'FilterMatch',
    'ProcessingStatus',
    'StageResult', 
    'PipelineResult',
    'PipelineStage',
    'AnalysisBatch',
    'AnalysisResult',
    
    # Core utilities
    'CostOptimizer',
    'ModelSelector',
    'TextProcessor',
    'ContentExtractor',
    'AnalysisCacheManager',
    'AnalysisBatchProcessor',
    'DataValidator',
    'ContextValidator',
    
    # Singletons
    'get_cost_optimizer',
    'get_text_processor', 
    'get_cache_manager',
    'get_batch_processor'
]
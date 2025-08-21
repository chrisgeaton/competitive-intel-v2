"""
Analysis package for competitive intelligence v2.

Multi-stage content analysis with cost optimization and AI-powered insights.
"""

from .pipeline import AnalysisPipeline, PipelineProcessor
from .filters import (
    ContentFilter,
    KeywordFilter,
    EntityFilter,
    RelevanceFilter,
    CompositeFilter,
    ContentFilterFactory
)
from .utils import (
    AnalysisStage, ContentPriority, FilterStrategy,
    PipelineStage, PipelineResult, ProcessingStatus
)

__all__ = [
    # Pipeline components
    'AnalysisPipeline',
    'PipelineProcessor',
    'PipelineStage', 
    'PipelineResult',
    'ProcessingStatus',
    
    # Filter components
    'ContentFilter',
    'FilterStrategy',
    'KeywordFilter',
    'EntityFilter',
    'RelevanceFilter',
    'CompositeFilter',
    'ContentFilterFactory',
    
    # Core types
    'AnalysisStage',
    'ContentPriority'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Competitive Intelligence Team'
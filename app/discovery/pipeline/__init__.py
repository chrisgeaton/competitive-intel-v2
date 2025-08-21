"""
Daily Processing Pipeline for Discovery Service competitive intelligence system.

This module contains the automated daily processing pipeline that orchestrates
content discovery, ML scoring, deduplication, and delivery for thousands of users
with efficient resource management and comprehensive error recovery.

Features:
- Automated daily content discovery for all active users
- ML-powered content scoring and relevance assessment
- Intelligent deduplication using content similarity detection
- Continuous ML model training from user engagement data
- Advanced job scheduling with error recovery and resource management
- Comprehensive system monitoring and performance analytics
- Secure API integration with existing authentication system
- Extensible architecture ready for podcast integration

Usage Example:
    from app.discovery.pipeline import PipelineAPIIntegration
    
    # Initialize pipeline with configuration
    config = {
        'batch_size': 50,
        'max_concurrent_users': 20,
        'quality_threshold': 0.5
    }
    
    pipeline_api = PipelineAPIIntegration(config)
    await pipeline_api.start_services()
    
    # Execute daily discovery
    result = await pipeline_api.run_daily_pipeline(context)
    
    # Stop services
    await pipeline_api.stop_services()
"""

# Core pipeline components
from .daily_discovery_pipeline import (
    DailyDiscoveryPipeline,
    UserDiscoveryContext,
    PipelineMetrics
)

from .content_processor import (
    ContentProcessor,
    ProcessingResult,
    MLScoringContext,
    QualityMetrics
)

from .ml_training_pipeline import (
    MLTrainingPipeline,
    TrainingMetrics,
    EngagementPattern,
    ModelTrainingData
)

from .job_scheduler import (
    JobScheduler,
    ScheduledJob,
    JobType,
    JobStatus,
    JobPriority,
    ResourceManager
)

from .monitoring_pipeline import (
    MonitoringPipeline,
    SystemHealth,
    PerformanceMetrics,
    UserEngagementMetrics,
    SystemMonitor,
    PerformanceTracker,
    EngagementAnalyzer
)

# Integration and API layer
from .pipeline_integration import (
    PipelineAPIIntegration,
    PipelineContext,
    PipelineAuthenticationManager,
    PipelineServiceIntegrator
)

# Logging and configuration
from .logging_config import (
    PipelineLoggingManager,
    LogEvent,
    LogLevel,
    ErrorRecoveryLogger,
    PerformanceLogger,
    AuditLogger,
    get_pipeline_logger,
    log_operation_start,
    log_operation_complete,
    log_user_context
)

# Test utilities (for development and testing)
from .test_pipeline_integration import (
    TestPipelineConfiguration
)

__version__ = "1.0.0"

__all__ = [
    # Core pipeline components
    'DailyDiscoveryPipeline',
    'UserDiscoveryContext',
    'PipelineMetrics',
    
    # Content processing
    'ContentProcessor',
    'ProcessingResult',
    'MLScoringContext',
    'QualityMetrics',
    
    # ML training
    'MLTrainingPipeline',
    'TrainingMetrics',
    'EngagementPattern',
    'ModelTrainingData',
    
    # Job scheduling
    'JobScheduler',
    'ScheduledJob',
    'JobType',
    'JobStatus',
    'JobPriority',
    'ResourceManager',
    
    # Monitoring
    'MonitoringPipeline',
    'SystemHealth',
    'PerformanceMetrics',
    'UserEngagementMetrics',
    'SystemMonitor',
    'PerformanceTracker',
    'EngagementAnalyzer',
    
    # Integration
    'PipelineAPIIntegration',
    'PipelineContext',
    'PipelineAuthenticationManager',
    'PipelineServiceIntegrator',
    
    # Logging
    'PipelineLoggingManager',
    'LogEvent',
    'LogLevel',
    'ErrorRecoveryLogger',
    'PerformanceLogger',
    'AuditLogger',
    'get_pipeline_logger',
    'log_operation_start',
    'log_operation_complete',
    'log_user_context',
    
    # Test utilities
    'TestPipelineConfiguration'
]

# Convenience function for quick pipeline setup
async def create_pipeline_system(config: dict) -> PipelineAPIIntegration:
    """
    Create and configure a complete pipeline system.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        PipelineAPIIntegration: Configured pipeline system
        
    Example:
        config = {
            'batch_size': 50,
            'max_concurrent_users': 20,
            'content_limit_per_user': 100,
            'quality_threshold': 0.5,
            'log_directory': 'logs/'
        }
        
        pipeline = await create_pipeline_system(config)
        await pipeline.start_services()
    """
    pipeline_api = PipelineAPIIntegration(config)
    await pipeline_api.start_services()
    return pipeline_api
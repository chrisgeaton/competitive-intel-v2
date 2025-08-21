"""
Comprehensive Logging Configuration for Pipeline System.

Provides structured logging, error recovery, performance monitoring,
and audit trails for the competitive intelligence pipeline system.
"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum

from ..utils import UnifiedErrorHandler


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: str
    process_id: int
    user_id: Optional[int] = None
    request_id: Optional[str] = None
    operation_id: Optional[str] = None
    duration_ms: Optional[float] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PipelineFormatter(logging.Formatter):
    """Custom formatter for pipeline logging."""
    
    def __init__(self, include_metadata: bool = True):
        self.include_metadata = include_metadata
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data."""
        
        # Create structured log event
        log_event = LogEvent(
            timestamp=datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=str(record.thread),
            process_id=record.process
        )
        
        # Add custom attributes if present
        for attr in ['user_id', 'request_id', 'operation_id', 'duration_ms', 'error_type']:
            if hasattr(record, attr):
                setattr(log_event, attr, getattr(record, attr))
        
        # Add exception info if present
        if record.exc_info:
            log_event.error_type = record.exc_info[0].__name__ if record.exc_info[0] else None
            log_event.stack_trace = self.formatException(record.exc_info)
        
        # Add metadata
        if self.include_metadata and hasattr(record, 'metadata'):
            log_event.metadata = record.metadata
        
        # Return JSON formatted string
        try:
            return json.dumps(log_event.to_dict(), ensure_ascii=False)
        except (TypeError, ValueError):
            # Fallback to simple formatting if JSON serialization fails
            return f"{log_event.timestamp} {log_event.level} {log_event.logger_name}: {log_event.message}"


class PipelineLogFilter(logging.Filter):
    """Filter for pipeline-specific logging."""
    
    def __init__(self, min_level: str = "INFO", include_modules: Optional[List[str]] = None):
        super().__init__()
        self.min_level = getattr(logging, min_level.upper())
        self.include_modules = include_modules or []
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on level and module."""
        
        # Check log level
        if record.levelno < self.min_level:
            return False
        
        # Check module inclusion
        if self.include_modules:
            return any(module in record.name for module in self.include_modules)
        
        return True


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.log_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.worker_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to async queue."""
        try:
            # Try to put record in queue without blocking
            if not self.log_queue.full():
                self.log_queue.put_nowait(record)
            # If queue is full, drop the record (prefer performance over completeness)
        except asyncio.QueueFull:
            pass  # Drop record if queue is full
    
    async def start_worker(self):
        """Start async log processing worker."""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_task = asyncio.create_task(self._process_logs())
    
    async def stop_worker(self):
        """Stop async log processing worker."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
    
    async def _process_logs(self):
        """Process logs from queue asynchronously."""
        while self.is_running:
            try:
                # Wait for log record with timeout
                record = await asyncio.wait_for(self.log_queue.get(), timeout=1.0)
                
                # Process record with target handler
                self.target_handler.emit(record)
                
                # Mark task done
                self.log_queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # Continue processing
            except Exception as e:
                # Log handler error (avoid infinite recursion)
                print(f"AsyncLogHandler error: {e}", file=sys.stderr)


class ErrorRecoveryLogger:
    """Specialized logger for error recovery and incident tracking."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize error logger
        self.error_logger = logging.getLogger('pipeline.error_recovery')
        self.error_logger.setLevel(logging.ERROR)
        
        # Add rotating file handler for errors
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'error_recovery.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        error_handler.setFormatter(PipelineFormatter())
        self.error_logger.addHandler(error_handler)
        
        # Track error patterns
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
    
    def log_error_with_recovery(self, error: Exception, context: Dict[str, Any],
                              recovery_action: str, recovery_success: bool):
        """Log error with recovery attempt information."""
        
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error tracking
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create error record
        error_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context,
            'recovery_action': recovery_action,
            'recovery_success': recovery_success,
            'occurrence_count': self.error_counts[error_type],
            'stack_trace': traceback.format_exc()
        }
        
        self.error_history.append(error_record)
        
        # Log with structured data
        self.error_logger.error(
            f"Error recovery attempt: {error_type}",
            extra={
                'metadata': error_record,
                'error_type': error_type
            },
            exc_info=True
        )
    
    def log_incident(self, incident_type: str, severity: str, description: str,
                    affected_components: List[str], resolution_steps: List[str]):
        """Log system incident with structured information."""
        
        incident_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'incident_type': incident_type,
            'severity': severity,
            'description': description,
            'affected_components': affected_components,
            'resolution_steps': resolution_steps
        }
        
        self.error_logger.critical(
            f"System incident: {incident_type} ({severity})",
            extra={'metadata': incident_record}
        )
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period."""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error['timestamp']).timestamp() > cutoff_time
        ]
        
        error_type_counts = {}
        for error in recent_errors:
            error_type = error['error_type']
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        return {
            'period_hours': hours,
            'total_errors': len(recent_errors),
            'error_types': error_type_counts,
            'most_common_error': max(error_type_counts.items(), key=lambda x: x[1])[0] if error_type_counts else None,
            'recovery_success_rate': sum(1 for e in recent_errors if e['recovery_success']) / max(len(recent_errors), 1)
        }


class PerformanceLogger:
    """Logger for performance monitoring and profiling."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize performance logger
        self.perf_logger = logging.getLogger('pipeline.performance')
        self.perf_logger.setLevel(logging.INFO)
        
        # Add rotating file handler for performance logs
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'performance.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        perf_handler.setFormatter(PipelineFormatter())
        self.perf_logger.addHandler(perf_handler)
        
        # Performance tracking
        self.operation_timings: Dict[str, List[float]] = {}
        self.slow_operations: List[Dict[str, Any]] = []
        self.slow_operation_threshold = 5.0  # seconds
    
    def log_operation_performance(self, operation_id: str, operation_type: str,
                                duration_ms: float, success: bool,
                                metadata: Optional[Dict[str, Any]] = None):
        """Log operation performance metrics."""
        
        # Track timing
        if operation_type not in self.operation_timings:
            self.operation_timings[operation_type] = []
        
        duration_seconds = duration_ms / 1000
        self.operation_timings[operation_type].append(duration_seconds)
        
        # Track slow operations
        if duration_seconds > self.slow_operation_threshold:
            slow_op = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'operation_id': operation_id,
                'operation_type': operation_type,
                'duration_seconds': duration_seconds,
                'success': success,
                'metadata': metadata or {}
            }
            self.slow_operations.append(slow_op)
        
        # Log performance data
        log_level = logging.WARNING if duration_seconds > self.slow_operation_threshold else logging.INFO
        
        self.perf_logger.log(
            log_level,
            f"Operation performance: {operation_type}",
            extra={
                'operation_id': operation_id,
                'duration_ms': duration_ms,
                'metadata': {
                    'operation_type': operation_type,
                    'duration_seconds': duration_seconds,
                    'success': success,
                    'is_slow': duration_seconds > self.slow_operation_threshold,
                    **(metadata or {})
                }
            }
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {}
        
        for operation_type, timings in self.operation_timings.items():
            if timings:
                summary[operation_type] = {
                    'count': len(timings),
                    'avg_duration': sum(timings) / len(timings),
                    'min_duration': min(timings),
                    'max_duration': max(timings),
                    'slow_operations': sum(1 for t in timings if t > self.slow_operation_threshold)
                }
        
        return {
            'operations': summary,
            'total_slow_operations': len(self.slow_operations),
            'slow_operation_threshold': self.slow_operation_threshold
        }


class AuditLogger:
    """Logger for audit trails and compliance."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize audit logger
        self.audit_logger = logging.getLogger('pipeline.audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Add rotating file handler for audit logs (larger retention)
        audit_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'audit.log',
            maxBytes=100*1024*1024,  # 100MB
            backupCount=20
        )
        audit_handler.setFormatter(PipelineFormatter())
        self.audit_logger.addHandler(audit_handler)
    
    def log_user_action(self, user_id: int, action: str, resource: str,
                       success: bool, ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       additional_data: Optional[Dict[str, Any]] = None):
        """Log user action for audit trail."""
        
        audit_record = {
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'success': success,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'additional_data': additional_data or {}
        }
        
        self.audit_logger.info(
            f"User action: {action} on {resource}",
            extra={
                'user_id': user_id,
                'metadata': audit_record
            }
        )
    
    def log_system_event(self, event_type: str, component: str, description: str,
                        metadata: Optional[Dict[str, Any]] = None):
        """Log system event for audit trail."""
        
        event_record = {
            'event_type': event_type,
            'component': component,
            'description': description,
            'metadata': metadata or {}
        }
        
        self.audit_logger.info(
            f"System event: {event_type} in {component}",
            extra={'metadata': event_record}
        )
    
    def log_data_access(self, user_id: int, data_type: str, data_id: str,
                       access_type: str, success: bool):
        """Log data access for compliance."""
        
        access_record = {
            'user_id': user_id,
            'data_type': data_type,
            'data_id': data_id,
            'access_type': access_type,  # read, write, delete
            'success': success
        }
        
        self.audit_logger.info(
            f"Data access: {access_type} {data_type}",
            extra={
                'user_id': user_id,
                'metadata': access_record
            }
        )


class PipelineLoggingManager:
    """Centralized logging manager for the entire pipeline system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_dir = Path(config.get('log_directory', 'logs/'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize specialized loggers
        self.error_recovery_logger = ErrorRecoveryLogger(self.log_dir)
        self.performance_logger = PerformanceLogger(self.log_dir)
        self.audit_logger = AuditLogger(self.log_dir)
        
        # Async handlers
        self.async_handlers: List[AsyncLogHandler] = []
        
        # Configure root logger
        self._configure_root_logger()
        
        # Configure pipeline loggers
        self._configure_pipeline_loggers()
    
    def _configure_root_logger(self):
        """Configure root logger settings."""
        
        # Set root log level
        logging.root.setLevel(logging.DEBUG)
        
        # Remove default handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Console handler for development
        if self.config.get('enable_console_logging', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s %(levelname)s %(name)s: %(message)s'
                )
            )
            logging.root.addHandler(console_handler)
    
    def _configure_pipeline_loggers(self):
        """Configure specialized pipeline loggers."""
        
        # Main pipeline logger
        pipeline_logger = logging.getLogger('pipeline')
        pipeline_logger.setLevel(logging.INFO)
        
        # Add rotating file handler
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'pipeline.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        main_handler.setFormatter(PipelineFormatter())
        main_handler.addFilter(PipelineLogFilter(min_level="INFO"))
        
        # Wrap in async handler if configured
        if self.config.get('enable_async_logging', True):
            async_handler = AsyncLogHandler(main_handler)
            self.async_handlers.append(async_handler)
            pipeline_logger.addHandler(async_handler)
        else:
            pipeline_logger.addHandler(main_handler)
        
        # Configure component-specific loggers
        components = [
            'pipeline.daily_discovery',
            'pipeline.content_processor', 
            'pipeline.ml_training',
            'pipeline.job_scheduler',
            'pipeline.monitoring',
            'pipeline.auth'
        ]
        
        for component in components:
            logger = logging.getLogger(component)
            logger.setLevel(logging.DEBUG if self.config.get('debug_mode', False) else logging.INFO)
            
            # Component-specific log file
            component_name = component.split('.')[-1]
            component_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f'{component_name}.log',
                maxBytes=20*1024*1024,  # 20MB
                backupCount=5
            )
            component_handler.setFormatter(PipelineFormatter())
            
            if self.config.get('enable_async_logging', True):
                async_handler = AsyncLogHandler(component_handler)
                self.async_handlers.append(async_handler)
                logger.addHandler(async_handler)
            else:
                logger.addHandler(component_handler)
    
    async def start_async_logging(self):
        """Start async log processing workers."""
        for handler in self.async_handlers:
            await handler.start_worker()
    
    async def stop_async_logging(self):
        """Stop async log processing workers."""
        for handler in self.async_handlers:
            await handler.stop_worker()
    
    def log_error_with_recovery(self, error: Exception, context: Dict[str, Any],
                              recovery_action: str, recovery_success: bool):
        """Log error with recovery information."""
        self.error_recovery_logger.log_error_with_recovery(
            error, context, recovery_action, recovery_success
        )
    
    def log_performance(self, operation_id: str, operation_type: str,
                       duration_ms: float, success: bool,
                       metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        self.performance_logger.log_operation_performance(
            operation_id, operation_type, duration_ms, success, metadata
        )
    
    def log_audit_event(self, event_type: str, **kwargs):
        """Log audit event."""
        if event_type == 'user_action':
            self.audit_logger.log_user_action(**kwargs)
        elif event_type == 'system_event':
            self.audit_logger.log_system_event(**kwargs)
        elif event_type == 'data_access':
            self.audit_logger.log_data_access(**kwargs)
    
    def get_logging_summary(self) -> Dict[str, Any]:
        """Get comprehensive logging summary."""
        return {
            'error_recovery': self.error_recovery_logger.get_error_summary(),
            'performance': self.performance_logger.get_performance_summary(),
            'configuration': {
                'log_directory': str(self.log_dir),
                'async_logging_enabled': self.config.get('enable_async_logging', True),
                'console_logging_enabled': self.config.get('enable_console_logging', True),
                'debug_mode': self.config.get('debug_mode', False),
                'async_handlers_count': len(self.async_handlers)
            }
        }


# Convenience functions for pipeline components

def get_pipeline_logger(name: str) -> logging.Logger:
    """Get configured pipeline logger."""
    return logging.getLogger(f'pipeline.{name}')

def log_operation_start(logger: logging.Logger, operation_id: str, 
                       operation_type: str, **metadata):
    """Log operation start with metadata."""
    logger.info(
        f"Starting operation: {operation_type}",
        extra={
            'operation_id': operation_id,
            'metadata': {
                'operation_type': operation_type,
                'phase': 'start',
                **metadata
            }
        }
    )

def log_operation_complete(logger: logging.Logger, operation_id: str,
                         operation_type: str, duration_ms: float,
                         success: bool, **metadata):
    """Log operation completion with metrics."""
    level = logging.INFO if success else logging.WARNING
    
    logger.log(
        level,
        f"Operation {'completed' if success else 'failed'}: {operation_type}",
        extra={
            'operation_id': operation_id,
            'duration_ms': duration_ms,
            'metadata': {
                'operation_type': operation_type,
                'phase': 'complete',
                'success': success,
                'duration_ms': duration_ms,
                **metadata
            }
        }
    )

def log_user_context(logger: logging.Logger, user_id: int, request_id: str):
    """Add user context to logger."""
    # Create a logger adapter with user context
    return logging.LoggerAdapter(logger, {
        'user_id': user_id,
        'request_id': request_id
    })
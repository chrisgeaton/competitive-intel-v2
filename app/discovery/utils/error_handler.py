"""
Standardized error handling for all discovery engines.
Consolidates inconsistent error patterns across the codebase.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class EngineErrorType(Enum):
    """Standardized error types for discovery engines."""
    RATE_LIMIT = "rate_limit"
    AUTH_FAILED = "auth_failed"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    PARSE_ERROR = "parse_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    FORBIDDEN = "forbidden"


class EngineException(Exception):
    """Standardized exception for discovery engines with rich error information."""
    
    def __init__(self, error_type: EngineErrorType, message: str, 
                 status_code: int = None, provider: str = None, 
                 context: Dict[str, Any] = None, is_retryable: bool = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.status_code = status_code
        self.provider = provider
        self.context = context or {}
        self.timestamp = datetime.now()
        
        # Auto-determine if error is retryable
        if is_retryable is None:
            self.is_retryable = self._determine_retryable()
        else:
            self.is_retryable = is_retryable
    
    def _determine_retryable(self) -> bool:
        """Determine if error is retryable based on error type."""
        non_retryable_types = {
            EngineErrorType.AUTH_FAILED,
            EngineErrorType.NOT_FOUND, 
            EngineErrorType.FORBIDDEN,
            EngineErrorType.PARSE_ERROR
        }
        return self.error_type not in non_retryable_types
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.error_type.value,
            'message': self.message,
            'status_code': self.status_code,
            'provider': self.provider,
            'is_retryable': self.is_retryable,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }


class UnifiedErrorHandler:
    """Standardized error handling for all discovery engines."""
    
    # HTTP status code to error type mappings
    ERROR_MAPPINGS = {
        400: EngineErrorType.PARSE_ERROR,
        401: EngineErrorType.AUTH_FAILED,
        403: EngineErrorType.FORBIDDEN,
        404: EngineErrorType.NOT_FOUND,
        429: EngineErrorType.RATE_LIMIT,
        500: EngineErrorType.SERVER_ERROR,
        502: EngineErrorType.SERVER_ERROR,
        503: EngineErrorType.SERVER_ERROR,
        504: EngineErrorType.TIMEOUT,
    }
    
    # Error message templates
    ERROR_MESSAGES = {
        EngineErrorType.RATE_LIMIT: "{provider} rate limit exceeded - quota: {quota_info}",
        EngineErrorType.AUTH_FAILED: "{provider} authentication failed - check API key",
        EngineErrorType.NOT_FOUND: "{provider} resource not found: {resource}",
        EngineErrorType.TIMEOUT: "{provider} request timed out after {timeout}s",
        EngineErrorType.SERVER_ERROR: "{provider} server error ({status_code}): {details}",
        EngineErrorType.NETWORK_ERROR: "{provider} network error: {details}",
        EngineErrorType.PARSE_ERROR: "{provider} response parsing failed: {details}",
        EngineErrorType.QUOTA_EXCEEDED: "{provider} quota exceeded - resets at {reset_time}",
        EngineErrorType.FORBIDDEN: "{provider} access forbidden - insufficient permissions"
    }
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("discovery.error_handler")
    
    def handle_http_error(self, status_code: int, provider: str, 
                         context: Dict[str, Any] = None) -> EngineException:
        """Convert HTTP status codes to standardized exceptions."""
        error_type = self.ERROR_MAPPINGS.get(status_code, EngineErrorType.SERVER_ERROR)
        context = context or {}
        
        # Build error message using template
        message_template = self.ERROR_MESSAGES.get(error_type, "{provider} error ({status_code})")
        message = message_template.format(
            provider=provider,
            status_code=status_code,
            quota_info=context.get('quota_info', 'unknown'),
            resource=context.get('resource', 'unknown'),
            timeout=context.get('timeout', 'unknown'),
            details=context.get('details', 'no details'),
            reset_time=context.get('reset_time', 'unknown')
        )
        
        exception = EngineException(
            error_type=error_type,
            message=message,
            status_code=status_code,
            provider=provider,
            context=context
        )
        
        # Log error with appropriate level
        if error_type in [EngineErrorType.AUTH_FAILED, EngineErrorType.SERVER_ERROR]:
            self.logger.error(f"Critical error: {exception.to_dict()}")
        elif error_type == EngineErrorType.RATE_LIMIT:
            self.logger.warning(f"Rate limit: {exception.to_dict()}")
        else:
            self.logger.info(f"Engine error: {exception.to_dict()}")
        
        return exception
    
    def handle_network_error(self, provider: str, original_error: Exception,
                           context: Dict[str, Any] = None) -> EngineException:
        """Handle network-related errors (timeouts, connection failures, etc.)."""
        context = context or {}
        context['original_error'] = str(original_error)
        
        # Determine specific error type based on original error
        error_type = EngineErrorType.NETWORK_ERROR
        if 'timeout' in str(original_error).lower():
            error_type = EngineErrorType.TIMEOUT
        
        message = self.ERROR_MESSAGES[error_type].format(
            provider=provider,
            details=str(original_error)
        )
        
        exception = EngineException(
            error_type=error_type,
            message=message,
            provider=provider,
            context=context
        )
        
        self.logger.warning(f"Network error: {exception.to_dict()}")
        return exception
    
    def handle_parse_error(self, provider: str, content: str, 
                          original_error: Exception, context: Dict[str, Any] = None) -> EngineException:
        """Handle content parsing errors."""
        context = context or {}
        context.update({
            'content_length': len(content) if content else 0,
            'content_preview': content[:200] if content else 'no content',
            'original_error': str(original_error)
        })
        
        message = self.ERROR_MESSAGES[EngineErrorType.PARSE_ERROR].format(
            provider=provider,
            details=str(original_error)
        )
        
        exception = EngineException(
            error_type=EngineErrorType.PARSE_ERROR,
            message=message,
            provider=provider,
            context=context
        )
        
        self.logger.error(f"Parse error: {exception.to_dict()}")
        return exception
    
    def should_retry(self, exception: EngineException, attempt: int, max_attempts: int) -> bool:
        """Determine if an error should trigger a retry."""
        if not exception.is_retryable:
            return False
        
        if attempt >= max_attempts:
            return False
        
        # Special logic for rate limits - longer wait time
        if exception.error_type == EngineErrorType.RATE_LIMIT:
            return attempt < min(max_attempts, 2)  # Max 2 retries for rate limits
        
        return True
    
    def get_retry_delay(self, exception: EngineException, attempt: int) -> float:
        """Get appropriate retry delay based on error type and attempt number."""
        base_delay = 1.0
        
        # Error-specific delays
        if exception.error_type == EngineErrorType.RATE_LIMIT:
            base_delay = 60.0  # Wait 1 minute for rate limits
        elif exception.error_type == EngineErrorType.TIMEOUT:
            base_delay = 5.0   # Wait 5 seconds for timeouts
        elif exception.error_type == EngineErrorType.SERVER_ERROR:
            base_delay = 10.0  # Wait 10 seconds for server errors
        
        # Exponential backoff
        return base_delay * (2 ** (attempt - 1))


# Global error handler instance
error_handler = UnifiedErrorHandler()
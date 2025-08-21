"""
Optimized async session management with connection pooling.
Consolidates duplicate session management code across discovery engines.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Optional


class AsyncSessionManager:
    """Reusable async session manager with connection pooling and optimization."""
    
    _instances: Dict[str, 'AsyncSessionManager'] = {}
    _logger = logging.getLogger("discovery.session_manager")
    
    def __init__(self, name: str, timeout: int = 30, headers: Dict[str, str] = None, 
                 max_connections: int = 100, max_per_host: int = 30):
        self.name = name
        self.timeout = timeout
        self.headers = headers or {}
        self.max_connections = max_connections
        self.max_per_host = max_per_host
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
    
    @classmethod
    def get_instance(cls, name: str, **kwargs) -> 'AsyncSessionManager':
        """Get or create a named session manager instance."""
        if name not in cls._instances:
            cls._instances[name] = cls(name, **kwargs)
        return cls._instances[name]
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create optimized aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            async with self._lock:
                # Double-check pattern for thread safety
                if self._session is None or self._session.closed:
                    await self._create_session()
        return self._session
    
    async def _create_session(self):
        """Create optimized aiohttp session."""
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=10,  # Connection timeout
            sock_read=5   # Socket read timeout
        )
        
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_per_host,
            enable_cleanup_closed=True,
            keepalive_timeout=60,
            ttl_dns_cache=300  # DNS cache TTL
        )
        
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self.headers,
            connector=connector,
            raise_for_status=False  # Handle status codes manually
        )
        
        self._logger.debug(f"Created optimized session for {self.name}")
    
    async def close(self):
        """Close the session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            self._logger.debug(f"Closed session for {self.name}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return await self.get_session()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - keep session alive for reuse."""
        # Don't close session on context exit to allow reuse
        pass
    
    @classmethod
    async def close_all(cls):
        """Close all session manager instances."""
        for manager in cls._instances.values():
            await manager.close()
        cls._instances.clear()
        cls._logger.info("Closed all session managers")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, '_session') and self._session and not self._session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
            except:
                pass  # Ignore errors during cleanup


# Pre-configured session managers for common use cases
def get_news_api_session() -> AsyncSessionManager:
    """Get session manager for news API providers."""
    return AsyncSessionManager.get_instance(
        "news_api",
        timeout=30,
        headers={
            'User-Agent': 'CompetitiveIntel/2.0 (News Discovery Engine)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        }
    )


def get_rss_session() -> AsyncSessionManager:
    """Get session manager for RSS feed monitoring."""
    return AsyncSessionManager.get_instance(
        "rss_monitor", 
        timeout=20,
        headers={
            'User-Agent': 'CompetitiveIntel/2.0 (RSS Monitor)',
            'Accept': 'text/xml, application/xml, application/rss+xml, */*',
            'Accept-Encoding': 'gzip, deflate'
        }
    )


def get_web_scraper_session() -> AsyncSessionManager:
    """Get session manager for web scraping."""
    return AsyncSessionManager.get_instance(
        "web_scraper",
        timeout=25,
        headers={
            'User-Agent': 'Mozilla/5.0 (compatible; CompetitiveIntel/2.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
    )
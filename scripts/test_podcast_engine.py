#!/usr/bin/env python3
"""
Test script for Podcast Discovery Engine integration.

This script tests the podcast engine functionality without requiring actual API keys.
It validates the architecture, configuration, and integration patterns.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.discovery.engines import PodcastDiscoveryEngine, create_podcast_engine, SourceManager
from app.discovery.engines.base_engine import SourceType, ContentType


async def test_podcast_engine_architecture():
    """Test podcast engine architecture and configuration."""
    print("="*60)
    print("PODCAST DISCOVERY ENGINE - ARCHITECTURE TEST")
    print("="*60)
    
    # Test 1: Engine Creation
    print("\n1. Testing Engine Creation...")
    try:
        # Create engine with dummy credentials for architecture testing
        engine = create_podcast_engine(
            api_key="test_key",
            api_secret="test_secret",
            max_episodes_per_show=3,
            recency_threshold_days=14
        )
        print("   [PASS] Engine created successfully")
        print(f"   Engine name: {engine.engine_name}")
        print(f"   Source type: {engine.source_type}")
    except Exception as e:
        print(f"   [FAIL] Engine creation failed: {e}")
        return False
    
    # Test 2: Configuration Validation
    print("\n2. Testing Configuration...")
    try:
        config = engine.config
        print(f"   [PASS] Configuration loaded")
        print(f"   Max episodes per show: {engine.max_episodes_per_show}")
        print(f"   Recency threshold: {engine.recency_threshold_days} days")
        print(f"   Language filter: {engine.language_filter}")
    except Exception as e:
        print(f"   [FAIL] Configuration test failed: {e}")
        return False
    
    # Test 3: Engine Status
    print("\n3. Testing Engine Status...")
    try:
        status = await engine.get_engine_status()
        print("   [PASS] Engine status retrieved")
        print(f"   Engine available: {status['is_available']}")
        print(f"   Configuration keys: {list(status['configuration'].keys())}")
    except Exception as e:
        print(f"   [FAIL] Engine status test failed: {e}")
        return False
    
    # Test 4: Source Type Integration
    print("\n4. Testing Source Type Integration...")
    try:
        # Verify SourceType.PODCAST exists
        podcast_source_type = SourceType.PODCAST
        print("   [PASS] SourceType.PODCAST available")
        print(f"   Source type value: {podcast_source_type.value}")
        
        # Verify ContentType.PODCAST exists  
        podcast_content_type = ContentType.PODCAST
        print("   [PASS] ContentType.PODCAST available")
        print(f"   Content type value: {podcast_content_type.value}")
    except Exception as e:
        print(f"   [FAIL] Source type integration failed: {e}")
        return False
    
    # Test 5: Authentication Structure
    print("\n5. Testing Authentication Structure...")
    try:
        from app.discovery.engines.podcast_engine import PodcastIndexAuth
        auth = PodcastIndexAuth("test_key", "test_secret")
        headers = auth.generate_headers()
        
        required_headers = ['X-Auth-Date', 'X-Auth-Key', 'Authorization', 'User-Agent']
        for header in required_headers:
            if header not in headers:
                raise Exception(f"Missing required header: {header}")
        
        print("   [PASS] Authentication headers generated")
        print(f"   Headers: {list(headers.keys())}")
    except Exception as e:
        print(f"   [FAIL] Authentication test failed: {e}")
        return False
    
    # Cleanup
    await engine.close()
    print("\n   [PASS] Engine cleanup completed")
    
    return True


async def test_source_manager_integration():
    """Test podcast engine integration with source manager."""
    print("\n" + "="*60)
    print("SOURCE MANAGER INTEGRATION TEST")
    print("="*60)
    
    # Test 1: Source Manager with Podcast Engine
    print("\n1. Testing Source Manager with Podcast Configuration...")
    try:
        config = {
            "max_concurrent_engines": 2,
            "default_timeout": 30,
            
            # Podcast configuration
            "podcast_index": {
                "enabled": True,
                "api_key": "test_key",
                "api_secret": "test_secret",
                "max_episodes_per_show": 3,
                "recency_threshold_days": 14
            },
            
            # RSS configuration (for comparison)
            "rss_monitor": {
                "enabled": True,
                "max_feeds": 10
            }
        }
        
        source_manager = SourceManager(config)
        print("   [PASS] Source manager created with podcast engine")
        print(f"   Engines configured: {list(source_manager.engine_configs.keys())}")
        print(f"   Engine statuses: {dict(source_manager.engine_statuses)}")
        
    except Exception as e:
        print(f"   [FAIL] Source manager integration failed: {e}")
        return False
    
    # Test 2: Engine Availability
    print("\n2. Testing Engine Availability...")
    try:
        available_engines = []
        for name, status in source_manager.engine_statuses.items():
            if status.value in ['active', 'inactive']:  # Not error state
                available_engines.append(name)
        
        print(f"   [PASS] Available engines: {available_engines}")
        
        if 'podcast_index' in available_engines:
            print("   [PASS] Podcast engine available in source manager")
        else:
            print("   [WARN] Podcast engine not available (likely due to missing credentials)")
            
    except Exception as e:
        print(f"   [FAIL] Engine availability test failed: {e}")
        return False
    
    # Test 3: Engine Configuration
    print("\n3. Testing Engine Configuration...")
    try:
        if 'podcast_index' in source_manager.engine_configs:
            podcast_config = source_manager.engine_configs['podcast_index']
            print("   [PASS] Podcast engine configuration found")
            print(f"   Engine class: {podcast_config.engine_class.__name__}")
            print(f"   Priority: {podcast_config.priority}")
            print(f"   Weight: {podcast_config.weight}")
        else:
            print("   [FAIL] Podcast engine configuration not found")
            return False
            
    except Exception as e:
        print(f"   [FAIL] Engine configuration test failed: {e}")
        return False
    
    return True


async def test_content_structure():
    """Test podcast content structure and metadata."""
    print("\n" + "="*60)
    print("CONTENT STRUCTURE TEST")
    print("="*60)
    
    # Test 1: Mock Content Item Creation
    print("\n1. Testing Podcast Content Structure...")
    try:
        from app.discovery.engines.base_engine import DiscoveredItem
        from datetime import datetime
        
        # Create a mock podcast episode item
        mock_item = DiscoveredItem(
            title="AI Insights Episode 42: The Future of Machine Learning",
            url="https://example.com/episode.mp3",
            content="Podcast: AI Insights | Episode: The Future of Machine Learning | Description: Deep dive into ML trends",
            source_name="PodcastIndex: AI Insights Podcast",
            source_type=SourceType.PODCAST,
            content_type=ContentType.PODCAST,
            published_at=datetime.now(),
            author="John Smith",
            description="A comprehensive discussion about the future of machine learning technology",
            keywords=["AI", "machine learning", "technology", "future"],
            metadata={
                'podcast_id': 123456,
                'podcast_title': 'AI Insights Podcast',
                'podcast_author': 'John Smith',
                'episode_id': 789012,
                'episode_number': 42,
                'duration_seconds': 1800,
                'duration_formatted': '30m',
                'enclosure_url': 'https://example.com/episode.mp3',
                'enclosure_type': 'audio/mpeg',
                'podcast_categories': {'Technology': 'Tech'},
                'language': 'en',
                'explicit': False,
                'ready_for_transcription': True,
                'discovery_source': 'podcastindex'
            }
        )
        
        print("   [PASS] Mock podcast episode created")
        print(f"   Title: {mock_item.title}")
        print(f"   Source type: {mock_item.source_type.value}")
        print(f"   Content type: {mock_item.content_type.value}")
        print(f"   Duration: {mock_item.metadata['duration_formatted']}")
        print(f"   Ready for transcription: {mock_item.metadata['ready_for_transcription']}")
        
    except Exception as e:
        print(f"   [FAIL] Content structure test failed: {e}")
        return False
    
    # Test 2: Metadata Validation
    print("\n2. Testing Metadata Completeness...")
    try:
        required_fields = [
            'podcast_id', 'podcast_title', 'episode_id', 'duration_seconds',
            'enclosure_url', 'discovery_source', 'ready_for_transcription'
        ]
        
        missing_fields = [field for field in required_fields if field not in mock_item.metadata]
        
        if not missing_fields:
            print("   [PASS] All required metadata fields present")
        else:
            print(f"   [FAIL] Missing metadata fields: {missing_fields}")
            return False
            
    except Exception as e:
        print(f"   [FAIL] Metadata validation failed: {e}")
        return False
    
    return True


async def main():
    """Run all podcast engine tests."""
    print("PODCAST DISCOVERY ENGINE - COMPREHENSIVE ARCHITECTURE TEST")
    print("Testing integration without requiring live API credentials")
    print("=" * 80)
    
    test_results = []
    
    # Run architecture tests
    result1 = await test_podcast_engine_architecture()
    test_results.append(("Engine Architecture", result1))
    
    # Run source manager integration tests
    result2 = await test_source_manager_integration()
    test_results.append(("Source Manager Integration", result2))
    
    # Run content structure tests
    result3 = await test_content_structure()
    test_results.append(("Content Structure", result3))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n[SUCCESS] All podcast engine architecture tests passed!")
        print("The podcast discovery engine is ready for integration.")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Review implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
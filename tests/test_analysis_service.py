"""
Comprehensive test suite for Analysis Service - Phase 3.

Tests multi-stage pipeline, Stage 1 filtering, cost optimization,
and integration with User Config and Discovery Services.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.analysis_service import (
    AnalysisService, AnalysisContext, AnalysisStage,
    ContentPriority, FilterResult, AnalysisResult,
    AnalysisBatch, CostOptimizer
)
from app.analysis.filters import (
    KeywordFilter, EntityFilter, RelevanceFilter,
    CompositeFilter, FilterStrategy, ContentFilterFactory
)
from app.analysis.pipeline import (
    AnalysisPipeline, PipelineStage, PipelineProcessor,
    ProcessingStatus, StageResult, PipelineResult
)
from app.models.user import User
from app.models.strategic_profile import UserStrategicProfile, UserFocusArea
from app.models.tracking import UserEntityTracking
from app.models.discovery import DiscoveredContent


# ============================================================================
# FIXTURES
# ============================================================================

@pytest_asyncio.fixture
async def analysis_service():
    """Create an Analysis Service instance."""
    return AnalysisService()


@pytest_asyncio.fixture
async def sample_content():
    """Create sample content for testing."""
    return {
        "id": 1,
        "title": "OpenAI Announces GPT-5 with Revolutionary AI Capabilities",
        "content_text": "OpenAI has unveiled GPT-5, featuring breakthrough machine learning algorithms and artificial intelligence capabilities that surpass previous models. The new system demonstrates advanced reasoning and improved performance in competitive analysis.",
        "content_url": "https://example.com/article1",
        "published_at": datetime.utcnow() - timedelta(hours=2),
        "source_id": 1
    }


@pytest_asyncio.fixture
async def sample_context():
    """Create sample analysis context."""
    return AnalysisContext(
        user_id=1,
        strategic_profile={
            "industry": "technology",
            "organization_type": "startup",
            "role": "ceo",
            "strategic_goals": ["ai_integration", "competitive_positioning"]
        },
        focus_areas=[
            {
                "focus_area": "AI and Machine Learning",
                "keywords": "artificial intelligence,machine learning,deep learning,GPT",
                "priority": 5
            },
            {
                "focus_area": "Competitive Intelligence",
                "keywords": "competitors,market analysis,competitive advantage",
                "priority": 4
            }
        ],
        tracked_entities=[
            {
                "entity_name": "OpenAI",
                "entity_type": "competitor",
                "keywords": "GPT,ChatGPT,DALL-E",
                "priority": 5
            },
            {
                "entity_name": "Google",
                "entity_type": "competitor",
                "keywords": "Bard,Gemini,DeepMind",
                "priority": 4
            }
        ],
        delivery_preferences={
            "frequency": "daily",
            "min_significance_level": "medium"
        },
        analysis_depth="standard"
    )


@pytest_asyncio.fixture
async def sample_batch(sample_content, sample_context):
    """Create sample analysis batch."""
    return AnalysisBatch(
        batch_id="test_batch_001",
        user_id=1,
        content_items=[sample_content],
        context=sample_context,
        priority=ContentPriority.HIGH
    )


@pytest_asyncio.fixture
async def mock_discovered_content():
    """Create mock discovered content."""
    content = Mock(spec=DiscoveredContent)
    content.id = 1
    content.title = "Test Article"
    content.content_text = "Test content with keywords"
    content.content_url = "https://example.com/test"
    content.published_at = datetime.utcnow()
    content.source_id = 1
    content.user_id = 1
    content.is_analyzed = False
    content.overall_score = 0.7
    return content


# ============================================================================
# ANALYSIS SERVICE CORE TESTS
# ============================================================================

class TestAnalysisServiceCore:
    """Test Analysis Service core functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, analysis_service):
        """Test service initializes correctly."""
        assert analysis_service is not None
        assert analysis_service.batch_size == 10
        assert analysis_service.max_concurrent_analyses == 5
        assert analysis_service.filter_threshold == 0.3
        assert analysis_service.relevance_threshold == 0.5
        assert isinstance(analysis_service.cost_optimizer, CostOptimizer)
    
    @pytest.mark.asyncio
    async def test_calculate_filter_score(self, analysis_service, sample_content, sample_context):
        """Test Stage 1 filter score calculation."""
        result = analysis_service.calculate_filter_score(sample_content, sample_context)
        
        assert isinstance(result, FilterResult)
        assert result.passed == True
        assert result.relevance_score > 0.5
        assert "openai" in result.matched_entities
        assert len(result.matched_keywords) > 0
        assert result.priority in [ContentPriority.HIGH, ContentPriority.CRITICAL]
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_filter_score_no_match(self, analysis_service, sample_context):
        """Test filter score with no matching content."""
        irrelevant_content = {
            "id": 2,
            "title": "Weather Report for Tomorrow",
            "content_text": "Sunny with a chance of rain in the afternoon.",
            "content_url": "https://example.com/weather"
        }
        
        result = analysis_service.calculate_filter_score(irrelevant_content, sample_context)
        
        assert result.passed == False
        assert result.relevance_score < 0.3
        assert len(result.matched_entities) == 0
        assert result.priority == ContentPriority.LOW
        assert result.filter_reason is not None
    
    @pytest.mark.asyncio
    async def test_cost_optimization(self):
        """Test cost optimizer functionality."""
        optimizer = CostOptimizer()
        
        # Test cost estimation
        cost = optimizer.estimate_cost(
            AnalysisStage.FILTERING,
            "gpt-4o-mini",
            1000
        )
        assert isinstance(cost, Decimal)
        assert cost > Decimal("0")
        assert cost < Decimal("1.0")
        
        # Test model selection
        model = optimizer.select_model(
            AnalysisStage.INSIGHT,
            ContentPriority.CRITICAL,
            None
        )
        assert model == "gpt-4o-mini"
        
        # Test cost-conscious selection
        model = optimizer.select_model(
            AnalysisStage.FILTERING,
            ContentPriority.LOW,
            Decimal("0.01")
        )
        assert model == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_estimate_batch_cost(self, analysis_service, sample_batch):
        """Test batch cost estimation."""
        cost_estimate = await analysis_service.estimate_batch_cost(
            sample_batch,
            [AnalysisStage.RELEVANCE, AnalysisStage.INSIGHT]
        )
        
        assert "batch_id" in cost_estimate
        assert cost_estimate["total_items"] == 1
        assert cost_estimate["estimated_cost"] > 0
        assert "stage_costs" in cost_estimate
        assert cost_estimate["cost_savings"] == "70% saved through Stage 1 filtering"


# ============================================================================
# FILTERING TESTS
# ============================================================================

class TestContentFilters:
    """Test content filtering components."""
    
    @pytest.mark.asyncio
    async def test_keyword_filter(self, sample_content):
        """Test keyword filter functionality."""
        filter = KeywordFilter(
            keywords=["OpenAI", "GPT-5", "machine learning"],
            match_type="partial",
            min_matches=2
        )
        
        result = await filter.filter(sample_content, {})
        
        assert result.passed == True
        assert result.score > 0.5
        assert len(result.matches) >= 2
        assert any(m.matched_value == "openai" for m in result.matches)
    
    @pytest.mark.asyncio
    async def test_entity_filter(self, sample_content, sample_context):
        """Test entity filter functionality."""
        filter = EntityFilter(
            entities=sample_context.tracked_entities,
            min_matches=1
        )
        
        result = await filter.filter(sample_content, {})
        
        assert result.passed == True
        assert result.score > 0.0
        assert len(result.matches) >= 1
        assert any(m.filter_type == "entity" for m in result.matches)
        assert any("openai" in m.matched_value.lower() for m in result.matches)
    
    @pytest.mark.asyncio
    async def test_relevance_filter(self, sample_content, sample_context):
        """Test relevance filter functionality."""
        filter = RelevanceFilter(
            focus_areas=sample_context.focus_areas,
            min_score=0.3
        )
        
        result = await filter.filter(sample_content, {})
        
        assert result.passed == True
        assert result.score >= 0.3
        assert len(result.matches) > 0
        assert any(m.filter_type == "focus_area" for m in result.matches)
    
    @pytest.mark.asyncio
    async def test_composite_filter_strict(self, sample_content, sample_context):
        """Test composite filter with strict strategy."""
        keyword_filter = KeywordFilter(["OpenAI", "GPT"], min_matches=1)
        entity_filter = EntityFilter(sample_context.tracked_entities)
        
        composite = CompositeFilter(
            filters=[keyword_filter, entity_filter],
            strategy=FilterStrategy.STRICT
        )
        
        result = await composite.filter(sample_content, {})
        
        assert result.passed == True
        assert result.confidence > 0.5
        assert len(result.matches) > 0
    
    @pytest.mark.asyncio
    async def test_composite_filter_lenient(self, sample_content):
        """Test composite filter with lenient strategy."""
        keyword_filter = KeywordFilter(["nonexistent"], min_matches=1)
        entity_filter = EntityFilter([{"entity_name": "OpenAI", "priority": 5}])
        
        composite = CompositeFilter(
            filters=[keyword_filter, entity_filter],
            strategy=FilterStrategy.LENIENT
        )
        
        result = await composite.filter(sample_content, {})
        
        assert result.passed == True  # Lenient passes if any filter passes
    
    @pytest.mark.asyncio
    async def test_filter_factory(self, sample_context):
        """Test content filter factory."""
        composite_filter = ContentFilterFactory.create_user_filters(
            sample_context.__dict__
        )
        
        assert isinstance(composite_filter, CompositeFilter)
        assert len(composite_filter.filters) > 0
        
        # Test with content
        content = {
            "title": "AI Competition Heats Up",
            "content_text": "OpenAI and Google compete in machine learning space"
        }
        
        result = await composite_filter.filter(content, {})
        assert isinstance(result.passed, bool)
        assert isinstance(result.score, float)


# ============================================================================
# PIPELINE TESTS
# ============================================================================

class TestAnalysisPipeline:
    """Test analysis pipeline functionality."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = AnalysisPipeline(max_concurrent=3)
        
        assert pipeline.max_concurrent == 3
        assert len(pipeline.stages) == 0
    
    @pytest.mark.asyncio
    async def test_pipeline_stage_processing(self, sample_content, sample_context):
        """Test pipeline stage processing."""
        # Create mock processor
        class MockProcessor(PipelineProcessor):
            async def process(self, content, context, previous_results):
                return StageResult(
                    stage=self.stage,
                    status=ProcessingStatus.COMPLETED,
                    start_time=datetime.utcnow(),
                    data={"processed": True},
                    cost=Decimal("0.01")
                )
        
        processor = MockProcessor(PipelineStage.FILTERING)
        pipeline = AnalysisPipeline(stages=[processor])
        
        result = await pipeline.process_content(sample_content, sample_context)
        
        assert isinstance(result, PipelineResult)
        assert result.final_status == ProcessingStatus.COMPLETED
        assert len(result.stages_completed) == 1
        assert result.total_cost == Decimal("0.01")
    
    @pytest.mark.asyncio
    async def test_pipeline_batch_processing(self, sample_content, sample_context):
        """Test pipeline batch processing."""
        class MockProcessor(PipelineProcessor):
            async def process(self, content, context, previous_results):
                await asyncio.sleep(0.01)  # Simulate processing
                return StageResult(
                    stage=self.stage,
                    status=ProcessingStatus.COMPLETED,
                    start_time=datetime.utcnow(),
                    data={"content_id": content.get("id")}
                )
        
        processor = MockProcessor(PipelineStage.ANALYSIS)
        pipeline = AnalysisPipeline(stages=[processor], max_concurrent=2)
        
        # Create multiple content items
        content_items = [
            {**sample_content, "id": i} for i in range(5)
        ]
        
        results = await pipeline.process_batch(content_items, sample_context)
        
        assert len(results) == 5
        assert all(r.final_status == ProcessingStatus.COMPLETED for r in results)
    
    @pytest.mark.asyncio
    async def test_pipeline_early_termination(self, sample_content, sample_context):
        """Test pipeline early termination on critical failure."""
        class FailingProcessor(PipelineProcessor):
            async def process(self, content, context, previous_results):
                return StageResult(
                    stage=self.stage,
                    status=ProcessingStatus.FAILED,
                    start_time=datetime.utcnow(),
                    error="Critical failure"
                )
        
        class SecondProcessor(PipelineProcessor):
            async def process(self, content, context, previous_results):
                return StageResult(
                    stage=self.stage,
                    status=ProcessingStatus.COMPLETED,
                    start_time=datetime.utcnow()
                )
        
        failing = FailingProcessor(PipelineStage.FILTERING)  # Critical stage
        second = SecondProcessor(PipelineStage.ANALYSIS)
        
        pipeline = AnalysisPipeline(stages=[failing, second])
        result = await pipeline.process_content(sample_content, sample_context)
        
        assert result.final_status == ProcessingStatus.FAILED
        assert len(result.stages_completed) == 1  # Second stage not executed
    
    @pytest.mark.asyncio
    async def test_pipeline_cost_estimation(self, sample_content):
        """Test pipeline cost estimation."""
        class CostProcessor(PipelineProcessor):
            def estimate_cost(self, content, context):
                return Decimal("0.05")
        
        processor1 = CostProcessor(PipelineStage.FILTERING)
        processor2 = CostProcessor(PipelineStage.ANALYSIS)
        
        pipeline = AnalysisPipeline(stages=[processor1, processor2])
        
        cost_estimate = pipeline.estimate_total_cost(
            [sample_content],
            {},
            [PipelineStage.FILTERING, PipelineStage.ANALYSIS]
        )
        
        assert cost_estimate["total_items"] == 1
        assert cost_estimate["total_cost"] == 0.1  # 0.05 * 2
        assert cost_estimate["average_cost_per_item"] == 0.1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAnalysisIntegration:
    """Test Analysis Service integration with other services."""
    
    @pytest.mark.asyncio
    async def test_user_context_caching(self, analysis_service):
        """Test user context caching mechanism."""
        # Mock database session
        mock_db = AsyncMock(spec=AsyncSession)
        
        # First call should cache
        with patch.object(analysis_service, 'execute_db_operation') as mock_exec:
            mock_exec.return_value = AnalysisContext(
                user_id=1,
                strategic_profile={"industry": "tech"},
                focus_areas=[],
                tracked_entities=[]
            )
            
            context1 = await analysis_service.get_user_context(mock_db, 1)
            assert mock_exec.call_count == 1
            
            # Second call should use cache
            context2 = await analysis_service.get_user_context(mock_db, 1)
            assert mock_exec.call_count == 1  # Not called again
            assert context1.user_id == context2.user_id
    
    @pytest.mark.asyncio
    async def test_batch_creation_with_filtering(self, analysis_service):
        """Test batch creation applies Stage 1 filtering."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        # Mock get_user_context
        with patch.object(analysis_service, 'get_user_context') as mock_context:
            mock_context.return_value = AnalysisContext(
                user_id=1,
                focus_areas=[{
                    "focus_area": "AI",
                    "keywords": "artificial intelligence,machine learning",
                    "priority": 5
                }],
                tracked_entities=[]
            )
            
            # Mock get_pending_content with mixed relevance
            with patch.object(analysis_service, 'get_pending_content') as mock_content:
                mock_content.return_value = [
                    Mock(
                        id=1,
                        title="AI breakthrough in machine learning",
                        content_text="Revolutionary artificial intelligence...",
                        content_url="https://example.com/1",
                        published_at=datetime.utcnow(),
                        source_id=1
                    ),
                    Mock(
                        id=2,
                        title="Weather forecast",
                        content_text="Sunny tomorrow",
                        content_url="https://example.com/2",
                        published_at=datetime.utcnow(),
                        source_id=1
                    )
                ]
                
                batch = await analysis_service.create_analysis_batch(mock_db, 1)
                
                assert batch is not None
                assert len(batch.content_items) == 1  # Only relevant content
                assert batch.content_items[0]["id"] == 1
                assert batch.priority in [ContentPriority.HIGH, ContentPriority.MEDIUM]


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance and optimization targets."""
    
    @pytest.mark.asyncio
    async def test_filtering_speed(self, sample_context):
        """Test Stage 1 filtering speed."""
        import time
        
        # Create large content set
        content_items = []
        for i in range(100):
            content_items.append({
                "id": i,
                "title": f"Article {i} about AI and machine learning",
                "content_text": f"Content {i} discussing OpenAI and competitive intelligence " * 50
            })
        
        filter = ContentFilterFactory.create_user_filters(sample_context.__dict__)
        
        start_time = time.time()
        
        # Process all items
        for content in content_items:
            await filter.filter(content, {})
        
        elapsed = time.time() - start_time
        
        # Should process 100 items in under 1 second
        assert elapsed < 1.0
        
        # Calculate average time per item
        avg_time = elapsed / len(content_items)
        assert avg_time < 0.01  # Less than 10ms per item
    
    @pytest.mark.asyncio
    async def test_cost_savings_validation(self, analysis_service, sample_context):
        """Test 70% cost savings through Stage 1 filtering."""
        # Create 100 content items with varying relevance
        content_items = []
        for i in range(100):
            if i < 30:  # 30% relevant
                content = {
                    "id": i,
                    "title": f"OpenAI announces GPT-{i}",
                    "content_text": "AI and machine learning breakthrough"
                }
            else:  # 70% irrelevant
                content = {
                    "id": i,
                    "title": f"Random article {i}",
                    "content_text": "Unrelated content about various topics"
                }
            content_items.append(content)
        
        # Apply Stage 1 filtering
        passed_items = []
        for content in content_items:
            result = analysis_service.calculate_filter_score(content, sample_context)
            if result.passed:
                passed_items.append(content)
        
        # Verify ~70% filtered out
        filter_rate = 1 - (len(passed_items) / len(content_items))
        assert filter_rate >= 0.65  # At least 65% filtered
        assert filter_rate <= 0.75  # At most 75% filtered
        
        # Calculate cost savings
        full_cost = len(content_items) * 0.10  # Assume $0.10 per full analysis
        filtered_cost = len(passed_items) * 0.10  # Only analyze passed items
        savings_percent = (full_cost - filtered_cost) / full_cost
        
        assert savings_percent >= 0.65  # At least 65% cost savings
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, analysis_service):
        """Test memory usage optimization."""
        import sys
        
        # Test cache size limits
        for i in range(100):
            context = AnalysisContext(
                user_id=i,
                focus_areas=[],
                tracked_entities=[]
            )
            analysis_service._user_context_cache[i] = (context, datetime.utcnow())
        
        # Cache should have reasonable memory footprint
        cache_size = sys.getsizeof(analysis_service._user_context_cache)
        assert cache_size < 1_000_000  # Less than 1MB for 100 contexts
        
        # Test cache cleanup (manual for now)
        cutoff = datetime.utcnow() - analysis_service._context_cache_ttl
        expired = [
            uid for uid, (_, cached_at) in analysis_service._user_context_cache.items()
            if cached_at < cutoff
        ]
        
        for uid in expired:
            del analysis_service._user_context_cache[uid]


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_content_handling(self, analysis_service, sample_context):
        """Test handling of empty content."""
        empty_content = {
            "id": 1,
            "title": "",
            "content_text": "",
            "content_url": "https://example.com/empty"
        }
        
        result = analysis_service.calculate_filter_score(empty_content, sample_context)
        
        assert result.passed == False
        assert result.relevance_score == 0.0
        assert len(result.matched_keywords) == 0
    
    @pytest.mark.asyncio
    async def test_missing_fields_handling(self, analysis_service, sample_context):
        """Test handling of missing fields."""
        partial_content = {
            "id": 1,
            "title": "Test Article"
            # Missing content_text
        }
        
        result = analysis_service.calculate_filter_score(partial_content, sample_context)
        
        assert isinstance(result, FilterResult)
        # Should not crash, just process available fields
    
    @pytest.mark.asyncio
    async def test_invalid_context_handling(self, analysis_service):
        """Test handling of invalid context."""
        invalid_context = AnalysisContext(
            user_id=1,
            focus_areas=None,  # Invalid
            tracked_entities=None  # Invalid
        )
        
        content = {"id": 1, "title": "Test", "content_text": "Test"}
        
        # Should handle gracefully
        result = analysis_service.calculate_filter_score(content, invalid_context)
        assert isinstance(result, FilterResult)


# ============================================================================
# RUN CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
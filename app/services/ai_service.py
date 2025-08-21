"""
AI Provider Service for competitive intelligence analysis.

Provides unified interface for OpenAI GPT-4 and Anthropic Claude integration
with cost optimization, error handling, and performance monitoring.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union

import openai
import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.analysis import AnalysisResult
from app.analysis.core import (
    AnalysisStage, AIProvider, AnalysisContext, FilterResult, AIModelConfig, AIResponse,
    BaseAIProvider, AIProviderError, AIProviderRateLimitError, AIProviderTimeoutError,
    AIProviderManager, create_ai_provider_manager
)


logger = logging.getLogger(__name__)




class OpenAIProvider(BaseAIProvider):
    """Optimized OpenAI GPT-4 provider implementation."""
    
    def __init__(self, config: AIModelConfig, service_config=None):
        super().__init__(config, service_config)
        if not settings.OPENAI_API_KEY:
            raise AIProviderError("OpenAI API key not configured", AIProvider.OPENAI)
        
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def _make_request(
        self,
        prompt: str,
        content: str,
        context: AnalysisContext,
        stage: AnalysisStage
    ) -> AIResponse:
        """Make actual OpenAI API request."""
        start_time = time.time()
        
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"} if self.config.supports_json else None
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            cost_cents = self.estimate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            
            return AIResponse(
                content=response.choices[0].message.content,
                usage=usage,
                model=self.config.model_name,
                provider=AIProvider.OPENAI,
                cost_cents=cost_cents,
                processing_time_ms=processing_time_ms,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "stage": stage.value
                }
            )
            
        except openai.RateLimitError as e:
            raise AIProviderRateLimitError(f"OpenAI rate limit exceeded: {e}", AIProvider.OPENAI)
        except openai.APITimeoutError as e:
            raise AIProviderTimeoutError(f"OpenAI timeout: {e}", AIProvider.OPENAI, 30)
        except Exception as e:
            raise AIProviderError(f"OpenAI error: {e}", AIProvider.OPENAI)


class AnthropicProvider(BaseAIProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        # Note: anthropic package would need to be installed
        # For now, this is a placeholder implementation
        logger.warning("Anthropic provider not fully implemented - using mock responses")
    
    async def analyze_content(
        self,
        prompt: str,
        content: str,
        context: AnalysisContext,
        stage: AnalysisStage
    ) -> AIResponse:
        """Analyze content using Anthropic Claude (mock implementation)."""
        # Mock implementation for now
        await asyncio.sleep(0.1)  # Simulate API call
        
        mock_response = {
            "analysis": "Mock analysis result",
            "relevance_score": 0.75,
            "confidence": 0.8
        }
        
        usage = {
            "input_tokens": self.count_tokens(prompt + content),
            "output_tokens": self.count_tokens(json.dumps(mock_response)),
            "total_tokens": 0
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        
        return AIResponse(
            content=json.dumps(mock_response),
            usage=usage,
            model=self.config.model_name,
            provider=AIProvider.ANTHROPIC,
            cost_cents=self.estimate_cost(usage["input_tokens"], usage["output_tokens"]),
            processing_time_ms=100,
            metadata={"stage": stage.value, "mock": True}
        )
    
    async def extract_insights(
        self,
        content: str,
        context: AnalysisContext,
        analysis_results: Dict[str, Any]
    ) -> AIResponse:
        """Extract strategic insights using Anthropic Claude (mock implementation)."""
        return await self.analyze_content(
            "Extract insights",
            content,
            context,
            AnalysisStage.INSIGHT_EXTRACTION
        )


class MockAIProvider(BaseAIProvider):
    """Mock AI provider for testing."""
    
    async def analyze_content(
        self,
        prompt: str,
        content: str,
        context: AnalysisContext,
        stage: AnalysisStage
    ) -> AIResponse:
        """Mock analysis for testing."""
        await asyncio.sleep(0.01)  # Simulate minimal processing time
        
        # Generate different responses based on stage
        if stage == AnalysisStage.FILTERING:
            mock_response = {
                "filter_passed": True,
                "filter_score": 0.85,
                "filter_priority": "high",
                "matched_keywords": ["AI", "strategic"],
                "matched_entities": ["OpenAI"],
                "filter_reason": "High relevance to strategic goals"
            }
        elif stage == AnalysisStage.RELEVANCE_ANALYSIS:
            mock_response = {
                "relevance_score": 0.78,
                "strategic_alignment": 0.82,
                "competitive_impact": 0.75,
                "urgency_score": 0.60
            }
        elif stage == AnalysisStage.INSIGHT_EXTRACTION:
            mock_response = {
                "key_insights": [
                    "Strategic AI partnership opportunity identified",
                    "Competitive advantage through early adoption"
                ],
                "action_items": [
                    "Evaluate partnership opportunities",
                    "Assess internal AI capabilities"
                ],
                "strategic_implications": [
                    "Market positioning improvement",
                    "Technology advancement acceleration"
                ],
                "risk_assessment": {
                    "level": "medium",
                    "factors": ["Technology dependency", "Market timing"]
                },
                "opportunity_assessment": {
                    "level": "high",
                    "factors": ["First mover advantage", "Strategic partnerships"]
                },
                "confidence_reasoning": "Strong alignment with stated strategic goals"
            }
        else:  # SUMMARY_GENERATION
            mock_response = {
                "executive_summary": "Strategic AI opportunity with high potential impact",
                "detailed_analysis": "Comprehensive analysis shows significant strategic value...",
                "confidence_reasoning": "High confidence based on multiple validation factors"
            }
        
        usage = {
            "input_tokens": self.count_tokens(prompt + content),
            "output_tokens": self.count_tokens(json.dumps(mock_response)),
            "total_tokens": 0
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        
        return AIResponse(
            content=json.dumps(mock_response),
            usage=usage,
            model=self.config.model_name,
            provider=AIProvider.MOCK,
            cost_cents=self.estimate_cost(usage["input_tokens"], usage["output_tokens"]),
            processing_time_ms=10,
            metadata={"stage": stage.value, "mock": True}
        )
    
    async def extract_insights(
        self,
        content: str,
        context: AnalysisContext,
        analysis_results: Dict[str, Any]
    ) -> AIResponse:
        """Mock insight extraction for testing."""
        return await self.analyze_content(
            "Extract insights",
            content,
            context,
            AnalysisStage.INSIGHT_EXTRACTION
        )


# Legacy compatibility class for existing code
class AIService:
    """Legacy AI service wrapper using centralized AI provider manager."""
    
    def __init__(self):
        self.ai_manager = create_ai_provider_manager()
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize AI provider manager."""
        await self.ai_manager.initialize()
    
    def get_provider(self, provider: AIProvider) -> BaseAIProvider:
        """Get AI provider instance."""
        provider_instance = self.ai_manager.providers.get(provider)
        if not provider_instance:
            raise AIProviderError(f"Provider {provider.value} not available", provider)
        return provider_instance
    
    def get_optimal_provider(
        self,
        priority: str,
        budget_cents: Optional[int] = None
    ) -> AIProvider:
        """Get optimal provider based on priority and budget."""
        return self.ai_manager.selection_strategy.select_optimal_provider(
            priority, AnalysisStage.RELEVANCE_ANALYSIS, budget_cents
        )
    
    async def analyze_content(
        self,
        content: str,
        context: AnalysisContext,
        stage: AnalysisStage,
        provider: Optional[AIProvider] = None,
        budget_cents: Optional[int] = None
    ) -> AIResponse:
        """Analyze content using optimal AI provider."""
        return await self.ai_manager.analyze_content(
            content, context, stage, provider, budget_cents
        )
    
    async def extract_insights(
        self,
        content: str,
        context: AnalysisContext,
        analysis_results: Dict[str, Any],
        provider: Optional[AIProvider] = None
    ) -> AIResponse:
        """Extract strategic insights from analyzed content."""
        # Use insight extraction stage
        return await self.ai_manager.analyze_content(
            content, context, AnalysisStage.INSIGHT_EXTRACTION, provider
        )
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return self.ai_manager.get_provider_stats()


# Global AI service instance with lazy initialization
class LazyAIService:
    """Lazy-initialized AI service singleton."""
    
    def __init__(self):
        self._service = None
    
    async def _ensure_initialized(self):
        """Ensure AI service is initialized."""
        if self._service is None:
            self._service = AIService()
            await self._service.initialize()
        return self._service
    
    async def analyze_content(self, *args, **kwargs):
        service = await self._ensure_initialized()
        return await service.analyze_content(*args, **kwargs)
    
    async def extract_insights(self, *args, **kwargs):
        service = await self._ensure_initialized()
        return await service.extract_insights(*args, **kwargs)
    
    def get_optimal_provider(self, *args, **kwargs):
        if self._service is None:
            # Fallback for synchronous calls
            return AIProvider.MOCK
        return self._service.get_optimal_provider(*args, **kwargs)


# Global AI service instance
ai_service = LazyAIService()
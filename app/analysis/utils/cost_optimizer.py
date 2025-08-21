"""
Cost optimization utilities for Analysis Service.

Centralized cost calculation and model selection logic.
"""

from decimal import Decimal
from typing import Dict, Optional

from .common_types import AnalysisStage, ContentPriority


class ModelSelector:
    """AI model selection based on requirements and constraints."""
    
    # Model capabilities and costs
    MODELS = {
        "gpt-4o-mini": {
            "input_cost": 0.00015,      # per 1K tokens
            "output_cost": 0.0006,      # per 1K tokens
            "quality": 0.85,            # Quality score
            "speed": 0.95,              # Speed score
            "best_for": ["filtering", "relevance", "general"]
        },
        "gpt-3.5-turbo": {
            "input_cost": 0.0005,
            "output_cost": 0.0015,
            "quality": 0.75,
            "speed": 0.90,
            "best_for": ["filtering", "quick_analysis"]
        },
        "gpt-4": {
            "input_cost": 0.03,
            "output_cost": 0.06,
            "quality": 0.95,
            "speed": 0.60,
            "best_for": ["insight", "summary", "complex_analysis"]
        },
        "gpt-4-turbo": {
            "input_cost": 0.01,
            "output_cost": 0.03,
            "quality": 0.90,
            "speed": 0.80,
            "best_for": ["insight", "summary"]
        }
    }
    
    @classmethod
    def select_optimal_model(
        cls,
        stage: AnalysisStage,
        priority: ContentPriority,
        cost_limit: Optional[Decimal] = None,
        quality_requirement: float = 0.7
    ) -> str:
        """Select optimal model based on requirements."""
        stage_name = stage.value.lower()
        
        # High priority or critical content preferences
        if priority in [ContentPriority.CRITICAL, ContentPriority.HIGH]:
            if stage in [AnalysisStage.INSIGHT, AnalysisStage.SUMMARY]:
                if cost_limit is None or cost_limit > Decimal("0.10"):
                    return "gpt-4o-mini"
                    
        # Cost-conscious selection
        if cost_limit and cost_limit < Decimal("0.05"):
            return "gpt-3.5-turbo"
            
        # Stage-specific selection
        if stage in [AnalysisStage.FILTERING, AnalysisStage.RELEVANCE]:
            return "gpt-4o-mini"  # Fast and cost-effective
        elif stage in [AnalysisStage.INSIGHT, AnalysisStage.SUMMARY]:
            return "gpt-4o-mini"  # Better for complex tasks
            
        # Default selection
        return "gpt-4o-mini"
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict:
        """Get model information."""
        return cls.MODELS.get(model_name, cls.MODELS["gpt-4o-mini"])


class CostOptimizer:
    """Cost optimization for AI analysis pipeline."""
    
    # Token estimates per operation type
    TOKEN_ESTIMATES = {
        "filtering": {"input": 500, "output": 100},
        "relevance": {"input": 1000, "output": 200},
        "insight": {"input": 1500, "output": 500},
        "summary": {"input": 2000, "output": 800},
        "preprocessing": {"input": 300, "output": 50},
        "enrichment": {"input": 800, "output": 150},
        "analysis": {"input": 1200, "output": 400},
        "scoring": {"input": 600, "output": 100},
        "insight_extraction": {"input": 1800, "output": 600},
        "summarization": {"input": 2500, "output": 1000},
        "postprocessing": {"input": 400, "output": 100}
    }
    
    def __init__(self):
        self.model_selector = ModelSelector()
    
    # Compatibility methods for existing tests
    def estimate_cost(
        self,
        stage: AnalysisStage,
        model: str = "gpt-4o-mini",
        content_length: int = 1000
    ) -> Decimal:
        """Compatibility method for estimate_stage_cost."""
        return self.estimate_stage_cost(stage, model, content_length)
    
    def select_model(
        self,
        stage: AnalysisStage,
        priority: ContentPriority,
        cost_limit: Optional[Decimal] = None
    ) -> str:
        """Compatibility method for model selection."""
        return self.model_selector.select_optimal_model(stage, priority, cost_limit)
        
    def estimate_stage_cost(
        self,
        stage: AnalysisStage,
        model: str = "gpt-4o-mini",
        content_length: int = 1000,
        custom_tokens: Optional[Dict[str, int]] = None
    ) -> Decimal:
        """Estimate cost for a specific analysis stage."""
        if model not in self.model_selector.MODELS:
            model = "gpt-4o-mini"
            
        stage_name = stage.value
        if stage_name not in self.TOKEN_ESTIMATES:
            return Decimal("0.01")  # Default estimate
            
        # Use custom tokens if provided, otherwise use estimates
        tokens = custom_tokens or self.TOKEN_ESTIMATES[stage_name]
        
        # Adjust tokens based on content length
        length_multiplier = max(1, content_length / 1000)
        
        input_tokens = tokens["input"] * length_multiplier
        output_tokens = tokens["output"]
        
        model_info = self.model_selector.get_model_info(model)
        input_cost = (input_tokens / 1000) * model_info["input_cost"]
        output_cost = (output_tokens / 1000) * model_info["output_cost"]
        
        return Decimal(str(round(input_cost + output_cost, 6)))
    
    def estimate_content_cost(
        self,
        stages: list,
        content_length: int = 1000,
        priority: ContentPriority = ContentPriority.MEDIUM,
        cost_limit: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """Estimate total cost for processing content through multiple stages."""
        total_cost = Decimal("0.00")
        stage_costs = {}
        
        for stage in stages:
            # Select optimal model for this stage
            model = self.model_selector.select_optimal_model(
                stage, priority, cost_limit
            )
            
            # Calculate stage cost
            stage_cost = self.estimate_stage_cost(stage, model, content_length)
            stage_costs[f"{stage.value}_{model}"] = stage_cost
            total_cost += stage_cost
            
        stage_costs["total"] = total_cost
        return stage_costs
    
    def optimize_batch_processing(
        self,
        content_items: list,
        stages: list,
        budget: Optional[Decimal] = None
    ) -> Dict[str, any]:
        """Optimize batch processing within budget constraints."""
        if not budget:
            budget = Decimal("10.00")  # Default budget
            
        # Calculate costs for different scenarios
        scenarios = {
            "high_quality": [],
            "balanced": [],
            "cost_effective": []
        }
        
        for content in content_items:
            content_length = len(content.get("content_text", ""))
            
            # High quality scenario
            hq_cost = self.estimate_content_cost(
                stages, content_length, ContentPriority.HIGH
            )["total"]
            scenarios["high_quality"].append(hq_cost)
            
            # Balanced scenario
            bal_cost = self.estimate_content_cost(
                stages, content_length, ContentPriority.MEDIUM
            )["total"]
            scenarios["balanced"].append(bal_cost)
            
            # Cost-effective scenario
            ce_cost = self.estimate_content_cost(
                stages, content_length, ContentPriority.LOW
            )["total"]
            scenarios["cost_effective"].append(ce_cost)
            
        # Select best scenario within budget
        total_costs = {
            scenario: sum(costs) 
            for scenario, costs in scenarios.items()
        }
        
        best_scenario = "cost_effective"
        for scenario, total_cost in total_costs.items():
            if total_cost <= budget:
                best_scenario = scenario
                break
                
        return {
            "recommended_scenario": best_scenario,
            "total_costs": total_costs,
            "budget": float(budget),
            "items_processed": len(content_items),
            "cost_per_item": float(total_costs[best_scenario] / len(content_items))
        }
    
    def calculate_savings(
        self,
        total_items: int,
        filtered_items: int,
        cost_per_item: Decimal
    ) -> Dict[str, any]:
        """Calculate cost savings from filtering."""
        full_cost = total_items * cost_per_item
        filtered_cost = filtered_items * cost_per_item
        savings = full_cost - filtered_cost
        savings_percent = (savings / full_cost * 100) if full_cost > 0 else 0
        
        return {
            "total_items": total_items,
            "filtered_items": filtered_items,
            "items_saved": total_items - filtered_items,
            "full_cost": float(full_cost),
            "filtered_cost": float(filtered_cost),
            "savings": float(savings),
            "savings_percent": float(savings_percent)
        }
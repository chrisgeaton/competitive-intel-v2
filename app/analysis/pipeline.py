"""
Analysis pipeline for multi-stage content processing.

Implements a flexible, cost-optimized pipeline for content analysis
with support for parallel processing and stage-based filtering.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession

# Import from centralized utilities
from .utils import (
    PipelineStage, ProcessingStatus, StageResult, PipelineResult,
    get_batch_processor
)


# Type definitions moved to centralized utilities


class PipelineProcessor:
    """Base processor for pipeline stages."""
    
    def __init__(self, stage: PipelineStage, logger: Optional[logging.Logger] = None):
        self.stage = stage
        self.logger = logger or logging.getLogger(__name__)
        
    async def process(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any],
        previous_results: List[StageResult]
    ) -> StageResult:
        """Process content through this stage."""
        raise NotImplementedError("Subclasses must implement process method")
        
    async def should_process(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any],
        previous_results: List[StageResult]
    ) -> bool:
        """Determine if this stage should process the content."""
        return True
        
    def estimate_cost(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Decimal:
        """Estimate processing cost for this stage."""
        return Decimal("0.00")


class AnalysisPipeline:
    """
    Multi-stage analysis pipeline with cost optimization.
    
    Features:
    - Configurable processing stages
    - Parallel processing support
    - Cost tracking and optimization
    - Stage-based filtering and early termination
    - Error handling and retry logic
    """
    
    def __init__(
        self,
        stages: Optional[List[PipelineProcessor]] = None,
        max_concurrent: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.stages = stages or []
        self.max_concurrent = max_concurrent
        self._stage_map: Dict[PipelineStage, PipelineProcessor] = {}
        
        # Get centralized batch processor
        self.batch_processor = get_batch_processor()
        
        # Build stage map
        for stage_processor in self.stages:
            self._stage_map[stage_processor.stage] = stage_processor
            
    def add_stage(self, processor: PipelineProcessor):
        """Add a processing stage to the pipeline."""
        self.stages.append(processor)
        self._stage_map[processor.stage] = processor
        
    def remove_stage(self, stage: PipelineStage):
        """Remove a processing stage from the pipeline."""
        if stage in self._stage_map:
            processor = self._stage_map[stage]
            self.stages.remove(processor)
            del self._stage_map[stage]
            
    async def process_content(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any],
        stages_to_run: Optional[List[PipelineStage]] = None,
        pipeline_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Process content through the pipeline.
        
        Args:
            content: Content to process
            context: Processing context (user preferences, etc.)
            stages_to_run: Specific stages to run (None = all stages)
            pipeline_id: Optional pipeline ID for tracking
            
        Returns:
            PipelineResult with all stage results
        """
        import hashlib
        
        # Generate pipeline ID if not provided
        if not pipeline_id:
            from datetime import datetime
            pipeline_id = hashlib.md5(
                f"{content.get('id', '')}-{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            
        # Initialize result
        result = PipelineResult(
            pipeline_id=pipeline_id,
            content_id=content.get("id", 0),
            user_id=context.get("user_id", 0)
        )
        
        # Determine which stages to run
        stages = self.stages
        if stages_to_run:
            stages = [s for s in self.stages if s.stage in stages_to_run]
            
        # Process through each stage
        for stage_processor in stages:
            try:
                # Check if stage should process
                should_process = await stage_processor.should_process(
                    content,
                    context,
                    result.stages_completed
                )
                
                if not should_process:
                    self.logger.info(
                        f"Skipping stage {stage_processor.stage.value} for content {content.get('id')}"
                    )
                    stage_result = StageResult(
                        stage=stage_processor.stage,
                        status=ProcessingStatus.SKIPPED,
                        start_time=datetime.now()
                    )
                    result.add_stage_result(stage_result)
                    continue
                    
                # Process stage
                self.logger.info(
                    f"Processing stage {stage_processor.stage.value} for content {content.get('id')}"
                )
                
                start_time = time.time()
                stage_result = await stage_processor.process(
                    content,
                    context,
                    result.stages_completed
                )
                
                # Update timing
                stage_result.processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Add to results
                result.add_stage_result(stage_result)
                
                # Check for early termination
                if stage_result.status == ProcessingStatus.FAILED:
                    self.logger.warning(
                        f"Stage {stage_processor.stage.value} failed: {stage_result.error}"
                    )
                    if self._should_terminate_on_failure(stage_processor.stage):
                        result.final_status = ProcessingStatus.FAILED
                        break
                        
            except Exception as e:
                self.logger.error(
                    f"Error in stage {stage_processor.stage.value}: {str(e)}"
                )
                stage_result = StageResult(
                    stage=stage_processor.stage,
                    status=ProcessingStatus.FAILED,
                    start_time=datetime.now(),
                    error=str(e)
                )
                result.add_stage_result(stage_result)
                
                if self._should_terminate_on_failure(stage_processor.stage):
                    result.final_status = ProcessingStatus.FAILED
                    break
                    
        # Set final status
        if result.final_status != ProcessingStatus.FAILED:
            result.final_status = ProcessingStatus.COMPLETED
            
        result.completed_at = datetime.now()
        
        self.logger.info(
            f"Pipeline {pipeline_id} completed: "
            f"stages={len(result.stages_completed)}, "
            f"time={result.total_processing_time_ms}ms, "
            f"cost=${result.total_cost}"
        )
        
        return result
        
    async def process_batch(
        self,
        content_items: List[Dict[str, Any]],
        context: Dict[str, Any],
        stages_to_run: Optional[List[PipelineStage]] = None
    ) -> List[PipelineResult]:
        """
        Process multiple content items through the pipeline.
        
        Uses concurrent processing up to max_concurrent limit.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_limit(content):
            async with semaphore:
                return await self.process_content(
                    content,
                    context,
                    stages_to_run
                )
                
        # Process all items concurrently
        tasks = [process_with_limit(content) for content in content_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Failed to process content item {i}: {str(result)}"
                )
            else:
                processed_results.append(result)
                
        return processed_results
        
    def estimate_total_cost(
        self,
        content_items: List[Dict[str, Any]],
        context: Dict[str, Any],
        stages_to_run: Optional[List[PipelineStage]] = None
    ) -> Dict[str, Any]:
        """
        Estimate total cost for processing content items.
        """
        stages = self.stages
        if stages_to_run:
            stages = [s for s in self.stages if s.stage in stages_to_run]
            
        total_cost = Decimal("0.00")
        stage_costs = {}
        
        for stage_processor in stages:
            stage_cost = Decimal("0.00")
            for content in content_items:
                stage_cost += stage_processor.estimate_cost(content, context)
            stage_costs[stage_processor.stage.value] = float(stage_cost)
            total_cost += stage_cost
            
        return {
            "total_items": len(content_items),
            "total_cost": float(total_cost),
            "average_cost_per_item": float(total_cost / len(content_items)) if content_items else 0,
            "stage_costs": stage_costs,
            "stages_to_run": [s.value for s in stages_to_run] if stages_to_run else "all"
        }
        
    def _should_terminate_on_failure(self, stage: PipelineStage) -> bool:
        """
        Determine if pipeline should terminate on stage failure.
        
        Critical stages like filtering should terminate the pipeline.
        """
        critical_stages = [
            PipelineStage.PREPROCESSING,
            PipelineStage.FILTERING
        ]
        return stage in critical_stages
        
    async def get_pipeline_stats(
        self,
        db: AsyncSession,
        user_id: Optional[int] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get pipeline processing statistics.
        """
        # This would query from a pipeline_results table
        stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "average_processing_time_ms": 0,
            "total_cost": 0.0,
            "stage_performance": {},
            "time_range_hours": time_range_hours
        }
        
        if user_id:
            stats["user_id"] = user_id
            
        for stage in PipelineStage:
            stats["stage_performance"][stage.value] = {
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "average_time_ms": 0,
                "total_cost": 0.0
            }
            
        return stats
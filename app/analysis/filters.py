"""
Stage 1 content filtering for cost optimization.

Implements efficient local filtering to eliminate irrelevant content
before expensive AI analysis, achieving 70% cost savings.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

# Import from centralized utilities
from .utils import (
    FilterStrategy, FilterMatch, FilterResult,
    get_text_processor, get_cache_manager
)


# Type definitions moved to centralized utilities


class ContentFilter(ABC):
    """Abstract base class for content filters."""
    
    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        required: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.logger = logger or logging.getLogger(__name__)
        
        # Use centralized text processor
        self.text_processor = get_text_processor()
        
    @abstractmethod
    async def filter(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> FilterResult:
        """Apply filter to content."""
        pass
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text using centralized processor."""
        return self.text_processor.preprocess_text(text, normalize=True)


class KeywordFilter(ContentFilter):
    """
    Filter based on keyword matching.
    
    Supports exact match, partial match, and regex patterns.
    """
    
    def __init__(
        self,
        keywords: List[str],
        match_type: str = "partial",  # 'exact', 'partial', 'regex'
        min_matches: int = 1,
        case_sensitive: bool = False,
        **kwargs
    ):
        super().__init__(name="keyword_filter", **kwargs)
        self.keywords = keywords
        self.match_type = match_type
        self.min_matches = min_matches
        self.case_sensitive = case_sensitive
        
        # Compile regex patterns if needed
        if match_type == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            self.patterns = [re.compile(kw, flags) for kw in keywords]
        else:
            self.patterns = []
            
    async def filter(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> FilterResult:
        """Apply keyword filter to content."""
        import time
        start_time = time.time()
        
        result = FilterResult(passed=False, score=0.0)
        
        # Extract text to search
        title = content.get("title", "")
        text = content.get("content_text", "")
        
        # Handle None values
        title = str(title) if title is not None else ""
        text = str(text) if text is not None else ""
        
        if not self.case_sensitive:
            title = title.lower()
            text = text.lower()
            
        full_text = f"{title} {text}"
        
        # Track matches
        matched_keywords = []
        
        for keyword in self.keywords:
            if not self.case_sensitive and self.match_type != "regex":
                keyword = keyword.lower()
                
            # Check for match based on type
            matched = False
            match_location = None
            
            if self.match_type == "exact":
                # Look for exact word match
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, title, re.IGNORECASE if not self.case_sensitive else 0):
                    matched = True
                    match_location = "title"
                elif re.search(pattern, text, re.IGNORECASE if not self.case_sensitive else 0):
                    matched = True
                    match_location = "content"
                    
            elif self.match_type == "partial":
                # Look for partial match
                if keyword in title:
                    matched = True
                    match_location = "title"
                elif keyword in text:
                    matched = True
                    match_location = "content"
                    
            elif self.match_type == "regex":
                # Use compiled regex pattern
                pattern = self.patterns[self.keywords.index(keyword)]
                if pattern.search(title):
                    matched = True
                    match_location = "title"
                elif pattern.search(text):
                    matched = True
                    match_location = "content"
                    
            if matched:
                matched_keywords.append(keyword)
                
                # Extract context around match
                context_text = None
                if match_location == "title":
                    context_text = title[:100]
                else:
                    # Find position in text and extract surrounding context
                    pos = text.find(keyword if not self.case_sensitive else keyword)
                    if pos >= 0:
                        start = max(0, pos - 50)
                        end = min(len(text), pos + len(keyword) + 50)
                        context_text = text[start:end]
                        
                result.add_match(FilterMatch(
                    filter_type="keyword",
                    matched_value=keyword,
                    match_location=match_location,
                    match_strength=1.0 if match_location == "title" else 0.7,
                    context=context_text
                ))
                
        # Calculate score
        if matched_keywords:
            result.score = len(matched_keywords) / len(self.keywords)
            result.passed = len(matched_keywords) >= self.min_matches
            result.confidence = min(1.0, len(matched_keywords) / self.min_matches)
        else:
            result.add_failure_reason(f"No keywords matched (required: {self.min_matches})")
            
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        return result


class EntityFilter(ContentFilter):
    """
    Filter based on entity matching (competitors, people, organizations, etc.).
    """
    
    def __init__(
        self,
        entities: List[Dict[str, Any]],
        min_matches: int = 1,
        check_aliases: bool = True,
        **kwargs
    ):
        super().__init__(name="entity_filter", **kwargs)
        self.entities = entities
        self.min_matches = min_matches
        self.check_aliases = check_aliases
        
        # Build entity lookup for efficient matching
        self._build_entity_lookup()
        
    def _build_entity_lookup(self):
        """Build lookup structures for efficient entity matching."""
        self.entity_names = set()
        self.entity_keywords = {}
        self.entity_priorities = {}
        
        for entity in self.entities:
            name = entity.get("entity_name", "").lower()
            if name:
                self.entity_names.add(name)
                self.entity_priorities[name] = entity.get("priority", 3)
                
                # Add keywords for this entity
                keywords = entity.get("keywords", "").lower().split(",")
                self.entity_keywords[name] = [kw.strip() for kw in keywords if kw.strip()]
                
                # Add aliases if available
                if self.check_aliases:
                    aliases = entity.get("aliases", "").lower().split(",")
                    for alias in aliases:
                        alias = alias.strip()
                        if alias:
                            self.entity_names.add(alias)
                            
    async def filter(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> FilterResult:
        """Apply entity filter to content."""
        import time
        start_time = time.time()
        
        result = FilterResult(passed=False, score=0.0)
        
        # Extract text to search
        title = self.preprocess_text(content.get("title", ""))
        text = self.preprocess_text(content.get("content_text", ""))
        full_text = f"{title} {text}"
        
        # Track matched entities
        matched_entities = []
        total_score = 0.0
        
        for entity_name in self.entity_names:
            if entity_name in full_text:
                # Determine match location
                match_location = "title" if entity_name in title else "content"
                match_strength = 1.0 if match_location == "title" else 0.7
                
                # Apply priority weighting
                priority = self.entity_priorities.get(entity_name, 3)
                match_strength *= (priority / 5.0)
                
                matched_entities.append(entity_name)
                total_score += match_strength
                
                # Extract context
                pos = full_text.find(entity_name)
                context_text = None
                if pos >= 0:
                    start = max(0, pos - 50)
                    end = min(len(full_text), pos + len(entity_name) + 50)
                    context_text = full_text[start:end]
                    
                result.add_match(FilterMatch(
                    filter_type="entity",
                    matched_value=entity_name,
                    match_location=match_location,
                    match_strength=match_strength,
                    context=context_text
                ))
                
                # Check for entity keywords
                if entity_name in self.entity_keywords:
                    for keyword in self.entity_keywords[entity_name]:
                        if keyword and keyword in full_text:
                            result.add_match(FilterMatch(
                                filter_type="entity_keyword",
                                matched_value=f"{entity_name}:{keyword}",
                                match_location="content",
                                match_strength=0.5,
                                context=None
                            ))
                            total_score += 0.2
                            
        # Calculate final score
        if matched_entities:
            # Normalize score
            result.score = min(1.0, total_score / max(1, len(self.entity_names)))
            result.passed = len(matched_entities) >= self.min_matches
            result.confidence = min(1.0, len(matched_entities) / self.min_matches)
        else:
            result.add_failure_reason(f"No entities matched (required: {self.min_matches})")
            
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        return result


class RelevanceFilter(ContentFilter):
    """
    Filter based on content relevance to user focus areas.
    """
    
    def __init__(
        self,
        focus_areas: List[Dict[str, Any]],
        min_score: float = 0.3,
        **kwargs
    ):
        super().__init__(name="relevance_filter", **kwargs)
        self.focus_areas = focus_areas
        self.min_score = min_score
        
        # Build focus area lookup
        self._build_focus_lookup()
        
    def _build_focus_lookup(self):
        """Build lookup structures for focus area matching."""
        self.focus_keywords = {}
        self.focus_priorities = {}
        
        for focus in self.focus_areas:
            area = focus.get("focus_area", "").lower()
            if area:
                keywords = focus.get("keywords", "").lower().split(",")
                self.focus_keywords[area] = [kw.strip() for kw in keywords if kw.strip()]
                self.focus_priorities[area] = focus.get("priority", 3)
                
    async def filter(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> FilterResult:
        """Apply relevance filter to content."""
        import time
        start_time = time.time()
        
        result = FilterResult(passed=False, score=0.0)
        
        # Extract text to search
        title = self.preprocess_text(content.get("title", ""))
        text = self.preprocess_text(content.get("content_text", ""))
        full_text = f"{title} {text}"
        
        # Calculate relevance for each focus area
        total_relevance = 0.0
        matched_areas = []
        
        for area, keywords in self.focus_keywords.items():
            area_score = 0.0
            area_matches = []
            
            for keyword in keywords:
                if keyword in full_text:
                    # Higher score for title matches
                    if keyword in title:
                        area_score += 2.0
                        match_location = "title"
                    else:
                        area_score += 1.0
                        match_location = "content"
                        
                    area_matches.append(keyword)
                    
                    result.add_match(FilterMatch(
                        filter_type="focus_area",
                        matched_value=f"{area}:{keyword}",
                        match_location=match_location,
                        match_strength=0.8,
                        context=None
                    ))
                    
            if area_matches:
                # Apply priority weighting
                priority = self.focus_priorities[area]
                area_score *= (priority / 5.0)
                
                total_relevance += area_score
                matched_areas.append(area)
                
        # Calculate final score
        if matched_areas:
            # Normalize by number of focus areas
            result.score = min(1.0, total_relevance / max(1, len(self.focus_areas) * 3))
            result.passed = result.score >= self.min_score
            result.confidence = result.score
        else:
            result.add_failure_reason(f"Relevance score {result.score:.2f} below threshold {self.min_score}")
            
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        return result


class CompositeFilter(ContentFilter):
    """
    Composite filter that combines multiple filters with configurable strategy.
    """
    
    def __init__(
        self,
        filters: List[ContentFilter],
        strategy: FilterStrategy = FilterStrategy.BALANCED,
        min_passing_filters: Optional[int] = None,
        **kwargs
    ):
        super().__init__(name="composite_filter", **kwargs)
        self.filters = filters
        self.strategy = strategy
        self.min_passing_filters = min_passing_filters or len(filters) // 2
        
    async def filter(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> FilterResult:
        """Apply composite filter to content."""
        import time
        import asyncio
        
        start_time = time.time()
        
        # Run all filters concurrently
        filter_tasks = [
            f.filter(content, context) for f in self.filters
        ]
        filter_results = await asyncio.gather(*filter_tasks)
        
        # Combine results based on strategy
        result = FilterResult(passed=False, score=0.0)
        
        passed_filters = []
        failed_filters = []
        total_score = 0.0
        all_matches = []
        
        for i, filter_result in enumerate(filter_results):
            filter_name = self.filters[i].name
            filter_weight = self.filters[i].weight
            
            if filter_result.passed:
                passed_filters.append(filter_name)
            else:
                failed_filters.append(filter_name)
                
            # Weighted score
            total_score += filter_result.score * filter_weight
            
            # Collect all matches
            all_matches.extend(filter_result.matches)
            
            # Collect failure reasons from required filters
            if self.filters[i].required and not filter_result.passed:
                result.add_failure_reason(f"Required filter '{filter_name}' failed")
                
        # Apply strategy
        total_weight = sum(f.weight for f in self.filters)
        result.score = total_score / total_weight if total_weight > 0 else 0
        
        if self.strategy == FilterStrategy.STRICT:
            # All filters must pass
            result.passed = len(passed_filters) == len(self.filters)
            
        elif self.strategy == FilterStrategy.BALANCED:
            # Most filters should pass
            result.passed = len(passed_filters) >= self.min_passing_filters
            
        elif self.strategy == FilterStrategy.LENIENT:
            # At least one filter should pass
            result.passed = len(passed_filters) > 0
            
        # Check required filters
        for i, f in enumerate(self.filters):
            if f.required and not filter_results[i].passed:
                result.passed = False
                break
                
        # Add matches and calculate confidence
        result.matches = all_matches
        if len(self.filters) > 0:
            result.confidence = len(passed_filters) / len(self.filters)
        else:
            result.confidence = 0.0
            
        if not result.passed and not result.failure_reasons:
            result.add_failure_reason(
                f"Failed composite filter: {len(passed_filters)}/{len(self.filters)} filters passed"
            )
            
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        return result


class ContentFilterFactory:
    """Factory for creating content filters based on user context."""
    
    @staticmethod
    def create_user_filters(context: Dict[str, Any]) -> CompositeFilter:
        """Create composite filter based on user context."""
        filters = []
        
        # Add keyword filter from focus areas
        focus_areas = context.get("focus_areas", [])
        if focus_areas:
            all_keywords = []
            for area in focus_areas:
                keywords = area.get("keywords", "").split(",")
                all_keywords.extend([kw.strip() for kw in keywords if kw.strip()])
                
            if all_keywords:
                filters.append(KeywordFilter(
                    keywords=all_keywords,
                    match_type="partial",
                    min_matches=1,
                    weight=1.0
                ))
                
        # Add entity filter
        entities = context.get("tracked_entities", [])
        if entities:
            filters.append(EntityFilter(
                entities=entities,
                min_matches=1,
                weight=1.5,
                required=False
            ))
            
        # Add relevance filter
        if focus_areas:
            filters.append(RelevanceFilter(
                focus_areas=focus_areas,
                min_score=0.3,
                weight=2.0
            ))
            
        # Create composite filter
        strategy = FilterStrategy.BALANCED
        if context.get("strict_filtering", False):
            strategy = FilterStrategy.STRICT
        elif context.get("lenient_filtering", False):
            strategy = FilterStrategy.LENIENT
            
        return CompositeFilter(
            filters=filters,
            strategy=strategy,
            min_passing_filters=max(1, len(filters) // 2)
        )
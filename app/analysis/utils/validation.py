"""
Validation utilities for Analysis Service.

Data validation, sanitization, and context validation functions.
"""

from typing import Dict, List, Any, Optional, Union
import re
from decimal import Decimal

from .common_types import (
    AnalysisContext, ContentPriority, AnalysisStage,
    FilterStrategy, AnalysisBatch
)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class DataValidator:
    """Validator for data integrity and format validation."""
    
    # Validation patterns
    PATTERNS = {
        "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        "url": re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
        "alphanumeric": re.compile(r'^[a-zA-Z0-9_-]+$'),
        "numeric": re.compile(r'^-?\d+(\.\d+)?$')
    }
    
    @classmethod
    def validate_content_item(cls, content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize content item."""
        if not isinstance(content, dict):
            raise ValidationError("Content must be a dictionary")
            
        # Required fields
        if "id" not in content:
            raise ValidationError("Content must have an 'id' field")
            
        # Sanitize and validate fields
        validated = {
            "id": cls._validate_id(content["id"]),
            "title": cls._sanitize_text(content.get("title", "")),
            "content_text": cls._sanitize_text(content.get("content_text", "")),
            "content_url": cls._validate_url(content.get("content_url")),
            "published_at": content.get("published_at"),  # Datetime validation in service
            "source_id": content.get("source_id", 0)
        }
        
        # Remove empty values
        return {k: v for k, v in validated.items() if v is not None and v != ""}
    
    @classmethod
    def validate_batch_items(cls, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a batch of content items."""
        if not isinstance(items, list):
            raise ValidationError("Items must be a list")
            
        if not items:
            raise ValidationError("Items list cannot be empty")
            
        validated_items = []
        for i, item in enumerate(items):
            try:
                validated_item = cls.validate_content_item(item)
                validated_items.append(validated_item)
            except ValidationError as e:
                raise ValidationError(f"Item {i} validation failed: {str(e)}")
                
        return validated_items
    
    @classmethod
    def validate_keywords(cls, keywords: Union[str, List[str]]) -> List[str]:
        """Validate and normalize keywords."""
        if isinstance(keywords, str):
            # Split comma-separated string
            keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        elif isinstance(keywords, list):
            keyword_list = [str(kw).strip() for kw in keywords if str(kw).strip()]
        else:
            raise ValidationError("Keywords must be string or list")
            
        # Validate individual keywords
        validated = []
        for keyword in keyword_list:
            if len(keyword) < 2:
                continue  # Skip very short keywords
            if len(keyword) > 100:
                keyword = keyword[:100]  # Truncate long keywords
            validated.append(keyword.lower())
            
        return validated
    
    @classmethod
    def validate_priority(cls, priority: Any) -> ContentPriority:
        """Validate and convert priority value."""
        if isinstance(priority, ContentPriority):
            return priority
            
        if isinstance(priority, str):
            try:
                return ContentPriority(priority.lower())
            except ValueError:
                return ContentPriority.MEDIUM
                
        if isinstance(priority, int):
            # Convert numeric priority (1-5) to enum
            priority_map = {
                5: ContentPriority.CRITICAL,
                4: ContentPriority.HIGH,
                3: ContentPriority.MEDIUM,
                2: ContentPriority.LOW,
                1: ContentPriority.LOW
            }
            return priority_map.get(priority, ContentPriority.MEDIUM)
            
        return ContentPriority.MEDIUM
    
    @classmethod
    def validate_stage(cls, stage: Any) -> AnalysisStage:
        """Validate and convert stage value."""
        if isinstance(stage, AnalysisStage):
            return stage
            
        if isinstance(stage, str):
            try:
                return AnalysisStage(stage.lower())
            except ValueError:
                raise ValidationError(f"Invalid analysis stage: {stage}")
                
        raise ValidationError("Stage must be string or AnalysisStage enum")
    
    @classmethod
    def validate_cost_limit(cls, cost_limit: Any) -> Optional[Decimal]:
        """Validate cost limit value."""
        if cost_limit is None:
            return None
            
        try:
            limit = Decimal(str(cost_limit))
            if limit < 0:
                raise ValidationError("Cost limit cannot be negative")
            if limit > Decimal("1000.00"):
                raise ValidationError("Cost limit too high (max: $1000)")
            return limit
        except (ValueError, TypeError):
            raise ValidationError("Invalid cost limit format")
    
    @classmethod
    def _validate_id(cls, id_value: Any) -> int:
        """Validate ID value."""
        try:
            id_int = int(id_value)
            if id_int <= 0:
                raise ValidationError("ID must be positive integer")
            return id_int
        except (ValueError, TypeError):
            raise ValidationError("Invalid ID format")
    
    @classmethod
    def _sanitize_text(cls, text: Any) -> str:
        """Sanitize text content."""
        if text is None:
            return ""
            
        text_str = str(text).strip()
        
        # Remove potentially harmful content
        text_str = re.sub(r'[^\x00-\x7F]+', ' ', text_str)  # ASCII only
        text_str = re.sub(r'\s+', ' ', text_str)  # Normalize whitespace
        
        # Truncate if too long
        if len(text_str) > 50000:  # 50KB limit
            text_str = text_str[:50000]
            
        return text_str
    
    @classmethod
    def _validate_url(cls, url: Any) -> Optional[str]:
        """Validate URL format."""
        if not url:
            return None
            
        url_str = str(url).strip()
        
        if not url_str.startswith(('http://', 'https://')):
            return None
            
        if len(url_str) > 2000:  # URL length limit
            return None
            
        # Basic URL validation
        if cls.PATTERNS["url"].match(url_str):
            return url_str
            
        return None


class ContextValidator:
    """Validator for analysis context and configuration."""
    
    @classmethod
    def validate_analysis_context(cls, context: Dict[str, Any]) -> AnalysisContext:
        """Validate and create AnalysisContext from dictionary."""
        if not isinstance(context, dict):
            raise ValidationError("Context must be a dictionary")
            
        # Required field
        if "user_id" not in context:
            raise ValidationError("Context must have user_id")
            
        user_id = DataValidator._validate_id(context["user_id"])
        
        # Validate optional fields
        strategic_profile = context.get("strategic_profile")
        if strategic_profile and not isinstance(strategic_profile, dict):
            raise ValidationError("Strategic profile must be a dictionary")
            
        focus_areas = context.get("focus_areas", [])
        if not isinstance(focus_areas, list):
            raise ValidationError("Focus areas must be a list")
            
        tracked_entities = context.get("tracked_entities", [])
        if not isinstance(tracked_entities, list):
            raise ValidationError("Tracked entities must be a list")
            
        delivery_preferences = context.get("delivery_preferences")
        if delivery_preferences and not isinstance(delivery_preferences, dict):
            raise ValidationError("Delivery preferences must be a dictionary")
            
        analysis_depth = context.get("analysis_depth", "standard")
        if analysis_depth not in ["quick", "standard", "deep"]:
            analysis_depth = "standard"
            
        cost_limit = DataValidator.validate_cost_limit(context.get("cost_limit"))
        
        return AnalysisContext(
            user_id=user_id,
            strategic_profile=strategic_profile,
            focus_areas=cls._validate_focus_areas(focus_areas),
            tracked_entities=cls._validate_tracked_entities(tracked_entities),
            delivery_preferences=delivery_preferences,
            analysis_depth=analysis_depth,
            cost_limit=cost_limit
        )
    
    @classmethod
    def validate_filter_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate filter configuration."""
        if not isinstance(config, dict):
            raise ValidationError("Filter config must be a dictionary")
            
        validated = {}
        
        # Validate keywords
        if "keywords" in config:
            validated["keywords"] = DataValidator.validate_keywords(config["keywords"])
            
        # Validate strategy
        if "strategy" in config:
            strategy_str = str(config["strategy"]).lower()
            try:
                validated["strategy"] = FilterStrategy(strategy_str)
            except ValueError:
                validated["strategy"] = FilterStrategy.BALANCED
        else:
            validated["strategy"] = FilterStrategy.BALANCED
            
        # Validate thresholds
        for field in ["min_score", "threshold", "confidence_threshold"]:
            if field in config:
                try:
                    value = float(config[field])
                    if 0.0 <= value <= 1.0:
                        validated[field] = value
                except (ValueError, TypeError):
                    pass
                    
        return validated
    
    @classmethod
    def validate_batch(cls, batch: Dict[str, Any]) -> AnalysisBatch:
        """Validate analysis batch."""
        if not isinstance(batch, dict):
            raise ValidationError("Batch must be a dictionary")
            
        # Required fields
        required_fields = ["batch_id", "user_id", "content_items", "context"]
        for field in required_fields:
            if field not in batch:
                raise ValidationError(f"Batch missing required field: {field}")
                
        batch_id = str(batch["batch_id"]).strip()
        if not batch_id or len(batch_id) > 50:
            raise ValidationError("Invalid batch_id")
            
        user_id = DataValidator._validate_id(batch["user_id"])
        
        content_items = DataValidator.validate_batch_items(batch["content_items"])
        
        context = cls.validate_analysis_context(batch["context"])
        
        priority = DataValidator.validate_priority(batch.get("priority", "medium"))
        
        return AnalysisBatch(
            batch_id=batch_id,
            user_id=user_id,
            content_items=content_items,
            context=context,
            priority=priority
        )
    
    @classmethod
    def _validate_focus_areas(cls, focus_areas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate focus areas list."""
        validated = []
        
        for area in focus_areas:
            if not isinstance(area, dict):
                continue
                
            validated_area = {}
            
            # Focus area name
            if "focus_area" in area:
                name = str(area["focus_area"]).strip()
                if name and len(name) <= 200:
                    validated_area["focus_area"] = name
                    
            # Keywords
            if "keywords" in area:
                keywords = DataValidator.validate_keywords(area["keywords"])
                if keywords:
                    validated_area["keywords"] = ",".join(keywords)
                    
            # Priority
            if "priority" in area:
                priority = area["priority"]
                if isinstance(priority, int) and 1 <= priority <= 5:
                    validated_area["priority"] = priority
                    
            if validated_area.get("focus_area"):
                validated.append(validated_area)
                
        return validated
    
    @classmethod
    def _validate_tracked_entities(cls, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate tracked entities list."""
        validated = []
        
        valid_entity_types = [
            "competitor", "technology", "person", "organization",
            "topic", "event", "market", "product"
        ]
        
        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            validated_entity = {}
            
            # Entity name
            if "entity_name" in entity:
                name = str(entity["entity_name"]).strip()
                if name and len(name) <= 200:
                    validated_entity["entity_name"] = name
                    
            # Entity type
            if "entity_type" in entity:
                entity_type = str(entity["entity_type"]).lower().strip()
                if entity_type in valid_entity_types:
                    validated_entity["entity_type"] = entity_type
                    
            # Keywords
            if "keywords" in entity:
                keywords = DataValidator.validate_keywords(entity["keywords"])
                if keywords:
                    validated_entity["keywords"] = ",".join(keywords)
                    
            # Priority
            if "priority" in entity:
                priority = entity["priority"]
                if isinstance(priority, int) and 1 <= priority <= 5:
                    validated_entity["priority"] = priority
                    
            if validated_entity.get("entity_name"):
                validated.append(validated_entity)
                
        return validated
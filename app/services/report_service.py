"""
Report Generation Service for Phase 4 - Strategic Intelligence Delivery

Multi-format report generation service that transforms analyzed content into 
actionable strategic intelligence reports with priority-based sections.

Features:
- Priority-based content curation using ContentPriority classifications
- Multi-format output: SendGrid Email, API JSON, Dashboard-ready formats
- Content deduplication with quality scoring preferences
- Strategic insights integration for "why relevant" explanations
- ASCII-only output for compatibility with Claude Code development
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from pydantic import BaseModel, Field

from app.database import get_db_session
from app.models.discovery import DiscoveredContent
from app.models.analysis import AnalysisResult, StrategicInsight
from app.analysis.core.shared_types import ContentPriority
from app.services.base_service import BaseIntelligenceService
from app.services.performance_optimizer import cached, performance_monitor, MemoryOptimizer


class ReportFormat(Enum):
    """Report output formats."""
    EMAIL_HTML = "email_html"
    API_JSON = "api_json" 
    DASHBOARD = "dashboard"


class ReportType(Enum):
    """Report generation types."""
    DAILY_DIGEST = "daily_digest"
    WEEKLY_SUMMARY = "weekly_summary"
    URGENT_ALERT = "urgent_alert"
    CUSTOM_QUERY = "custom_query"


@dataclass
class ContentItem:
    """Structured content item for report generation."""
    content_id: int
    title: str
    url: str
    priority: ContentPriority
    overall_score: float
    published_at: datetime
    source_name: str
    
    # Strategic context
    strategic_insights: List[Dict[str, Any]]
    relevance_explanation: str
    matched_entities: List[str]
    matched_focus_areas: List[str]
    
    # Analysis scores
    strategic_alignment: float
    competitive_impact: float
    urgency_score: float


@dataclass 
class ReportSection:
    """Priority-based report section."""
    priority: ContentPriority
    title: str
    description: str
    items: List[ContentItem]
    total_count: int
    section_summary: str


class ReportGenerationRequest(BaseModel):
    """Request model for report generation."""
    user_id: int
    report_type: ReportType
    output_formats: List[ReportFormat]
    date_range_days: int = Field(default=1, ge=1, le=30)
    min_priority: ContentPriority = ContentPriority.MEDIUM
    max_items_per_section: int = Field(default=10, ge=1, le=50)
    include_low_priority: bool = Field(default=False)
    custom_filters: Optional[Dict[str, Any]] = None


class ReportOutput(BaseModel):
    """Generated report output."""
    report_id: str
    user_id: int
    report_type: ReportType
    format: ReportFormat
    content: str
    metadata: Dict[str, Any]
    generated_at: datetime
    content_items_count: int
    sections_count: int


class ReportService(BaseIntelligenceService):
    """
    Strategic Report Generation Service
    
    Transforms analyzed content into actionable intelligence reports with:
    - Priority-based content organization
    - Multi-format output generation
    - Content deduplication and quality filtering
    - Strategic insights integration
    """
    
    def __init__(self):
        super().__init__("report_service")
    
    @performance_monitor("report_generation")
    async def generate_report(
        self,
        request: ReportGenerationRequest
    ) -> List[ReportOutput]:
        """
        Generate strategic intelligence report in requested formats.
        
        Args:
            request: Report generation configuration
            
        Returns:
            List of ReportOutput objects for each requested format
        """
        async with get_db_session() as db:
            # Step 1: Retrieve and curate content
            content_items = await self._retrieve_content_for_report(db, request)
            
            # Step 2: Apply deduplication and quality filtering
            # Memory optimization before deduplication
            if len(content_items) > 100:
                MemoryOptimizer.force_garbage_collection()
            
            filtered_items = await self._deduplicate_and_filter_content(content_items)
            
            # Step 3: Organize into priority-based sections
            report_sections = await self._organize_into_sections(
                filtered_items, request.max_items_per_section
            )
            
            # Step 4: Generate reports in all requested formats
            reports = []
            for format_type in request.output_formats:
                report = await self._generate_format_specific_report(
                    db, request, report_sections, format_type
                )
                reports.append(report)
            
            return reports
    
    async def _retrieve_content_for_report(
        self,
        db: AsyncSession,
        request: ReportGenerationRequest
    ) -> List[ContentItem]:
        """
        Retrieve analyzed content based on report parameters.
        """
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=request.date_range_days)
        
        # Build query for analyzed content with strategic insights
        query = select(
            DiscoveredContent.id,
            DiscoveredContent.title,
            DiscoveredContent.content_url,
            DiscoveredContent.overall_score,
            DiscoveredContent.published_at,
            DiscoveredContent.source_id,
            AnalysisResult.filter_priority,
            AnalysisResult.strategic_alignment,
            AnalysisResult.competitive_impact,
            AnalysisResult.urgency_score,
            AnalysisResult.filter_matched_entities,
            AnalysisResult.filter_matched_keywords
        ).select_from(
            DiscoveredContent
        ).join(
            AnalysisResult, DiscoveredContent.id == AnalysisResult.content_id
        ).where(
            and_(
                AnalysisResult.user_id == request.user_id,
                AnalysisResult.filter_passed == True,
                DiscoveredContent.published_at >= start_date,
                DiscoveredContent.published_at <= end_date
            )
        ).order_by(
            desc(AnalysisResult.strategic_alignment),
            desc(DiscoveredContent.overall_score)
        )
        
        # Apply priority filtering
        if not request.include_low_priority:
            priority_values = []
            if request.min_priority in [ContentPriority.CRITICAL, ContentPriority.HIGH, ContentPriority.MEDIUM]:
                priority_values.extend(["critical", "high", "medium"])
            elif request.min_priority == ContentPriority.HIGH:
                priority_values.extend(["critical", "high"])
            elif request.min_priority == ContentPriority.CRITICAL:
                priority_values.append("critical")
            
            if priority_values:
                query = query.where(AnalysisResult.filter_priority.in_(priority_values))
        
        result = await db.execute(query)
        rows = result.fetchall()
        
        # Convert to ContentItem objects with strategic context
        content_items = []
        for row in rows:
            # Get strategic insights for this content
            insights = await self._get_strategic_insights(db, row.id)
            
            # Get source name
            source_name = await self._get_source_name(db, row.source_id)
            
            # Create relevance explanation
            relevance_explanation = self.create_relevance_explanation(
                row.filter_matched_entities or [],
                row.filter_matched_keywords or [],
                float(row.strategic_alignment or 0.0),
                float(row.competitive_impact or 0.0),
                float(row.urgency_score or 0.0)
            )
            
            content_item = ContentItem(
                content_id=row.id,
                title=row.title,
                url=row.content_url,
                priority=ContentPriority.from_score(float(row.strategic_alignment or 0.0)),
                overall_score=float(row.overall_score or 0.0),
                published_at=row.published_at,
                source_name=source_name,
                strategic_insights=[insight.to_dict() for insight in insights],
                relevance_explanation=relevance_explanation,
                matched_entities=row.filter_matched_entities or [],
                matched_focus_areas=row.filter_matched_keywords or [],
                strategic_alignment=float(row.strategic_alignment or 0.0),
                competitive_impact=float(row.competitive_impact or 0.0),
                urgency_score=float(row.urgency_score or 0.0)
            )
            content_items.append(content_item)
        
        return content_items
    
    async def _get_strategic_insights(
        self, 
        db: AsyncSession, 
        content_id: int
    ) -> List[StrategicInsight]:
        """Get strategic insights for content item."""
        query = select(StrategicInsight).where(
            StrategicInsight.content_id == content_id
        ).order_by(desc(StrategicInsight.relevance_score))
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def _get_source_name(self, db: AsyncSession, source_id: int) -> str:
        """Get source name for content item."""
        from app.models.discovery import DiscoveredSource
        
        query = select(DiscoveredSource.source_name).where(
            DiscoveredSource.id == source_id
        )
        result = await db.execute(query)
        source_name = result.scalar()
        return source_name or f"Source {source_id}"
    
    
    async def _deduplicate_and_filter_content(
        self, 
        content_items: List[ContentItem]
    ) -> List[ContentItem]:
        """
        Apply deduplication and quality filtering.
        
        Keeps highest scoring source for duplicate content.
        """
        # Group by URL similarity and title similarity
        url_groups = {}
        title_groups = {}
        
        for item in content_items:
            # URL-based grouping
            url_hash = hashlib.md5(item.url.encode()).hexdigest()
            if url_hash not in url_groups:
                url_groups[url_hash] = []
            url_groups[url_hash].append(item)
            
            # Title-based grouping (simple similarity)
            title_key = item.title.lower().strip()[:100]  # First 100 chars
            if title_key not in title_groups:
                title_groups[title_key] = []
            title_groups[title_key].append(item)
        
        # Deduplicate: keep highest overall_score for each group
        filtered_items = []
        processed_urls = set()
        
        for url_hash, items in url_groups.items():
            if len(items) == 1:
                filtered_items.append(items[0])
                processed_urls.add(items[0].url)
            else:
                # Keep highest scoring item
                best_item = max(items, key=lambda x: x.overall_score)
                filtered_items.append(best_item)
                processed_urls.add(best_item.url)
        
        return filtered_items
    
    async def _organize_into_sections(
        self,
        content_items: List[ContentItem],
        max_items_per_section: int
    ) -> List[ReportSection]:
        """
        Organize content into priority-based sections.
        """
        # Group by priority
        priority_groups = {
            ContentPriority.CRITICAL: [],
            ContentPriority.HIGH: [],
            ContentPriority.MEDIUM: [],
            ContentPriority.LOW: []
        }
        
        for item in content_items:
            priority_groups[item.priority].append(item)
        
        # Create sections
        sections = []
        
        section_configs = {
            ContentPriority.CRITICAL: {
                "title": "CRITICAL INTELLIGENCE",
                "description": "Urgent items requiring immediate attention and strategic response"
            },
            ContentPriority.HIGH: {
                "title": "HIGH PRIORITY INSIGHTS", 
                "description": "Important strategic intelligence for near-term planning and action"
            },
            ContentPriority.MEDIUM: {
                "title": "STRATEGIC UPDATES",
                "description": "Relevant intelligence for ongoing strategic awareness and planning"
            },
            ContentPriority.LOW: {
                "title": "BACKGROUND INTELLIGENCE",
                "description": "Contextual information for comprehensive strategic understanding"
            }
        }
        
        for priority in [ContentPriority.CRITICAL, ContentPriority.HIGH, 
                        ContentPriority.MEDIUM, ContentPriority.LOW]:
            items = priority_groups[priority]
            
            if not items:
                continue
                
            # Sort by strategic alignment and overall score
            items.sort(key=lambda x: (x.strategic_alignment, x.overall_score), reverse=True)
            
            # Limit items per section
            section_items = items[:max_items_per_section]
            
            # Create section summary
            section_summary = self._create_section_summary(section_items, priority)
            
            section = ReportSection(
                priority=priority,
                title=section_configs[priority]["title"],
                description=section_configs[priority]["description"],
                items=section_items,
                total_count=len(items),
                section_summary=section_summary
            )
            sections.append(section)
        
        return sections
    
    def _create_section_summary(
        self,
        items: List[ContentItem],
        priority: ContentPriority
    ) -> str:
        """Create summary for report section."""
        if not items:
            return "No items in this priority level."
        
        summary_parts = [
            f"{len(items)} items",
            f"avg score: {sum(item.overall_score for item in items) / len(items):.2f}"
        ]
        
        # Add priority-specific insights
        if priority == ContentPriority.CRITICAL:
            urgent_count = sum(1 for item in items if item.urgency_score > 0.8)
            if urgent_count > 0:
                summary_parts.append(f"{urgent_count} urgent")
        
        return f"({', '.join(summary_parts)})"
    
    async def _generate_format_specific_report(
        self,
        db: AsyncSession,
        request: ReportGenerationRequest,
        sections: List[ReportSection],
        format_type: ReportFormat
    ) -> ReportOutput:
        """Generate report in specific format."""
        
        # Generate unique report ID
        report_id = hashlib.md5(
            f"{request.user_id}-{datetime.utcnow().isoformat()}-{format_type.value}".encode()
        ).hexdigest()[:12]
        
        # Get user context for personalization
        user_context = await self.get_user_strategic_context(db, request.user_id)
        
        if format_type == ReportFormat.EMAIL_HTML:
            content = await self._generate_email_html(sections, user_context)
        elif format_type == ReportFormat.API_JSON:
            content = await self._generate_api_json(sections, user_context)
        elif format_type == ReportFormat.DASHBOARD:
            content = await self._generate_dashboard_format(sections, user_context)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Calculate metadata
        total_items = sum(len(section.items) for section in sections)
        
        metadata = {
            "user_context": user_context,
            "generation_params": {
                "date_range_days": request.date_range_days,
                "min_priority": request.min_priority.value,
                "max_items_per_section": request.max_items_per_section
            },
            "content_stats": {
                "sections_count": len(sections),
                "total_items": total_items,
                "priority_breakdown": {
                    section.priority.value: len(section.items) 
                    for section in sections
                }
            }
        }
        
        return ReportOutput(
            report_id=report_id,
            user_id=request.user_id,
            report_type=request.report_type,
            format=format_type,
            content=content,
            metadata=metadata,
            generated_at=datetime.utcnow(),
            content_items_count=total_items,
            sections_count=len(sections)
        )
    
    
    async def _generate_email_html(
        self,
        sections: List[ReportSection],
        user_context: Dict[str, Any]
    ) -> str:
        """Generate HTML email format."""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Strategic Intelligence Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
        .content {{ padding: 20px; }}
        .section {{ margin-bottom: 30px; border-left: 4px solid #3498db; padding-left: 15px; }}
        .section.critical {{ border-color: #e74c3c; }}
        .section.high {{ border-color: #f39c12; }}
        .section.medium {{ border-color: #3498db; }}
        .section.low {{ border-color: #95a5a6; }}
        .section-title {{ font-size: 18px; font-weight: bold; margin-bottom: 5px; color: #2c3e50; }}
        .section-desc {{ font-size: 14px; color: #7f8c8d; margin-bottom: 15px; }}
        .article {{ margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 6px; }}
        .article-title {{ font-size: 16px; font-weight: bold; margin-bottom: 8px; }}
        .article-title a {{ color: #2c3e50; text-decoration: none; }}
        .article-title a:hover {{ text-decoration: underline; }}
        .article-meta {{ font-size: 12px; color: #7f8c8d; margin-bottom: 8px; }}
        .article-relevance {{ font-size: 13px; color: #5d6d7e; font-style: italic; }}
        .footer {{ background-color: #ecf0f1; padding: 15px; text-align: center; font-size: 12px; color: #7f8c8d; border-radius: 0 0 8px 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Strategic Intelligence Report</h1>
            <p>Personalized for {user_context.get('user_name', 'User')} | {user_context.get('industry', 'Unknown Industry')} | {user_context.get('role', 'Unknown Role')}</p>
            <p>Generated on {datetime.utcnow().strftime('%B %d, %Y at %I:%M %p UTC')}</p>
        </div>
        <div class="content">
"""
        
        for section in sections:
            if not section.items:
                continue
                
            priority_class = section.priority.value
            html_content += f"""
            <div class="section {priority_class}">
                <div class="section-title">{section.title}</div>
                <div class="section-desc">{section.description} {section.section_summary}</div>
"""
            
            for item in section.items:
                html_content += f"""
                <div class="article">
                    <div class="article-title">
                        <a href="{item.url}" target="_blank">{item.title}</a>
                    </div>
                    <div class="article-meta">
                        Source: {item.source_name} | Published: {item.published_at.strftime('%m/%d/%Y')} | Score: {item.overall_score:.2f}
                    </div>
                    <div class="article-relevance">
                        Relevant because: {item.relevance_explanation}
                    </div>
                </div>
"""
            
            html_content += "</div>\n"
        
        html_content += f"""
        </div>
        <div class="footer">
            Generated with Claude Code Competitive Intelligence v2<br>
            Strategic Intelligence tailored for {user_context.get('industry', 'your industry')} professionals
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
    
    async def _generate_api_json(
        self,
        sections: List[ReportSection],
        user_context: Dict[str, Any]
    ) -> str:
        """Generate API JSON format."""
        
        report_data = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "user_context": user_context,
                "format": "api_json"
            },
            "sections": []
        }
        
        for section in sections:
            section_data = {
                "priority": section.priority.value,
                "title": section.title,
                "description": section.description,
                "summary": section.section_summary,
                "total_available": section.total_count,
                "items_included": len(section.items),
                "content_items": []
            }
            
            for item in section.items:
                item_data = {
                    "content_id": item.content_id,
                    "title": item.title,
                    "url": item.url,
                    "source_name": item.source_name,
                    "published_at": item.published_at.isoformat(),
                    "scores": {
                        "overall_score": item.overall_score,
                        "strategic_alignment": item.strategic_alignment,
                        "competitive_impact": item.competitive_impact,
                        "urgency_score": item.urgency_score
                    },
                    "relevance_explanation": item.relevance_explanation,
                    "matched_entities": item.matched_entities,
                    "matched_focus_areas": item.matched_focus_areas,
                    "strategic_insights": item.strategic_insights
                }
                section_data["content_items"].append(item_data)
            
            report_data["sections"].append(section_data)
        
        return json.dumps(report_data, indent=2, default=str)
    
    async def _generate_dashboard_format(
        self,
        sections: List[ReportSection],
        user_context: Dict[str, Any]
    ) -> str:
        """Generate dashboard-ready format."""
        
        dashboard_data = {
            "dashboard_config": {
                "user_name": user_context.get('user_name', 'User'),
                "industry": user_context.get('industry', 'Unknown'),
                "role": user_context.get('role', 'Unknown'),
                "generated_at": datetime.utcnow().isoformat()
            },
            "summary_stats": {
                "total_sections": len(sections),
                "total_items": sum(len(section.items) for section in sections),
                "priority_distribution": {
                    section.priority.value: len(section.items) 
                    for section in sections
                }
            },
            "priority_sections": []
        }
        
        for section in sections:
            section_widget = {
                "widget_type": "priority_section",
                "priority": section.priority.value,
                "title": section.title,
                "description": section.description,
                "item_count": len(section.items),
                "total_available": section.total_count,
                "display_config": {
                    "color_scheme": {
                        "critical": "#e74c3c",
                        "high": "#f39c12", 
                        "medium": "#3498db",
                        "low": "#95a5a6"
                    }.get(section.priority.value, "#3498db"),
                    "urgency_indicator": section.priority.value in ["critical", "high"]
                },
                "content_preview": []
            }
            
            # Add top 5 items for dashboard preview
            for item in section.items[:5]:
                preview_item = {
                    "content_id": item.content_id,
                    "title": item.title[:100] + "..." if len(item.title) > 100 else item.title,
                    "url": item.url,
                    "source": item.source_name,
                    "published_date": item.published_at.strftime('%Y-%m-%d'),
                    "score": round(item.overall_score, 2),
                    "relevance_summary": item.relevance_explanation[:150] + "..." if len(item.relevance_explanation) > 150 else item.relevance_explanation,
                    "tags": {
                        "entities": item.matched_entities[:3],
                        "focus_areas": item.matched_focus_areas[:3]
                    }
                }
                section_widget["content_preview"].append(preview_item)
            
            dashboard_data["priority_sections"].append(section_widget)
        
        return json.dumps(dashboard_data, indent=2, default=str)
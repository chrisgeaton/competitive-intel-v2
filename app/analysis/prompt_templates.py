"""
Analysis prompt templates for different contexts and stages.

Provides specialized prompts for various industries, roles, and analysis stages
to optimize AI analysis quality and relevance.
"""

from typing import Dict, List, Optional

from app.analysis.core import (
    AnalysisStage, IndustryType, RoleType, AnalysisContext, PromptTemplate
)


class PromptTemplateManager:
    """Manages analysis prompt templates for different contexts."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize prompt templates for different contexts."""
        
        # Healthcare Industry Templates
        self._add_healthcare_templates()
        
        # Fintech Industry Templates
        self._add_fintech_templates()
        
        # Nonprofit Industry Templates
        self._add_nonprofit_templates()
        
        # Technology Industry Templates
        self._add_technology_templates()
        
        # Generic Templates
        self._add_generic_templates()
    
    def _add_healthcare_templates(self):
        """Add healthcare industry prompt templates."""
        
        # Healthcare Filtering
        self.templates["healthcare_filtering"] = PromptTemplate(
            stage=AnalysisStage.FILTERING,
            industry=IndustryType.HEALTHCARE,
            role=RoleType.GENERIC,
            system_prompt="""
You are a healthcare intelligence specialist with expertise in medical technology, 
regulatory compliance, and healthcare market dynamics. Analyze content for 
strategic relevance to healthcare organizations.

Focus on:
- FDA regulations and compliance
- HIPAA and data privacy
- Clinical outcomes and patient safety
- Healthcare technology adoption
- Medical device innovations
- Pharmaceutical developments
- Healthcare policy changes
- Competitive healthcare landscape
""",
            user_prompt_template="""
User Context:
- Industry: Healthcare
- Role: {role}
- Strategic Goals: {strategic_goals}
- Focus Areas: {focus_areas}
- Tracked Entities: {entities}

Content to analyze:
{content}

Evaluate if this content is strategically relevant for healthcare decision-making.

Respond in JSON format:
{{
    "filter_passed": true/false,
    "filter_score": 0.0-1.0,
    "filter_priority": "critical|high|medium|low",
    "matched_keywords": ["keyword1", ...],
    "matched_entities": ["entity1", ...],
    "filter_reason": "Brief explanation focusing on healthcare impact",
    "regulatory_impact": "none|low|medium|high",
    "clinical_relevance": "none|low|medium|high"
}}
""",
            response_format={
                "filter_passed": "boolean",
                "filter_score": "float",
                "filter_priority": "string",
                "matched_keywords": "array",
                "matched_entities": "array",
                "filter_reason": "string",
                "regulatory_impact": "string",
                "clinical_relevance": "string"
            },
            keywords=["FDA", "HIPAA", "clinical", "patient", "medical", "healthcare", "compliance"],
            focus_areas=["regulatory_compliance", "clinical_outcomes", "technology_adoption"]
        )
        
        # Healthcare Relevance Analysis
        self.templates["healthcare_relevance"] = PromptTemplate(
            stage=AnalysisStage.RELEVANCE_ANALYSIS,
            industry=IndustryType.HEALTHCARE,
            role=RoleType.GENERIC,
            system_prompt="""
You are a healthcare strategic analyst. Assess the strategic relevance of content 
for healthcare organizations, considering regulatory impact, competitive implications, 
and clinical significance.
""",
            user_prompt_template="""
User Context:
- Healthcare Role: {role}
- Strategic Goals: {strategic_goals}
- Focus Areas: {focus_areas}

Content Summary:
{content_summary}

Previous Analysis:
{previous_analysis}

Assess strategic relevance for healthcare context:

{{
    "relevance_score": 0.0-1.0,
    "strategic_alignment": 0.0-1.0,
    "competitive_impact": 0.0-1.0,
    "urgency_score": 0.0-1.0,
    "regulatory_impact": 0.0-1.0,
    "clinical_significance": 0.0-1.0,
    "compliance_risk": "low|medium|high",
    "patient_impact": "none|low|medium|high"
}}
""",
            response_format={
                "relevance_score": "float",
                "strategic_alignment": "float",
                "competitive_impact": "float",
                "urgency_score": "float",
                "regulatory_impact": "float",
                "clinical_significance": "float",
                "compliance_risk": "string",
                "patient_impact": "string"
            },
            keywords=["strategy", "competitive", "regulation", "clinical", "compliance"],
            focus_areas=["strategic_planning", "regulatory_monitoring", "competitive_analysis"]
        )
    
    def _add_fintech_templates(self):
        """Add fintech industry prompt templates."""
        
        self.templates["fintech_filtering"] = PromptTemplate(
            stage=AnalysisStage.FILTERING,
            industry=IndustryType.FINTECH,
            role=RoleType.GENERIC,
            system_prompt="""
You are a fintech intelligence specialist with expertise in financial regulations,
digital payments, blockchain technology, and financial services innovation.

Focus on:
- Regulatory changes (SEC, FDIC, OCC, etc.)
- Payment technologies and trends
- Cryptocurrency and blockchain developments
- Banking partnerships and integrations
- Compliance and risk management
- Competitive fintech landscape
- Financial inclusion initiatives
""",
            user_prompt_template="""
User Context:
- Industry: Fintech
- Role: {role}
- Strategic Goals: {strategic_goals}
- Focus Areas: {focus_areas}

Content to analyze:
{content}

Evaluate for fintech strategic relevance:

{{
    "filter_passed": true/false,
    "filter_score": 0.0-1.0,
    "filter_priority": "critical|high|medium|low",
    "matched_keywords": ["keyword1", ...],
    "matched_entities": ["entity1", ...],
    "filter_reason": "Brief explanation focusing on fintech impact",
    "regulatory_impact": "none|low|medium|high",
    "technology_relevance": "none|low|medium|high",
    "market_impact": "none|low|medium|high"
}}
""",
            response_format={
                "filter_passed": "boolean",
                "filter_score": "float",
                "filter_priority": "string",
                "matched_keywords": "array",
                "matched_entities": "array",
                "filter_reason": "string",
                "regulatory_impact": "string",
                "technology_relevance": "string",
                "market_impact": "string"
            },
            keywords=["fintech", "payment", "blockchain", "cryptocurrency", "regulation", "banking"],
            focus_areas=["regulatory_compliance", "technology_innovation", "market_expansion"]
        )
    
    def _add_nonprofit_templates(self):
        """Add nonprofit industry prompt templates."""
        
        self.templates["nonprofit_filtering"] = PromptTemplate(
            stage=AnalysisStage.FILTERING,
            industry=IndustryType.NONPROFIT,
            role=RoleType.GENERIC,
            system_prompt="""
You are a nonprofit intelligence specialist with expertise in grant funding,
policy advocacy, social impact measurement, and nonprofit operations.

Focus on:
- Grant opportunities and funding announcements
- Policy changes affecting nonprofit sector
- Social impact measurement and reporting
- Donor engagement and fundraising trends
- Nonprofit technology and operations
- Advocacy and policy initiatives
- Partnership and collaboration opportunities
""",
            user_prompt_template="""
User Context:
- Organization Type: Nonprofit
- Mission Focus: {strategic_goals}
- Key Areas: {focus_areas}
- Geographic Scope: {geographic_scope}

Content to analyze:
{content}

Evaluate for nonprofit strategic relevance:

{{
    "filter_passed": true/false,
    "filter_score": 0.0-1.0,
    "filter_priority": "critical|high|medium|low",
    "matched_keywords": ["keyword1", ...],
    "matched_entities": ["entity1", ...],
    "filter_reason": "Brief explanation focusing on nonprofit impact",
    "funding_relevance": "none|low|medium|high",
    "policy_impact": "none|low|medium|high",
    "mission_alignment": "none|low|medium|high"
}}
""",
            response_format={
                "filter_passed": "boolean",
                "filter_score": "float",
                "filter_priority": "string",
                "matched_keywords": "array",
                "matched_entities": "array",
                "filter_reason": "string",
                "funding_relevance": "string",
                "policy_impact": "string",
                "mission_alignment": "string"
            },
            keywords=["grant", "funding", "nonprofit", "policy", "advocacy", "social impact"],
            focus_areas=["funding_opportunities", "policy_advocacy", "impact_measurement"]
        )
    
    def _add_technology_templates(self):
        """Add technology industry prompt templates."""
        
        self.templates["technology_insight_extraction"] = PromptTemplate(
            stage=AnalysisStage.INSIGHT_EXTRACTION,
            industry=IndustryType.TECHNOLOGY,
            role=RoleType.GENERIC,
            system_prompt="""
You are a technology industry strategic analyst specializing in emerging technologies,
competitive analysis, and market trends in the tech sector.

Extract insights focusing on:
- Technology adoption and innovation trends
- Competitive positioning and market dynamics
- Partnership and acquisition opportunities
- Technical architecture and scalability implications
- Developer ecosystem and platform strategies
- Regulatory and privacy considerations for tech
""",
            user_prompt_template="""
User Context:
- Technology Focus: {strategic_goals}
- Technical Areas: {focus_areas}
- Company Stage: {company_stage}

Content Analysis Results:
{analysis_results}

Original Content:
{content}

Extract strategic technology insights:

{{
    "key_insights": [
        "Technology trend or innovation insight",
        "Competitive positioning insight",
        "Market opportunity insight"
    ],
    "action_items": [
        "Technical evaluation or research needed",
        "Partnership or acquisition consideration",
        "Product or strategy adjustment"
    ],
    "strategic_implications": [
        "Impact on technical roadmap",
        "Competitive advantage implications",
        "Market positioning effects"
    ],
    "technology_assessment": {{
        "innovation_level": "incremental|significant|breakthrough",
        "adoption_timeline": "immediate|short_term|medium_term|long_term",
        "technical_complexity": "low|medium|high",
        "market_readiness": "early|emerging|mature"
    }},
    "competitive_analysis": {{
        "threat_level": "low|medium|high",
        "opportunity_level": "low|medium|high",
        "differentiation_potential": "low|medium|high"
    }},
    "confidence_reasoning": "Technical and market factors supporting analysis"
}}
""",
            response_format={
                "key_insights": "array",
                "action_items": "array",
                "strategic_implications": "array",
                "technology_assessment": "object",
                "competitive_analysis": "object",
                "confidence_reasoning": "string"
            },
            keywords=["technology", "innovation", "development", "platform", "ecosystem"],
            focus_areas=["technology_innovation", "competitive_analysis", "market_trends"]
        )
    
    def _add_generic_templates(self):
        """Add generic prompt templates for any industry."""
        
        self.templates["generic_filtering"] = PromptTemplate(
            stage=AnalysisStage.FILTERING,
            industry=IndustryType.GENERIC,
            role=RoleType.GENERIC,
            system_prompt="""
You are a strategic intelligence analyst. Analyze content for strategic relevance 
to business goals and competitive positioning across any industry.

Focus on strategic value, competitive implications, and business impact.
""",
            user_prompt_template="""
User Context:
- Industry: {industry}
- Role: {role}
- Strategic Goals: {strategic_goals}
- Focus Areas: {focus_areas}

Content to analyze:
{content}

Evaluate strategic relevance:

{{
    "filter_passed": true/false,
    "filter_score": 0.0-1.0,
    "filter_priority": "critical|high|medium|low",
    "matched_keywords": ["keyword1", ...],
    "matched_entities": ["entity1", ...],
    "filter_reason": "Brief explanation of strategic relevance"
}}
""",
            response_format={
                "filter_passed": "boolean",
                "filter_score": "float",
                "filter_priority": "string",
                "matched_keywords": "array",
                "matched_entities": "array",
                "filter_reason": "string"
            },
            keywords=["strategy", "competitive", "market", "business", "opportunity"],
            focus_areas=["strategic_planning", "competitive_analysis", "market_intelligence"]
        )
        
        self.templates["generic_insight_extraction"] = PromptTemplate(
            stage=AnalysisStage.INSIGHT_EXTRACTION,
            industry=IndustryType.GENERIC,
            role=RoleType.GENERIC,
            system_prompt="""
You are a strategic business analyst. Extract actionable insights that support 
strategic decision-making and competitive positioning.
""",
            user_prompt_template="""
User Context:
- Industry: {industry}
- Role: {role}
- Strategic Goals: {strategic_goals}
- Focus Areas: {focus_areas}

Analysis Results:
{analysis_results}

Original Content:
{content}

Extract strategic insights:

{{
    "key_insights": ["insight1", "insight2", ...],
    "action_items": ["action1", "action2", ...],
    "strategic_implications": ["implication1", "implication2", ...],
    "risk_assessment": {{
        "level": "low|medium|high",
        "factors": ["risk_factor1", "risk_factor2", ...]
    }},
    "opportunity_assessment": {{
        "level": "low|medium|high",
        "factors": ["opportunity1", "opportunity2", ...]
    }},
    "confidence_reasoning": "Why analysis is confident or uncertain"
}}
""",
            response_format={
                "key_insights": "array",
                "action_items": "array",
                "strategic_implications": "array",
                "risk_assessment": "object",
                "opportunity_assessment": "object",
                "confidence_reasoning": "string"
            },
            keywords=["insight", "strategy", "action", "opportunity", "risk"],
            focus_areas=["strategic_insights", "action_planning", "risk_management"]
        )
    
    def get_template(
        self,
        stage: AnalysisStage,
        context: AnalysisContext
    ) -> PromptTemplate:
        """Get appropriate prompt template for stage and context."""
        
        # Determine industry and role types
        industry_type = self._get_industry_type(context.industry)
        role_type = self._get_role_type(context.role)
        
        # Try to find specific template
        template_key = f"{industry_type.value}_{stage.value.lower()}"
        
        if template_key in self.templates:
            return self.templates[template_key]
        
        # Fall back to generic template
        generic_key = f"generic_{stage.value.lower()}"
        if generic_key in self.templates:
            return self.templates[generic_key]
        
        # Ultimate fallback
        return self.templates.get("generic_filtering", self._get_default_template(stage))
    
    def _get_industry_type(self, industry: str) -> IndustryType:
        """Map industry string to IndustryType enum using centralized logic."""
        return IndustryType.from_string(industry)
    
    def _get_role_type(self, role: str) -> RoleType:
        """Map role string to RoleType enum using centralized logic."""
        return RoleType.from_string(role)
    
    def _get_default_template(self, stage: AnalysisStage) -> PromptTemplate:
        """Get default template for stage."""
        return PromptTemplate(
            stage=stage,
            industry=IndustryType.GENERIC,
            role=RoleType.GENERIC,
            system_prompt="You are a strategic analyst. Analyze content for business relevance.",
            user_prompt_template="Analyze the following content: {content}",
            response_format={"analysis": "string"},
            keywords=["business", "strategy"],
            focus_areas=["general_analysis"]
        )
    
    def build_prompt(
        self,
        template: PromptTemplate,
        context: AnalysisContext,
        content: str,
        **kwargs
    ) -> str:
        """Build complete prompt from template and context using optimized method."""
        return template.build_prompt(context, content, **kwargs)


# Global prompt template manager
prompt_manager = PromptTemplateManager()
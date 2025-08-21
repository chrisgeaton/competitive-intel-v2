"""
Strategic profile schemas with enhanced validation for industry types and strategic goals.
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, validator


class IndustryType(str, Enum):
    """Enumeration of supported industry types."""
    HEALTHCARE = "healthcare"
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    EDUCATION = "education"
    NONPROFIT = "nonprofit"
    GOVERNMENT = "government"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    CONSULTING = "consulting"
    ENERGY = "energy"
    TELECOMMUNICATIONS = "telecommunications"
    AUTOMOTIVE = "automotive"
    AEROSPACE = "aerospace"
    PHARMACEUTICALS = "pharmaceuticals"
    BIOTECHNOLOGY = "biotechnology"
    MEDIA = "media"
    REAL_ESTATE = "real_estate"
    AGRICULTURE = "agriculture"
    FOOD_BEVERAGE = "food_beverage"
    TRANSPORTATION = "transportation"
    LOGISTICS = "logistics"
    HOSPITALITY = "hospitality"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"
    LEGAL = "legal"
    ARCHITECTURE = "architecture"
    CONSTRUCTION = "construction"
    INSURANCE = "insurance"
    BANKING = "banking"
    OTHER = "other"


class OrganizationType(str, Enum):
    """Enumeration of organization types."""
    STARTUP = "startup"
    SMALL_BUSINESS = "small_business"
    MEDIUM_ENTERPRISE = "medium_enterprise"
    LARGE_ENTERPRISE = "large_enterprise"
    MULTINATIONAL = "multinational"
    NONPROFIT = "nonprofit"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    HEALTHCARE_SYSTEM = "healthcare_system"
    RESEARCH_INSTITUTE = "research_institute"
    CONSULTING_FIRM = "consulting_firm"
    OTHER = "other"


class OrganizationSize(str, Enum):
    """Enumeration of organization sizes."""
    MICRO = "micro"          # 1-9 employees
    SMALL = "small"          # 10-49 employees
    MEDIUM = "medium"        # 50-249 employees
    LARGE = "large"          # 250-999 employees
    ENTERPRISE = "enterprise" # 1000+ employees


class StrategicGoalCategory(str, Enum):
    """Enumeration of strategic goal categories."""
    MARKET_EXPANSION = "market_expansion"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    PRODUCT_DEVELOPMENT = "product_development"
    DIGITAL_TRANSFORMATION = "digital_transformation"
    CUSTOMER_ACQUISITION = "customer_acquisition"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    COST_REDUCTION = "cost_reduction"
    REVENUE_GROWTH = "revenue_growth"
    BRAND_AWARENESS = "brand_awareness"
    TALENT_ACQUISITION = "talent_acquisition"
    TECHNOLOGY_ADOPTION = "technology_adoption"
    SUSTAINABILITY = "sustainability"
    COMPLIANCE = "compliance"
    RISK_MANAGEMENT = "risk_management"
    PARTNERSHIP_DEVELOPMENT = "partnership_development"
    INNOVATION = "innovation"
    GLOBAL_EXPANSION = "global_expansion"
    MERGER_ACQUISITION = "merger_acquisition"
    CUSTOMER_RETENTION = "customer_retention"
    DATA_ANALYTICS = "data_analytics"
    OTHER = "other"


class UserRole(str, Enum):
    """Enumeration of user roles."""
    CEO = "ceo"
    CTO = "cto"
    CFO = "cfo"
    COO = "coo"
    CMO = "cmo"
    PRESIDENT = "president"
    VP_STRATEGY = "vp_strategy"
    VP_PRODUCT = "vp_product"
    VP_MARKETING = "vp_marketing"
    VP_SALES = "vp_sales"
    VP_OPERATIONS = "vp_operations"
    DIRECTOR = "director"
    SENIOR_MANAGER = "senior_manager"
    MANAGER = "manager"
    ANALYST = "analyst"
    CONSULTANT = "consultant"
    RESEARCHER = "researcher"
    PRODUCT_MANAGER = "product_manager"
    PROJECT_MANAGER = "project_manager"
    BUSINESS_ANALYST = "business_analyst"
    STRATEGY_ANALYST = "strategy_analyst"
    MARKET_RESEARCHER = "market_researcher"
    COMPETITIVE_ANALYST = "competitive_analyst"
    OTHER = "other"


class StrategicProfileBase(BaseModel):
    """Base strategic profile schema."""
    industry: Optional[IndustryType] = Field(None, description="Primary industry sector")
    organization_type: Optional[OrganizationType] = Field(None, description="Type of organization")
    role: Optional[UserRole] = Field(None, description="User's role in the organization")
    strategic_goals: Optional[List[StrategicGoalCategory]] = Field(
        default_factory=list,
        description="Key strategic objectives and focus areas",
        max_items=10
    )
    organization_size: Optional[OrganizationSize] = Field(None, description="Size of the organization")

    @validator('strategic_goals', pre=True)
    def validate_strategic_goals(cls, v):
        """Validate strategic goals list."""
        if v is None:
            return []
        if isinstance(v, str):
            # Handle single string input
            return [v]
        if isinstance(v, list):
            # Remove duplicates while preserving order
            seen = set()
            unique_goals = []
            for goal in v:
                if goal not in seen:
                    seen.add(goal)
                    unique_goals.append(goal)
            return unique_goals
        return v

    @validator('strategic_goals')
    def validate_strategic_goals_length(cls, v):
        """Validate strategic goals don't exceed maximum."""
        if v and len(v) > 10:
            raise ValueError("Maximum 10 strategic goals allowed")
        return v


class StrategicProfileCreate(StrategicProfileBase):
    """Schema for creating a strategic profile."""
    
    class Config:
        schema_extra = {
            "example": {
                "industry": "technology",
                "organization_type": "startup",
                "role": "ceo",
                "strategic_goals": [
                    "market_expansion",
                    "product_development",
                    "competitive_intelligence"
                ],
                "organization_size": "small"
            }
        }


class StrategicProfileUpdate(StrategicProfileBase):
    """Schema for updating a strategic profile."""
    
    class Config:
        schema_extra = {
            "example": {
                "industry": "healthcare",
                "strategic_goals": [
                    "digital_transformation",
                    "compliance",
                    "operational_efficiency"
                ]
            }
        }


class StrategicProfileResponse(StrategicProfileBase):
    """Schema for strategic profile responses."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "user_id": 123,
                "industry": "technology",
                "organization_type": "startup",
                "role": "ceo",
                "strategic_goals": [
                    "market_expansion",
                    "product_development",
                    "competitive_intelligence"
                ],
                "organization_size": "small",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z"
            }
        }


class StrategicProfileAnalytics(BaseModel):
    """Schema for strategic profile analytics and insights."""
    profile_completeness: float = Field(description="Profile completeness percentage (0-100)")
    missing_fields: List[str] = Field(description="List of missing required fields")
    recommended_goals: List[StrategicGoalCategory] = Field(
        description="Recommended strategic goals based on industry and role"
    )
    industry_trends: Optional[dict] = Field(description="Industry-specific trends and insights")
    
    class Config:
        schema_extra = {
            "example": {
                "profile_completeness": 85.0,
                "missing_fields": ["organization_size"],
                "recommended_goals": ["digital_transformation", "data_analytics"],
                "industry_trends": {
                    "top_goals": ["market_expansion", "product_development"],
                    "emerging_trends": ["ai_adoption", "sustainability"]
                }
            }
        }


class BulkStrategicProfileUpdate(BaseModel):
    """Schema for bulk updating strategic profiles."""
    profiles: List[StrategicProfileUpdate] = Field(
        description="List of profile updates",
        max_items=100
    )
    
    @validator('profiles')
    def validate_profiles_count(cls, v):
        """Validate profiles list length."""
        if len(v) > 100:
            raise ValueError("Maximum 100 profiles allowed per bulk operation")
        return v
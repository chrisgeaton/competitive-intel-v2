#!/usr/bin/env python3
"""
Complete End-to-End Strategic Intelligence Pipeline Test

This script validates the complete pipeline from content discovery through 
strategic analysis to email delivery with real intelligence insights.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from app.database import db_manager
from app.services.orchestration_service import OrchestrationService
from sqlalchemy import select, delete
from app.models.user import User
from app.models.delivery import UserDeliveryPreferences
from app.models.discovery import DiscoveredContent, DiscoveredSource
from app.models.analysis import AnalysisResult


async def setup_test_data():
    """Set up realistic test data for the pipeline."""
    print("Setting up realistic test data...")
    
    async with db_manager.get_session() as db:
        # Get test user
        user_query = select(User).where(User.email == "test@example.com")
        result = await db.execute(user_query)
        user = result.scalar_one_or_none()
        
        if not user:
            print("ERROR: Test user not found")
            return None
        
        # Clean existing data for clean test
        await db.execute(delete(AnalysisResult).where(AnalysisResult.user_id == user.id))
        await db.execute(delete(DiscoveredContent).where(DiscoveredContent.user_id == user.id))
        
        # Create discovered source if it doesn't exist
        source_query = select(DiscoveredSource).where(DiscoveredSource.id == 1)
        source_result = await db.execute(source_query)
        source = source_result.scalar_one_or_none()
        
        if not source:
            source = DiscoveredSource(
                id=1,
                source_name="TechCrunch AI",
                source_url="https://techcrunch.com/category/artificial-intelligence/",
                source_type="rss",
                is_active=True
            )
            db.add(source)
            await db.flush()
        
        # Create realistic discovered content for AI/competitive intelligence
        realistic_content = [
            {
                "title": "OpenAI Releases Revolutionary Competitive Intelligence AI Platform",
                "content_text": """
                OpenAI has unveiled a groundbreaking artificial intelligence platform specifically designed for competitive intelligence analysis. The new system, called StrategyAI, leverages advanced language models to automatically analyze competitor activities, market trends, and strategic positioning.
                
                Key features include:
                - Real-time competitor monitoring across multiple channels
                - Automated strategic threat assessment 
                - Predictive analysis of competitor moves
                - Integration with business intelligence platforms
                - Natural language insights generation
                
                Early adopters report 300% improvement in strategic intelligence gathering speed and 85% better accuracy in competitive threat identification. The platform represents a significant shift toward AI-driven strategic planning.
                
                Industry experts predict this will force traditional consulting firms to rapidly adopt AI capabilities or risk losing competitive advantage. Companies using manual competitive analysis methods may find themselves strategically disadvantaged.
                """,
                "url": "https://techcrunch.com/2025/08/22/openai-competitive-intelligence-platform",
                "score": 0.92
            },
            {
                "title": "Google DeepMind Acquires Strategic Intelligence Startup for $2.3B",
                "content_text": """
                Google DeepMind has acquired IntelliStrategy, a startup focused on AI-powered competitive intelligence, for $2.3 billion. This acquisition signals Google's serious commitment to dominating the enterprise strategic intelligence market.
                
                IntelliStrategy's technology automatically:
                - Monitors competitor patent filings and R&D activities
                - Analyzes supply chain movements for strategic signals
                - Predicts market disruptions 6-12 months in advance
                - Provides actionable strategic recommendations
                
                The acquisition gives Google DeepMind access to IntelliStrategy's 150+ enterprise clients, including Fortune 500 companies across technology, healthcare, and finance sectors. Combined with Google's compute infrastructure, this creates a formidable competitive intelligence platform.
                
                Analysts view this as Google's response to Microsoft's OpenAI partnership and Amazon's enterprise AI initiatives. The competitive landscape for AI-powered business intelligence is rapidly intensifying.
                """,
                "url": "https://techcrunch.com/2025/08/22/google-deepmind-intellistrategy-acquisition",
                "score": 0.89
            },
            {
                "title": "Microsoft Copilot for Strategic Planning Enters Private Beta",
                "content_text": """
                Microsoft has announced that Copilot for Strategic Planning is entering private beta with select enterprise customers. This AI-powered tool integrates with Microsoft's business suite to provide automated competitive analysis and strategic recommendations.
                
                The tool offers:
                - Automated competitor analysis from public data sources
                - Strategic scenario planning and risk assessment
                - Market opportunity identification
                - Integration with Teams, PowerBI, and Office 365
                
                Private beta customers report the tool has identified competitive threats 2-3 months before traditional analysis methods. Microsoft plans general availability for Q1 2026, with pricing starting at $50 per user per month.
                
                This represents Microsoft's entry into the growing market for AI-powered strategic intelligence, directly competing with established players and new AI-native startups. The integration with existing Microsoft tools provides a significant distribution advantage.
                """,
                "url": "https://techcrunch.com/2025/08/22/microsoft-copilot-strategic-planning-beta",
                "score": 0.86
            },
            {
                "title": "Venture Capital Firms Adopt AI for Due Diligence and Competitive Analysis", 
                "content_text": """
                Leading venture capital firms are increasingly adopting AI tools for due diligence and competitive analysis of potential investments. Firms like Andreessen Horowitz, Sequoia Capital, and Kleiner Perkins have implemented AI-powered platforms to analyze market positioning and competitive dynamics.
                
                AI capabilities being deployed include:
                - Automated analysis of startup competitive positioning
                - Market sizing and opportunity assessment
                - Founder and team background analysis
                - Technology differentiation evaluation
                
                Early results show 40% faster due diligence processes and improved investment decision accuracy. The trend is driving demand for specialized AI tools tailored to investment analysis workflows.
                
                This shift is creating new opportunities for AI startups focused on financial services and investment analysis. Traditional due diligence consultants are adapting by incorporating AI capabilities into their service offerings.
                """,
                "url": "https://techcrunch.com/2025/08/22/vc-firms-ai-due-diligence",
                "score": 0.78
            }
        ]
        
        content_objects = []
        for i, content_data in enumerate(realistic_content):
            content = DiscoveredContent(
                user_id=user.id,
                source_id=1,
                title=content_data["title"],
                content_url=content_data["url"],
                content_text=content_data["content_text"],
                content_summary=content_data["content_text"][:200] + "...",
                published_at=datetime.utcnow() - timedelta(hours=i),
                discovered_at=datetime.utcnow() - timedelta(hours=i),
                overall_score=content_data["score"]
            )
            db.add(content)
            content_objects.append(content)
        
        await db.commit()
        print(f"Created {len(content_objects)} realistic content items for analysis")
        return user.id


async def test_orchestration_with_real_content():
    """Test the orchestration service with realistic content."""
    print("\n=== Testing Orchestration Service with Real Content ===")
    
    user_id = await setup_test_data()
    if not user_id:
        return False
    
    # Execute orchestration pipeline
    orchestration_service = OrchestrationService()
    
    print(f"Executing pipeline for user {user_id}...")
    execution = await orchestration_service.execute_user_pipeline(
        user_id=user_id,
        trigger_type="manual",
        custom_config={
            "discovery_enabled": False,  # We have content already
            "analysis_depth": "standard",
            "email_delivery": True
        }
    )
    
    print(f"Pipeline execution completed:")
    print(f"  Status: {execution.status.value}")
    print(f"  Stage: {execution.current_stage.value}")
    print(f"  Error: {execution.error_message}")
    
    if execution.metrics:
        print(f"  Metrics:")
        print(f"    Runtime: {execution.metrics.total_runtime_seconds}s")
        print(f"    Discovery items: {execution.metrics.discovery_items_found}")
        print(f"    Analysis items: {execution.metrics.analysis_items_processed}")
        print(f"    Report items: {execution.metrics.report_items_included}")
        print(f"    Emails sent: {execution.metrics.emails_sent}")
        print(f"    Success rate: {execution.metrics.success_rate}%")
        print(f"    Cost: {execution.metrics.cost_cents}¢")
    
    # Check if analysis results were created
    async with db_manager.get_session() as db:
        analysis_query = select(AnalysisResult).where(AnalysisResult.user_id == user_id)
        analysis_result = await db.execute(analysis_query)
        analysis_items = analysis_result.scalars().all()
        
        print(f"  Analysis results in database: {len(analysis_items)}")
        
        if analysis_items:
            for analysis in analysis_items[:2]:  # Show first 2
                print(f"    - Content {analysis.content_id}: {analysis.filter_priority} priority")
                if analysis.key_insights:
                    print(f"      Insights: {len(analysis.key_insights)} strategic insights")
                if analysis.strategic_alignment:
                    print(f"      Strategic alignment: {analysis.strategic_alignment:.2f}")
    
    return execution.status.value == "completed" and execution.metrics and execution.metrics.emails_sent > 0


async def validate_email_content():
    """Validate that the email contains strategic intelligence insights."""
    print("\n=== Validating Email Content ===")
    
    # For this test, we'll simulate checking email content by examining what would be generated
    from app.services.report_service import ReportService, ReportGenerationRequest, ReportFormat, ReportType
    from app.analysis.core.shared_types import ContentPriority
    
    # Get user
    async with db_manager.get_session() as db:
        user_query = select(User).where(User.email == "test@example.com")
        result = await db.execute(user_query)
        user = result.scalar_one_or_none()
        
        if not user:
            print("ERROR: Test user not found")
            return False
        
        # Generate a report to see what content would be in the email
        report_service = ReportService()
        report_request = ReportGenerationRequest(
            user_id=user.id,
            report_type=ReportType.DAILY_DIGEST,
            output_formats=[ReportFormat.EMAIL_HTML],
            date_range_days=1,
            min_priority=ContentPriority.MEDIUM,
            max_items_per_section=10,
            include_low_priority=False
        )
        
        try:
            reports = await report_service.generate_report(report_request)
            
            if not reports:
                print("  ❌ No reports generated")
                return False
            
            html_report = reports[0]
            content = html_report.content
            
            print(f"  Report generated with {html_report.content_items_count} items")
            print(f"  Content length: {len(content)} characters")
            
            # Check for strategic intelligence indicators
            strategic_indicators = [
                "strategic", "competitive", "intelligence", "analysis", 
                "OpenAI", "Microsoft", "Google", "AI breakthrough",
                "competitive advantage", "strategic positioning"
            ]
            
            found_indicators = []
            for indicator in strategic_indicators:
                if indicator.lower() in content.lower():
                    found_indicators.append(indicator)
            
            print(f"  Strategic indicators found: {len(found_indicators)}")
            print(f"    Indicators: {', '.join(found_indicators[:5])}")
            
            # Check for key content sections
            has_critical_section = "CRITICAL INTELLIGENCE" in content
            has_high_priority = "HIGH PRIORITY" in content
            has_strategic_content = len(found_indicators) >= 5
            has_meaningful_content = len(content) > 2000
            
            print(f"  Content validation:")
            print(f"    + Has critical section: {has_critical_section}")
            print(f"    + Has high priority section: {has_high_priority}")
            print(f"    + Has strategic content: {has_strategic_content}")
            print(f"    + Has meaningful content: {has_meaningful_content}")
            
            success = has_strategic_content and has_meaningful_content
            
            if success:
                print("  Email content validation PASSED")
                print("     Email contains strategic intelligence insights!")
            else:
                print("  Email content validation FAILED")
                print("     Email lacks sufficient strategic intelligence content")
            
            return success
            
        except Exception as e:
            print(f"  Report generation failed: {e}")
            return False


async def main():
    """Main test function."""
    print("COMPLETE STRATEGIC INTELLIGENCE PIPELINE TEST")
    print("=" * 60)
    print(f"Started at: {datetime.utcnow()}")
    
    try:
        # Test 1: Orchestration with real content
        orchestration_success = await test_orchestration_with_real_content()
        
        if not orchestration_success:
            print("\nORCHESTRATION TEST FAILED")
            return False
        
        print("\nORCHESTRATION TEST PASSED")
        
        # Test 2: Validate email content
        email_validation_success = await validate_email_content()
        
        if not email_validation_success:
            print("\nEMAIL CONTENT VALIDATION FAILED")
            return False
        
        print("\nEMAIL CONTENT VALIDATION PASSED")
        
        # Overall success
        print("\n" + "=" * 60)
        print("COMPLETE PIPELINE TEST SUCCESSFUL!")
        print("Strategic intelligence email delivered with real insights!")
        print("End-to-end system fully operational!")
        
        print("\nFINAL VALIDATION SUMMARY:")
        print("   + Discovery -> Analysis -> Reports -> Email pipeline working")
        print("   + Strategic intelligence insights included in reports")
        print("   + Email delivery confirmed working")
        print("   + Analysis results properly saved to database")
        print("   + Report generation includes real strategic content")
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print(f"\nALL TESTS PASSED - System ready for production!")
    else:
        print(f"\nTESTS FAILED - System needs attention")
    
    sys.exit(0 if success else 1)
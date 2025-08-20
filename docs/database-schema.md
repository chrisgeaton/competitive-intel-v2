# Database Schema Documentation

## Overview
PostgreSQL database schema for Competitive Intelligence v2 system. Designed for strategic intelligence beyond competitor tracking, supporting diverse industries and use cases.

**Database**: `competitive_intelligence`  
**Tables**: 15 core tables with proper indexes and foreign keys  
**Location**: `database/schema.sql`

## Core Design Principles
1. **Strategic Profile-Driven**: Users define goals, system finds relevant intelligence
2. **Flexible Entity Tracking**: Support competitors, topics, regulations, opportunities
3. **Self-Service Ready**: Users can manage their own preferences
4. **Cost-Optimized Analysis**: Track AI analysis costs and performance
5. **Quality-Scored Sources**: Prevent information overload with source quality metrics

## Table Groups

### 👥 Users & Authentication
Core user management and session handling

**users** - Primary user records
- `id` (SERIAL PRIMARY KEY)
- `email` (UNIQUE, for login)
- `name` (display name)
- `password_hash` (for future self-service auth)
- `subscription_status` (trial, active, cancelled)
- `created_at`, `last_login`, `is_active`

**user_sessions** - JWT session management
- `user_id` → users(id)
- `token` (UNIQUE JWT token)
- `expires_at` (session expiration)

### 🎯 Strategic Profile System
User's strategic context and goals

**user_strategic_profile** - Core strategic context
- `user_id` → users(id)
- `industry` (healthcare, nonprofit, fintech, etc.)
- `organization_type` (startup, enterprise, nonprofit)
- `role` (ceo, product_manager, strategy_analyst)
- `strategic_goals` (TEXT[] array of strategic focus areas)
- `organization_size` (small, medium, large)

**user_focus_areas** - Specific intelligence focus areas
- `user_id` → users(id)
- `focus_area` (AI integration, regulatory changes, funding opportunities)
- `keywords` (TEXT[] related search terms)
- `priority` (1=low, 2=medium, 3=high, 4=critical)

### 🏢 Entity Tracking System
Flexible tracking of any entity type

**tracking_entities** - Entities to monitor
- `name` (entity name)
- `entity_type` (competitor, organization, topic, person, technology)
- `domain` (website if applicable)
- `industry`, `description`
- `metadata` (JSONB for flexible entity-specific data)

**user_entity_tracking** - User's tracking preferences
- `user_id` → users(id)
- `entity_id` → tracking_entities(id)
- `priority` (1-4 priority level)
- `custom_keywords` (TEXT[] user-specific keywords)
- `tracking_enabled` (boolean flag)

### 📡 Source Discovery System
Smart content source management

**content_sources** - Information sources
- `name`, `source_type` (rss, google_news, web_scrape, social_media, podcast)
- `url`, `config` (JSONB source-specific configuration)
- `quality_score`, `reliability_score` (0.0-1.0 ratings)
- `check_frequency_minutes` (collection interval)
- `is_active`, `last_checked`

**entity_source_coverage** - Source relevance mapping
- `entity_id` → tracking_entities(id)
- `source_id` → content_sources(id)
- `relevance_score` (0.0-1.0 how relevant source is for entity)

### 📰 Content & Analysis System
Article storage and AI analysis results

**articles** - Raw collected content
- `source_id` → content_sources(id)
- `url` (UNIQUE), `title`, `content`, `author`
- `published_at`, `discovered_at`
- `content_hash` (for deduplication)
- `metadata` (JSONB source-specific data)

**article_analysis** - AI analysis results
- `article_id` → articles(id)
- `analysis_stage` (1=quick_classification, 2=deep_analysis)
- `analysis_type` (relevance, strategic_insights, urgency)
- `result` (JSONB analysis results)
- `confidence_score`, `cost_cents`, `model_used`

**strategic_insights** - Extracted strategic intelligence
- `article_id` → articles(id)
- `insight_type` (opportunity, threat, trend, competitive_move)
- `title`, `description`
- `strategic_significance` (low, medium, high, critical)
- `urgency` (routine, timely, urgent, breaking)
- `affected_entities` (INTEGER[] array of entity IDs)
- `confidence_score`

### 📧 Delivery & Reporting System
User reports and engagement tracking

**user_delivery_preferences** - How users want reports delivered
- `user_id` → users(id)
- `frequency` (daily, weekly, real_time)
- `delivery_time`, `timezone`
- `max_articles_per_report`
- `min_significance_level` (low, medium, high, critical)
- `content_format` (executive_summary, full, bullet_points)
- `email_enabled`, `urgent_alerts_enabled`, `digest_mode`

**user_reports** - Generated reports
- `user_id` → users(id)
- `report_type` (daily, weekly, urgent_alert)
- `subject`, `content_html`, `content_text`
- `article_count`, `insight_count`
- `status` (draft, sent, failed)
- `scheduled_for`, `sent_at`, `opened_at`

**report_engagement** - User engagement tracking
- `report_id` → user_reports(id)
- `article_id` → articles(id)
- `action_type` (opened, clicked, saved, shared)
- `action_timestamp`

### 📊 System Monitoring
Performance and health tracking

**system_metrics** - System performance data
- `metric_type` (collection_rate, analysis_cost, user_engagement)
- `metric_value`, `metadata` (JSONB)
- `recorded_at`

## Key Indexes for Performance

### User Operations
- `idx_users_email` - Fast user lookup by email
- `idx_user_sessions_token` - Fast session validation

### Content Queries
- `idx_articles_published_at` - Time-based article queries
- `idx_articles_url_hash` - Deduplication checks
- `idx_article_analysis_article_id` - Analysis lookups

### Strategic Intelligence
- `idx_strategic_insights_urgency` - Urgent item queries
- `idx_strategic_insights_significance` - High-impact insights
- `idx_user_entity_tracking_user_id` - User's tracking preferences

### Reporting
- `idx_user_reports_scheduled_for` - Report scheduling
- `idx_user_reports_status` - Report status queries

## Data Relationships

### User → Strategic Profile (1:1)
Each user has one strategic profile defining their intelligence needs

### User → Focus Areas (1:Many)
Users can have multiple focus areas with different priorities

### User → Entity Tracking (Many:Many)
Users track multiple entities, entities tracked by multiple users

### Entity → Source Coverage (Many:Many)
Entities covered by multiple sources, sources cover multiple entities

### Article → Analysis → Insights (1:Many:Many)
Articles have multiple analysis results, generate multiple insights

### User → Reports → Engagement (1:Many:Many)
Users receive multiple reports, each with multiple engagement events

## Strategic Intelligence Flow

1. **User Setup**: Define strategic profile and focus areas
2. **Entity Discovery**: System suggests relevant entities to track
3. **Source Mapping**: Sources automatically linked to entities
4. **Content Collection**: Articles collected from relevant sources
5. **AI Analysis**: Two-stage analysis with cost optimization
6. **Insight Extraction**: Strategic insights extracted from analysis
7. **Report Generation**: Personalized reports based on user preferences
8. **Engagement Tracking**: Monitor user interaction for optimization

## Example Data Scenarios

### Healthcare User
```sql
-- Strategic profile
industry: 'healthcare'
role: 'product_manager'
strategic_goals: ['AI integration', 'regulatory compliance', 'competitive positioning']

-- Focus areas
'EHR AI integration' (priority: 4)
'FDA AI regulations' (priority: 3)
'Epic competitive moves' (priority: 4)

-- Entity tracking
Epic (competitor, priority: 4)
FDA AI guidance (topic, priority: 3)
Microsoft Healthcare (organization, priority: 2)
```

### Nonprofit User
```sql
-- Strategic profile
industry: 'nonprofit'
organization_type: 'nonprofit'
strategic_goals: ['funding opportunities', 'regulatory changes', 'best practices']

-- Focus areas
'Fishery management grants' (priority: 4)
'Ocean conservation policy' (priority: 3)
'Sustainable fishing practices' (priority: 3)

-- Entity tracking
NOAA (organization, priority: 4)
Ocean Conservancy (organization, priority: 3)
Marine Stewardship Council (organization, priority: 2)
```

## Migration from Current System

### Current → New Mapping
- `config.json users` → `users` + `user_strategic_profile`
- `config.json competitors` → `tracking_entities` (type: competitor)
- `user preferences` → `user_delivery_preferences`
- SQLite articles → PostgreSQL `articles` (with proper normalization)

### Data Preservation
All current functionality preserved with enhanced capabilities:
- ✅ User email settings → `user_delivery_preferences`
- ✅ Competitor tracking → `user_entity_tracking`
- ✅ Article storage → `articles` (with better metadata)
- ✅ Analysis results → `article_analysis` (with cost tracking)

## Schema Evolution Strategy
- **Backwards compatible**: New columns added with defaults
- **Versioned migrations**: Track schema changes over time
- **Rollback capability**: Each migration can be reversed
- **Data validation**: Constraints ensure data integrity
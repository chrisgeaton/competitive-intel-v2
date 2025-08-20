-- Competitive Intelligence v2 Database Schema
-- Clean, extensible design for personalized strategic intelligence

-- ================================
-- USERS & AUTHENTICATION
-- ================================

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255), -- For future authentication
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    subscription_status VARCHAR(50) DEFAULT 'trial' -- trial, active, cancelled
);

CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- USER STRATEGIC PROFILE
-- ================================

-- Core strategic context that drives everything else
CREATE TABLE user_strategic_profile (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    industry VARCHAR(100), -- healthcare, nonprofit, fintech, etc.
    organization_type VARCHAR(100), -- startup, enterprise, nonprofit, etc.
    role VARCHAR(100), -- ceo, product_manager, strategy_analyst, etc.
    strategic_goals TEXT[], -- array of strategic goals/focus areas
    organization_size VARCHAR(50), -- small, medium, large
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Key focus areas that user wants intelligence on
CREATE TABLE user_focus_areas (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    focus_area VARCHAR(255) NOT NULL, -- "AI integration", "regulatory changes", "funding opportunities"
    keywords TEXT[], -- related keywords for this focus area
    priority INTEGER DEFAULT 3, -- 1=low, 2=medium, 3=high, 4=critical
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- TRACKING ENTITIES (Dynamic)
-- ================================

-- Entities to track (competitors, organizations, topics, etc.)
CREATE TABLE tracking_entities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL, -- competitor, organization, topic, person, technology
    domain VARCHAR(255), -- website domain if applicable
    description TEXT,
    industry VARCHAR(100),
    metadata JSONB, -- flexible storage for entity-specific data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(name, entity_type)
);

-- User's specific tracking preferences for entities
CREATE TABLE user_entity_tracking (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    entity_id INTEGER REFERENCES tracking_entities(id) ON DELETE CASCADE,
    priority INTEGER DEFAULT 3, -- 1=low, 2=medium, 3=high, 4=critical
    custom_keywords TEXT[], -- additional keywords specific to this user's interest
    tracking_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, entity_id)
);

-- ================================
-- CONTENT SOURCES & DISCOVERY
-- ================================

-- Information sources (RSS, APIs, websites, etc.)
CREATE TABLE content_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- rss, google_news, web_scrape, social_media, podcast
    url VARCHAR(500),
    config JSONB, -- source-specific configuration (API keys, scraping rules, etc.)
    quality_score DECIMAL(3,2) DEFAULT 0.5, -- 0.0 to 1.0 quality rating
    reliability_score DECIMAL(3,2) DEFAULT 0.5, -- 0.0 to 1.0 reliability rating
    is_active BOOLEAN DEFAULT true,
    last_checked TIMESTAMP,
    check_frequency_minutes INTEGER DEFAULT 60,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Link sources to entities (which sources cover which entities)
CREATE TABLE entity_source_coverage (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES tracking_entities(id) ON DELETE CASCADE,
    source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
    relevance_score DECIMAL(3,2) DEFAULT 0.5, -- how relevant this source is for this entity
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(entity_id, source_id)
);

-- ================================
-- CONTENT & ANALYSIS
-- ================================

-- Raw articles/content collected
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES content_sources(id),
    url VARCHAR(1000) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    author VARCHAR(255),
    published_at TIMESTAMP,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_hash VARCHAR(64), -- for deduplication
    metadata JSONB -- source-specific metadata
);

-- AI analysis results for articles
CREATE TABLE article_analysis (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    analysis_stage INTEGER NOT NULL, -- 1=quick_classification, 2=deep_analysis
    analysis_type VARCHAR(50) NOT NULL, -- relevance, strategic_insights, urgency, etc.
    result JSONB NOT NULL, -- AI analysis results
    confidence_score DECIMAL(3,2),
    cost_cents INTEGER, -- cost in cents
    model_used VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Strategic insights extracted from articles
CREATE TABLE strategic_insights (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    insight_type VARCHAR(50) NOT NULL, -- opportunity, threat, trend, competitive_move
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    strategic_significance VARCHAR(50), -- low, medium, high, critical
    urgency VARCHAR(50), -- routine, timely, urgent, breaking
    affected_entities INTEGER[], -- array of entity IDs
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- USER DELIVERY PREFERENCES
-- ================================

CREATE TABLE user_delivery_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    
    -- Delivery schedule
    frequency VARCHAR(50) DEFAULT 'daily', -- daily, weekly, real_time
    delivery_time TIME DEFAULT '08:00:00',
    timezone VARCHAR(50) DEFAULT 'UTC',
    weekend_delivery BOOLEAN DEFAULT false,
    
    -- Content preferences
    max_articles_per_report INTEGER DEFAULT 10,
    min_significance_level VARCHAR(50) DEFAULT 'medium', -- low, medium, high, critical
    content_format VARCHAR(50) DEFAULT 'executive_summary', -- full, summary, bullet_points
    
    -- Notification preferences
    email_enabled BOOLEAN DEFAULT true,
    urgent_alerts_enabled BOOLEAN DEFAULT true,
    digest_mode BOOLEAN DEFAULT true, -- combine multiple updates
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- REPORTS & NOTIFICATIONS
-- ================================

-- Generated reports for users
CREATE TABLE user_reports (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    report_type VARCHAR(50) NOT NULL, -- daily, weekly, urgent_alert
    subject VARCHAR(500),
    content_html TEXT,
    content_text TEXT,
    article_count INTEGER DEFAULT 0,
    insight_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'draft', -- draft, sent, failed
    scheduled_for TIMESTAMP,
    sent_at TIMESTAMP,
    opened_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Track user engagement with reports
CREATE TABLE report_engagement (
    id SERIAL PRIMARY KEY,
    report_id INTEGER REFERENCES user_reports(id) ON DELETE CASCADE,
    article_id INTEGER REFERENCES articles(id),
    action_type VARCHAR(50), -- opened, clicked, saved, shared
    action_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- SYSTEM MONITORING
-- ================================

-- System health and performance tracking
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    metric_type VARCHAR(50) NOT NULL, -- collection_rate, analysis_cost, user_engagement
    metric_value DECIMAL(10,4),
    metadata JSONB,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- INDEXES FOR PERFORMANCE
-- ================================

-- User lookups
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_user_sessions_token ON user_sessions(token);

-- Content queries
CREATE INDEX idx_articles_published_at ON articles(published_at);
CREATE INDEX idx_articles_url_hash ON articles(content_hash);
CREATE INDEX idx_article_analysis_article_id ON article_analysis(article_id);

-- Strategic insights
CREATE INDEX idx_strategic_insights_article_id ON strategic_insights(article_id);
CREATE INDEX idx_strategic_insights_urgency ON strategic_insights(urgency);
CREATE INDEX idx_strategic_insights_significance ON strategic_insights(strategic_significance);

-- User tracking
CREATE INDEX idx_user_entity_tracking_user_id ON user_entity_tracking(user_id);
CREATE INDEX idx_user_entity_tracking_entity_id ON user_entity_tracking(entity_id);

-- Reports
CREATE INDEX idx_user_reports_user_id ON user_reports(user_id);
CREATE INDEX idx_user_reports_status ON user_reports(status);
CREATE INDEX idx_user_reports_scheduled_for ON user_reports(scheduled_for);
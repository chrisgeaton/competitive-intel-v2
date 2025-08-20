# Architecture Decisions - Competitive Intelligence v2

## Core Architectural Choices

### 1. Database: PostgreSQL (Local) ✅
**Decision**: Use PostgreSQL locally instead of SQLite  
**Reasoning**: 
- Production-ready features (concurrent users, advanced queries)
- Better data types (JSONB, arrays)
- Easier scaling path to cloud when ready
- Proper foreign key constraints and indexes

**Implementation**: Docker container for local development  
**Alternative Considered**: SQLite (rejected due to scalability limitations)

### 2. Service Architecture: Modular Monolith → Microservices ✅
**Decision**: Build modular services that can be extracted later  
**Reasoning**:
- Faster initial development (single codebase)
- Clear service boundaries in code
- Easy extraction to microservices when scaling
- Simpler debugging and deployment initially

**Services Designed**:
1. **User Config Service** (foundation)
2. **Discovery Service** (smart source finding)
3. **Analysis Service** (AI pipeline)
4. **Scoring Service** (relevance engine)

### 3. User Configuration: Database-Only ✅
**Decision**: Eliminate config files, use database as single source of truth  
**Reasoning**:
- Solves current pain points (manual user management, restart requirements)
- Enables self-service user preferences
- Real-time configuration changes
- Audit trail and version history

**Replaces**: JSON config files + SQLite hybrid approach

### 4. Database Driver: psycopg3 ✅
**Decision**: Use psycopg3 instead of psycopg2  
**Reasoning**:
- Modern async/await support
- Better Windows compatibility
- Improved performance
- Active development and support

**Installation**: `pip install "psycopg[binary]"` worked successfully

### 5. Web Framework: FastAPI ✅
**Decision**: Use FastAPI for all API services  
**Reasoning**:
- Automatic API documentation
- Type validation with Pydantic
- Modern async support
- JWT authentication built-in
- Fast development cycle

### 6. Strategic Intelligence Focus ✅
**Decision**: Expand beyond competitor tracking to strategic intelligence  
**Reasoning**:
- Broader market opportunity (nonprofit example: fishery management)
- More valuable user proposition
- Flexible entity tracking (competitors, topics, regulations, opportunities)
- User-defined strategic goals drive relevance

**Example Use Cases**:
- Healthcare: Competitor analysis + regulatory changes
- Nonprofit: Funding opportunities + industry best practices
- Fintech: Market trends + regulatory updates

### 7. Analysis Pipeline: Preserve Two-Stage Approach ✅
**Decision**: Keep proven two-stage AI analysis with enhancements  
**Reasoning**:
- Current cost optimization works well ($0.50-2.00/day per user)
- Stage 1 filtering saves 70% on irrelevant content
- Stage 2 provides deep strategic insights
- Add personalization based on user strategic goals

**Enhancements Planned**:
- Context-aware analysis (nonprofit vs enterprise focus)
- User strategic goals influence analysis prompts
- Dynamic source discovery based on analysis results

### 8. Development Strategy: Parallel Build ✅
**Decision**: Build new system alongside old system  
**Reasoning**:
- No disruption to current 3 alpha users
- Ability to compare systems side-by-side
- No migration pressure during development
- Clean slate architecture design

**Testing Plan**: Use Chris as test user with different industry focus

### 9. Source Discovery: AI-Enhanced ✅
**Decision**: AI-powered source recommendation with quality scoring  
**Reasoning**:
- Current manual source management doesn't scale
- AI can suggest relevant sources based on user goals
- Quality scoring prevents information overload
- Smart source expansion based on user engagement

**Sources Supported**:
- RSS feeds (current strength)
- Google News API
- Web scraping with rate limiting
- Podcast transcripts
- Social media monitoring
- Google Alerts automation

### 10. Authentication: JWT with Future Self-Service ✅
**Decision**: JWT-based authentication ready for self-service  
**Reasoning**:
- Stateless authentication
- Frontend-friendly
- Ready for user self-registration
- Session management built-in

**Current**: Admin-managed users  
**Future**: Self-service registration and preference management

## Technical Debt Resolved

### From Current System
- ❌ Two sources of truth (config.json + database)
- ❌ Manual user management requiring developer intervention
- ❌ Restart required for configuration changes
- ❌ Inconsistent data models
- ❌ Circular dependencies and tight coupling
- ❌ Limited scalability and error handling

### In New System
- ✅ Single database source of truth
- ✅ Self-service user management
- ✅ Real-time configuration updates
- ✅ Consistent, normalized data models
- ✅ Clear service boundaries
- ✅ Built for scale from day one

## Key Design Patterns

### 1. Strategic Profile-Driven
Users define strategic goals and focus areas rather than just competitor lists
- Enables personalized analysis
- Supports diverse industries and use cases
- Drives intelligent source discovery

### 2. Entity-Centric Tracking
Flexible tracking of any entity type (competitors, topics, people, technologies)
- Supports strategic intelligence beyond competitors
- Enables relationship mapping
- Facilitates trend analysis

### 3. Quality-Scored Sources
All sources have quality and reliability scores
- Prevents information overload
- Enables automatic source pruning
- Supports source recommendation algorithms

### 4. Cost-Optimized Analysis
Preserve two-stage analysis with cost tracking
- Maintains economic viability
- Enables cost-per-user optimization
- Supports business model validation

## Future Scalability Decisions

### Database Scaling Path
1. **Current**: PostgreSQL local
2. **MVP**: PostgreSQL cloud (AWS RDS/Google Cloud SQL)
3. **Scale**: Read replicas + caching layer
4. **Enterprise**: Sharding by user/organization

### Service Extraction Path
1. **Current**: Modular monolith
2. **MVP**: Extract high-load services (Analysis)
3. **Scale**: Full microservices with API gateway
4. **Enterprise**: Event-driven architecture

### Cost Optimization Path
1. **Current**: OpenAI + two-stage optimization
2. **MVP**: Multi-provider AI (OpenAI + Anthropic + local models)
3. **Scale**: Custom model fine-tuning
4. **Enterprise**: Hybrid cloud + edge processing

## Decisions Deferred
- **Frontend Framework**: React planned, details TBD
- **Hosting Provider**: Local first, cloud provider TBD
- **Monitoring Stack**: Basic logging planned, full observability TBD
- **CI/CD Pipeline**: Manual deployment initially, automation TBD